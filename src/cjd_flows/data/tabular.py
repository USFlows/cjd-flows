"""MAF tabular density-estimation benchmarks (POWER, GAS, HEPMASS, MINIBOONE, BSDS300).

The raw data is the archive of Papamakarios et al. (2017), "Masked
Autoregressive Flow for Density Estimation" (zenodo record 1161203,
``data.tar.gz``). The preprocessing replicates ``datasets/*.py`` of
github.com/gpapamak/maf exactly (same random seed, column removal, noise,
normalization and splits), so numbers are comparable with the literature.

Processed splits are cached as ``.npy`` files under
``<root>/maf/processed`` (``root`` defaults to ``$CJD_FLOWS_DATA`` or
``~/.cache/cjd_flows``).

Expected shapes ``(train, val, test)``:

=========  ====  ==========  ========  ========
name        D     train       val       test
=========  ====  ==========  ========  ========
power        6   1,659,917   184,435   204,928
gas          8     852,174    94,685   105,206
hepmass     21     315,123    35,013   174,987
miniboone   43      29,556     3,284     3,648
bsds300     63   1,000,000    50,000   250,000
=========  ====  ==========  ========  ========

Optional dependencies: ``pandas`` (gas, hepmass) and ``h5py`` (bsds300).
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from cjd_flows.data._download import PathLike, download_url, resolve_root

MAF_URL = "https://zenodo.org/records/1161203/files/data.tar.gz?download=1"

TABULAR_DATASETS = ("power", "gas", "hepmass", "miniboone", "bsds300")

_MEMBERS: Dict[str, Tuple[str, ...]] = {
    "power": ("data/power/data.npy",),
    "gas": ("data/gas/ethylene_CO.pickle",),
    "hepmass": ("data/hepmass/1000_train.csv", "data/hepmass/1000_test.csv"),
    "miniboone": ("data/miniboone/data.npy",),
    "bsds300": ("data/BSDS300/BSDS300.hdf5",),
}


# ------------------------------------------------------------------ raw data
def _maf_dir(root: Optional[PathLike]) -> Path:
    return resolve_root(root) / "maf"


def _ensure_raw(name: str, root: Optional[PathLike], download: bool) -> Path:
    """Makes sure the raw files of ``name`` are extracted; returns the extraction dir."""
    base = _maf_dir(root)
    raw = base / "raw"
    if all((raw / m).exists() for m in _MEMBERS[name]):
        return raw
    archive = base / "data.tar.gz"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"{archive} not found and download=False")
        download_url(MAF_URL, archive, timeout=120.0, retries=5)
    raw.mkdir(parents=True, exist_ok=True)
    wanted = set(_MEMBERS[name])
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.lstrip("./") in wanted and m.isfile()]
        for m in members:
            m.name = m.name.lstrip("./")
        try:
            tf.extractall(raw, members=members, filter="data")
        except TypeError:  # Python < 3.10.12 without extraction filters
            tf.extractall(raw, members=members)
    missing = [m for m in _MEMBERS[name] if not (raw / m).exists()]
    if missing:
        raise FileNotFoundError(f"archive {archive} lacks {missing}")
    return raw


# ------------------------------------------------- MAF preprocessing replicas
def _split_tail(data: np.ndarray, frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    n = int(frac * data.shape[0])
    return data[0:-n], data[-n:]


def _power(raw: Path):
    rng = np.random.RandomState(42)
    data = np.load(raw / "data/power/data.npy")
    rng.shuffle(data)
    n = data.shape[0]
    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    voltage_noise = 0.01 * rng.rand(n, 1)
    gap_noise = 0.001 * rng.rand(n, 1)
    sm_noise = rng.rand(n, 3)
    time_noise = np.zeros((n, 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise
    data, test = _split_tail(data)
    train, val = _split_tail(data)
    both = np.vstack((train, val))
    mu, s = both.mean(axis=0), both.std(axis=0)
    return (train - mu) / s, (val - mu) / s, (test - mu) / s


def _gas(raw: Path):
    import pandas as pd

    data = pd.read_pickle(raw / "data/gas/ethylene_CO.pickle")
    data = data.drop(columns=["Meth", "Eth", "Time"])

    def corr_numbers(df):
        return (df.corr() > 0.98).to_numpy().sum(axis=1)

    b = corr_numbers(data)
    while np.any(b > 1):
        col = data.columns[np.where(b > 1)[0][0]]
        data = data.drop(columns=[col])
        b = corr_numbers(data)
    data = (data - data.mean()) / data.std()
    data = data.to_numpy()
    train, test = _split_tail(data)
    train, val = _split_tail(train)
    return train, val, test


def _hepmass(raw: Path):
    import pandas as pd

    path = raw / "data/hepmass"
    train = pd.read_csv(path / "1000_train.csv", index_col=False)
    test = pd.read_csv(path / "1000_test.csv", index_col=False)
    # keep only signal (label 1), drop the label column
    train = train[train[train.columns[0]] == 1].drop(columns=train.columns[0])
    test = test[test[test.columns[0]] == 1].drop(columns=test.columns[0])
    test = test.drop(columns=test.columns[-1])  # "Because the data set is messed up!"
    mu, s = train.mean(), train.std()
    train = ((train - mu) / s).to_numpy()
    # test columns are aligned positionally in the MAF code (as_matrix after (test-mu)/s
    # with mismatching column names would misalign); replicate by positional arithmetic
    test = (test.to_numpy() - mu.to_numpy()) / s.to_numpy()
    remove = []
    for i, feature in enumerate(train.T):
        _, counts = np.unique(feature, return_counts=True)
        if counts[0] > 5:  # count of the smallest value (sorted Counter items)[0]
            remove.append(i)
    keep = np.array([i for i in range(train.shape[1]) if i not in remove])
    train, test = train[:, keep], test[:, keep]
    train, val = _split_tail(train)
    return train, val, test


def _miniboone(raw: Path):
    data = np.load(raw / "data/miniboone/data.npy")
    data, test = _split_tail(data)
    train, val = _split_tail(data)
    both = np.vstack((train, val))
    mu, s = both.mean(axis=0), both.std(axis=0)
    return (train - mu) / s, (val - mu) / s, (test - mu) / s


def _bsds300(raw: Path):
    import h5py

    with h5py.File(raw / "data/BSDS300/BSDS300.hdf5", "r") as f:
        return np.asarray(f["train"]), np.asarray(f["validation"]), np.asarray(f["test"])


_PROCESSORS = {
    "power": _power,
    "gas": _gas,
    "hepmass": _hepmass,
    "miniboone": _miniboone,
    "bsds300": _bsds300,
}


def load_tabular(
    name: str,
    root: Optional[PathLike] = None,
    download: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Loads a MAF tabular benchmark as ``(train, val, test)`` tensors.

    Args:
        name: one of ``power``, ``gas``, ``hepmass``, ``miniboone``, ``bsds300``.
        root: cache directory (default ``$CJD_FLOWS_DATA`` or ``~/.cache/cjd_flows``).
        download: download the ~860 MB archive if necessary.
        dtype: dtype of the returned tensors.

    Returns:
        Three tensors of shape ``(N_split, D)``.
    """
    name = name.lower()
    if name not in _PROCESSORS:
        raise ValueError(f"unknown tabular dataset {name!r}; choose from {TABULAR_DATASETS}")
    proc = _maf_dir(root) / "processed"
    files = [proc / f"{name}_{s}.npy" for s in ("train", "val", "test")]
    if not all(f.exists() for f in files):
        raw = _ensure_raw(name, root, download)
        splits = _PROCESSORS[name](raw)
        proc.mkdir(parents=True, exist_ok=True)
        for f, arr in zip(files, splits):
            tmp = f.with_name(f.stem + ".tmp.npy")
            np.save(tmp, np.ascontiguousarray(arr, dtype=np.float32))
            tmp.replace(f)
    return tuple(torch.from_numpy(np.load(f)).to(dtype) for f in files)  # type: ignore[return-value]
