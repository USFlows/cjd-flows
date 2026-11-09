"""Image datasets (MNIST, Fashion-MNIST, CIFAR-10) with on-the-fly dequantization.

The raw 8-bit images are kept as ``uint8`` tensors; every access draws fresh
uniform dequantization noise and applies a constant-det affine scaling:

* ``scale="unit"``:      ``y = (x + u) / 256          ∈ [0, 1)``
* ``scale="symmetric"``: ``y = 2 (x + u) / 256 − 1    ∈ [−1, 1)``

``dataset.preprocessing_log_det`` is the per-sample ``log|det ∂y/∂x̃|`` w.r.t.
the dequantized raw data ``x̃ = x + u ∈ [0, 256)^D``; use
``dataset.bits_per_dim(log_prob)`` to report bits/dim.

Data is loaded through ``torchvision`` if it is installed (``backend="auto"``)
and otherwise downloaded directly (MNIST/Fashion IDX files and the *binary*
CIFAR-10 archive -- no pickle involved).
"""

from __future__ import annotations

import gzip
import math
import tarfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cjd_flows.data._download import PathLike, download_url, resolve_root
from cjd_flows.data.preprocess import Compose, Preprocessing, ToSymmetric, ToUnitInterval, UniformDequantize, bits_per_dim

IMAGE_DATASETS = ("mnist", "fashion_mnist", "cifar10")

_IDX = {
    "mnist": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        {
            "train": (("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                      ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432")),
            "test": (("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
                     ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")),
        },
    ),
    "fashion_mnist": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        {
            "train": (("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
                      ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe")),
            "test": (("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
                     ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")),
        },
    ),
}
_CIFAR_BIN = ("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "c32a1d4ab5d03f1284b67883e8d87530")


# ------------------------------------------------------------------ raw I/O
def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    zero, dtype_code, ndim = data[0] * 256 + data[1], data[2], data[3]
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path} is not an unsigned-byte IDX file")
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    return arr.reshape(dims)


def _load_raw(name: str, split: str, root: Path, download: bool) -> Tuple[Tensor, Tensor]:
    d = root / "raw" / name
    if name in _IDX:
        url, files = _IDX[name]
        paths = []
        for fname, md5 in files[split]:
            p = d / fname
            if not p.exists():
                if not download:
                    raise FileNotFoundError(f"{p} not found and download=False")
                download_url(url + fname, p, md5=md5)
            paths.append(p)
        x = torch.from_numpy(_read_idx(paths[0]).copy()).unsqueeze(1)
        y = torch.from_numpy(_read_idx(paths[1]).astype(np.int64))
        return x, y
    if name == "cifar10":
        archive = d / "cifar-10-binary.tar.gz"
        if not archive.exists():
            if not download:
                raise FileNotFoundError(f"{archive} not found and download=False")
            download_url(_CIFAR_BIN[0], archive, md5=_CIFAR_BIN[1])
        names = [f"data_batch_{i}.bin" for i in range(1, 6)] if split == "train" else ["test_batch.bin"]
        recs = []
        with tarfile.open(archive, "r:gz") as tf:
            members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
            for n in names:
                f = tf.extractfile(members[n])
                assert f is not None
                recs.append(np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 1 + 3 * 32 * 32))
        rec = np.concatenate(recs)
        y = torch.from_numpy(rec[:, 0].astype(np.int64))
        x = torch.from_numpy(rec[:, 1:].copy()).reshape(-1, 3, 32, 32)
        return x, y
    raise ValueError(name)


def _load_torchvision(name: str, split: str, root: Path, download: bool) -> Tuple[Tensor, Tensor]:
    import torchvision

    cls = {"mnist": torchvision.datasets.MNIST, "fashion_mnist": torchvision.datasets.FashionMNIST,
           "cifar10": torchvision.datasets.CIFAR10}[name]
    ds = cls(str(root / "torchvision"), train=(split == "train"), download=download)
    data = ds.data
    x = torch.as_tensor(np.asarray(data)) if not isinstance(data, Tensor) else data
    if x.dim() == 3:  # (N, H, W)
        x = x.unsqueeze(1)
    else:  # CIFAR: (N, H, W, C)
        x = x.permute(0, 3, 1, 2)
    y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.int64)
    return x.contiguous().to(torch.uint8), y


def load_image_tensors(
    name: str,
    split: str = "train",
    root: Optional[PathLike] = None,
    download: bool = True,
    backend: str = "auto",
) -> Tuple[Tensor, Tensor]:
    """Raw ``uint8`` images ``(N, C, H, W)`` and ``int64`` labels ``(N,)``."""
    name = name.lower().replace("-", "_")
    if name == "fashionmnist":
        name = "fashion_mnist"
    if name not in IMAGE_DATASETS:
        raise ValueError(f"unknown image dataset {name!r}; choose from {IMAGE_DATASETS}")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    root_p = resolve_root(root)
    if backend == "auto":
        try:
            import torchvision  # noqa: F401

            backend = "torchvision"
        except ImportError:
            backend = "raw"
    if backend == "torchvision":
        return _load_torchvision(name, split, root_p, download)
    if backend == "raw":
        return _load_raw(name, split, root_p, download)
    raise ValueError(f"unknown backend {backend!r}")


# ------------------------------------------------------------------ dataset
class DequantizedImageDataset(Dataset):
    """Dataset of 8-bit images returning ``(y, label)`` with fresh dequantization noise.

    Args:
        images: ``uint8`` tensor ``(N, C, H, W)``.
        labels: ``(N,)`` labels.
        dequantize: add ``U[0,1)`` noise (otherwise ``x + 0.5`` is used, the bin centre).
        scale: ``"unit"`` ([0,1)) or ``"symmetric"`` ([-1,1)).
        flatten: return vectors ``(C·H·W,)`` instead of images.

    Attributes:
        event_shape, dim, preprocessing, preprocessing_log_det (per sample,
        w.r.t. ``x̃ ∈ [0,256)^D``), n_bins.

    Besides ``__getitem__`` it provides a vectorized :meth:`get_batch` (used by
    :func:`cjd_flows.train.fit`) and :meth:`to` to keep the ``uint8`` storage on
    the GPU.
    """

    n_bins = 256

    def __init__(
        self,
        images: Tensor,
        labels: Tensor,
        dequantize: bool = True,
        scale: str = "symmetric",
        flatten: bool = False,
    ):
        if images.dtype != torch.uint8:
            raise TypeError("images must be uint8")
        self.images = images
        self.labels = labels
        self.dequantize = dequantize
        self.flatten = flatten
        if scale == "unit":
            self.scaling: Preprocessing = ToUnitInterval(self.n_bins)
        elif scale == "symmetric":
            self.scaling = ToSymmetric(self.n_bins)
        else:
            raise ValueError("scale must be 'unit' or 'symmetric'")
        self.scale = scale
        self.preprocessing = Compose(UniformDequantize(self.n_bins), self.scaling) if dequantize else self.scaling
        shape = tuple(images.shape[1:])
        self.event_shape: Tuple[int, ...] = (math.prod(shape),) if flatten else shape
        self.dim = math.prod(shape)
        self.preprocessing_log_det = self.scaling.constant_log_abs_det(self.dim)

    def __len__(self) -> int:
        return self.images.shape[0]

    def to(self, device) -> "DequantizedImageDataset":
        """Moves the ``uint8`` storage to ``device`` (in place); returns ``self``."""
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        return self

    def _process(self, x: Tensor) -> Tensor:
        x = x.float()
        x = x + torch.rand_like(x) if self.dequantize else x + 0.5
        y = self.scaling(x)
        return y.flatten(1) if self.flatten else y

    def get_batch(self, idx: Union[Tensor, Sequence[int]]) -> Tuple[Tensor, Tensor]:
        """Vectorized access: ``(y, labels)`` for a batch of indices."""
        idx = torch.as_tensor(idx, device=self.images.device)
        return self._process(self.images[idx]), self.labels[idx]

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        return self._process(self.images[i].unsqueeze(0))[0], self.labels[i]

    def bits_per_dim(self, log_prob: Union[Tensor, float]) -> Union[Tensor, float]:
        """Bits/dim of model log-likelihoods (nats, model space) on this dataset."""
        return bits_per_dim(log_prob, self.dim, self.preprocessing_log_det)

    def to_raw(self, y: Tensor) -> Tensor:
        """Maps model-space data back to ``uint8`` images (quantizing)."""
        x = self.scaling.inverse(y).floor().clamp(0, self.n_bins - 1)
        if self.flatten:
            x = x.reshape(-1, *self.images.shape[1:])
        return x.to(torch.uint8)


def load_images(
    name: str,
    root: Optional[PathLike] = None,
    classes: Optional[Union[int, Sequence[int]]] = None,
    split: str = "train",
    dequantize: bool = True,
    scale: str = "symmetric",
    flatten: bool = False,
    download: bool = True,
    backend: str = "auto",
    exclude_classes: bool = False,
) -> DequantizedImageDataset:
    """Loads MNIST / Fashion-MNIST / CIFAR-10 as a dequantized dataset.

    Args:
        name: ``"mnist"``, ``"fashion_mnist"`` or ``"cifar10"``.
        root: cache directory (default ``$CJD_FLOWS_DATA`` or ``~/.cache/cjd_flows``).
        classes: keep only these labels (one-class / subset filter).
        split: ``"train"`` or ``"test"``.
        dequantize: uniform dequantization (fresh noise on every access).
        scale: ``"symmetric"`` ([-1, 1)) or ``"unit"`` ([0, 1)).
        flatten: return flat vectors.
        download: download if missing.
        backend: ``"auto"`` (torchvision if installed), ``"torchvision"`` or ``"raw"``.
        exclude_classes: if ``True`` keep all labels *except* ``classes``
            (handy for the "rest" set in one-class protocols).

    Returns:
        A :class:`DequantizedImageDataset`.
    """
    x, y = load_image_tensors(name, split, root, download, backend)
    if classes is not None:
        cl = torch.as_tensor([classes] if isinstance(classes, int) else list(classes))
        mask = torch.isin(y, cl)
        if exclude_classes:
            mask = ~mask
        x, y = x[mask], y[mask]
    return DequantizedImageDataset(x, y, dequantize=dequantize, scale=scale, flatten=flatten)
