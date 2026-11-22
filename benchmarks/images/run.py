"""MNIST / Fashion-MNIST benchmarks on the GPU: density (bits/dim) and one-class anomaly detection.

Compact matrix: couplings {additive, gin, affine reference} x bases {gaussian,
ggmix-monotone, radial-mixture}; ConvNet conditioners (USFlow defaults for
images), 8 blocks, hidden (128, 128), ~1.2M parameters for every variant.

Protocol: fixed epoch budget; the validation split is carved out of the training
data (fixed seed, identical for every run) and selects the checkpoint (best
validation NLL, ``restore="best"``; no early stopping, so every model gets the
same budget). The test split is evaluated once, for the selected model. Runs
whose training aborts (non-finite losses) are recorded as failed.

For the affine reference (no constant determinant) the ``nll`` anomaly score is
computed with its own ``log_prob``: :func:`cjd_flows.scores.anomaly_scores`
assumes the library contract (constant ``log_abs_det``) and would drop the
input-dependent coupling determinants.

* ``density``: train on the training set (minus 5000 validation images), report
  test bits/dim.
* ``oneclass``: for each class c, train on the training images of class c (minus
  10 % validation), score the test set (class c = normal, all other classes =
  anomalous) with every available anomaly score and report AUROC / AUPR /
  FPR@95.

Each run is one (task, dataset, class, coupling, base, seed). Runs can be split
across processes with ``--shard k --num-shards n`` (one JSONL file per shard)
and are resumed on restart.

Usage::

    PYTHONPATH=src:. MIOPEN_FIND_MODE=FAST python benchmarks/images/run.py \\
        --task oneclass --datasets mnist fashion_mnist --seeds 0 --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import traceback
from pathlib import Path

COUPLINGS = ["additive", "gin", "affine"]
BASES = ["gaussian", "ggmix-monotone", "radial-mixture"]
SHAPE = (1, 28, 28)


def grid(task, datasets, seeds, couplings=COUPLINGS, bases=BASES, classes=range(10), epochs=20):
    specs = []
    for ds, coupling, base, seed in itertools.product(datasets, couplings, bases, seeds):
        common = dict(task=task, dataset=ds, coupling=coupling, base=base, seed=seed, blocks=8,
                      hidden=[128, 128], epochs=epochs, batch_size=128, lr=1e-3)
        if task == "density":
            specs.append(dict(common, cls=None))
        else:
            specs += [dict(common, cls=c) for c in classes]
    return specs


def build_model(spec):
    import torch

    torch.manual_seed(spec["seed"])
    if spec["coupling"] == "affine":
        from benchmarks.baselines import affine_reference_flow

        return affine_reference_flow(SHAPE, blocks=spec["blocks"], hidden=tuple(spec["hidden"]), base=spec["base"])
    from cjd_flows.models import USFlow

    return USFlow(SHAPE, blocks=spec["blocks"], hidden=tuple(spec["hidden"]), coupling=spec["coupling"],
                  base=spec["base"])


def _split(dataset, n_val):
    """Fixed train/validation split (same for every run and seed)."""
    import torch
    from torch.utils.data import random_split

    n_val = int(n_val)
    return random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(0))


def load_data(spec):
    from cjd_flows.data import load_images

    ds = spec["dataset"]
    if spec["task"] == "density":
        train, val = _split(load_images(ds, split="train"), 5000)
        return train, val, load_images(ds, split="test"), None
    c = spec["cls"]
    full = load_images(ds, split="train", classes=c)
    train, val = _split(full, round(0.1 * len(full)))
    test_in = load_images(ds, split="test", classes=c)
    test_out = load_images(ds, split="test", classes=c, exclude_classes=True)
    return train, val, test_in, test_out


def _nll_scores(flow, data, batch_size=1000):
    """Per-sample ``-log p(x)`` from the flow's own ``log_prob`` (on the GPU)."""
    import torch
    from torch.utils.data import DataLoader

    out = []
    for batch in DataLoader(data, batch_size=batch_size):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(-flow.log_prob(x.to("cuda")).float().cpu())
    return torch.cat(out)


def _read_records(path):
    """Complete JSON records of a results file.

    Only newline-terminated lines count as complete: an unterminated last line
    (interrupted append) is ignored here and truncated before the next append,
    so its run is redone. A malformed terminated line raises.
    """
    text = path.read_text()
    lines = text.split("\n")[:-1]  # the last segment is either "" or an unterminated tail
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError(f"{path}:{i + 1}: malformed record") from None
    return records


def run_one(spec, meta) -> dict:
    import torch

    from benchmarks.common import count_parameters
    from cjd_flows.data import bits_per_dim
    from cjd_flows.scores import anomaly_scores, detection_metrics
    from cjd_flows.train import evaluate_nll, fit

    rec = dict(spec, meta=meta)
    try:
        t0 = time.time()
        train, val, test_a, test_b = load_data(spec)
        flow = build_model(spec)
        rec["params"] = count_parameters(flow)
        res = fit(flow, train, val, epochs=spec["epochs"], batch_size=spec["batch_size"], lr=spec["lr"],
                  ema_decay=0.999, grad_clip=5.0, device="cuda", restore="best", val_batch_size=1000,
                  verbose=False, seed=spec["seed"])
        flow.eval()
        rec.update(best_epoch=res.best_epoch, restored=res.restored, steps_done=res.steps,
                   best_val_nll=float(res.best_val_loss) if res.best_val_loss is not None else None)
        if res.aborted:
            raise RuntimeError(f"training aborted after {res.steps} steps (non-finite losses)")
        ref = test_a.dataset if hasattr(test_a, "dataset") else test_a
        dim, pld = ref.dim, ref.preprocessing_log_det
        with torch.no_grad():
            nll = float(evaluate_nll(flow, test_a, batch_size=1000, device="cuda"))
            rec["test_bpd" if spec["task"] == "density" else "normal_test_bpd"] = float(bits_per_dim(-nll, dim, pld))
            if spec["task"] == "oneclass":
                s_in = anomaly_scores(flow, test_a, device="cuda", batch_size=1000)
                s_out = anomaly_scores(flow, test_b, device="cuda", batch_size=1000)
                if spec["coupling"] == "affine":  # input-dependent determinants: use the flow's log_prob
                    s_in["nll"], s_out["nll"] = _nll_scores(flow, test_a), _nll_scores(flow, test_b)
                rec["metrics"] = {k: detection_metrics(s_in[k], s_out[k]) for k in s_in}
        rec["minutes"] = (time.time() - t0) / 60
        rec["ok"] = True
    except Exception as e:  # record failures instead of stopping the sweep
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["density", "oneclass"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--couplings", nargs="+", default=COUPLINGS)
    ap.add_argument("--bases", nargs="+", default=BASES)
    ap.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="benchmarks/runs/images")
    args = ap.parse_args()

    from benchmarks.common import run_metadata

    meta = run_metadata("cuda")  # source and hardware of this process, captured before any run
    specs = grid(args.task, args.datasets, args.seeds, args.couplings, args.bases, args.classes, args.epochs)
    specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]
    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)

    keys = sorted({k for s in specs for k in s})

    def resume_key(d):
        return json.dumps({k: d.get(k) for k in keys}, sort_keys=True)

    done = set()
    for f in out.glob("results*.jsonl"):
        for r in _read_records(f):
            if r.get("ok"):
                done.add(resume_key(r))
    specs = [s for s in specs if resume_key(s) not in done]
    print(f"[shard {args.shard}] {len(specs)} runs to do", flush=True)
    own = out / f"results_shard{args.shard}.jsonl"
    if own.exists():
        data = own.read_bytes()
        if data and not data.endswith(b"\n"):  # drop an incomplete tail (its run is re-done)
            own.write_bytes(data[: data.rfind(b"\n") + 1])
    with open(own, "a") as f:
        for spec in specs:
            rec = run_one(spec, meta)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            tag = f"{spec['dataset']}|c{spec['cls']}|{spec['coupling']}|{spec['base']}|s{spec['seed']}"
            if rec["ok"]:
                main_metric = rec.get("test_bpd") or rec["metrics"].get("nll", {}).get("auroc")
                print(f"[{tag}] {main_metric:.4f} ({rec['minutes']:.1f} min)", flush=True)
            else:
                print(f"[{tag}] FAILED {rec['error']}", flush=True)


if __name__ == "__main__":
    main()
