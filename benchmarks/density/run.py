"""Density-estimation benchmarks (toy 2-D, asymmetric GMM, MAF tabular suite).

Runs a grid of (dataset x base x coupling x seed) configurations in a process
pool and appends one JSON record per run to ``<out>/results.jsonl``. Runs on the
CPU by default; ``--device cuda`` trains on a GPU, with every pool worker its own
GPU process (keep ``--workers`` small). The device is not part of a run's
identity: a sweep can be resumed on another device, and ``meta.device`` records
where each run happened.

Suites: ``toy`` (bases x couplings), ``arch`` (couplings x mixing layers, incl. the
affine reference without constant determinant), ``tabular`` (MAF suite: additive,
autoregressive and the affine reference x four bases), ``smoke``.

Usage::

    PYTHONPATH=src:. python benchmarks/density/run.py --suite toy --workers 8
    PYTHONPATH=src:. python benchmarks/density/run.py --suite tabular --workers 8
    PYTHONPATH=src:. python benchmarks/density/run.py --suite tabular --workers 2 --device cuda
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import time
import traceback
from pathlib import Path


def _average_ranks(a):
    """Ranks with ties replaced by their average rank (as in scipy.stats.rankdata)."""
    import torch

    values, order = a.sort()
    _, inverse, counts = torch.unique_consecutive(values, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0)
    average = (ends - counts + ends - 1).double() / 2
    ranks = torch.empty(a.shape, dtype=torch.float64)
    ranks[order] = average[inverse]
    return ranks


def spearman(a, b):
    """Spearman rank correlation with tie handling; ``None`` if either input is constant (undefined)."""
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) == 0.0:
        return None
    return float((ra * rb).sum() / denom)


def load_data(spec):
    import torch

    from cjd_flows.data import toy

    name = spec["dataset"]
    if name == "asymmetric_gmm":
        d = spec["d"]
        g = toy.asymmetric_gmm(d)
        torch.manual_seed(1000 + spec["seed"])
        tr, va, te = (g.sample(n).float() for n in (50_000, 10_000, 20_000))
        return tr, va, te, lambda x: g.log_prob(x.double())
    if name in toy.TOY_DATASETS:
        tr = toy.make_toy(name, 50_000, seed=10 + spec["seed"])
        va = toy.make_toy(name, 10_000, seed=20 + spec["seed"])
        te = toy.make_toy(name, 20_000, seed=30 + spec["seed"])
        try:
            toy.toy_log_prob(name, te[:2])
            exact = lambda x: toy.toy_log_prob(name, x)  # noqa: E731
        except NotImplementedError:
            exact = None
        return tr, va, te, exact
    from cjd_flows.data.tabular import load_tabular

    tr, va, te = load_tabular(name)
    return tr, va, te, None


def build_model(spec, D):
    """USFlow variants or the affine reference (no constant determinant) for ``spec``."""
    common = dict(blocks=spec["blocks"], hidden=tuple(spec["hidden"]), mixing=spec.get("mixing", "lu"),
                  base=spec["base"], p=spec.get("p", 2.0), n_components=spec.get("n_components", 4))
    coupling = spec["coupling"]
    if coupling == "affine":
        from benchmarks.baselines import affine_reference_flow

        return affine_reference_flow((D,), **common)
    from cjd_flows.models import USFlow

    if coupling == "masked":  # VeriFlow-style: masked additive couplings in adjoint blocks, ReLU
        return USFlow((D,), coupling="additive", masked=True, conjugate=True, activation="relu", **common)
    return USFlow((D,), coupling=coupling, **common)


def run_one(spec, meta=None, device="cpu") -> dict:
    import torch

    torch.set_num_threads(spec.get("threads", 4))
    from benchmarks.common import count_parameters, device_name, run_metadata
    from cjd_flows.train import evaluate_nll, fit

    rec = dict(spec)
    # captured once at the start of the sweep (the checkout may change while it runs)
    rec["meta"] = dict(meta) if meta is not None else run_metadata("cpu")
    try:
        # resolved here, in the worker: the sweep's parent process stays off the GPU
        rec["meta"]["device"] = device_name(device)
        t0 = time.time()
        tr, va, te, exact = load_data(spec)
        D = tr.shape[1]
        torch.manual_seed(spec["seed"])
        flow = build_model(spec, D)
        rec["params"] = count_parameters(flow)
        steps = spec["steps"]
        bs = spec.get("batch_size", 512)
        epochs = max(1, math.ceil(steps * bs / len(tr)))
        res = fit(
            flow, tr, va, epochs=epochs, max_steps=steps, batch_size=bs, lr=spec.get("lr", 1e-3),
            weight_decay=spec.get("weight_decay", 0.0), ema_decay=0.999, grad_clip=5.0,
            device=device, restore="best", val_batch_size=8192,
        )
        flow.eval()
        rec["test_nll"] = float(evaluate_nll(flow, te, batch_size=8192, device=device))
        flow.to("cpu")  # the alignment statistics below compare with CPU reference densities
        rec["best_val_nll"] = float(res.best_val_loss) if res.best_val_loss is not None else None
        rec["steps_done"] = res.steps
        # convergence check: best epoch vs epochs run, and the validation curve
        rec["best_epoch"], rec["epochs_run"] = res.best_epoch, res.epochs
        rec["val_curve"] = [float(v) for v in res.val_losses]
        rec["aborted"] = bool(res.aborted)
        with torch.no_grad():
            x = te[:20_000]
            lq = flow.log_prob(x).double()
            z = flow.transform(x)
            r = z.flatten(1).norm(p=spec.get("p", 2.0) if spec.get("p", 2.0) != float("inf") else float("inf"), dim=1)
            rec["spearman_norm_vs_model"] = spearman(-r.double(), lq)
            if exact is not None:
                lp = exact(x).double()
                rec["kl_estimate"] = float((lp - lq).mean())
                rec["true_test_nll"] = float(-lp.mean())
                rec["spearman_norm_vs_true"] = spearman(-r.double(), lp)
                rec["spearman_model_vs_true"] = spearman(lq, lp)
            rec["log_abs_det"] = None if spec["coupling"] == "affine" else float(flow.log_abs_det())
        rec["minutes"] = (time.time() - t0) / 60
        rec["ok"] = True
    except Exception as e:  # record failures instead of crashing the pool
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    return rec


def toy_grid(seeds):
    specs = []
    for ds, base, coup, seed in itertools.product(
        ["two_moons", "eight_gaussians", "checkerboard", "rings", "pinwheel", "spirals"],
        ["gaussian", "ggmix", "ggmix-monotone", "spline", "khinchine", "radial-mixture"],
        ["additive", "gin"],
        seeds,
    ):
        specs.append(dict(dataset=ds, base=base, coupling=coup, seed=seed, blocks=8,
                          hidden=[128, 128], steps=6000, threads=2))
    for d, base, coup, seed in itertools.product(
        [2, 8, 32, 128], ["gaussian", "ggmix", "spline", "radial-mixture"], ["additive", "gin"], seeds
    ):
        specs.append(dict(dataset="asymmetric_gmm", d=d, base=base, coupling=coup, seed=seed,
                          blocks=8, hidden=[max(128, 4 * d)] * 2, steps=6000, threads=2))
    return specs


ARCH_COUPLINGS = ["additive", "masked", "gin", "autoregressive", "affine"]
ARCH_MIXINGS = ["lu", "orthogonal", "householder", "permutation"]


def arch_grid(seeds):
    """Architecture comparison: coupling x mixing for a Gaussian and a monotone radial base."""
    specs = []
    data = [("two_moons", None), ("eight_gaussians", None), ("checkerboard", None), ("rings", None),
            ("asymmetric_gmm", 8), ("asymmetric_gmm", 32)]
    for (ds, d), coup, mix, base, seed in itertools.product(
        data, ARCH_COUPLINGS, ARCH_MIXINGS, ["gaussian", "ggmix-monotone"], seeds
    ):
        spec = dict(dataset=ds, base=base, coupling=coup, mixing=mix, seed=seed, blocks=8, steps=6000, threads=2)
        if d is None:
            spec["hidden"] = [128, 128]
        else:
            spec.update(d=d, hidden=[max(128, 4 * d)] * 2)
        specs.append(spec)
    return specs


def tabular_grid(seeds):
    cfg = {
        "power": dict(blocks=10, hidden=[256, 256], steps=40_000),
        "gas": dict(blocks=10, hidden=[256, 256], steps=40_000),
        "hepmass": dict(blocks=10, hidden=[512, 512], steps=30_000),
        "miniboone": dict(blocks=10, hidden=[512, 512], steps=8_000, weight_decay=1e-4),
        "bsds300": dict(blocks=10, hidden=[512, 512], steps=30_000),
    }
    specs = []
    for (ds, c), base, coup, seed in itertools.product(
        cfg.items(), ["gaussian", "ggmix-monotone", "spline", "radial-mixture"], ["additive", "autoregressive", "affine"],
        seeds,
    ):
        specs.append(dict(dataset=ds, base=base, coupling=coup, mixing="lu", seed=seed, threads=4, **c))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["toy", "arch", "tabular", "smoke"], default="toy")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="benchmarks/runs/density")
    ap.add_argument("--filter", default=None, help="substring filter on dataset name")
    ap.add_argument("--bases", nargs="+", default=None, help="restrict base distributions")
    ap.add_argument("--couplings", nargs="+", default=None, help="restrict coupling types")
    ap.add_argument("--steps-scale", type=float, default=1.0, help="multiply the step budget")
    ap.add_argument("--threads", type=int, default=None, help="torch threads per worker")
    ap.add_argument("--device", default="cpu",
                    help="training device, e.g. cpu or cuda (each worker is one GPU process; keep --workers small)")
    args = ap.parse_args()
    if args.device == "cpu":  # keep CPU workers from initialising a GPU runtime (inherited by the pool)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

    if args.suite == "toy":
        specs = toy_grid(args.seeds)
    elif args.suite == "arch":
        specs = arch_grid(args.seeds)
    elif args.suite == "tabular":
        specs = tabular_grid(args.seeds)
    else:
        specs = [dict(dataset="two_moons", base=b, coupling=c, mixing="lu", seed=0, blocks=4, hidden=[32, 32],
                      steps=200, threads=2)
                 for b in ("gaussian", "ggmix") for c in ARCH_COUPLINGS]
    if args.filter:
        specs = [s for s in specs if args.filter in s["dataset"]]
    if args.bases:
        specs = [s for s in specs if s["base"] in args.bases]
    if args.couplings:
        specs = [s for s in specs if s["coupling"] in args.couplings]
    for s in specs:
        if args.steps_scale != 1.0:
            s["steps"] = int(s["steps"] * args.steps_scale)
        if args.threads:
            s["threads"] = args.threads
    out = Path(args.out) / args.suite
    out.mkdir(parents=True, exist_ok=True)
    # resume: compare on the union of all spec keys (specs differ, e.g. only GMM specs have ``d``)
    keys = sorted({k for s in specs for k in s})

    defaults = {"mixing": "lu"}  # records from before ``mixing`` was a spec field used LU

    def resume_key(d):
        return json.dumps({k: d.get(k, defaults.get(k)) for k in keys}, sort_keys=True)

    from benchmarks.common import drop_incomplete_tail, read_records, run_metadata

    done = set()
    res_file = out / "results.jsonl"
    if res_file.exists():
        for r in read_records(res_file):
            if r.get("ok"):
                done.add(resume_key(r))
    specs = [s for s in specs if resume_key(s) not in done]
    print(f"{len(specs)} runs to do", flush=True)
    drop_incomplete_tail(res_file)  # an interrupted append; its run is redone

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    meta = run_metadata("cpu")  # source of the whole sweep; each worker fills in its device
    with ctx.Pool(args.workers) as pool, open(res_file, "a") as f:
        for rec in pool.imap_unordered(functools.partial(run_one, meta=meta, device=args.device), specs):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            msg = rec.get("error") or f"test_nll={rec.get('test_nll'):.4f} kl={rec.get('kl_estimate')}"
            print(f"[{rec['dataset']}{rec.get('d','')}|{rec['base']}|{rec['coupling']}|s{rec['seed']}] {msg} "
                  f"({rec.get('minutes', 0):.1f} min)", flush=True)


if __name__ == "__main__":
    main()
