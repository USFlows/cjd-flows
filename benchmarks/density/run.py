"""CPU density-estimation benchmarks (toy 2-D, asymmetric GMM, MAF tabular suite).

Runs a grid of (dataset x base x coupling x seed) configurations in a process
pool and appends one JSON record per run to ``<out>/results.jsonl``.

Usage::

    PYTHONPATH=src:. python benchmarks/density/run.py --suite toy --workers 8
    PYTHONPATH=src:. python benchmarks/density/run.py --suite tabular --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")


def spearman(a, b) -> float:
    import torch

    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


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


def run_one(spec) -> dict:
    import torch

    torch.set_num_threads(spec.get("threads", 4))
    from cjd_flows.models import USFlow
    from cjd_flows.train import evaluate_nll, fit

    rec = dict(spec)
    try:
        t0 = time.time()
        tr, va, te, exact = load_data(spec)
        D = tr.shape[1]
        torch.manual_seed(spec["seed"])
        flow = USFlow(
            (D,),
            blocks=spec["blocks"],
            coupling=spec["coupling"],
            hidden=tuple(spec["hidden"]),
            mixing=spec.get("mixing", "lu"),
            base=spec["base"],
            p=spec.get("p", 2.0),
            n_components=spec.get("n_components", 4),
        )
        rec["params"] = sum(p.numel() for p in flow.parameters())
        steps = spec["steps"]
        bs = spec.get("batch_size", 512)
        epochs = max(1, math.ceil(steps * bs / len(tr)))
        res = fit(
            flow, tr, va, epochs=epochs, max_steps=steps, batch_size=bs, lr=spec.get("lr", 1e-3),
            weight_decay=spec.get("weight_decay", 0.0), ema_decay=0.999, grad_clip=5.0,
            device="cpu", restore="best", val_batch_size=8192,
        )
        flow.eval()
        rec["test_nll"] = float(evaluate_nll(flow, te, batch_size=8192, device="cpu"))
        rec["best_val_nll"] = float(res.best_val_loss) if res.best_val_loss is not None else None
        rec["steps_done"] = res.steps
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
            rec["log_abs_det"] = float(flow.log_abs_det())
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
        cfg.items(), ["gaussian", "ggmix", "spline", "radial-mixture"], ["additive", "autoregressive"], seeds
    ):
        specs.append(dict(dataset=ds, base=base, coupling=coup, seed=seed, threads=4, **c))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["toy", "tabular", "smoke"], default="toy")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="benchmarks/runs/density")
    ap.add_argument("--filter", default=None, help="substring filter on dataset name")
    ap.add_argument("--bases", nargs="+", default=None, help="restrict base distributions")
    ap.add_argument("--couplings", nargs="+", default=None, help="restrict coupling types")
    ap.add_argument("--steps-scale", type=float, default=1.0, help="multiply the step budget")
    ap.add_argument("--threads", type=int, default=None, help="torch threads per worker")
    args = ap.parse_args()

    if args.suite == "toy":
        specs = toy_grid(args.seeds)
    elif args.suite == "tabular":
        specs = tabular_grid(args.seeds)
    else:
        specs = [dict(dataset="two_moons", base=b, coupling=c, seed=0, blocks=4, hidden=[32, 32], steps=200, threads=2)
                 for b in ("gaussian", "ggmix") for c in ("additive", "gin")]
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
    done = set()
    res_file = out / "results.jsonl"
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                done.add(json.dumps({k: r[k] for k in specs[0] if k in r}, sort_keys=True))
    specs = [s for s in specs if json.dumps(s, sort_keys=True) not in done]
    print(f"{len(specs)} runs to do", flush=True)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool, open(res_file, "a") as f:
        for rec in pool.imap_unordered(run_one, specs):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            msg = rec.get("error") or f"test_nll={rec.get('test_nll'):.4f} kl={rec.get('kl_estimate')}"
            print(f"[{rec['dataset']}{rec.get('d','')}|{rec['base']}|{rec['coupling']}|s{rec['seed']}] {msg} "
                  f"({rec.get('minutes', 0):.1f} min)", flush=True)


if __name__ == "__main__":
    main()
