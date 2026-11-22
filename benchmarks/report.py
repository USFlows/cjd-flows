"""Aggregates benchmark results into markdown tables (written next to the run outputs, not versioned) and a JSON summary.

Usage::

    python benchmarks/report.py --runs benchmarks/runs --out benchmarks/runs/report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(vals, digits=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{mean(vals):.{digits}f} ± {stdev(vals):.{digits}f}"


def provenance(records) -> str:
    """One line stating which library versions, commits and devices produced ``records``."""
    metas = [r.get("meta") or {} for r in records]
    versions = sorted({m.get("library_version", "unknown") for m in metas})
    commits = sorted({(m.get("git_commit") or "unknown")[:7] + ("+dirty" if m.get("git_dirty") else "") for m in metas})
    devices = sorted({m.get("device", "unknown") for m in metas})
    seeds = sorted({r.get("seed") for r in records if r.get("seed") is not None})
    return (f"*{len(records)} runs; cjd-flows {', '.join(versions)}; commit {', '.join(commits)}; "
            f"device {', '.join(devices)}; seeds {seeds}.*\n")


def pivot(records, row_key, col_key, val_key, digits=3, lower_better=True):
    rows = sorted({row_key(r) for r in records}, key=str)
    cols = sorted({col_key(r) for r in records}, key=str)
    cell = defaultdict(list)
    for r in records:
        if r.get(val_key) is not None:
            cell[(row_key(r), col_key(r))].append(r[val_key])
    lines = ["| | " + " | ".join(map(str, cols)) + " |", "|---" * (len(cols) + 1) + "|"]
    for row in rows:
        means = {c: mean(cell[(row, c)]) for c in cols if cell[(row, c)]}
        best = (min if lower_better else max)(means.values()) if means else None
        out = []
        for c in cols:
            s = fmt(cell[(row, c)], digits)
            if cell[(row, c)] and best is not None and abs(means[c] - best) < 1e-12:
                s = f"**{s}**"
            out.append(s)
        lines.append(f"| {row} | " + " | ".join(out) + " |")
    return "\n".join(lines)


def density_section(runs: Path) -> str:
    out = []
    toy = [r for r in load_jsonl(runs / "density" / "toy" / "results.jsonl") if r.get("ok")]
    if toy:
        out.append("## Toy 2-D densities (test NLL in nats, lower is better)\n")
        out.append("8 coupling blocks, MLP conditioners (2×128), 6000 steps, EMA. Rows: dataset / coupling; columns: base distribution.\n")
        recs = [r for r in toy if r["dataset"] != "asymmetric_gmm"]
        out.append(pivot(recs, lambda r: f"{r['dataset']} / {r['coupling']}", lambda r: r["base"], "test_nll"))
        out.append("\n" + provenance(recs))
        kl = [r for r in recs if r.get("kl_estimate") is not None]
        if kl:
            out.append("\n\n### KL(p_data ‖ p_model) estimate (nats) on datasets with known density\n")
            out.append(pivot(kl, lambda r: f"{r['dataset']} / {r['coupling']}", lambda r: r["base"], "kl_estimate"))
            out.append("\n" + provenance(kl))
        gmm = [r for r in toy if r["dataset"] == "asymmetric_gmm"]
        if gmm:
            out.append("\n\n## Asymmetric GMM of the USF paper (d = 2 … 128)\n")
            out.append("### KL estimate (nats)\n")
            out.append(pivot(gmm, lambda r: f"d={r['d']:>3} / {r['coupling']}", lambda r: r["base"], "kl_estimate", 3))
            out.append("\n" + provenance(gmm))
            out.append("\n\n### Spearman correlation between −latent norm and true log-density (alignment; higher is better)\n")
            out.append(pivot(gmm, lambda r: f"d={r['d']:>3} / {r['coupling']}", lambda r: r["base"], "spearman_norm_vs_true", 3, lower_better=False))
            out.append("\n" + provenance(gmm))
    tab = [r for r in load_jsonl(runs / "density" / "tabular" / "results.jsonl") if r.get("ok")]
    if tab:
        out.append("\n\n## MAF tabular suite (test log-likelihood in nats, higher is better)\n")
        for r in tab:
            r["test_ll"] = -r["test_nll"]
        out.append(pivot(tab, lambda r: f"{r['dataset']} / {r['coupling']}", lambda r: r["base"], "test_ll", 2, lower_better=False))
        out.append("\n" + provenance(tab))
        out.append(
            "\n\nReference (test LL, nats): RealNVP power −0.17, gas 8.33, hepmass −18.71, miniboone −13.55, bsds300 153.28; "
            "Glow 0.17 / 8.15 / −18.92 / −11.35 / 155.07; MAF 0.24 / 10.08 / −17.70 / −11.75 / 155.69; "
            "RQ-NSF (AR) 0.66 / 13.09 / −14.01 / −9.22 / 157.31."
        )
    return "\n".join(out)


def arch_section(runs: Path) -> str:
    """Architecture comparison (density/run.py --suite arch): couplings x mixing layers per base."""
    recs = [r for r in load_jsonl(runs / "density" / "arch" / "results.jsonl") if r.get("ok")]
    if not recs:
        return ""

    def row(r):
        ds = f"gmm d={r['d']}" if r["dataset"] == "asymmetric_gmm" else r["dataset"]
        return f"{ds} / {r['coupling']}"

    out = ["\n\n## Architecture comparison (test NLL in nats, lower is better)\n",
           "8 blocks, MLP conditioners (2×128; GMM: 2×max(128, 4d)), 6000 steps, EMA, selection on the "
           "validation split. `masked` = VeriFlow-style masked couplings in adjoint LU blocks; `affine` = "
           "the same trunk with affine couplings (no constant determinant, benchmark-only reference).\n"]
    for base in sorted({r["base"] for r in recs}):
        sel = [r for r in recs if r["base"] == base]
        out.append(f"\n### Base: {base} (rows: dataset / coupling; columns: mixing layer)\n")
        out.append(pivot(sel, row, lambda r: r.get("mixing", "lu"), "test_nll"))
        out.append("\n" + provenance(sel))
    aligned = [r for r in recs if r.get("spearman_norm_vs_true") is not None]
    if aligned:
        out.append("\n### Alignment: Spearman correlation between −latent norm and true log-density "
                   "(higher is better; mean over mixing layers and seeds)\n")
        out.append(pivot(aligned, row, lambda r: r["base"], "spearman_norm_vs_true", 3, lower_better=False))
        out.append("\n" + provenance(aligned))
    return "\n".join(out)


def cifar_section(runs: Path) -> str:
    rows = []
    for d in sorted((runs / "cifar").glob("*-s*")):
        if not d.is_dir():
            continue
        args = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
        metrics = load_jsonl(d / "metrics.jsonl")
        evals = [m for m in metrics if "test_bpd" in m]
        final = json.loads((d / "final.json").read_text()) if (d / "final.json").exists() else None
        rows.append(
            dict(
                name=d.name,
                variant=args.get("variant"),
                size=args.get("size"),
                noise=args.get("noise", 0.0),
                params=next((m["n_params"] for m in metrics if "n_params" in m), None),
                last_eval=evals[-1]["test_bpd"] if evals else None,
                epoch=evals[-1]["epoch"] if evals else None,
                final=final["final_test_bpd"] if final else None,
                minutes=final["minutes"] if final else None,
                seed=args.get("seed"),
                meta=final.get("meta") if final else None,
            )
        )
    if not rows:
        return ""
    out = ["\n\n## CIFAR-10 scaling study (test bits/dim, lower is better)\n",
           "| run | variant | size | params | noise σ | test bpd (full test set) | last eval (2k subset) | epoch | minutes |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['name']} | {r['variant']} | {r['size']} | {r['params']/1e6 if r['params'] else 0:.1f}M | {r['noise']} | "
            f"{r['final'] if r['final'] is None else round(r['final'], 4)} | "
            f"{r['last_eval'] if r['last_eval'] is None else round(r['last_eval'], 4)} | "
            f"{r['epoch'] if r['epoch'] is None else round(r['epoch'], 1)} | "
            f"{r['minutes'] if r['minutes'] is None else round(r['minutes'], 1)} |"
        )
    out.append("\n" + provenance(rows))
    out.append("\nReference CIFAR-10 bpd: NICE ≈4.48, RealNVP 3.49, Glow 3.35, Flow++ 3.08, Jet (ImageNet-21k pretraining) 3.02.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="benchmarks/runs")
    ap.add_argument("--out", default="benchmarks/runs/report.md")
    args = ap.parse_args()
    runs = Path(args.runs)
    md = "# Benchmark results\n\nGenerated by `benchmarks/report.py`.\n\n" + density_section(runs) + arch_section(runs) + cifar_section(runs) + "\n"
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
