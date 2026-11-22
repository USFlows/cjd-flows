import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
report = pytest.importorskip("benchmarks.report")  # repository-only (not in the sdist)
arch_section, images_section, load_jsonl = report.arch_section, report.images_section, report.load_jsonl

META = {"library_version": "1.0.0", "git_commit": "abcdef0123", "git_dirty": False, "device": "cpu"}


def _write(path, records, tail=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)


def test_load_jsonl_skips_unterminated_tail(tmp_path):
    f = tmp_path / "r.jsonl"
    _write(f, [{"a": 1}, {"a": 2}], tail='{"a": 3, "b"')
    assert load_jsonl(f) == [{"a": 1}, {"a": 2}]
    assert load_jsonl(tmp_path / "missing.jsonl") == []


def _oneclass(ds, cls, coupling, base, auroc, best_epoch=20):
    metrics = {"nll": {"auroc": auroc}, "latent_norm": {"auroc": auroc - 0.01}}
    return dict(task="oneclass", dataset=ds, cls=cls, coupling=coupling, base=base, seed=0, epochs=20,
                best_epoch=best_epoch, normal_test_bpd=3.0, metrics=metrics, meta=META, ok=True)


def test_images_section(tmp_path):
    runs = tmp_path
    oc = [_oneclass("mnist", c, cp, "gaussian", 0.8 + 0.01 * c) for c in range(3) for cp in ("additive", "gin")]
    oc.append(dict(_oneclass("mnist", 0, "gin", "gaussian", 0.1), ok=False))  # failed runs are ignored
    _write(runs / "images" / "oneclass" / "results_shard0.jsonl", oc[:3])
    _write(runs / "images" / "oneclass" / "results_shard1.jsonl", oc[3:])
    dens = [dict(task="density", dataset="mnist", cls=None, coupling="gin", base="gaussian", seed=s, epochs=20,
                 best_epoch=12, test_bpd=1.5 + 0.1 * s, meta=META, ok=True) for s in (0, 1)]
    _write(runs / "images" / "density" / "results_shard0.jsonl", dens)
    md = images_section(runs)
    assert "## MNIST / Fashion-MNIST density" in md
    assert "| mnist, 20 ep / gin | **1.550 ± 0.071** |" in md
    assert "last epoch in 0 of 2 runs." in md
    assert "### Score: nll" in md and "### Score: latent_norm" in md
    # mean over the three classes 0.80, 0.81, 0.82
    assert "| mnist, 20 ep / additive | **0.810 ± 0.010** |" in md
    assert "| mnist, 20 ep / 2 | **0.820** | **0.820** |" in md  # per-class table, ties are both bold
    assert "last epoch in 6 of 6 runs (possible sign that the budget is too short" in md
    assert "Incomplete sweep" not in md
    assert images_section(tmp_path / "empty") == ""
    _write(runs / "images" / "oneclass" / "results_shard2.jsonl", [_oneclass("mnist", 3, "gin", "gaussian", 0.9)])
    assert "on mnist, 20 ep the cells average over different sets of classes" in images_section(runs)
    # equal counts but different classes are flagged as well: additive {1, 2, 3}, gin {0, 1, 2}
    _write(runs / "images" / "oneclass" / "results_shard2.jsonl", [_oneclass("mnist", 3, "additive", "gaussian", 0.9)])
    _write(runs / "images" / "oneclass" / "results_shard0.jsonl", oc[1:3])  # drops (class 0, additive)
    assert "different sets of classes" in images_section(runs)


def test_images_normal_bpd_is_averaged_per_class(tmp_path):
    recs = [dict(_oneclass("mnist", 0, "gin", "gaussian", 0.9), seed=0, normal_test_bpd=1.0),
            dict(_oneclass("mnist", 0, "gin", "gaussian", 0.9), seed=1, normal_test_bpd=3.0),
            dict(_oneclass("mnist", 1, "gin", "gaussian", 0.9), seed=0, normal_test_bpd=5.0)]
    _write(tmp_path / "images" / "oneclass" / "results_shard0.jsonl", recs)
    md = images_section(tmp_path)
    assert "| mnist, 20 ep / gin | **3.500 ± 2.121** |" in md  # classes 2.0 and 5.0, not runs 1, 3, 5


def test_arch_alignment_tables(tmp_path):
    recs = [dict(dataset="asymmetric_gmm", d=8, coupling=c, base="gaussian", mixing="lu", seed=s, test_nll=13.0,
                 spearman_norm_vs_true=nt, spearman_model_vs_true=0.99, meta=META, ok=True)
            for c, nt in (("additive", 0.99), ("affine", 0.8)) for s in (0, 1)]
    _write(tmp_path / "density" / "arch" / "results.jsonl", recs)
    md = arch_section(tmp_path)
    assert "between −latent norm and true log-density" in md
    assert "between model log-density and true log-density" in md
    assert "| gmm d=8 / affine | **0.800 ± 0.000** |" in md


def test_images_budgets_are_not_pooled(tmp_path):
    recs = [_oneclass("mnist", 0, "gin", "gaussian", 0.8), dict(_oneclass("mnist", 0, "gin", "gaussian", 0.9), epochs=100)]
    _write(tmp_path / "images" / "oneclass" / "results_shard0.jsonl", recs)
    md = images_section(tmp_path)
    assert "| mnist, 20 ep / gin | **0.800** |" in md and "| mnist, 100 ep / gin | **0.900** |" in md


def test_last_epoch_note_ignores_single_epoch_runs(tmp_path):
    _write(tmp_path / "images" / "oneclass" / "results_shard0.jsonl",
           [dict(_oneclass("mnist", 0, "gin", "gaussian", 0.8), epochs=1, best_epoch=1)])
    assert "Selected checkpoint" not in images_section(tmp_path)
