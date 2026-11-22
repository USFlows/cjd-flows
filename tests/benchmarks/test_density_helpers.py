import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
run = pytest.importorskip("benchmarks.density.run")


def test_spearman_matches_scipy_with_ties():
    stats = pytest.importorskip("scipy.stats")
    torch.manual_seed(0)
    a = torch.randint(0, 5, (200,)).double()  # many ties
    b = a + torch.randn(200, dtype=torch.float64)
    assert run.spearman(a, b) == pytest.approx(stats.spearmanr(a.numpy(), b.numpy()).statistic, abs=1e-12)


def test_spearman_undefined_for_constant_input():
    a = torch.randn(50, dtype=torch.float64)
    assert run.spearman(a, torch.full((50,), -3.4657, dtype=torch.float64)) is None
    assert run.spearman(a, a) == pytest.approx(1.0)


def test_run_one_records_convergence_information():
    spec = dict(dataset="two_moons", base="gaussian", coupling="additive", mixing="lu", seed=0, blocks=2,
                hidden=[16], steps=40, batch_size=256, threads=1)
    rec = run.run_one(spec, meta={})
    assert rec["ok"], rec.get("error")
    assert rec["epochs_run"] >= 1 and 1 <= rec["best_epoch"] <= rec["epochs_run"]
    assert len(rec["val_curve"]) == rec["epochs_run"]
    assert min(rec["val_curve"]) == pytest.approx(rec["best_val_nll"])
