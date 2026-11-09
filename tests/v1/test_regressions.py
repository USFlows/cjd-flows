"""Regression tests for bugs found in the v1 review."""

import math

import pytest
import torch

from cjd_flows import Flow, StandardNormal, USFlow
from cjd_flows.bijections import ActNorm, LULinear, Squeeze
from cjd_flows.models import BASES
from cjd_flows.radial import _numerics
from cjd_flows.testing import check_flow_normalization
from cjd_flows.train import NoiseAugmentation, _make_optimizer, fit


# ------------------------------------------------------------------ Flow shapes
def test_flow_infers_event_shape_from_shape_changing_first_layer():
    sq = Squeeze((1, 4, 4))
    flow = Flow([sq, LULinear(sq.out_shape)], StandardNormal(sq.out_shape))
    assert flow.event_shape == (1, 4, 4)
    assert flow.latent_shape == (4, 2, 2)
    x = flow.sample((3,))
    assert x.shape == (3, 1, 4, 4)
    # sampling must be the generative map, not a reshape of latents
    torch.manual_seed(0)
    z = flow.base.sample((3,))
    torch.manual_seed(0)
    assert torch.allclose(flow.sample((3,)), flow.generate(z))
    assert flow.log_prob(x).shape == (3,)


def test_flow_infers_event_shape_from_inverse_shape_changing_layer():
    sq = Squeeze((1, 4, 4))
    flow = Flow([sq.inv], StandardNormal((1, 4, 4)))
    assert flow.event_shape == (4, 2, 2)
    assert flow.sample((2,)).shape == (2, 4, 2, 2)


def test_flow_rejects_incompatible_event_shape():
    with pytest.raises(ValueError):
        Flow([LULinear((3,))], StandardNormal((4,)), event_shape=(3,))


# ------------------------------------------------------------------ testing utils
def test_check_flow_normalization_without_parameters():
    sq = Squeeze((1, 2, 2))
    flow = Flow([sq], StandardNormal(sq.out_shape))
    # 4-D: crude MC estimate, only checks that it runs and is in the right ballpark
    assert 0.5 < check_flow_normalization(flow, n=50_000, radius=5.0) < 1.5


def test_check_flow_normalization_does_not_data_init_actnorm():
    flow = Flow([ActNorm((2,))], StandardNormal((2,)))
    assert flow.training
    check_flow_normalization(flow, n=1000)
    assert flow.training  # mode restored
    assert not bool(flow.bijection[0].initialized)
    assert torch.all(flow.bijection[0].log_scale == 0)


# ------------------------------------------------------------------ training
def test_soft_training_with_batch_context_and_validation():
    torch.manual_seed(0)
    flow = USFlow((2,), blocks=2, hidden=(8,), context_dim=2)
    x = torch.randn(128, 2)
    y = torch.randint(0, 2, (128,)).float()
    ds = torch.utils.data.TensorDataset(x, y)
    res = fit(flow, ds, ds, epochs=1, batch_size=64, batch_context=True, noise=NoiseAugmentation("soft", sigma_max=0.1))
    assert len(res.val_losses) == 1 and math.isfinite(res.val_losses[0])


def test_fp16_nonfinite_loss_is_skipped():
    torch.manual_seed(0)
    flow = USFlow((2,), blocks=1, hidden=(8,))
    x = torch.randn(64, 2)
    x[0, 0] = float("inf")
    with pytest.warns(UserWarning, match="aborting"):
        res = fit(flow, x, epochs=1, batch_size=64, amp="fp16", max_nonfinite=1)
    assert res.skipped_steps == 1 and res.aborted


def test_scale_lr_group_contains_only_scalars_and_base():
    flow = USFlow((2,), blocks=1, hidden=(8,), base="gg")
    opt = _make_optimizer(flow, "adamw", 1e-3, 0.0, (0.9, 0.99), scale_lr_mult=10.0)
    fast = [g for g in opt.param_groups if g["lr"] == pytest.approx(1e-2)]
    assert len(fast) == 1
    ids = {id(p) for p in fast[0]["params"]}
    base_ids = {id(p) for p in flow.base.parameters()}
    for p in fast[0]["params"]:
        assert p.dim() == 0 or id(p) in base_ids
    # the conditioner's 1-element output bias (2-D data, split coupling) is a plain weight
    coupling = flow.bijection[1]
    assert coupling.conditioner.out.bias.numel() == 1
    assert id(coupling.conditioner.out.bias) not in ids
    assert id(flow.bijection[-1].log_scale) in ids  # GlobalScale


# ------------------------------------------------------------------ ONNX-friendly numerics
def test_lanczos_lgamma_matches_torch():
    x = torch.cat([torch.logspace(-4, 4, 200, dtype=torch.float64), torch.tensor([0.5, 1.0, 2.0, 1536.0], dtype=torch.float64)])
    ref = torch.lgamma(x)
    got = _numerics._lgamma_lanczos(x)
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_export_numerics_paths_match(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(1000, dtype=torch.float64) * 30
    s = torch.sort(torch.rand(50, dtype=torch.float64)).values
    q = torch.rand(300, dtype=torch.float64) * 1.2 - 0.1
    ref = (
        _numerics.log_expm1_ratio(x),
        _numerics.logcumsumexp(x),
        _numerics.searchsorted_right(s, q),
        _numerics.lgamma(x.abs() + 1e-3),
    )
    monkeypatch.setattr(_numerics, "_exporting", lambda: True)
    got = (
        _numerics.log_expm1_ratio(x),
        _numerics.logcumsumexp(x),
        _numerics.searchsorted_right(s, q),
        _numerics.lgamma(x.abs() + 1e-3),
    )
    assert torch.allclose(got[0], ref[0], atol=1e-9)
    assert torch.allclose(got[1], ref[1], atol=1e-9)
    assert torch.equal(got[2], ref[2])
    assert torch.allclose(got[3], ref[3], rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------------ ONNX export
def _onnx_roundtrip(flow, tmp_path, mode, x):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from cjd_flows.export import to_onnx, verify_onnx

    path = to_onnx(flow, tmp_path / f"{mode}.onnx", mode=mode, batch_size=2)
    return verify_onnx(flow, path, x)


@pytest.mark.parametrize("base", [b for b in BASES if b != "gaussian"])
def test_onnx_log_prob_radial_bases(tmp_path, base):
    torch.manual_seed(0)
    flow = USFlow((4,), blocks=2, hidden=(8,), base=base).eval()
    _onnx_roundtrip(flow, tmp_path, "log_prob", torch.randn(5, 4))


@pytest.mark.parametrize("mixing", ["lu", "orthogonal", "householder", "actnorm"])
@pytest.mark.parametrize("mode", ["forward", "inverse"])
def test_onnx_linear_layers_both_directions(tmp_path, mixing, mode):
    torch.manual_seed(0)
    flow = USFlow((2, 4, 4), blocks=2, hidden=(8,), mixing=mixing, conjugate=True)
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.05 * torch.randn_like(p))
    _onnx_roundtrip(flow.eval(), tmp_path, mode, torch.randn(3, 2, 4, 4))


def test_onnx_inverse_of_squeeze_flow_uses_latent_shape(tmp_path):
    sq = Squeeze((1, 4, 4))
    flow = Flow([sq, LULinear(sq.out_shape)], StandardNormal(sq.out_shape)).eval()
    _onnx_roundtrip(flow, tmp_path, "inverse", torch.randn(3, *sq.out_shape))
    _onnx_roundtrip(flow, tmp_path, "log_prob", torch.randn(3, 1, 4, 4))
