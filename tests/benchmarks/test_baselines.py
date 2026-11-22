"""Regression tests for the benchmark-only affine reference (non-constant determinant).

``benchmarks/`` is repository tooling and not part of the sdist, so these tests
skip when it is not available.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
baselines = pytest.importorskip("benchmarks.baselines")


def _perturbed(mixing, base, shape=(3,), scale=0.3):
    torch.manual_seed(0)
    flow = baselines.affine_reference_flow(shape, blocks=3, hidden=(16,), mixing=mixing, base=base).double()
    with torch.no_grad():
        for p in flow.parameters():  # move the conditioners away from their zero-output init
            p.add_(torch.randn_like(p) * scale)
    return flow.eval()


@pytest.mark.parametrize("mixing", ["lu", "orthogonal", "householder", "permutation"])
@pytest.mark.parametrize("base", ["gaussian", "ggmix"])
def test_affine_reference_log_prob_matches_autograd(mixing, base):
    flow = _perturbed(mixing, base)
    x = torch.randn(6, 3, dtype=torch.float64)
    z = flow.transform(x)
    logdet = torch.stack([
        torch.linalg.slogdet(torch.autograd.functional.jacobian(lambda v: flow.transform(v[None])[0], xi))[1]
        for xi in x
    ])
    assert torch.allclose(flow.log_prob(x), flow.base.log_prob(z) + logdet, rtol=0.0, atol=1e-10)
    assert torch.allclose(flow.bijection.inverse(z), x, rtol=0.0, atol=1e-10)
    assert logdet.std() > 1e-3  # the determinant really depends on x


def test_affine_coupling_keeps_float64_and_half_inputs():
    flow = _perturbed("lu", "gaussian")
    x = torch.randn(4, 3, dtype=torch.float64)
    assert flow.log_prob(x).dtype == torch.float64
    coupling = next(m for m in flow.modules() if isinstance(m, baselines.AffineCoupling))
    log_scale, _ = coupling._st(torch.randn(4, 2, dtype=torch.bfloat16))
    assert log_scale.dtype == torch.float32  # half precision is promoted, not kept


def test_affine_reference_image_shape():
    # smaller perturbation: ConvNet conditioners otherwise produce log-scales near -13, and the
    # inverse amplifies rounding errors by exp(-log_scale) (conditioning, not a precision bug)
    flow = _perturbed("lu", "gaussian", shape=(2, 4, 4), scale=0.05)
    x = torch.randn(3, 2, 4, 4, dtype=torch.float64)
    assert flow.log_prob(x).shape == (3,)
    assert torch.allclose(flow.bijection.inverse(flow.transform(x)), x, rtol=0.0, atol=1e-10)


def test_half_precision_flow_keeps_fp32_log_density():
    flow = _perturbed("permutation", "gaussian").to(torch.bfloat16)
    x = torch.randn(4, 3, dtype=torch.bfloat16)
    lp = flow.log_prob(x)
    assert lp.dtype == torch.float32  # the fp32 log-determinant accumulator is not cast down
    assert torch.isfinite(lp).all()
