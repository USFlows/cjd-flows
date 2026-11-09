"""Tests for the high-level builders in :mod:`cjd_flows.models`."""

import itertools

import pytest
import torch

from cjd_flows import Flow, StandardNormal, USFlow, VeriFlow
from cjd_flows.bijections import GlobalScale, LULinear, SplitAdditiveCoupling, Squeeze
from cjd_flows.conditioners import ConvNet
from cjd_flows.core import InverseBijection
from cjd_flows.models import BASES, make_base
from cjd_flows.testing import check_bijection, check_flow_normalization, jacobian_log_abs_det

MIXINGS = ["lu", "sl-lu", "orthogonal", "householder", "permutation", "actnorm", None]
COUPLINGS = [("additive", False), ("additive", True), ("gin", False), ("autoregressive", False)]


def _perturb(module: torch.nn.Module, scale: float = 0.2) -> None:
    """Moves all parameters away from the (identity) initialization."""
    with torch.no_grad():
        for p in module.parameters():
            fan = p.shape[-1] if p.dim() > 0 else 1
            p.add_(scale * torch.randn_like(p) / fan**0.5)


def _grid():
    for shape, (coupling, masked), mixing, conjugate in itertools.product(
        [(5,), (2, 4, 4)], COUPLINGS, MIXINGS, [False, True]
    ):
        if coupling == "autoregressive" and len(shape) > 1:
            continue
        yield pytest.param(
            shape, coupling, masked, mixing, conjugate,
            id=f"{'x'.join(map(str, shape))}-{coupling}{'-masked' if masked else ''}-{mixing}{'-conj' if conjugate else ''}",
        )


@pytest.mark.parametrize("shape,coupling,masked,mixing,conjugate", list(_grid()))
def test_usflow_bijection_contract(shape, coupling, masked, mixing, conjugate):
    torch.manual_seed(0)
    flow = USFlow(
        shape, blocks=3, coupling=coupling, masked=masked, mixing=mixing, conjugate=conjugate, hidden=(8,)
    ).double()
    flow.train()
    flow.log_prob(torch.randn(32, *shape, dtype=torch.float64))  # ActNorm data init
    _perturb(flow)
    flow.eval()
    check_bijection(flow.bijection, torch.randn(6, *shape, dtype=torch.float64))
    if conjugate and mixing is not None:
        # A ∘ C ∘ A^{-1} blocks: the inverse shares the mixing layer's parameters
        invs = [b for b in flow.bijection if isinstance(b, InverseBijection)]
        assert len(invs) == 3
        assert all(any(b is inv.bijection for b in flow.bijection) for inv in invs)


@pytest.mark.parametrize("shape", [(5,), (2, 4, 4)])
def test_usflow_with_context(shape):
    torch.manual_seed(0)
    flow = USFlow(shape, blocks=2, hidden=(8,), context_dim=3, conjugate=True).double()
    _perturb(flow)
    x = torch.randn(4, *shape, dtype=torch.float64)
    ctx = torch.randn(4, 3, dtype=torch.float64)
    check_bijection(flow.bijection, x, context=ctx)
    assert not torch.allclose(flow.log_prob(x, ctx), flow.log_prob(x, ctx + 1))
    # missing context -> default context (zeros)
    assert torch.allclose(flow.log_prob(x), flow.log_prob(x, torch.zeros(4, 3, dtype=torch.float64)))


@pytest.mark.parametrize("conditioner", ["mlp", "jet"])
def test_usflow_image_conditioners(conditioner):
    torch.manual_seed(0)
    kw = {"embed_dim": 16, "depth": 1, "num_heads": 2} if conditioner == "jet" else {}
    flow = USFlow((2, 4, 4), blocks=2, conditioner=conditioner, hidden=(8,), conditioner_kwargs=kw).double()
    _perturb(flow)
    check_bijection(flow.bijection, torch.randn(3, 2, 4, 4, dtype=torch.float64))


@pytest.mark.parametrize("base", BASES)
def test_usflow_normalization_2d(base):
    torch.manual_seed(0)
    flow = USFlow((2,), blocks=3, hidden=(16,), base=base)
    _perturb(flow, 0.1)
    torch.manual_seed(1)
    est = check_flow_normalization(flow, n=400_000, radius=9.0)
    assert abs(est - 1.0) < 0.06, est


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("shape", [(5,), (2, 4, 4)])
def test_usflow_sampling_shapes(base, shape):
    torch.manual_seed(0)
    flow = USFlow(shape, blocks=2, hidden=(8,), base=base).eval()
    x = flow.sample((3, 2))
    assert x.shape == (3, 2, *shape)
    lp = flow.log_prob(x)
    assert lp.shape == (3, 2) and torch.isfinite(lp).all()
    assert flow.sample().shape == shape
    assert flow.rsample((4,)).shape == (4, *shape)


def test_make_base_names():
    for name in BASES:
        base = make_base(name, (2, 3))
        assert tuple(base.event_shape) == (2, 3)
        assert base.log_prob(torch.randn(4, 2, 3)).shape == (4,)
    with pytest.raises(ValueError):
        make_base("nope", (2,))


def test_piecewise_affine_flags():
    assert USFlow((4,), coupling="additive", activation="relu").is_piecewise_affine()
    assert USFlow((4,), coupling="additive", masked=True, activation="leaky_relu").is_piecewise_affine()
    assert USFlow((1, 4, 4), coupling="additive", activation="relu", conjugate=True).is_piecewise_affine()
    assert USFlow((4,), coupling="autoregressive", activation="relu").is_piecewise_affine()
    assert not USFlow((4,), coupling="gin", activation="relu").is_piecewise_affine()
    assert not USFlow((4,), coupling="additive", activation="gelu").is_piecewise_affine()
    assert not USFlow((2, 4, 4), conditioner="jet", hidden=(8,),
                      conditioner_kwargs={"embed_dim": 8, "depth": 1, "num_heads": 2}).is_piecewise_affine()


def test_veriflow_preset():
    torch.manual_seed(0)
    flow = VeriFlow((4,), blocks=2, hidden=(8, 8))
    assert flow.is_piecewise_affine()
    assert flow.base.p == 1.0 and flow.base.is_monotone()
    layers = list(flow.bijection)
    assert isinstance(layers[0], LULinear) and isinstance(layers[2], InverseBijection)
    assert layers[2].bijection is layers[0]
    assert isinstance(layers[-1], GlobalScale)
    flow = flow.double()
    _perturb(flow)
    check_bijection(flow.bijection, torch.randn(5, 4, dtype=torch.float64))


def test_usflow_conjugate_state_dict_roundtrip():
    torch.manual_seed(0)
    a = USFlow((3,), blocks=2, hidden=(8,), conjugate=True, base="ggmix")
    _perturb(a)
    torch.manual_seed(1)
    b = USFlow((3,), blocks=2, hidden=(8,), conjugate=True, base="ggmix")
    b.load_state_dict(a.state_dict())
    x = torch.randn(7, 3)
    assert torch.allclose(a.log_prob(x), b.log_prob(x))


def test_usflow_invalid_arguments():
    with pytest.raises(ValueError):
        USFlow((2, 4, 4), coupling="autoregressive")
    with pytest.raises(ValueError):
        USFlow((4,), coupling="nope")
    with pytest.raises(ValueError):
        USFlow((4,), mixing="nope")


# ------------------------------------------------------------- shape-changing flows
def _squeeze_flow():
    sq = Squeeze((1, 4, 4))
    lat = sq.out_shape  # (4, 2, 2)
    coupling = SplitAdditiveCoupling(lat, lambda i, o: ConvNet(i[0], o[0], spatial_ndim=2, hidden=(8,)))
    return Flow([sq, LULinear(lat), coupling, GlobalScale(lat, init=0.7)], StandardNormal(lat))


def test_flow_with_squeeze_shapes_and_density():
    torch.manual_seed(0)
    flow = _squeeze_flow().double()
    _perturb(flow)
    assert flow.event_shape == (1, 4, 4)
    assert flow.latent_shape == (4, 2, 2)
    assert flow.dim == 16

    x = torch.randn(5, 1, 4, 4, dtype=torch.float64)
    z = flow.transform(x)
    assert z.shape == (5, 4, 2, 2)
    assert torch.allclose(flow.generate(z), x, atol=1e-10)

    # log p(x) = log p_Z(f(x)) + log|det J_f| with the autograd determinant
    ref = jacobian_log_abs_det(lambda v: flow.transform(v), x[:3])
    assert torch.allclose(ref, flow.log_abs_det().expand(3), atol=1e-8)
    assert torch.allclose(flow.log_prob(x), flow.base.log_prob(z) + flow.log_abs_det())

    s = flow.sample((2, 3))
    assert s.shape == (2, 3, 1, 4, 4)
    assert flow.log_prob(s).shape == (2, 3)


def test_flow_with_squeeze_normalization():
    torch.manual_seed(0)
    sq = Squeeze((1, 2))  # 2-D data (1 channel, 2 positions) -> (2, 1)
    flow = Flow([sq, LULinear(sq.out_shape), GlobalScale(sq.out_shape, init=1.5)], StandardNormal(sq.out_shape))
    _perturb(flow, 0.3)
    assert flow.event_shape == (1, 2) and flow.latent_shape == (2, 1)
    torch.manual_seed(1)
    est = check_flow_normalization(flow, n=400_000, radius=9.0)
    assert abs(est - 1.0) < 0.05, est
