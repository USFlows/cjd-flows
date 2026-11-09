import math

import pytest
import torch

from cjd_flows.bijections.autoregressive import MaskedConvAutoregressive, ShiftAutoregressive
from cjd_flows.bijections.coupling import (
    GINCoupling,
    MaskedAdditiveCoupling,
    SplitAdditiveCoupling,
    channel_mask,
    checkerboard_mask,
    random_mask,
    split_indices,
)
from cjd_flows.conditioners import MADE, MLP, ConvNet, JetConditioner
from cjd_flows.core import Chain
from cjd_flows.testing import check_bijection, jacobian_log_abs_det

CTX = 2


def perturb(module, scale=0.3):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * scale)
    return module


# ------------------------------------------------------------------ conditioner factories
def mlp_factory(ctx):
    return lambda i, o: MLP(i, o, hidden=(16, 16), context_dim=ctx)


def conv_factory(ctx):
    return lambda i, o: ConvNet(i[0], o[0], spatial_ndim=len(i) - 1, hidden=(8, 8), context_dim=ctx)


def jet_factory(ctx):
    def f(i, o):
        if len(i) == 1:
            return JetConditioner(i, embed_dim=16, depth=1, num_heads=2, context_dim=ctx, token_size=2, out_dim=o[0])
        return JetConditioner(i, patch_size=2, embed_dim=16, depth=1, num_heads=2, context_dim=ctx, out_dim=o[0])
    return f


FACTORIES = {"mlp": mlp_factory, "conv": conv_factory, "jet": jet_factory}

# (event_shape, split, allowed conditioner kinds)
CASES = [
    ((6,), "channel", ("mlp", "jet")),
    ((6,), "interleaved", ("mlp", "jet")),
    ((6,), "random", ("mlp",)),
    ((2, 4, 4), "channel", ("mlp", "conv", "jet")),
    ((2, 4, 4), "spatial-row", ("mlp", "conv", "jet")),
    ((2, 4, 4), "spatial-col", ("mlp", "conv", "jet")),
    ((2, 4, 4), "checkerboard", ("mlp", "conv", "jet")),
    ((4, 4, 4), "channel", ("mlp", "conv", "jet")),
    ((3, 4, 4), "channel", ("mlp", "conv", "jet")),  # odd channel count
    ((2, 8), "channel", ("mlp", "conv", "jet")),
    ((2, 8), "checkerboard", ("mlp", "conv", "jet")),
    ((1, 4, 4, 4), "checkerboard", ("mlp", "conv")),  # 3 spatial dims
]
SPLIT_PARAMS = [
    pytest.param(shape, split, kind, ctx, id=f"{shape}-{split}-{kind}-ctx{ctx}")
    for shape, split, kinds in CASES
    for kind in kinds
    for ctx in (0, CTX)
]


def _inputs(shape, ctx, B=5):
    x = torch.randn(B, *shape, dtype=torch.float64)
    c = torch.randn(B, ctx, dtype=torch.float64) if ctx else None
    return x, c


@pytest.mark.parametrize("cls", [SplitAdditiveCoupling, GINCoupling])
@pytest.mark.parametrize("shape,split,kind,ctx", SPLIT_PARAMS)
def test_split_couplings(cls, shape, split, kind, ctx):
    torch.manual_seed(0)
    for parity in (0, 1):
        layer = cls(shape, FACTORIES[kind](ctx), split=split, parity=parity).double()
        x, c = _inputs(shape, ctx)
        # identity at init (zero-initialized conditioner)
        assert torch.equal(layer(x, c), x)
        assert torch.equal(layer.inverse(x, c), x)
        perturb(layer)
        z = layer(x, c)
        assert not torch.allclose(z, x)
        check_bijection(layer, x, c)
        assert layer.volume_preserving
        if ctx:
            assert not torch.allclose(layer(x, c), layer(x, c + 1.0))


@pytest.mark.parametrize("mask", ["checkerboard", "channel", "random"])
@pytest.mark.parametrize("shape", [(6,), (2, 4, 4), (4, 4, 4), (2, 8)])
@pytest.mark.parametrize("kind", ["mlp", "conv", "jet"])
@pytest.mark.parametrize("ctx", [0, CTX])
def test_masked_additive(mask, shape, kind, ctx):
    if kind == "conv" and len(shape) == 1:
        pytest.skip("conv needs spatial input")
    torch.manual_seed(0)
    for parity in (0, 1):
        cond = FACTORIES[kind](ctx)(shape, shape)
        layer = MaskedAdditiveCoupling(shape, cond, mask=mask, parity=parity).double()
        x, c = _inputs(shape, ctx)
        assert torch.equal(layer(x, c), x)
        perturb(layer)
        check_bijection(layer, x, c)
        # masked positions pass through unchanged
        m = layer.mask.bool().expand_as(x)
        assert torch.equal(layer(x, c)[m], x[m])


def test_masks():
    assert checkerboard_mask((6,)).tolist() == [1, 0, 1, 0, 1, 0]
    assert checkerboard_mask((6,), 1).tolist() == [0, 1, 0, 1, 0, 1]
    m = checkerboard_mask((2, 3, 3))
    assert torch.equal(m[0], m[1]) and m[0, 0, 0] == 1 and m[0, 0, 1] == 0
    assert torch.equal(checkerboard_mask((2, 4, 4)) + checkerboard_mask((2, 4, 4), 1), torch.ones(2, 4, 4))
    cm = channel_mask((4, 2, 2))
    assert cm[:2].eq(1).all() and cm[2:].eq(0).all()
    assert channel_mask((6,)).tolist() == [1, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="channel"):
        channel_mask((1, 4, 4))
    r = random_mask((2, 3, 3), seed=1)
    assert r.sum() == 9 and torch.equal(r, random_mask((2, 3, 3), seed=1))
    layer = MaskedAdditiveCoupling((2, 4, 4), ConvNet(2, 2), mask="channel")
    assert "mask" in dict(layer.named_buffers())


def test_split_indices_partition_and_errors():
    for shape, split in [((7,), "channel"), ((7,), "interleaved"), ((3, 4, 6), "checkerboard"),
                         ((3, 4, 6), "spatial-row"), ((2, 2, 4, 6), "spatial-col"), ((5, 4), "random")]:
        ia, ib, sa, sb = split_indices(shape, split)
        n = math.prod(shape)
        assert sorted(torch.cat([ia, ib]).tolist()) == list(range(n))
        assert math.prod(sa) == len(ia) and math.prod(sb) == len(ib)
    custom = torch.zeros(2, 3, dtype=torch.bool)
    custom[0, 1] = True
    ia, ib, sa, sb = split_indices((2, 3), custom)
    assert ia.tolist() == [1] and sa == (1,) and sb == (5,)
    with pytest.raises(ValueError):
        split_indices((1, 4, 4), "channel")
    with pytest.raises(ValueError):
        split_indices((2, 3, 3), "spatial-row")


def test_split_custom_mask():
    sel = random_mask((2, 4, 4), seed=0).bool()
    layer = SplitAdditiveCoupling((2, 4, 4), lambda i, o: MLP(i, o, hidden=(8,)), split=sel).double()
    perturb(layer)
    x = torch.randn(4, 2, 4, 4, dtype=torch.float64)
    check_bijection(layer, x)
    assert torch.equal(layer(x)[:, sel], x[:, sel])


def test_split_with_module_conditioner():
    cond = ConvNet(2, 2, spatial_ndim=2, hidden=(8,))
    layer = SplitAdditiveCoupling((4, 4, 4), cond, split="channel").double()
    assert layer.conditioner is cond
    perturb(layer)
    check_bijection(layer, torch.randn(3, 4, 4, 4, dtype=torch.float64))


def test_gin_log_det_zero_vs_autograd():
    torch.manual_seed(0)
    layer = GINCoupling((2, 4, 4), conv_factory(0), split="checkerboard").double()
    perturb(layer, scale=1.0)
    x = torch.randn(4, 2, 4, 4, dtype=torch.float64)
    ref = jacobian_log_abs_det(layer, x)
    assert torch.allclose(ref, torch.zeros_like(ref), atol=1e-10)
    assert layer.log_abs_det().item() == 0.0
    # the layer actually scales (not just shifts)
    _, xb = layer._split(x)
    s, _ = layer._scale_shift(layer._params(layer._split(x)[0], None))
    assert s.abs().max() > 1e-3 and s.abs().max() <= 2 * layer.clamp


def test_piecewise_affine_flags():
    assert SplitAdditiveCoupling((6,), mlp_factory(0)).piecewise_affine
    assert not SplitAdditiveCoupling((6,), jet_factory(0)).piecewise_affine
    assert not SplitAdditiveCoupling((2, 4, 4), lambda i, o: ConvNet(i[0], o[0], gated=True)).piecewise_affine
    assert MaskedAdditiveCoupling((2, 4, 4), ConvNet(2, 2)).piecewise_affine
    assert not GINCoupling((6,), mlp_factory(0)).piecewise_affine
    assert ShiftAutoregressive(4, MADE(4)).piecewise_affine
    assert not ShiftAutoregressive(4, MADE(4, activation="gelu")).piecewise_affine
    chain = Chain(SplitAdditiveCoupling((6,), mlp_factory(0)), GINCoupling((6,), mlp_factory(0)))
    assert not chain.is_piecewise_affine() and chain.volume_preserving


# ------------------------------------------------------------------ autoregressive
@pytest.mark.parametrize("order", [None, "reverse", "random"])
@pytest.mark.parametrize("ctx", [0, CTX])
def test_shift_autoregressive(order, ctx):
    torch.manual_seed(0)
    D = 6
    layer = ShiftAutoregressive(D, MADE(D, hidden=(32, 32), order=order, context_dim=ctx)).double()
    x, c = _inputs((D,), ctx)
    assert torch.equal(layer(x, c), x)
    perturb(layer)
    check_bijection(layer, x, c)
    # Jacobian = I + strictly lower triangular in the autoregressive order
    order_t = layer.made.order
    cc = None if c is None else c[:1]
    J = torch.autograd.functional.jacobian(lambda v: layer(v.unsqueeze(0), cc).squeeze(0), x[0])
    Jp = J[order_t][:, order_t]
    assert torch.allclose(torch.diagonal(Jp), torch.ones(D, dtype=torch.float64))
    assert torch.all(torch.triu(Jp, diagonal=1) == 0)
    assert torch.tril(Jp, diagonal=-1).abs().sum() > 0


def test_shift_autoregressive_validates_made():
    with pytest.raises(ValueError):
        ShiftAutoregressive(4, MADE(5))
    with pytest.raises(ValueError):
        ShiftAutoregressive(4, MADE(4, out_multiplier=2))


@pytest.mark.parametrize("shape", [(2, 6), (2, 3, 3), (1, 2, 2, 3)])
@pytest.mark.parametrize("ctx", [0, CTX])
def test_masked_conv_autoregressive(shape, ctx):
    torch.manual_seed(0)
    layer = MaskedConvAutoregressive(shape, hidden=(8, 8, 6), context_dim=ctx).double()
    x, c = _inputs(shape, ctx)
    assert torch.equal(layer(x, c), x)
    perturb(layer)
    check_bijection(layer, x, c)
    # causal structure: output at raster position p only depends on positions < p (+ identity)
    d = math.prod(shape)
    C = shape[0]
    P = d // C
    cc = None if c is None else c[:1]
    J = torch.autograd.functional.jacobian(lambda v: layer(v.reshape(1, *shape), cc).reshape(-1), x[0].reshape(-1))
    pos = torch.arange(d) % P  # raster position of each flat index (C-major layout)
    off = J - torch.eye(d, dtype=J.dtype)
    allowed = pos[:, None] > pos[None, :]
    assert torch.all(off[~allowed] == 0)
    assert off[allowed].abs().sum() > 0


# ------------------------------------------------------------------ cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_couplings_cuda():
    dev = "cuda"
    layers = [
        (SplitAdditiveCoupling((2, 8, 8), conv_factory(1), split="checkerboard"), (2, 8, 8)),
        (GINCoupling((2, 8, 8), jet_factory(1), split="channel"), (2, 8, 8)),
        (MaskedAdditiveCoupling((6,), MLP(6, 6, context_dim=1), mask="checkerboard"), (6,)),
        (ShiftAutoregressive(6, MADE(6, context_dim=1)), (6,)),
        (MaskedConvAutoregressive((2, 3, 3), context_dim=1), (2, 3, 3)),
    ]
    for layer, shape in layers:
        layer = perturb(layer, 0.1).to(dev)
        x = torch.randn(4, *shape, device=dev)
        c = torch.randn(4, 1, device=dev)
        z = layer(x, c)
        assert z.device.type == "cuda"
        assert torch.allclose(layer.inverse(z, c), x, atol=1e-4)
        assert layer.log_abs_det().device.type == "cuda"
