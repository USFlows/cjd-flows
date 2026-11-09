
import pytest
import torch

from cjd_flows.conditioners import MADE, MLP, ConvNet, JetConditioner, lipschitz_upper_bound

torch.manual_seed(0)


def perturb(module, scale=0.2):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * scale)
    return module


def train_step(net, x, ctx):
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    out = net(x, ctx)
    loss = (out - torch.randn_like(out) - ctx.sum(1).reshape(-1, *([1] * (out.dim() - 1)))).square().mean()
    loss.backward()
    opt.step()


# ------------------------------------------------------------------ shapes / zero init
@pytest.mark.parametrize("norm", [None, "layernorm"])
@pytest.mark.parametrize("ctx", [0, 3])
def test_mlp_shapes_and_zero_init(norm, ctx):
    net = MLP(6, 4, hidden=(16, 16, 8), context_dim=ctx, norm=norm, dropout=0.1)
    x = torch.randn(5, 6)
    c = torch.randn(5, ctx) if ctx else None
    out = net(x, c)
    assert out.shape == (5, 4)
    assert torch.all(out == 0)


def test_mlp_shape_io():
    net = MLP((2, 4, 4), (3, 4, 4), hidden=(16,))
    out = net(torch.randn(2, 2, 4, 4))
    assert out.shape == (2, 3, 4, 4)


@pytest.mark.parametrize("nd", [1, 2, 3])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("norm", [None, "layernorm"])
def test_convnet_shapes(nd, gated, norm):
    net = ConvNet(3, 5, spatial_ndim=nd, hidden=(8, 12, 12), gated=gated, norm=norm, context_dim=2, dilation=2)
    x = torch.randn(2, 3, *([6] * nd))
    out = net(x, torch.randn(2, 2))
    assert out.shape == (2, 5, *([6] * nd))
    assert torch.all(out == 0)
    perturb(net)
    assert net(x, torch.randn(2, 2)).shape == (2, 5, *([6] * nd))


@pytest.mark.parametrize("in_dims,out_dim,expected", [
    ((6,), None, (6,)),
    ((6,), 10, (10,)),
    ((2, 8), 4, (4, 8)),
    ((2, 4, 4), None, (2, 4, 4)),
    ((1, 4, 4), 2, (2, 4, 4)),
    ((2, 4, 4, 4), None, (2, 4, 4, 4)),
])
def test_jet_shapes(in_dims, out_dim, expected):
    net = JetConditioner(in_dims, patch_size=2, embed_dim=16, depth=2, num_heads=2, context_dim=1, token_size=4, out_dim=out_dim)
    x = torch.randn(3, *in_dims)
    out = net(x, torch.randn(3, 1))
    assert out.shape == (3, *expected)
    assert torch.all(out == 0)
    assert net(x).shape == (3, *expected)


def test_jet_padding_mask_and_checkpointing():
    net = JetConditioner((2, 8), patch_size=1, embed_dim=16, depth=2, num_heads=2, causal=True, grad_checkpointing=True)
    perturb(net)
    net.train()
    x = torch.randn(3, 2, 8, requires_grad=True)
    mask = torch.ones(3, 8, dtype=torch.bool)
    mask[:, -2:] = False
    out = net(x, padding_mask=mask)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # causal: output at token 0 does not depend on later tokens
    net.eval()
    x2 = x.detach().clone()
    x2[:, :, 5] += 1.0
    assert torch.allclose(net(x.detach())[:, :, 0], net(x2)[:, :, 0], atol=1e-6)


# ------------------------------------------------------------------ context
@pytest.mark.parametrize("make", [
    lambda: (MLP(4, 4, hidden=(16, 16), context_dim=2), (4,)),
    lambda: (ConvNet(2, 2, spatial_ndim=2, hidden=(8, 8), context_dim=2), (2, 4, 4)),
    lambda: (MADE(4, hidden=(16, 16), context_dim=2), (4,)),
    lambda: (JetConditioner((2, 4, 4), patch_size=2, embed_dim=16, depth=1, num_heads=2, context_dim=2), (2, 4, 4)),
])
def test_context_changes_output_after_training(make):
    torch.manual_seed(1)
    net, shape = make()
    x = torch.randn(8, *shape)
    ctx = torch.randn(8, 2)
    for _ in range(2):
        train_step(net, x, ctx)
    with torch.no_grad():
        a = net(x, ctx)
        b = net(x, ctx + 1.0)
    assert not torch.allclose(a, b)


# ------------------------------------------------------------------ MADE
@pytest.mark.parametrize("order", [None, "reverse", "random", [2, 0, 4, 1, 3]])
@pytest.mark.parametrize("mult", [1, 2])
def test_made_autoregressive(order, mult):
    D = 5
    made = MADE(D, hidden=(16, 16), order=order, seed=3, out_multiplier=mult).double()
    perturb(made)
    x = torch.randn(D, dtype=torch.float64)
    J = torch.autograd.functional.jacobian(lambda v: made(v.unsqueeze(0)).squeeze(0), x)
    deg = made.degrees
    allowed = (deg.repeat(mult)[:, None] > deg[None, :])
    assert torch.all(J[~allowed] == 0)
    assert torch.equal(made.connectivity(), allowed & made.connectivity())
    # with enough hidden units, all allowed connections exist
    assert torch.equal(made.connectivity(), allowed)


def test_made_masks_are_buffers():
    made = MADE(4, hidden=(8,))
    names = dict(made.named_buffers())
    assert "order" in names and "degrees" in names
    assert any(n.endswith("mask") for n in names)


# ------------------------------------------------------------------ flags
def test_piecewise_affine_flags():
    assert MLP(4, 4).piecewise_affine
    assert MLP(4, 4, activation="leaky_relu").piecewise_affine
    assert not MLP(4, 4, activation="gelu").piecewise_affine
    assert not MLP(4, 4, norm="layernorm").piecewise_affine
    assert ConvNet(2, 2).piecewise_affine
    assert not ConvNet(2, 2, gated=True).piecewise_affine
    assert not ConvNet(2, 2, norm="layernorm").piecewise_affine
    assert MADE(4).piecewise_affine
    assert not MADE(4, activation="tanh").piecewise_affine
    assert not JetConditioner((4,), embed_dim=16, num_heads=2, depth=1).piecewise_affine


# ------------------------------------------------------------------ Lipschitz
def _empirical_lipschitz(net, shape, n=20):
    best = 0.0
    for _ in range(n):
        x = torch.randn(1, *shape, dtype=torch.float64)
        J = torch.autograd.functional.jacobian(lambda v: net(v.reshape(1, *shape)).reshape(-1), x.reshape(-1))
        best = max(best, float(torch.linalg.matrix_norm(J, ord=2)))
    return best


@pytest.mark.parametrize("sn", [False, True])
def test_lipschitz_bound_mlp(sn):
    torch.manual_seed(0)
    net = MLP(5, 5, hidden=(16, 16, 12), spectral_norm=sn).double()
    perturb(net)
    with torch.no_grad():  # converge the power iteration of spectral_norm
        for _ in range(30):
            net(torch.randn(2, 5, dtype=torch.float64))
    net.eval()
    bound = lipschitz_upper_bound(net)
    emp = _empirical_lipschitz(net, (5,))
    assert 0 < emp <= bound + 1e-9
    if sn:
        # 4 linear layers with sigma ~ 1, 2 residual blocks -> ~ (1+1)*(1+1)*|gain|*...
        gain = abs(float(net.out_gain.gain.detach()))
        assert bound <= 4.0 * gain * 1.05 + 1e-9


@pytest.mark.parametrize("sn", [False, True])
@pytest.mark.parametrize("nd", [1, 2])
def test_lipschitz_bound_convnet(sn, nd):
    torch.manual_seed(0)
    net = ConvNet(2, 2, spatial_ndim=nd, hidden=(4, 4), spectral_norm=sn, dilation=1 + nd % 2).double()
    perturb(net)
    net.eval()
    shape = (2,) + (5,) * nd
    bound = lipschitz_upper_bound(net)
    emp = _empirical_lipschitz(net, shape, n=5)
    assert 0 < emp <= bound + 1e-9


def test_lipschitz_bound_rejects_norm():
    with pytest.raises(ValueError):
        lipschitz_upper_bound(MLP(3, 3, norm="layernorm"))
    with pytest.raises(ValueError):
        lipschitz_upper_bound(ConvNet(2, 2, hidden=(4, 4), gated=True))
    with pytest.raises(TypeError):
        lipschitz_upper_bound(MADE(3))


# ------------------------------------------------------------------ cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_conditioners_cuda():
    dev = "cuda"
    nets = [
        (MLP(4, 4, context_dim=1), (4,)),
        (ConvNet(2, 2, context_dim=1, gated=True, norm="layernorm"), (2, 8, 8)),
        (MADE(4, context_dim=1), (4,)),
        (JetConditioner((2, 8, 8), patch_size=2, embed_dim=16, depth=1, num_heads=2, context_dim=1), (2, 8, 8)),
    ]
    for net, shape in nets:
        net = perturb(net).to(dev)
        out = net(torch.randn(3, *shape, device=dev), torch.randn(3, 1, device=dev))
        assert out.device.type == "cuda" and torch.isfinite(out).all()
