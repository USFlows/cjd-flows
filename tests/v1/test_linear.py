import math
import os
import time

import pytest
import torch

from cjd_flows.bijections.linear import (
    ActNorm,
    AffineFromMatrix,
    FixedOrthogonal,
    GlobalScale,
    HouseholderLinear,
    LULinear,
    OrthogonalLinear,
    Permutation,
    Reverse,
    Shift,
    merge_linear,
)
from cjd_flows.core import Chain
from cjd_flows.testing import check_bijection

SHAPES = [(6,), (3, 4, 4), (2, 8), (2, 2, 2, 2)]

FACTORIES = {
    "lu": lambda s: LULinear(s, identity_init=False),
    "lu_id": lambda s: LULinear(s),
    "lu_unit": lambda s: LULinear(s, unit_det=True, identity_init=False),
    "lu_nopermute_nobias": lambda s: LULinear(s, permute=False, bias=False, identity_init=False),
    "householder": lambda s: HouseholderLinear(s, n_reflections=3, bias=True),
    "householder_odd": lambda s: HouseholderLinear(s, n_reflections=1),
    "cayley": lambda s: OrthogonalLinear(s, method="cayley", init_std=0.5),
    "exp": lambda s: OrthogonalLinear(s, method="exp", init_std=0.5),
    "actnorm": lambda s: ActNorm(s),
    "actnorm_zs": lambda s: ActNorm(s, zero_sum=True),
    "global": lambda s: GlobalScale(s, init=1.7),
    "perm": lambda s: Permutation(s),
    "reverse": lambda s: Reverse(s),
    "fixed_orth": lambda s: FixedOrthogonal(s, seed=0),
    "fixed_orth64": lambda s: FixedOrthogonal(s, seed=0, dtype=torch.float64),
    "affine": lambda s: AffineFromMatrix(torch.randn(s[0], s[0]) + 2 * torch.eye(s[0]), torch.randn(s[0]), s),
}

FLOAT32_FACTORIES = {k: v for k, v in FACTORIES.items() if not k.endswith("64")}


def _randomize(layer):
    """Perturb parameters so the tests do not run at the identity."""
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return layer


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("name", list(FACTORIES))
def test_check_bijection(name, shape):
    torch.manual_seed(0)
    layer = FACTORIES[name](shape)
    x = torch.randn(8, *shape, dtype=torch.float64) * 2 + 0.5
    layer = layer.double()
    if isinstance(layer, ActNorm):
        layer(x)  # data init
    _randomize(layer)
    check_bijection(layer, x)
    assert layer.piecewise_affine


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("name", list(FACTORIES))
def test_matrix_consistency(name, shape):
    """forward == as_dense, inverse_weight == W^{-1}, log_abs_det == n_spatial*logdet(W)."""
    torch.manual_seed(1)
    layer = _randomize(FACTORIES[name](shape).double()).eval()
    x = torch.randn(5, *shape, dtype=torch.float64)
    W, b = layer.to_matrix()
    A, c = layer.as_dense()
    torch.testing.assert_close(layer(x).flatten(1), x.flatten(1) @ A.T + c)
    torch.testing.assert_close(layer.inverse_weight() @ W, torch.eye(shape[0], dtype=torch.float64))
    n_sp = math.prod(shape[1:])
    tol = dict(atol=1e-5, rtol=0) if name == "fixed_orth" else {}  # float32-built matrix
    torch.testing.assert_close(layer.log_abs_det(), n_sp * torch.linalg.slogdet(W).logabsdet, **tol)
    torch.testing.assert_close(layer.log_abs_det(), torch.linalg.slogdet(A).logabsdet, **tol)


def test_lu_unit_det_exact():
    torch.manual_seed(0)
    layer = LULinear((5, 3, 3), unit_det=True).double()
    with torch.no_grad():
        layer.log_s.copy_(torch.randn(5, dtype=torch.float64) * 3)
    assert layer.log_abs_det().abs().item() < 1e-12
    assert abs(abs(torch.linalg.det(layer.weight()).item()) - 1) < 1e-10
    # anisotropic scaling remains
    assert layer.log_diag().std() > 0.5
    assert layer.volume_preserving


def test_lu_soft_clamp():
    layer = LULinear(4, max_log_scale=8.0)
    with torch.no_grad():
        layer.log_s.fill_(100.0)
    assert layer.log_diag().max().item() <= 8.0
    hard = LULinear(4, max_log_scale=2.0, soft_clamp=False)
    with torch.no_grad():
        hard.log_s.fill_(-5.0)
    assert torch.allclose(hard.log_diag(), torch.full((4,), -2.0))


def test_lu_identity_init():
    layer = LULinear((4, 2, 2), permute=False)
    assert torch.equal(layer.weight(), torch.eye(4))
    layer = LULinear((4, 2, 2), identity_init=False)
    W = layer.weight()
    torch.testing.assert_close(W @ W.T, torch.eye(4), atol=1e-5, rtol=0)


def test_lu_prior():
    shape = (4, 3, 3)
    layer = LULinear(shape, prior_scale=2.0, max_log_scale=None).double()
    s = torch.randn(4, dtype=torch.float64)
    with torch.no_grad():
        layer.log_s.copy_(s)
    sigma0 = 2.0 / (9 * math.sqrt(4))
    assert math.isclose(layer.prior_std(), sigma0)
    ref = torch.distributions.Normal(0.0, sigma0).log_prob(s).sum()
    torch.testing.assert_close(layer.log_prior(include_constants=True), ref)
    torch.testing.assert_close(layer.log_prior(), -0.5 * (s / sigma0).square().sum())
    # total log-det ~ N(0, prior_scale^2) under the prior
    samples = torch.randn(200_000, 4, dtype=torch.float64) * sigma0
    total = 9 * samples.sum(1)
    assert abs(total.std().item() - 2.0) < 0.02
    assert LULinear(4).log_prior().item() == 0.0


def test_householder_matches_dense():
    torch.manual_seed(0)
    C, k = 7, 5
    layer = HouseholderLinear((C, 3, 2), n_reflections=k).double()
    V = layer.vectors.detach()
    W = torch.eye(C, dtype=torch.float64)
    for i in range(k):  # W = H_k ... H_1
        v = V[i]
        H = torch.eye(C, dtype=torch.float64) - 2 * torch.outer(v, v) / v.dot(v)
        W = H @ W
    torch.testing.assert_close(layer.weight(), W)
    x = torch.randn(4, C, 3, 2, dtype=torch.float64)
    ref = torch.einsum("ij,bj...->bi...", W, x)
    torch.testing.assert_close(layer(x), ref)
    assert math.isclose(torch.linalg.det(W).item(), (-1) ** k, rel_tol=1e-10)
    assert layer.log_abs_det().item() == 0.0 and layer.volume_preserving


def test_orthogonal_is_orthogonal():
    for method in ("cayley", "exp"):
        layer = OrthogonalLinear(6, method=method, init_std=1.0).double()
        W = layer.weight()
        torch.testing.assert_close(W @ W.T, torch.eye(6, dtype=torch.float64))
        assert math.isclose(torch.linalg.det(W).item(), 1.0, rel_tol=1e-10)
        assert torch.equal(OrthogonalLinear(6, method=method).weight(), torch.eye(6))


@pytest.mark.parametrize("shape", [(6,), (3, 4, 4), (2, 2, 2, 2)], ids=str)
def test_actnorm_data_init(shape):
    torch.manual_seed(0)
    x = torch.randn(64, *shape, dtype=torch.float64) * torch.linspace(0.5, 3, shape[0], dtype=torch.float64).view(
        1, -1, *([1] * (len(shape) - 1))
    ) + 4.0
    dims = [0] + list(range(2, x.dim()))
    layer = ActNorm(shape).double()
    assert not layer.initialized
    z = layer(x)
    assert layer.initialized
    torch.testing.assert_close(z.mean(dims), torch.zeros(shape[0], dtype=torch.float64))
    torch.testing.assert_close(z.std(dims, unbiased=False), torch.ones(shape[0], dtype=torch.float64))
    # second call does not re-init
    s = layer.log_scale.clone()
    layer(x * 10)
    assert torch.equal(s, layer.log_scale)
    # zero-sum variant: zero mean, equal std, unit det
    zs = ActNorm(shape, zero_sum=True).double()
    z = zs(x)
    torch.testing.assert_close(z.mean(dims), torch.zeros(shape[0], dtype=torch.float64))
    std = z.std(dims, unbiased=False)
    torch.testing.assert_close(std, std[:1].expand_as(std))
    assert abs(zs.log_abs_det().item()) < 1e-12
    # eval mode: no init
    ev = ActNorm(shape).eval()
    ev(x.float())
    assert not ev.initialized


def test_actnorm_state_dict_keeps_init():
    a = ActNorm((3, 2, 2))
    a(torch.randn(10, 3, 2, 2) * 5)
    b = ActNorm((3, 2, 2))
    b.load_state_dict(a.state_dict())
    assert b.initialized
    x = torch.randn(4, 3, 2, 2)
    torch.testing.assert_close(a(x), b(x))


def test_global_scale():
    shape = (3, 4, 4)
    g = GlobalScale(shape, init=2.0).double()
    assert math.isclose(g.log_abs_det().item(), 48 * math.log(2.0), rel_tol=1e-6)
    torch.manual_seed(0)
    z = torch.randn(4000, 3, dtype=torch.float64) * 3.0
    g = GlobalScale(3).double()
    c = g.set_from_latents(z, method="mle")
    assert abs(c - 1 / 3) < 0.01
    # MLE property: derivative of E[log N(cz)] + D log c at optimum is 0
    log_c = g.log_scale.detach().clone().requires_grad_(True)
    obj = -0.5 * (z * log_c.exp()).square().sum(1).mean() + 3 * log_c
    (grad,) = torch.autograd.grad(obj, log_c)
    assert abs(grad.item()) < 1e-8
    # det method: anisotropic covariance diag(1, 4, 16) -> |Σ|^{-1/(2D)} = 64^{-1/6} = 0.5
    z = torch.randn(200_000, 3, dtype=torch.float64) * torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    c = g.set_from_latents(z, method="det")
    assert abs(c - 0.5) < 0.01


def test_permutation_validation():
    p = Permutation(4, perm=[2, 0, 3, 1])
    x = torch.arange(4.0).unsqueeze(0)
    assert p(x).tolist() == [[2.0, 0.0, 3.0, 1.0]]
    assert Reverse(4)(x).tolist() == [[3.0, 2.0, 1.0, 0.0]]
    with pytest.raises(ValueError):
        Permutation(3, perm=[0, 0, 1])


def test_inverse_does_not_use_dense_inverse(monkeypatch):
    """The LU forward/inverse never call torch.inverse / linalg.inv."""
    layer = LULinear((4, 3, 3), identity_init=False)

    def boom(*a, **k):
        raise AssertionError("dense inverse used")

    monkeypatch.setattr(torch, "inverse", boom)
    monkeypatch.setattr(torch.linalg, "inv", boom)
    x = torch.randn(2, 4, 3, 3)
    layer.inverse(layer(x))


@pytest.mark.parametrize("shape", [(6,), (3, 4, 4)], ids=str)
def test_merge_linear(shape):
    torch.manual_seed(0)
    from cjd_flows.core import Identity

    act = ActNorm(shape).double()
    act(torch.randn(16, *shape, dtype=torch.float64))
    layers = [
        LULinear(shape, identity_init=False),
        HouseholderLinear(shape, 2, bias=True),
        act,
        Identity(),
        OrthogonalLinear(shape, init_std=0.3),
        Chain(GlobalScale(shape, 1.3), LULinear(shape, identity_init=False).inv),
        Permutation(shape),
    ]
    chain = Chain(*[_randomize(layer.double()) for layer in layers]).eval()
    merged = merge_linear(chain)
    assert len(merged) == 3
    assert isinstance(merged[0], AffineFromMatrix) and isinstance(merged[2], AffineFromMatrix)
    x = torch.randn(7, *shape, dtype=torch.float64)
    torch.testing.assert_close(merged(x), chain(x))
    torch.testing.assert_close(merged.inverse(x), chain.inverse(x))
    torch.testing.assert_close(merged.log_abs_det(), chain.log_abs_det())
    check_bijection(merged, x)


def test_double_and_state_dict_roundtrip():
    shape = (3, 4, 4)
    for name, fac in FLOAT32_FACTORIES.items():
        torch.manual_seed(0)
        a = _randomize(fac(shape)).eval()
        torch.manual_seed(123)
        b = fac(shape).eval()
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, *shape)
        torch.testing.assert_close(a(x), b(x), msg=name)
        torch.testing.assert_close(a.inverse(x), b.inverse(x), msg=name)
        d = a.double()
        assert d(x.double()).dtype == torch.float64
        assert d.log_abs_det().dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_cuda_smoke():
    shape = (3, 4, 4)
    for name, fac in FLOAT32_FACTORIES.items():
        layer = fac(shape).cuda()
        x = torch.randn(4, *shape, device="cuda")
        z = layer(x)
        assert z.device.type == "cuda"
        torch.testing.assert_close(layer.inverse(z), x, atol=1e-4, rtol=1e-4, msg=name)
        ld = layer.log_abs_det()
        assert ld.device.type == "cuda" or ld.numel() == 1
    merged = merge_linear(Chain(LULinear(shape).cuda(), Permutation(shape).cuda()))
    assert merged[0].W.device.type == "cuda"


def test_compile_lu_forward():
    if os.environ.get("CJD_SKIP_SLOW"):
        pytest.skip("slow test disabled")
    layer = LULinear((8, 4, 4), identity_init=False)
    x = torch.randn(4, 8, 4, 4)
    t0 = time.time()
    try:
        compiled = torch.compile(layer)
        z = compiled(x)
    except Exception as e:  # compile toolchain missing in some environments
        pytest.skip(f"torch.compile unavailable: {e}")
    if time.time() - t0 > 60:
        pytest.skip("torch.compile took longer than 60s")
    torch.testing.assert_close(z, layer(x))


@pytest.mark.parametrize("shape", [(16,), (16, 2)], ids=str)
def test_lu_inverse_paths_agree(shape):
    """Few vectors -> direct triangular solves; many -> W^{-1} GEMM. Both exact."""
    torch.manual_seed(0)
    layer = _randomize(LULinear(shape, identity_init=False).double())
    for B in (1, 3, 40):  # B * n_spatial below and above C
        x = torch.randn(B, *shape, dtype=torch.float64)
        torch.testing.assert_close(layer.inverse(layer(x)), x)


@pytest.mark.parametrize("per", ["coordinate", "channel"])
@pytest.mark.parametrize("shape", [(6,), (3, 4, 4), (2, 2, 2, 2)], ids=str)
def test_shift(per, shape):
    torch.manual_seed(0)
    x = torch.randn(64, *shape, dtype=torch.float64) + torch.randn(*shape, dtype=torch.float64) * 3
    layer = Shift(shape, per=per).double()
    assert not layer.initialized and layer.volume_preserving and layer.piecewise_affine
    z = layer(x)
    assert layer.initialized
    dims = [0] if per == "coordinate" else [0] + list(range(2, x.dim()))
    torch.testing.assert_close(z.mean(dims), torch.zeros_like(z.mean(dims)))
    s = layer.shift.clone()
    layer(x * 10)  # no re-init
    assert torch.equal(s, layer.shift)
    _randomize(layer)
    check_bijection(layer, x)
    assert layer.log_abs_det().shape == () and layer.log_abs_det().item() == 0.0
    # eval mode and data_init=False: no init
    ev = Shift(shape, per=per).eval()
    ev(x.float())
    assert not ev.initialized
    assert Shift(shape, data_init=False).initialized
    with pytest.raises(ValueError):
        Shift(shape, per="bogus")


def test_shift_state_dict_keeps_init():
    a = Shift((3, 2, 2))
    a(torch.randn(10, 3, 2, 2) + 5)
    b = Shift((3, 2, 2))
    b.load_state_dict(a.state_dict())
    assert b.initialized
    x = torch.randn(4, 3, 2, 2)
    torch.testing.assert_close(a(x), b(x))
