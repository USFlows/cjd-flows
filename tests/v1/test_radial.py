"""Tests for the pyro-free radial base distributions (``cjd_flows.radial``)."""

from __future__ import annotations

import math

import pytest
import torch

from cjd_flows.core import Flow, Identity
from cjd_flows.radial import (
    GammaMixture,
    GeneralizedGamma,
    GeneralizedGammaMixture,
    KhinchineNorm,
    LogNormalMixture,
    MonotoneProfileNorm,
    NormDistribution,
    RadialBase,
    RadialMixture,
    SplineNorm,
    StudentTNorm,
    WeibullMixture,
    density_threshold_radius,
    fit_norm_to_radii,
    hdr_pvalue,
    log_unit_ball_volume,
    lp_norm,
    rq_spline,
    sample_unit_sphere,
    udl_intervals,
    udl_log_threshold,
    udl_radius,
)

stats = pytest.importorskip("scipy.stats")

INF = float("inf")
PS = [1.0, 2.0, INF, 1.5]


@pytest.fixture(autouse=True)
def float64_and_seed():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


FAMILIES = {
    "gg": lambda d, p: GeneralizedGamma(d, p),
    "gg_mono": lambda d, p: GeneralizedGamma(d, p, monotone=True),
    "ggmix": lambda d, p: GeneralizedGammaMixture(d, p, 3),
    "ggmix_mono": lambda d, p: GeneralizedGammaMixture(d, p, 2, monotone=True),
    "gamma": lambda d, p: GammaMixture(d, p, 2),
    "weibull": lambda d, p: WeibullMixture(d, p, 2),
    "weibull_mono": lambda d, p: WeibullMixture(d, p, 1, monotone=True),
    "lognormal": lambda d, p: LogNormalMixture(d, p, 3),
    "studentt": lambda d, p: StudentTNorm(d, p, df=3.0),
    "khinchine": lambda d, p: KhinchineNorm(d, p),
    "khinchine_ln": lambda d, p: KhinchineNorm(d, p, mixing=LogNormalMixture(d, p, 2)),
    "spline": lambda d, p: SplineNorm(d, p),
    "spline_logistic": lambda d, p: SplineNorm(d, p, base="logistic"),
    "monotone_profile": lambda d, p: MonotoneProfileNorm(d, p),
}

EXACTLY_MONOTONE = {"gg_mono", "ggmix_mono", "weibull_mono", "studentt", "khinchine", "khinchine_ln", "monotone_profile"}


def perturb(norm: NormDistribution, scale: float = 0.3) -> NormDistribution:
    """Moves all parameters away from their (often special) initial values."""
    with torch.no_grad():
        for name, prm in norm.named_parameters():
            if "unnorm" in name or "ramp_logits" in name:
                prm.add_(torch.randn_like(prm))
                if "ramp_logits" in name:
                    prm.copy_(torch.randn_like(prm) - 2.0)
            else:
                prm.add_(scale * torch.randn_like(prm))
    return norm


def integral(norm: NormDistribution, n: int = 400_001) -> float:
    lo, hi = norm.log_radius_bounds()
    s = torch.linspace(lo - 2.0, hi + 2.0, n)
    with torch.no_grad():
        v = norm.log_prob_log_radius(s).exp()
    return float(torch.trapezoid(v, s))


# ============================================================== geometry
def test_unit_ball_volume_known_values():
    assert math.isclose(math.exp(log_unit_ball_volume(2, 2.0)), math.pi, rel_tol=1e-12)
    assert math.isclose(math.exp(log_unit_ball_volume(3, 2.0)), 4 / 3 * math.pi, rel_tol=1e-12)
    assert math.isclose(math.exp(log_unit_ball_volume(2, 1.0)), 2.0, rel_tol=1e-12)
    assert math.isclose(math.exp(log_unit_ball_volume(3, 1.0)), 8 / 6, rel_tol=1e-12)
    assert math.isclose(math.exp(log_unit_ball_volume(5, INF)), 32.0, rel_tol=1e-12)
    # large p approaches the cube
    assert math.isclose(log_unit_ball_volume(3, 1e6), 3 * math.log(2), rel_tol=1e-4)


@pytest.mark.parametrize("p", [0.5, 1.0, 1.5, 2.0, 7.0, INF])
def test_lp_norm(p):
    z = torch.randn(5, 3, 4)
    flat = z.reshape(5, -1)
    ref = flat.abs().amax(-1) if math.isinf(p) else (flat.abs() ** p).sum(-1) ** (1 / p)
    assert torch.allclose(lp_norm(z, p, 2), ref)
    big = 1e30 * torch.ones(2, 10)
    assert torch.isfinite(lp_norm(big, 37.0, 1)).all()


@pytest.mark.parametrize("p", PS + [0.7])
def test_sample_unit_sphere(p):
    u = sample_unit_sphere(20000, (3,), p)
    assert u.shape == (20000, 3)
    assert torch.allclose(lp_norm(u, p, 1), torch.ones(20000), atol=1e-10)
    # sign symmetry / centering
    assert u.mean(0).abs().max() < 0.03
    assert ((u > 0).double().mean(0) - 0.5).abs().max() < 0.02


def test_sample_unit_sphere_inf_faces():
    u = sample_unit_sphere(30000, (4,), INF)
    on_face = u.abs() == 1.0
    assert (on_face.sum(-1) >= 1).all()
    face_pos = (u == 1.0).double().mean(0)
    face_neg = (u == -1.0).double().mean(0)
    assert (face_pos - 1 / 8).abs().max() < 0.01 and (face_neg - 1 / 8).abs().max() < 0.01
    # the other coordinates are uniform on [-1, 1]
    rest = u[:, 1][~on_face[:, 1]]
    assert stats.kstest(rest.numpy(), "uniform", args=(-1, 2)).pvalue > 1e-3


def test_sample_unit_sphere_cone_measure_l1():
    # The cone measure on the 2-D L1 sphere has a uniform first coordinate.
    u = sample_unit_sphere(20000, (2,), 1.0)
    assert stats.kstest(u[:, 0].numpy(), "uniform", args=(-1, 2)).pvalue > 1e-3


# ============================================================ norm laws
@pytest.mark.parametrize("name", list(FAMILIES))
@pytest.mark.parametrize("d", [1, 3, 10, 784])
@pytest.mark.parametrize("p", PS)
def test_norm_normalization_default(name, d, p):
    norm = FAMILIES[name](d, p)
    assert abs(integral(norm) - 1.0) < 2e-5


@pytest.mark.parametrize("name", list(FAMILIES))
@pytest.mark.parametrize("d", [2, 50])
@pytest.mark.parametrize("p", [1.5, INF])
def test_norm_normalization_perturbed(name, d, p):
    norm = perturb(FAMILIES[name](d, p))
    assert abs(integral(norm) - 1.0) < 2e-5


@pytest.mark.parametrize("name", list(FAMILIES))
@pytest.mark.parametrize("d,p", [(3, 2.0), (40, 1.5)])
def test_norm_samples_match_cdf(name, d, p):
    norm = perturb(FAMILIES[name](d, p), 0.2)
    with torch.no_grad():
        r = norm.sample((3000,))
        assert r.shape == (3000,) and (r > 0).all()
        res = stats.kstest(r.numpy(), lambda x: norm.cdf(torch.as_tensor(x)).numpy())
    assert res.pvalue > 1e-3, res


@pytest.mark.parametrize("name", list(FAMILIES))
def test_norm_cdf_matches_density_and_icdf(name):
    norm = perturb(FAMILIES[name](5, 2.0), 0.2)
    with torch.no_grad():
        q = torch.tensor([0.01, 0.2, 0.5, 0.9, 0.999])
        r = norm.icdf(q)
        assert torch.allclose(norm.cdf(r), q, atol=2e-6)
        # cdf equals the integral of the density
        s = torch.linspace(norm.log_radius_bounds()[0] - 2, math.log(float(r[2])), 200_001)
        mass = torch.trapezoid(norm.log_prob_log_radius(s).exp(), s)
        assert abs(float(mass) - 0.5) < 1e-5


@pytest.mark.parametrize("name", list(FAMILIES))
def test_norm_gradients_finite(name):
    norm = perturb(FAMILIES[name](6, 1.5), 0.2)
    r = norm.rsample((64,))
    loss = norm.log_prob(r).mean() + norm.log_prob(torch.rand(16) * 5).mean()
    params = [q for q in norm.parameters() if q.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    assert any(g is not None for g in grads)
    for g in grads:
        assert g is None or torch.isfinite(g).all()


def test_rsample_is_reparameterized():
    norm = GeneralizedGamma(4, 2.0)
    r = norm.rsample((100,))
    (g,) = torch.autograd.grad(r.mean(), [norm.loc_global])
    assert g.abs() > 0


def test_gg_special_cases():
    r = torch.linspace(0.05, 6, 50)
    chi = GeneralizedGamma(3, 2.0)  # chi(3)
    chi_ref = torch.tensor(stats.chi(3).logpdf(r.numpy()))
    assert torch.allclose(chi.log_prob(r), chi_ref, atol=1e-10)
    gam = GammaMixture(5, 1.0, shape=2.5, scale=0.7)
    assert torch.allclose(gam.log_prob(r), torch.tensor(stats.gamma(2.5, scale=0.7).logpdf(r.numpy())), atol=1e-10)
    wb = WeibullMixture(5, 2.0, shape=1.7, scale=1.3)
    assert torch.allclose(wb.log_prob(r), torch.tensor(stats.weibull_min(1.7, scale=1.3).logpdf(r.numpy())), atol=1e-10)
    ln = LogNormalMixture(5, 2.0, loc=0.3, scale=0.4)
    assert torch.allclose(ln.log_prob(r), torch.tensor(stats.lognorm(0.4, scale=math.exp(0.3)).logpdf(r.numpy())), atol=1e-10)


def test_monotonicity_flags():
    d = 8
    assert GeneralizedGamma(d, 2.0, a=d).profile_is_monotone()
    assert GeneralizedGamma(d, 2.0, a=d - 0.5).profile_is_monotone()
    assert not GeneralizedGamma(d, 2.0, a=d + 0.5).profile_is_monotone()
    assert GeneralizedGamma(d, 2.0, monotone=True).profile_is_monotone()
    assert GammaMixture(d, 1.0, shape=d).profile_is_monotone()
    assert not GammaMixture(d, 1.0, shape=2 * d).profile_is_monotone()
    assert GammaMixture(d, 1.0, monotone=True).profile_is_monotone()
    assert WeibullMixture(d, 2.0, monotone=True).profile_is_monotone()
    assert not WeibullMixture(d, 2.0, shape=d + 1.0).profile_is_monotone()
    assert not LogNormalMixture(d, 2.0).profile_is_monotone()
    assert KhinchineNorm(d, 2.0, mixing=LogNormalMixture(d, 2.0)).profile_is_monotone()
    assert StudentTNorm(d, 2.0).profile_is_monotone()
    assert MonotoneProfileNorm(d, 2.0).profile_is_monotone()
    assert not SplineNorm(d, 2.0).profile_is_monotone()
    # mixed shapes fall back to a numerical check that must agree with a direct one
    mix = GeneralizedGammaMixture(d, 2.0, 2, a=[d - 1.0, d + 3.0])
    s = torch.linspace(*mix.log_radius_bounds(), 20000)
    lh = mix.log_profile_log_radius(s)
    assert mix.profile_is_monotone() == bool((lh.diff() <= 1e-9).all())


@pytest.mark.parametrize("name", list(FAMILIES))
def test_monotone_flag_consistent_with_profile(name):
    norm = perturb(FAMILIES[name](5, 2.0), 0.2)
    s = torch.linspace(*norm.log_radius_bounds(), 20000)
    with torch.no_grad():
        lh = norm.log_profile_log_radius(s)
    numerically_monotone = bool((lh.diff() <= 1e-7 * (1 + lh[1:].abs())).all())
    if norm.profile_is_monotone():
        assert numerically_monotone
    if name in EXACTLY_MONOTONE:
        assert norm.profile_is_monotone()


def test_log_profile_consistent_with_log_prob():
    for name, make in FAMILIES.items():
        norm = perturb(make(7, 1.5), 0.2)
        r = torch.linspace(0.3, 4.0, 13)
        with torch.no_grad():
            lhs = norm.log_profile(r)
            from cjd_flows.radial import log_sphere_measure

            rhs = norm.log_prob(r) - log_sphere_measure(7, 1.5, r)
            # volume coordinate t = V r^d has density h(r)
            t = math.exp(log_unit_ball_volume(7, 1.5)) * r**7
            vol = norm.log_prob_volume(t)
        assert torch.allclose(lhs, rhs, atol=1e-8), name
        assert torch.allclose(lhs, vol, atol=1e-8), name


@pytest.mark.parametrize("d", [2, 10, 784, 3072])
@pytest.mark.parametrize("p", [1.0, 2.0, INF])
def test_khinchine_default_equals_reference(d, p):
    ref = GeneralizedGamma(d, p)
    kh = KhinchineNorm(d, p)
    lo, hi = ref.log_radius_bounds()
    s = torch.linspace(lo, hi, 3000)
    with torch.no_grad():
        lp_ref = ref.log_prob_log_radius(s)
        mask = lp_ref > lp_ref.max() - 25
        err = (kh.log_prob_log_radius(s) - lp_ref)[mask].abs().max()
    assert err < 2e-5
    r = torch.exp(s[mask][::100])
    assert torch.allclose(kh.cdf(r), ref.cdf(r), atol=1e-5)


def test_studentt_matches_multivariate_t():
    d, nu = 3, 4.0
    base = RadialBase((d,), 2.0, StudentTNorm(d, 2.0, df=nu, theta=1.0))
    z = torch.randn(50, d) * 2
    ref = (
        math.lgamma((nu + d) / 2) - math.lgamma(nu / 2) - d / 2 * math.log(nu * math.pi)
        - (nu + d) / 2 * torch.log1p((z**2).sum(-1) / nu)
    )
    assert torch.allclose(base.log_prob(z), ref, atol=1e-10)
    # 1-D sanity: Student-t
    b1 = RadialBase((1,), 2.0, StudentTNorm(1, 2.0, df=nu, theta=1.0))
    x = torch.linspace(-5, 5, 11).reshape(-1, 1)
    assert torch.allclose(b1.log_prob(x), torch.distributions.StudentT(nu).log_prob(x[:, 0]), atol=1e-10)


def test_spline_identity_init_and_invertibility():
    k = 6
    x = torch.linspace(-7, 7, 101)
    from cjd_flows.radial.splines import identity_derivative_init

    w, h = torch.zeros(k), torch.zeros(k)
    dd = torch.full((k - 1,), identity_derivative_init())
    y, lad = rq_spline(x, w, h, dd, 5.0)
    assert torch.allclose(y, x, atol=1e-12) and torch.allclose(lad, torch.zeros_like(x), atol=1e-12)
    w, h, dd = torch.randn(k), torch.randn(k), torch.randn(k - 1)
    y, lad = rq_spline(x, w, h, dd, 5.0)
    xr, lad_inv = rq_spline(y, w, h, dd, 5.0, inverse=True)
    assert torch.allclose(xr, x, atol=1e-10)
    assert torch.allclose(lad, -lad_inv, atol=1e-10)
    xg = x.clone().requires_grad_(True)
    yg, ladg = rq_spline(xg, w, h, dd, 5.0)
    (grad,) = torch.autograd.grad(yg.sum(), xg)
    assert torch.allclose(grad.log(), ladg, atol=1e-8)
    assert (y.diff() > 0).all()


# ============================================================ RadialBase
def test_gaussian_is_exact():
    for shape in [(3,), (2, 4, 4)]:
        base = RadialBase.gaussian(shape)
        z = torch.randn(17, *shape) * 1.7
        ref = torch.distributions.Normal(0.0, 1.0).log_prob(z).flatten(1).sum(-1)
        assert torch.allclose(base.log_prob(z), ref, atol=1e-11, rtol=1e-13)
        assert not any(p.requires_grad for p in base.parameters())


def test_laplace_and_generalized_gaussian_are_exact():
    base = RadialBase.laplace((5,))
    z = torch.randn(20, 5) * 2
    ref = torch.distributions.Laplace(0.0, 1.0).log_prob(z).sum(-1)
    assert torch.allclose(base.log_prob(z), ref, atol=1e-11)
    p = 3.0
    gg = RadialBase.generalized_gaussian((4,), p)
    log_c = math.log(2) + math.log(p) / p + math.lgamma(1 + 1 / p)
    ref = (-(z[:, :4].abs() ** p) / p - log_c).sum(-1)
    assert torch.allclose(gg.log_prob(z[:, :4]), ref, atol=1e-11)
    with pytest.raises(ValueError):
        RadialBase.generalized_gaussian((4,), INF)


def _mc_mass(base, d: int, half_width: float, n: int = 1_000_000) -> float:
    z = (torch.rand(n, d) * 2 - 1) * half_width
    with torch.no_grad():
        return float(base.log_prob(z).exp().mean() * (2 * half_width) ** d)


def _is_mass(base, loc: torch.Tensor, scale: float, n: int = 1_000_000) -> tuple:
    """Importance-sampling estimate of ∫ p with a heavy-tailed product-Cauchy proposal."""
    prop = torch.distributions.Cauchy(loc, torch.full_like(loc, scale))
    z = prop.sample((n,))
    with torch.no_grad():
        w = (base.log_prob(z) - prop.log_prob(z).sum(-1)).exp()
    return float(w.mean()), float(w.std() / math.sqrt(n))


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("name", ["gg", "khinchine_ln", "lognormal", "studentt", "monotone_profile", "spline"])
def test_radial_log_prob_normalization_mc(d, p, name):
    norm = perturb(FAMILIES[name](d, p), 0.15)
    loc = torch.tensor([0.3, -0.2, 0.1][:d])
    base = RadialBase((d,), p, norm, loc=loc)
    with torch.no_grad():
        scale = float(norm.icdf(torch.tensor(0.5)))
    est, se = _is_mass(base, loc, scale)
    assert abs(est - 1.0) < max(5 * se, 2e-3), (est, se)


@pytest.mark.parametrize("p", PS)
def test_radial_samples_have_right_norm_law(p):
    norm = perturb(GeneralizedGammaMixture(4, p, 2), 0.2)
    base = RadialBase((2, 2), p, norm, loc=torch.ones(2, 2))
    z = base.sample((4000,))
    assert z.shape == (4000, 2, 2)
    with torch.no_grad():
        r = base.norm_of(z)
        res = stats.kstest(r.numpy(), lambda x: norm.cdf(torch.as_tensor(x)).numpy())
    assert res.pvalue > 1e-3


def test_radial_base_checks_and_flow():
    with pytest.raises(ValueError):
        RadialBase((3,), 2.0, GeneralizedGamma(4, 2.0))
    with pytest.raises(ValueError):
        RadialBase((3,), 2.0, GeneralizedGamma(3, 1.0))
    base = RadialBase((3,), 1.5, learn_loc=True)
    flow = Flow(Identity(), base)
    x = torch.randn(8, 3)
    assert torch.allclose(flow.log_prob(x), base.log_prob(x))
    assert flow.sample((5,)).shape == (5, 3)
    loss = -base.log_prob(x).mean()
    loss.backward()
    assert torch.isfinite(base.loc.grad).all()
    assert base.is_monotone()


@pytest.mark.parametrize("d", [784, 3072])
@pytest.mark.parametrize("name", list(FAMILIES))
def test_high_dimensional(d, name):
    p = 2.0 if d == 784 else 1.0
    base = RadialBase((d,), p, FAMILIES[name](d, p))
    z = base.sample((64,))
    assert torch.isfinite(z).all()
    lp = base.log_prob(z)
    assert torch.isfinite(lp).all()
    # data at the wrong scale and the origin must still give finite values
    lp2 = base.log_prob(torch.cat([3 * z[:4], 0.1 * z[:4], torch.zeros(1, d)]))
    assert not torch.isnan(lp2).any() and (lp2 < float("inf")).all()
    assert torch.isfinite(lp2[:8]).all()
    params = [q for q in base.parameters() if q.requires_grad]
    if params:
        grads = torch.autograd.grad(-lp.mean(), params, allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_float32_high_dim_and_double(name):
    torch.set_default_dtype(torch.float32)
    d = 3072
    base = RadialBase((d,), 2.0, FAMILIES[name](d, 2.0))
    z = base.sample((32,))
    assert z.dtype == torch.float32
    assert torch.isfinite(base.log_prob(z)).all()
    base = base.double()
    z = base.sample((8,))
    assert z.dtype == torch.float64
    lp = base.log_prob(z)
    assert lp.dtype == torch.float64 and torch.isfinite(lp).all()


def test_gaussian_high_dim_exact():
    base = RadialBase.gaussian((3, 32, 32))
    z = torch.randn(4, 3, 32, 32)
    ref = torch.distributions.Normal(0.0, 1.0).log_prob(z).flatten(1).sum(-1)
    assert torch.allclose(base.log_prob(z), ref, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("name", list(FAMILIES))
def test_cuda(name):
    torch.set_default_dtype(torch.float32)
    base = RadialBase((50,), 2.0, FAMILIES[name](50, 2.0)).to("cuda")
    z = base.sample((16,))
    assert z.device.type == "cuda"
    lp = base.log_prob(z)
    assert torch.isfinite(lp).all()
    lp.mean().backward()
    if base.is_monotone():
        assert torch.isfinite(udl_radius(base, 0.9))
    assert torch.isfinite(hdr_pvalue(base, z)).all()


# ============================================================= mixtures
def test_radial_mixture_normalization_and_sampling():
    d = 2
    norms = [GeneralizedGamma(d, 1.5, theta=0.6), StudentTNorm(d, 1.5, df=6.0), KhinchineNorm(d, 1.5)]
    locs = torch.tensor([[2.0, 0.0], [-2.0, 1.0], [0.0, -2.0]])
    mix = RadialMixture((d,), p=1.5, norms=norms, locs=locs, weights=[0.5, 0.2, 0.3])
    est = _mc_mass(mix, d, 25.0, 2_000_000)
    assert abs(est - 1.0) < 0.02
    z = mix.sample((20000,))
    assert z.shape == (20000, 2)
    assert torch.allclose(z.mean(0), (mix.weights.unsqueeze(-1) * locs).sum(0), atol=0.15)
    # posterior of the dominant component near its centre
    assert mix.responsibilities(locs[:1])[0, 0] > 0.9
    lp = mix.log_prob(z[:100])
    lp.mean().backward()
    assert torch.isfinite(mix.locs.grad).all() and torch.isfinite(mix.logits.grad).all()


def test_level_set_sandwich():
    d = 2
    mix = RadialMixture(
        (d,), p=2.0, norms=[GeneralizedGamma(d, 2.0, theta=0.8), GeneralizedGamma(d, 2.0, theta=1.4)],
        locs=torch.tensor([[1.0, 0.0], [-1.0, 0.5]]), weights=[0.4, 0.6],
    )
    log_t = torch.tensor(-3.0)
    inner, outer = mix.level_set_sandwich(log_t)
    assert inner.shape == (2,) and (inner <= outer).all() and (inner > 0).all()
    z = (torch.rand(200_000, d) * 2 - 1) * 6
    with torch.no_grad():
        lp = mix.log_prob(z)
        rk = mix.component_norms(z)
    in_inner = (rk <= inner).any(-1)
    in_outer = (rk <= outer).any(-1)
    assert (lp[in_inner] >= log_t - 1e-9).all()
    assert in_outer[lp >= log_t].all()
    # threshold above the max density gives empty balls
    inner_e, _ = mix.level_set_sandwich(torch.tensor(5.0))
    assert (inner_e == 0).all()
    with pytest.raises(ValueError):
        RadialMixture((d,), p=2.0, norms=[LogNormalMixture(d, 2.0)]).level_set_sandwich(-3.0)


# ============================================================ level sets
@pytest.mark.parametrize("name", ["gg_mono", "khinchine", "studentt", "monotone_profile"])
@pytest.mark.parametrize("p", [1.0, 2.0, INF])
def test_udl_radius_mass(name, p):
    d = 5
    base = RadialBase((d,), p, perturb(FAMILIES[name](d, p), 0.15))
    z = base.sample((40000,))
    for q in [0.5, 0.9, 0.99]:
        rq = udl_radius(base, q)
        frac = float((base.norm_of(z) <= rq).double().mean())
        assert abs(frac - q) < 4 * math.sqrt(q * (1 - q) / 40000) + 1e-3
        # the ball is an upper level set: inside is denser than outside
        lp = base.log_prob(z)
        inside = base.norm_of(z) <= rq
        assert lp[inside].min() >= lp[~inside].max() - 1e-9


def test_udl_radius_requires_monotone():
    with pytest.raises(ValueError):
        udl_radius(RadialBase((3,), 2.0, LogNormalMixture(3, 2.0)), 0.9)


@pytest.mark.parametrize("name", ["gg", "khinchine", "lognormal", "spline"])
def test_hdr_pvalue_uniform_under_model(name):
    base = RadialBase((4,), 1.5, perturb(FAMILIES[name](4, 1.5), 0.15))
    z = base.sample((5000,))
    with torch.no_grad():
        pv = hdr_pvalue(base, z)
        lp = base.log_prob(z)
    assert ((pv >= 0) & (pv <= 1)).all()
    assert stats.kstest(pv.numpy(), "uniform").pvalue > 1e-3
    # more likely points have smaller values
    order = torch.argsort(lp, descending=True)
    assert (pv[order].diff() >= -1e-3).all()


def test_density_threshold_radius():
    norm = GeneralizedGamma(6, 2.0, a=5.0)  # monotone, unbounded at 0
    thr = torch.tensor([-8.0, -5.0, -2.0, 3.0])
    rho = density_threshold_radius(norm, thr)
    assert torch.allclose(norm.log_profile(rho), thr, atol=1e-8)
    bounded = RadialBase.gaussian((6,))
    top = float(bounded.log_profile(torch.tensor(0.0)))
    rho = density_threshold_radius(bounded, torch.tensor([top + 1.0, top - 2.0]))
    assert rho[0] == 0 and abs(float(rho[1]) - 2.0) < 1e-8  # -r^2/2 = -2


def test_udl_intervals_monotone_and_shell():
    base = RadialBase((3,), 2.0, GeneralizedGamma(3, 2.0))
    iv = udl_intervals(base, q=0.9)
    assert iv.shape == (1, 2) and iv[0, 0] == 0
    assert abs(float(iv[0, 1]) - float(udl_radius(base, 0.9))) < 5e-3
    shell = RadialBase((3,), 2.0, LogNormalMixture(3, 2.0, loc=1.0, scale=0.2))
    iv = udl_intervals(shell, q=0.8)
    assert iv.shape == (1, 2) and iv[0, 0] > 0
    norm = shell.norm
    mass = float((norm.cdf(iv[0, 1]) - norm.cdf(iv[0, 0])).detach())
    assert abs(mass - 0.8) < 5e-3
    assert abs(float((norm.log_profile(iv[0, 0]) - norm.log_profile(iv[0, 1])).detach())) < 0.05
    thr = udl_log_threshold(shell, 0.8)
    assert torch.allclose(udl_intervals(shell, log_threshold=thr), iv)
    # two separated shells -> two intervals at a high threshold
    two = LogNormalMixture(3, 2.0, 2, loc=[0.0, 2.0], scale=[0.1, 0.1], spread=0.0)
    iv2 = udl_intervals(two, q=0.5)
    assert iv2.shape == (2, 2) and (iv2[1:, 0] > iv2[:-1, 1]).all()
    assert udl_intervals(two, log_threshold=1e3).shape == (0, 2)
    with pytest.raises(ValueError):
        udl_intervals(two)


# ================================================================ fitting
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_fit_norm_to_radii(optimizer):
    target = GeneralizedGamma(10, 2.0, theta=2.0, beta=1.5)
    radii = target.sample((5000,))
    norm = GeneralizedGamma(10, 2.0)
    before = float(-norm.log_prob(radii).mean().detach())
    losses = fit_norm_to_radii(norm, radii, steps=300 if optimizer == "adam" else 100, lr=0.05 if optimizer == "adam" else 0.5, optimizer=optimizer)
    after = float(-norm.log_prob(radii).mean().detach())
    best = float(-target.log_prob(radii).mean().detach())
    assert after < before and after - best < 0.01
    assert len(losses) > 0


def test_fix_scale_option():
    for norm in [GeneralizedGamma(5, learn_scale=False), LogNormalMixture(5, n_components=3, learn_scale=False),
                 StudentTNorm(5, learn_scale=False), GeneralizedGammaMixture(5, 2.0, 3, learn_scale=False)]:
        names = {n for n, _ in norm.named_parameters()}
        assert "loc_global" not in names
    mix = GeneralizedGammaMixture(5, 2.0, 3, learn_scale=False)
    mean0 = float(mix.log_theta.mean().detach())
    with torch.no_grad():
        mix.loc_offsets.add_(torch.randn(3))
    assert abs(float(mix.log_theta.mean().detach()) - mean0) < 1e-12
