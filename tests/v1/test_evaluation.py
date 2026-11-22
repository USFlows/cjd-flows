import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cjd_flows import USFlow  # noqa: E402
from cjd_flows.evaluation import RadialFlowEvaluator  # noqa: E402


def _flow(base="laplace", p=1.0, d=4):
    torch.manual_seed(0)
    flow = USFlow((d,), blocks=2, hidden=(16,), base=base, p=p).double().eval()
    with torch.no_grad():
        for prm in flow.parameters():
            prm.add_(torch.randn_like(prm) * 0.05)
    return flow


def _samples(flow, n=1500):
    torch.manual_seed(1)
    np.random.seed(1)
    with torch.no_grad():
        return flow.sample(n)


def test_l1_flow_samples_match_base():
    flow = _flow()
    ev = RadialFlowEvaluator(flow, _samples(flow))
    assert ev.p == 1.0 and ev.latents.shape == (1500, 4)
    assert ev.ks_norm_statistic().pvalue > 0.01
    assert ev.wasserstein_norm_distance() < 0.1 * float(ev._latent_norms().mean())
    assert not ev.test_sign_symmetry(method="sign", combine="stouffer")["reject"]
    p_value, reject = ev.test_uniformity_simplex(n_samples_ref=300, n_boot=100)
    assert not reject and 0 <= p_value <= 1
    res = ev.test_l1_radial_symmetry(sign_method="sign", sign_combine="stouffer")
    assert not res["l1_radial_rejected"]


def test_detects_norm_misfit():
    flow = _flow()
    x = _samples(flow)
    # pushing the samples away from the centre stretches the latent norms
    shifted = flow.base.loc + 1.5 * (flow.transform(x) - flow.base.loc)
    x_bad = flow.bijection.inverse(shifted).detach()
    ev = RadialFlowEvaluator(flow, x_bad)
    assert ev.ks_norm_statistic().pvalue < 1e-6
    good = RadialFlowEvaluator(flow, x).wasserstein_norm_distance()
    assert ev.wasserstein_norm_distance() > 5 * good


def test_gaussian_base_is_l2_radial():
    flow = _flow(base="gaussian", p=2.0)
    ev = RadialFlowEvaluator(flow, _samples(flow))
    assert ev.p == 2.0
    assert ev.ks_norm_statistic().pvalue > 0.01
    with pytest.raises(ValueError):
        ev.test_sign_symmetry()  # L1 only


def test_non_radial_base_needs_p():
    flow = _flow(base="radial-mixture", p=2.0)
    x = _samples(flow, 200)
    with pytest.raises(ValueError):
        RadialFlowEvaluator(flow, x)
    with pytest.raises(ValueError):
        RadialFlowEvaluator(flow, x, p=2.0)  # norm distribution missing
    from cjd_flows.radial import RadialBase

    gauss = RadialBase.gaussian((4,))
    ev = RadialFlowEvaluator(flow, x, p=2.0, norm_distribution=gauss.norm.double(), loc=torch.zeros(4))
    assert np.isfinite(ev.wasserstein_norm_distance())
    with pytest.raises(ValueError):
        RadialFlowEvaluator(flow, x, p=-1.0, norm_distribution=gauss.norm)


def test_plots_and_independence_tests():
    flow = _flow(d=2)
    ev = RadialFlowEvaluator(flow, _samples(flow, 300))
    rc_before = dict(plt.rcParams)
    ref = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    for ax in [
        ev.qq_plot_norms(n_samples=500),
        ev.kde_plot_norms(n_samples=500),
        ev.pp_plot_norms(),
        ev.nll_norm_scatter_plot(ref, n_samples=300),
        ev.logprob_reference_scatter_plot(ref, n_samples=300),
    ]:
        assert isinstance(ax, plt.Axes)
    assert dict(plt.rcParams) == rc_before  # no global style changes
    plt.close("all")

    hsic, p_value = ev.hs_independence_test(n_permutations=50)
    assert np.isfinite(hsic) and 0 <= p_value <= 1
    chi2_stat, p_value = ev.binned_uniformity_test(n_bins=4)
    assert np.isfinite(chi2_stat) and 0 <= p_value <= 1


def _stretched(flow, x, factor=1.5):
    """Data whose latents are stretched along the first coordinate (directions no longer radial)."""
    loc = getattr(flow.base, "loc", 0.0)
    z = flow.transform(x) - loc
    z[:, 0] *= factor
    return flow.bijection.inverse(z + loc).detach()


@pytest.mark.parametrize("p,base", [(1.0, "laplace"), (2.0, "gaussian")])
def test_binned_uniformity_uses_cone_measure(p, base):
    # equal-probability Cartesian cells would reject these radial samples
    flow = _flow(base=base, p=p, d=2)
    x = _samples(flow, 400)
    np.random.seed(0)
    assert RadialFlowEvaluator(flow, x).binned_uniformity_test(n_bins=4)[1] > 0.01  # was ~1e-39
    np.random.seed(0)
    assert RadialFlowEvaluator(flow, _stretched(flow, x, 2.0)).binned_uniformity_test(n_bins=4)[1] < 0.01


def test_simplex_uniformity_tests_are_calibrated():
    flow = _flow(d=3)
    ev = RadialFlowEvaluator(flow, _samples(flow, 300))
    np.random.seed(0)
    p_bh, reject_bh = ev.test_uniformity_simplex(method="bhattacharyya", n_boot=200)
    assert not reject_bh and p_bh > 0.05
    np.random.seed(0)
    bad = RadialFlowEvaluator(flow, _stretched(flow, _samples(flow, 300), 2.0))
    assert bad.test_uniformity_simplex(n_samples_ref=300, n_boot=100)[1]
    with pytest.raises(ValueError):
        ev.test_uniformity_simplex(method="nope")


def test_pp_plot_falls_back_without_cdf():
    class SampleOnly(torch.distributions.Distribution):
        arg_constraints = {}

        def __init__(self):
            super().__init__(validate_args=False)

        def sample(self, sample_shape=()):
            return torch.distributions.Chi2(torch.tensor(4.0, dtype=torch.float64)).sample(sample_shape).sqrt()

    flow = _flow(base="gaussian", p=2.0)
    ev = RadialFlowEvaluator(flow, _samples(flow, 200), p=2.0, norm_distribution=SampleOnly(), loc=torch.zeros(4))
    with pytest.raises(NotImplementedError):
        SampleOnly().cdf(torch.ones(1))
    assert isinstance(ev.pp_plot_norms(n_samples=500), plt.Axes)
    plt.close("all")


def test_hsic_matches_trace_formula():
    flow = _flow(d=3)
    ev = RadialFlowEvaluator(flow, _samples(flow, 120))
    torch.manual_seed(0)
    hsic, p_value = ev.hs_independence_test(n_permutations=30)
    # reference: the explicit trace(H K_n H H K_d H) / n^2
    norms = torch.linalg.vector_norm(ev.latents, ord=1, dim=1).unsqueeze(1)
    dirs = ev.latents / norms
    norms = (norms - norms.mean()) / norms.std()
    dirs = (dirs - dirs.mean(0)) / dirs.std(0)
    n = len(norms)
    H = torch.eye(n, dtype=norms.dtype) - 1.0 / n
    ref = torch.trace(H @ ev._rbf_kernel(norms) @ H @ H @ ev._rbf_kernel(dirs) @ H) / n**2
    assert abs(hsic - ref.item()) < 1e-10
    assert 0 <= p_value <= 1


def test_binned_uniformity_rejects_high_dimensions():
    flow = _flow(d=24)
    ev = RadialFlowEvaluator(flow, _samples(flow, 50))
    with pytest.raises(ValueError):
        ev.binned_uniformity_test(n_bins=10)


def test_legacy_location_reexports():
    pytest.importorskip("pyro")
    from cjd_flows.explib.eval import RadialFlowEvaluator as Legacy

    assert Legacy is RadialFlowEvaluator
