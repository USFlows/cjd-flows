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


def test_binned_uniformity_rejects_high_dimensions():
    flow = _flow(d=24)
    ev = RadialFlowEvaluator(flow, _samples(flow, 50))
    with pytest.raises(ValueError):
        ev.binned_uniformity_test(n_bins=10)


def test_legacy_location_reexports():
    pytest.importorskip("pyro")
    from cjd_flows.explib.eval import RadialFlowEvaluator as Legacy

    assert Legacy is RadialFlowEvaluator
