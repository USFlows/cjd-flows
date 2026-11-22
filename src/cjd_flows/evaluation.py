"""Latent-space diagnostics for flows with an Lp-radial base distribution.

:class:`RadialFlowEvaluator` maps data to the latent space of a trained flow
and checks how well the latents match the radial structure of the base:

* **norm distribution**: Wasserstein distance and Kolmogorov-Smirnov test
  between the latent norms ``||z - loc||_p`` and the base norm law; QQ, PP and
  KDE plots;
* **radiality**: sign symmetry of the coordinates and uniformity of the
  normalized absolute latents on the simplex (L1 bases), binned directional
  uniformity, and a permutation HSIC test of norm/direction independence;
* **density alignment**: scatter plots of the negative log-likelihood against
  latent norms, and of learned against reference log-probabilities.

Mismatches in the norm diagnostics tell over- from under-approximation of the
density level sets (latent norms too large / too small compared to the base).

Works with v1 flows (:class:`~cjd_flows.Flow` with a
:class:`~cjd_flows.RadialBase` or :class:`~cjd_flows.StandardNormal` base) and
with the pyro-based 0.9 flows (``base_distribution`` and ``backward``).
Needs the ``eval`` extra (``pip install "cjd-flows[eval]"``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.stats import binomtest, chi2, ks_2samp, norm, wasserstein_distance, wilcoxon
    from sklearn.neighbors import KernelDensity
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('cjd_flows.evaluation needs the "eval" extra: pip install "cjd-flows[eval]"') from e

__all__ = ["RadialFlowEvaluator"]

_PLOT_RC = {
    "text.usetex": False,
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
}


def _plot_context():
    """Plot style for the diagnostics, without changing the global matplotlib state."""
    import matplotlib

    return matplotlib.rc_context({**plt.style.library["ggplot"], **_PLOT_RC})


def _radial_parts(flow):
    """``(p, norm_distribution, loc, to_latent)`` of a v1 or 0.9 flow, or ``None`` entries if unknown."""
    base = getattr(flow, "base", None)
    if base is not None and hasattr(flow, "transform"):  # v1 Flow
        from cjd_flows.core import StandardNormal
        from cjd_flows.radial import RadialBase

        if isinstance(base, RadialBase):
            return base.p, base.norm, base.loc, flow.transform
        if isinstance(base, StandardNormal):
            # standard normal = L2-radial with a chi(d) norm law
            gauss = RadialBase.gaussian(base.event_shape)
            return 2.0, gauss.norm, gauss.loc, flow.transform
        return None, None, None, flow.transform
    # pyro-based 0.9 flow
    bd = getattr(flow, "base_distribution", None)
    p = getattr(bd, "p", None)
    norm_distribution = getattr(bd, "norm_distribution", None)
    loc = getattr(bd, "loc", None)
    return p, norm_distribution, loc, flow.backward


class RadialFlowEvaluator:
    """Diagnostics of the latent radial structure of a trained flow.

    Args:
        flow: trained flow. For a v1 :class:`~cjd_flows.Flow` with a
            :class:`~cjd_flows.RadialBase` or :class:`~cjd_flows.StandardNormal`
            base, and for 0.9 flows with a ``RadialDistribution`` base, ``p``,
            the norm distribution and ``loc`` are read from the flow.
        data: data tensor ``(N, *event_shape)`` to evaluate on.
        device: device for the computations.
        p: norm exponent, required if the base is not radial (e.g. a
            :class:`~cjd_flows.RadialMixture`); overrides the flow's otherwise.
        norm_distribution: distribution of the latent norm (needs ``sample``;
            ``cdf`` is used if available), required with ``p``.
        loc: latent centre, required if the base has none.
    """

    def __init__(
        self,
        flow,
        data: torch.Tensor,
        device="cpu",
        p: Optional[float] = None,
        norm_distribution=None,
        loc: Optional[torch.Tensor] = None,
    ):
        self.flow = flow.to(device)
        self.data = data.to(device)
        self.device = device
        self.dim = math.prod(self.data.shape[1:])

        flow_p, flow_norm, flow_loc, self._to_latent = _radial_parts(flow)
        if p is None:
            p = flow_p
            norm_distribution = flow_norm if norm_distribution is None else norm_distribution
        if p is None:
            raise ValueError("p must be specified for flows without an Lp-radial base")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise TypeError("p must be an integer or float")
        if p <= 0:
            raise ValueError("p must be a positive number")
        if norm_distribution is None:
            raise ValueError("norm_distribution must be specified together with p")
        self.p = float(p)
        self.norm_distribution = norm_distribution

        if loc is None:
            loc = flow_loc
        if loc is None:
            raise ValueError("loc must be specified for flows without a base centre")
        self.loc = torch.as_tensor(loc).to(device)

        with torch.no_grad():
            self.latents = self._latents_of(self.data)

    # ------------------------------------------------------------------ helpers
    def _latents_of(self, x: torch.Tensor) -> torch.Tensor:
        """Centred, flattened latents ``(N, D)`` of data ``x``."""
        z = self._to_latent(x) - self.loc.to(x.dtype)
        return z.reshape(z.shape[0], -1)

    def _latent_norms(self) -> np.ndarray:
        return torch.linalg.vector_norm(self.latents, ord=self.p, dim=1).detach().cpu().numpy()

    def _norm_samples(self, n_samples: int) -> np.ndarray:
        with torch.no_grad():
            return self.norm_distribution.sample((n_samples,)).detach().cpu().numpy().reshape(-1)

    # ------------------------------------------------------- norm distribution
    def wasserstein_norm_distance(self, n_samples=10000):
        """Wasserstein-1 distance between the latent norms and the base norm law."""
        return wasserstein_distance(self._latent_norms(), self._norm_samples(n_samples))

    def ks_norm_statistic(self, n_samples=10000):
        """Two-sample Kolmogorov-Smirnov test of latent norms vs. base norm samples.

        Returns the scipy result (``statistic``, ``pvalue``).
        """
        return ks_2samp(self._latent_norms(), self._norm_samples(n_samples))

    def qq_plot_norms(self, ax=None, n_samples=10000):
        """QQ plot of the latent norms against the base norm law."""
        latent_q = np.quantile(self._latent_norms(), np.linspace(0, 1, 100))
        sample_q = np.quantile(self._norm_samples(n_samples), np.linspace(0, 1, 100))
        with _plot_context():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(sample_q, latent_q, alpha=0.7)
            lo, hi = min(sample_q.min(), latent_q.min()), max(sample_q.max(), latent_q.max())
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set_title("QQ-plot of Latent Norms")
            ax.set_xlabel("Base Distribution Quantiles")
            ax.set_ylabel("Data Latent Quantiles")
        return ax

    def kde_plot_norms(self, ax=None, n_samples=10000, bandwidth=0.2):
        """Kernel density estimates of the latent norms and the base norm law."""
        latent_norms = self._latent_norms()
        sample_norms = self._norm_samples(n_samples)
        kde_latent = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(latent_norms.reshape(-1, 1))
        kde_base = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(sample_norms.reshape(-1, 1))
        grid = np.linspace(
            min(latent_norms.min(), sample_norms.min()), max(latent_norms.max(), sample_norms.max()), 1000
        )[:, None]
        with _plot_context():
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 6))
            ax.plot(grid, np.exp(kde_latent.score_samples(grid)), label="Data Latents")
            ax.plot(grid, np.exp(kde_base.score_samples(grid)), label="Base Distribution")
            ax.set_title("KDE of Norm Distributions")
            ax.set_xlabel("Norm Value")
            ax.set_ylabel("Density")
            ax.legend()
        return ax

    def pp_plot_norms(self, ax=None, n_samples=10000):
        """PP plot: empirical CDF of the latent norms against the base norm CDF.

        Uses ``norm_distribution.cdf`` if available, otherwise a Monte-Carlo CDF.
        """
        sorted_norms = np.sort(self._latent_norms())
        n = len(sorted_norms)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = None
        if hasattr(self.norm_distribution, "cdf"):
            try:
                with torch.no_grad():
                    r = torch.as_tensor(sorted_norms, device=self.device, dtype=self.latents.dtype)
                    theoretical_cdf = self.norm_distribution.cdf(r).detach().cpu().numpy()
            except NotImplementedError:  # e.g. torch.distributions without a CDF
                theoretical_cdf = None
        if theoretical_cdf is None:
            samples = np.sort(self._norm_samples(n_samples))
            theoretical_cdf = np.searchsorted(samples, sorted_norms) / n_samples
        with _plot_context():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(theoretical_cdf, empirical_cdf, alpha=0.7)
            ax.plot([0, 1], [0, 1], "r--")
            ax.set_title("PP-plot of Norm Distributions")
            ax.set_xlabel("Theoretical CDF (Base Distribution)")
            ax.set_ylabel("Empirical CDF (Data Latents)")
            ax.grid(True)
        return ax

    # --------------------------------------------------------------- radiality
    def binned_uniformity_test(self, n_bins=10, n_reference=None):
        """Chi-squared test of the binned directions ``z / ||z||_p`` against an Lp-radial law.

        Every coordinate of the direction is binned into ``n_bins`` cells on
        ``[-1, 1]``. The expected cell probabilities are those of the cone
        measure on the Lp unit sphere (the direction law of every Lp-radial
        distribution), estimated from ``n_reference`` exact simulations
        (default ``max(100_000, 50 N)``). Cells with an expected count below 5
        are pooled. Only meaningful for low dimensions (``n_bins ** dim``
        cells). Returns ``(chi2, p_value)``.
        """
        total_bins = n_bins**self.dim
        if total_bins > 2**62:
            raise ValueError(f"n_bins ** dim = {n_bins}**{self.dim} cells is too many; use a low-dimensional latent")
        directions = self.latents / torch.linalg.vector_norm(self.latents, ord=self.p, dim=1, keepdim=True)
        directions = directions.detach().cpu().numpy()
        n = len(directions)
        if n_reference is None:
            n_reference = max(100_000, 50 * n)
        ref = self._simulate_lp_directions(n_reference, self.dim)
        obs_bins, obs_counts = np.unique(self._bin_index(directions, n_bins), return_counts=True)
        ref_bins, ref_counts = np.unique(self._bin_index(ref, n_bins), return_counts=True)
        cells = np.union1d(obs_bins, ref_bins)
        observed = np.zeros(len(cells))
        observed[np.searchsorted(cells, obs_bins)] = obs_counts
        expected = np.zeros(len(cells))
        expected[np.searchsorted(cells, ref_bins)] = ref_counts * (n / n_reference)
        rare = expected < 5
        if rare.any():
            observed = np.append(observed[~rare], observed[rare].sum())
            expected = np.append(expected[~rare], expected[rare].sum())
            if expected[-1] == 0:  # observations only in cells the reference never hit
                expected[-1] = n / n_reference
        if len(expected) < 2:
            raise ValueError("Too few populated cells for a chi-squared test; use more data or fewer bins")
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        p_value = 1 - chi2.cdf(chi2_stat, df=len(expected) - 1)
        return chi2_stat, p_value

    def _simulate_lp_directions(self, n, d):
        """Exact samples of the cone measure on the Lp unit sphere."""
        if math.isinf(self.p):
            g = np.random.uniform(-1.0, 1.0, size=(n, d))
            return g / np.abs(g).max(axis=1, keepdims=True)
        # i.i.d. coordinates with density proportional to exp(-|x|^p), normalized
        g = np.random.gamma(1.0 / self.p, 1.0, size=(n, d)) ** (1.0 / self.p)
        g *= np.random.choice([-1.0, 1.0], size=(n, d))
        return g / (np.abs(g) ** self.p).sum(axis=1, keepdims=True) ** (1.0 / self.p)

    @staticmethod
    def _bin_index(directions, n_bins):
        edges = np.linspace(-1, 1, n_bins + 1)
        index = np.zeros(len(directions), dtype=np.int64)
        for dim in range(directions.shape[1]):
            idx = np.clip(np.digitize(directions[:, dim], edges) - 1, 0, n_bins - 1)
            index += idx.astype(np.int64) * (n_bins**dim)
        return index

    def hs_independence_test(self, n_permutations=1000):
        """Permutation HSIC test of H0: latent norm and direction are independent.

        Returns ``(hsic, p_value)``.
        """
        norms = torch.linalg.vector_norm(self.latents, ord=self.p, dim=1).unsqueeze(1)
        directions = self.latents / norms
        norms = (norms - norms.mean()) / norms.std()
        directions = (directions - directions.mean(dim=0)) / directions.std(dim=0)
        K_n = self._rbf_kernel(norms)
        K_d = self._rbf_kernel(directions)
        n = len(norms)
        H = torch.eye(n, device=K_n.device, dtype=K_n.dtype) - 1.0 / n
        K_n = H @ K_n @ H
        K_d = H @ K_d @ H
        hsic_value = torch.trace(K_n @ K_d) / (n * n)
        permuted = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = torch.randperm(n, device=K_d.device)
            permuted[i] = torch.trace(K_n @ K_d[perm][:, perm]).item()
        p_value = (permuted / (n * n) >= hsic_value.item()).mean()
        return hsic_value.item(), p_value

    def _rbf_kernel(self, X, sigma=None):
        """RBF kernel matrix with the median heuristic for ``sigma``."""
        X_norm = torch.sum(X**2, dim=1).reshape(-1, 1)
        pairwise_dist = (X_norm + X_norm.T - 2 * X @ X.T).clamp_min(0)
        if sigma is None:
            sigma = torch.median(pairwise_dist[pairwise_dist > 0]).sqrt()
        return torch.exp(-pairwise_dist / (2 * sigma**2))

    def test_uniformity_simplex(self, alpha=0.05, method="energy", n_samples_ref=1000, n_boot=1000):
        """Uniformity of ``|z| / ||z||_1`` on the simplex (L1 bases only).

        Args:
            alpha: significance level.
            method: ``"energy"`` (energy distance) or ``"bhattacharyya"``
                (KS distance of centred log-coordinates); both with
                p-values from simulated uniform samples.
            n_samples_ref: reference samples for the energy distance.
            n_boot: simulation repetitions for the p-value.

        Returns:
            ``(p_value, reject)``.
        """
        if self.p != 1:
            raise ValueError(f"Uniformity test requires L1 norm (p=1), current p={self.p}")
        abs_latents = torch.abs(self.latents)
        row_sums = abs_latents.sum(dim=1, keepdim=True)
        valid = (row_sums > 1e-8).squeeze(1)
        if valid.sum() < 10:
            raise ValueError("Insufficient non-zero latent vectors for uniformity test")
        u = (abs_latents[valid] / row_sums[valid]).detach().cpu().numpy()
        if method == "energy":
            return self._energy_uniformity_test(u, alpha, n_samples_ref, n_boot)
        if method == "bhattacharyya":
            return self._bhattacharyya_uniformity_test(u, alpha, n_boot)
        raise ValueError(f"Unknown method: {method}")

    def _energy_uniformity_test(self, u, alpha, n_samples_ref, n_boot):
        n, d = u.shape
        ref = self._simulate_uniform_simplex(n_samples_ref, d)
        stat_obs = self._energy_distance(u, ref)
        stat_boot = np.array([self._energy_distance(self._simulate_uniform_simplex(n, d), ref) for _ in range(n_boot)])
        p_value = np.mean(stat_boot >= stat_obs)
        return p_value, p_value < alpha

    def _bhattacharyya_uniformity_test(self, u, alpha, n_boot=1000):
        """KS distance of the centred log-coordinates to a Gumbel law, calibrated by simulation.

        The centred log-coordinates of a uniform simplex sample are neither
        exactly Gumbel nor independent, so the p-value comes from the same
        statistic on ``n_boot`` simulated uniform samples of the same size.
        Pooling all coordinates makes it weak against anisotropic latents;
        the energy test (default) has much more power there.
        """

        def statistic(v):
            y = -np.log(v)
            return stats.kstest((y - y.mean(axis=1, keepdims=True)).ravel(), "gumbel_r").statistic

        n, d = u.shape
        stat_obs = statistic(u)
        stat_boot = np.array([statistic(self._simulate_uniform_simplex(n, d)) for _ in range(n_boot)])
        p_value = np.mean(stat_boot >= stat_obs)
        return p_value, p_value < alpha

    def _simulate_uniform_simplex(self, n, d):
        """Uniform samples on the simplex (normalized exponentials)."""
        e = np.random.exponential(scale=1.0, size=(n, d))
        return e / e.sum(axis=1, keepdims=True)

    def _energy_distance(self, X, Y):
        """Energy distance between samples ``X`` and ``Y`` (rows)."""
        n, m = X.shape[0], Y.shape[0]
        xx = np.sum(X**2, axis=1)
        yy = np.sum(Y**2, axis=1)
        d_xx = xx[:, None] + xx[None, :] - 2 * X @ X.T
        d_yy = yy[:, None] + yy[None, :] - 2 * Y @ Y.T
        d_xy = xx[:, None] + yy[None, :] - 2 * X @ Y.T
        # round-off can make squared distances slightly negative
        term1 = np.sum(np.sqrt(np.clip(d_xy, 0.0, None))) / (n * m)
        term2 = np.sum(np.sqrt(np.clip(d_xx, 0.0, None))) / (n * n)
        term3 = np.sum(np.sqrt(np.clip(d_yy, 0.0, None))) / (m * m)
        return 2 * term1 - term2 - term3

    def test_sign_symmetry(self, alpha=0.05, method="sign", combine="stouffer"):
        """Per-coordinate sign symmetry of the latents (L1 bases only).

        Args:
            alpha: significance level.
            method: ``"sign"`` (binomial) or ``"wilcoxon"`` (signed rank).
            combine: ``"fisher"``, ``"stouffer"`` (sign test only) or ``None``
                (Bonferroni).

        Returns:
            dict with ``p_values``, ``reject``, ``combined_p`` and ``method``.
        """
        if self.p != 1:
            raise ValueError(f"Sign symmetry test requires L1 norm (p=1), current p={self.p}")
        if combine == "stouffer" and method != "sign":
            raise ValueError("Stouffer method requires sign test")
        p_values, z_scores = [], []
        latents = self.latents.detach().cpu().numpy()
        for j in range(latents.shape[1]):
            coord = latents[:, j]
            if method == "sign":
                n_pos = int((coord > 0).sum())
                p_val = binomtest(n_pos, len(coord), p=0.5, alternative="two-sided").pvalue
                z_scores.append((n_pos - len(coord) / 2) / np.sqrt(len(coord) / 4))
            elif method == "wilcoxon":
                _, p_val = wilcoxon(coord, zero_method="wilcox", alternative="two-sided")
                z_scores.append(norm.ppf(1 - p_val / 2) * np.sign(np.median(coord)))
            else:
                raise ValueError(f"Unknown method: {method}")
            p_values.append(p_val)

        combined_p = None
        if combine == "fisher":
            combined_p = 1 - chi2.cdf(-2 * np.sum(np.log(p_values)), 2 * len(p_values))
            reject = combined_p < alpha
        elif combine == "stouffer":
            z_combined = np.sum(z_scores) / np.sqrt(len(z_scores))
            combined_p = 2 * (1 - norm.cdf(np.abs(z_combined)))
            reject = combined_p < alpha
        else:
            reject = any(pv < alpha / self.dim for pv in p_values)
        return {
            "p_values": p_values,
            "reject": reject,
            "combined_p": combined_p,
            "method": f"{method} with {combine}" if combine else f"{method} with Bonferroni",
        }

    def test_l1_radial_symmetry(self, alpha=0.05, sign_method="wilcoxon", sign_combine="fisher", uniform_method="energy"):
        """Sign symmetry and simplex uniformity, each at level ``alpha / 2`` (L1 bases only).

        Returns:
            dict with the sign and uniformity results and ``l1_radial_rejected``.
        """
        if self.p != 1:
            raise ValueError(f"L1-radial test requires p=1, current p={self.p}")
        sign = self.test_sign_symmetry(alpha=alpha / 2, method=sign_method, combine=sign_combine)
        uniformity_pval, uniformity_reject = self.test_uniformity_simplex(alpha=alpha / 2, method=uniform_method)
        return {
            "sign_pvals": sign["p_values"],
            "sign_reject": sign["reject"],
            "sign_combined_p": sign["combined_p"],
            "sign_method": sign["method"],
            "uniformity_pval": uniformity_pval,
            "uniformity_reject": uniformity_reject,
            "l1_radial_rejected": sign["reject"] or uniformity_reject,
        }

    # -------------------------------------------------------- density alignment
    def nll_norm_scatter_plot(self, ref_distribution, ax=None, n_samples=10000):
        """Scatter of the reference NLL against the latent norms of reference samples.

        ``ref_distribution`` needs ``sample`` and ``log_prob`` (e.g. the true
        data distribution of a toy dataset). The legend reports Pearson,
        Spearman and Kendall correlations.
        """
        with torch.no_grad():
            samples = ref_distribution.sample((n_samples,)).to(self.device)
            nlls = -ref_distribution.log_prob(samples).cpu().numpy()
            latent_norms = torch.linalg.vector_norm(self._latents_of(samples), ord=self.p, dim=1).cpu().numpy()
        pearson_r, _ = stats.pearsonr(nlls, latent_norms)
        spearman_rho, _ = stats.spearmanr(nlls, latent_norms)
        kendall_tau, _ = stats.kendalltau(nlls, latent_norms)
        with _plot_context():
            if ax is None:
                _, ax = plt.subplots()
            ax.scatter(nlls, latent_norms, alpha=0.5)
            ax.set_xlabel("Negative Log-Likelihood")
            ax.set_ylabel("Latent Norm")
            ax.set_title("Negative Log-Likelihood vs Latent Norm")
            ax.text(
                0.6, 0.05,
                f"Pearson R: {pearson_r:.2f}\nSpearman Rho: {spearman_rho:.2f}\nKendall Tau: {kendall_tau:.2f}",
                transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.5),
            )
        return ax

    def logprob_reference_scatter_plot(self, ref_distribution, ax=None, n_samples=10000):
        """Scatter of the flow's log-probabilities against reference log-probabilities."""
        with torch.no_grad():
            samples = ref_distribution.sample((n_samples,)).to(self.device)
            ref_log_probs = ref_distribution.log_prob(samples).cpu().numpy()
            learned_log_probs = self.flow.log_prob(samples).cpu().numpy()
        pearson_r, _ = stats.pearsonr(ref_log_probs, learned_log_probs)
        spearman_rho, _ = stats.spearmanr(ref_log_probs, learned_log_probs)
        kendall_tau, _ = stats.kendalltau(ref_log_probs, learned_log_probs)
        with _plot_context():
            if ax is None:
                _, ax = plt.subplots()
            ax.scatter(ref_log_probs, learned_log_probs, alpha=0.5)
            ax.set_xlabel("Reference Log-Probability")
            ax.set_ylabel("Estimated Log-Probability")
            ax.set_title("Log-Probability Comparison")
            ax.text(
                0.6, 0.05,
                f"Pearson R: {pearson_r:.2f}\nSpearman Rho: {spearman_rho:.2f}\nKendall Tau: {kendall_tau:.2f}",
                transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.5),
            )
        return ax
