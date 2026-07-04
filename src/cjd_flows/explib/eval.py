from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2, binned_statistic
from sklearn.metrics import mutual_info_score
import scipy.stats as stats
from scipy.stats import binomtest, wilcoxon
from sklearn.neighbors import KernelDensity

from cjd_flows.distributions import RadialDistribution

class RadialFlowEvaluator:
    def __init__(self, flow, data, device='cpu', p: Optional[float] = None, norm_distribution: Optional[torch.distributions.Distribution] = None, loc: Optional[torch.Tensor] = None):
        """
        Evaluator for USFlow models with RadialDistribution base distribution.
        
        Args:
            flow: Trained USFlow model with RadialDistribution base
            data: Dataset tensor for evaluation
            device: Device for computation
        """
        self.flow = flow.to(device)
        self.data = data.to(device)
        self.device = device
        
        self.dim = torch.prod(torch.tensor(self.data.shape[1:])).item()

        if isinstance(flow.base_distribution, RadialDistribution):
            # Get p-norm from base distribution
            self.p = flow.base_distribution.p
            self.norm_distribution = flow.base_distribution.norm_distribution
        else:
            if p is None:
                raise ValueError("p-norm must be specified for non-RadialDistribution base distributions")
            if not isinstance(p, (int, float)):
                raise TypeError("p must be an integer or float")
            if p <= 0:
                raise ValueError("p must be a positive number")
            self.p = p
            self.norm_distribution = norm_distribution

        if hasattr(flow.base_distribution, 'loc'):
            self.loc = flow.base_distribution.loc.to(device)
        else:
            if loc is None:
                raise ValueError("loc must be specified for non-RadialDistribution base distributions")
            self.loc = loc.to(device)

        # Precompute latent representations
        with torch.no_grad():
            self.latents = self.flow.backward(self.data) - self.loc
            self.latents = self.latents.view(self.latents.shape[0], -1)
        

    def wasserstein_norm_distance(self, n_samples=10000):
        """
        Compute Wasserstein distance between:
        1. Norm distribution of base distribution
        2. Empirical p-norms of latent representations
        
        Returns:
            wasserstein_dist: Wasserstein distance
        """
        # Get empirical latent norms
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        
        # Sample from base norm distribution
        base_norm_dist = self.norm_distribution
        sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        # Compute Wasserstein distance
        return wasserstein_distance(latent_norms, sample_norms)
    
    def ks_norm_statistic(self, n_samples=10000):
        """
        Compute Kolmogorov-Smirnov statistic for norm distributions.
        
        Returns:
            ks_stat: KS statistic
            p_value: Associated p-value
        """
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        base_norm_dist = self.norm_distribution
        sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        return ks_2samp(latent_norms, sample_norms)
    
    def qq_plot_norms(self, ax=None, n_samples=10000):
        """
        Generate QQ-plot comparing:
        1. Quantiles of empirical latent norms
        2. Quantiles of base norm distribution samples
        """
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12
        })
        plt.style.use('ggplot')

        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        base_norm_dist = self.norm_distribution
        sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        latent_quantiles = np.quantile(latent_norms, np.linspace(0, 1, 100))
        sample_quantiles = np.quantile(sample_norms, np.linspace(0, 1, 100))
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(sample_quantiles, latent_quantiles, alpha=0.7)
        min_val = min(sample_quantiles.min(), latent_quantiles.min())
        max_val = max(sample_quantiles.max(), latent_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title('QQ-plot of Latent Norms')
        ax.set_xlabel('Base Distribution Quantiles')
        ax.set_ylabel('Data Latent Quantiles')
        return ax
    
    def kde_plot_norms(self, ax=None, n_samples=10000):
        """
        Generate KDE plots comparing:
        1. Empirical latent norms distribution
        2. Base norm distribution
        """
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12
        })
        plt.style.use('ggplot')

        with torch.no_grad():
            latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
            base_norm_dist = self.norm_distribution
            sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        # Create KDE models
        kde_latent = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(latent_norms.reshape(-1, 1))
        kde_base = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample_norms.reshape(-1, 1))
        
        # Evaluate on grid
        x_grid = np.linspace(
            min(latent_norms.min(), sample_norms.min()),
            max(latent_norms.max(), sample_norms.max()),
            1000
        )[:, np.newaxis]
        
        log_dens_latent = kde_latent.score_samples(x_grid)
        log_dens_base = kde_base.score_samples(x_grid)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_grid, np.exp(log_dens_latent), label='Data Latents')
        ax.plot(x_grid, np.exp(log_dens_base), label='Base Distribution')
        ax.set_title('KDE of Norm Distributions')
        ax.set_xlabel('Norm Value')
        ax.set_ylabel('Density')
        ax.legend()
        return ax
    
    def pp_plot_norms(self, ax=None, n_samples=10000):
        """
        Generate PP-plot comparing:
        1. Empirical CDF of latent norms
        2. Theoretical CDF of base norm distribution
        
        Args:
            ax: Matplotlib axis (optional)
            n_samples: Number of samples for theoretical distribution
            
        Returns:
            ax: Matplotlib axis
        """
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12
        })
        plt.style.use('ggplot')

        # Get empirical latent norms
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        
        # Compute empirical CDF
        n = len(latent_norms)
        empirical_cdf = np.arange(1, n+1) / n
        sorted_norms = np.sort(latent_norms)
        
        # Get theoretical CDF (if available)
        base_norm_dist = self.norm_distribution
        if hasattr(base_norm_dist, 'cdf'):
            # Use analytical CDF if available
            theoretical_cdf = base_norm_dist.cdf(
                torch.tensor(sorted_norms).to(self.device)
            ).detach().cpu().numpy()
        else:
            # Approximate via sampling
            sample_norms = base_norm_dist.sample((n_samples,)).detach().cpu().numpy()
            sample_sorted = np.sort(sample_norms)
            theoretical_cdf = np.searchsorted(sample_sorted, sorted_norms) / n_samples
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(theoretical_cdf, empirical_cdf, alpha=0.7)
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_title('PP-plot of Norm Distributions')
        ax.set_xlabel('Theoretical CDF (Base Distribution)')
        ax.set_ylabel('Empirical CDF (Data Latents)')
        ax.grid(True)
        
        return ax
    
    def binned_uniformity_test(self, n_bins=10):
        """
        Binned uniformity test for latent directions.
        Computes chi-squared statistic for binned directional data.
        
        Returns:
            chi2_stat: Chi-squared statistic
            p_value: Associated p-value
        """
        # Normalize to unit sphere
        directions = self.latents / torch.norm(self.latents, p=self.p, dim=1, keepdim=True)
        directions = directions.cpu().numpy()
        
        # Create bins in each dimension
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_indices = np.zeros(len(directions), dtype=int)
        
        # Multi-dimensional binning
        for dim in range(self.dim):
            bin_indices_dim = np.digitize(directions[:, dim], bin_edges) - 1
            bin_indices += bin_indices_dim * (n_bins ** dim)
        
        # Count bins
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        n_observed = len(unique_bins)
        
        # Expected counts (uniform distribution)
        total_bins = n_bins ** self.dim
        expected = len(directions) / total_bins
        
        # Chi-squared test
        chi2_stat = np.sum((counts - expected) ** 2 / expected)
        p_value = 1 - chi2.cdf(chi2_stat, df=n_observed - 1)
        
        return chi2_stat, p_value
    
    def hs_independence_test(self, n_permutations=1000):
        """
        Hilbert-Schmidt Independence Criterion for:
        H0: Norm and direction are independent
        
        Returns:
            hsic_value: HSIC statistic
            p_value: Estimated p-value via permutation test
        """
        # Compute norms and directions
        norms = torch.norm(self.latents, p=self.p, dim=1).unsqueeze(1)
        directions = self.latents / norms
        
        # Center and scale
        norms = (norms - norms.mean()) / norms.std()
        directions = (directions - directions.mean(dim=0)) / directions.std(dim=0)
        
        # Compute kernels
        K_n = self._rbf_kernel(norms)
        K_d = self._rbf_kernel(directions)
        
        # Center kernels
        n = len(norms)
        H = torch.eye(n) - torch.ones(n, n) / n
        K_n = H @ K_n @ H
        K_d = H @ K_d @ H
        
        # Compute HSIC
        hsic_value = torch.trace(K_n @ K_d) / (n * n)
        
        # Permutation test for p-value
        permuted_values = []
        for _ in range(n_permutations):
            perm_idx = torch.randperm(n)
            K_d_perm = K_d[perm_idx][:, perm_idx]
            permuted_values.append(torch.trace(K_n @ K_d_perm).item())
        
        permuted_values = np.array(permuted_values) / (n * n)
        p_value = (permuted_values >= hsic_value.item()).mean()
        
        return hsic_value.item(), p_value
    
    def _rbf_kernel(self, X, sigma=None):
        """Compute RBF kernel matrix"""
        n = X.shape[0]
        X_norm = torch.sum(X**2, dim=1).reshape(-1, 1)
        pairwise_dist = X_norm + X_norm.T - 2 * torch.mm(X, X.T)
        
        if sigma is None:
            sigma = torch.median(pairwise_dist[pairwise_dist > 0]).sqrt()
        
        gamma = 1.0 / (2 * sigma**2)
        K = torch.exp(-gamma * pairwise_dist)
        return K
    
    def test_uniformity_simplex(self, alpha=0.05, method='energy', n_samples_ref=1000, n_boot=1000):
        """
        Test uniformity of normalized absolute latents on the simplex.
        
        Args:
            alpha: Significance level
            method: 'energy' for energy distance test, 'bhattacharyya' for transformed residuals test
            n_samples_ref: Number of reference samples for energy distance
            n_boot: Number of bootstrap samples for p-value calculation
            
        Returns:
            p_value: Computed p-value for uniformity test
            reject: Boolean indicating rejection of uniformity
        """
        if self.p != 1:
            raise ValueError("Uniformity test requires L1 norm (p=1), current p={}".format(self.p))
            
        # Compute absolute values and normalize to simplex
        abs_latents = torch.abs(self.latents)
        row_sums = abs_latents.sum(dim=1, keepdim=True)
        valid_rows = (row_sums > 1e-8).squeeze()
        
        if valid_rows.sum() < 10:  # Ensure sufficient valid samples
            raise ValueError("Insufficient non-zero latent vectors for uniformity test")
            
        u = abs_latents[valid_rows] / row_sums[valid_rows]
        u_np = u.cpu().numpy()
        
        if method == 'energy':
            return self._energy_uniformity_test(u_np, alpha, n_samples_ref, n_boot)
        elif method == 'bhattacharyya':
            return self._bhattacharyya_uniformity_test(u_np, alpha)
        else:
            raise ValueError("Unknown method: {}".format(method))

    def _energy_uniformity_test(self, u, alpha, n_samples_ref, n_boot):
        """Energy distance test for uniformity on simplex"""
        d = u.shape[1]
        n = u.shape[0]
        
        # Generate reference uniform sample
        ref = self._simulate_uniform_simplex(n_samples_ref, d)
        
        # Compute observed energy distance
        stat_obs = self._energy_distance(u, ref)
        
        # Bootstrap distribution under null
        stat_boot = []
        for _ in range(n_boot):
            boot_sample = self._simulate_uniform_simplex(n, d)
            stat_boot.append(self._energy_distance(boot_sample, ref))
        
        # Calculate p-value
        p_value = np.mean(np.array(stat_boot) >= stat_obs)
        reject = p_value < alpha
        return p_value, reject

    def _bhattacharyya_uniformity_test(self, u, alpha):
        """Bhattacharyya transformation test for uniformity"""
        # Transform to negative logs
        y = -np.log(u)
        
        # Compute residuals (centered logs)
        residuals = y - y.mean(axis=1, keepdims=True)
        
        # Flatten residuals and test against standard Gumbel
        flat_residuals = residuals.flatten()
        ks_stat, p_value = stats.kstest(flat_residuals, 'gumbel_r')
        reject = p_value < alpha
        return p_value, reject

    def _simulate_uniform_simplex(self, n, d):
        """Generate uniform samples on simplex using exponential distribution"""
        exp_samples = np.random.exponential(scale=1.0, size=(n, d))
        row_sums = exp_samples.sum(axis=1, keepdims=True)
        return exp_samples / row_sums

    def _energy_distance(self, X, Y):
        """Compute energy distance between samples X and Y"""
        n = X.shape[0]
        m = Y.shape[0]
        
        # Compute pairwise distances
        xx = np.sum(X**2, axis=1)
        yy = np.sum(Y**2, axis=1)
        xy = np.dot(X, Y.T)
        
        d_xx = xx[:, None] + xx[None, :] - 2 * np.dot(X, X.T)
        d_yy = yy[:, None] + yy[None, :] - 2 * np.dot(Y, Y.T)
        d_xy = xx[:, None] + yy[None, :] - 2 * xy
        
        term1 = np.sum(np.sqrt(d_xy)) / (n * m)
        term2 = np.sum(np.sqrt(d_xx)) / (n * n)
        term3 = np.sum(np.sqrt(d_yy)) / (m * m)
        
        return 2 * term1 - term2 - term3

    def test_sign_symmetry(self, alpha=0.05, method='sign', combine='stouffer'):
        """
        Test sign symmetry with options for high-dimensional aggregation.
        
        Args:
            alpha: Significance level
            method: 'sign' or 'wilcoxon'
            combine: 'fisher', 'stouffer', or None for Bonferroni
            
        Returns:
            result: Dictionary containing p-values and rejection decision
        """
        if self.p != 1:
            raise ValueError("Sign symmetry test requires L1 norm (p=1), current p={}".format(self.p))
            
        p_values = []
        z_scores = []  # For Stouffer's method
        
        # Compute p-values for each dimension
        for j in range(self.latents.shape[1]):
            coord = self.latents[:, j].cpu().numpy()
            
            if method == 'sign':
                n_pos = (coord > 0).sum()
                test_result = binomtest(n_pos, len(coord), p=0.5, alternative='two-sided')
                p_val = test_result.pvalue
                z_scores.append((n_pos - len(coord)/2) / np.sqrt(len(coord)/4))
                
            elif method == 'wilcoxon':
                _, p_val = wilcoxon(coord, zero_method='wilcox', alternative='two-sided')
                # For Fisher only (Stouffer not recommended with Wilcoxon in high-d)
                z_scores.append(norm.ppf(1 - p_val/2) * np.sign(np.median(coord)))
                
            p_values.append(p_val)
        
        # Handle different combination methods
        combined_p = None
        if combine == 'fisher':
            chi2_stat = -2 * np.sum(np.log(p_values))
            df = 2 * len(p_values)
            combined_p = 1 - chi2.cdf(chi2_stat, df)
            reject = combined_p < alpha
            
        elif combine == 'stouffer':
            if method != 'sign':
                raise ValueError("Stouffer method requires sign test")
            z_combined = np.sum(z_scores) / np.sqrt(len(z_scores))
            combined_p = 2 * (1 - norm.cdf(np.abs(z_combined)))  # Two-sided
            reject = combined_p < alpha
            
        else:  # Bonferroni
            per_test_alpha = alpha / self.dim
            reject = any(p < per_test_alpha for p in p_values)
        
        return {
            'p_values': p_values,
            'reject': reject,
            'combined_p': combined_p,
            'method': f"{method} with {combine}" if combine else f"{method} with Bonferroni"
        }

    def test_l1_radial_symmetry(self, alpha=0.05, sign_method='wilcoxon', 
                               sign_combine='fisher', uniform_method='energy'):
        """
        Combined test with improved high-dimensional handling.
        
        Args:
            alpha: Overall significance level
            sign_method: 'sign' or 'wilcoxon'
            sign_combine: 'fisher', 'stouffer', or None
            uniform_method: 'energy' or 'bhattacharyya'
            
        Returns:
            result: Dictionary with test outcomes
        """
        if self.p != 1:
            raise ValueError("L1-radial test requires p=1, current p={}".format(self.p))
            
        # Test sign symmetry with alpha/2
        sign_result = self.test_sign_symmetry(
            alpha=alpha/2, 
            method=sign_method, 
            combine=sign_combine
        )
        
        # Test uniformity with alpha/2
        uniformity_pval, uniformity_reject = self.test_uniformity_simplex(
            alpha=alpha/2, method=uniform_method
        )
        
        # Combine results
        l1_radial_rejected = sign_result['reject'] or uniformity_reject
        
        return {
            'sign_pvals': sign_result['p_values'],
            'sign_reject': sign_result['reject'],
            'sign_combined_p': sign_result['combined_p'],
            'sign_method': sign_result['method'],
            'uniformity_pval': uniformity_pval,
            'uniformity_reject': uniformity_reject,
            'l1_radial_rejected': l1_radial_rejected
        }

    def nll_norm_scatter_plot(self, ref_distribution, ax=None, n_samples=10000):
        """
        Scatter plot of log-probabilities of latent norms vs base distribution.
        
        Args:
            ref_distribution: Reference distribution for nll computation
            ax: Matplotlib axis (optional)
            n_samples: Number of samples for base distribution
        """
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12
        })
        plt.style.use('ggplot')

        if ax is None:
            fig, ax = plt.subplots()

        # Sample from the reference distribution
        base_samples = ref_distribution.sample((n_samples,)).to(self.device)
        base_samples = base_samples.view(base_samples.shape[0], -1)

        # Compute log-probabilities
        with torch.no_grad():
            nlls = -ref_distribution.log_prob(base_samples).cpu().numpy()
            latent_norms = (self.flow.backward(base_samples) - self.loc).norm(p=self.p, dim=1).cpu().numpy()

        # Compute Pearson correlation
        pearson_r, _ = stats.pearsonr(nlls, latent_norms)
        spearman_rho, _ = stats.spearmanr(nlls, latent_norms)
        kendall_tau, _ = stats.kendalltau(nlls, latent_norms)



        # Scatter plot
        ax.scatter(nlls, latent_norms, alpha=0.5)
        ax.set_xlabel("Negative Log-Likelihood")
        ax.set_ylabel("Latent Norm")
        ax.set_title("Negative Log-Likelihood vs Latent Norm")

        ax.text(0.6, 0.05, f"Pearson R: {pearson_r:.2f}\nSpearman Rho: {spearman_rho:.2f}\nKendall Tau: {kendall_tau:.2f}", 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

        return ax
    
    def logprob_reference_scatter_plot(self, ref_distribution, ax=None, n_samples=10000):
        """
        Scatter estimated log-probs against reference distribution log-probs.
        Args:
            ref_distribution: Reference distribution for log-prob computation
            ax: Matplotlib axis (optional)
            n_samples: Number of samples for base distribution
        """
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12
        })
        plt.style.use('ggplot')

        if ax is None:
            fig, ax = plt.subplots()

        # Sample from the reference distribution
        base_samples = ref_distribution.sample((n_samples,)).to(self.device)

        # Compute log-probabilities
        with torch.no_grad():
            ref_log_probs = ref_distribution.log_prob(base_samples).cpu().numpy()
            learned_log_probs = self.flow.log_prob(base_samples).cpu().numpy()

        pearson_r, _ = stats.pearsonr(ref_log_probs, learned_log_probs)
        spearman_rho, _ = stats.spearmanr(ref_log_probs, learned_log_probs)
        kendall_tau, _ = stats.kendalltau(ref_log_probs, learned_log_probs)

        # Scatter plot
        ax.scatter(ref_log_probs, learned_log_probs, alpha=0.5)
        ax.set_xlabel("Reference Log-Probability")
        ax.set_ylabel("Estimated Log-Probability")
        ax.set_title("Log-Probability Comparison")
        ax.text(0.6, 0.05, f"Pearson R: {pearson_r:.2f}\nSpearman Rho: {spearman_rho:.2f}\nKendall Tau: {kendall_tau:.2f}",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

        return ax