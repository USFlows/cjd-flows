"""Learnable 1-D distributions of the radius ``R = ||Z - loc||_p``.

Every family is a :class:`NormDistribution`, i.e. an ``nn.Module`` that knows
the ambient dimension ``d`` and norm exponent ``p`` and provides

* ``log_prob(r)`` / ``log_prob_log_radius(s)`` (density of ``R`` resp. ``log R``),
* ``log_profile(r)`` = ``log h(r)``, the log-density of the *d-dimensional*
  radial distribution at any point of norm ``r``,
* ``sample`` / ``rsample`` (reparameterized w.r.t. all parameters except
  mixture weights), ``cdf``, ``icdf``,
* ``profile_is_monotone(d)`` -- whether ``h`` is non-increasing, i.e. whether
  upper density level sets are balls.

Parametrization conventions
---------------------------
* All parameters are unconstrained (``log``/``softplus``/logits).
* Families are initialized relative to the *reference* law of
  :func:`~cjd_flows.radial.geometry.reference_log_radius` (``chi(d)`` for
  ``p = 2``, ``Gamma(d, 1)`` for ``p = 1``), so the default works for any
  ``d``. Location-type parameters (log-scales) are stored in *standardized*
  units ``u`` with ``location = center + s0 * u`` where ``s0 ≈ 1/sqrt(β d)`` is
  the reference standard deviation of ``log R``; one unit of ``u`` is one
  natural "radial standard deviation" regardless of ``d``.
* ``learn_scale=False`` fixes the overall scale of the radius (it is
  redundant with the global log-determinant of a flow). Mixtures keep the
  relative offsets of the components learnable but pin their mean.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cjd_flows.core import ShapeLike, as_shape

from ._numerics import (
    GridTable,
    bisect_increasing,
    inverse_trigamma,
    log_panel_integrals,
    make_table,
    safe_log,
    table_cdf,
    table_icdf,
)
from .geometry import log_unit_ball_volume, reference_exponent, reference_log_radius
from .splines import identity_derivative_init, rq_spline

__all__ = [
    "NormDistribution",
    "GeneralizedGammaMixture",
    "GeneralizedGamma",
    "GammaMixture",
    "WeibullMixture",
    "LogNormalMixture",
    "StudentTNorm",
    "KhinchineNorm",
    "SplineNorm",
    "LogRadiusFlow",
    "MonotoneProfileNorm",
    "fit_norm_to_radii",
]

_LOG_EPS = -28.0  # log of the tail mass ignored by support bounds (~7e-13)
_EULER_GAMMA = 0.5772156649015329


def _tensor(value, ref: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def _log_gamma_variate_bounds(k: Tensor) -> Tuple[Tensor, Tensor]:
    """Conservative bounds on ``log G`` for ``G ~ Gamma(k, 1)`` (tail mass ≲ 1e-12)."""
    k = k.to(torch.float64)
    lo_tail = (_LOG_EPS + torch.lgamma(k + 1)) / k  # P(G < x) <= x^k / Γ(k+1)
    lo_clt = torch.special.digamma(k) - 25 * torch.sqrt(torch.special.polygamma(1, k))
    lo = torch.maximum(lo_tail, lo_clt)
    hi = torch.log(k + torch.sqrt(60 * k) + 60)
    return lo, hi


class NormDistribution(nn.Module):
    """Base class for distributions of the radius ``R`` on ``(0, ∞)``.

    Subclasses implement :meth:`log_prob_log_radius`, :meth:`rsample` and
    :meth:`log_radius_bounds`; exact :meth:`cdf` / :meth:`icdf` /
    :meth:`log_profile_log_radius` overrides are optional (the defaults use a
    tabulated density on the log-radius axis and bisection).

    Args:
        d: ambient dimension of the radial distribution.
        p: exponent of the Lp norm (``p > 0`` or ``inf``).
    """

    #: Number of nodes of the default log-radius quadrature grid.
    grid_size: int = 4096

    def __init__(self, d: int, p: float = 2.0):
        super().__init__()
        if int(d) < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        p = float(p)
        if not p > 0:
            raise ValueError(f"p must be positive, got {p}")
        self.d = int(d)
        self.p = p
        self.ref_exponent = reference_exponent(p)
        self.ref_loc, self.ref_scale = reference_log_radius(self.d, p)
        self.log_dV = math.log(self.d) + log_unit_ball_volume(self.d, p)
        self.register_buffer("_ref", torch.zeros(()), persistent=False)

    # ----------------------------------------------------------- utilities
    def _param(self, name: str, value: Tensor, learnable: bool) -> None:
        value = value.to(self._ref.dtype).clone()
        if learnable:
            setattr(self, name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def _register_locations(self, init: Sequence[float], learn_scale: bool, spread: float) -> None:
        """Registers ``K`` location parameters ``center + s0 * (u_global + u_k - mean(u))``."""
        init_t = torch.as_tensor(init, dtype=torch.float64).reshape(-1)
        k = init_t.numel()
        self._loc_center = float(init_t.mean())
        u = (init_t - self._loc_center) / self.ref_scale
        if k > 1:
            u = u + spread * torch.linspace(-1.0, 1.0, k, dtype=torch.float64)
        self._param("loc_offsets", u, k > 1)
        self._param("loc_global", torch.zeros((), dtype=torch.float64), learn_scale)

    def _locations(self) -> Tensor:
        u = self.loc_offsets
        return self._loc_center + self.ref_scale * (self.loc_global + u - u.mean())

    @property
    def _dtype(self) -> torch.dtype:
        return self._ref.dtype

    @property
    def _device(self) -> torch.device:
        return self._ref.device

    # ------------------------------------------------------------ density
    def log_prob_log_radius(self, s: Tensor) -> Tensor:
        """Log-density of ``S = log R`` at ``s``."""
        raise NotImplementedError

    def log_prob(self, r: Tensor) -> Tensor:
        """Log-density of ``R`` at ``r > 0`` (any shape)."""
        s = safe_log(_tensor(r, self._ref))
        return self.log_prob_log_radius(s) - s

    def log_profile_log_radius(self, s: Tensor, d: Optional[int] = None) -> Tensor:
        """``log h(e^s)`` -- log-density of the radial distribution at norm ``e^s``.

        ``h(r) = p_R(r) / (d V_p r^{d-1})``. ``d`` defaults to ``self.d``.
        """
        if d is None or d == self.d:
            return self.log_prob_log_radius(s) - self.d * s - self.log_dV
        log_dv = math.log(d) + log_unit_ball_volume(d, self.p)
        return self.log_prob_log_radius(s) - d * s - log_dv

    def log_profile(self, r: Tensor) -> Tensor:
        """``log h(r)`` -- the log-density of the d-dimensional radial distribution at norm ``r``."""
        return self.log_profile_log_radius(safe_log(_tensor(r, self._ref)))

    def log_prob_volume(self, t: Tensor) -> Tensor:
        """Log-density of the volume coordinate ``T = V_p R^d`` (equals ``log h(r)``)."""
        t = _tensor(t, self._ref)
        s = (safe_log(t) - (self.log_dV - math.log(self.d))) / self.d
        return self.log_profile_log_radius(s)

    def forward(self, r: Tensor) -> Tensor:
        return self.log_prob(r)

    # ----------------------------------------------------------- sampling
    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        """Reparameterized samples of shape ``sample_shape``."""
        q = torch.rand(as_shape(sample_shape), dtype=self._dtype, device=self._device)
        return self.icdf(q)

    def sample(self, sample_shape: ShapeLike = ()) -> Tensor:
        """Samples of shape ``sample_shape`` (no gradient)."""
        with torch.no_grad():
            return self.rsample(sample_shape)

    # ------------------------------------------------------ cdf / support
    def log_radius_bounds(self) -> Tuple[float, float]:
        """Interval ``[lo, hi]`` of ``log R`` that contains all but ~1e-12 of the mass."""
        raise NotImplementedError

    def grid_table(self, n: Optional[int] = None) -> GridTable:
        """Tabulated density of ``log R`` on a uniform grid over :meth:`log_radius_bounds`."""
        lo, hi = self.log_radius_bounds()
        s = torch.linspace(lo, hi, n or self.grid_size, dtype=self._dtype, device=self._device)
        return make_table(s, self.log_prob_log_radius(s))

    def cdf(self, r: Tensor) -> Tensor:
        """``P(R <= r)``."""
        return table_cdf(self.grid_table(), safe_log(_tensor(r, self._ref)))

    def icdf(self, q: Tensor) -> Tensor:
        """Quantile function (inverse of :meth:`cdf`)."""
        return torch.exp(table_icdf(self.grid_table(), _tensor(q, self._ref)))

    def _icdf_bisection(self, q: Tensor) -> Tensor:
        q = _tensor(q, self._ref)
        lo, hi = self.log_radius_bounds()
        s = bisect_increasing(lambda x: self.cdf(torch.exp(x)), q, lo - 1.0, hi + 1.0, n_iter=80)
        return torch.exp(s)

    # ------------------------------------------------------- monotonicity
    def _numeric_profile_monotone(self, d: int, n: int = 4096) -> bool:
        with torch.no_grad():
            lo, hi = self.log_radius_bounds()
            s = torch.linspace(lo, hi, n, dtype=torch.float64, device=self._device)
            lh = self.log_profile_log_radius(s.to(self._dtype), d).to(torch.float64)
            diff = lh[1:] - lh[:-1]
            tol = 1e-7 * (1.0 + lh[1:].abs())
            return bool((diff <= tol).all())

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        """Whether the radial profile ``h`` (in dimension ``d``) is non-increasing.

        The default is a numerical check on the support grid.
        """
        return self._numeric_profile_monotone(d or self.d)

    def extra_repr(self) -> str:
        return f"d={self.d}, p={self.p}"


# =========================================================================
# Generalized gamma (Stacy) family
# =========================================================================
class GeneralizedGammaMixture(NormDistribution):
    """Mixture of generalized gamma (Stacy) distributions.

    Component density (shape ``a``, power ``β``, scale ``θ``)::

        p(r) = β / (θ Γ(a/β)) (r/θ)^{a-1} exp(-(r/θ)^β),   R = θ G^{1/β}, G ~ Gamma(a/β).

    Special cases: Gamma (``β = 1``), Weibull (``a = β``), chi (``a = d, β = 2,
    θ = sqrt 2``), Lp-generalized-normal radius (``a = d, β = p, θ = p^{1/p}``).
    The radial profile of a component is ``h(r) ∝ r^{a-d} exp(-(r/θ)^β)``: it is
    **monotone iff a ≤ d** and **bounded at 0 iff a ≥ d**. ``monotone=True``
    ties ``a = d`` for all components (bounded and monotone).

    Args:
        d: dimension.
        p: norm exponent.
        n_components: number of components ``K``.
        a, beta, theta: initial values (scalars or length-K). Defaults give the
            reference law (``a = d``, ``β = p`` or ``2`` for ``p = inf``,
            ``θ = β^{1/β}``).
        weights: initial mixture weights (default uniform).
        monotone: tie ``a = d``.
        learn_shape: learn ``a`` and ``β``.
        learn_scale: learn the global scale (mean of ``log θ_k``).
        learn_weights: learn the mixture weights.
        spread: initial spread of the components' ``log θ`` in reference
            standard deviations (``u_k = spread * linspace(-1, 1, K)``).
    """

    _shape_mode = "free"

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        n_components: int = 1,
        a: Optional[Union[float, Sequence[float]]] = None,
        beta: Optional[Union[float, Sequence[float]]] = None,
        theta: Optional[Union[float, Sequence[float]]] = None,
        weights: Optional[Sequence[float]] = None,
        monotone: bool = False,
        learn_shape: bool = True,
        learn_scale: bool = True,
        learn_weights: bool = True,
        spread: float = 1.0,
    ):
        super().__init__(d, p)
        k = int(n_components)
        if k < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = k
        self.monotone = bool(monotone)
        a0, b0, t0 = self._default_init()
        a_t = torch.as_tensor(a0 if a is None else a, dtype=torch.float64).expand(k).clone()
        b_t = torch.as_tensor(b0 if beta is None else beta, dtype=torch.float64).expand(k).clone()
        t_t = torch.as_tensor(t0 if theta is None else theta, dtype=torch.float64).expand(k).clone()
        if (a_t <= 0).any() or (b_t <= 0).any() or (t_t <= 0).any():
            raise ValueError("a, beta and theta must be positive")
        self._init_shape(a_t, b_t, learn_shape)
        self._register_locations(t_t.log().tolist(), learn_scale, spread)
        w = torch.full((k,), 1.0 / k, dtype=torch.float64) if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        self._param("logits", w.log(), learn_weights and k > 1)

    # -------------------------------------------------------- parameters
    def _default_init(self) -> Tuple[float, float, float]:
        b = self.ref_exponent
        return float(self.d), b, b ** (1.0 / b)

    def _init_shape(self, a: Tensor, beta: Tensor, learn: bool) -> None:
        if self.monotone:
            self.register_buffer("log_a", torch.full_like(a, math.log(self.d)).to(self._dtype))
        else:
            self._param("log_a", a.log(), learn)
        self._param("log_beta", beta.log(), learn)

    def _shape_params(self) -> Tuple[Tensor, Tensor]:
        """``(log a, log β)``, each of shape ``(K,)``."""
        return self.log_a, self.log_beta

    @property
    def log_theta(self) -> Tensor:
        """``log θ_k`` of shape ``(K,)``."""
        return self._locations()

    @property
    def a(self) -> Tensor:
        return self._shape_params()[0].exp()

    @property
    def beta(self) -> Tensor:
        return self._shape_params()[1].exp()

    @property
    def theta(self) -> Tensor:
        return self.log_theta.exp()

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.logits, -1)

    # ------------------------------------------------------------ density
    def _component_terms(self, s: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        log_a, log_b = self._shape_params()
        a, b = log_a.exp(), log_b.exp()
        x = s.unsqueeze(-1) - self.log_theta  # (..., K)
        base = torch.log_softmax(self.logits, -1) + log_b - torch.lgamma(a / b) - torch.exp(b * x)
        return base, a, x, log_a

    def log_prob_log_radius(self, s: Tensor) -> Tensor:
        base, a, x, _ = self._component_terms(s)
        return torch.logsumexp(base + a * x, dim=-1)

    def log_profile_log_radius(self, s: Tensor, d: Optional[int] = None) -> Tensor:
        # log h = log β - d log θ - lgamma(a/β) + (a - d) x - (r/θ)^β - log(d V_p): no cancellation.
        d_ = self.d if d is None else d
        log_dv = self.log_dV if d is None or d == self.d else math.log(d) + log_unit_ball_volume(d, self.p)
        base, a, x, _ = self._component_terms(s)
        comp = base + (a - d_) * x - d_ * self.log_theta
        return torch.logsumexp(comp, dim=-1) - log_dv

    def cdf(self, r: Tensor) -> Tensor:
        r = _tensor(r, self._ref)
        a, b = self.a, self.beta
        z = torch.exp(b * (safe_log(r).unsqueeze(-1) - self.log_theta))
        conc = (a / b).expand_as(z)
        return (self.weights * torch.special.gammainc(conc, z)).sum(-1).clamp(0.0, 1.0)

    def icdf(self, q: Tensor) -> Tensor:
        return self._icdf_bisection(q)

    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        shape = as_shape(sample_shape)
        n = math.prod(shape)
        a, b, lt = self.a, self.beta, self.log_theta
        if self.n_components == 1:
            idx = torch.zeros(n, dtype=torch.long, device=self._device)
        else:
            idx = torch.distributions.Categorical(logits=self.logits.detach()).sample((n,))
        conc = (a / b)[idx]
        g = torch.distributions.Gamma(conc, torch.ones_like(conc)).rsample()
        s = lt[idx] + safe_log(g) / b[idx]
        return torch.exp(s).reshape(shape)

    def log_radius_bounds(self) -> Tuple[float, float]:
        with torch.no_grad():
            a, b, lt = self.a.double(), self.beta.double(), self.log_theta.double()
            lo_g, hi_g = _log_gamma_variate_bounds(a / b)
            return float((lt + lo_g / b).min()), float((lt + hi_g / b).max())

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        d = d or self.d
        a = self.a.detach()
        if self.monotone and d == self.d:
            return True
        if bool((a <= d * (1 + 1e-9)).all()):
            return True
        if bool((a > d * (1 + 1e-9)).all()):
            return False  # h(r) -> 0 as r -> 0
        return self._numeric_profile_monotone(d)


class GeneralizedGamma(GeneralizedGammaMixture):
    """Single generalized gamma (Stacy) norm distribution ``GG(a, β, θ)``.

    The default (``a = d, β = p, θ = p^{1/p}``; ``β = 2`` for ``p = inf``) is
    the norm law of the Lp-generalized normal ``p(z) ∝ exp(-||z||_p^β / β)``,
    e.g. ``chi(d)`` for ``p = 2``. ``monotone=True`` ties ``a = d``.
    """

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        a: Optional[float] = None,
        beta: Optional[float] = None,
        theta: Optional[float] = None,
        monotone: bool = False,
        learn_shape: bool = True,
        learn_scale: bool = True,
    ):
        super().__init__(
            d, p, 1, a=a, beta=beta, theta=theta, monotone=monotone,
            learn_shape=learn_shape, learn_scale=learn_scale, learn_weights=False,
        )


class GammaMixture(GeneralizedGammaMixture):
    """Mixture of Gamma distributions (``β = 1`` fixed).

    The radial profile of ``Gamma(k, θ)`` is monotone iff ``k ≤ d`` and
    bounded iff ``k ≥ d`` (``monotone=True`` ties ``k = d``, i.e. Laplace-type
    profiles ``exp(-r/θ)``). By default the shape is moment-matched to the
    reference law of ``log R`` (which gives ``k ≈ 2d`` for ``p = 2``, a shell
    profile).
    """

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        n_components: int = 1,
        shape: Optional[Union[float, Sequence[float]]] = None,
        scale: Optional[Union[float, Sequence[float]]] = None,
        weights: Optional[Sequence[float]] = None,
        monotone: bool = False,
        learn_shape: bool = True,
        learn_scale: bool = True,
        learn_weights: bool = True,
        spread: float = 1.0,
    ):
        super().__init__(
            d, p, n_components, a=shape, beta=1.0, theta=scale, weights=weights, monotone=monotone,
            learn_shape=learn_shape, learn_scale=learn_scale, learn_weights=learn_weights, spread=spread,
        )

    def _default_init(self) -> Tuple[float, float, float]:
        if self.monotone:
            k = float(self.d)
        else:
            k = inverse_trigamma(self.ref_scale**2)
        log_theta = self.ref_loc - float(torch.special.digamma(torch.tensor(k, dtype=torch.float64)))
        return k, 1.0, math.exp(log_theta)

    def _init_shape(self, a: Tensor, beta: Tensor, learn: bool) -> None:
        if self.monotone:
            self.register_buffer("log_a", torch.full_like(a, math.log(self.d)).to(self._dtype))
        else:
            self._param("log_a", a.log(), learn)
        self.register_buffer("log_beta", torch.zeros_like(a).to(self._dtype))


class WeibullMixture(GeneralizedGammaMixture):
    """Mixture of Weibull distributions (``a = β = k``).

    The profile ``h ∝ r^{k-d} exp(-(r/λ)^k)`` is monotone iff ``k ≤ d``;
    ``monotone=True`` parametrizes ``k = d · sigmoid(raw)``. The default shape
    matches the reference standard deviation of ``log R``
    (``k = π / (sqrt 6 s0)``).
    """

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        n_components: int = 1,
        shape: Optional[Union[float, Sequence[float]]] = None,
        scale: Optional[Union[float, Sequence[float]]] = None,
        weights: Optional[Sequence[float]] = None,
        monotone: bool = False,
        learn_shape: bool = True,
        learn_scale: bool = True,
        learn_weights: bool = True,
        spread: float = 1.0,
    ):
        super().__init__(
            d, p, n_components, a=shape, beta=shape, theta=scale, weights=weights, monotone=monotone,
            learn_shape=learn_shape, learn_scale=learn_scale, learn_weights=learn_weights, spread=spread,
        )

    def _default_init(self) -> Tuple[float, float, float]:
        k = math.pi / (math.sqrt(6.0) * self.ref_scale)
        if self.monotone:
            k = min(k, 0.5 * self.d)
        log_lam = self.ref_loc + _EULER_GAMMA / k
        return k, k, math.exp(log_lam)

    def _init_shape(self, a: Tensor, beta: Tensor, learn: bool) -> None:
        if self.monotone:
            frac = (a / self.d).clamp(1e-6, 1 - 1e-6)
            self._param("shape_raw", torch.log(frac) - torch.log1p(-frac), learn)
        else:
            self._param("shape_raw", a.log(), learn)

    def _shape_params(self) -> Tuple[Tensor, Tensor]:
        if self.monotone:
            log_k = math.log(self.d) + F.logsigmoid(self.shape_raw)
        else:
            log_k = self.shape_raw
        return log_k, log_k

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        return bool((self.a.detach() <= (d or self.d) * (1 + 1e-9)).all()) or super().profile_is_monotone(d)


# =========================================================================
# Log-normal mixture
# =========================================================================
class LogNormalMixture(NormDistribution):
    """Mixture of log-normal distributions, ``log R ~ Σ π_k N(μ_k, σ_k²)``.

    **Never monotone**: ``p_R`` vanishes faster than any power of ``r`` at 0,
    so ``h(r) -> 0`` as ``r -> 0`` and the profile is a "shell". Upper density
    level sets are then annuli, not balls. Default: one component matching
    the reference mean and standard deviation of ``log R``.

    Args:
        d, p: dimension and norm exponent.
        n_components: number of components.
        loc: initial ``μ_k`` (absolute, in log-radius units).
        scale: initial ``σ_k``.
        weights: initial mixture weights.
        learn_scale: learn the global location (mean of ``μ_k``).
        learn_shape: learn ``σ_k``.
        learn_weights: learn the mixture weights.
        spread: initial spread of ``μ_k`` in reference standard deviations.
    """

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        n_components: int = 1,
        loc: Optional[Union[float, Sequence[float]]] = None,
        scale: Optional[Union[float, Sequence[float]]] = None,
        weights: Optional[Sequence[float]] = None,
        learn_scale: bool = True,
        learn_shape: bool = True,
        learn_weights: bool = True,
        spread: float = 1.0,
    ):
        super().__init__(d, p)
        k = int(n_components)
        self.n_components = k
        mu = torch.as_tensor(self.ref_loc if loc is None else loc, dtype=torch.float64).expand(k).clone()
        sig = torch.as_tensor(self.ref_scale if scale is None else scale, dtype=torch.float64).expand(k).clone()
        self._register_locations(mu.tolist(), learn_scale, spread)
        self._param("log_sigma", sig.log(), learn_shape)
        w = torch.full((k,), 1.0 / k, dtype=torch.float64) if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        self._param("logits", w.log(), learn_weights and k > 1)

    @property
    def mu(self) -> Tensor:
        return self._locations()

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.logits, -1)

    def log_prob_log_radius(self, s: Tensor) -> Tensor:
        z = (s.unsqueeze(-1) - self.mu) / self.sigma
        comp = torch.log_softmax(self.logits, -1) - 0.5 * z**2 - self.log_sigma - 0.5 * math.log(2 * math.pi)
        return torch.logsumexp(comp, -1)

    def cdf(self, r: Tensor) -> Tensor:
        s = safe_log(_tensor(r, self._ref))
        z = (s.unsqueeze(-1) - self.mu) / self.sigma
        return (self.weights * torch.special.ndtr(z)).sum(-1)

    def icdf(self, q: Tensor) -> Tensor:
        if self.n_components == 1:
            q = _tensor(q, self._ref)
            return torch.exp(self.mu[0] + self.sigma[0] * torch.special.ndtri(q))
        return self._icdf_bisection(q)

    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        shape = as_shape(sample_shape)
        n = math.prod(shape)
        if self.n_components == 1:
            idx = torch.zeros(n, dtype=torch.long, device=self._device)
        else:
            idx = torch.distributions.Categorical(logits=self.logits.detach()).sample((n,))
        eps = torch.randn(n, dtype=self._dtype, device=self._device)
        return torch.exp(self.mu[idx] + self.sigma[idx] * eps).reshape(shape)

    def log_radius_bounds(self) -> Tuple[float, float]:
        with torch.no_grad():
            mu, sig = self.mu, self.sigma
            return float((mu - 9 * sig).min()), float((mu + 9 * sig).max())

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        return False


# =========================================================================
# Student-t type radial law
# =========================================================================
class StudentTNorm(NormDistribution):
    """Norm law of the Lp-radial multivariate-t analogue (heavy tails, monotone).

    Profile ``h(r) ∝ (1 + (r/θ)^β / ν)^{-(ν + d)/β}`` with ``β = p`` (``2`` for
    ``p = inf``). With ``X = (R/θ)^β / ν`` one gets
    ``X ~ BetaPrime(d/β, ν/β)``, hence the exact normalizer::

        log p_R(r) = α log x - (α + β') log(1 + x) - log B(α, β') + log β - log r,
        α = d/β,  β' = ν/β.

    Sampling: ``X = G1 / G2`` with ``G1 ~ Gamma(α)``, ``G2 ~ Gamma(β')``
    (reparameterized). ``ν -> ∞`` recovers the reference-type profile
    ``exp(-(r/θ)^β / β)``. The default scale matches the reference mean of
    ``log R``.

    Args:
        d, p: dimension and norm exponent.
        df: degrees of freedom ``ν`` (initial value).
        theta: initial scale (default: match the reference mean of ``log R``).
        learn_df: learn ``ν``.
        learn_scale: learn ``θ``.
    """

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        df: float = 5.0,
        theta: Optional[float] = None,
        learn_df: bool = True,
        learn_scale: bool = True,
    ):
        super().__init__(d, p)
        self.beta = self.ref_exponent
        if theta is None:
            alpha, bp = d / self.beta, df / self.beta
            dg = torch.special.digamma(torch.tensor([alpha, bp], dtype=torch.float64))
            log_theta = self.ref_loc - (math.log(df) + float(dg[0] - dg[1])) / self.beta
        else:
            log_theta = math.log(theta)
        self._register_locations([log_theta], learn_scale, 0.0)
        self._param("log_df", torch.tensor(math.log(df), dtype=torch.float64), learn_df)

    @property
    def df(self) -> Tensor:
        return self.log_df.exp()

    @property
    def log_theta(self) -> Tensor:
        return self._locations()[0]

    def _terms(self, s: Tensor):
        nu = self.df
        alpha = self.d / self.beta
        bp = nu / self.beta
        log_x = self.beta * (s - self.log_theta) - self.log_df
        lbeta = math.lgamma(alpha) + torch.lgamma(bp) - torch.lgamma(alpha + bp)
        return alpha, bp, log_x, lbeta

    def log_prob_log_radius(self, s: Tensor) -> Tensor:
        alpha, bp, log_x, lbeta = self._terms(s)
        return alpha * log_x - (alpha + bp) * F.softplus(log_x) - lbeta + math.log(self.beta)

    def log_profile_log_radius(self, s: Tensor, d: Optional[int] = None) -> Tensor:
        if d is not None and d != self.d:
            return super().log_profile_log_radius(s, d)
        alpha, bp, log_x, lbeta = self._terms(s)
        const = -self.d * self.log_theta - alpha * self.log_df - lbeta + math.log(self.beta) - self.log_dV
        return const - (alpha + bp) * F.softplus(log_x)

    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        shape = as_shape(sample_shape)
        alpha = torch.full(shape, self.d / self.beta, dtype=self._dtype, device=self._device)
        bp = (self.df / self.beta).expand(shape)
        g1 = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).rsample()
        g2 = torch.distributions.Gamma(bp, torch.ones_like(bp)).rsample()
        log_x = safe_log(g1) - safe_log(g2)
        return torch.exp(self.log_theta + (self.log_df + log_x) / self.beta)

    def log_radius_bounds(self) -> Tuple[float, float]:
        with torch.no_grad():
            k = torch.tensor([self.d / self.beta, float(self.df) / self.beta], dtype=torch.float64)
            lo, hi = _log_gamma_variate_bounds(k)
            lx_lo, lx_hi = float(lo[0] - hi[1]), float(hi[0] - lo[1])
            lt, ldf = float(self.log_theta), float(self.log_df)
            return lt + (ldf + lx_lo) / self.beta, lt + (ldf + lx_hi) / self.beta

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        return True


# =========================================================================
# Khinchine mixtures of uniform balls
# =========================================================================
class KhinchineNorm(NormDistribution):
    """Radially decreasing law via Khinchine's representation ``R = C · U^{1/d}``.

    Every radially decreasing density is a mixture of uniform distributions on
    Lp balls; here the ball radius ``C`` follows an arbitrary learnable
    :class:`NormDistribution` ``mixing`` and ``U ~ Unif(0, 1)``. The profile
    ``h(r) = V_p^{-1} ∫_r^∞ c^{-d} m_C(c) dc`` is **monotone by construction**.

    On the log axis ``log R = log C - E/d`` with ``E ~ Exp(1)``, so::

        p_{log R}(s) = d ∫_s^∞ exp(-d (σ - s)) q(σ) dσ,   q = density of log C.

    The integral is evaluated with the exponential-linear rule (``log q``
    interpolated linearly, exponential kernel integrated exactly; all in
    log space) on a uniform grid of ``n_grid`` nodes over the support of
    ``log C``. Since ``exp(-d (σ - s)) = e^{d s} e^{-d σ}``, the full-panel
    integrals are shared by all queries (suffix ``logcumsumexp``); only the
    partial panel ``[s, σ_j]`` depends on the query, so the cost is
    ``O(n_grid + batch)``. Differentiable w.r.t. ``mixing``'s parameters and
    the query; the error is second order in the grid spacing (≈1e-5 in
    ``log p`` with the default grid). The CDF has the closed form
    ``F_R(r) = F_C(r) + p_{log R}(log r) / d``.

    Default mixing: ``GG(a = d + β, β, β^{1/β})``, for which ``R`` follows
    exactly the reference law (``chi(d)`` for ``p = 2``).

    Args:
        d, p: dimension and norm exponent.
        mixing: distribution of the ball radius ``C`` (its ``d`` is irrelevant).
        n_grid: number of quadrature nodes over the support of ``log C``.
    """

    def __init__(self, d: int, p: float = 2.0, mixing: Optional[NormDistribution] = None, n_grid: int = 8192):
        super().__init__(d, p)
        if mixing is None:
            b = self.ref_exponent
            mixing = GeneralizedGamma(d, p, a=d + b, beta=b, theta=b ** (1.0 / b))
        self.mixing = mixing
        self.n_grid = int(n_grid)

    def _grid(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Quadrature nodes: ``n_grid`` over the support of ``log C`` plus ``n_grid/4``
        below it, covering the extra spread ``E/d`` of ``log R`` (the integrand
        ``q(σ) e^{-dσ}`` need not decay to the left of the support of ``log C``)."""
        lo, hi = self.mixing.log_radius_bounds()
        lo_ext = lo + (_LOG_EPS - 2.0) / self.d
        n_ext = max(self.n_grid // 4, 2)
        left = torch.linspace(lo_ext, lo, n_ext + 1, dtype=dtype, device=device)[:-1]
        main = torch.linspace(lo, hi, self.n_grid, dtype=dtype, device=device)
        return torch.cat([left, main])

    def log_prob_log_radius(self, s: Tensor) -> Tensor:
        # p_{log R}(s) = d e^{d (s - c)} ∫_s^∞ g(σ) dσ with g(σ) = q(σ) e^{-d (σ - c)} (c: grid centre).
        # Panel integrals of g over the fixed grid are query independent -> suffix sums; only the
        # partial panel [s, σ_j] depends on the query. Cost O(N + B).
        d = self.d
        shape = s.shape
        flat = s.reshape(-1)
        grid = self._grid(s.dtype, s.device)
        lo, hi = float(grid[0]), float(grid[-1])
        c = 0.5 * (lo + hi)
        g_grid = self.mixing.log_prob_log_radius(grid) - d * (grid - c)
        panels = log_panel_integrals(grid, g_grid)  # (N-1,)
        tail = g_grid[-1:] - math.log(d)  # ∫ beyond the grid, assuming q flat there
        # suffix[j] = log(Σ_{i >= j} panel_i + tail), j = 0..N-1
        suffix = torch.logcumsumexp(torch.cat([panels, tail]).flip(0), 0).flip(0)

        g_s = self.mixing.log_prob_log_radius(flat) - d * (flat - c)
        j = torch.searchsorted(grid.contiguous(), flat.detach().contiguous(), right=True)  # first σ_j > s
        n = grid.shape[0]
        inside = j < n
        jc = j.clamp(max=n - 1)
        nodes = torch.stack([flat, torch.maximum(grid[jc], flat)], dim=-1)
        partial = log_panel_integrals(nodes, torch.stack([g_s, g_grid[jc]], dim=-1)).squeeze(-1)
        total_inside = torch.logaddexp(partial, suffix[jc])
        total = torch.where(inside, total_inside, g_s - math.log(d))
        return (math.log(d) + d * (flat - c) + total).reshape(shape)

    def cdf(self, r: Tensor) -> Tensor:
        r = _tensor(r, self._ref)
        s = safe_log(r)
        return (self.mixing.cdf(r) + torch.exp(self.log_prob_log_radius(s) - math.log(self.d))).clamp(0.0, 1.0)

    def icdf(self, q: Tensor) -> Tensor:
        return self._icdf_bisection(q)

    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        shape = as_shape(sample_shape)
        c = self.mixing.rsample(shape)
        u = torch.rand(shape, dtype=self._dtype, device=self._device)
        return c * torch.exp(safe_log(u) / self.d)

    def log_radius_bounds(self) -> Tuple[float, float]:
        lo, hi = self.mixing.log_radius_bounds()
        return lo + (_LOG_EPS - 2.0) / self.d, hi

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        if d is None or d == self.d:
            return True
        return self._numeric_profile_monotone(d)


# =========================================================================
# Spline flow on the standardized log-radius
# =========================================================================
class SplineNorm(NormDistribution):
    """Normalizing flow on the standardized log-radius.

    ``u = (log r - μ0) / s0`` (reference mean/std of ``log R``), then a
    learnable affine map ``v = (u - m) e^{-ℓ}`` and a monotone
    rational-quadratic spline ``y = g(v)`` on ``[-B, B]`` (identity outside)
    map to a standard normal (``base="normal"``) or logistic
    (``base="logistic"``) variable. Density by change of variables, sampling
    by the inverse spline. Initialized as the identity, i.e. ``log R ~
    N(μ0, s0²)`` (the log-normal approximation of the reference law).

    Monotonicity: with a normal base the left tail is log-normal, hence the
    profile is never monotone. With a logistic base ``p_R(r) ∝ r^{κ-1}`` near 0
    with ``κ = e^{-ℓ}/s0``; the profile is then monotone iff ``κ ≤ d`` and the
    profile is non-increasing on the support grid (checked numerically).

    Args:
        d, p: dimension and norm exponent.
        num_bins: number of spline bins.
        bound: half-width ``B`` of the spline domain (in standardized units).
        base: ``"normal"`` or ``"logistic"``.
        learn_scale: learn the affine map (``m``, ``ℓ``).
    """

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        num_bins: int = 8,
        bound: float = 5.0,
        base: str = "normal",
        learn_scale: bool = True,
    ):
        super().__init__(d, p)
        if base not in ("normal", "logistic"):
            raise ValueError("base must be 'normal' or 'logistic'")
        self.base = base
        self.bound = float(bound)
        self.num_bins = int(num_bins)
        k = self.num_bins
        self.unnorm_widths = nn.Parameter(torch.zeros(k))
        self.unnorm_heights = nn.Parameter(torch.zeros(k))
        self.unnorm_derivatives = nn.Parameter(torch.full((k - 1,), identity_derivative_init()))
        self._param("shift", torch.zeros(()), learn_scale)
        self._param("log_scale", torch.zeros(()), learn_scale)

    def _spline(self, v: Tensor, inverse: bool = False) -> Tuple[Tensor, Tensor]:
        return rq_spline(
            v, self.unnorm_widths, self.unnorm_heights, self.unnorm_derivatives, self.bound, inverse=inverse
        )

    def _base_log_prob(self, y: Tensor) -> Tensor:
        if self.base == "normal":
            return -0.5 * y**2 - 0.5 * math.log(2 * math.pi)
        return -F.softplus(-y) - F.softplus(y)

    def _base_cdf(self, y: Tensor) -> Tensor:
        return torch.special.ndtr(y) if self.base == "normal" else torch.sigmoid(y)

    def _base_icdf(self, q: Tensor) -> Tensor:
        return torch.special.ndtri(q) if self.base == "normal" else torch.logit(q)

    def _to_y(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        u = (s - self.ref_loc) / self.ref_scale
        v = (u - self.shift) * torch.exp(-self.log_scale)
        y, lad = self._spline(v)
        return y, lad - self.log_scale - math.log(self.ref_scale)

    def _from_y(self, y: Tensor) -> Tensor:
        v, _ = self._spline(y, inverse=True)
        u = self.shift + torch.exp(self.log_scale) * v
        return self.ref_loc + self.ref_scale * u

    def log_prob_log_radius(self, s: Tensor) -> Tensor:
        y, lad = self._to_y(s)
        return self._base_log_prob(y) + lad

    def cdf(self, r: Tensor) -> Tensor:
        y, _ = self._to_y(safe_log(_tensor(r, self._ref)))
        return self._base_cdf(y)

    def icdf(self, q: Tensor) -> Tensor:
        return torch.exp(self._from_y(self._base_icdf(_tensor(q, self._ref))))

    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        shape = as_shape(sample_shape)
        if self.base == "normal":
            y = torch.randn(shape, dtype=self._dtype, device=self._device)
        else:
            u = torch.rand(shape, dtype=self._dtype, device=self._device).clamp(1e-12, 1 - 1e-12)
            y = torch.logit(u)
        return torch.exp(self._from_y(y))

    def log_radius_bounds(self) -> Tuple[float, float]:
        with torch.no_grad():
            y_max = 7.5 if self.base == "normal" else 28.0
            s = self._from_y(torch.tensor([-y_max, y_max], dtype=self._dtype, device=self._device))
            return float(s[0]), float(s[1])

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        if self.base == "normal":
            return False
        kappa = float(torch.exp(-self.log_scale.detach())) / self.ref_scale
        if kappa > (d or self.d) * (1 + 1e-9):
            return False
        return self._numeric_profile_monotone(d or self.d)


#: Alias: a 1-D flow on the standardized log-radius.
LogRadiusFlow = SplineNorm


# =========================================================================
# Directly parametrized monotone profile
# =========================================================================
class MonotoneProfileNorm(NormDistribution):
    """Radial law with a directly parametrized, non-increasing profile.

    ``-log h(r) = φ(r) + const`` with the increasing function::

        φ(r) = (r/θ)^β + κ Σ_k softplus(α_k) w softplus((u - c_k) / w),
        u = (log r - μ0) / s0,

    i.e. a generalized-normal baseline plus non-negative combinations of
    softplus ramps in the standardized log-radius (``κ = d s0`` is the natural
    slope of ``φ`` near the typical radius). The ramps start at ``α_k = -20``
    so the initial law is the reference one. The normalizer
    ``∫ exp(-φ(r)) r^{d-1} dr`` is computed by the exponential-linear rule
    on a grid in ``log r`` that is adapted (without gradient) to the region
    holding the mass; sampling inverts the tabulated CDF (differentiable).

    Args:
        d, p: dimension and norm exponent.
        n_ramps: number of softplus ramps ``K``.
        ramp_range: range of the ramp centres ``c_k`` in standardized units.
        n_grid: number of quadrature nodes.
        beta, theta: initial baseline power and scale (default: reference).
        learn_scale: learn ``θ``.
        learn_shape: learn ``β``.
    """

    def __init__(
        self,
        d: int,
        p: float = 2.0,
        n_ramps: int = 16,
        ramp_range: Tuple[float, float] = (-6.0, 6.0),
        n_grid: int = 4096,
        beta: Optional[float] = None,
        theta: Optional[float] = None,
        learn_scale: bool = True,
        learn_shape: bool = True,
    ):
        super().__init__(d, p)
        b = self.ref_exponent if beta is None else float(beta)
        t = b ** (1.0 / b) if theta is None else float(theta)
        # baseline exp(-(r/θ)^β); the reference profile is exp(-r^β/β) i.e. θ = β^{1/β}
        self._param("log_beta", torch.tensor(math.log(b), dtype=torch.float64), learn_shape)
        self._register_locations([math.log(t)], learn_scale, 0.0)
        self.n_grid = int(n_grid)
        centers = torch.linspace(ramp_range[0], ramp_range[1], n_ramps, dtype=torch.float64)
        self.register_buffer("ramp_centers", centers.to(self._dtype))
        self.ramp_width = float(centers[1] - centers[0]) if n_ramps > 1 else 1.0
        self.ramp_slope = self.d * self.ref_scale
        self.ramp_logits = nn.Parameter(torch.full((n_ramps,), -20.0))

    @property
    def log_theta(self) -> Tensor:
        return self._locations()[0]

    def phi(self, s: Tensor) -> Tensor:
        """``φ(e^s)`` (increasing in ``s``)."""
        beta = self.log_beta.exp()
        base = torch.exp(beta * (s - self.log_theta))
        u = (s - self.ref_loc) / self.ref_scale
        w = self.ramp_width
        ramps = F.softplus((u.unsqueeze(-1) - self.ramp_centers) / w) * w
        return base + self.ramp_slope * (ramps * F.softplus(self.ramp_logits)).sum(-1)

    def _support(self) -> Tuple[float, float]:
        with torch.no_grad():
            beta = float(self.log_beta.exp())
            lt = float(self.log_theta)
            center = lt + math.log(self.d / beta) / beta
            sd = 1.0 / math.sqrt(beta * self.d)
            s = torch.linspace(center - 50 * sd - 3.0, center + 15 * sd, 8192, dtype=torch.float64, device=self._device)
            ell = (-self.phi(s.to(self._dtype)) + self.d * s.to(self._dtype)).double()
            keep = ell >= ell.max() - 60.0
            idx = torch.nonzero(keep).flatten()
            step = float(s[1] - s[0])
            return float(s[idx[0]]) - 2 * step, float(s[idx[-1]]) + 2 * step

    def grid_table(self, n: Optional[int] = None) -> GridTable:
        lo, hi = self._support()
        s = torch.linspace(lo, hi, n or self.n_grid, dtype=self._dtype, device=self._device)
        return make_table(s, -self.phi(s) + self.d * s)

    def log_normalizer(self) -> Tensor:
        """``log ∫ exp(-φ(r)) r^{d-1} dr``."""
        return self.grid_table().log_total

    def log_prob_log_radius(self, s: Tensor) -> Tensor:
        return -self.phi(s) + self.d * s - self.log_normalizer()

    def log_profile_log_radius(self, s: Tensor, d: Optional[int] = None) -> Tensor:
        if d is not None and d != self.d:
            return super().log_profile_log_radius(s, d)
        return -self.phi(s) - self.log_normalizer() - self.log_dV

    def cdf(self, r: Tensor) -> Tensor:
        return table_cdf(self.grid_table(), safe_log(_tensor(r, self._ref)))

    def log_radius_bounds(self) -> Tuple[float, float]:
        return self._support()

    def profile_is_monotone(self, d: Optional[int] = None) -> bool:
        if d is None or d == self.d:
            return True
        return self._numeric_profile_monotone(d)


# =========================================================================
# Utilities
# =========================================================================
def fit_norm_to_radii(
    norm: NormDistribution,
    radii: Tensor,
    steps: int = 500,
    lr: float = 1e-2,
    optimizer: str = "adam",
    batch_size: Optional[int] = None,
) -> List[float]:
    """Maximum-likelihood refit of a norm distribution to observed radii.

    Typical use: after training a flow, refit the base's norm law to the
    latent norms ``||f(x) - loc||_p`` of the training data (Draxler et al.,
    2024 suggest such a post-hoc 1-D correction since it closes the
    expressivity gap of volume-preserving flows).

    Args:
        norm: the norm distribution (modified in place).
        radii: 1-D tensor of observed radii.
        steps: optimization steps (Adam steps, or LBFGS iterations).
        lr: learning rate.
        optimizer: ``"adam"`` or ``"lbfgs"`` (full batch).
        batch_size: mini-batch size for Adam (default: full batch).

    Returns:
        List of the per-step mean negative log-likelihoods.
    """
    params = [q for q in norm.parameters() if q.requires_grad]
    if not params:
        return []
    radii = radii.detach().reshape(-1).to(norm._ref)
    losses: List[float] = []
    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -norm.log_prob(radii).mean()
            loss.backward()
            losses.append(float(loss.detach()))
            return loss

        opt.step(closure)
        return losses
    if optimizer != "adam":
        raise ValueError("optimizer must be 'adam' or 'lbfgs'")
    opt = torch.optim.Adam(params, lr=lr)
    n = radii.shape[0]
    for _ in range(steps):
        batch = radii if batch_size is None or batch_size >= n else radii[torch.randint(0, n, (batch_size,), device=radii.device)]
        opt.zero_grad()
        loss = -norm.log_prob(batch).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses
