"""Lp-radial base distributions and multi-centre radial mixtures."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from cjd_flows.core import BaseDistribution, ShapeLike, as_shape

from .geometry import lp_norm, sample_unit_sphere
from .norms import GeneralizedGamma, NormDistribution

__all__ = ["RadialBase", "RadialMixture"]


def _check_norm(norm: NormDistribution, d: int, p: float) -> None:
    if not isinstance(norm, NormDistribution):
        raise TypeError(f"norm must be a NormDistribution, got {type(norm)}")
    if norm.d != d:
        raise ValueError(f"norm distribution was built for d={norm.d}, but the event has d={d}")
    if norm.p != p:
        raise ValueError(f"norm distribution was built for p={norm.p}, but the base uses p={p}")


class RadialBase(BaseDistribution):
    """Lp-radial distribution ``p(z) = h(||z - loc||_p)`` with a learnable norm law.

    ``Z = loc + R · U`` with ``R ~ norm`` and ``U`` uniform on the Lp unit
    sphere w.r.t. the cone measure; ``h(r) = p_R(r) / (d V_p r^{d-1})``.

    Args:
        event_shape: event shape, ``d = prod(event_shape)``.
        p: norm exponent (``p > 0`` or ``inf``).
        norm: distribution of ``R`` (built for the same ``d`` and ``p``).
            Default: learnable :class:`GeneralizedGamma` initialized at the
            reference law (standard normal for ``p = 2``).
        loc: centre (default zeros).
        learn_loc: make the centre a parameter.
    """

    def __init__(
        self,
        event_shape: ShapeLike,
        p: float = 2.0,
        norm: Optional[NormDistribution] = None,
        loc: Optional[Tensor] = None,
        learn_loc: bool = False,
    ):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        self.p = float(p)
        d = self.dim
        self.norm = GeneralizedGamma(d, self.p) if norm is None else norm
        _check_norm(self.norm, d, self.p)
        loc_t = torch.zeros(self.event_shape) if loc is None else torch.as_tensor(loc, dtype=torch.get_default_dtype()).reshape(self.event_shape)
        if learn_loc:
            self.loc = nn.Parameter(loc_t.clone())
        else:
            self.register_buffer("loc", loc_t.clone())

    # ------------------------------------------------------- constructors
    @classmethod
    def gaussian(cls, event_shape: ShapeLike, learnable: bool = False) -> RadialBase:
        """Standard normal as ``chi(d)`` norm with ``p = 2`` (exact)."""
        d = math.prod(as_shape(event_shape))
        norm = GeneralizedGamma(d, 2.0, a=d, beta=2.0, theta=math.sqrt(2.0), learn_shape=learnable, learn_scale=learnable)
        return cls(event_shape, 2.0, norm)

    @classmethod
    def laplace(cls, event_shape: ShapeLike, learnable: bool = False) -> RadialBase:
        """Product of standard Laplace distributions (``p = 1``, ``R ~ Gamma(d, 1)``)."""
        d = math.prod(as_shape(event_shape))
        norm = GeneralizedGamma(d, 1.0, a=d, beta=1.0, theta=1.0, learn_shape=learnable, learn_scale=learnable)
        return cls(event_shape, 1.0, norm)

    @classmethod
    def generalized_gaussian(cls, event_shape: ShapeLike, p: float, learnable: bool = False) -> RadialBase:
        """Product of generalized normals ``∝ exp(-|z_i|^p / p)`` (finite ``p``)."""
        if math.isinf(p):
            raise ValueError("p = inf has no generalized normal; use RadialBase(event_shape, p=float('inf'))")
        d = math.prod(as_shape(event_shape))
        norm = GeneralizedGamma(d, p, a=d, beta=p, theta=p ** (1.0 / p), learn_shape=learnable, learn_scale=learnable)
        return cls(event_shape, p, norm)

    # ----------------------------------------------------------- density
    @property
    def dim(self) -> int:
        return math.prod(self.event_shape)

    def norm_of(self, z: Tensor) -> Tensor:
        """``||z - loc||_p`` over the event dimensions."""
        return lp_norm(z - self.loc, self.p, len(self.event_shape))

    def log_profile(self, r: Tensor) -> Tensor:
        """``log h(r)`` -- the log-density at any point with ``||z - loc||_p = r``."""
        return self.norm.log_profile(r)

    def log_prob(self, z: Tensor) -> Tensor:
        return self.norm.log_profile(self.norm_of(z))

    def is_monotone(self) -> bool:
        """Whether the profile is non-increasing (upper level sets are Lp balls)."""
        return self.norm.profile_is_monotone(self.dim)

    # ----------------------------------------------------------- sampling
    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        """Samples, differentiable w.r.t. ``loc`` and the norm parameters."""
        shape = as_shape(sample_shape)
        n = math.prod(shape)
        r = self.norm.rsample((n,)).to(self.loc.dtype)
        u = sample_unit_sphere(n, self.event_shape, self.p, device=self.loc.device, dtype=self.loc.dtype)
        z = self.loc + r.reshape(n, *([1] * len(self.event_shape))) * u
        return z.reshape(*shape, *self.event_shape)

    def sample(self, sample_shape: ShapeLike = ()) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)


class RadialMixture(BaseDistribution):
    """Mixture of ``K`` Lp-radial distributions with distinct learnable centres.

    ``p(z) = Σ_k π_k h_k(||z - μ_k||_p)`` (exact ``log_prob`` via logsumexp).

    Args:
        event_shape: event shape.
        n_components: ``K`` (required if ``norms`` is not given).
        p: norm exponent shared by all components.
        norms: per-component norm distributions (default: independent
            learnable :class:`GeneralizedGamma` at the reference law).
        locs: initial centres ``(K, *event_shape)`` (default
            ``init_scale * randn``).
        weights: initial mixture weights (default uniform).
        learn_locs: learn the centres.
        learn_weights: learn the mixture weights.
        init_scale: scale of the random initial centres.
    """

    def __init__(
        self,
        event_shape: ShapeLike,
        n_components: Optional[int] = None,
        p: float = 2.0,
        norms: Optional[Sequence[NormDistribution]] = None,
        locs: Optional[Tensor] = None,
        weights: Optional[Sequence[float]] = None,
        learn_locs: bool = True,
        learn_weights: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        self.p = float(p)
        d = self.dim
        if norms is None:
            if n_components is None:
                raise ValueError("either n_components or norms must be given")
            norms = [GeneralizedGamma(d, self.p) for _ in range(n_components)]
        self.norms = nn.ModuleList(norms)
        k = len(self.norms)
        for nm in self.norms:
            _check_norm(nm, d, self.p)
        locs_t = init_scale * torch.randn(k, *self.event_shape) if locs is None else torch.as_tensor(locs, dtype=torch.get_default_dtype()).reshape(k, *self.event_shape)
        if learn_locs:
            self.locs = nn.Parameter(locs_t.clone())
        else:
            self.register_buffer("locs", locs_t.clone())
        w = torch.full((k,), 1.0 / k) if weights is None else torch.as_tensor(weights, dtype=torch.get_default_dtype())
        if learn_weights:
            self.logits = nn.Parameter(w.log())
        else:
            self.register_buffer("logits", w.log())

    @property
    def dim(self) -> int:
        return math.prod(self.event_shape)

    @property
    def n_components(self) -> int:
        return len(self.norms)

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.logits, -1)

    def component_norms(self, z: Tensor) -> Tensor:
        """``||z - μ_k||_p`` of shape ``(..., K)``."""
        n_ev = len(self.event_shape)
        diff = z.unsqueeze(z.dim() - n_ev) - self.locs
        return lp_norm(diff, self.p, n_ev)

    def component_log_prob(self, z: Tensor) -> Tensor:
        """``log h_k(||z - μ_k||_p)`` (without weights), shape ``(..., K)``."""
        r = self.component_norms(z)
        return torch.stack([nm.log_profile(r[..., k]) for k, nm in enumerate(self.norms)], dim=-1)

    def log_prob(self, z: Tensor) -> Tensor:
        return torch.logsumexp(self.component_log_prob(z) + torch.log_softmax(self.logits, -1), dim=-1)

    def responsibilities(self, z: Tensor) -> Tensor:
        """Posterior component probabilities ``(..., K)``."""
        return torch.softmax(self.component_log_prob(z) + torch.log_softmax(self.logits, -1), dim=-1)

    def rsample(self, sample_shape: ShapeLike = ()) -> Tensor:
        """Samples, reparameterized w.r.t. centres and norm parameters (not the weights)."""
        shape = as_shape(sample_shape)
        n = math.prod(shape)
        idx = torch.distributions.Categorical(logits=self.logits.detach()).sample((n,))
        out = self.locs.new_zeros(n, *self.event_shape)
        for k, nm in enumerate(self.norms):
            ids = torch.nonzero(idx == k).flatten()
            m = ids.numel()
            if m == 0:
                continue
            r = nm.rsample((m,)).to(self.locs.dtype)
            u = sample_unit_sphere(m, self.event_shape, self.p, device=self.locs.device, dtype=self.locs.dtype)
            vals = self.locs[k] + r.reshape(m, *([1] * len(self.event_shape))) * u
            out = out.index_put((ids,), vals)
        return out.reshape(*shape, *self.event_shape)

    def sample(self, sample_shape: ShapeLike = ()) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def is_monotone(self) -> bool:
        """Whether every component profile is non-increasing."""
        return all(nm.profile_is_monotone(self.dim) for nm in self.norms)

    def level_set_sandwich(self, log_t: Union[float, Tensor]) -> Tuple[Tensor, Tensor]:
        """Per-component ball radii sandwiching the upper level set ``{p ≥ t}``.

        With ``B_k(ρ) = {z : ||z - μ_k||_p ≤ ρ}``::

            ∪_k B_k(inner_k)  ⊆  {p ≥ t}  ⊆  ∪_k B_k(outer_k)

        where ``inner_k`` solves ``π_k h_k(ρ) = t`` and ``outer_k`` solves
        ``π_k h_k(ρ) = t / K`` (since ``p ≥ π_k h_k`` and
        ``p ≤ K max_k π_k h_k``). Requires monotone component profiles. A
        radius of ``0`` denotes an empty ball.

        Args:
            log_t: log-density threshold ``log t``.

        Returns:
            ``(inner, outer)``, each of shape ``(K,)``.
        """
        from .levelsets import density_threshold_radius

        if not self.is_monotone():
            raise ValueError("level_set_sandwich requires monotone component profiles")
        log_w = torch.log_softmax(self.logits.detach(), -1)
        log_t = torch.as_tensor(log_t, dtype=log_w.dtype, device=log_w.device)
        k = self.n_components
        inner = torch.stack([density_threshold_radius(nm, log_t - log_w[i]) for i, nm in enumerate(self.norms)])
        outer = torch.stack(
            [density_threshold_radius(nm, log_t - math.log(k) - log_w[i]) for i, nm in enumerate(self.norms)]
        )
        return inner, outer
