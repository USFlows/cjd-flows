"""Density level sets and highest-density regions of radial distributions.

For a radial density ``p(z) = h(||z - loc||_p)`` with a **monotone**
(non-increasing) profile ``h``, every upper density level set (UDL)
``{z : p(z) ≥ t}`` is an Lp ball around ``loc``. Hence

* the UDL of probability mass ``q`` is the ball of radius ``F_R^{-1}(q)``,
* the mass of the region that is *more likely* than ``z`` is ``F_R(||z||)``
  (``Uniform(0, 1)`` under the model; ``1 - F_R(||z||)`` is the anomaly
  p-value).

For non-monotone ("shell") profiles UDLs are unions of annuli; these are
computed on a log-radius grid by :func:`udl_intervals`. Since flows in this
library have a constant Jacobian determinant, these sets map back to exact
density level sets in data space.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from ._numerics import bisect_increasing, safe_log
from .base import RadialBase
from .norms import NormDistribution

__all__ = [
    "udl_radius",
    "hdr_pvalue",
    "density_threshold_radius",
    "udl_log_threshold",
    "udl_intervals",
]

BaseOrNorm = Union[RadialBase, NormDistribution]


def _norm(base: BaseOrNorm) -> NormDistribution:
    return base.norm if isinstance(base, RadialBase) else base


def _require_monotone(norm: NormDistribution, fn: str) -> None:
    if not norm.profile_is_monotone():
        raise ValueError(f"{fn} requires a monotone profile; use udl_intervals for shell profiles")


def udl_radius(base: BaseOrNorm, q: Union[float, Tensor], check_monotone: bool = True) -> Tensor:
    """Radius ``r_q`` with ``P(||Z - loc|| ≤ r_q) = q``.

    For a monotone profile the ball of radius ``r_q`` is the upper density
    level set (= highest density region) of mass ``q``.
    """
    norm = _norm(base)
    if check_monotone:
        _require_monotone(norm, "udl_radius")
    return norm.icdf(torch.as_tensor(q, dtype=norm._dtype, device=norm._device))


def _grid(norm: NormDistribution, n_grid: int) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Log-radius nodes, log-profile, normalized cell masses and spacing."""
    with torch.no_grad():
        table = norm.grid_table(n_grid)
        s, ell = table.s, table.ell
        delta = float(s[1] - s[0])
        log_h = norm.log_profile_log_radius(s)
        w = torch.softmax(ell, dim=0)
        return s, log_h, w, delta


def _sorted_masses(log_h: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
    order = torch.argsort(log_h, descending=True)
    return log_h[order], torch.cumsum(w[order], 0)


def hdr_pvalue(base: BaseOrNorm, z: Tensor, n_grid: int = 65536) -> Tensor:
    """Mass of the region that is more likely than ``z``: ``P(h(R) ≥ h(||z||))``.

    For a monotone profile this is exactly ``F_R(||z - loc||_p)``. For shell
    profiles it is computed on a log-radius grid. Under the model the result
    is ``Uniform(0, 1)``; large values indicate atypical points.

    Args:
        base: a :class:`RadialBase` (``z`` are points) or a
            :class:`NormDistribution` (``z`` are radii).
        z: points ``(B, *event_shape)`` or radii.
        n_grid: grid size for non-monotone profiles.
    """
    norm = _norm(base)
    r = base.norm_of(z) if isinstance(base, RadialBase) else z
    if norm.profile_is_monotone():
        return norm.cdf(r)
    s, log_h, w, _ = _grid(norm, n_grid)
    lh_sorted, cum = _sorted_masses(log_h, w)
    thr = norm.log_profile(r).detach()
    # number of grid cells with log_h >= thr
    cnt = torch.searchsorted((-lh_sorted).contiguous(), (-thr).reshape(-1).contiguous(), right=True)
    mass = torch.cat([cum.new_zeros(1), cum])[cnt]
    return mass.reshape(thr.shape).clamp(0.0, 1.0)


def density_threshold_radius(base: BaseOrNorm, log_density_threshold: Union[float, Tensor]) -> Tensor:
    """Radius ``ρ`` with ``log h(ρ) = log_density_threshold`` (monotone profiles).

    ``{z : log p(z) ≥ threshold}`` is the ball of radius ``ρ`` around
    ``loc``. Returns ``0`` if the threshold exceeds ``sup h`` (empty set).
    Vectorized over thresholds (bisection in ``log r``, no gradient).
    """
    norm = _norm(base)
    _require_monotone(norm, "density_threshold_radius")
    thr = torch.as_tensor(log_density_threshold, dtype=norm._dtype, device=norm._device)
    lo, hi = norm.log_radius_bounds()
    lo, hi = lo - 50.0, hi + 50.0
    with torch.no_grad():
        s = bisect_increasing(lambda x: -norm.log_profile_log_radius(x), -thr, lo, hi, n_iter=120)
        top = norm.log_profile_log_radius(torch.full_like(thr, lo))
        r = torch.exp(s)
        return torch.where(thr > top, torch.zeros_like(r), r)


def udl_log_threshold(base: BaseOrNorm, q: float, n_grid: int = 65536) -> Tensor:
    """Log-density threshold ``log t_q`` whose upper level set has mass ``q``."""
    norm = _norm(base)
    _, log_h, w, _ = _grid(norm, n_grid)
    lh_sorted, cum = _sorted_masses(log_h, w)
    idx = torch.searchsorted(cum, torch.tensor([float(q)], dtype=cum.dtype, device=cum.device)).clamp(max=cum.numel() - 1)
    return lh_sorted[idx[0]]


def udl_intervals(
    base: BaseOrNorm,
    q: Optional[float] = None,
    log_threshold: Optional[Union[float, Tensor]] = None,
    n_grid: int = 65536,
) -> Tensor:
    """Upper density level set as a union of radial intervals (any profile).

    Exactly one of ``q`` (probability mass) or ``log_threshold`` (log-density
    threshold) must be given. The set ``{r : log h(r) ≥ threshold}`` is
    computed on a uniform log-radius grid over the support (resolution
    ``(hi - lo) / n_grid`` in ``log r``); fully vectorized.

    Returns:
        Tensor ``(M, 2)`` of disjoint intervals ``[a_m, b_m]`` of radii
        (``M = 0`` for an empty set). An interval reaching the lower end of the
        support starts at ``0``.
    """
    if (q is None) == (log_threshold is None):
        raise ValueError("exactly one of q or log_threshold must be given")
    norm = _norm(base)
    s, log_h, w, delta = _grid(norm, n_grid)
    if q is not None:
        lh_sorted, cum = _sorted_masses(log_h, w)
        idx = torch.searchsorted(cum, torch.tensor([float(q)], dtype=cum.dtype, device=cum.device))
        thr = lh_sorted[idx.clamp(max=cum.numel() - 1)[0]]
    else:
        thr = torch.as_tensor(log_threshold, dtype=log_h.dtype, device=log_h.device)
    mask = log_h >= thr
    false = mask.new_zeros(1)
    prev = torch.cat([false, mask[:-1]])
    nxt = torch.cat([mask[1:], false])
    starts = s[mask & ~prev] - delta / 2
    ends = s[mask & ~nxt] + delta / 2
    out = torch.stack([starts.exp(), ends.exp()], dim=-1)
    if mask[0]:
        out[0, 0] = 0.0
    return out
