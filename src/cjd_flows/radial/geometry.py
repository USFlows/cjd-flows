"""Geometry of Lp balls and spheres.

For ``p > 0`` (including ``p = inf``) the Lp unit ball in ``R^d`` has volume

    V_p = (2 Γ(1 + 1/p))^d / Γ(1 + d/p),       V_inf = 2^d,

and the "sphere measure" of radius ``r`` (the derivative of the ball volume
``V_p r^d``) is ``d V_p r^{d-1}``. A radial density ``p(z) = h(||z||_p)``
therefore induces the norm density ``p_R(r) = h(r) d V_p r^{d-1}``. All
quantities are computed in log space with ``lgamma``.

Reference law
-------------
Every norm family in :mod:`cjd_flows.radial.norms` is initialized relative to
the norm law of a *reference* distribution ``p(z) ∝ exp(-||z||_p^β / β)``
with ``β = p`` for finite ``p`` and ``β = 2`` for ``p = inf``. For ``p = 2``
this is the standard normal (``R ~ chi(d)``), for ``p = 1`` the standard
Laplace product distribution (``R ~ Gamma(d, 1)``). Its radius is
``R = (β G)^{1/β}`` with ``G ~ Gamma(d/β)``, so ``log R`` has mean
``(log β + ψ(d/β)) / β`` and standard deviation ``sqrt(ψ'(d/β)) / β``
(≈ ``(1/β) log d`` and ≈ ``1/sqrt(β d)`` in high dimensions).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from cjd_flows.core import ShapeLike, as_shape

from ._numerics import safe_log


def _check_p(p: float) -> float:
    p = float(p)
    if not p > 0:
        raise ValueError(f"p must be positive (or inf), got {p}")
    return p


def reference_exponent(p: float) -> float:
    """Exponent ``β`` of the reference profile ``exp(-r^β/β)`` (``p``, or ``2`` for ``p = inf``)."""
    p = _check_p(p)
    return 2.0 if math.isinf(p) else p


def reference_log_radius(d: int, p: float) -> Tuple[float, float]:
    """Mean and standard deviation of ``log ||Z||_p`` under the reference law.

    Args:
        d: dimension.
        p: norm exponent.

    Returns:
        ``(mu0, s0)`` as Python floats.
    """
    beta = reference_exponent(p)
    k = torch.tensor(d / beta, dtype=torch.float64)
    mu0 = (math.log(beta) + float(torch.special.digamma(k))) / beta
    s0 = math.sqrt(float(torch.special.polygamma(1, k))) / beta
    return mu0, s0


def log_unit_ball_volume(d: int, p: float) -> float:
    """``log V_p`` -- log-volume of the Lp unit ball in ``R^d``."""
    p = _check_p(p)
    if math.isinf(p):
        return d * math.log(2.0)
    return d * (math.log(2.0) + math.lgamma(1.0 + 1.0 / p)) - math.lgamma(1.0 + d / p)


def log_sphere_measure(d: int, p: float, r: Union[Tensor, float]) -> Tensor:
    """``log(d V_p r^{d-1})`` -- log of the Lp "surface" measure at radius ``r``.

    ``r^{d-1}`` is never formed; ``r`` is clamped to the smallest positive
    normal number.
    """
    r = torch.as_tensor(r)
    return math.log(d) + log_unit_ball_volume(d, p) + (d - 1) * safe_log(r)


def lp_norm(z: Tensor, p: float, event_ndim: int = 1) -> Tensor:
    """Lp norm over the last ``event_ndim`` dimensions.

    For ``p ∉ {1, 2, inf}`` the input is rescaled by its max-abs entry to
    avoid overflow of ``|z|^p``.
    """
    p = _check_p(p)
    flat = z.flatten(start_dim=z.dim() - event_ndim) if event_ndim > 0 else z.unsqueeze(-1)
    if p in (1.0, 2.0) or math.isinf(p):
        return torch.linalg.vector_norm(flat, ord=p, dim=-1)
    m = flat.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(flat.dtype).tiny)
    return torch.linalg.vector_norm(flat / m, ord=p, dim=-1) * m.squeeze(-1)


def sample_unit_sphere(
    n: int,
    event_shape: ShapeLike,
    p: float,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Samples ``n`` points on the Lp unit sphere w.r.t. the cone measure.

    The cone measure is the law of ``Z / ||Z||_p`` for any Lp-radial ``Z``
    (it is *not* the surface measure unless ``p ∈ {2, inf}``).

    * ``p = 2``: normalized Gaussian vectors.
    * finite ``p``: ``|z_i|^p ~ Gamma(1/p)`` with a random sign (i.e. ``z_i``
      has density ``∝ exp(-|z_i|^p)``), normalized by ``||z||_p``. Computed in
      log space, so arbitrary ``p`` works without under/overflow.
    * ``p = inf``: uniform on the surface of the cube ``[-1, 1]^d``: one face
      coordinate and its sign are chosen uniformly, the other coordinates are
      uniform on ``[-1, 1]``.

    Returns:
        Tensor of shape ``(n, *event_shape)``.
    """
    p = _check_p(p)
    shape = as_shape(event_shape)
    d = math.prod(shape)
    dtype = dtype or torch.get_default_dtype()
    if math.isinf(p):
        x = torch.rand(n, d, device=device, dtype=dtype) * 2 - 1
        face = torch.randint(0, d, (n, 1), device=device)
        sign = torch.randint(0, 2, (n, 1), device=device).to(dtype) * 2 - 1
        x = x.scatter(1, face, sign)
        return x.reshape(n, *shape)
    if p == 2.0:
        x = torch.randn(n, d, device=device, dtype=dtype)
    else:
        conc = torch.full((n, d), 1.0 / p, device=device, dtype=dtype)
        g = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample()
        lz = safe_log(g) / p
        lz = lz - lz.amax(dim=-1, keepdim=True)
        sign = torch.randint(0, 2, (n, d), device=device).to(dtype) * 2 - 1
        x = lz.exp() * sign
    x = x / lp_norm(x, p, 1).unsqueeze(-1)
    return x.reshape(n, *shape)
