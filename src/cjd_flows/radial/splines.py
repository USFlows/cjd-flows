"""Monotone rational-quadratic splines (Durkan et al., 2019) for scalar inputs.

The spline maps ``[-B, B]`` onto itself and is the identity outside (linear
tails with unit slope; the boundary derivatives are fixed to ``1``).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_MIN_WIDTH = 1e-3
DEFAULT_MIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def identity_derivative_init(min_derivative: float = DEFAULT_MIN_DERIVATIVE) -> float:
    """Unnormalized derivative value for which the spline derivative equals one."""
    return math.log(math.expm1(1.0 - min_derivative))


def _knots(unnorm: Tensor, bound: float, min_size: float) -> Tuple[Tensor, Tensor]:
    k = unnorm.shape[-1]
    w = torch.softmax(unnorm, dim=-1)
    w = min_size + (1 - min_size * k) * w
    cum = torch.cumsum(w, dim=-1)
    cum = F.pad(cum, (1, 0), value=0.0)
    cum = 2 * bound * cum - bound
    cum = torch.cat([cum.new_full((1,), -bound), cum[1:-1], cum.new_full((1,), bound)])
    return cum, cum[1:] - cum[:-1]


def rq_spline(
    x: Tensor,
    unnorm_widths: Tensor,
    unnorm_heights: Tensor,
    unnorm_derivatives: Tensor,
    bound: float,
    inverse: bool = False,
    min_width: float = DEFAULT_MIN_WIDTH,
    min_height: float = DEFAULT_MIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[Tensor, Tensor]:
    """Applies a 1-D monotone RQ spline element-wise.

    Args:
        x: inputs of any shape.
        unnorm_widths: ``(K,)`` bin width logits.
        unnorm_heights: ``(K,)`` bin height logits.
        unnorm_derivatives: ``(K - 1,)`` interior knot derivatives (pre-softplus).
        bound: half-width ``B`` of the spline interval.
        inverse: apply the inverse map.

    Returns:
        ``(y, log_abs_det)`` where ``log_abs_det = log|dy/dx|`` of the map that
        was applied (forward or inverse), element-wise.
    """
    cw, widths = _knots(unnorm_widths, bound, min_width)
    ch, heights = _knots(unnorm_heights, bound, min_height)
    ones = unnorm_derivatives.new_ones(1)
    derivs = torch.cat([ones, min_derivative + F.softplus(unnorm_derivatives), ones])

    inside = (x >= -bound) & (x <= bound)
    xi = x.clamp(-bound, bound)
    knots = ch if inverse else cw
    idx = torch.searchsorted(knots[1:-1].contiguous(), xi.detach().reshape(-1).contiguous(), right=True)
    idx = idx.reshape(x.shape)

    x_k, w_k = cw[idx], widths[idx]
    y_k, h_k = ch[idx], heights[idx]
    d_k, d_k1 = derivs[idx], derivs[idx + 1]
    s_k = h_k / w_k

    if not inverse:
        theta = (xi - x_k) / w_k
        t1mt = theta * (1 - theta)
        denom = s_k + (d_k1 + d_k - 2 * s_k) * t1mt
        y = y_k + h_k * (s_k * theta**2 + d_k * t1mt) / denom
        num_d = s_k**2 * (d_k1 * theta**2 + 2 * s_k * t1mt + d_k * (1 - theta) ** 2)
        lad = torch.log(num_d) - 2 * torch.log(denom)
    else:
        dy = xi - y_k
        c0 = d_k1 + d_k - 2 * s_k
        a = h_k * (s_k - d_k) + dy * c0
        b = h_k * d_k - dy * c0
        c = -s_k * dy
        disc = (b**2 - 4 * a * c).clamp_min(0.0)
        theta = (2 * c) / (-b - torch.sqrt(disc))
        theta = theta.clamp(0.0, 1.0)
        y = theta * w_k + x_k
        t1mt = theta * (1 - theta)
        denom = s_k + c0 * t1mt
        num_d = s_k**2 * (d_k1 * theta**2 + 2 * s_k * t1mt + d_k * (1 - theta) ** 2)
        lad = -(torch.log(num_d) - 2 * torch.log(denom))

    y = torch.where(inside, y, x)
    lad = torch.where(inside, lad, torch.zeros_like(lad))
    return y, lad
