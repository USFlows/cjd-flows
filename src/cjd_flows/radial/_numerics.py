"""Private numerical helpers for 1-D densities on the log-radius axis.

All quadratures in this package integrate densities that are represented by
their logarithm ``ell(s)`` on a grid of nodes ``s``. Between two nodes the
log-density is interpolated *linearly*, so every panel integral
``∫ exp(ell)`` has a closed form ("exponential-linear trapezoid rule"). This
rule is exact for densities whose logarithm is piecewise linear (in
particular for exponential kernels such as ``exp(-d (σ - s))``) and does not
underflow because everything is computed with ``logsumexp``.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Tuple

import torch
from torch import Tensor

#: Lower clamp for log-densities, avoids ``-inf - (-inf)`` in differences.
LOG_FLOOR = -1e30


def safe_log(x: Tensor) -> Tensor:
    """``log(x)`` with ``x`` clamped to the smallest positive normal number."""
    return torch.log(x.clamp_min(torch.finfo(x.dtype).tiny))


def _exporting() -> bool:
    return torch.onnx.is_in_onnx_export()


# Lanczos approximation (g = 7, n = 9); relative error ~1e-15 for arguments >= 1.
_LANCZOS_G = 7.0
_LANCZOS_COEF = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def _lgamma_lanczos(x: Tensor) -> Tensor:
    """``log Γ(x)`` for ``x > 0`` from elementary ops only (ONNX exportable).

    Uses ``log Γ(x) = log Γ(x + 1) - log x`` so that the Lanczos series is
    always evaluated at arguments ``>= 1``.
    """
    z = x  # Lanczos form for Γ(z + 1)
    a = torch.full_like(z, _LANCZOS_COEF[0])
    for i, c in enumerate(_LANCZOS_COEF[1:], start=1):
        a = a + c / (z + i)
    t = z + (_LANCZOS_G + 0.5)
    lg_zp1 = 0.5 * math.log(2 * math.pi) + (z + 0.5) * torch.log(t) - t + torch.log(a)
    return lg_zp1 - torch.log(x)


def lgamma(x: Tensor) -> Tensor:
    """``torch.lgamma`` (for positive arguments), replaced by an elementary-op
    approximation while exporting to ONNX (which has no ``lgamma`` operator)."""
    return _lgamma_lanczos(x) if _exporting() else torch.lgamma(x)


def logcumsumexp(x: Tensor, dim: int = -1) -> Tensor:
    """``torch.logcumsumexp``; while exporting to ONNX (no such operator) it is
    computed as ``m + log(cumsum(exp(x - m)))`` with ``m = max(x)``, so
    prefix sums smaller than ``exp(m)`` times the dtype's underflow
    threshold become ``-inf``."""
    if _exporting():
        m = x.detach().amax(dim=dim, keepdim=True).clamp_min(LOG_FLOOR)
        return torch.log(torch.cumsum(torch.exp(x - m), dim=dim)) + m
    return torch.logcumsumexp(x, dim=dim)


def searchsorted_right(sorted_seq: Tensor, values: Tensor) -> Tensor:
    """``torch.searchsorted(sorted_seq, values, right=True)`` for a 1-D ``sorted_seq``.

    While exporting to ONNX (no ``searchsorted`` operator) the index is
    computed as the number of entries ``<= value`` by broadcasting.
    """
    if _exporting():
        return (sorted_seq <= values.unsqueeze(-1)).sum(-1)
    return torch.searchsorted(sorted_seq.contiguous(), values.contiguous(), right=True)


def log_expm1_ratio(x: Tensor) -> Tensor:
    """Numerically stable ``log((exp(x) - 1) / x)`` (with value ``0`` at ``x = 0``)."""
    if _exporting():
        # ONNX has no expm1: use 1 - exp(-|x|), accurate enough for |x| >= 1e-2,
        # and the series (error < 1e-11) below that
        small = x.abs() < 1e-2
        xs = torch.where(small, torch.ones_like(x), x)
        ax = xs.abs()
        big = torch.relu(xs) + torch.log(1 - torch.exp(-ax)) - torch.log(ax)
        return torch.where(small, x / 2 + x * x / 24, big)
    small = x.abs() < 1e-4
    xs = torch.where(small, torch.ones_like(x), x)
    ax = xs.abs()
    big = torch.relu(xs) + torch.log(-torch.expm1(-ax)) - torch.log(ax)
    series = x / 2 + x * x / 24
    return torch.where(small, series, big)


def log_panel_integrals(s: Tensor, ell: Tensor) -> Tensor:
    """Log-integrals of ``exp(ell)`` over consecutive panels ``[s_i, s_{i+1}]``.

    ``ell`` is interpolated linearly between nodes. Zero-width panels give
    ``-inf``.

    Args:
        s: nodes, non-decreasing along the last dim, shape ``(..., n)``.
        ell: log-integrand at the nodes, shape ``(..., n)``.

    Returns:
        Tensor of shape ``(..., n - 1)``.
    """
    delta = s[..., 1:] - s[..., :-1]
    pos = delta > 0
    safe_delta = torch.where(pos, delta, torch.ones_like(delta))
    ell = ell.clamp_min(LOG_FLOOR)
    val = ell[..., :-1] + torch.log(safe_delta) + log_expm1_ratio(ell[..., 1:] - ell[..., :-1])
    return torch.where(pos, val, torch.full_like(val, -float("inf")))


class GridTable(NamedTuple):
    """Tabulated density of ``S = log R`` on a uniform grid.

    Attributes:
        s: grid nodes ``(n,)`` (no gradient).
        ell: log-density at the nodes ``(n,)`` (may carry gradients).
        log_cdf: normalized log-CDF at the nodes ``(n,)`` (``-inf`` at node 0).
        log_total: log of the (unnormalized) integral over the grid.
    """

    s: Tensor
    ell: Tensor
    log_cdf: Tensor
    log_total: Tensor


def make_table(s: Tensor, ell: Tensor) -> GridTable:
    """Builds a :class:`GridTable` from nodes and log-density values."""
    panels = log_panel_integrals(s, ell)
    cum = logcumsumexp(panels, dim=-1)
    log_total = cum[-1]
    log_cdf = torch.cat([cum.new_full((1,), -float("inf")), cum - log_total])
    return GridTable(s, ell, log_cdf, log_total)


def _locate(s_grid: Tensor, s_query: Tensor) -> Tuple[Tensor, Tensor]:
    n = s_grid.shape[0]
    idx = torch.searchsorted(s_grid.contiguous(), s_query.detach().contiguous(), right=True) - 1
    idx = idx.clamp(0, n - 2)
    return idx, s_grid[idx]


def table_cdf(table: GridTable, s_query: Tensor) -> Tensor:
    """CDF of ``S`` at ``s_query`` (any shape) from a :class:`GridTable`."""
    s, ell = table.s, table.ell.clamp_min(LOG_FLOOR)
    flat = s_query.reshape(-1)
    idx, s0 = _locate(s, flat)
    delta = s[idx + 1] - s0
    t = ((flat - s0) / delta).clamp(0.0, 1.0)
    width = (t * delta).clamp_min(torch.finfo(s.dtype).tiny)
    partial = ell[idx] + torch.log(width) + log_expm1_ratio(t * (ell[idx + 1] - ell[idx])) - table.log_total
    partial = torch.where(t > 0, partial, torch.full_like(partial, -float("inf")))
    out = torch.logaddexp(table.log_cdf[idx], partial).exp().clamp(0.0, 1.0)
    out = torch.where(flat < s[0], torch.zeros_like(out), out)
    out = torch.where(flat > s[-1], torch.ones_like(out), out)
    return out.reshape(s_query.shape)


def table_icdf(table: GridTable, q: Tensor) -> Tensor:
    """Inverse CDF (on the ``S = log R`` axis) from a :class:`GridTable`.

    Inverts the exponential-linear interpolant exactly, so the result is
    differentiable w.r.t. the tabulated log-density (implicit
    reparameterization).
    """
    s, ell = table.s, table.ell.clamp_min(LOG_FLOOR)
    n = s.shape[0]
    flat = q.reshape(-1).to(s.dtype)
    cdf_nodes = table.log_cdf.exp()
    idx = torch.searchsorted(cdf_nodes.detach().contiguous(), flat.detach().contiguous(), right=True) - 1
    idx = idx.clamp(0, n - 2)
    delta = s[idx + 1] - s[idx]
    dq = (flat - cdf_nodes[idx]).clamp_min(0.0)
    l0 = ell[idx] - table.log_total  # normalized log-density at left node
    k = (ell[idx + 1] - ell[idx]) / delta
    small = (k * delta).abs() < 1e-6
    k_safe = torch.where(small, torch.ones_like(k), k)
    arg = (k_safe * dq * torch.exp(-l0)).clamp_min(-1.0 + 1e-12)
    x_big = torch.log1p(arg) / k_safe
    x_small = dq * torch.exp(-l0)
    x = torch.where(small, x_small, x_big)
    x = torch.minimum(x.clamp_min(0.0), delta)
    return (s[idx] + x).reshape(q.shape)


def bisect_increasing(
    fn: Callable[[Tensor], Tensor], target: Tensor, lo: float, hi: float, n_iter: int = 80
) -> Tensor:
    """Vectorized bisection for ``fn(x) = target`` with ``fn`` non-decreasing.

    Returns the midpoint of the final bracket (values outside ``fn([lo, hi])``
    are clamped to the bracket ends). Runs without gradient.
    """
    with torch.no_grad():
        a = torch.full_like(target, lo)
        b = torch.full_like(target, hi)
        for _ in range(n_iter):
            m = 0.5 * (a + b)
            below = fn(m) < target
            a = torch.where(below, m, a)
            b = torch.where(below, b, m)
        return 0.5 * (a + b)


def inverse_trigamma(y: float, n_iter: int = 50) -> float:
    """Solves ``trigamma(k) = y`` for ``k > 0`` (Newton on ``1/k``)."""
    t = torch.tensor(y, dtype=torch.float64)
    k = 1.0 / t + 0.5
    for _ in range(n_iter):
        f = torch.special.polygamma(1, k) - t
        fp = torch.special.polygamma(2, k)
        k = (k - f / fp).clamp_min(k / 10)
    return float(k)
