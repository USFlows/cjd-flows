"""Shift-only (volume-preserving) autoregressive layers.

* :class:`ShiftAutoregressive` -- ``z_i = x_i + mu_i(x_<i)`` with a
  :class:`~cjd_flows.conditioners.MADE` (vectors).
* :class:`MaskedConvAutoregressive` -- *experimental* spatial version with
  PixelCNN-style causal convolutions (images / sequences / volumes).

The normalizing direction (``forward``) is a single parallel pass; the
generative direction (``inverse``) is sequential. The Jacobian is unit lower
triangular (in the autoregressive order), hence ``log|det J| = 0``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cjd_flows.conditioners import MADE, _prepare_context, make_activation
from cjd_flows.core import ShapeLike, VolumePreserving, as_shape

__all__ = ["ShiftAutoregressive", "MaskedConvAutoregressive", "causal_kernel_mask"]


class ShiftAutoregressive(VolumePreserving):
    """Additive autoregressive layer ``z_i = x_i + mu_i(x_{<i}, context)``.

    ``forward`` (data -> latent) costs one MADE pass. ``inverse`` solves the
    triangular system by the fixed-point iteration ``x <- z - mu(x)``, which is
    exact after ``D`` passes (after pass ``k`` the first ``k`` dimensions of
    the autoregressive order are exact).

    Args:
        features: number of dimensions ``D``.
        made: a :class:`MADE` with ``features == D`` and ``out_multiplier == 1``.
            Defaults to ``MADE(features)``.

    ``piecewise_affine`` follows the MADE.
    """

    def __init__(self, features: int, made: Optional[MADE] = None):
        super().__init__()
        if made is None:
            made = MADE(features)
        if made.features != features or made.out_multiplier != 1:
            raise ValueError("made must have features == features and out_multiplier == 1")
        self.features = int(features)
        self.made = made
        self.piecewise_affine = bool(made.piecewise_affine)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return x + self.made(x, context)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        x = z
        # mu of the first dimension in the order is constant -> D passes suffice
        for _ in range(self.features):
            x = z - self.made(x, context)
        return x


def causal_kernel_mask(kernel_size: int, ndim: int, include_center: bool) -> Tensor:
    """Raster-order causal mask for an ``ndim``-D kernel (PixelCNN type A/B).

    Kernel offsets that precede the centre in row-major (raster) order are 1;
    the centre itself is 1 iff ``include_center`` (type B).
    """
    K = kernel_size**ndim
    center = K // 2
    m = torch.arange(K) < center
    if include_center:
        m[center] = True
    return m.reshape((kernel_size,) * ndim).to(torch.get_default_dtype())


class _MaskedConv(nn.Module):
    def __init__(self, conv: nn.modules.conv._ConvNd, mask: Tensor):
        super().__init__()
        self.conv = conv
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        c = self.conv
        return c._conv_forward(x, c.weight * self.mask, c.bias)


_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}


class MaskedConvAutoregressive(VolumePreserving):
    """**Experimental.** Spatial autoregressive shift layer with causal convolutions.

    ``z[:, :, p] = x[:, :, p] + mu(x[:, :, <p])[:, :, p]`` where ``<p`` are the
    positions preceding ``p`` in raster order (all channels). The shift
    network is a PixelCNN-style stack: a type-A masked convolution (excludes
    the centre), type-B masked hidden convolutions with residual connections,
    and a zero-initialized type-B output convolution. Channels at the same
    position do not condition on each other.

    ``forward`` is one parallel pass; ``inverse`` needs ``prod(spatial)``
    sequential passes (raster scan) and is therefore slow for large inputs.

    Args:
        event_shape: ``(C, *spatial)`` with 1-3 spatial dims.
        hidden: hidden channel widths.
        kernel_size: odd kernel size.
        activation: activation name or module.
        context_dim: context size (added to the first hidden layer through an
            unmasked linear map, broadcast over space).
    """

    def __init__(
        self,
        event_shape: ShapeLike,
        hidden: Sequence[int] = (64, 64),
        kernel_size: int = 3,
        activation: Union[str, nn.Module] = "relu",
        context_dim: int = 0,
    ):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        C, *spatial = self.event_shape
        nd = len(spatial)
        if nd not in _CONV:
            raise ValueError("event_shape must be (C, *spatial) with 1-3 spatial dims")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        hidden = tuple(int(h) for h in hidden)
        self.spatial_ndim = nd
        self.context_dim = int(context_dim)
        Conv = _CONV[nd]
        pad = kernel_size // 2
        mask_a = causal_kernel_mask(kernel_size, nd, include_center=False)
        mask_b = causal_kernel_mask(kernel_size, nd, include_center=True)
        self.act, _, pa = make_activation(activation)
        self.piecewise_affine = bool(pa)

        self.inp = _MaskedConv(Conv(C, hidden[0], kernel_size, padding=pad), mask_a)
        # context enters through an unmasked 1x1 path (no spatial information)
        self.ctx = nn.Linear(self.context_dim, hidden[0], bias=False) if self.context_dim > 0 else None
        self.blocks = nn.ModuleList(
            _MaskedConv(Conv(hidden[i - 1], hidden[i], kernel_size, padding=pad), mask_b) for i in range(1, len(hidden))
        )
        self.projs = nn.ModuleList(
            Conv(hidden[i - 1], hidden[i], 1, bias=False) if hidden[i - 1] != hidden[i] else nn.Identity()
            for i in range(1, len(hidden))
        )
        out = Conv(hidden[-1], C, kernel_size, padding=pad)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.out = _MaskedConv(out, mask_b)

    @property
    def num_positions(self) -> int:
        return math.prod(self.event_shape[1:])

    def shift(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """The autoregressive shift ``mu(x)``."""
        h = self.inp(x)
        ctx = _prepare_context(context, self.context_dim, x.shape[0], x)
        if ctx is not None:
            h = h + self.ctx(ctx).reshape(x.shape[0], -1, *([1] * self.spatial_ndim))
        for blk, proj in zip(self.blocks, self.projs):
            h = proj(h) + blk(self.act(h))
        return self.out(self.act(h))

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return x + self.shift(x, context)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        x = z
        for _ in range(self.num_positions):
            x = z - self.shift(x, context)
        return x
