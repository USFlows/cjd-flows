"""Shift-only (volume-preserving) autoregressive layers.

* :class:`ShiftAutoregressive` -- ``z_i = x_i + mu_i(x_<i)`` with a
  :class:`~cjd_flows.conditioners.MADE` (vectors).
* :class:`MaskedConvAutoregressive` -- *experimental* spatial version with
  PixelCNN-style causal convolutions (images / sequences / volumes).
* :class:`TransformerAutoregressive` -- token-autoregressive layer with a
  causal transformer (volume-preserving variant of TarFlow), inverse with a
  key/value cache.

The normalizing direction (``forward``) is a single parallel pass; the
generative direction (``inverse``) is sequential. The Jacobian is unit lower
triangular (in the autoregressive order), hence ``log|det J| = 0``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from cjd_flows.conditioners import MADE, _JetBlock, _prepare_context, _TokenMLP, make_activation
from cjd_flows.core import ShapeLike, VolumePreserving, as_shape

__all__ = ["ShiftAutoregressive", "MaskedConvAutoregressive", "TransformerAutoregressive", "causal_kernel_mask"]


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


# --------------------------------------------------------------------------- transformer
class TransformerAutoregressive(VolumePreserving):
    """Token-autoregressive layer with a causal transformer.

    For a sequence of ``N`` tokens with ``D`` features each (event shape
    ``(N, D)``, e.g. the output of :class:`~cjd_flows.bijections.Patchify`)::

        z_t = x_t + mu_t(x_<t, context)                         (scale=False)
        z_t = x_t * exp(s_t(x_<t, context)) + mu_t(x_<t, context)  (scale=True)

    where ``s_t`` is bounded by ``clamp * tanh(. / clamp)`` and centred over the
    ``D`` features of each token, so ``sum_d s_{t,d} = 0``. The Jacobian is
    block lower triangular with diagonal blocks ``I`` (or ``diag(exp(s_t))``),
    hence ``log|det J| = 0`` in both cases. With ``scale=True`` this is the
    GIN-style (volume-preserving) counterpart of TarFlow (Zhai et al. 2024),
    whose per-token scales change the volume.

    The shift network is a pre-LN transformer with causal attention, QK-norm
    and adaLN-zero context modulation (the blocks of
    :class:`~cjd_flows.conditioners.JetConditioner`). Its inputs are a learned
    start token followed by ``x_0, ..., x_{N-2}``, so output ``t`` only sees
    ``x_<t``. The output head is zero-initialized (identity at init).

    ``forward`` (data -> latent) is one parallel pass. ``inverse`` decodes the
    tokens one by one and reuses the attention keys/values of earlier tokens,
    so it costs about one parallel pass of compute spread over ``N``
    sequential steps.

    Alternate ``reverse`` between stacked layers so that every token is
    conditioned on every other one in some layer.

    Numerics: with ``scale=False`` the inverse is exact up to rounding. With
    ``scale=True`` rounding errors of earlier tokens are multiplied by up to
    ``exp(|s|)`` in later ones, so large log-scales make the sequential
    inverse (sampling) ill-conditioned over long sequences. ``forward`` and
    ``log_prob`` are not affected. Check ``inverse(forward(x))`` on trained
    models, or lower ``clamp``.

    Args:
        event_shape: ``(N, D)``: number of tokens and features per token.
        embed_dim: transformer width.
        depth: number of transformer blocks.
        num_heads: attention heads (default ``max(1, embed_dim // 64)``).
        mlp_ratio: token MLP hidden width / embed_dim.
        context_dim: context size (``0``: unconditional).
        reverse: process the tokens in reverse order.
        scale: add zero-sum log-scales per token (see above).
        clamp: soft bound of the raw log-scale (``None`` disables it).
        qk_norm: RMSNorm on per-head queries/keys.
        nonlinearity: token MLP activation (default GELU).
        grad_checkpointing: recompute block activations in backward.
    """

    piecewise_affine = False

    def __init__(
        self,
        event_shape: ShapeLike,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        context_dim: int = 0,
        reverse: bool = False,
        scale: bool = False,
        clamp: Optional[float] = 2.0,
        qk_norm: bool = True,
        nonlinearity: Optional[nn.Module] = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        if len(self.event_shape) != 2:
            raise ValueError(f"event_shape must be (num_tokens, token_dim), got {self.event_shape}")
        N, D = self.event_shape
        if num_heads is None:
            num_heads = max(1, embed_dim // 64)
        self.num_tokens, self.token_dim = N, D
        self.context_dim = int(context_dim)
        self.reverse = bool(reverse)
        self.scale = bool(scale)
        self.clamp = clamp
        self.grad_checkpointing = grad_checkpointing

        self.inp = nn.Linear(D, embed_dim)
        self.start = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, N, embed_dim) * 0.02)
        # unconditional layers use a learned constant embedding; conditional
        # layers embed the (zero-filled if missing) context, so they carry no
        # parameter that would stay unused (DistributedDataParallel)
        if self.context_dim > 0:
            self.ctx_embed = _TokenMLP(self.context_dim, embed_dim, embed_dim)
            self.null_ctx = None
        else:
            self.ctx_embed = None
            self.null_ctx = nn.Parameter(torch.zeros(embed_dim))
        self.blocks = nn.ModuleList(
            _JetBlock(embed_dim, num_heads, mlp_ratio, qk_norm, True, nonlinearity) for _ in range(depth)
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 2 * D if self.scale else D)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _embed_context(self, context: Optional[Tensor], B: int, ref: Tensor) -> Tensor:
        if self.ctx_embed is None:
            return self.null_ctx.to(ref.dtype).unsqueeze(0).expand(B, -1)
        return self.ctx_embed(_prepare_context(context, self.context_dim, B, ref))

    def _params(self, x: Tensor, c: Tensor) -> Tensor:
        """Shift (and log-scale) parameters for all tokens, ``(B, N, D or 2D)``."""
        B = x.shape[0]
        h = torch.cat([self.start.expand(B, -1, -1).to(x.dtype), self.inp(x[:, :-1])], dim=1)
        h = h + self.pos_embed.to(h.dtype)
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(block, h, c, use_reentrant=False)
            else:
                h = block(h, c)
        return self.head(self.norm_out(h))

    def _scale_shift(self, params: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.scale:
            return params, None
        s, t = params.chunk(2, dim=-1)
        if self.clamp is not None:
            s = self.clamp * torch.tanh(s / self.clamp)
        return t, s - s.mean(dim=-1, keepdim=True)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.reverse:
            x = x.flip(1)
        t, s = self._scale_shift(self._params(x, self._embed_context(context, x.shape[0], x)))
        z = x + t if s is None else x * s.exp() + t
        return z.flip(1) if self.reverse else z

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.reverse:
            z = z.flip(1)
        B, N = z.shape[0], self.num_tokens
        c = self._embed_context(context, B, z)
        caches = [dict() for _ in self.blocks]
        h = self.start.expand(B, -1, -1).to(z.dtype)
        xs = []
        for i in range(N):
            h = h + self.pos_embed[:, i : i + 1].to(h.dtype)
            for block, cache in zip(self.blocks, caches):
                h = block.step(h, c, cache)
            t, s = self._scale_shift(self.head(self.norm_out(h))[:, 0])
            x_i = z[:, i] - t if s is None else (z[:, i] - t) * (-s).exp()
            xs.append(x_i)
            if i + 1 < N:
                h = self.inp(x_i).unsqueeze(1)
        x = torch.stack(xs, dim=1)
        return x.flip(1) if self.reverse else x
