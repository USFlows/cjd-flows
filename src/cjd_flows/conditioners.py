"""Conditioner networks for coupling and autoregressive layers.

All conditioners are plain :class:`torch.nn.Module` objects with the signature
``forward(x, context=None)`` and are pyro-free. Their **last layer is
zero-initialized**, so every coupling / autoregressive layer built on top of
them starts as the identity map.

Each conditioner exposes ``piecewise_affine: bool``. It is ``True`` iff the
network is a piecewise-affine function of its input (ReLU / LeakyReLU
activations, no normalization, no gating, no attention), which is what exact
SMT/MILP verification of the resulting flow requires. Dropout is ignored for
this flag since it is the identity in evaluation mode.

Context handling:
    * :class:`MLP` and :class:`MADE`: the context vector is concatenated to the
      input (``MADE``: added through an unmasked linear map, so every output
      may depend on it).
    * :class:`ConvNet`: the context is broadcast over space and appended as
      extra input channels.
    * :class:`JetConditioner`: adaLN-zero modulation in every block.

If a conditioner was built with ``context_dim > 0`` and ``context`` is
``None``, a zero context is used. If it was built with ``context_dim == 0``,
a passed context is ignored (so contextual and non-contextual layers can be
mixed in one flow).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm

__all__ = [
    "MLP",
    "ConvNet",
    "MADE",
    "JetConditioner",
    "LayerNormChannels",
    "sincos_pos_embed",
    "lipschitz_upper_bound",
    "make_activation",
]

ShapeLike = Union[int, Sequence[int]]

# Activation name -> (factory, Lipschitz constant, piecewise affine)
_ACTIVATIONS = {
    "relu": (nn.ReLU, 1.0, True),
    "leaky_relu": (lambda: nn.LeakyReLU(0.01), 1.0, True),
    "elu": (nn.ELU, 1.0, False),
    "tanh": (nn.Tanh, 1.0, False),
    "gelu": (nn.GELU, 1.13, False),  # max |GELU'| ~= 1.129
    "silu": (nn.SiLU, 1.1, False),  # max |SiLU'| ~= 1.0998
}


def make_activation(activation: Union[str, nn.Module]) -> Tuple[nn.Module, float, bool]:
    """Builds an activation module.

    Args:
        activation: one of ``relu, leaky_relu, elu, tanh, gelu, silu`` or an
            ``nn.Module`` instance (treated as non-piecewise-affine with
            unknown Lipschitz constant ``inf``, except ``nn.ReLU``/``nn.LeakyReLU``).

    Returns:
        ``(module, lipschitz_constant, piecewise_affine)``.
    """
    if isinstance(activation, nn.Module):
        if isinstance(activation, nn.ReLU):
            return activation, 1.0, True
        if isinstance(activation, nn.LeakyReLU):
            return activation, max(1.0, abs(activation.negative_slope)), True
        return activation, math.inf, False
    key = str(activation).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{activation}'. Choose from {sorted(_ACTIVATIONS)}.")
    factory, lip, pa = _ACTIVATIONS[key]
    return factory(), lip, pa


def _as_shape(shape: ShapeLike) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(s) for s in shape)


def _prepare_context(context: Optional[Tensor], context_dim: int, batch: int, ref: Tensor) -> Optional[Tensor]:
    """Returns a ``(B, context_dim)`` context (zeros if missing) or ``None`` if unused."""
    if context_dim == 0:
        return None
    if context is None:
        return ref.new_zeros(batch, context_dim)
    context = context.to(device=ref.device, dtype=ref.dtype)
    if context.numel() == context_dim:
        return context.reshape(1, context_dim).expand(batch, context_dim)
    return context.reshape(batch, context_dim)


class _ZeroGain(nn.Module):
    """Scalar gain initialized to zero (used to zero-init spectrally normalized output layers)."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gain


# --------------------------------------------------------------------------- spectral norms
def _matrix_spectral_norm(w: Tensor) -> float:
    return float(torch.linalg.matrix_norm(w.detach().double(), ord=2))


def _linear_bound(layer: nn.Linear) -> float:
    return _matrix_spectral_norm(layer.weight)


def _conv_bound(layer: nn.modules.conv._ConvNd) -> float:
    """Upper bound on the operator norm of a stride-1 zero-padded convolution.

    Uses ``min(sqrt(K) * ||W_mat||_2, sum_k ||W_k||_2)`` where ``W_mat`` is the
    ``(C_out, C_in * K)`` reshaped kernel and ``W_k`` the ``(C_out, C_in)``
    slice at kernel offset ``k`` (``K`` offsets). Both are valid bounds: the
    first because every input pixel appears in at most ``K`` unfolded patches,
    the second by the triangle inequality over shifted 1x1 convolutions.
    """
    w = layer.weight.detach().double()
    c_out = w.shape[0]
    K = math.prod(w.shape[2:])
    b1 = math.sqrt(K) * _matrix_spectral_norm(w.reshape(c_out, -1))
    wk = w.reshape(c_out, w.shape[1], K).permute(2, 0, 1)
    b2 = float(torch.linalg.matrix_norm(wk, ord=2).sum())
    return min(b1, b2)


def _layer_bound(layer: nn.Module) -> float:
    if isinstance(layer, nn.Linear):
        return _linear_bound(layer)
    if isinstance(layer, nn.modules.conv._ConvNd):
        return _conv_bound(layer)
    raise TypeError(f"No Lipschitz bound for {type(layer)}")


def lipschitz_upper_bound(module: nn.Module) -> float:
    """Upper bound on the Lipschitz constant (w.r.t. the l2 norm) of a conditioner.

    Supported: :class:`MLP` and :class:`ConvNet` without normalization layers
    and without gating. The bound is the product of per-layer bounds, where a
    residual block ``h -> P h + W act(h)`` contributes ``L_P + L_act * L_W``
    (``L_P = 1`` for the identity skip, i.e. ``1 + L`` per residual block).
    Linear layer norms are computed exactly (SVD of the *effective*, possibly
    spectrally-normalized weight); convolution norms use the bound described in
    :func:`_conv_bound`. The bound refers to evaluation mode (dropout = identity)
    and to the joint input ``(x, context)``, hence also bounds the Lipschitz
    constant in ``x`` alone.

    For a ReLU network this is a true upper bound; for smooth activations the
    activation's maximal slope is used (e.g. 1.13 for GELU).

    Args:
        module: an :class:`MLP` or :class:`ConvNet`.

    Returns:
        The bound as a Python float (``inf`` if the activation's Lipschitz constant is unknown).
    """
    fn = getattr(module, "_lipschitz_upper_bound", None)
    if fn is None:
        raise TypeError(f"lipschitz_upper_bound is not implemented for {type(module).__name__}")
    return float(fn())


# --------------------------------------------------------------------------- MLP
class _MLPBlock(nn.Module):
    """``h -> skip(h) + Linear(dropout(act(norm(h))))`` (``skip`` = id / projection / none)."""

    def __init__(self, d_in, d_out, activation, lip_act, residual, norm, dropout, sn):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if norm == "layernorm" else nn.Identity()
        self.act = activation
        self.lip_act = lip_act
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_in, d_out)
        if sn:
            self.linear = _spectral_norm(self.linear)
        self.residual = residual
        if residual and d_in != d_out:
            self.proj = nn.Linear(d_in, d_out, bias=False)
            if sn:
                self.proj = _spectral_norm(self.proj)
        else:
            self.proj = None

    def forward(self, h: Tensor) -> Tensor:
        out = self.linear(self.drop(self.act(self.norm(h))))
        if not self.residual:
            return out
        return out + (h if self.proj is None else self.proj(h))

    def bound(self) -> float:
        inner = self.lip_act * _layer_bound(self.linear)
        if not self.residual:
            return inner
        skip = 1.0 if self.proj is None else _layer_bound(self.proj)
        return skip + inner


class MLP(nn.Module):
    """Residual MLP conditioner ``x[, context] -> out``.

    Architecture: ``Linear(in + ctx, h0)`` -> residual blocks
    ``h -> h + Linear(act(norm(h)))`` for every subsequent hidden width ->
    ``act`` -> zero-initialized ``Linear(h_last, out)``. With ``hidden=(256, 256)``
    this is a standard 2-hidden-layer MLP (3 linear layers) with one skip connection.

    Input/output may be shapes: if ``in_features`` is a tuple the input is
    flattened, if ``out_features`` is a tuple the output is reshaped, so an
    MLP can be used as conditioner for image-shaped couplings.

    Args:
        in_features: input size (int) or event shape.
        out_features: output size (int) or event shape.
        hidden: hidden widths (non-empty).
        activation: activation name or module.
        context_dim: size of the context vector (concatenated to the input).
        residual: use skip connections (projection if widths differ).
        norm: ``None`` or ``"layernorm"`` (pre-norm in each block).
        dropout: dropout probability inside blocks.
        spectral_norm: wrap all linear layers in
            :func:`torch.nn.utils.parametrizations.spectral_norm`. The output
            layer is then followed by a scalar gain initialized to zero (a
            spectrally normalized zero matrix is undefined).
    """

    def __init__(
        self,
        in_features: ShapeLike,
        out_features: ShapeLike,
        hidden: Sequence[int] = (256, 256),
        activation: Union[str, nn.Module] = "relu",
        context_dim: int = 0,
        residual: bool = True,
        norm: Optional[str] = None,
        dropout: float = 0.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        hidden = tuple(int(h) for h in hidden)
        if len(hidden) == 0 or any(h <= 0 for h in hidden):
            raise ValueError("hidden must be a non-empty sequence of positive ints")
        if norm not in (None, "layernorm"):
            raise ValueError(f"norm must be None or 'layernorm', got {norm!r}")
        self.in_shape = _as_shape(in_features)
        self.out_shape = _as_shape(out_features)
        self.in_features = math.prod(self.in_shape)
        self.out_features = math.prod(self.out_shape)
        self.context_dim = int(context_dim)
        self.norm = norm
        self.spectral_norm = spectral_norm

        act, lip, pa = make_activation(activation)
        self._lip_act = lip
        self.piecewise_affine = bool(pa and norm is None)

        self.inp = nn.Linear(self.in_features + self.context_dim, hidden[0])
        if spectral_norm:
            self.inp = _spectral_norm(self.inp)
        self.blocks = nn.ModuleList(
            _MLPBlock(hidden[i - 1], hidden[i], act, lip, residual, norm, dropout, spectral_norm)
            for i in range(1, len(hidden))
        )
        self.out_norm = nn.LayerNorm(hidden[-1]) if norm == "layernorm" else nn.Identity()
        self.out_act = act
        self.out = nn.Linear(hidden[-1], self.out_features)
        if spectral_norm:
            self.out = _spectral_norm(self.out)
            self.out_gain = _ZeroGain()
        else:
            nn.init.zeros_(self.out.weight)
            self.out_gain = nn.Identity()
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        B = x.shape[0]
        h = x.reshape(B, self.in_features)
        ctx = _prepare_context(context, self.context_dim, B, h)
        if ctx is not None:
            h = torch.cat([h, ctx], dim=1)
        h = self.inp(h)
        for blk in self.blocks:
            h = blk(h)
        out = self.out_gain(self.out(self.out_act(self.out_norm(h))))
        return out.reshape(B, *self.out_shape)

    def _lipschitz_upper_bound(self) -> float:
        if self.norm is not None:
            raise ValueError("Lipschitz bound is not available with normalization layers")
        bound = _layer_bound(self.inp)
        for blk in self.blocks:
            bound *= blk.bound()
        bound *= self._lip_act * _layer_bound(self.out)
        if isinstance(self.out_gain, _ZeroGain):
            bound *= abs(float(self.out_gain.gain.detach()))
        return bound


# --------------------------------------------------------------------------- ConvNet
_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}


class LayerNormChannels(nn.Module):
    """LayerNorm over the channel dimension of ``(B, C, *spatial)`` tensors."""

    def __init__(self, channels: int, spatial_ndim: int, eps: float = 1e-5):
        super().__init__()
        shape = (1, channels) + (1,) * spatial_ndim
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class _ConvBlock(nn.Module):
    """Residual conv block.

    * plain:  ``h -> skip(h) + Conv_k(dropout(act(norm(h))))``
    * gated:  ``h -> skip(h) + val * sigmoid(gate)``, with
      ``[val, gate] = Conv_1(act(Conv_k(dropout(act(norm(h))))))`` (legacy GatedConvND).
    """

    def __init__(self, c_in, c_out, nd, k, dilation, activation, lip_act, residual, gated, norm, dropout, sn):
        super().__init__()
        Conv = _CONV[nd]
        pad = dilation * (k - 1) // 2
        wrap = _spectral_norm if sn else (lambda m: m)
        self.norm = LayerNormChannels(c_in, nd) if norm == "layernorm" else nn.Identity()
        self.act = activation
        self.lip_act = lip_act
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gated = gated
        self.conv = wrap(Conv(c_in, c_out, k, padding=pad, dilation=dilation))
        self.gate_conv = wrap(Conv(c_out, 2 * c_out, 1)) if gated else None
        self.residual = residual
        self.proj = wrap(Conv(c_in, c_out, 1, bias=False)) if residual and c_in != c_out else None

    def forward(self, h: Tensor) -> Tensor:
        out = self.conv(self.drop(self.act(self.norm(h))))
        if self.gated:
            val, gate = self.gate_conv(self.act(out)).chunk(2, dim=1)
            out = val * torch.sigmoid(gate)
        if not self.residual:
            return out
        return out + (h if self.proj is None else self.proj(h))

    def bound(self) -> float:
        if self.gated:
            raise ValueError("Lipschitz bound is not available for gated blocks")
        inner = self.lip_act * _layer_bound(self.conv)
        if not self.residual:
            return inner
        return (1.0 if self.proj is None else _layer_bound(self.proj)) + inner


class ConvNet(nn.Module):
    """Residual N-D convolutional conditioner ``(B, C_in, *S) -> (B, C_out, *S)``.

    Architecture: ``Conv_k(C_in + ctx, h0)`` -> residual (optionally gated)
    blocks for every subsequent hidden width -> ``act`` -> zero-initialized
    ``Conv_k(h_last, C_out)``. All convolutions are stride 1 with "same"
    padding ``dilation * (k - 1) // 2`` (``kernel_size`` must be odd), so the
    spatial shape is preserved. The context is broadcast over space and
    appended as extra input channels.

    Args:
        in_channels: input channels.
        out_channels: output channels (exactly).
        spatial_ndim: number of spatial dims (1, 2 or 3).
        hidden: hidden channel widths (non-empty).
        kernel_size: odd kernel size.
        activation: activation name or module.
        context_dim: size of the context vector.
        gated: use gated blocks (``val * sigmoid(gate)``; not piecewise affine).
        norm: ``None`` or ``"layernorm"`` (channel LayerNorm, pre-norm).
        dropout: dropout probability inside blocks.
        spectral_norm: spectrally normalize all convolutions (torch normalizes
            the reshaped ``(C_out, C_in*K)`` kernel, which is *not* the operator
            norm; see :func:`lipschitz_upper_bound`). The output conv is then
            followed by a zero-initialized scalar gain.
        dilation: dilation of the ``k x k`` convolutions.
        residual: use skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_ndim: int = 2,
        hidden: Sequence[int] = (64, 64),
        kernel_size: int = 3,
        activation: Union[str, nn.Module] = "relu",
        context_dim: int = 0,
        gated: bool = False,
        norm: Optional[str] = None,
        dropout: float = 0.0,
        spectral_norm: bool = False,
        dilation: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        if spatial_ndim not in _CONV:
            raise ValueError(f"spatial_ndim must be 1, 2 or 3, got {spatial_ndim}")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve the spatial shape")
        if norm not in (None, "layernorm"):
            raise ValueError(f"norm must be None or 'layernorm', got {norm!r}")
        hidden = tuple(int(h) for h in hidden)
        if len(hidden) == 0 or any(h <= 0 for h in hidden):
            raise ValueError("hidden must be a non-empty sequence of positive ints")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spatial_ndim = int(spatial_ndim)
        self.context_dim = int(context_dim)
        self.norm = norm
        self.gated = gated

        act, lip, pa = make_activation(activation)
        self._lip_act = lip
        self.piecewise_affine = bool(pa and norm is None and not gated)

        Conv = _CONV[spatial_ndim]
        pad = dilation * (kernel_size - 1) // 2
        wrap = _spectral_norm if spectral_norm else (lambda m: m)
        self.inp = wrap(Conv(self.in_channels + self.context_dim, hidden[0], kernel_size, padding=pad, dilation=dilation))
        self.blocks = nn.ModuleList(
            _ConvBlock(hidden[i - 1], hidden[i], spatial_ndim, kernel_size, dilation, act, lip, residual, gated, norm, dropout, spectral_norm)
            for i in range(1, len(hidden))
        )
        self.out_norm = LayerNormChannels(hidden[-1], spatial_ndim) if norm == "layernorm" else nn.Identity()
        self.out_act = act
        self.out = Conv(hidden[-1], self.out_channels, kernel_size, padding=pad, dilation=dilation)
        nn.init.zeros_(self.out.bias)
        if spectral_norm:
            self.out = _spectral_norm(self.out)
            self.out_gain = _ZeroGain()
        else:
            nn.init.zeros_(self.out.weight)
            self.out_gain = nn.Identity()

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        B = x.shape[0]
        ctx = _prepare_context(context, self.context_dim, B, x)
        if ctx is not None:
            ctx = ctx.reshape(B, self.context_dim, *([1] * self.spatial_ndim)).expand(B, self.context_dim, *x.shape[2:])
            x = torch.cat([x, ctx], dim=1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_gain(self.out(self.out_act(self.out_norm(h))))

    def _lipschitz_upper_bound(self) -> float:
        if self.norm is not None:
            raise ValueError("Lipschitz bound is not available with normalization layers")
        bound = _layer_bound(self.inp)
        for blk in self.blocks:
            bound *= blk.bound()
        bound *= self._lip_act * _layer_bound(self.out)
        if isinstance(self.out_gain, _ZeroGain):
            bound *= abs(float(self.out_gain.gain.detach()))
        return bound


# --------------------------------------------------------------------------- MADE
class _MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: Tensor):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask.to(torch.get_default_dtype()))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked autoregressive MLP (Germain et al., 2015).

    Output ``i`` (for each of the ``out_multiplier`` output blocks) depends only
    on the inputs that precede dimension ``i`` in ``order``; the first
    dimension in the order receives a constant (bias, or context-dependent)
    output. The context (if any) enters every hidden unit of the first layer
    through an unmasked linear map. The output layer is zero-initialized.

    Args:
        features: number of input dimensions ``D``.
        hidden: hidden widths (non-empty).
        context_dim: size of the context vector.
        activation: activation name or module.
        order: ``None`` (identity order), ``"reverse"``, ``"random"`` or a
            permutation (``order[k]`` is the dimension at position ``k``).
        seed: seed for ``order="random"``.
        out_multiplier: number of outputs per input dimension. The output has
            shape ``(B, out_multiplier * D)``, block ``j`` occupying
            ``[:, j*D:(j+1)*D]``.

    Buffers:
        order: the permutation. degrees: ``degrees[d] = position of d + 1``.
    """

    def __init__(
        self,
        features: int,
        hidden: Sequence[int] = (256, 256),
        context_dim: int = 0,
        activation: Union[str, nn.Module] = "relu",
        order: Union[None, str, Sequence[int], Tensor] = None,
        seed: int = 0,
        out_multiplier: int = 1,
    ):
        super().__init__()
        D = int(features)
        hidden = tuple(int(h) for h in hidden)
        if len(hidden) == 0 or any(h <= 0 for h in hidden):
            raise ValueError("hidden must be a non-empty sequence of positive ints")
        self.features = D
        self.context_dim = int(context_dim)
        self.out_multiplier = int(out_multiplier)

        if order is None:
            perm = torch.arange(D)
        elif isinstance(order, str):
            if order == "reverse":
                perm = torch.arange(D - 1, -1, -1)
            elif order == "random":
                g = torch.Generator().manual_seed(seed)
                perm = torch.randperm(D, generator=g)
            else:
                raise ValueError(f"Unknown order '{order}'")
        else:
            perm = torch.as_tensor(order, dtype=torch.long)
            if sorted(perm.tolist()) != list(range(D)):
                raise ValueError("order must be a permutation of range(features)")
        degrees = torch.empty(D, dtype=torch.long)
        degrees[perm] = torch.arange(1, D + 1)
        self.register_buffer("order", perm)
        self.register_buffer("degrees", degrees)

        act, _, pa = make_activation(activation)
        self.act = act
        self.piecewise_affine = bool(pa)

        max_deg = max(D - 1, 1)
        layers = []
        deg_prev = degrees
        for w in hidden:
            deg_h = torch.arange(w) % max_deg + 1
            mask = (deg_h[:, None] >= deg_prev[None, :])
            layers.append(_MaskedLinear(len(deg_prev), w, mask))
            deg_prev = deg_h
        self.hidden_layers = nn.ModuleList(layers)
        deg_out = degrees.repeat(self.out_multiplier)
        self.out = _MaskedLinear(len(deg_prev), D * self.out_multiplier, deg_out[:, None] > deg_prev[None, :])
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.ctx = nn.Linear(self.context_dim, hidden[0], bias=False) if self.context_dim > 0 else None

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        h = self.hidden_layers[0](x)
        ctx = _prepare_context(context, self.context_dim, x.shape[0], x)
        if ctx is not None:
            h = h + self.ctx(ctx)
        h = self.act(h)
        for layer in self.hidden_layers[1:]:
            h = self.act(layer(h))
        return self.out(h)

    def connectivity(self) -> Tensor:
        """Boolean ``(D * out_multiplier, D)`` matrix: may output ``i`` depend on input ``j``."""
        m = self.hidden_layers[0].mask
        for layer in list(self.hidden_layers[1:]) + [self.out]:
            m = layer.mask @ m
        return m > 0


# --------------------------------------------------------------------------- Jet / ViT
def sincos_pos_embed(embed_dim: int, grid_shape: Iterable[int]) -> Tensor:
    """Fixed sine-cosine positional embedding for an N-D token grid.

    The embedding dimension is split evenly across grid axes (sin/cos pairs);
    remainder dimensions are zero.

    Returns:
        Tensor of shape ``(prod(grid_shape), embed_dim)`` (float32).
    """
    grid_shape = tuple(int(s) for s in grid_shape)
    ndim = len(grid_shape)
    dim_per_axis = (embed_dim // ndim) // 2 * 2
    if dim_per_axis <= 0:
        raise ValueError(f"embed_dim={embed_dim} too small for {ndim} grid axes")
    coords = torch.meshgrid(*[torch.arange(s, dtype=torch.float64) for s in grid_shape], indexing="ij")
    embs = []
    omega = torch.arange(dim_per_axis // 2, dtype=torch.float64)
    omega = 1.0 / (10000.0 ** (omega / (dim_per_axis / 2)))
    for axis in range(ndim):
        angles = coords[axis].reshape(-1)[:, None] * omega[None, :]
        embs.append(torch.cat([angles.sin(), angles.cos()], dim=1))
    emb = torch.cat(embs, dim=1)
    if emb.shape[1] < embed_dim:
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], embed_dim - emb.shape[1])], dim=1)
    return emb.float()


class _TokenMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, activation: Optional[nn.Module] = None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), activation if activation is not None else nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _SelfAttention(nn.Module):
    """Multi-head self-attention with optional QK-RMSNorm, causal and key-padding masks."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, causal: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        else:
            mask = padding_mask.to(torch.bool)[:, None, None, :]
            if self.causal:
                mask = mask & torch.ones(N, N, dtype=torch.bool, device=x.device).tril()
            # every token may attend to itself (no fully-masked rows)
            mask = mask | torch.eye(N, dtype=torch.bool, device=x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))

    def step(self, x: Tensor, cache: dict) -> Tensor:
        """Incremental causal attention for new tokens ``x`` of shape ``(B, T, D)``.

        Keys and values of all earlier calls are kept in ``cache`` (a dict,
        empty on the first call), so the result equals rows ``P..P+T-1`` of a
        causal :meth:`forward` over the whole sequence (``P`` = cached length).
        """
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
        mask = None
        if T > 1:
            P = k.shape[2] - T
            mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(diagonal=P)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, T, D))

    def step_static(self, x: Tensor, k_buf: Tensor, v_buf: Tensor, pos: Tensor) -> Tensor:
        """Single-token causal attention with preallocated key/value buffers.

        ``x`` is ``(B, 1, D)``, ``k_buf``/``v_buf`` are zero-initialized
        ``(B, num_heads, L, head_dim)`` buffers and ``pos`` is a long tensor of
        shape ``(1,)`` with the position of ``x``. The new key/value are written
        into the buffers in place and the query attends to positions ``<= pos``.
        All shapes are fixed, so a decoding loop can be compiled or captured as
        a CUDA graph. In-place writes: use without autograd only.
        """
        B, _, D = x.shape
        q, k, v = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        k_buf.index_copy_(2, pos, k.to(k_buf.dtype))
        v_buf.index_copy_(2, pos, v.to(v_buf.dtype))
        mask = (torch.arange(k_buf.shape[2], device=x.device) <= pos).view(1, 1, 1, -1)
        out = F.scaled_dot_product_attention(q, k_buf.to(q.dtype), v_buf.to(q.dtype), attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, 1, D))


class _JetBlock(nn.Module):
    """Pre-LN ViT block with adaLN-zero modulation (gates start closed -> identity)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qk_norm=True, causal=False, activation=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = _SelfAttention(dim, num_heads, qk_norm=qk_norm, causal=causal)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = _TokenMLP(dim, int(dim * mlp_ratio), dim, activation)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: Tensor, c: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + sc1) + s1, padding_mask=padding_mask)
        return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + s2)

    def step(self, x: Tensor, c: Tensor, cache: dict) -> Tensor:
        """Incremental version of :meth:`forward` for causal blocks (see :meth:`_SelfAttention.step`)."""
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + g1 * self.attn.step(self.norm1(x) * (1 + sc1) + s1, cache)
        return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + s2)

    def modulation(self, c: Tensor) -> Tuple[Tensor, ...]:
        """The six adaLN modulation tensors for context embedding ``c`` (each ``(B, 1, dim)``)."""
        return self.ada(c).unsqueeze(1).chunk(6, dim=-1)

    def step_static(self, x: Tensor, mod: Tuple[Tensor, ...], k_buf: Tensor, v_buf: Tensor, pos: Tensor) -> Tensor:
        """Single-token step with precomputed modulation (see :meth:`_SelfAttention.step_static`)."""
        s1, sc1, g1, s2, sc2, g2 = mod
        x = x + g1 * self.attn.step_static(self.norm1(x) * (1 + sc1) + s1, k_buf, v_buf, pos)
        return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + s2)


class JetConditioner(nn.Module):
    """ViT conditioner (Jet, Kolesnikov et al. 2024) with adaLN-zero context.

    Pipeline: patchify (strided conv / token chunking) -> fixed sin-cos
    positional embedding -> ``depth`` pre-LN blocks with QK-norm attention and
    adaLN-zero context modulation -> LayerNorm -> zero-initialized token-wise
    head -> unpatchify. Output is exactly zero at initialization.

    Topologies:
        * vector, ``in_dims == [D]``: chunked into tokens of ``token_size``
          entries (zero-padded). Output ``(B, out_dim)`` (default ``D``).
        * spatial, ``in_dims == [C, *spatial]`` (1-3 spatial dims): patches of
          side ``patch_size`` (must divide all spatial dims). Output
          ``(B, out_dim, *spatial)`` (default ``out_dim = C``).

    If ``context`` is ``None`` (or ``context_dim == 0``) a learned null
    embedding is used. Not piecewise affine.

    Args:
        in_dims: input event shape.
        patch_size: patch side length (spatial inputs).
        embed_dim: token width.
        depth: number of blocks.
        num_heads: attention heads.
        mlp_ratio: token MLP hidden width / embed_dim.
        context_dim: context size (``0``: unconditional). Note: the legacy
            default was ``1``; ``context=None`` still works with any value.
        token_size: entries per token (vector inputs).
        qk_norm: RMSNorm on per-head queries/keys.
        causal: causal attention over the token order.
        nonlinearity: token MLP activation (default GELU).
        grad_checkpointing: recompute block activations in backward.
        out_dim: output features (vector) / channels (spatial). New in v1.
    """

    piecewise_affine = False

    def __init__(
        self,
        in_dims: Iterable[int],
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        context_dim: int = 0,
        token_size: int = 8,
        qk_norm: bool = True,
        causal: bool = False,
        nonlinearity: Optional[nn.Module] = None,
        grad_checkpointing: bool = False,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        in_dims = [int(d) for d in in_dims]
        if not 1 <= len(in_dims) <= 4:
            raise ValueError(f"Unsupported in_dims {in_dims}")
        self.in_dims = in_dims
        self.embed_dim = embed_dim
        self.context_dim = int(context_dim)
        self.grad_checkpointing = grad_checkpointing
        self.is_vector = len(in_dims) == 1

        if self.is_vector:
            d_in = in_dims[0]
            self.out_dim = int(out_dim) if out_dim is not None else d_in
            token_size = min(token_size, d_in)
            self.token_size = token_size
            n_tokens = math.ceil(d_in / token_size)
            self.n_pad = n_tokens * token_size - d_in
            self.out_token = math.ceil(self.out_dim / n_tokens)
            self.patch = nn.Linear(token_size, embed_dim)
            self.head = nn.Linear(embed_dim, self.out_token)
            grid_shape: Tuple[int, ...] = (n_tokens,)
        else:
            c_in, *spatial = in_dims
            rank = len(spatial)
            if any(s % patch_size for s in spatial):
                raise ValueError(f"patch_size {patch_size} must divide spatial dims {spatial}")
            self.out_dim = int(out_dim) if out_dim is not None else c_in
            self.patch_size = patch_size
            self.patch = _CONV[rank](c_in, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.head = nn.Linear(embed_dim, self.out_dim * patch_size**rank)
            grid_shape = tuple(s // patch_size for s in spatial)
        self.grid_shape = grid_shape
        self.register_buffer("pos_embed", sincos_pos_embed(embed_dim, grid_shape).unsqueeze(0))

        self.ctx_embed = _TokenMLP(self.context_dim, embed_dim, embed_dim) if self.context_dim > 0 else None
        self.null_ctx = nn.Parameter(torch.zeros(embed_dim))
        self.blocks = nn.ModuleList(
            _JetBlock(embed_dim, num_heads, mlp_ratio, qk_norm, causal, nonlinearity) for _ in range(depth)
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def n_tokens(self) -> int:
        return math.prod(self.grid_shape)

    def _embed_context(self, context: Optional[Tensor], B: int, ref: Tensor) -> Tensor:
        if context is None or self.ctx_embed is None:
            return self.null_ctx.to(ref.dtype).unsqueeze(0).expand(B, -1)
        return self.ctx_embed(_prepare_context(context, self.context_dim, B, ref))

    def _tokenize(self, x: Tensor) -> Tensor:
        if self.is_vector:
            if self.n_pad:
                x = F.pad(x, (0, self.n_pad))
            tokens = self.patch(x.reshape(x.shape[0], -1, self.token_size))
        else:
            tokens = self.patch(x).flatten(2).transpose(1, 2)
        return tokens + self.pos_embed.to(tokens.dtype)

    def _untokenize(self, out: Tensor, B: int) -> Tensor:
        if self.is_vector:
            return out.reshape(B, -1)[:, : self.out_dim]
        _, *spatial = self.in_dims
        rank = len(spatial)
        p = self.patch_size
        out = out.reshape(B, *self.grid_shape, self.out_dim, *([p] * rank))
        perm = [0, 1 + rank]
        for axis in range(rank):
            perm += [1 + axis, 2 + rank + axis]
        return out.permute(*perm).reshape(B, self.out_dim, *spatial)

    def forward(self, x: Tensor, context: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        """Args:
            x: ``(B, *in_dims)``.
            context: optional ``(B, context_dim)`` (or broadcastable single context).
            padding_mask: optional bool ``(B, n_tokens)``; True = valid token.
        """
        if self.is_vector and x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        B = x.shape[0]
        tokens = self._tokenize(x)
        c = self._embed_context(context, B, tokens)
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, c, padding_mask, use_reentrant=False)
            else:
                tokens = block(tokens, c, padding_mask=padding_mask)
        return self._untokenize(self.head(self.norm_out(tokens)), B)
