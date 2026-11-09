"""Volume-preserving coupling layers.

* :class:`MaskedAdditiveCoupling` -- ``y = x + (1 - m) * t(m * x)`` with a
  binary mask buffer (checkerboard / channel / random / custom).
* :class:`SplitAdditiveCoupling` -- Jet-style pairing: the input is split by
  index buffers into a conditioning half ``x_a`` and a transformed half
  ``x_b``; ``y_b = x_b + t(x_a)``. No compute is wasted on masked zeros.
* :class:`GINCoupling` -- same split, but affine with zero-sum log-scale
  (Sorrenson et al., 2020): ``y_b = x_b * exp(s - mean(s)) + t``. Still
  volume preserving, but not piecewise affine.

All layers have ``log|det J| = 0``. Conditioners are called as
``conditioner(x_a, context)`` (``context`` is only passed when not ``None``)
and should be zero-initialized (all conditioners in
:mod:`cjd_flows.conditioners` are), so that every layer starts as the identity.

Mask convention: ``1`` = conditioning (passed through), ``0`` = transformed.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from cjd_flows.core import ShapeLike, VolumePreserving, as_shape

__all__ = [
    "checkerboard_mask",
    "channel_mask",
    "random_mask",
    "split_indices",
    "MaskedAdditiveCoupling",
    "SplitAdditiveCoupling",
    "GINCoupling",
]

ConditionerFactory = Callable[[Tuple[int, ...], Tuple[int, ...]], nn.Module]


# --------------------------------------------------------------------------- masks
def checkerboard_mask(event_shape: ShapeLike, parity: int = 0) -> Tensor:
    """Checkerboard mask over the spatial dims (identical for every channel).

    For vectors ``(D,)`` the mask alternates over the dimensions
    (``[1, 0, 1, 0, ...]`` for ``parity=0``), as in the legacy code.

    Returns:
        Float tensor of ``event_shape`` with ones on the conditioning positions.
    """
    shape = as_shape(event_shape)
    if len(shape) == 1:
        return ((torch.arange(shape[0]) + parity) % 2 == 0).to(torch.get_default_dtype())
    spatial = shape[1:]
    coords = torch.meshgrid(*[torch.arange(s) for s in spatial], indexing="ij")
    m = (sum(coords) + parity) % 2 == 0
    return m.unsqueeze(0).expand(shape).to(torch.get_default_dtype()).contiguous()


def channel_mask(event_shape: ShapeLike, parity: int = 0) -> Tensor:
    """Mask selecting the first ``C // 2`` channels (``parity=0``) or the rest (``parity=1``).

    For vectors, the channel dimension is ``D``.

    Raises:
        ValueError: if the event has a single channel.
    """
    shape = as_shape(event_shape)
    C = shape[0]
    if C < 2:
        raise ValueError(
            f"channel_mask needs at least 2 channels, got event_shape={shape}. "
            "Use checkerboard_mask or squeeze the input first."
        )
    m = (torch.arange(C) < C // 2)
    if parity % 2:
        m = ~m
    m = m.reshape(C, *([1] * (len(shape) - 1))).expand(shape)
    return m.to(torch.get_default_dtype()).contiguous()


def random_mask(event_shape: ShapeLike, seed: int = 0) -> Tensor:
    """Random mask with ``floor(n / 2)`` ones (reproducible via ``seed``)."""
    shape = as_shape(event_shape)
    n = math.prod(shape)
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(n)
    m[torch.randperm(n, generator=g)[: n // 2]] = 1.0
    return m.reshape(shape).to(torch.get_default_dtype())


def _resolve_mask(event_shape: Tuple[int, ...], mask: Union[str, Tensor], parity: int, seed: int) -> Tensor:
    if isinstance(mask, str):
        if mask == "checkerboard":
            return checkerboard_mask(event_shape, parity)
        if mask == "channel":
            return channel_mask(event_shape, parity)
        if mask == "random":
            return random_mask(event_shape, seed)
        raise ValueError(f"Unknown mask '{mask}'")
    mask = torch.as_tensor(mask)
    if tuple(mask.shape) != event_shape:
        mask = mask.expand(event_shape)
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask must be binary")
    return mask.to(torch.get_default_dtype()).contiguous()


def _call(conditioner: nn.Module, x: Tensor, context: Optional[Tensor]) -> Tensor:
    return conditioner(x) if context is None else conditioner(x, context)


# --------------------------------------------------------------------------- masked
class MaskedAdditiveCoupling(VolumePreserving):
    """Additive masked coupling ``y = x + (1 - m) * t(m * x, context)``.

    Args:
        event_shape: event shape of the input.
        conditioner: module mapping ``(B, *event_shape)`` -> ``(B, *event_shape)``.
        mask: binary tensor of ``event_shape`` (1 = conditioning) or one of
            ``"checkerboard"``, ``"channel"``, ``"random"``.
        parity: parity for string masks (``"checkerboard"``, ``"channel"``).
        seed: seed for ``mask="random"``.

    ``piecewise_affine`` follows the conditioner.
    """

    def __init__(
        self,
        event_shape: ShapeLike,
        conditioner: nn.Module,
        mask: Union[str, Tensor] = "checkerboard",
        parity: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        self.conditioner = conditioner
        self.register_buffer("mask", _resolve_mask(self.event_shape, mask, parity, seed))
        self.piecewise_affine = bool(getattr(conditioner, "piecewise_affine", False))

    def _shift(self, x: Tensor, context: Optional[Tensor]) -> Tensor:
        m = self.mask.to(x.dtype)
        return (1 - m) * _call(self.conditioner, x * m, context)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return x + self._shift(x, context)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return z - self._shift(z, context)


# --------------------------------------------------------------------------- split
def split_indices(
    event_shape: ShapeLike,
    split: Union[str, Tensor] = "channel",
    parity: int = 0,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, Tuple[int, ...], Tuple[int, ...]]:
    """Computes flat index sets for a two-way split of an event.

    Args:
        event_shape: ``(D,)`` or ``(C, *spatial)``.
        split: one of

            * ``"channel"`` -- first ``C // 2`` channels vs. the rest (for
              vectors: first ``D // 2`` dims vs. the rest; alias ``"half"``),
            * ``"interleaved"`` (vectors) / ``"checkerboard"`` -- even vs. odd
              dims (vectors) or checkerboard over the spatial dims; the
              last spatial dim must be even and the halves have shape
              ``(C, *spatial[:-1], spatial[-1] // 2)``,
            * ``"spatial-row"`` -- even/odd indices along the *first* spatial axis,
            * ``"spatial-col"`` -- even/odd indices along the *last* spatial axis,
            * ``"random"`` -- a random half (seeded); halves are flat vectors,
            * a boolean tensor of ``event_shape`` (True = conditioning half);
              halves are flat vectors.
        parity: ``1`` swaps the roles of the two halves.
        seed: seed for ``split="random"``.

    Returns:
        ``(idx_a, idx_b, shape_a, shape_b)``: flat indices into the flattened
        event for the conditioning half ``a`` and the transformed half ``b``
        and the event shapes of the halves.
    """
    shape = as_shape(event_shape)
    n = math.prod(shape)
    flat = torch.arange(n).reshape(shape)

    def _bool_split(sel: Tensor, shp_a, shp_b):
        return flat[sel].reshape(-1), flat[~sel].reshape(-1), shp_a, shp_b

    if isinstance(split, Tensor):
        sel = split.to(torch.bool).expand(shape)
        na = int(sel.sum())
        if na == 0 or na == n:
            raise ValueError("custom split must select a proper, non-empty subset")
        ia, ib, sa, sb = _bool_split(sel, (na,), (n - na,))
    elif split == "random":
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        na = n // 2
        if na == 0:
            raise ValueError("event too small to split")
        ia, ib, sa, sb = perm[:na].sort().values, perm[na:].sort().values, (na,), (n - na,)
    elif len(shape) == 1:
        D = shape[0]
        if D < 2:
            raise ValueError("cannot split a 1-dimensional event")
        if split in ("channel", "half"):
            sel = torch.arange(D) < D // 2
        elif split in ("interleaved", "checkerboard"):
            sel = torch.arange(D) % 2 == 0
        else:
            raise ValueError(f"split '{split}' is not available for vector events")
        na = int(sel.sum())
        ia, ib, sa, sb = _bool_split(sel, (na,), (D - na,))
    else:
        C, *spatial = shape
        if split in ("channel", "half"):
            if C < 2:
                raise ValueError(
                    f"channel split needs at least 2 channels, got event_shape={shape}; "
                    "use a spatial split or squeeze the input first"
                )
            ca = C // 2
            sel = (torch.arange(C) < ca).reshape(C, *([1] * len(spatial))).expand(shape)
            ia, ib, sa, sb = _bool_split(sel, (ca, *spatial), (C - ca, *spatial))
        elif split in ("spatial-row", "spatial-col", "checkerboard", "interleaved"):
            axis = 0 if split == "spatial-row" else len(spatial) - 1
            if spatial[axis] % 2:
                raise ValueError(f"split '{split}' needs an even size along spatial axis {axis}, got {spatial}")
            coords = torch.meshgrid(*[torch.arange(s) for s in spatial], indexing="ij")
            if split in ("checkerboard", "interleaved"):
                sel = sum(coords) % 2 == 0
            else:
                sel = coords[axis] % 2 == 0
            sel = sel.unsqueeze(0).expand(shape)
            half = list(spatial)
            half[axis] //= 2
            ia, ib, sa, sb = _bool_split(sel, (C, *half), (C, *half))
        else:
            raise ValueError(f"Unknown split '{split}'")
    if parity % 2:
        ia, ib, sa, sb = ib, ia, sb, sa
    return ia, ib, tuple(sa), tuple(sb)


class _SplitCoupling(VolumePreserving):
    """Shared machinery of split couplings (gather halves via index buffers)."""

    _out_factor = 1

    def __init__(
        self,
        event_shape: ShapeLike,
        conditioner: Union[nn.Module, ConditionerFactory],
        split: Union[str, Tensor] = "channel",
        parity: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        ia, ib, sa, sb = split_indices(self.event_shape, split, parity, seed)
        self.shape_a, self.shape_b = sa, sb
        self.out_shape = (self._out_factor * sb[0], *sb[1:])
        perm = torch.cat([ia, ib])
        self.register_buffer("idx_a", ia)
        self.register_buffer("idx_b", ib)
        self.register_buffer("inv_perm", torch.argsort(perm))
        if not isinstance(conditioner, nn.Module):
            conditioner = conditioner(sa, self.out_shape)
        self.conditioner = conditioner

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]
        xf = x.reshape(B, -1)
        xa = xf.index_select(1, self.idx_a).reshape(B, *self.shape_a)
        xb = xf.index_select(1, self.idx_b).reshape(B, *self.shape_b)
        return xa, xb

    def _merge(self, xa: Tensor, yb: Tensor) -> Tensor:
        B = xa.shape[0]
        y = torch.cat([xa.reshape(B, -1), yb.reshape(B, -1)], dim=1).index_select(1, self.inv_perm)
        return y.reshape(B, *self.event_shape)

    def _params(self, xa: Tensor, context: Optional[Tensor]) -> Tensor:
        out = _call(self.conditioner, xa, context)
        B = xa.shape[0]
        if tuple(out.shape[1:]) != self.out_shape:
            out = out.reshape(B, *self.out_shape)
        return out

    def _transform_b(self, xb: Tensor, params: Tensor, inverse: bool) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        xa, xb = self._split(x)
        return self._merge(xa, self._transform_b(xb, self._params(xa, context), inverse=False))

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        za, zb = self._split(z)
        return self._merge(za, self._transform_b(zb, self._params(za, context), inverse=True))


class SplitAdditiveCoupling(_SplitCoupling):
    """Additive split coupling ``y_a = x_a``, ``y_b = x_b + t(x_a, context)``.

    Args:
        event_shape: input event shape.
        conditioner: a module mapping ``(B, *shape_a) -> (B, *shape_b)``, or a
            factory ``f(shape_a, shape_b) -> module`` (shapes are event shapes,
            channel first), e.g.
            ``lambda i, o: ConvNet(i[0], o[0], spatial_ndim=len(i) - 1)`` or
            ``lambda i, o: MLP(i, o)``.
        split: see :func:`split_indices`.
        parity: ``1`` swaps the halves.
        seed: seed for ``split="random"``.

    Attributes:
        shape_a, shape_b: event shapes of the conditioning / transformed half.
    """

    def __init__(self, event_shape, conditioner, split="channel", parity=0, seed=0):
        super().__init__(event_shape, conditioner, split, parity, seed)
        self.piecewise_affine = bool(getattr(self.conditioner, "piecewise_affine", False))

    def _transform_b(self, xb: Tensor, t: Tensor, inverse: bool) -> Tensor:
        return xb - t if inverse else xb + t


class GINCoupling(_SplitCoupling):
    """Volume-preserving affine split coupling (GIN, Sorrenson et al. 2020).

    ``y_b = x_b * exp(s - mean(s)) + t`` with ``[s, t] = conditioner(x_a)``
    split along dim 1 and ``s`` bounded by ``clamp * tanh(s / clamp)``. The
    mean is taken over all transformed dimensions of each sample, so the
    log-scales sum to zero and ``log|det J| = 0``.

    Args:
        event_shape: input event shape.
        conditioner: module mapping ``(B, *shape_a) -> (B, 2 * C_b, *spatial_b)``
            (vectors: ``(B, 2 * D_b)``), or a factory ``f(shape_a, out_shape)``
            where ``out_shape`` already has the doubled channel count.
        split, parity, seed: see :func:`split_indices`.
        clamp: soft bound of the raw log-scale (``None`` disables it).
    """

    _out_factor = 2
    piecewise_affine = False

    def __init__(self, event_shape, conditioner, split="channel", parity=0, seed=0, clamp: Optional[float] = 2.0):
        super().__init__(event_shape, conditioner, split, parity, seed)
        self.clamp = clamp
        self.piecewise_affine = False

    def _scale_shift(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        s, t = params.chunk(2, dim=1)
        if self.clamp is not None:
            s = self.clamp * torch.tanh(s / self.clamp)
        B = s.shape[0]
        s = s - s.reshape(B, -1).mean(dim=1).reshape(B, *([1] * (s.dim() - 1)))
        return s, t

    def _transform_b(self, xb: Tensor, params: Tensor, inverse: bool) -> Tensor:
        s, t = self._scale_shift(params)
        if inverse:
            return (xb - t) * torch.exp(-s)
        return xb * torch.exp(s) + t
