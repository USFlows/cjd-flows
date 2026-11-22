"""Shape-changing volume-preserving bijections (pure permutations / orthonormal maps).

All layers here change the event shape. They expose ``event_shape`` (input)
and ``out_shape`` (output) attributes; :meth:`inverse` maps back from
``out_shape``. Their Jacobian is a permutation or an orthonormal matrix, so
``log|det J| = 0``.

Note that :func:`cjd_flows.testing.check_bijection` asserts shape
preservation, so these layers are tested with dedicated checks.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from cjd_flows.core import ShapeLike, VolumePreserving, as_shape

__all__ = ["Squeeze", "HaarWavelet", "Patchify", "Flatten", "Reshape"]


def _check_divisible(spatial: Tuple[int, ...], f: int, name: str) -> None:
    if len(spatial) == 0:
        raise ValueError(f"{name} needs at least one spatial dimension")
    if any(s % f for s in spatial):
        raise ValueError(f"{name}: spatial dims {spatial} must be divisible by {f}")


def _split_blocks(x: Tensor, f: int) -> Tensor:
    """``(B, C, S1, ..., Sk) -> (B, C, f, ..., f, S1/f, ..., Sk/f)``."""
    B, C = x.shape[:2]
    spatial = x.shape[2:]
    k = len(spatial)
    shape: List[int] = [B, C]
    for s in spatial:
        shape += [s // f, f]
    x = x.reshape(shape)
    # positions of the block offsets and of the coarse coordinates
    offsets = [3 + 2 * i for i in range(k)]
    coarse = [2 + 2 * i for i in range(k)]
    return x.permute(0, 1, *offsets, *coarse)


def _merge_blocks(x: Tensor, f: int, k: int) -> Tensor:
    """Inverse of :func:`_split_blocks`: ``(B, C, f..f, S/f..) -> (B, C, S1, ..., Sk)``."""
    B, C = x.shape[:2]
    coarse = x.shape[2 + k :]
    perm = [0, 1]
    for i in range(k):
        perm += [2 + k + i, 2 + i]
    x = x.permute(perm)
    return x.reshape(B, C, *[s * f for s in coarse])


class _ShapeChanging(VolumePreserving):
    piecewise_affine = True

    def __init__(self, event_shape: ShapeLike):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        self.register_buffer("_ref", torch.zeros(()), persistent=False)

    out_shape: Tuple[int, ...]

    def extra_repr(self) -> str:
        return f"event_shape={self.event_shape}, out_shape={self.out_shape}"


class Squeeze(_ShapeChanging):
    """Space-to-depth: ``(C, S1, ..., Sk) -> (C·f^k, S1/f, ..., Sk/f)`` for ``k = 1..3``.

    Output channel ``c·f^k + o`` holds input channel ``c`` at block offset ``o``
    (offsets enumerated row-major), as in Glow/RealNVP.

    Args:
        event_shape: ``(C, *spatial)``.
        factor: block size ``f``.
    """

    def __init__(self, event_shape: ShapeLike, factor: int = 2):
        super().__init__(event_shape)
        C, *spatial = self.event_shape
        _check_divisible(tuple(spatial), factor, "Squeeze")
        self.factor = factor
        self.k = len(spatial)
        self.out_shape = (C * factor**self.k, *[s // factor for s in spatial])

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return _split_blocks(x, self.factor).reshape(x.shape[0], *self.out_shape)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        B = z.shape[0]
        C = self.event_shape[0]
        z = z.reshape(B, C, *([self.factor] * self.k), *self.out_shape[1:])
        return _merge_blocks(z, self.factor, self.k)


class HaarWavelet(_ShapeChanging):
    """One level of the orthonormal Haar wavelet transform as space-to-depth.

    ``(C, S1, ..., Sk) -> (2^k·C, S1/2, ..., Sk/2)``. Each ``2×…×2`` block is
    multiplied by ``H_k = H ⊗ … ⊗ H`` with ``H = [[1, 1], [1, -1]] / √2``.
    The output is **subband-major**: channels ``[0, C)`` are the low-pass
    (approximation, scaled average ``2^{k/2}·mean``) coefficients, followed by
    the ``2^k - 1`` detail subbands of ``C`` channels each. ``|det| = 1``.

    Args:
        event_shape: ``(C, *spatial)`` with even spatial sizes.
    """

    def __init__(self, event_shape: ShapeLike):
        super().__init__(event_shape)
        C, *spatial = self.event_shape
        _check_divisible(tuple(spatial), 2, "HaarWavelet")
        self.k = len(spatial)
        self.out_shape = (C * 2**self.k, *[s // 2 for s in spatial])

    def _butterfly(self, blocks: Tensor) -> Tensor:
        """Applies ``H ⊗ … ⊗ H`` on the ``k`` offset axes (dims ``2..1+k``) of ``blocks``.

        Uses exact sum/difference butterflies (no stored matrix), so the map is
        orthonormal to the precision of the input dtype.
        """
        r = 1.0 / math.sqrt(2.0)
        for i in range(self.k):
            a, b = blocks.unbind(2 + i)
            blocks = torch.stack(((a + b) * r, (a - b) * r), dim=2 + i)
        return blocks

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        B, C = x.shape[:2]
        n = 2**self.k
        blocks = self._butterfly(_split_blocks(x, 2))  # (B, C, 2..2, coarse)
        y = blocks.reshape(B, C, n, -1).transpose(1, 2)  # subband-major
        return y.reshape(B, *self.out_shape)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        B = z.shape[0]
        C = self.event_shape[0]
        n = 2**self.k
        x = z.reshape(B, n, C, -1).transpose(1, 2)
        x = x.reshape(B, C, *([2] * self.k), *self.out_shape[1:])
        # H ⊗ … ⊗ H is symmetric and orthogonal, hence self-inverse
        return _merge_blocks(self._butterfly(x), 2, self.k)


class Patchify(_ShapeChanging):
    """Token layout ``(C, S1, ..., Sk) -> (N_tokens, p^k·C)`` for transformer conditioners.

    ``N_tokens = prod(S_i / p)`` (row-major over patch positions) and each
    token's features are ordered ``(p, …, p, C)`` (channels last), i.e.
    ``b c (h p1) (w p2) -> b (h w) (p1 p2 c)``. A pure permutation.

    Note that after this layer dim 1 indexes *tokens*, so channel-wise layers
    applied afterwards would mix tokens.

    Args:
        event_shape: ``(C, *spatial)``.
        patch_size: patch side length ``p``.
    """

    def __init__(self, event_shape: ShapeLike, patch_size: int):
        super().__init__(event_shape)
        C, *spatial = self.event_shape
        _check_divisible(tuple(spatial), patch_size, "Patchify")
        self.patch_size = patch_size
        self.k = len(spatial)
        self.grid = tuple(s // patch_size for s in spatial)
        self.out_shape = (math.prod(self.grid), patch_size**self.k * C)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        B = x.shape[0]
        k = self.k
        blocks = _split_blocks(x, self.patch_size)  # (B, C, p.., grid..)
        perm = [0, *range(2 + k, 2 + 2 * k), *range(2, 2 + k), 1]  # (B, grid.., p.., C)
        return blocks.permute(perm).reshape(B, *self.out_shape)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        B = z.shape[0]
        k = self.k
        C = self.event_shape[0]
        p = self.patch_size
        z = z.reshape(B, *self.grid, *([p] * k), C)
        perm = [0, 1 + 2 * k, *range(1 + k, 1 + 2 * k), *range(1, 1 + k)]  # (B, C, p.., grid..)
        return _merge_blocks(z.permute(perm), p, k)


class Reshape(_ShapeChanging):
    """Reshapes events from ``in_shape`` to ``out_shape`` (row-major)."""

    def __init__(self, in_shape: ShapeLike, out_shape: ShapeLike):
        super().__init__(in_shape)
        self.out_shape = as_shape(out_shape)
        if math.prod(self.out_shape) != math.prod(self.event_shape):
            raise ValueError(f"cannot reshape {self.event_shape} to {self.out_shape}")

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return x.reshape(x.shape[0], *self.out_shape)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return z.reshape(z.shape[0], *self.event_shape)


class Flatten(Reshape):
    """Flattens events ``(C, *spatial) -> (prod,)``."""

    def __init__(self, event_shape: ShapeLike):
        shape = as_shape(event_shape)
        super().__init__(shape, (math.prod(shape),))
