"""Data preprocessing with tracked log-determinants.

CJD flows have a constant Jacobian determinant. To report likelihoods in the
*original* data space (e.g. bits/dim on 8-bit images) the change of variables
of the preprocessing must be accounted for. All preprocessing steps here record
``log|det ∂y/∂x|`` per sample:

* affine steps (:class:`AffineScale`, :class:`ToUnitInterval`,
  :class:`ToSymmetric`, :class:`Standardize`) have a **constant** log-det and
  therefore preserve the CJD property of the overall model
  ``x -> preprocessing -> flow``;
* :class:`Logit` is **not** constant-det (``constant_det = False``): composing it
  with a CJD flow yields a model whose density level sets are no longer
  pre-images of base level sets. Use it only if you know what you are doing.

Bits/dim convention (:func:`bits_per_dim`): 8-bit data ``x ∈ {0,…,255}^D`` is
dequantized to ``x̃ = x + u ∈ [0, 256)^D`` with ``u ~ U[0,1)^D``. The model sees
``y = g(x̃)`` and

    bpd = (−log p_model(y) − log|det ∂y/∂x̃|) / (D ln 2),

which is the negative (variational lower bound of the) discrete
log-likelihood in bits per dimension.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

Number = Union[float, int]


def _event_size(x_or_shape) -> int:
    if isinstance(x_or_shape, Tensor):
        return math.prod(x_or_shape.shape[1:])
    if isinstance(x_or_shape, int):
        return x_or_shape
    return math.prod(x_or_shape)


class Preprocessing:
    """Base class of a (fixed, non-learnable) preprocessing map ``y = g(x)``.

    Attributes:
        constant_det: whether ``log|det ∂g/∂x|`` is independent of ``x``.
    """

    constant_det: bool = True

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse(self, y: Tensor) -> Tensor:
        raise NotImplementedError

    def log_abs_det(self, x: Tensor) -> Tensor:
        """Per-sample ``log|det ∂g/∂x|`` evaluated at the *input* batch ``x`` (shape ``(B,)``)."""
        return x.new_full((x.shape[0],), self.constant_log_abs_det(x.shape[1:]))

    def constant_log_abs_det(self, event_shape: Union[int, Sequence[int]]) -> float:
        """Per-sample log-det for constant-det maps (raises otherwise)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


class UniformDequantize(Preprocessing):
    """Adds fresh uniform noise ``u ~ U[0,1)`` to integer data.

    ``x ∈ {0, …, n_bins−1}`` becomes ``x̃ = x + u ∈ [0, n_bins)``. The map is
    stochastic but volume preserving (log-det 0); the inverse is ``floor``.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins = int(n_bins)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.float() if not x.is_floating_point() else x
        return x + torch.rand_like(x)

    def inverse(self, y: Tensor) -> Tensor:
        return y.floor().clamp(0, self.n_bins - 1)

    def constant_log_abs_det(self, event_shape) -> float:
        return 0.0


class AffineScale(Preprocessing):
    """``y = scale · x + shift`` with a scalar (or broadcastable) ``scale``."""

    def __init__(self, scale: Union[Number, Tensor], shift: Union[Number, Tensor] = 0.0):
        self.scale = scale
        self.shift = shift

    def __call__(self, x: Tensor) -> Tensor:
        x = x.float() if not x.is_floating_point() else x
        return x * _as(self.scale, x) + _as(self.shift, x)

    def inverse(self, y: Tensor) -> Tensor:
        return (y - _as(self.shift, y)) / _as(self.scale, y)

    def constant_log_abs_det(self, event_shape) -> float:
        d = _event_size(event_shape)
        if isinstance(self.scale, Tensor) and self.scale.numel() > 1:
            shape = (event_shape,) if isinstance(event_shape, int) else tuple(event_shape)
            return float(self.scale.detach().double().abs().log().expand(shape).sum())
        return d * math.log(abs(float(self.scale)))


class ToUnitInterval(AffineScale):
    """Maps ``[0, n_bins)`` to ``[0, 1)``: ``y = x / n_bins`` (log-det ``−D log n_bins``)."""

    def __init__(self, n_bins: int = 256):
        super().__init__(1.0 / n_bins, 0.0)
        self.n_bins = int(n_bins)


class ToSymmetric(AffineScale):
    """Maps ``[0, n_bins)`` to ``[−1, 1)``: ``y = 2x / n_bins − 1`` (log-det ``D log(2/n_bins)``)."""

    def __init__(self, n_bins: int = 256):
        super().__init__(2.0 / n_bins, -1.0)
        self.n_bins = int(n_bins)


def to_unit_interval(x: Tensor, n_bins: int = 256) -> Tensor:
    """``x / n_bins``; per-sample log-det ``−D log n_bins``."""
    return ToUnitInterval(n_bins)(x)


def to_symmetric(x: Tensor, n_bins: int = 256) -> Tensor:
    """``2x / n_bins − 1``; per-sample log-det ``D (log 2 − log n_bins)``."""
    return ToSymmetric(n_bins)(x)


class Logit(Preprocessing):
    """RealNVP-style logit transform ``y = logit(α + (1−2α) u)`` for ``u ∈ [0, 1]``.

    Warning:
        NOT constant-det (``constant_det = False``). Combined with a CJD flow
        the overall model loses the density level-set preservation property.
    """

    constant_det = False

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)

    def __call__(self, u: Tensor) -> Tensor:
        s = self.alpha + (1 - 2 * self.alpha) * u
        return torch.log(s) - torch.log1p(-s)

    def inverse(self, y: Tensor) -> Tensor:
        return (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)

    def log_abs_det(self, u: Tensor) -> Tensor:
        s = self.alpha + (1 - 2 * self.alpha) * u
        ld = math.log(1 - 2 * self.alpha) - torch.log(s) - torch.log1p(-s)
        return ld.flatten(1).sum(-1)

    def constant_log_abs_det(self, event_shape) -> float:
        raise ValueError("Logit has an input-dependent log-determinant; use log_abs_det(x).")


class Standardize(Preprocessing):
    """Feature-wise standardization ``y = (x − mean) / std`` (constant log-det ``−Σ log std``).

    Args:
        mean, std: statistics over the event shape; use :meth:`fit` to estimate them.
        eps: lower bound for ``std``.
    """

    def __init__(self, mean: Optional[Tensor] = None, std: Optional[Tensor] = None, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, x: Tensor, unbiased: bool = False) -> "Standardize":
        x = x.double()
        self.mean = x.mean(0).float()
        self.std = x.std(0, unbiased=unbiased).clamp_min(self.eps).float()
        return self

    def _check(self):
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardize is not fitted; call fit(x) first.")

    def __call__(self, x: Tensor) -> Tensor:
        self._check()
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse(self, y: Tensor) -> Tensor:
        self._check()
        return y * self.std.to(y) + self.mean.to(y)

    def constant_log_abs_det(self, event_shape=None) -> float:
        self._check()
        return -float(self.std.double().log().sum())


class Compose(Preprocessing):
    """Sequential composition; log-dets add up."""

    def __init__(self, *steps: Preprocessing):
        self.steps = list(steps)

    @property
    def constant_det(self) -> bool:  # type: ignore[override]
        return all(s.constant_det for s in self.steps)

    def __call__(self, x: Tensor) -> Tensor:
        for s in self.steps:
            x = s(x)
        return x

    def inverse(self, y: Tensor) -> Tensor:
        for s in reversed(self.steps):
            y = s.inverse(y)
        return y

    def log_abs_det(self, x: Tensor) -> Tensor:
        total = x.new_zeros(x.shape[0], dtype=torch.float32 if not x.is_floating_point() else x.dtype)
        for s in self.steps:
            total = total + s.log_abs_det(x)
            x = s(x)
        return total

    def constant_log_abs_det(self, event_shape) -> float:
        return sum(s.constant_log_abs_det(event_shape) for s in self.steps)


def bits_per_dim(
    log_prob: Union[Tensor, float],
    dim: int,
    preprocessing_log_det: Union[Tensor, float, None] = None,
    n_bins: int = 256,
) -> Union[Tensor, float]:
    """Converts model log-likelihoods to bits/dim in the dequantized data space.

    Args:
        log_prob: ``log p_model(y)`` in model space (nats), per sample or averaged.
        dim: number of dimensions ``D``.
        preprocessing_log_det: ``log|det ∂y/∂x̃|`` of the preprocessing from the
            dequantized raw data ``x̃ ∈ [0, n_bins)^D`` to model space ``y``
            (per sample). ``None`` means the model data is ``x̃ / n_bins ∈ [0,1)^D``,
            i.e. ``−D log n_bins``.
        n_bins: number of quantization levels (only used when
            ``preprocessing_log_det`` is ``None``).

    Returns:
        ``(−log p(y) − log|det ∂y/∂x̃|) / (D ln 2)``. Equivalently, if ``L``
        denotes the log-det relative to the ``[0, 1)`` scaling,
        ``(−log p(y) − L + D log n_bins) / (D ln 2)``.

    Example:
        The uniform density on ``[0, 256)^D`` (identity preprocessing) has
        ``log p = −D log 256`` and gives exactly 8 bits/dim.
    """
    if preprocessing_log_det is None:
        preprocessing_log_det = -dim * math.log(n_bins)
    return (-log_prob - preprocessing_log_det) / (dim * math.log(2.0))


def _as(v, ref: Tensor):
    return v.to(ref) if isinstance(v, Tensor) else v
