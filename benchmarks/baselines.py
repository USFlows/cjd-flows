"""Reference models without a constant Jacobian determinant (benchmark-only).

They violate the library contract (``log_abs_det`` is not constant), so they
live here and not in :mod:`cjd_flows`. They share the trunk of
:func:`cjd_flows.models.USFlow` so that benchmarks measure what the constant
determinant costs, not differences in architecture or training.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor

from cjd_flows.bijections.coupling import _SplitCoupling
from cjd_flows.bijections.linear import GlobalScale, LULinear, Shift
from cjd_flows.core import Chain, Flow, ShapeLike, as_shape

__all__ = ["AffineCoupling", "AffineJetCoupling", "VariableDetFlow", "affine_reference_flow"]


class AffineCoupling(_SplitCoupling):
    """Jet-style affine coupling ``y_b = x_b * 2 sigmoid(s) + t`` (non-constant det)."""

    _out_factor = 2
    piecewise_affine = False
    volume_preserving = False

    def _st(self, params: Tensor):
        s, t = params.chunk(2, dim=1)
        s = s.float() if s.dtype in (torch.float16, torch.bfloat16) else s  # >= fp32
        log_scale = torch.nn.functional.logsigmoid(s) + math.log(2.0)
        return log_scale, t

    def forward_with_logdet(self, x: Tensor, context: Optional[Tensor] = None):
        xa, xb = self._split(x)
        log_scale, t = self._st(self._params(xa, context))
        yb = xb * log_scale.exp().to(xb.dtype) + t
        return self._merge(xa, yb), log_scale.flatten(1).sum(1)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return self.forward_with_logdet(x, context)[0]

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        za, zb = self._split(z)
        log_scale, t = self._st(self._params(za, context))
        return self._merge(za, (zb - t) * (-log_scale).exp().to(zb.dtype))


#: Name used by the CIFAR study.
AffineJetCoupling = AffineCoupling


class VariableDetFlow(Flow):
    """Flow that also accumulates per-sample log-determinants (for the affine reference)."""

    def log_prob(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        ctx = self._context(context, x.shape[0], x)
        ld = x.new_zeros(x.shape[0], dtype=torch.promote_types(x.dtype, torch.float32))
        z = x
        for b in self.bijection:
            if hasattr(b, "forward_with_logdet"):
                z, ldj = b.forward_with_logdet(z, ctx)
                ld = ld + ldj
            else:
                z = b(z, ctx)
                ld = ld + b.log_abs_det()
        return self.base.log_prob(z) + ld  # keeps the >= fp32 accumulator (type promotion)


def affine_reference_flow(
    event_shape: ShapeLike,
    *,
    blocks: int = 8,
    hidden: Sequence[int] = (256, 256),
    activation: str = "relu",
    mixing: Optional[str] = "lu",
    base="gaussian",
    p: float = 2.0,
    n_components: int = 4,
) -> VariableDetFlow:
    """:func:`~cjd_flows.models.USFlow` trunk with affine instead of additive couplings.

    ``[mix_i, affine coupling_i]`` blocks with alternating parity, then a
    final LU layer, a data-initialized :class:`Shift` and :class:`GlobalScale`,
    exactly as in ``USFlow`` (vector or image event shapes).
    """
    from cjd_flows.models import _conditioner_factory, _mixing_layer, make_base

    shape = as_shape(event_shape)
    spatial = len(shape) > 1
    layers = []
    for i in range(blocks):
        mix = _mixing_layer(mixing, shape)
        if mix is not None:
            layers.append(mix)
        split = ("channel" if (shape[0] > 1 and i % 2 == 0) else "checkerboard") if spatial else "channel"
        fac = _conditioner_factory(shape, "auto", tuple(hidden), activation, 0)
        layers.append(AffineCoupling(shape, fac, split=split, parity=i % 2))
    layers += [LULinear(shape), Shift(shape), GlobalScale(shape)]
    return VariableDetFlow(Chain(*layers), make_base(base, shape, p=p, n_components=n_components))
