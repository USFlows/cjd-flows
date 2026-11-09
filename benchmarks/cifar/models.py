"""Model zoo for the CIFAR-10 scaling study.

Three variants with identical trunk (Squeeze x2 -> L split couplings with
Jet-style ViT conditioners, 4:1 channel:spatial split ratio, random channel
permutations before channel couplings):

* ``us-gauss``  : additive (volume-preserving) couplings + GlobalScale, standard normal base
* ``us-radial`` : same flow, learnable radial base distribution
* ``affine``    : non-uniformly-scaling baseline: Jet-style affine couplings
                  with bounded scale ``2*sigmoid(s)`` + GlobalScale, Gaussian base

The affine baseline is NOT a constant-Jacobian flow and therefore lives here,
not in the library.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from cjd_flows.bijections.coupling import SplitAdditiveCoupling, _SplitCoupling
from cjd_flows.bijections.linear import GlobalScale, Permutation
from cjd_flows.bijections.reshape import Squeeze
from cjd_flows.conditioners import JetConditioner
from cjd_flows.core import Chain, Flow, StandardNormal

SIZES = {
    # name: (embed_dim, depth, n_couplings)
    "tiny": (128, 1, 4),
    "small": (256, 2, 8),
    "medium": (384, 3, 12),
    "large": (512, 4, 16),
}

SPATIAL = ("checkerboard", "spatial-row", "spatial-col")


class AffineJetCoupling(_SplitCoupling):
    """Jet-style affine coupling ``y_b = x_b * 2 sigmoid(s) + t`` (non-constant det)."""

    _out_factor = 2
    piecewise_affine = False
    volume_preserving = False

    def _st(self, params: Tensor):
        s, t = params.chunk(2, dim=1)
        log_scale = torch.nn.functional.logsigmoid(s.float()) + math.log(2.0)
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


class VariableDetFlow(Flow):
    """Flow wrapper that also accumulates per-sample log-dets (for the affine baseline)."""

    def log_prob(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        ctx = self._context(context, x.shape[0], x)
        ld = x.new_zeros(x.shape[0], dtype=torch.float32)
        z = x
        for b in self.bijection:
            if hasattr(b, "forward_with_logdet"):
                z, l = b.forward_with_logdet(z, ctx)
                ld = ld + l
            else:
                z = b(z, ctx)
                ld = ld + b.log_abs_det()
        return self.base.log_prob(z) + ld


def make_conditioner(embed_dim: int, depth: int, context_dim: int = 0):
    def factory(shape_in, shape_out):
        return JetConditioner(
            list(shape_in),
            patch_size=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=max(1, embed_dim // 64),
            out_dim=shape_out[0],
            context_dim=context_dim,
        )

    return factory


def build_flow(
    variant: str = "us-radial",
    size: str = "small",
    image_shape=(3, 32, 32),
    base: Optional[nn.Module] = None,
    context_dim: int = 0,
    seed: int = 0,
) -> Flow:
    """Builds one of the study variants. ``base`` overrides the base distribution."""
    torch.manual_seed(seed)
    E, depth, L = SIZES[size]
    ev = tuple(image_shape)
    layers = []
    for _ in range(2):
        sq = Squeeze(ev, 2)
        layers.append(sq)
        ev = tuple(sq.out_shape)
    cond = make_conditioner(E, depth, context_dim)
    n_spatial = 0
    for i in range(L):
        if i % 5 == 4:
            split = SPATIAL[n_spatial % 3]
            n_spatial += 1
            parity = (i // 5) % 2
        else:
            layers.append(Permutation(ev))
            split, parity = "channel", i % 2
        if variant == "affine":
            layers.append(AffineJetCoupling(ev, cond, split=split, parity=parity))
        else:
            layers.append(SplitAdditiveCoupling(ev, cond, split=split, parity=parity))
    layers.append(GlobalScale(ev, init=1.0))

    if base is None:
        if variant == "us-radial":
            from cjd_flows.radial import RadialBase  # noqa: F401  (configured in train script)

            raise ValueError("pass a radial base explicitly")
        base = StandardNormal(ev)
    cls = VariableDetFlow if variant == "affine" else Flow
    return cls(Chain(*layers), base, context_dim=context_dim, event_shape=image_shape)
