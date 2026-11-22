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

from typing import Optional

import torch
from torch import nn

from benchmarks.baselines import AffineJetCoupling, VariableDetFlow
from cjd_flows.bijections.coupling import SplitAdditiveCoupling
from cjd_flows.bijections.linear import GlobalScale, Permutation, Shift
from cjd_flows.bijections.reshape import Squeeze
from cjd_flows.conditioners import JetConditioner
from cjd_flows.core import Chain, Flow, StandardNormal

SIZES = {
    # name: (embed_dim, depth, n_couplings)
    "tiny": (128, 1, 4),
    "small": (256, 2, 8),
    "medium": (384, 3, 12),
    "large": (512, 4, 16),
    "deep": (256, 1, 24),
}

SPATIAL = ("checkerboard", "spatial-row", "spatial-col")


def make_conditioner(embed_dim: int, depth: int, context_dim: int = 0, kind: str = "jet"):
    if kind == "conv":
        from cjd_flows.conditioners import ConvNet

        def conv_factory(shape_in, shape_out):
            return ConvNet(shape_in[0], shape_out[0], spatial_ndim=len(shape_in) - 1,
                           hidden=(embed_dim,) * (depth + 1), kernel_size=3, context_dim=context_dim)

        return conv_factory

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
    conditioner: str = "jet",
    shift: bool = True,
) -> Flow:
    """Builds one of the study variants. ``base`` overrides the base distribution.

    ``shift`` adds a data-initialized :class:`Shift` before the global scale
    (without it, short runs end with off-centre latents and colour-cast samples).
    """
    torch.manual_seed(seed)
    E, depth, L = SIZES[size]
    ev = tuple(image_shape)
    layers = []
    for _ in range(2):
        sq = Squeeze(ev, 2)
        layers.append(sq)
        ev = tuple(sq.out_shape)
    cond = make_conditioner(E, depth, context_dim, conditioner)
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
    if shift:
        layers.append(Shift(ev))
    layers.append(GlobalScale(ev, init=1.0))

    if base is None:
        if variant == "us-radial":
            from cjd_flows.radial import RadialBase  # noqa: F401  (configured in train script)

            raise ValueError("pass a radial base explicitly")
        base = StandardNormal(ev)
    cls = VariableDetFlow if variant == "affine" else Flow
    return cls(Chain(*layers), base, context_dim=context_dim, event_shape=image_shape)
