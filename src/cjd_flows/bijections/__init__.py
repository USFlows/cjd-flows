"""Bijections with constant Jacobian determinant."""

from cjd_flows.bijections.autoregressive import MaskedConvAutoregressive, ShiftAutoregressive
from cjd_flows.bijections.coupling import (
    GINCoupling,
    MaskedAdditiveCoupling,
    SplitAdditiveCoupling,
    channel_mask,
    checkerboard_mask,
    random_mask,
    split_indices,
)
from cjd_flows.bijections.linear import (
    ActNorm,
    AffineFromMatrix,
    ChannelLinear,
    FixedOrthogonal,
    GlobalScale,
    HouseholderLinear,
    LULinear,
    OrthogonalLinear,
    Permutation,
    Reverse,
    Shift,
    merge_linear,
)
from cjd_flows.bijections.reshape import Flatten, HaarWavelet, Patchify, Reshape, Squeeze
from cjd_flows.core import Bijection, Chain, Identity, InverseBijection, VolumePreserving

__all__ = [
    "Bijection", "VolumePreserving", "Chain", "Identity", "InverseBijection",
    "MaskedAdditiveCoupling", "SplitAdditiveCoupling", "GINCoupling",
    "checkerboard_mask", "channel_mask", "random_mask", "split_indices",
    "ShiftAutoregressive", "MaskedConvAutoregressive",
    "ChannelLinear", "LULinear", "HouseholderLinear", "OrthogonalLinear", "ActNorm",
    "GlobalScale", "Shift", "Permutation", "Reverse", "FixedOrthogonal", "AffineFromMatrix", "merge_linear",
    "Squeeze", "HaarWavelet", "Patchify", "Reshape", "Flatten",
]
