"""High-level model builders.

:func:`USFlow` assembles a uniformly scaling (constant-Jacobian-determinant)
flow from the library's building blocks::

    from cjd_flows.models import USFlow
    flow = USFlow((2,), blocks=8, base="ggmix")          # 2-D density
    flow = USFlow((1, 28, 28), blocks=6, base="ggmix")   # images (ConvNet conditioners)

Every flow has the structure

    [block_1, ..., block_n, final_linear, GlobalScale]

with ``block_i = [mix_i, coupling_i]`` or, with ``conjugate=True``,
``block_i = [mix_i, coupling_i, mix_i^{-1}]`` (the "adjoint" blocks
``A^{-1} ∘ C ∘ A`` of VeriFlow / USFlows). Only the final linear layer and the
global scale change volume, so ``log|det J|`` is a single learnable constant.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

from torch import nn

from cjd_flows.bijections.autoregressive import ShiftAutoregressive
from cjd_flows.bijections.coupling import GINCoupling, MaskedAdditiveCoupling, SplitAdditiveCoupling
from cjd_flows.bijections.linear import (
    ActNorm,
    GlobalScale,
    HouseholderLinear,
    LULinear,
    OrthogonalLinear,
    Permutation,
)
from cjd_flows.conditioners import MLP, MADE, ConvNet, JetConditioner
from cjd_flows.core import BaseDistribution, Bijection, Chain, Flow, ShapeLike, StandardNormal, as_shape

BASES = (
    "gaussian",
    "laplace",
    "gg",
    "gg-monotone",
    "ggmix",
    "ggmix-monotone",
    "spline",
    "monotone",
    "khinchine",
    "student-t",
    "lognormal-mix",
    "radial-mixture",
)


def make_base(
    base: Union[str, BaseDistribution],
    event_shape: ShapeLike,
    p: float = 2.0,
    n_components: int = 4,
    learn_scale: bool = False,
) -> BaseDistribution:
    """Creates a base distribution by name.

    Args:
        base: one of :data:`BASES` or a :class:`BaseDistribution` instance.
            ``"gaussian"`` is a fixed standard normal; all other names give an
            Lp-radial distribution (with norm ``p``) whose 1-D norm
            distribution is learnable:

            * ``"laplace"`` -- fixed ``p``-generalized normal (Laplace for p=1),
            * ``"gg"`` / ``"gg-monotone"`` -- generalized gamma (monotone ties ``a=d``),
            * ``"ggmix"`` / ``"ggmix-monotone"`` -- mixture of generalized gammas,
            * ``"spline"`` -- rational-quadratic spline on the log-radius,
            * ``"monotone"`` -- monotone (radially decreasing) profile,
            * ``"khinchine"`` -- mixture of uniform balls (monotone by construction),
            * ``"student-t"`` -- heavy-tailed radial t,
            * ``"lognormal-mix"`` -- log-normal mixture (shell profile, *not* monotone),
            * ``"radial-mixture"`` -- mixture of radial components with
              learnable centres (multi-modal latent).
        event_shape: latent event shape.
        p: Lp norm exponent of the radial base.
        n_components: number of mixture components (``*mix`` / ``radial-mixture``).
        learn_scale: learn the global radial scale. Defaults to ``False``
            because the flow's :class:`GlobalScale` already covers it.
    """
    if isinstance(base, BaseDistribution):
        return base
    from cjd_flows import radial as R

    shape = as_shape(event_shape)
    d = math.prod(shape)
    if base == "gaussian":
        return StandardNormal(shape)
    if base == "laplace":
        return R.RadialBase.generalized_gaussian(shape, p) if p != 2.0 else R.RadialBase.gaussian(shape)
    if base == "radial-mixture":
        return R.RadialMixture(shape, n_components=n_components, p=p)
    norms = {
        "gg": lambda: R.GeneralizedGamma(d, p, learn_scale=learn_scale),
        "gg-monotone": lambda: R.GeneralizedGamma(d, p, monotone=True, learn_scale=learn_scale),
        "ggmix": lambda: R.GeneralizedGammaMixture(d, p, n_components=n_components, learn_scale=learn_scale),
        "ggmix-monotone": lambda: R.GeneralizedGammaMixture(
            d, p, n_components=n_components, monotone=True, learn_scale=learn_scale
        ),
        "spline": lambda: R.SplineNorm(d, p, num_bins=16, bound=6.0, learn_scale=learn_scale),
        "monotone": lambda: R.MonotoneProfileNorm(d, p, learn_scale=learn_scale),
        "khinchine": lambda: R.KhinchineNorm(d, p),
        "student-t": lambda: R.StudentTNorm(d, p, learn_scale=learn_scale),
        "lognormal-mix": lambda: R.LogNormalMixture(d, p, n_components=n_components, learn_scale=learn_scale),
    }
    if base not in norms:
        raise ValueError(f"unknown base {base!r}; choose from {BASES}")
    return R.RadialBase(shape, p=p, norm=norms[base]())


def _mixing_layer(kind: Optional[str], event_shape) -> Optional[Bijection]:
    if kind is None or kind == "none":
        return None
    if kind == "lu":
        return LULinear(event_shape)
    if kind in ("sl-lu", "lu-unit"):
        return LULinear(event_shape, unit_det=True)
    if kind == "orthogonal":
        return OrthogonalLinear(event_shape)
    if kind == "householder":
        return HouseholderLinear(event_shape, n_reflections=min(event_shape[0], 8))
    if kind == "permutation":
        return Permutation(event_shape)
    if kind == "actnorm":
        return ActNorm(event_shape, zero_sum=True)
    raise ValueError(f"unknown mixing layer {kind!r}")


def _conditioner_factory(
    event_shape, conditioner, hidden, activation, context_dim, out_factor=1, **kw
) -> Callable:
    spatial = len(event_shape) > 1
    if conditioner == "auto":
        conditioner = "conv" if spatial else "mlp"
    if callable(conditioner) and not isinstance(conditioner, str):
        return conditioner
    if conditioner == "mlp":
        return lambda i, o: MLP(i, o, hidden=hidden, activation=activation, context_dim=context_dim, **kw)
    if conditioner == "conv":
        return lambda i, o: ConvNet(
            i[0], o[0], spatial_ndim=len(i) - 1, hidden=hidden, activation=activation, context_dim=context_dim, **kw
        )
    if conditioner == "jet":
        return lambda i, o: JetConditioner(
            list(i), out_dim=o[0], context_dim=context_dim, patch_size=kw.pop("patch_size", 1), **kw
        )
    raise ValueError(f"unknown conditioner {conditioner!r}")


def USFlow(
    event_shape: ShapeLike,
    *,
    blocks: int = 8,
    coupling: str = "additive",
    conditioner: Union[str, Callable] = "auto",
    hidden: Sequence[int] = (256, 256),
    activation: str = "relu",
    mixing: Optional[str] = "lu",
    conjugate: bool = False,
    final_linear: bool = True,
    global_scale: bool = True,
    base: Union[str, BaseDistribution] = "gaussian",
    p: float = 2.0,
    n_components: int = 4,
    context_dim: int = 0,
    masked: bool = False,
    conditioner_kwargs: Optional[dict] = None,
) -> Flow:
    """Builds a uniformly scaling flow.

    Args:
        event_shape: data shape, ``(D,)`` for vectors or ``(C, *spatial)``.
        blocks: number of coupling blocks.
        coupling: ``"additive"`` (NICE-style, piecewise affine with ReLU
            conditioners), ``"gin"`` (affine with zero-sum log-scale; more
            expressive per block, not piecewise affine) or ``"autoregressive"``
            (shift-only MADE; vectors only).
        conditioner: ``"auto"`` (MLP for vectors, ConvNet for images),
            ``"mlp"``, ``"conv"``, ``"jet"`` or a factory
            ``f(shape_in, shape_out) -> nn.Module``.
        hidden: hidden widths/channels of the conditioners.
        activation: conditioner nonlinearity (``relu``/``leaky_relu`` keep the
            flow piecewise affine).
        mixing: linear layer before every coupling: ``"lu"``, ``"sl-lu"``
            (unit determinant), ``"orthogonal"``, ``"householder"``,
            ``"permutation"``, ``"actnorm"`` or ``None``.
        conjugate: use adjoint blocks ``A^{-1} C A`` (requires ``mixing``).
        final_linear: append a free LU layer at the end.
        global_scale: append a learnable scalar scale (the "uniform scaling").
        base: base distribution name or instance, see :func:`make_base`.
        p: norm of the radial base.
        n_components: mixture components of the base (if applicable).
        context_dim: dimension of an optional conditioning vector.
        masked: use masked (instead of split) additive couplings -- the
            original USFlows formulation. Split couplings are cheaper.
        conditioner_kwargs: extra keyword arguments for the conditioner.
    """
    shape = as_shape(event_shape)
    spatial = len(shape) > 1
    kw = dict(conditioner_kwargs or {})
    layers = []
    for i in range(blocks):
        parity = i % 2
        mix = _mixing_layer(mixing, shape)
        if mix is not None:
            layers.append(mix)
        if coupling == "autoregressive":
            if spatial:
                raise ValueError("autoregressive coupling is only implemented for vector data")
            made = MADE(shape[0], hidden=hidden, context_dim=context_dim, activation=activation,
                        order="reverse" if parity else None)
            layers.append(ShiftAutoregressive(shape[0], made))
        else:
            if spatial:
                split = "channel" if (shape[0] > 1 and i % 2 == 0) else "checkerboard"
            else:
                split = "channel"
            if coupling == "additive" and masked:
                fac = _conditioner_factory(shape, conditioner, hidden, activation, context_dim, **kw)
                cond = fac(shape, shape)
                mask = "checkerboard" if (split == "checkerboard" or not spatial) else "channel"
                layers.append(MaskedAdditiveCoupling(shape, cond, mask=mask, parity=parity))
            elif coupling == "additive":
                fac = _conditioner_factory(shape, conditioner, hidden, activation, context_dim, **kw)
                layers.append(SplitAdditiveCoupling(shape, fac, split=split, parity=parity))
            elif coupling == "gin":
                fac = _conditioner_factory(shape, conditioner, hidden, activation, context_dim, **kw)
                layers.append(GINCoupling(shape, fac, split=split, parity=parity))
            else:
                raise ValueError(f"unknown coupling {coupling!r}")
        if conjugate and mix is not None:
            layers.append(mix.inv)
    if final_linear:
        layers.append(LULinear(shape))
    if global_scale:
        layers.append(GlobalScale(shape))
    return Flow(Chain(*layers), make_base(base, shape, p=p, n_components=n_components),
                context_dim=context_dim)


def VeriFlow(
    event_shape: ShapeLike,
    *,
    blocks: int = 5,
    hidden: Sequence[int] = (128, 128, 128),
    base: Union[str, BaseDistribution] = "ggmix-monotone",
    p: float = 1.0,
    **kwargs,
) -> Flow:
    """Verification-friendly preset: piecewise-affine (ReLU) conditioners,
    masked additive couplings in adjoint LU blocks, and an L1-radial base with a
    *monotone* learnable profile, so latent density level sets are L1 balls
    (polytopes) and their pre-images are exactly encodable in MILP/SMT."""
    return USFlow(
        event_shape,
        blocks=blocks,
        hidden=hidden,
        coupling="additive",
        masked=True,
        mixing="lu",
        conjugate=True,
        activation="relu",
        base=base,
        p=p,
        **kwargs,
    )


__all__ = ["USFlow", "VeriFlow", "make_base", "BASES"]

# silence unused-import warnings for re-exported helpers
_ = nn
