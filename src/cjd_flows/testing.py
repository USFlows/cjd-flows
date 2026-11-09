"""Public test utilities.

Use these helpers to verify custom bijections (e.g. with your own conditioner)
against the library contract::

    from cjd_flows.testing import check_bijection
    check_bijection(my_layer, torch.randn(8, 4, 6, 6, dtype=torch.float64))
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from cjd_flows.core import Bijection, Flow, _zero_like_params


def jacobian_log_abs_det(f, x: Tensor) -> Tensor:
    """Per-sample ``log|det J_f(x)|`` via autograd (for small event sizes).

    Args:
        f: map from ``(B, *event)`` to ``(B, *event)``.
        x: input batch.

    Returns:
        Tensor of shape ``(B,)``.
    """
    event = x.shape[1:]
    d = math.prod(event)

    def single(xi: Tensor) -> Tensor:
        return f(xi.reshape(1, *event)).reshape(d)

    out = []
    for xi in x:
        J = torch.autograd.functional.jacobian(single, xi.reshape(d))
        out.append(torch.linalg.slogdet(J).logabsdet)
    return torch.stack(out)


def check_bijection(
    bijection: Bijection,
    x: Tensor,
    context: Optional[Tensor] = None,
    atol: Optional[float] = None,
    check_jacobian: bool = True,
    max_jacobian_dim: int = 256,
) -> None:
    """Asserts that ``bijection`` satisfies the CJD-Flows contract on input ``x``.

    Checks:
        1. shape preservation of forward and inverse,
        2. ``inverse(forward(x)) == x`` and ``forward(inverse(z)) == z``,
        3. ``log_abs_det()`` is a finite 0-dim tensor,
        4. (optional) ``log_abs_det()`` equals the autograd log-determinant for
           every sample -- which also verifies that it is constant,
        5. gradients w.r.t. all parameters are finite.

    Use ``float64`` inputs (and ``bijection.double()``) for tight tolerances.
    """
    if atol is None:
        atol = 1e-8 if x.dtype == torch.float64 else 1e-4
    rtol = 1e-6 if x.dtype == torch.float64 else 1e-4

    z = bijection(x, context)
    assert z.shape == x.shape, f"forward changed shape {x.shape} -> {z.shape}"
    x_rec = bijection.inverse(z, context)
    assert x_rec.shape == x.shape, f"inverse changed shape {z.shape} -> {x_rec.shape}"
    err = (x_rec - x).abs().max().item()
    assert err < atol * 10 * max(1.0, x.abs().max().item()), f"inverse(forward(x)) != x (max err {err:.3e})"

    z2 = torch.randn_like(x)
    err2 = (bijection(bijection.inverse(z2, context), context) - z2).abs().max().item()
    assert err2 < atol * 10 * max(1.0, z2.abs().max().item()), f"forward(inverse(z)) != z (max err {err2:.3e})"

    ladj = bijection.log_abs_det()
    assert isinstance(ladj, Tensor) and ladj.dim() == 0, "log_abs_det() must return a 0-dim tensor"
    assert torch.isfinite(ladj), "log_abs_det() is not finite"

    if check_jacobian and math.prod(x.shape[1:]) <= max_jacobian_dim:
        ref = jacobian_log_abs_det(lambda v: bijection(v, None if context is None else context[:1]), x[:4] if context is None else x[:1])
        assert torch.allclose(
            ref, ladj.detach().to(ref).expand_as(ref), atol=max(atol * 100, 1e-5), rtol=rtol * 100
        ), f"log_abs_det {ladj.item():.6f} != autograd {ref.tolist()}"

    params = [p for p in bijection.parameters() if p.requires_grad]
    if params:
        loss = bijection(x, context).square().sum() + ladj
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for p, g in zip(params, grads):
            if g is not None:
                assert torch.isfinite(g).all(), "non-finite gradient"


def check_flow_normalization(flow: Flow, n: int = 200_000, radius: float = 6.0) -> float:
    """Monte-Carlo estimate of ``∫ p(x) dx`` over a box (for 1D/2D flows).

    Returns the estimate; a well-defined flow yields a value close to 1 if the
    box covers (almost) all the mass. The flow is evaluated in eval mode (so
    e.g. :class:`~cjd_flows.bijections.ActNorm` is not data-initialized on the
    box samples); its training mode is restored afterwards.
    """
    d = flow.dim
    ref = _zero_like_params(flow)
    x = (torch.rand(n, d, dtype=ref.dtype, device=ref.device) * 2 - 1) * radius
    was_training = flow.training
    flow.eval()
    try:
        with torch.no_grad():
            p = flow.log_prob(x.reshape(n, *flow.event_shape)).exp()
    finally:
        flow.train(was_training)
    return float(p.mean() * (2 * radius) ** d)
