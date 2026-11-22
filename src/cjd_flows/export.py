"""ONNX export of CJD flows.

:func:`to_onnx` wraps the flow in a tiny stateless ``nn.Module`` for one of
four computations and exports it:

=============  ==========  =============  ==========================================
mode           input       output         computation
=============  ==========  =============  ==========================================
``log_prob``   ``x``       ``log_prob``   ``log p(x)``
``forward``    ``x``       ``z``          ``z = f(x)`` (normalizing direction)
``inverse``    ``z``       ``x``          ``x = f⁻¹(z)``
``sample``     ``z``       ``x``          ``x = f⁻¹(z)`` with externally drawn ``z ~ p_Z``
=============  ==========  =============  ==========================================

Sampling is exported as a *deterministic* graph taking the latent ``z`` as
input (no RNG inside the graph); draw ``z`` with ``flow.base.sample``.
Conditional flows (``context_dim > 0``) get a second input ``context``
unless ``with_context=False`` (then ``flow.default_context`` is baked in).

The legacy TorchScript exporter (``dynamo=False``) is used if available
(it needs only ``onnx``); otherwise the dynamo exporter (needs ``onnxscript``).
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn

MODES = ("log_prob", "sample", "forward", "inverse")
_IO = {
    "log_prob": ("x", "log_prob"),
    "forward": ("x", "z"),
    "inverse": ("z", "x"),
    "sample": ("z", "x"),
}


class _Wrapper(nn.Module):
    def __init__(self, flow: nn.Module, mode: str, with_context: bool, freeze_linear: bool = False):
        super().__init__()
        self.flow = flow
        self.mode = mode
        self.with_context = with_context
        # Optionally replace the channel-linear layers by frozen matrices (their
        # parametrizations use triangular solves / matrix exponentials / inverses
        # that ONNX cannot express, at least in the inverse direction).
        self.bijection = _freeze_linear(flow.bijection) if freeze_linear else flow.bijection

    def _ctx(self, t: Tensor, context: Optional[Tensor]) -> Optional[Tensor]:
        if self.flow.context_dim == 0:
            return None
        if context is not None:
            return context
        return t.new_full((t.shape[0], self.flow.context_dim), float(self.flow.default_context))

    def forward(self, t: Tensor, context: Optional[Tensor] = None) -> Tensor:
        ctx = self._ctx(t, context)
        if self.mode == "log_prob":
            z = self.bijection(t, ctx)
            return self.flow.base.log_prob(z) + self.bijection.log_abs_det()
        if self.mode == "forward":
            return self.bijection(t, ctx)
        return self.bijection.inverse(t, ctx)


def _freeze_linear(bijection: nn.Module) -> nn.Module:
    from cjd_flows.bijections.linear import merge_linear

    with torch.no_grad():
        return merge_linear(bijection, min_group=1)


def export_module(
    flow: nn.Module, mode: str = "log_prob", with_context: Optional[bool] = None, freeze_linear: bool = False
) -> nn.Module:
    """The wrapper ``nn.Module`` that :func:`to_onnx` exports (useful for other exporters).

    Args:
        flow: the flow.
        mode: one of :data:`MODES`.
        with_context: add a ``context`` input (default: iff ``flow.context_dim > 0``).
        freeze_linear: replace (runs of) channel-linear layers by frozen
            :class:`~cjd_flows.bijections.AffineFromMatrix` layers (see :func:`to_onnx`).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if with_context is None:
        with_context = getattr(flow, "context_dim", 0) > 0
    return _Wrapper(flow, mode, bool(with_context), freeze_linear=freeze_linear).eval()


def _ref(flow: nn.Module) -> Tensor:
    for t in flow.parameters():
        return t
    for t in flow.buffers():
        if t.is_floating_point():
            return t
    return torch.zeros(())


def to_onnx(
    flow: nn.Module,
    path: Union[str, Path],
    mode: str = "log_prob",
    batch_size: int = 1,
    dynamic_batch: bool = True,
    opset: Optional[int] = None,
    with_context: Optional[bool] = None,
    example: Optional[Tensor] = None,
    dynamo: Optional[bool] = None,
    freeze_linear: bool = True,
) -> Path:
    """Exports ``flow`` to ONNX.

    Args:
        flow: a :class:`~cjd_flows.core.Flow`.
        path: output ``.onnx`` file.
        mode: ``"log_prob"``, ``"sample"``, ``"forward"`` or ``"inverse"`` (see module doc).
        batch_size: batch size of the example input used for tracing.
        dynamic_batch: make the batch dimension symbolic.
        opset: ONNX opset (default 17 for the legacy exporter, exporter default otherwise).
        with_context: add a ``context`` input for conditional flows (default:
            ``True`` iff ``flow.context_dim > 0``).
        example: optional example input (shape ``(B, *event_shape)``, or
            ``(B, *latent_shape)`` for ``"sample"``/``"inverse"``).
        dynamo: force the exporter (``None``: legacy TorchScript exporter if
            available, else dynamo).
        freeze_linear: export the channel-linear layers (LU, orthogonal,
            Householder, ActNorm, ...) as frozen matrices via
            :func:`~cjd_flows.bijections.merge_linear` (consecutive ones are
            merged). Required for parametrizations that ONNX cannot express,
            e.g. the triangular solves of ``LULinear.inverse`` or the Cayley
            map of ``OrthogonalLinear``.

    Returns:
        The path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = export_module(flow, mode, with_context, freeze_linear=freeze_linear)
    ref = _ref(flow)
    if example is None:
        # latent-input graphs take z, whose shape differs from x for shape-changing flows
        shape = flow.event_shape if mode in ("log_prob", "forward") else getattr(flow, "latent_shape", flow.event_shape)
        example = torch.randn(batch_size, *shape, device=ref.device, dtype=ref.dtype)
    in_name, out_name = _IO[mode]
    args = (example,)
    input_names = [in_name]
    if wrapper.with_context:
        args = (example, example.new_full((example.shape[0], flow.context_dim), float(flow.default_context or 0.0)))
        input_names.append("context")
    dyn: Optional[Dict[str, Dict[int, str]]] = None
    if dynamic_batch:
        dyn = {n: {0: "batch"} for n in input_names + [out_name]}

    export_sig = inspect.signature(torch.onnx.export).parameters
    has_flag = "dynamo" in export_sig

    def legacy():
        kw = {"dynamo": False} if has_flag else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper, args, str(path), input_names=input_names, output_names=[out_name],
                dynamic_axes=dyn, opset_version=opset or 17, **kw,
            )

    def via_dynamo():
        shapes = None
        if dynamic_batch:
            b = torch.export.Dim("batch", min=1)
            shapes = tuple({0: b} for _ in args)
        torch.onnx.export(
            wrapper, args, str(path), input_names=input_names, output_names=[out_name],
            dynamic_shapes=shapes, opset_version=opset, dynamo=True,
        )

    was_training = flow.training
    flow.eval()
    try:
        with torch.no_grad():
            if dynamo is True:
                via_dynamo()
            elif dynamo is False or not has_flag:
                legacy()
            else:
                try:
                    legacy()
                except Exception as e:  # legacy exporter unavailable/failed -> dynamo
                    try:
                        via_dynamo()
                    except Exception as e2:
                        raise RuntimeError(f"ONNX export failed (legacy: {e!r}; dynamo: {e2!r})") from e2
        return path
    finally:
        flow.train(was_training)


def verify_onnx(
    flow: nn.Module,
    path: Union[str, Path],
    x: Tensor,
    context: Optional[Tensor] = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> float:
    """Compares an exported model (via onnxruntime) with the torch flow.

    The mode is inferred from the ONNX output name. ``x`` is the graph input
    (data for ``log_prob``/``forward``, latents for ``sample``/``inverse``).

    Returns:
        The maximum absolute deviation. Raises ``AssertionError`` if it exceeds
        ``atol + rtol·|reference|``; raises ``ImportError`` without onnxruntime.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out_name = sess.get_outputs()[0].name
    in_names = [i.name for i in sess.get_inputs()]
    mode = {"log_prob": "log_prob", "z": "forward", "x": "inverse"}[out_name]
    wrapper = export_module(flow, mode, with_context=len(in_names) > 1)
    with torch.no_grad():
        ref = wrapper(x, context).detach().cpu().double().numpy()
    feed = {in_names[0]: x.detach().cpu().numpy()}
    if len(in_names) > 1:
        if context is None:
            context = x.new_full((x.shape[0], flow.context_dim), float(flow.default_context or 0.0))
        feed[in_names[1]] = context.detach().cpu().numpy().astype(feed[in_names[0]].dtype)
    got = sess.run([out_name], feed)[0].astype(np.float64)
    err = float(np.abs(got - ref).max())
    tol = atol + rtol * float(np.abs(ref).max())
    assert err <= tol, f"ONNX output deviates from torch by {err:.3e} (tol {tol:.3e})"
    return err


__all__ = ["MODES", "export_module", "to_onnx", "verify_onnx"]
