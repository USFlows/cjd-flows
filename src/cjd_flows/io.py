"""Saving and loading flows together with their builder configuration.

Flows built with a registered builder (:func:`~cjd_flows.models.USFlow`,
:func:`~cjd_flows.models.VeriFlow`, or your own function decorated with
:func:`register_builder`) remember the arguments they were built with
(``flow.builder_config``). :func:`save` stores these arguments as JSON
together with the weights, the dtype and the library version, so that
:func:`load` can rebuild the flow without any code from the caller::

    from cjd_flows import USFlow
    from cjd_flows.io import load, save

    flow = USFlow((2,), blocks=8, base="ggmix")
    ...  # train
    save(flow, "flow.pt")
    flow = load("flow.pt")           # rebuilt from the stored configuration

The file is a :func:`torch.save` archive of tensors, strings and numbers only,
read with ``torch.load(..., weights_only=True)``; loading never executes
pickled code. Flows that were not built by a registered builder, or whose
builder arguments are not JSON-serializable (e.g. a custom conditioner
factory or a base distribution instance), are saved without configuration and
loaded into a model you construct yourself: ``load(path, model=my_flow)``.

Optimizer, scheduler and EMA state are not part of the file.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn

__all__ = ["save", "load", "load_metadata", "register_builder", "FORMAT_VERSION"]

#: Version of the file layout written by :func:`save`.
FORMAT_VERSION = 1
_FORMAT = "cjd-flows"

_BUILDERS: Dict[str, Callable[..., nn.Module]] = {}
# While load() replays a builder: delegated builder name -> saved configuration
_REPLAY: contextvars.ContextVar[Optional[Dict[str, dict]]] = contextvars.ContextVar("cjd_flows_replay", default=None)

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- config capture
def _to_json(value: Any) -> Any:
    """JSON-compatible copy of ``value`` (tuples become lists); raises TypeError otherwise."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("dict keys must be strings")
        return {k: _to_json(v) for k, v in value.items()}
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def register_builder(fn: Optional[Callable] = None, *, name: Optional[str] = None):
    """Registers a model builder so that :func:`load` can rebuild its flows.

    The decorated function must return an :class:`~torch.nn.Module`. Its call
    arguments, with defaults filled in, are recorded on the result as
    ``builder_config`` (``{"builder": name, "kwargs": {...}}``) if they are
    JSON-serializable
    (numbers, strings, ``None``, lists/tuples, dicts with string keys);
    otherwise ``builder_config`` is ``None`` and the reason is kept in
    ``builder_config_error``.

    When a registered builder delegates to another one (e.g. ``VeriFlow``
    calls ``USFlow``) and returns its model, the outer call is recorded and the
    inner, fully resolved configuration is kept under ``"delegate"``.
    :func:`load` replays the outer builder and substitutes the saved inner
    arguments in its call of the delegated builder (when the explicitly passed
    arguments match), so the defaults of every builder in the chain are frozen
    at save time and everything the outer builder does is replayed as well.

    Can be used as ``@register_builder`` or ``@register_builder(name="MyFlow")``.
    """

    def decorate(f: Callable) -> Callable:
        key = name or f.__name__
        existing = _BUILDERS.get(key)
        if existing is not None and not _same_builder(existing.__wrapped__, f):
            raise ValueError(f"a different builder named {key!r} is already registered")
        signature = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            saved = (_REPLAY.get() or {}).get(key)
            if saved is not None and _matches(signature, args, kwargs, saved["kwargs"]):
                args, kwargs = _split_args(signature, saved["kwargs"])
            # snapshot the arguments before the call: the builder may mutate them
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            for p in signature.parameters.values():  # flatten **kwargs
                if p.kind is inspect.Parameter.VAR_KEYWORD and p.name in arguments:
                    arguments.update(arguments.pop(p.name))
                elif p.kind is inspect.Parameter.VAR_POSITIONAL and p.name in arguments:
                    raise TypeError(f"builder {key!r}: *{p.name} arguments cannot be recorded")
            try:
                recorded, error = _to_json(arguments), None  # deep copy
            except TypeError as e:
                recorded, error = None, f"builder {key!r}: argument not JSON-serializable ({e})"
            model = f(*args, **kwargs)
            inner = getattr(model, "builder_config", None)  # set by a delegated registered builder
            config = None
            if recorded is not None:
                config = {"builder": key, "kwargs": recorded}
                if inner is not None:
                    config["delegate"] = inner
            model.builder_config = config
            model.builder_config_error = error
            return model

        _BUILDERS[key] = wrapper
        return wrapper

    return decorate(fn) if fn is not None else decorate


def _qualname(f: Callable) -> str:
    return f"{getattr(f, '__module__', '')}.{getattr(f, '__qualname__', repr(f))}"


def _same_builder(a: Callable, b: Callable) -> bool:
    """True for the same function, or a re-definition of it (e.g. a module reload).

    Closures are only the same if they are the same object: two closures from
    one factory share code and name but capture different values.
    """
    if a is b:
        return True
    return (
        _qualname(a) == _qualname(b)
        and getattr(a, "__code__", None) == getattr(b, "__code__", None)
        and getattr(a, "__closure__", None) is None
        and getattr(b, "__closure__", None) is None
    )


def _split_args(signature: inspect.Signature, recorded: dict):
    """``(args, kwargs)`` for a call with recorded arguments (positional-only ones positionally)."""
    kwargs = dict(recorded)
    args = []
    for p in signature.parameters.values():
        if p.kind is inspect.Parameter.POSITIONAL_ONLY and p.name in kwargs:
            args.append(kwargs.pop(p.name))
    return args, kwargs


def _matches(signature: inspect.Signature, args, kwargs, recorded: dict) -> bool:
    """True if the explicitly passed arguments agree with the recorded ones."""
    try:
        explicit = dict(signature.bind(*args, **kwargs).arguments)
        for p in signature.parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD and p.name in explicit:
                explicit.update(explicit.pop(p.name))
        explicit = _to_json(explicit)
    except TypeError:
        return False
    return all(k in recorded and recorded[k] == v for k, v in explicit.items())


def _call_builder(config: dict) -> nn.Module:
    """Replays ``config``: calls its builder with the recorded arguments, and
    its delegated builders with their recorded (default-frozen) arguments."""
    builder = _BUILDERS.get(config["builder"])
    if builder is None:
        raise ValueError(
            f"unknown builder {config['builder']!r}; import the module that registers it before loading"
        )
    delegates = {}
    inner = config.get("delegate")
    while inner is not None:
        delegates.setdefault(inner["builder"], inner)
        inner = inner.get("delegate")
    token = _REPLAY.set(delegates)
    try:
        args, kwargs = _split_args(inspect.signature(builder.__wrapped__), config["kwargs"])
        return builder(*args, **kwargs)
    finally:
        _REPLAY.reset(token)


# --------------------------------------------------------------------------- save / load
def _float_dtype(module: nn.Module) -> Optional[torch.dtype]:
    for t in list(module.parameters()) + list(module.buffers()):
        if t.is_floating_point():
            return t.dtype
    return None


def save(flow: nn.Module, path: PathLike, metadata: Optional[dict] = None) -> Path:
    """Saves the weights of ``flow`` together with its builder configuration.

    Args:
        flow: a flow (any :class:`~torch.nn.Module`). If it carries a
            ``builder_config`` (built by a registered builder), :func:`load`
            can rebuild it; otherwise a warning is issued and the file has to
            be loaded with ``load(path, model=...)``.
        path: output file.
        metadata: optional JSON-serializable dict stored alongside (e.g.
            training settings or metrics).

    Returns:
        The path written to.
    """
    path = Path(path)
    config = getattr(flow, "builder_config", None)
    if config is None:
        reason = getattr(flow, "builder_config_error", None) or "the flow was not built by a registered builder"
        warnings.warn(
            f"saving without builder configuration ({reason}); load it with load(path, model=...)",
            stacklevel=2,
        )
    try:
        meta_json = json.dumps(_to_json(metadata or {}))
    except TypeError as e:
        raise TypeError(f"metadata must be JSON-serializable: {e}") from e
    from cjd_flows import __version__

    dtype = _float_dtype(flow)
    payload = {
        "format": _FORMAT,
        "format_version": FORMAT_VERSION,
        "library_version": __version__,
        "config": json.dumps(config) if config is not None else "",
        "dtype": str(dtype).replace("torch.", "") if dtype is not None else "",
        "metadata": meta_json,
        "state_dict": {k: v.detach().cpu() for k, v in flow.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _read(path: PathLike) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError(f"{path} is not a file written by cjd_flows.io.save")
    if int(payload.get("format_version", -1)) > FORMAT_VERSION:
        raise ValueError(
            f"{path} uses file format {payload['format_version']}; this version of cjd-flows reads up to "
            f"{FORMAT_VERSION}. Upgrade cjd-flows."
        )
    return payload


def load_metadata(path: PathLike) -> dict:
    """Returns the builder configuration, versions and user metadata of a saved file (no model is built)."""
    payload = _read(path)
    return {
        "config": json.loads(payload["config"]) if payload["config"] else None,
        "library_version": payload["library_version"],
        "format_version": payload["format_version"],
        "dtype": payload["dtype"] or None,
        "metadata": json.loads(payload["metadata"]),
    }


def load(
    path: PathLike,
    model: Optional[nn.Module] = None,
    map_location: Union[str, torch.device, None] = None,
    strict: bool = True,
) -> nn.Module:
    """Loads a flow saved with :func:`save`.

    Args:
        path: file written by :func:`save`.
        model: module to load the weights into. If ``None``, the flow is
            rebuilt from the stored builder configuration.
        map_location: device to move the flow to (default: CPU).
        strict: passed to :meth:`~torch.nn.Module.load_state_dict`.

    Returns:
        The flow in eval mode, with the saved dtype (when rebuilt) and the
        saved weights and buffers (including data-dependent initialization
        flags such as :class:`~cjd_flows.bijections.Shift` ``initialized``).
    """
    payload = _read(path)
    saved_version = str(payload["library_version"])
    from cjd_flows import __version__

    if saved_version.split(".")[0] != __version__.split(".")[0]:
        warnings.warn(
            f"{path} was saved with cjd-flows {saved_version}, this is {__version__}", stacklevel=2
        )
    if model is None:
        if not payload["config"]:
            raise ValueError(
                f"{path} has no builder configuration; construct the model yourself and pass it as model=..."
            )
        config = json.loads(payload["config"])
        import cjd_flows.models  # noqa: F401  (registers the built-in builders)

        model = _call_builder(config)
        if payload["dtype"]:
            model = model.to(getattr(torch, payload["dtype"]))
        model.builder_config = config
        model.builder_config_error = None
    model.load_state_dict(payload["state_dict"], strict=strict)
    if map_location is not None:
        model = model.to(map_location)
    return model.eval()
