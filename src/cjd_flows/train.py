"""Maximum-likelihood training of CJD flows.

:func:`fit` is a compact but complete trainer: AdamW with linear warmup and
cosine decay, gradient clipping, EMA weights, validation with early stopping,
non-finite step protection, optional AMP / ``torch.compile``, MAP priors and
noise augmentation (SoftFlow-style "soft training", fixed Gaussian noise as in
TarFlow, or a decaying noise curriculum).

Losses are reported as mean negative log-likelihood in **nats per sample**;
use :meth:`TrainResult.bits_per_dim` or :func:`nats_to_bits_per_dim` to
convert.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger("cjd_flows.train")

DataLike = Union[Tensor, Dataset, DataLoader]


# ---------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of a module's parameters.

    Only floating-point parameters are averaged; buffers are left untouched.

    Args:
        module: the module whose parameters are tracked.
        decay: EMA decay ``β`` (shadow ``← β·shadow + (1−β)·param``).
        warmup: use the effective decay ``min(β, (1+n)/(10+n))`` for the first
            updates so that early (random) weights are forgotten quickly.

    Example::

        ema = EMA(flow, 0.999)
        ...; opt.step(); ema.update()
        with ema.swap():          # evaluate with averaged weights
            val = evaluate_nll(flow, x_val)
    """

    def __init__(self, module: nn.Module, decay: float = 0.999, warmup: bool = True):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.module = module
        self.decay = float(decay)
        self.warmup = warmup
        self.num_updates = 0
        self.shadow: Dict[str, Tensor] = {
            n: p.detach().clone() for n, p in module.named_parameters() if p.is_floating_point()
        }

    def _params(self) -> Iterator[Tuple[str, Tensor]]:
        for n, p in self.module.named_parameters():
            if n in self.shadow:
                yield n, p

    @torch.no_grad()
    def update(self) -> None:
        """Updates the shadow parameters from the current module parameters."""
        self.num_updates += 1
        d = self.decay
        if self.warmup:
            d = min(d, (1 + self.num_updates) / (10 + self.num_updates))
        for n, p in self._params():
            s = self.shadow[n]
            if s.device != p.device or s.dtype != p.dtype:
                s = self.shadow[n] = s.to(p)
            s.lerp_(p.detach(), 1.0 - d)

    @torch.no_grad()
    def copy_to(self, module: Optional[nn.Module] = None) -> None:
        """Writes the averaged parameters into ``module`` (default: the tracked module)."""
        module = module if module is not None else self.module
        for n, p in module.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

    @contextlib.contextmanager
    def swap(self):
        """Context manager that temporarily loads the EMA weights into the module."""
        backup = {n: p.detach().clone() for n, p in self._params()}
        self.copy_to()
        try:
            yield self.module
        finally:
            with torch.no_grad():
                for n, p in self._params():
                    p.copy_(backup[n])

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": dict(self.shadow)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


# --------------------------------------------------------- noise augmentation
class NoiseAugmentation:
    """Noise augmentation / soft training of flows.

    Modes:
        * ``"soft"`` (SoftFlow, Kim et al. 2020): per-sample
          ``σ ~ U(0, sigma_max)``, ``x ← x + σ ε`` and the flow is conditioned
          on ``σ / sigma_max`` (requires ``flow.context_dim >= 1``; at test time
          ``flow.default_context = 0`` selects the clean distribution).
        * ``"gaussian"`` (TarFlow-style): fixed ``x ← x + sigma ε`` without
          context. Use :func:`tweedie_denoise` on samples afterwards.
        * ``"curriculum"``: ``x ← x + σ(t) ε`` with ``σ`` decaying from
          ``sigma_max`` to ``0`` over the first ``decay_frac`` of training
          (``schedule`` ``"linear"`` or ``"cosine"``), no context.

    Args:
        mode: see above.
        sigma: noise level for ``"gaussian"``; also the default ``sigma_max``.
        sigma_max: maximal noise level for ``"soft"`` / ``"curriculum"``.
        decay_frac: fraction of the total steps over which the curriculum decays.
        schedule: curriculum shape.
    """

    MODES = ("soft", "gaussian", "curriculum")

    def __init__(
        self,
        mode: str = "soft",
        sigma: Optional[float] = None,
        sigma_max: Optional[float] = None,
        decay_frac: float = 0.5,
        schedule: str = "cosine",
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.sigma_max = float(sigma_max if sigma_max is not None else (sigma if sigma is not None else 0.1))
        self.sigma = float(sigma if sigma is not None else self.sigma_max)
        self.decay_frac = decay_frac
        self.schedule = schedule

    @property
    def uses_context(self) -> bool:
        return self.mode == "soft"

    def current_sigma(self, step: int, total_steps: int) -> float:
        """Noise level of the ``"gaussian"``/``"curriculum"`` modes at ``step``."""
        if self.mode == "gaussian":
            return self.sigma
        if self.mode == "curriculum":
            t = min(1.0, step / max(1.0, self.decay_frac * total_steps))
            if self.schedule == "linear":
                return self.sigma_max * (1.0 - t)
            return self.sigma_max * 0.5 * (1.0 + math.cos(math.pi * t))
        raise ValueError("current_sigma is not defined for soft training (sigma is random)")

    def __call__(self, x: Tensor, step: int = 0, total_steps: int = 1) -> Tuple[Tensor, Optional[Tensor]]:
        """Returns ``(x_noisy, context)``; ``context`` is ``(B, 1)`` for soft training, else ``None``."""
        eps = torch.randn_like(x)
        if self.mode == "soft":
            u = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
            sigma = (u * self.sigma_max).reshape(-1, *([1] * (x.dim() - 1)))
            return x + sigma * eps, u.unsqueeze(1)
        s = self.current_sigma(step, total_steps)
        return (x + s * eps if s > 0 else x), None

    def __repr__(self) -> str:
        return f"NoiseAugmentation(mode={self.mode!r}, sigma={self.sigma}, sigma_max={self.sigma_max})"


def tweedie_denoise(flow: nn.Module, x: Tensor, sigma: float, context: Optional[Tensor] = None) -> Tensor:
    """Tweedie / score-based denoising ``x + σ² ∇_x log p(x)``.

    Needs one backward pass. For CJD flows the log-det term has zero gradient
    w.r.t. ``x``, so the score is ``J_f(x)^T ∇ log p_Z(f(x))``.

    Args:
        flow: a flow with ``log_prob(x, context)``, trained on ``σ``-noisy data.
        x: noisy samples.
        sigma: noise level (for soft-trained flows pass the matching
            ``context = σ / σ_max``).
        context: optional context.
    """
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        lp = flow.log_prob(x_, context) if context is not None else flow.log_prob(x_)
        (g,) = torch.autograd.grad(lp.sum(), x_)
    return (x_ + sigma**2 * g).detach()


# ----------------------------------------------------------- result / utils
@dataclass
class TrainResult:
    """Outcome of :func:`fit`.

    Attributes:
        train_losses: mean training NLL (nats/sample) per epoch (on the noisy
            data if noise augmentation is used; without the prior term).
        val_losses: mean validation NLL per evaluation (clean data).
        best_val_loss, best_epoch: best validation loss and its epoch (1-based).
        best_state: ``state_dict`` (on CPU) of the best weights (EMA weights if
            EMA is enabled), or ``None`` without validation data.
        steps: number of optimizer steps taken.
        epochs: number of (partial) epochs run.
        wall_time: seconds.
        skipped_steps: steps skipped because of non-finite loss/gradients.
        stopped_early: early stopping triggered.
        aborted: training stopped after too many consecutive non-finite steps.
        restored: which weights the flow holds after training.
        step_losses: ``(step, loss)`` pairs logged every ``log_every`` steps.
    """

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    best_state: Optional[Dict[str, Tensor]] = None
    steps: int = 0
    epochs: int = 0
    wall_time: float = 0.0
    skipped_steps: int = 0
    stopped_early: bool = False
    aborted: bool = False
    restored: str = "last"
    step_losses: List[Tuple[int, float]] = field(default_factory=list)

    def bits_per_dim(
        self, dim: int, preprocessing_log_det: Optional[float] = None, which: str = "val", n_bins: int = 256
    ) -> List[float]:
        """Converts the recorded losses (``which`` = ``"val"`` or ``"train"``) to bits/dim.

        ``preprocessing_log_det`` is ``log|det ∂y/∂x̃|`` from the raw (dequantized)
        data ``x̃ ∈ [0, n_bins)^D`` to model space ``y``; ``None`` assumes
        ``y = x̃ / n_bins``. See :func:`cjd_flows.data.bits_per_dim`.
        """
        losses = self.val_losses if which == "val" else self.train_losses
        return [nats_to_bits_per_dim(v, dim, preprocessing_log_det, n_bins) for v in losses]


def nats_to_bits_per_dim(
    nll_nats: Union[float, Tensor], dim: int, preprocessing_log_det: Optional[float] = None, n_bins: int = 256
):
    """``(nll − log|det ∂y/∂x̃|) / (D ln 2)`` (nll in nats per sample).

    Same convention as :func:`cjd_flows.data.bits_per_dim` (``None`` means the
    model data is ``x̃ / n_bins ∈ [0, 1)^D``).
    """
    from cjd_flows.data.preprocess import bits_per_dim

    return bits_per_dim(-nll_nats, dim, preprocessing_log_det, n_bins)


class _IndexBatcher:
    """Fast batching for tensors / TensorDatasets / datasets with ``get_batch``."""

    def __init__(self, source, batch_size: int, shuffle: bool, drop_last: bool, with_second: bool):
        self.source = source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.with_second = with_second
        self.n = len(source) if not isinstance(source, Tensor) else source.shape[0]

    def __len__(self) -> int:
        return self.n // self.batch_size if self.drop_last else math.ceil(self.n / self.batch_size)

    def _fetch(self, idx: Tensor):
        s = self.source
        if isinstance(s, Tensor):
            return s[idx.to(s.device)]
        if isinstance(s, TensorDataset):
            ts = s.tensors
            return tuple(t[idx.to(t.device)] for t in ts) if len(ts) > 1 else ts[0][idx.to(ts[0].device)]
        return s.get_batch(idx)

    def __iter__(self):
        perm = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(len(self)):
            yield self._fetch(perm[i * self.batch_size : (i + 1) * self.batch_size])


def _make_batches(data: DataLike, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False):
    if isinstance(data, DataLoader):
        return data
    if isinstance(data, Tensor) or (isinstance(data, TensorDataset)) or (
        isinstance(data, Dataset) and callable(getattr(data, "get_batch", None)) and num_workers == 0
    ):
        return _IndexBatcher(data, batch_size, shuffle, drop_last, True)
    if isinstance(data, Dataset):
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )
    raise TypeError(f"unsupported data type {type(data)}")


def _num_samples(data: DataLike, batches) -> int:
    if isinstance(data, Tensor):
        return data.shape[0]
    if isinstance(data, DataLoader):
        try:
            return len(data.dataset)  # type: ignore[arg-type]
        except TypeError:
            return len(data) * (data.batch_size or 1)
    try:
        return len(data)  # type: ignore[arg-type]
    except TypeError:
        return len(batches) * 1


def _unpack(batch, device, dtype, batch_context: bool) -> Tuple[Tensor, Optional[Tensor]]:
    """Extracts ``(x, context)`` from a batch (tensor, tuple/list or dict)."""
    ctx = None
    if isinstance(batch, Tensor):
        x = batch
    elif isinstance(batch, dict):
        x = batch.get("x", batch.get("input"))
        if batch_context:
            ctx = batch.get("context")
    elif isinstance(batch, (tuple, list)):
        x = batch[0]
        if batch_context and len(batch) > 1:
            ctx = batch[1]
    else:
        raise TypeError(f"cannot interpret batch of type {type(batch)}")
    x = x.to(device=device, dtype=dtype, non_blocking=True)
    if ctx is not None:
        ctx = ctx.to(device=device, dtype=dtype, non_blocking=True)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(1)
    return x, ctx


def _prior_modules(model: nn.Module) -> List[nn.Module]:
    """Top-most (sub)modules exposing a callable ``log_prior()`` (no double counting)."""
    found: List[Tuple[str, nn.Module]] = []
    for name, m in model.named_modules():  # pre-order: parents come first
        if any(p == "" or name.startswith(p + ".") for p, _ in found):
            continue
        if callable(getattr(m, "log_prior", None)):
            found.append((name, m))
    return [m for _, m in found]


def _cpu_state(model: nn.Module) -> Dict[str, Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _make_optimizer(model: nn.Module, optimizer, lr, weight_decay, betas, scale_lr_mult: float = 1.0) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if callable(optimizer) and not isinstance(optimizer, str):
        return optimizer(params)
    # "Scale-like" parameters: 0-dim scalars (e.g. GlobalScale.log_scale) and the
    # parameters of the base distribution. A single scale has to track the
    # shrinking latent variance during training; Adam moves each parameter by
    # roughly ``lr`` per step, so these get a larger learning rate. (Tensors that
    # merely have one element, e.g. the output bias of a conditioner with one
    # output in 2-D, are ordinary network weights.)
    base = getattr(model, "base", None)
    special = {id(p) for p in params if p.dim() == 0}
    if isinstance(base, nn.Module):
        special |= {id(p) for p in base.parameters()}
    scale = [p for p in params if id(p) in special]
    decay = [p for p in params if p.dim() >= 2 and id(p) not in special]
    no_decay = [p for p in params if p.dim() < 2 and id(p) not in special]
    groups = [
        g
        for g in (
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": scale, "weight_decay": 0.0, "lr": lr * scale_lr_mult},
        )
        if g["params"]
    ]
    name = optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=betas[0])
    raise ValueError(f"unknown optimizer {optimizer!r}")


def _lr_lambda(warmup: int, total: int, schedule: str, min_lr_ratio: float) -> Callable[[int], float]:
    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            t = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))
        if schedule == "linear":
            t = min(1.0, (step - warmup) / max(1, total - warmup))
            return 1.0 - (1 - min_lr_ratio) * t
        raise ValueError(f"unknown schedule {schedule!r}")

    return f


def _device_of(model: nn.Module) -> torch.device:
    for t in model.parameters():
        return t.device
    for t in model.buffers():
        return t.device
    return torch.device("cpu")


def _dtype_of(model: nn.Module) -> torch.dtype:
    for t in model.parameters():
        if t.is_floating_point():
            return t.dtype
    for t in model.buffers():
        if t.is_floating_point():
            return t.dtype
    return torch.get_default_dtype()


@torch.no_grad()
def evaluate_nll(
    flow: nn.Module,
    data: DataLike,
    batch_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
    context: Optional[Tensor] = None,
    batch_context: bool = False,
    amp: Union[bool, str] = False,
    pad_context: int = 0,
) -> float:
    """Mean negative log-likelihood (nats per sample) of ``flow`` on ``data``.

    Args:
        pad_context: number of zero columns appended to contexts taken from
            the batches (``batch_context=True``). Soft-trained conditional
            flows use one extra column for the noise level, which is ``0``
            for clean data.
    """
    device = torch.device(device) if device is not None else _device_of(flow)
    dtype = _dtype_of(flow)
    was_training = flow.training
    flow.eval()
    total, n = 0.0, 0
    try:
        for batch in _make_batches(data, batch_size, False, 0):
            x, ctx = _unpack(batch, device, dtype, batch_context)
            if ctx is not None and pad_context > 0:
                ctx = torch.cat([ctx, ctx.new_zeros(ctx.shape[0], pad_context)], dim=1)
            if ctx is None and context is not None:
                ctx = context
            with _autocast(device, amp):
                lp = flow.log_prob(x, ctx) if ctx is not None else flow.log_prob(x)
            total += float(-lp.double().sum())
            n += x.shape[0]
    finally:
        flow.train(was_training)
    return total / max(n, 1)


def _autocast(device: torch.device, amp: Union[bool, str]):
    if not amp:
        return contextlib.nullcontext()
    dt = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dt)


# ---------------------------------------------------------------------- fit
def fit(
    flow: nn.Module,
    train_data: DataLike,
    val_data: Optional[DataLike] = None,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: Union[str, Callable[[Sequence[nn.Parameter]], torch.optim.Optimizer]] = "adamw",
    betas: Tuple[float, float] = (0.9, 0.95),
    warmup_steps: Optional[int] = None,
    warmup_frac: float = 0.02,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    scale_lr_mult: float = 10.0,
    normalize_loss: bool = True,
    ema_decay: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    max_steps: Optional[int] = None,
    restore: str = "best",
    device: Optional[Union[str, torch.device]] = None,
    num_workers: int = 0,
    amp: Union[bool, str] = False,
    compile: bool = False,
    noise: Optional[NoiseAugmentation] = None,
    batch_context: bool = False,
    prior_weight: Optional[float] = None,
    max_nonfinite: int = 20,
    val_batch_size: Optional[int] = None,
    log_every: Optional[int] = None,
    verbose: bool = False,
    callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = None,
    seed: Optional[int] = None,
) -> TrainResult:
    """Trains ``flow`` by maximum likelihood (or MAP).

    Args:
        flow: a :class:`cjd_flows.core.Flow` (anything with ``log_prob(x[, context])``).
        train_data: tensor ``(N, *event)``, ``TensorDataset``, any ``Dataset``
            returning a tensor or an ``(x, label, ...)`` tuple, or a ``DataLoader``.
        val_data: same types; evaluated after every epoch (clean data, no noise).
        epochs: number of epochs.
        batch_size: batch size (ignored for DataLoaders).
        lr, weight_decay, betas: optimizer settings. Weight decay only applies
            to parameters with ``dim >= 2``.
        optimizer: ``"adamw"``, ``"adam"``, ``"sgd"`` or ``callable(params) -> Optimizer``.
        warmup_steps: linear warmup steps (overrides ``warmup_frac``).
        warmup_frac: warmup as a fraction of the total number of steps.
        schedule: ``"cosine"``, ``"linear"`` or ``"constant"`` after warmup.
        min_lr_ratio: final lr as a fraction of ``lr`` (cosine/linear).
        grad_clip: max gradient norm (``None`` disables clipping).
        scale_lr_mult: learning-rate multiplier for scalar parameters (e.g. the
            flow's global scale) and the base-distribution parameters. These
            have to follow the latent variance as the couplings learn; with a
            plain Adam step size they throttle training for thousands of steps.
        normalize_loss: backpropagate the NLL in nats *per dimension*, so that
            ``grad_clip`` does not depend on the data dimension.
        ema_decay: EMA decay (``None`` disables EMA). Validation uses the EMA weights.
        early_stopping_patience: stop after this many epochs without
            validation improvement (requires ``val_data``).
        max_steps: hard limit on optimizer steps (also sets the schedule length).
        restore: weights to keep at the end: ``"best"`` (best validation;
            falls back to ``"ema"``/``"last"`` without validation data),
            ``"last"`` or ``"ema"``.
        device: training device (default: the flow's device). The flow is moved there.
        num_workers: DataLoader workers for generic datasets.
        amp: ``False``, ``"bf16"`` (or ``True``) or ``"fp16"`` autocast.
        compile: wrap the flow with ``torch.compile`` for the training forward pass.
        noise: a :class:`NoiseAugmentation` (soft training / noise augmentation).
        batch_context: pass the second element of ``(x, c)`` batches as context.
        prior_weight: weight of ``log_prior()`` terms exposed by the flow's
            (sub)modules; the prior is scaled by ``1/N_train`` so that the
            objective is the per-sample MAP objective. ``None`` = 1.0 if such
            modules exist; ``0`` disables.
        max_nonfinite: abort after this many consecutive non-finite steps.
        val_batch_size: batch size for validation (default ``max(batch_size, 1024)``).
        log_every: log the running training loss every this many steps.
        verbose: print one line per epoch.
        callback: called after every epoch with a dict (``epoch``, ``step``,
            ``train_loss``, ``val_loss``, ``lr``, ``flow``); returning ``True``
            stops training.
        seed: random seed.

    Returns:
        A :class:`TrainResult`.
    """
    if restore not in ("best", "last", "ema"):
        raise ValueError("restore must be 'best', 'last' or 'ema'")
    if restore == "ema" and ema_decay is None:
        raise ValueError("restore='ema' requires ema_decay")
    if early_stopping_patience is not None and val_data is None:
        raise ValueError("early stopping requires val_data")
    if amp is True:
        amp = "bf16"
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device) if device is not None else _device_of(flow)
    flow.to(device)
    dtype = _dtype_of(flow)

    context_dim = int(getattr(flow, "context_dim", 0) or 0)
    if noise is not None and noise.uses_context:
        if context_dim < 1:
            raise ValueError("soft training requires a flow with context_dim >= 1")
        if getattr(flow, "default_context", 0.0) not in (0, 0.0):
            warnings.warn("soft training: flow.default_context should be 0 (the clean data distribution)", stacklevel=2)

    batches = _make_batches(train_data, batch_size, True, num_workers)
    try:
        steps_per_epoch = len(batches)
    except TypeError:
        steps_per_epoch = None
    if steps_per_epoch is None and max_steps is None:
        raise ValueError("max_steps is required for iterable datasets without len()")
    total_steps = epochs * steps_per_epoch if steps_per_epoch is not None else max_steps
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    n_train = _num_samples(train_data, batches)

    opt = _make_optimizer(flow, optimizer, lr, weight_decay, betas, scale_lr_mult)
    warmup = warmup_steps if warmup_steps is not None else int(round(warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(warmup, total_steps, schedule, min_lr_ratio))
    scaler = torch.amp.GradScaler(device.type) if amp == "fp16" else None

    ema = EMA(flow, ema_decay) if ema_decay is not None else None
    priors = _prior_modules(flow)
    if prior_weight is None:
        prior_weight = 1.0 if priors else 0.0
    use_prior = prior_weight != 0.0 and len(priors) > 0

    model = torch.compile(flow) if compile else flow
    clip = grad_clip if grad_clip is not None else math.inf
    params = [p for p in flow.parameters() if p.requires_grad]
    val_bs = val_batch_size or max(batch_size, 1024)
    # soft training appends the noise level to the batch context; validation is on clean data (σ = 0)
    val_pad = 1 if (noise is not None and noise.uses_context and batch_context) else 0

    res = TrainResult()
    best = math.inf
    bad_epochs = 0
    consecutive_bad = 0
    step = 0
    t0 = time.time()
    stop = False
    flow.train()

    for epoch in range(1, epochs + 1):
        ep_sum, ep_n = 0.0, 0
        run_sum, run_n = 0.0, 0
        for batch in batches:
            if max_steps is not None and step >= max_steps:
                stop = True
                break
            x, ctx = _unpack(batch, device, dtype, batch_context)
            if noise is not None:
                x, nctx = noise(x, step, total_steps)
                if nctx is not None:
                    ctx = nctx if ctx is None else torch.cat([ctx, nctx], dim=1)
            if ctx is not None and ctx.shape[1] != context_dim:
                raise ValueError(f"context has {ctx.shape[1]} columns but flow.context_dim={context_dim}")

            with _autocast(device, amp):
                lp = model(x, ctx) if ctx is not None else model(x)
                nll = -lp.float().mean()
                loss = nll
                if use_prior:
                    lpr = sum(m.log_prior() for m in priors)
                    loss = nll - prior_weight * lpr / n_train
                if normalize_loss:
                    # nats per dimension: makes grad_clip independent of the data dimension
                    loss = loss / x[0].numel()

            ok = bool(torch.isfinite(loss))
            unscaled = False
            if ok:
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    unscaled = True
                else:
                    loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, clip) if params else torch.zeros(())
                ok = bool(torch.isfinite(gnorm))
            if not ok:
                opt.zero_grad(set_to_none=True)
                if unscaled:
                    # non-finite gradients: let the scaler lower its scale (an update without
                    # a preceding unscale_ -- i.e. a non-finite forward loss -- would raise)
                    scaler.update()  # type: ignore[union-attr]
                res.skipped_steps += 1
                consecutive_bad += 1
                logger.warning("non-finite loss/gradient at step %d (skipped)", step)
                if consecutive_bad >= max_nonfinite:
                    warnings.warn(f"aborting: {consecutive_bad} consecutive non-finite steps", stacklevel=2)
                    res.aborted = True
                    stop = True
                    break
                continue  # skipped steps do not advance the step counter / schedule
            consecutive_bad = 0
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            if ema is not None:
                ema.update()
            step += 1
            v = float(nll.detach())
            ep_sum += v * x.shape[0]
            ep_n += x.shape[0]
            run_sum += v
            run_n += 1
            if log_every and step % log_every == 0:
                res.step_losses.append((step, run_sum / run_n))
                logger.info("step %d  loss %.4f  lr %.3g", step, run_sum / run_n, sched.get_last_lr()[0])
                run_sum, run_n = 0.0, 0

        if ep_n == 0 and stop:
            break
        res.epochs = epoch
        res.train_losses.append(ep_sum / max(ep_n, 1))

        val_loss = None
        if val_data is not None:
            ctx_mgr = ema.swap() if ema is not None else contextlib.nullcontext()
            with ctx_mgr:
                val_loss = evaluate_nll(
                    flow, val_data, val_bs, device, batch_context=batch_context, amp=amp, pad_context=val_pad
                )
                if val_loss < best:
                    best = val_loss
                    res.best_val_loss = val_loss
                    res.best_epoch = epoch
                    res.best_state = _cpu_state(flow)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            res.val_losses.append(val_loss)
            flow.train()

        msg = f"epoch {epoch}: train {res.train_losses[-1]:.4f}" + (f"  val {val_loss:.4f}" if val_loss is not None else "")
        logger.info(msg)
        if verbose:
            print(msg, flush=True)
        if callback is not None:
            info = {"epoch": epoch, "step": step, "train_loss": res.train_losses[-1], "val_loss": val_loss,
                    "lr": sched.get_last_lr()[0], "flow": flow}
            if callback(info):
                stop = True
        if early_stopping_patience is not None and bad_epochs >= early_stopping_patience:
            res.stopped_early = True
            stop = True
        if stop or (max_steps is not None and step >= max_steps):
            break

    res.steps = step
    # ------------------------------------------------------------- restore
    target = restore
    if target == "best" and res.best_state is None:
        target = "ema" if ema is not None else "last"
    if target == "best":
        flow.load_state_dict(res.best_state)
    elif target == "ema":
        ema.copy_to()  # type: ignore[union-attr]
    res.restored = target
    flow.eval()
    res.wall_time = time.time() - t0
    return res


__all__ = [
    "EMA",
    "NoiseAugmentation",
    "TrainResult",
    "evaluate_nll",
    "fit",
    "nats_to_bits_per_dim",
    "tweedie_denoise",
]
