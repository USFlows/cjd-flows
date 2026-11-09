"""Anomaly / OOD scores and detection metrics.

Convention: **higher scores mean "more anomalous"**, and detection metrics
treat the out-of-distribution / anomalous samples as the positive class.

Scores (:func:`anomaly_scores`):

* ``nll`` -- ``−log p(x)``.
* ``latent_norm`` -- ``‖f(x) − loc‖_p`` of the latent code (``base.norm_of`` if
  the base provides it, else ``base.loc``/``base.p`` if present, else L2).
  For a CJD flow with a radially *decreasing* base this ranks samples
  exactly like ``nll``.
* ``typicality`` -- ``−log p_R(r)`` with ``r`` the latent norm, if the base has a
  radial ``norm`` distribution (``base.norm.log_prob``); for
  :class:`~cjd_flows.core.StandardNormal` the chi distribution is used.
  Otherwise a fitted :class:`TypicalityScorer` gives ``|nll − mean_ref nll|``.
* ``hdr`` -- ``F_R(r)`` (``base.norm.cdf``; chi CDF for StandardNormal). For
  radially decreasing bases this is the probability mass of the highest
  density region whose boundary passes through ``x`` (an HDR p-value is
  ``1 − hdr``).

Metrics: :func:`auroc`, :func:`aupr`, :func:`fpr_at_tpr` in pure torch.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from cjd_flows.core import StandardNormal

ALL_KINDS = ("nll", "latent_norm", "typicality", "hdr")
DataLike = Union[Tensor, Dataset, DataLoader]


# ------------------------------------------------------------ batching
def _iter_batches(data: DataLike, batch_size: int) -> Iterable[Tensor]:
    if isinstance(data, Tensor):
        for i in range(0, data.shape[0], batch_size):
            yield data[i : i + batch_size]
        return
    if isinstance(data, Dataset) and callable(getattr(data, "get_batch", None)):
        n = len(data)  # type: ignore[arg-type]
        for i in range(0, n, batch_size):
            b = data.get_batch(torch.arange(i, min(i + batch_size, n)))
            yield b[0] if isinstance(b, (tuple, list)) else b
        return
    loader = data if isinstance(data, DataLoader) else DataLoader(data, batch_size=batch_size, shuffle=False)
    for b in loader:
        yield b[0] if isinstance(b, (tuple, list)) else b


def _ref(flow: nn.Module) -> Tensor:
    for t in flow.parameters():
        return t
    for t in flow.buffers():
        if t.is_floating_point():
            return t
    return torch.zeros(())


@torch.no_grad()
def batched_apply(
    flow: nn.Module,
    data: DataLike,
    fn: Callable[[nn.Module, Tensor], Dict[str, Tensor]],
    batch_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Tensor]:
    """Applies ``fn(flow, x_batch) -> {name: (B,) tensor}`` batch-wise and concatenates (on CPU)."""
    ref = _ref(flow)
    device = torch.device(device) if device is not None else ref.device
    out: Dict[str, list] = {}
    was_training = flow.training
    flow.eval()
    try:
        for x in _iter_batches(data, batch_size):
            x = x.to(device=device, dtype=ref.dtype)
            for k, v in fn(flow, x).items():
                out.setdefault(k, []).append(v.detach().cpu())
    finally:
        flow.train(was_training)
    return {k: torch.cat(v) for k, v in out.items()}


# ------------------------------------------------------------ radial info
def _flat(z: Tensor, n_event: int) -> Tensor:
    return z.flatten(start_dim=z.dim() - n_event)


def latent_norm(base: nn.Module, z: Tensor) -> Tensor:
    """Norm of latent codes w.r.t. the base: ``base.norm_of(z)`` or ``‖z − loc‖_p`` (default L2)."""
    if callable(getattr(base, "norm_of", None)):
        return base.norm_of(z)
    n_event = len(getattr(base, "event_shape", (z.shape[-1],)))
    loc = getattr(base, "loc", None)
    if isinstance(loc, Tensor):
        z = z - loc.to(z)
    p = getattr(base, "p", 2.0)
    p = float(p.detach()) if isinstance(p, Tensor) else float(p)
    return torch.linalg.vector_norm(_flat(z, n_event), ord=p, dim=-1)


def _chi_log_prob(r: Tensor, d: int) -> Tensor:
    r = r.double()
    k = d / 2.0
    lp = (d - 1) * torch.log(r) - r**2 / 2 - (k - 1) * math.log(2) - math.lgamma(k)
    return lp


def _chi_cdf(r: Tensor, d: int) -> Tensor:
    r = r.double()
    return torch.special.gammainc(torch.full_like(r, d / 2.0), r**2 / 2)


def _radial_fns(base: nn.Module) -> Tuple[Optional[Callable], Optional[Callable]]:
    """``(log_prob_R, cdf_R)`` of the base's radial (norm) distribution if available."""
    norm = getattr(base, "norm", None)
    if norm is not None and callable(getattr(norm, "log_prob", None)) and not isinstance(norm, (int, float)):
        cdf = getattr(norm, "cdf", None)
        return norm.log_prob, (cdf if callable(cdf) else None)
    if isinstance(base, StandardNormal):
        d = base.dim
        return (lambda r: _chi_log_prob(r, d)), (lambda r: _chi_cdf(r, d))
    return None, None


# ------------------------------------------------------------ typicality
class TypicalityScorer:
    """Reference-based typicality score ``|nll(x) − mean_ref nll|``.

    Usage::

        ts = TypicalityScorer().fit(flow, x_train)
        s = ts.score(flow, x_test)
    """

    def __init__(self):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def fit(self, flow: nn.Module, x_ref: DataLike, batch_size: int = 1024, context: Optional[Tensor] = None) -> "TypicalityScorer":
        nll = _nll_only(flow, x_ref, batch_size, context)
        self.mean = float(nll.double().mean())
        self.std = float(nll.double().std())
        return self

    def score_from_nll(self, nll: Tensor) -> Tensor:
        if self.mean is None:
            raise RuntimeError("TypicalityScorer is not fitted")
        return (nll - self.mean).abs()

    def score(self, flow: nn.Module, x: DataLike, batch_size: int = 1024, context: Optional[Tensor] = None) -> Tensor:
        return self.score_from_nll(_nll_only(flow, x, batch_size, context))


def _lp(flow, x, context):
    return flow.log_prob(x, context) if context is not None else flow.log_prob(x)


def _nll_only(flow, data, batch_size, context) -> Tensor:
    return batched_apply(flow, data, lambda f, x: {"nll": -_lp(f, x, context)}, batch_size)["nll"]


# ------------------------------------------------------------ scores
def anomaly_scores(
    flow: nn.Module,
    x: DataLike,
    kinds: Optional[Sequence[str]] = None,
    batch_size: int = 1024,
    context: Optional[Tensor] = None,
    reference: Optional[TypicalityScorer] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Tensor]:
    """Computes anomaly scores (higher = more anomalous) batch-wise.

    Args:
        flow: a :class:`~cjd_flows.core.Flow`.
        x: tensor, dataset (``x`` or ``(x, y)`` items) or DataLoader.
        kinds: subset of ``("nll", "latent_norm", "typicality", "hdr")``.
            ``None`` computes every score that is available for this flow;
            explicitly requested but unavailable scores raise ``ValueError``.
        batch_size: evaluation batch size.
        context: optional context for conditional flows (default: flow default).
        reference: fitted :class:`TypicalityScorer`, used for ``typicality``
            if the base has no radial norm distribution.
        device: evaluation device (default: the flow's device).

    Returns:
        ``{kind: (N,) tensor}`` on CPU.
    """
    base = flow.base
    lp_r, cdf_r = _radial_fns(base)
    available = {"nll", "latent_norm"}
    if lp_r is not None or reference is not None:
        available.add("typicality")
    if cdf_r is not None:
        available.add("hdr")
    if kinds is None:
        kinds = [k for k in ALL_KINDS if k in available]
    else:
        kinds = list(kinds)
        for k in kinds:
            if k not in ALL_KINDS:
                raise ValueError(f"unknown score {k!r}; choose from {ALL_KINDS}")
            if k not in available:
                raise ValueError(
                    f"score {k!r} is not available: the base has no radial norm distribution"
                    + (" and no TypicalityScorer reference was given" if k == "typicality" else "")
                )

    def fn(f, xb):
        out: Dict[str, Tensor] = {}
        z = f.transform(xb, context)
        nll = -(f.base.log_prob(z) + f.log_abs_det())
        if "nll" in kinds:
            out["nll"] = nll
        r = None
        if any(k in kinds for k in ("latent_norm", "typicality", "hdr")):
            r = latent_norm(f.base, z)
        if "latent_norm" in kinds:
            out["latent_norm"] = r
        if "typicality" in kinds:
            out["typicality"] = -lp_r(r).to(nll.dtype) if lp_r is not None else reference.score_from_nll(nll)  # type: ignore[union-attr]
        if "hdr" in kinds:
            out["hdr"] = cdf_r(r).to(nll.dtype)  # type: ignore[misc]
        return out

    res = batched_apply(flow, x, fn, batch_size, device)
    return {k: res[k] for k in kinds}


# ------------------------------------------------------------ metrics
def _prep(scores_in: Tensor, scores_out: Tensor) -> Tuple[Tensor, Tensor]:
    s_in = torch.as_tensor(scores_in).detach().flatten().double().cpu()
    s_out = torch.as_tensor(scores_out).detach().flatten().double().cpu()
    if s_in.numel() == 0 or s_out.numel() == 0:
        raise ValueError("both score sets must be non-empty")
    if torch.isnan(s_in).any() or torch.isnan(s_out).any():
        raise ValueError("scores contain NaN")
    return s_in, s_out


def _avg_ranks(s: Tensor) -> Tensor:
    """1-based ranks with ties replaced by their average rank."""
    sorted_s, order = torch.sort(s, stable=True)
    n = s.numel()
    _, inv, counts = torch.unique_consecutive(sorted_s, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    starts = ends - counts.double() + 1
    avg = (starts + ends) / 2
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = avg[inv]
    return ranks


def auroc(scores_in: Tensor, scores_out: Tensor) -> float:
    """Area under the ROC curve, ``P(s_out > s_in) + ½ P(s_out = s_in)`` (out = positive)."""
    s_in, s_out = _prep(scores_in, scores_out)
    n_in, n_out = s_in.numel(), s_out.numel()
    ranks = _avg_ranks(torch.cat([s_in, s_out]))
    u = ranks[n_in:].sum() - n_out * (n_out + 1) / 2
    return float(u / (n_in * n_out))


def _desc_counts(s_in: Tensor, s_out: Tensor) -> Tuple[Tensor, Tensor]:
    """Cumulative TP/FP counts at each distinct threshold (scores descending)."""
    s = torch.cat([s_in, s_out])
    y = torch.cat([torch.zeros(s_in.numel()), torch.ones(s_out.numel())]).double()
    order = torch.argsort(s, descending=True, stable=True)
    s, y = s[order], y[order]
    tp = torch.cumsum(y, 0)
    fp = torch.cumsum(1 - y, 0)
    last = torch.ones_like(s, dtype=torch.bool)
    last[:-1] = s[1:] != s[:-1]  # last index of every group of tied scores
    return tp[last], fp[last]


def aupr(scores_in: Tensor, scores_out: Tensor) -> float:
    """Average precision with the out-of-distribution samples as positives (sklearn semantics)."""
    s_in, s_out = _prep(scores_in, scores_out)
    tp, fp = _desc_counts(s_in, s_out)
    precision = tp / (tp + fp)
    recall = tp / s_out.numel()
    dr = torch.diff(recall, prepend=recall.new_zeros(1))
    return float((dr * precision).sum())


def fpr_at_tpr(scores_in: Tensor, scores_out: Tensor, tpr: float = 0.95) -> float:
    """False-positive rate at the first threshold reaching true-positive rate ``tpr``."""
    s_in, s_out = _prep(scores_in, scores_out)
    tp, fp = _desc_counts(s_in, s_out)
    tprs = tp / s_out.numel()
    idx = int(torch.searchsorted(tprs, torch.tensor([tpr], dtype=tprs.dtype) - 1e-12)[0])
    idx = min(idx, tprs.numel() - 1)
    return float(fp[idx] / s_in.numel())


def detection_metrics(scores_in: Tensor, scores_out: Tensor, tpr: float = 0.95) -> Dict[str, float]:
    """``{"auroc", "aupr", "fpr@95"}`` (key uses ``round(100 * tpr)``)."""
    return {
        "auroc": auroc(scores_in, scores_out),
        "aupr": aupr(scores_in, scores_out),
        f"fpr@{round(100 * tpr)}": fpr_at_tpr(scores_in, scores_out, tpr),
    }


# ------------------------------------------------------------ protocol
def one_class_protocol(
    classes: Iterable[Any],
    make_flow: Callable[[Any], nn.Module],
    get_data: Callable[[Any], Tuple[DataLike, DataLike, DataLike]],
    fit_kwargs: Optional[Dict[str, Any]] = None,
    kinds: Optional[Sequence[str]] = ("nll",),
    batch_size: int = 1024,
    train_fn: Optional[Callable[..., Any]] = None,
) -> Dict[Any, Dict[str, Dict[str, float]]]:
    """One-class anomaly detection protocol ("train on one class, test vs. the rest").

    For every class ``c``: ``train, test_in, test_out = get_data(c)``, a fresh
    flow ``make_flow(c)`` is trained with ``train_fn(flow, train, **fit_kwargs)``
    (default :func:`cjd_flows.train.fit`) and scored on ``test_in`` (normal)
    and ``test_out`` (anomalous).

    Returns:
        ``{c: {kind: {"auroc", "aupr", "fpr@95"}}}``.
    """
    if train_fn is None:
        from cjd_flows.train import fit as train_fn  # type: ignore[no-redef]
    results: Dict[Any, Dict[str, Dict[str, float]]] = {}
    for c in classes:
        train, test_in, test_out = get_data(c)
        flow = make_flow(c)
        train_fn(flow, train, **(fit_kwargs or {}))
        s_in = anomaly_scores(flow, test_in, kinds, batch_size)
        s_out = anomaly_scores(flow, test_out, kinds, batch_size)
        results[c] = {k: detection_metrics(s_in[k], s_out[k]) for k in s_in}
    return results


__all__ = [
    "ALL_KINDS",
    "TypicalityScorer",
    "anomaly_scores",
    "aupr",
    "auroc",
    "batched_apply",
    "detection_metrics",
    "fpr_at_tpr",
    "latent_norm",
    "one_class_protocol",
]
