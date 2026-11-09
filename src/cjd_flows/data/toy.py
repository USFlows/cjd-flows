"""Toy datasets (2-D benchmarks and an n-D asymmetric Gaussian mixture).

All samplers return ``float32`` tensors of shape ``(n, d)`` and accept a
``seed`` (or a ``torch.Generator``) for reproducibility. The 2-D generators
follow the widely used FFJORD / "inf_train_gen" definitions.

Exact log densities are available for :func:`eight_gaussians`,
:func:`checkerboard` and :class:`GaussianMixture` (e.g. :func:`asymmetric_gmm`)
via :func:`toy_log_prob` / ``GaussianMixture.log_prob``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Union

import torch
from torch import Tensor, nn

GenLike = Union[int, torch.Generator, None]


def _gen(seed: GenLike) -> Optional[torch.Generator]:
    if isinstance(seed, torch.Generator):
        return seed
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _randn(g, *shape) -> Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _rand(g, *shape) -> Tensor:
    return torch.rand(*shape, generator=g, dtype=torch.float64)


def _shuffle(x: Tensor, g) -> Tensor:
    return x[torch.randperm(x.shape[0], generator=g)]


def _out(x: Tensor) -> Tensor:
    return x.to(torch.float32).contiguous()


# --------------------------------------------------------------------- 2-D
def two_moons(n: int, noise: float = 0.1, seed: GenLike = None) -> Tensor:
    """Two interleaving half circles (sklearn ``make_moons``), scaled by 2 and centred."""
    g = _gen(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = torch.linspace(0, math.pi, n_out, dtype=torch.float64)
    t_in = torch.linspace(0, math.pi, n_in, dtype=torch.float64)
    outer = torch.stack([t_out.cos(), t_out.sin()], 1)
    inner = torch.stack([1 - t_in.cos(), 1 - t_in.sin() - 0.5], 1)
    x = torch.cat([outer, inner]) + noise * _randn(g, n, 2)
    x = x * 2 + torch.tensor([-1.0, -0.2], dtype=torch.float64)
    return _out(_shuffle(x, g))


def circles(n: int, factor: float = 0.5, noise: float = 0.08, seed: GenLike = None) -> Tensor:
    """Two concentric circles (sklearn ``make_circles``) scaled by 3."""
    g = _gen(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = torch.linspace(0, 2 * math.pi, n_out + 1, dtype=torch.float64)[:-1]
    t_in = torch.linspace(0, 2 * math.pi, n_in + 1, dtype=torch.float64)[:-1]
    x = torch.cat(
        [torch.stack([t_out.cos(), t_out.sin()], 1), factor * torch.stack([t_in.cos(), t_in.sin()], 1)]
    )
    x = x + noise * _randn(g, n, 2)
    return _out(_shuffle(x * 3, g))


def rings(n: int, noise: float = 0.08, seed: GenLike = None) -> Tensor:
    """Four concentric rings with radii 3, 2.25, 1.5, 0.75."""
    g = _gen(seed)
    ns = [n // 4] * 3
    ns = [n - sum(ns)] + ns
    parts = []
    for k, m in zip([1.0, 0.75, 0.5, 0.25], reversed(ns)):
        t = torch.linspace(0, 2 * math.pi, m + 1, dtype=torch.float64)[:-1]
        parts.append(k * torch.stack([t.cos(), t.sin()], 1))
    x = torch.cat(parts) * 3.0
    x = _shuffle(x, g) + noise * _randn(g, n, 2)
    return _out(x)


_EIGHT_SCALE = 4.0
_EIGHT_STD = 0.5


def _eight_centers() -> Tensor:
    s = 1.0 / math.sqrt(2)
    c = torch.tensor(
        [(1, 0), (-1, 0), (0, 1), (0, -1), (s, s), (s, -s), (-s, s), (-s, -s)], dtype=torch.float64
    )
    return c * _EIGHT_SCALE


def eight_gaussians(n: int, seed: GenLike = None) -> Tensor:
    """Mixture of 8 isotropic Gaussians on a circle (radius 4/1.414, std 0.5/1.414)."""
    g = _gen(seed)
    idx = torch.randint(0, 8, (n,), generator=g)
    x = _randn(g, n, 2) * _EIGHT_STD + _eight_centers()[idx]
    return _out(x / 1.414)


def eight_gaussians_mixture() -> "GaussianMixture":
    """The exact density of :func:`eight_gaussians` as a :class:`GaussianMixture`."""
    means = _eight_centers() / 1.414
    var = torch.full((8, 2), (_EIGHT_STD / 1.414) ** 2, dtype=torch.float64)
    return GaussianMixture(means, var)


def checkerboard(n: int, seed: GenLike = None) -> Tensor:
    """Uniform density on 8 squares of a 4x4 checkerboard on ``[-4, 4]^2`` (density 1/32)."""
    g = _gen(seed)
    x1 = _rand(g, n) * 4 - 2
    x2_ = _rand(g, n) - torch.randint(0, 2, (n,), generator=g).double() * 2
    x2 = x2_ + torch.floor(x1).remainder(2)
    return _out(torch.stack([x1, x2], 1) * 2)


def checkerboard_log_prob(x: Tensor) -> Tensor:
    """Exact log density of :func:`checkerboard` (``-log 32`` on the support, ``-inf`` outside)."""
    u = x / 2
    inside = (u.abs() <= 2).all(-1)
    on = (torch.floor(u[:, 0]) + torch.floor(u[:, 1])).remainder(2) == 0
    lp = torch.full(x.shape[:1], -math.inf, dtype=x.dtype, device=x.device)
    return lp.masked_fill(inside & on, -math.log(32.0))


def pinwheel(
    n: int,
    num_classes: int = 5,
    radial_std: float = 0.3,
    tangential_std: float = 0.1,
    rate: float = 0.25,
    seed: GenLike = None,
) -> Tensor:
    """Pinwheel with ``num_classes`` spiralling arms."""
    g = _gen(seed)
    per = n // num_classes
    m = per * num_classes
    rads = torch.linspace(0, 2 * math.pi, num_classes + 1, dtype=torch.float64)[:-1]
    feats = _randn(g, m, 2) * torch.tensor([radial_std, tangential_std], dtype=torch.float64)
    feats[:, 0] += 1.0
    labels = torch.arange(num_classes).repeat_interleave(per)
    ang = rads[labels] + rate * feats[:, 0].exp()
    rot = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], 1).reshape(-1, 2, 2)
    x = 2 * torch.einsum("ti,tij->tj", feats, rot)
    if m < n:  # top up so that exactly n points are returned
        x = torch.cat([x, x[torch.randint(0, m, (n - m,), generator=g)]])
    return _out(_shuffle(x, g))


def swiss_roll(n: int, noise: float = 1.0, seed: GenLike = None) -> Tensor:
    """2-D swiss roll (sklearn ``make_swiss_roll`` projected to ``(x, z)``, scaled by 1/5)."""
    g = _gen(seed)
    t = 1.5 * math.pi * (1 + 2 * _rand(g, n))
    x = torch.stack([t * t.cos(), t * t.sin()], 1) + noise * _randn(g, n, 2)
    return _out(x / 5.0)


def spirals(n: int, noise: float = 0.1, seed: GenLike = None) -> Tensor:
    """Two interleaved spirals."""
    g = _gen(seed)
    h = n // 2
    r = _rand(g, h, 1).sqrt() * 540 * (2 * math.pi) / 360
    d1x = -r.cos() * r + _rand(g, h, 1) * 0.5
    d1y = r.sin() * r + _rand(g, h, 1) * 0.5
    a = torch.cat([d1x, d1y], 1)
    x = torch.cat([a, -a]) / 3
    if x.shape[0] < n:
        x = torch.cat([x, x[:1]])
    x = x + noise * _randn(g, n, 2)
    return _out(_shuffle(x, g))


# ------------------------------------------------------------- mixtures
class GaussianMixture(nn.Module):
    """Gaussian mixture with diagonal covariances and exact ``log_prob`` / ``sample``.

    Args:
        means: ``(K, d)`` component means.
        variances: ``(K, d)`` diagonal covariances.
        weights: ``(K,)`` mixture weights (uniform if ``None``).
    """

    def __init__(self, means: Tensor, variances: Tensor, weights: Optional[Tensor] = None):
        super().__init__()
        means = torch.as_tensor(means, dtype=torch.float64)
        variances = torch.as_tensor(variances, dtype=torch.float64).expand_as(means)
        if weights is None:
            weights = torch.full((means.shape[0],), 1.0 / means.shape[0], dtype=torch.float64)
        weights = torch.as_tensor(weights, dtype=torch.float64)
        self.register_buffer("means", means.clone())
        self.register_buffer("variances", variances.clone())
        self.register_buffer("weights", weights / weights.sum())

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    def log_prob(self, x: Tensor) -> Tensor:
        """Exact log density of shape ``(B,)`` (computed in float64, returned in ``x.dtype``)."""
        xd = x.to(self.means.dtype).unsqueeze(1)
        comp = -0.5 * (
            ((xd - self.means) ** 2 / self.variances).sum(-1)
            + self.variances.log().sum(-1)
            + self.dim * math.log(2 * math.pi)
        )
        return torch.logsumexp(comp + self.weights.log(), dim=1).to(x.dtype)

    @torch.no_grad()
    def sample(self, n: int, seed: GenLike = None) -> Tensor:
        g = _gen(seed)
        idx = torch.multinomial(self.weights.cpu(), n, replacement=True, generator=g)
        eps = _randn(g, n, self.dim)
        x = self.means.cpu()[idx] + eps * self.variances.cpu()[idx].sqrt()
        return _out(x).to(self.means.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.log_prob(x)


def asymmetric_gmm(d: int, high: float = 5.0, low: float = 0.5) -> GaussianMixture:
    """Asymmetric two-component GMM of the USF paper.

    Means ``+1_d`` and ``−1_d``, equal weights, ``Σ1 = I`` and
    ``Σ2 = diag(high, …, high, low, …, low)`` (first ``d // 2`` entries ``high``).
    """
    means = torch.stack([torch.ones(d), -torch.ones(d)]).double()
    v2 = torch.full((d,), low, dtype=torch.float64)
    v2[: d // 2] = high
    variances = torch.stack([torch.ones(d, dtype=torch.float64), v2])
    return GaussianMixture(means, variances)


def asymmetric_gmm_data(n: int, d: int = 2, seed: GenLike = None) -> Tensor:
    """Samples from :func:`asymmetric_gmm`."""
    return asymmetric_gmm(d).sample(n, seed)


# ------------------------------------------------------------- registry
TOY_DATASETS: Dict[str, Callable[..., Tensor]] = {
    "two_moons": two_moons,
    "moons": two_moons,
    "circles": circles,
    "rings": rings,
    "eight_gaussians": eight_gaussians,
    "8gaussians": eight_gaussians,
    "checkerboard": checkerboard,
    "pinwheel": pinwheel,
    "swiss_roll": swiss_roll,
    "swissroll": swiss_roll,
    "spirals": spirals,
    "2spirals": spirals,
    "asymmetric_gmm": asymmetric_gmm_data,
}


def make_toy(name: str, n: int, seed: GenLike = None, **kwargs) -> Tensor:
    """Samples ``n`` points of the toy dataset ``name`` (see :data:`TOY_DATASETS`)."""
    try:
        fn = TOY_DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown toy dataset {name!r}; choose from {sorted(TOY_DATASETS)}") from None
    return fn(n, seed=seed, **kwargs)


def toy_log_prob(name: str, x: Tensor, **kwargs) -> Tensor:
    """Exact log density of a toy dataset (where available).

    Supported: ``eight_gaussians``, ``checkerboard``, ``asymmetric_gmm`` (pass ``d``
    implicitly via ``x.shape[1]``).
    """
    if name in ("eight_gaussians", "8gaussians"):
        return eight_gaussians_mixture().to(x.device).log_prob(x)
    if name == "checkerboard":
        return checkerboard_log_prob(x)
    if name == "asymmetric_gmm":
        return asymmetric_gmm(x.shape[1], **kwargs).to(x.device).log_prob(x)
    raise NotImplementedError(f"no closed-form density for toy dataset {name!r}")


__all__: Sequence[str] = [
    "two_moons",
    "circles",
    "rings",
    "eight_gaussians",
    "eight_gaussians_mixture",
    "checkerboard",
    "checkerboard_log_prob",
    "pinwheel",
    "swiss_roll",
    "spirals",
    "GaussianMixture",
    "asymmetric_gmm",
    "asymmetric_gmm_data",
    "TOY_DATASETS",
    "make_toy",
    "toy_log_prob",
]
