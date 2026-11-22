"""Channel-wise linear bijections ("one-star convolutions").

Every layer in this module acts on the channel dimension (dim ``1``) and is
shared across all spatial positions, i.e. for ``x`` of shape
``(B, C, *spatial)``

    y[:, :, s] = W @ x[:, :, s] + b        for every spatial position s,

and for vector data ``(B, D)`` it is an ordinary dense affine layer (``D`` is
the channel dimension). This is a 1x1 convolution for images, but works for any
number of spatial dimensions. The Jacobian determinant is constant:

    log|det J| = n_spatial * log|det W|,     n_spatial = prod(event_shape[1:]),

which is why all constructors take ``event_shape``.

Implementation notes:
    * The forward (normalizing / training) direction never inverts a matrix:
      it is a single batched matmul ``baddbmm`` on a ``(B, C, N)`` view, which
      benchmarked fastest on GPU (forward + backward) compared to 1x1
      convolution, ``F.linear`` on channels-last views and ``einsum``.
    * Inverses use the cheap inverse of each parametrization (triangular
      solves, transposes, reversed reflections, ...).
    * :meth:`ChannelLinear.to_matrix` / :meth:`ChannelLinear.as_dense` expose
      the affine map explicitly (e.g. for verifier export), and
      :func:`merge_linear` collapses consecutive channel-linear layers into one
      frozen :class:`AffineFromMatrix`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cjd_flows.core import Bijection, Chain, InverseBijection, ShapeLike, as_shape

__all__ = [
    "ChannelLinear",
    "LULinear",
    "HouseholderLinear",
    "OrthogonalLinear",
    "ActNorm",
    "GlobalScale",
    "Shift",
    "Permutation",
    "Reverse",
    "FixedOrthogonal",
    "AffineFromMatrix",
    "merge_linear",
    "apply_channel_matrix",
]


# --------------------------------------------------------------------------- helpers
def _flat(x: Tensor) -> Tensor:
    """``(B, C, *spatial) -> (B, C, N)`` view (``N = 1`` for vectors)."""
    return x.reshape(x.shape[0], x.shape[1], -1)


def apply_channel_matrix(x: Tensor, W: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Computes ``W @ x[:, :, s] + b`` for every spatial position ``s``.

    Args:
        x: input of shape ``(B, C, *spatial)`` or ``(B, C)``.
        W: matrix of shape ``(C_out, C)``.
        b: optional bias of shape ``(C_out,)``.

    Returns:
        Tensor of shape ``(B, C_out, *spatial)``.
    """
    if x.dim() == 2:
        return F.linear(x, W, b)
    B, C = x.shape[:2]
    xf = _flat(x)
    Wb = W.expand(B, *W.shape)
    if b is None:
        y = torch.bmm(Wb, xf)
    else:
        y = torch.baddbmm(b.view(1, -1, 1), Wb, xf)
    return y.reshape(B, W.shape[0], *x.shape[2:])


def _channel_view(v: Tensor, x: Tensor) -> Tensor:
    """Reshapes a per-channel vector ``(C,)`` for broadcasting against ``x``."""
    return v.view(1, -1, *([1] * (x.dim() - 2)))


def _soft_clamp(s: Tensor, bound: Optional[float], soft: bool) -> Tensor:
    if bound is None:
        return s
    if soft:
        return bound * torch.tanh(s / bound)
    return s.clamp(-bound, bound)


# --------------------------------------------------------------------------- base class
class ChannelLinear(Bijection):
    """Abstract channel-wise affine bijection ``y = W x + b`` (applied on dim 1).

    Subclasses implement :meth:`weight` and may override :meth:`bias`,
    :meth:`inverse_weight`, :meth:`log_abs_det_weight`, :meth:`forward` and
    :meth:`inverse` with cheaper parametrization-specific versions.

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.

    Attributes:
        event_shape: the event shape.
        channels: ``C`` (the matrix size).
        n_spatial: ``prod(event_shape[1:])`` (1 for vectors).
    """

    piecewise_affine = True

    def __init__(self, event_shape: ShapeLike):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        self.channels = self.event_shape[0]
        self.n_spatial = math.prod(self.event_shape[1:])

    # -- parametrization ------------------------------------------------------
    def weight(self) -> Tensor:
        """The ``(C, C)`` matrix ``W``."""
        raise NotImplementedError

    def bias(self) -> Optional[Tensor]:
        """The ``(C,)`` bias ``b`` or ``None``."""
        return None

    def inverse_weight(self) -> Tensor:
        """``W^{-1}`` (generic fallback via ``torch.linalg.inv``; not used for training)."""
        return torch.linalg.inv(self.weight())

    def log_abs_det_weight(self) -> Tensor:
        """``log|det W|`` of the channel matrix (0-dim)."""
        return torch.linalg.slogdet(self.weight()).logabsdet

    def log_abs_det(self) -> Tensor:
        return self.n_spatial * self.log_abs_det_weight()

    # -- maps -----------------------------------------------------------------
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return apply_channel_matrix(x, self.weight(), self.bias())

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        b = self.bias()
        if b is not None:
            z = z - _channel_view(b, z)
        return apply_channel_matrix(z, self.inverse_weight())

    # -- export ---------------------------------------------------------------
    def to_matrix(self) -> Tuple[Tensor, Tensor]:
        """Returns ``(W, b)`` of the channel map (``b`` is zeros if there is no bias)."""
        W = self.weight()
        b = self.bias()
        if b is None:
            b = W.new_zeros(W.shape[0])
        return W, b

    def as_dense(self) -> Tuple[Tensor, Tensor]:
        """Returns ``(A, c)`` of the full affine map on flattened events.

        ``A`` has shape ``(D, D)`` with ``D = prod(event_shape)`` such that
        ``forward(x).flatten(1) == x.flatten(1) @ A.T + c``. Since events are
        flattened channel-major, ``A = kron(W, I_{n_spatial})``.
        """
        W, b = self.to_matrix()
        eye = torch.eye(self.n_spatial, device=W.device, dtype=W.dtype)
        return torch.kron(W.contiguous(), eye), b.repeat_interleave(self.n_spatial)

    def extra_repr(self) -> str:
        return f"event_shape={self.event_shape}"


# --------------------------------------------------------------------------- LU
class LULinear(ChannelLinear):
    """Channel-wise LU-parametrized linear layer ``W = P L U`` with bias.

    ``P`` is a fixed permutation (buffer), ``L`` is unit lower triangular and
    ``U`` is upper triangular with diagonal ``sign * exp(log_s)`` (``sign`` is a
    fixed ``±1`` buffer), so ``log|det W| = sum(log_s)`` and the layer is always
    invertible. The forward direction is one matmul with ``W``; the inverse uses
    two triangular solves (applied either directly to the data when there are
    fewer vectors than channels, or to ``I`` to form ``W^{-1}`` followed by one
    matmul otherwise -- whichever is faster). ``torch.inverse`` is never used.

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        unit_det: if ``True`` the log-diagonal is projected to zero sum
            (``log_s - mean(log_s)``), i.e. ``W ∈ SL(C)`` up to sign: ``|det W| = 1``
            while anisotropic scaling remains possible (GIN trick).
        identity_init: if ``True`` initialize ``L = U = I`` (so ``W = P``). Otherwise
            ``W`` is initialized as a random orthogonal matrix (Glow-style).
        permute: use a random fixed channel permutation ``P``; otherwise ``P = I``
            (with ``identity_init=False`` the orthogonal init is then ``P^T Q``).
        prior_scale: scale ``σ`` of the bilateral log-normal prior used by
            :meth:`log_prior` (``None`` disables the prior).
        max_log_scale: bound on each ``|log_s|`` (``None`` disables clamping).
        soft_clamp: use a smooth ``tanh`` clamp instead of a hard clamp.
        bias: whether to learn a bias.
    """

    def __init__(
        self,
        event_shape: ShapeLike,
        unit_det: bool = False,
        identity_init: bool = True,
        permute: bool = True,
        prior_scale: Optional[float] = None,
        max_log_scale: Optional[float] = 8.0,
        soft_clamp: bool = True,
        bias: bool = True,
    ):
        super().__init__(event_shape)
        C = self.channels
        self.unit_det = unit_det
        self.prior_scale = prior_scale
        self.max_log_scale = max_log_scale
        self.soft_clamp = soft_clamp
        self.volume_preserving = unit_det

        if identity_init:
            perm = torch.randperm(C) if permute else torch.arange(C)
            L0 = torch.zeros(C, C)
            U0 = torch.zeros(C, C)
            sign = torch.ones(C)
            log_s = torch.zeros(C)
        else:
            Q = torch.linalg.qr(torch.randn(C, C, dtype=torch.float64))[0]
            P, L, U = torch.linalg.lu(Q)
            perm = P.argmax(dim=1) if permute else torch.arange(C)
            L0, U0 = L.float(), U.float()
            d = torch.diagonal(U)
            sign = d.sign().float()
            log_s = d.abs().log().float()
        if unit_det:
            log_s = log_s - log_s.mean()
        if max_log_scale is not None and soft_clamp:
            # invert the tanh soft clamp so that log_diag() == log_s at init
            log_s = max_log_scale * torch.atanh((log_s / max_log_scale).clamp(-0.999, 0.999))

        self.lower = nn.Parameter(L0.tril(-1).clone())
        self.upper = nn.Parameter(U0.triu(1).clone())
        self.log_s = nn.Parameter(log_s)
        self.bias_ = nn.Parameter(torch.zeros(C)) if bias else None
        self.register_buffer("sign", sign)
        self.register_buffer("perm", perm.long())
        self.register_buffer("inv_perm", torch.argsort(perm).long())

    # -- parametrization ------------------------------------------------------
    def log_diag(self) -> Tensor:
        """Effective ``log|u_ii|`` after clamping and (optional) zero-sum projection."""
        s = _soft_clamp(self.log_s, self.max_log_scale, self.soft_clamp)
        if self.unit_det:
            s = s - s.mean()
        return s

    def L(self) -> Tensor:
        eye = torch.eye(self.channels, device=self.lower.device, dtype=self.lower.dtype)
        return self.lower.tril(-1) + eye

    def U(self) -> Tensor:
        # eye * v instead of torch.diag(v): ONNX has no diag operator
        eye = torch.eye(self.channels, device=self.upper.device, dtype=self.upper.dtype)
        return self.upper.triu(1) + eye * (self.sign * self.log_diag().exp())

    def weight(self) -> Tensor:
        return (self.L() @ self.U())[self.perm]

    def bias(self) -> Optional[Tensor]:
        return self.bias_

    def inverse_weight(self) -> Tensor:
        eye = torch.eye(self.channels, device=self.lower.device, dtype=self.lower.dtype)
        Linv = torch.linalg.solve_triangular(self.L(), eye, upper=False, unitriangular=True)
        LUinv = torch.linalg.solve_triangular(self.U(), Linv, upper=True)
        return LUinv[:, self.perm]

    def log_abs_det_weight(self) -> Tensor:
        return self.log_diag().sum()

    # -- maps -----------------------------------------------------------------
    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.bias_ is not None:
            z = z - _channel_view(self.bias_, z)
        B, C = z.shape[:2]
        n_vectors = B * self.n_spatial
        if n_vectors >= C:
            # many vectors: W^{-1} via two triangular solves against I (O(C^3)),
            # then one GEMM -- much faster than solving for every vector.
            return apply_channel_matrix(z, self.inverse_weight())
        # few vectors: solve directly, as one 2-D (C, B*N) right-hand side
        z = z.index_select(1, self.inv_perm)
        X = z.reshape(B, C, -1).transpose(0, 1).reshape(C, -1)
        y = torch.linalg.solve_triangular(self.L(), X, upper=False, unitriangular=True)
        y = torch.linalg.solve_triangular(self.U(), y, upper=True)
        return y.reshape(C, B, -1).transpose(0, 1).reshape(z.shape)

    # -- prior ----------------------------------------------------------------
    def prior_std(self) -> Optional[float]:
        """Per-entry std ``σ0 = prior_scale / (n_spatial * sqrt(C))`` of ``log|u_ii|``."""
        if self.prior_scale is None:
            return None
        return self.prior_scale / (self.n_spatial * math.sqrt(self.channels))

    def log_prior(self, include_constants: bool = False) -> Tensor:
        """Log-density of the bilateral log-normal prior on ``diag(U)``.

        Each ``|u_ii|`` is LogNormal(0, σ0²) (with a symmetric random sign, hence
        "bilateral"), i.e. ``log|u_ii| ~ N(0, σ0²)`` i.i.d. with

            σ0 = prior_scale / (n_spatial * sqrt(C)).

        Then ``log|det J| = n_spatial * Σ_i log|u_ii| ~ N(0, prior_scale²)``, so
        ``prior_scale`` directly controls the prior spread of the layer's
        total log-volume change regardless of channel count and resolution.

        Returns the (Gaussian) log-density of the log-diagonal
        ``Σ_i log N(log|u_ii|; 0, σ0²)`` -- i.e. the density w.r.t. the
        ``log_s`` parametrization, *without* the ``-Σ log|u_ii|`` change-of-
        variables term that would give the density of ``|u_ii|`` itself.
        Returns ``0`` if ``prior_scale`` is ``None``.

        Args:
            include_constants: include the Gaussian normalization constants.
        """
        s = self.log_diag()
        sigma = self.prior_std()
        if sigma is None:
            return s.new_zeros(())
        lp = -0.5 * (s / sigma).square().sum()
        if include_constants:
            lp = lp - s.numel() * math.log(sigma * math.sqrt(2 * math.pi))
        return lp

    def extra_repr(self) -> str:
        return f"event_shape={self.event_shape}, unit_det={self.unit_det}"


# --------------------------------------------------------------------------- Householder
class HouseholderLinear(ChannelLinear):
    """Product of ``k`` Householder reflections ``W = H_k ⋯ H_1`` (plus optional bias).

    ``H_i = I - 2 v_i v_iᵀ / ‖v_i‖²``. Forward and inverse apply the reflections
    as rank-1 updates in ``O(k·C)`` per vector; no ``C×C`` matrix is formed.
    ``det W = (-1)^k`` so the layer is volume preserving.

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        n_reflections: number ``k`` of reflections (defaults to ``C``).
        bias: whether to learn a bias.
    """

    volume_preserving = True

    def __init__(self, event_shape: ShapeLike, n_reflections: Optional[int] = None, bias: bool = False):
        super().__init__(event_shape)
        k = self.channels if n_reflections is None else int(n_reflections)
        self.n_reflections = k
        self.vectors = nn.Parameter(torch.randn(k, self.channels))
        self.bias_ = nn.Parameter(torch.zeros(self.channels)) if bias else None

    def bias(self) -> Optional[Tensor]:
        return self.bias_

    def _reflect(self, xf: Tensor, reverse: bool) -> Tensor:
        """Applies the reflections to ``xf`` of shape ``(B, C, N)``."""
        V = self.vectors
        coef = 2.0 / V.square().sum(dim=1)  # (k,)
        order = range(self.n_reflections - 1, -1, -1) if reverse else range(self.n_reflections)
        for i in order:
            v = V[i]
            dots = torch.einsum("c,bcn->bn", v, xf)
            xf = xf - (coef[i] * v).view(1, -1, 1) * dots.unsqueeze(1)
        return xf

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        y = self._reflect(_flat(x), reverse=False).reshape(x.shape)
        if self.bias_ is not None:
            y = y + _channel_view(self.bias_, y)
        return y

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.bias_ is not None:
            z = z - _channel_view(self.bias_, z)
        return self._reflect(_flat(z), reverse=True).reshape(z.shape)

    def weight(self) -> Tensor:
        eye = torch.eye(self.channels, device=self.vectors.device, dtype=self.vectors.dtype)
        # rows of the result are (W e_j)^T, i.e. the result is W^T
        return self._reflect(eye.unsqueeze(-1), reverse=False).squeeze(-1).T

    def inverse_weight(self) -> Tensor:
        return self.weight().T

    def log_abs_det_weight(self) -> Tensor:
        return self.vectors.new_zeros(())

    def extra_repr(self) -> str:
        return f"event_shape={self.event_shape}, n_reflections={self.n_reflections}"


# --------------------------------------------------------------------------- orthogonal
class OrthogonalLinear(ChannelLinear):
    """Learnable special-orthogonal channel mixing (``det W = 1``) with optional bias.

    ``W`` is the Cayley transform ``(I - A)(I + A)^{-1}`` or the matrix
    exponential ``exp(A)`` of a skew-symmetric ``A`` (initialized at ``0``, so
    ``W = I``). The inverse is ``Wᵀ``. Note that the Cayley map cannot represent
    rotations with eigenvalue ``-1``; both maps cover only ``SO(C)``.

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        method: ``"cayley"`` or ``"exp"``.
        bias: whether to learn a bias.
        init_std: std of a random perturbation of ``A`` at init (0 = identity).
    """

    volume_preserving = True

    def __init__(self, event_shape: ShapeLike, method: str = "cayley", bias: bool = True, init_std: float = 0.0):
        super().__init__(event_shape)
        if method not in ("cayley", "exp"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.skew_raw = nn.Parameter(init_std * torch.randn(self.channels, self.channels))
        self.bias_ = nn.Parameter(torch.zeros(self.channels)) if bias else None

    def skew(self) -> Tensor:
        A = self.skew_raw.triu(1)
        return A - A.T

    def weight(self) -> Tensor:
        A = self.skew()
        if self.method == "exp":
            return torch.linalg.matrix_exp(A)
        eye = torch.eye(self.channels, device=A.device, dtype=A.dtype)
        # (I - A) and (I + A)^{-1} commute
        return torch.linalg.solve(eye + A, eye - A)

    def bias(self) -> Optional[Tensor]:
        return self.bias_

    def inverse_weight(self) -> Tensor:
        return self.weight().T

    def log_abs_det_weight(self) -> Tensor:
        return self.skew_raw.new_zeros(())

    def extra_repr(self) -> str:
        return f"event_shape={self.event_shape}, method={self.method!r}"


# --------------------------------------------------------------------------- ActNorm
class ActNorm(ChannelLinear):
    """Per-channel affine ``y = exp(s) * x + b`` with data-dependent initialization.

    On the first forward call in training mode, ``s`` and ``b`` are set so that
    the output has zero mean and unit variance per channel on that batch; the
    persistent buffer ``initialized`` records this (so a loaded state dict is not
    re-initialized).

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        zero_sum: if ``True`` the effective log-scale is ``s - mean(s)`` so that
            ``|det| = 1``. The data init then equalizes the per-channel std (to
            the geometric mean of the input stds) instead of making it one.
        eps: numerical floor for the std in the data init.
    """

    def __init__(self, event_shape: ShapeLike, zero_sum: bool = False, eps: float = 1e-6):
        super().__init__(event_shape)
        self.zero_sum = zero_sum
        self.eps = eps
        self.volume_preserving = zero_sum
        self.log_scale = nn.Parameter(torch.zeros(self.channels))
        self.shift = nn.Parameter(torch.zeros(self.channels))
        self.register_buffer("initialized", torch.tensor(False))

    def scale_log(self) -> Tensor:
        s = self.log_scale
        return s - s.mean() if self.zero_sum else s

    @torch.no_grad()
    def data_init(self, x: Tensor) -> None:
        """Initializes ``s`` and ``b`` from a batch ``x``."""
        dims = [0] + list(range(2, x.dim()))
        mean = x.mean(dim=dims)
        std = x.std(dim=dims, unbiased=False).clamp_min(self.eps)
        self.log_scale.copy_(-std.log())
        s = self.scale_log()
        self.shift.copy_(-mean * s.exp())
        self.initialized.fill_(True)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.training and not bool(self.initialized):
            self.data_init(x.detach())
        s = self.scale_log()
        return x * _channel_view(s.exp(), x) + _channel_view(self.shift, x)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        s = self.scale_log()
        return (z - _channel_view(self.shift, z)) * _channel_view((-s).exp(), z)

    def weight(self) -> Tensor:
        return torch.eye(self.channels, device=self.log_scale.device, dtype=self.log_scale.dtype) * self.scale_log().exp()

    def inverse_weight(self) -> Tensor:
        return torch.eye(self.channels, device=self.log_scale.device, dtype=self.log_scale.dtype) * (-self.scale_log()).exp()

    def bias(self) -> Optional[Tensor]:
        return self.shift

    def log_abs_det_weight(self) -> Tensor:
        return self.scale_log().sum()

    def extra_repr(self) -> str:
        return f"event_shape={self.event_shape}, zero_sum={self.zero_sum}"


# --------------------------------------------------------------------------- global scale
class GlobalScale(ChannelLinear):
    """Uniform scaling ``y = exp(log_scale) * x`` with a single scalar parameter.

    ``log|det J| = D * log_scale`` with ``D = prod(event_shape)``.

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        init: initial scale factor (``> 0``).
    """

    def __init__(self, event_shape: ShapeLike, init: float = 1.0):
        super().__init__(event_shape)
        if init <= 0:
            raise ValueError("init must be positive")
        self.dim = math.prod(self.event_shape)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init)))

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return x * self.log_scale.exp()

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return z * (-self.log_scale).exp()

    def weight(self) -> Tensor:
        return self.log_scale.exp() * torch.eye(
            self.channels, device=self.log_scale.device, dtype=self.log_scale.dtype
        )

    def inverse_weight(self) -> Tensor:
        return (-self.log_scale).exp() * torch.eye(
            self.channels, device=self.log_scale.device, dtype=self.log_scale.dtype
        )

    def log_abs_det_weight(self) -> Tensor:
        return self.channels * self.log_scale

    def log_abs_det(self) -> Tensor:
        return self.dim * self.log_scale

    @torch.no_grad()
    def set_from_latents(self, z: Tensor, method: str = "mle") -> float:
        """Sets the scale in closed form for a standard-normal base.

        ``z`` are samples of the *input* of this layer (the latents produced by
        all preceding layers, before scaling).

        Args:
            z: tensor ``(N, *event_shape)``.
            method:
                ``"mle"``: exact maximum-likelihood scale for fixed preceding
                layers, ``c = sqrt(D / E‖z‖²)`` (maximizes
                ``E[log N(c z)] + D log c``).
                ``"det"``: Draxler et al.'s optimal uniform scale
                ``c = |Σ_z|^{-1/(2D)}``, optimal when the preceding volume-
                preserving part can whiten the latents (requires ``N > D``).

        Returns:
            The new scale factor ``c``.
        """
        zf = z.reshape(z.shape[0], -1).double()
        D = zf.shape[1]
        if method == "mle":
            log_c = 0.5 * (math.log(D) - zf.square().sum(1).mean().log().item())
        elif method == "det":
            zc = zf - zf.mean(0, keepdim=True)
            cov = zc.T @ zc / (zf.shape[0] - 1)
            sign, logdet = torch.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("latent covariance is singular; need more samples than dimensions")
            log_c = -logdet.item() / (2 * D)
        else:
            raise ValueError(f"unknown method {method!r}")
        self.log_scale.fill_(log_c)
        return math.exp(log_c)


# --------------------------------------------------------------------------- shift
class Shift(Bijection):
    """Learnable translation ``y = x + b`` (volume preserving, ``log|det J| = 0``).

    Constant-Jacobian flows built only from additive couplings have no explicit
    bias; the couplings have to centre the latents themselves. A ``Shift``
    before the global scale gives the flow a direct, volume-preserving bias.
    With ``data_init=True`` it is set to minus the mean of the first training
    batch (like :class:`ActNorm`; the persistent buffer ``initialized`` prevents
    re-initialization after loading a state dict), so training starts with
    centred latents. It does not prevent the couplings from moving the latents
    off-centre again during training.

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        per: ``"coordinate"`` (one offset per entry of the event, the default)
            or ``"channel"`` (one offset per channel, shared across positions).
        data_init: initialize from the first forward call in training mode.
    """

    piecewise_affine = True
    volume_preserving = True

    def __init__(self, event_shape: ShapeLike, per: str = "coordinate", data_init: bool = True):
        super().__init__()
        if per not in ("coordinate", "channel"):
            raise ValueError("per must be 'coordinate' or 'channel'")
        self.event_shape = as_shape(event_shape)
        self.per = per
        shape = self.event_shape if per == "coordinate" else (self.event_shape[0],) + (1,) * (len(self.event_shape) - 1)
        self.shift = nn.Parameter(torch.zeros(shape))
        self.register_buffer("initialized", torch.tensor(not data_init))

    @torch.no_grad()
    def data_init(self, x: Tensor) -> None:
        """Sets the offset to minus the mean of ``x`` (``(B, *event_shape)``)."""
        dims = [0] if self.per == "coordinate" else [0] + list(range(2, x.dim()))
        self.shift.copy_(-x.mean(dim=dims, keepdim=True).reshape(self.shift.shape))
        self.initialized.fill_(True)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.training and not bool(self.initialized):
            self.data_init(x.detach())
        return x + self.shift

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return z - self.shift

    def log_abs_det(self) -> Tensor:
        return self.shift.new_zeros(())

    def extra_repr(self) -> str:
        return f"event_shape={self.event_shape}, per={self.per!r}"


# --------------------------------------------------------------------------- permutations
class Permutation(ChannelLinear):
    """Fixed channel permutation ``y[:, i] = x[:, perm[i]]`` (buffer).

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        perm: permutation of ``range(C)``; random if ``None``.
    """

    volume_preserving = True

    def __init__(self, event_shape: ShapeLike, perm: Optional[Sequence[int]] = None):
        super().__init__(event_shape)
        C = self.channels
        perm = torch.randperm(C) if perm is None else torch.as_tensor(perm, dtype=torch.long)
        if perm.shape != (C,) or not torch.equal(perm.sort().values, torch.arange(C)):
            raise ValueError("perm must be a permutation of range(C)")
        self.register_buffer("perm", perm.long())
        self.register_buffer("inv_perm", torch.argsort(perm).long())
        self.register_buffer("_ref", torch.zeros(()), persistent=False)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return x.index_select(1, self.perm)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return z.index_select(1, self.inv_perm)

    def weight(self) -> Tensor:
        return torch.eye(self.channels, device=self._ref.device, dtype=self._ref.dtype)[self.perm]

    def inverse_weight(self) -> Tensor:
        return self.weight().T

    def log_abs_det_weight(self) -> Tensor:
        return self._ref.new_zeros(())


class Reverse(Permutation):
    """Reverses the channel order."""

    def __init__(self, event_shape: ShapeLike):
        C = as_shape(event_shape)[0]
        super().__init__(event_shape, perm=list(range(C - 1, -1, -1)))


class FixedOrthogonal(ChannelLinear):
    """Fixed random Haar-distributed orthogonal channel mixing (buffer).

    Args:
        event_shape: ``(C, *spatial)`` or ``(D,)``.
        seed: optional seed for reproducibility.
        dtype: dtype of the stored matrix (default: ``torch.get_default_dtype()``).
            The matrix is orthogonal to the precision of this dtype; a layer
            built in float32 and converted with ``.double()`` is orthogonal only
            up to ~1e-7, so ``log_abs_det() == 0`` is then exact only to that
            precision. Pass ``dtype=torch.float64`` for float64 work.
    """

    volume_preserving = True

    def __init__(self, event_shape: ShapeLike, seed: Optional[int] = None, dtype: Optional[torch.dtype] = None):
        super().__init__(event_shape)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        else:
            g.seed()
        A = torch.randn(self.channels, self.channels, generator=g, dtype=torch.float64)
        Q, R = torch.linalg.qr(A)
        Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
        self.register_buffer("W", Q.to(dtype or torch.get_default_dtype()))

    def weight(self) -> Tensor:
        return self.W

    def inverse_weight(self) -> Tensor:
        return self.W.T

    def log_abs_det_weight(self) -> Tensor:
        return self.W.new_zeros(())


# --------------------------------------------------------------------------- frozen affine
class AffineFromMatrix(ChannelLinear):
    """Frozen general channel-wise affine map ``y = W x + b``.

    ``W^{-1}`` and ``log|det W|`` are precomputed (in float64) and stored as
    buffers. Used by :func:`merge_linear` / ``simplify()``.

    Args:
        W: ``(C, C)`` invertible matrix.
        b: ``(C,)`` bias or ``None``.
        event_shape: event shape (defaults to ``(C,)``).
    """

    def __init__(self, W: Tensor, b: Optional[Tensor] = None, event_shape: Optional[ShapeLike] = None):
        W = torch.as_tensor(W).detach()
        C = W.shape[0]
        super().__init__(event_shape if event_shape is not None else (C,))
        if W.shape != (C, C) or self.channels != C:
            raise ValueError(f"W must be ({self.channels}, {self.channels}), got {tuple(W.shape)}")
        dtype = W.dtype if W.is_floating_point() else torch.get_default_dtype()
        W64 = W.double()
        b = torch.zeros(C, dtype=dtype, device=W.device) if b is None else torch.as_tensor(b).detach()
        self.register_buffer("W", W.to(dtype).clone())
        self.register_buffer("W_inv", torch.linalg.inv(W64).to(dtype))
        self.register_buffer("b", b.to(device=W.device, dtype=dtype).clone())
        self.register_buffer("logabsdet", torch.linalg.slogdet(W64).logabsdet.to(dtype))

    def weight(self) -> Tensor:
        return self.W

    def inverse_weight(self) -> Tensor:
        # One Newton-Schulz step X <- X (2I - W X): restores full precision
        # e.g. after ``.double()`` of a layer built in float32 (cost O(C^3)).
        X = self.W_inv
        return 2 * X - X @ (self.W @ X)

    def bias(self) -> Optional[Tensor]:
        return self.b

    def log_abs_det_weight(self) -> Tensor:
        return self.logabsdet


# --------------------------------------------------------------------------- merging
def _as_affine(b: Bijection) -> Optional[Tuple[Tuple[int, ...], Tensor, Tensor]]:
    """``(event_shape, W, b)`` in float64 if ``b`` is a channel-linear layer, else ``None``."""
    if isinstance(b, ChannelLinear):
        W, c = b.to_matrix()
        return b.event_shape, W.detach().double(), c.detach().double()
    if isinstance(b, InverseBijection) and isinstance(b.bijection, ChannelLinear):
        inner = b.bijection
        W, c = inner.to_matrix()
        Winv = torch.linalg.inv(W.detach().double())
        return inner.event_shape, Winv, -(Winv @ c.detach().double())
    return None


def _flatten_chain(chain: Bijection) -> List[Bijection]:
    if isinstance(chain, Chain):
        out: List[Bijection] = []
        for b in chain:
            out.extend(_flatten_chain(b))
        return out
    return [chain]


@torch.no_grad()
def merge_linear(chain: Bijection, min_group: int = 2) -> Chain:
    """Collapses runs of consecutive channel-linear layers into :class:`AffineFromMatrix`.

    Nested :class:`Chain` objects are flattened; inverses of channel-linear
    layers are merged too. Layers in a run must share the same ``event_shape``.
    The composition ``f2 ∘ f1`` has ``W = W2 W1`` and ``b = W2 b1 + b2``
    (computed in float64, then cast to the dtype of the first layer).

    Args:
        chain: a :class:`Chain` (or single bijection).
        min_group: minimum run length to merge; shorter runs are kept as is.

    Returns:
        A new :class:`Chain` (sharing the non-merged modules with ``chain``).
    """
    layers = _flatten_chain(chain)
    out: List[Bijection] = []
    run: List[Bijection] = []

    def flush():
        if len(run) >= min_group:
            ref = run[0]
            p = next(iter(ref.parameters()), None)
            if p is None:
                p = next((t for t in ref.buffers() if t.is_floating_point()), None)
            dtype = p.dtype if p is not None else torch.get_default_dtype()
            device = p.device if p is not None else None
            shape, W, c = _as_affine(run[0])
            for layer in run[1:]:
                _, W2, c2 = _as_affine(layer)
                W, c = W2 @ W, W2 @ c + c2
            out.append(AffineFromMatrix(W.to(device=device, dtype=dtype), c.to(device=device, dtype=dtype), shape))
        else:
            out.extend(run)
        run.clear()

    run_shape: Optional[Tuple[int, ...]] = None
    for layer in layers:
        aff = _as_affine(layer)
        if aff is not None and (not run or run_shape == aff[0]):
            run.append(layer)
        else:
            flush()
            if aff is not None:
                run.append(layer)
            else:
                out.append(layer)
        run_shape = aff[0] if aff is not None else None
    flush()
    return Chain(*out)
