"""Core abstractions of CJD-Flows.

The central invariant of the library is that every bijection has a *constant*
Jacobian determinant, i.e. ``log|det J_f(x)|`` does not depend on ``x`` (nor on
an optional context). The contract therefore separates the map from its
log-determinant:

* ``Bijection.forward(x, context=None) -> z``   (normalizing direction: data -> latent)
* ``Bijection.inverse(z, context=None) -> x``   (generative direction: latent -> data)
* ``Bijection.log_abs_det() -> Tensor``         (0-dim tensor, depends only on parameters)

so that a flow's log-density is

    log p(x) = log p_Z(f(x)) + log|det J_f|

where the second summand is a single scalar shared by all inputs. Consequently,
density level sets in data space are exactly the pre-images of density level
sets of the base distribution ("density level-set preservation").

Conventions
-----------
* Tensors have exactly one leading batch dimension followed by the event
  dimensions, i.e. shape ``(B, *event_shape)``. Vector data has
  ``event_shape == (D,)``; image-like data has ``(C, H, W)`` (any number of
  spatial dimensions is allowed). The *channel* dimension is always dim ``1``.
* Bijections are plain :class:`torch.nn.Module` objects without hidden caches,
  so they work with ``torch.compile``, ``state_dict``, ``.to()``, ``.double()``,
  DDP, etc.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

ShapeLike = Union[int, Sequence[int], torch.Size]


def as_shape(shape: ShapeLike) -> Tuple[int, ...]:
    """Normalizes an ``int``/sequence to a tuple of ints."""
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(s) for s in shape)


def _zero_like_params(module: nn.Module) -> Tensor:
    """0-dim zero tensor on the device/dtype of the module's parameters/buffers."""
    for t in module.parameters():
        return t.new_zeros(())
    for t in module.buffers():
        if t.is_floating_point():
            return t.new_zeros(())
    return torch.zeros(())


class Bijection(nn.Module):
    """Base class for all bijections with constant Jacobian determinant.

    Subclasses implement :meth:`forward`, :meth:`inverse` and (if the
    determinant is not one) :meth:`log_abs_det`.

    Class attributes:
        piecewise_affine: ``True`` if the map is piecewise affine provided its
            conditioner networks are piecewise affine (e.g. ReLU MLPs/CNNs).
            Such flows admit exact SMT/MILP encodings.
        volume_preserving: ``True`` if ``|det J| == 1`` by construction.
    """

    piecewise_affine: bool = True
    volume_preserving: bool = False

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:  # noqa: D401
        raise NotImplementedError

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    def log_abs_det(self) -> Tensor:
        """Constant ``log|det J_f|`` of the forward (normalizing) map as a 0-dim tensor."""
        return _zero_like_params(self)

    def forward_and_log_det(
        self, x: Tensor, context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Convenience API compatible with nflows/normflows-style code.

        Returns ``(z, ladj)`` where ``ladj`` has shape ``(B,)``.
        """
        z = self.forward(x, context)
        return z, self.log_abs_det().expand(x.shape[0])

    def inverse_and_log_det(
        self, z: Tensor, context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x = self.inverse(z, context)
        return x, (-self.log_abs_det()).expand(z.shape[0])

    @property
    def inv(self) -> Bijection:
        """The inverse bijection (a lightweight view sharing parameters)."""
        return InverseBijection(self)

    def is_piecewise_affine(self) -> bool:
        """Whether this bijection (including its submodules) is piecewise affine."""
        return self.piecewise_affine


class VolumePreserving(Bijection):
    """Marker base class for bijections with ``|det J| == 1``."""

    volume_preserving = True

    def log_abs_det(self) -> Tensor:
        return _zero_like_params(self)


class InverseBijection(Bijection):
    """Swaps the directions of a bijection."""

    def __init__(self, bijection: Bijection):
        super().__init__()
        self.bijection = bijection
        self.piecewise_affine = bijection.piecewise_affine
        self.volume_preserving = bijection.volume_preserving

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return self.bijection.inverse(x, context)

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return self.bijection.forward(z, context)

    def log_abs_det(self) -> Tensor:
        return -self.bijection.log_abs_det()

    @property
    def inv(self) -> Bijection:
        return self.bijection

    def is_piecewise_affine(self) -> bool:
        return self.bijection.is_piecewise_affine()


class Chain(Bijection):
    """Composition ``f = f_n ∘ ... ∘ f_1`` (applied in list order in the forward direction)."""

    def __init__(self, *bijections: Union[Bijection, Iterable[Bijection]]):
        super().__init__()
        if len(bijections) == 1 and not isinstance(bijections[0], nn.Module):
            bijections = tuple(bijections[0])
        for b in bijections:
            if not isinstance(b, Bijection):
                raise TypeError(f"Chain elements must be Bijections, got {type(b)}")
        self.bijections = nn.ModuleList(bijections)
        self.volume_preserving = all(b.volume_preserving for b in bijections)

    def __len__(self) -> int:
        return len(self.bijections)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Chain(*list(self.bijections)[idx])
        return self.bijections[idx]

    def __iter__(self):
        return iter(self.bijections)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        for b in self.bijections:
            x = b(x, context)
        return x

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        for b in reversed(self.bijections):
            z = b.inverse(z, context)
        return z

    def log_abs_det(self) -> Tensor:
        total = _zero_like_params(self)
        for b in self.bijections:
            total = total + b.log_abs_det()
        return total

    def is_piecewise_affine(self) -> bool:
        return all(b.is_piecewise_affine() for b in self.bijections)


class Identity(VolumePreserving):
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return x

    def inverse(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return z


class BaseDistribution(nn.Module):
    """Interface for (learnable) base distributions.

    Subclasses provide ``event_shape``, ``log_prob(z) -> (B,)`` and
    ``sample(sample_shape) -> (*sample_shape, *event_shape)``.
    """

    event_shape: Tuple[int, ...]

    def log_prob(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, sample_shape: ShapeLike = ()) -> Tensor:
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tensor:
        return self.log_prob(z)


class StandardNormal(BaseDistribution):
    """Isotropic standard normal base distribution (non-learnable)."""

    def __init__(self, event_shape: ShapeLike):
        super().__init__()
        self.event_shape = as_shape(event_shape)
        self.register_buffer("_dummy", torch.zeros(()), persistent=False)

    @property
    def dim(self) -> int:
        return math.prod(self.event_shape)

    def log_prob(self, z: Tensor) -> Tensor:
        z = z.flatten(start_dim=z.dim() - len(self.event_shape))
        return -0.5 * (z**2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)

    def sample(self, sample_shape: ShapeLike = ()) -> Tensor:
        shape = as_shape(sample_shape) + self.event_shape
        return torch.randn(shape, device=self._dummy.device, dtype=self._dummy.dtype)


def _input_event_shape(bijection: Bijection) -> Optional[Tuple[int, ...]]:
    """Event shape expected by ``bijection`` (``None`` if no layer declares one).

    Uses the ``event_shape`` attribute of the first layer of (nested) chains;
    for an inverted shape-changing layer the input is its ``out_shape``.
    """
    if isinstance(bijection, Chain):
        return _input_event_shape(bijection.bijections[0]) if len(bijection) else None
    if isinstance(bijection, InverseBijection):
        inner = bijection.bijection
        shape = getattr(inner, "out_shape", None)
        if shape is None:
            return _input_event_shape(inner) if not isinstance(inner, Chain) else None
        return as_shape(shape)
    shape = getattr(bijection, "event_shape", None)
    if shape is None:
        return None
    try:
        return as_shape(shape)
    except (TypeError, ValueError):
        return None


class Flow(nn.Module):
    """A normalizing flow with constant Jacobian determinant.

    Args:
        bijection: normalizing map ``f`` (data -> latent). Lists are wrapped in
            a :class:`Chain`.
        base: base distribution ``p_Z`` over the latent space.
        context_dim: dimension of an optional context vector passed to all
            conditioners (e.g. class embeddings or the soft-training noise scale).
        default_context: value used when ``context`` is omitted but
            ``context_dim > 0`` (``0.0`` for soft training, where sigma=0 is
            the clean data distribution).
        event_shape: shape of a single data point. Defaults to the
            ``event_shape`` of the first layer of the bijection (e.g. the input
            shape of a leading ``Squeeze``) and otherwise to the base
            distribution's event shape. Pass it explicitly if the first layer
            does not declare its input shape but the bijection changes the shape.

    The model computes ``log p(x) = log p_Z(f(x)) + log|det J_f|``.
    """

    def __init__(
        self,
        bijection: Union[Bijection, Sequence[Bijection]],
        base: BaseDistribution,
        context_dim: int = 0,
        default_context: Optional[float] = 0.0,
        event_shape: Optional[ShapeLike] = None,
    ):
        super().__init__()
        if not isinstance(bijection, Bijection):
            bijection = Chain(*bijection)
        if event_shape is None:
            # A shape-changing first layer (e.g. ``Squeeze``) knows the data shape;
            # falling back to the base's shape would silently scramble samples.
            event_shape = _input_event_shape(bijection)
        self._event_shape = None if event_shape is None else as_shape(event_shape)
        latent = getattr(base, "event_shape", None)
        if self._event_shape is not None and latent is not None:
            if math.prod(self._event_shape) != math.prod(as_shape(latent)):
                raise ValueError(
                    f"event_shape {self._event_shape} and the base's event_shape "
                    f"{tuple(latent)} have different sizes"
                )
        self.bijection = bijection
        self.base = base
        self.context_dim = context_dim
        self.default_context = default_context

    # ------------------------------------------------------------------ utils
    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of a single data point."""
        if self._event_shape is not None:
            return self._event_shape
        return tuple(self.base.event_shape)

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        """Shape of a single latent point."""
        return tuple(self.base.event_shape)

    @property
    def dim(self) -> int:
        return math.prod(self.event_shape)

    def _context(self, context: Optional[Tensor], batch: int, ref: Tensor) -> Optional[Tensor]:
        if self.context_dim == 0:
            return context
        if context is None:
            if self.default_context is None:
                raise ValueError("This flow requires a context.")
            return ref.new_full((batch, self.context_dim), float(self.default_context))
        if context.dim() == 1:
            context = context.unsqueeze(-1) if self.context_dim == 1 else context.unsqueeze(0)
        if context.shape[0] == 1 and batch > 1:
            context = context.expand(batch, -1)
        return context

    def _ref(self) -> Tensor:
        return _zero_like_params(self)

    # --------------------------------------------------------------- density
    def log_abs_det(self) -> Tensor:
        """The constant ``log|det J_f|`` of the normalizing map."""
        return self.bijection.log_abs_det()

    def transform(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """Maps data to latent space, ``z = f(x)``."""
        return self.bijection(x, self._context(context, x.shape[0], x))

    def generate(self, z: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """Maps latents to data space, ``x = f^{-1}(z)``."""
        return self.bijection.inverse(z, self._context(context, z.shape[0], z))

    def log_prob(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """Exact log-density ``log p(x)`` of shape ``(B,)``.

        Inputs with more than one leading batch dimension are flattened and
        reshaped back.
        """
        n_event = len(self.event_shape)
        batch_shape = x.shape[: x.dim() - n_event]
        if len(batch_shape) != 1:
            x = x.reshape(-1, *self.event_shape)
        z = self.transform(x, context)
        lp = self.base.log_prob(z) + self.log_abs_det()
        return lp.reshape(batch_shape) if len(batch_shape) != 1 else lp

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return self.log_prob(x, context)

    def latent_log_prob(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """``log p_Z(f(x))`` -- equals ``log p(x)`` up to the constant ``log|det J_f|``."""
        return self.base.log_prob(self.transform(x, context))

    @torch.no_grad()
    def sample(
        self,
        sample_shape: ShapeLike = (),
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Draws samples ``x = f^{-1}(z)`` with ``z ~ p_Z``."""
        sample_shape = as_shape(sample_shape)
        n = math.prod(sample_shape) if sample_shape else 1
        z = self.base.sample((n,))
        x = self.generate(z, context)
        return x.reshape(*sample_shape, *self.event_shape)

    def rsample(self, sample_shape: ShapeLike = (), context: Optional[Tensor] = None) -> Tensor:
        """Like :meth:`sample` but differentiable w.r.t. the flow parameters."""
        sample_shape = as_shape(sample_shape)
        n = math.prod(sample_shape) if sample_shape else 1
        z = self.base.sample((n,))
        return self.generate(z, context).reshape(*sample_shape, *self.event_shape)

    def is_piecewise_affine(self) -> bool:
        return self.bijection.is_piecewise_affine()

    def fit(self, *args, **kwargs):
        """Trains the flow; see :func:`cjd_flows.train.fit`."""
        from cjd_flows.train import fit

        return fit(self, *args, **kwargs)
