import numpy as np

from torch.utils.data import Dataset, default_collate

from pyro import distributions as dist
from typing import List, Dict, Literal, Any, Iterable, Optional, Type, Union, Tuple
import torch
from cjd_flows.distributions import RadialDistribution, Independent
from cjd_flows.sophia import SophiaG

from cjd_flows.transforms import (
    ScaleTransform,
    MaskedCoupling,
    LUTransform,
    InverseTransform,
    BaseTransform,
    BlockAffineTransform,
    HouseholderTransform,
    SequentialAffineTransform
)

def _get_batch(data: Union[torch.Tensor, Dataset], indices: np.ndarray) -> torch.Tensor:
    """Returns the data points ``data[indices]`` as a single tensor.

    Supports tensors, datasets that accept fancy indexing (e.g.
    :class:`torch.utils.data.TensorDataset`) and generic
    map-style datasets (item-wise access + collation). If items are tuples
    (e.g. ``(x, label)``), only the first entry is returned.
    """
    if isinstance(data, torch.Tensor):
        return data[torch.as_tensor(indices, device=data.device)]
    try:
        batch = data[indices]
    except (TypeError, IndexError, KeyError, ValueError, NotImplementedError):
        batch = default_collate([data[int(i)] for i in indices])
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(np.asarray(batch), dtype=torch.get_default_dtype())
    return batch


class Flow(torch.nn.Module):
    """Base implementation of a flow model"""

    # Export mode determines whether the log_prob or the sample function is exported to onnx
    export_modes = Literal["log_prob", "sample"]
    export: export_modes = "log_prob"
    device = "cpu"  

    def forward(self, x: torch.Tensor):
        """Dummy implementation of forward method for onnx export. The self.export attribute
        determines whether the log_prob or the sample function is exported to onnx.
        To obtain the typical foward behavior, use the _forward method."""
        if self.export == "log_prob":
            return self.log_prob(x)
        elif self.export == "sample":
            return self.sample()
        elif self.export == "forward":
            return self._forward(x)
        elif self.export == "backward":
            return self.backward(x)
        else:
            raise ValueError(f"Unknown export mode {self.export}")
    
    def _default_context(
        self, batch_size: int, device: Union[str, torch.device]
    ) -> Optional[torch.Tensor]:
        """Context used when none is given explicitly.

        For soft-trained models (SoftFlow) the conditioners expect a noise-level
        context; the clean model corresponds to noise level 0, so a zero context
        of shape (batch, 1) is used. For all other models no context is used.
        The same default is applied in every direction (forward, backward,
        log_prob, sample) so that they are mutually consistent.
        """
        if getattr(self, "soft_training", False):
            return torch.zeros(batch_size, 1, device=device)
        return None

    @staticmethod
    def _apply_layer(layer, direction: str, x: torch.Tensor, context: Optional[torch.Tensor]):
        fn = getattr(layer, direction)
        if context is None:
            return fn(x)
        return fn(x, context=context)

    def _forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """Internal forward method that applies all layers in the flow
        
        Args:
            x: input tensor to the flow.
            context: optional context (defaults to :meth:`_default_context`).
        Returns:
            x: output tensor after applying all layers in the flow.
        """
        if context is None:
            context = self._default_context(x.shape[0], x.device)
        for layer in self.layers:
            x = self._apply_layer(layer, "forward", x, context)
        return x

    def backward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        """Internal backward method that applies all layers in the flow in reverse order
        
        Args:
            x: input tensor to the flow.
            context: optional context (defaults to :meth:`_default_context`).
        Returns:
            x: output tensor after applying all layers in the flow in reverse order.
        """
        if context is None:
            context = self._default_context(x.shape[0], x.device)
        for layer in reversed(self.layers):
            x = self._apply_layer(layer, "backward", x, context)
        return x

    def __init__(
        self,
        base_distribution,
        layers,
        soft_training: bool = False,
        training_noise_prior=None,
        device: str = "cpu",
        *args,
        **kwargs,
    ) -> None:
        if training_noise_prior is None:
            training_noise_prior = dist.Uniform(0, 1e-6)
            
        super().__init__(*args, **kwargs)

        self.soft_training = soft_training
        self.training_noise_prior = training_noise_prior
        self.layers = layers
        self.trainable_layers = torch.nn.ModuleList(
            [l for l in layers if isinstance(l, torch.nn.Module)]
        )
        self.base_distribution = base_distribution
        self.to(device)
        self.device = device

        # Redeclare all batch dimensions to event dimensions
        # This is a sanitary measure to avoid pyro from creating a batch of transforms
        # rather than a single transform.
        batch_shape = self.base_distribution.batch_shape
        if len(batch_shape) > 0:
            self.base_distribution = Independent(
                self.base_distribution, len(batch_shape)
            )

        self.transform = dist.TransformedDistribution(self.base_distribution, layers)

    def log_prior(self) -> torch.Tensor:
        """Returns the log prior of the model parameters. The model is trained in maximum posterior fashion, i.e.
        $$argmax_{\\theta} \\log p_{\\theta}(D) + \\log p_{prior}(\\theta)$$ By default, this ia the constant zero, which amounts
        to maximum likelihood training (improper uniform prior).
        """
        return 0
    

    def fit(
        self,
        data_train: Dataset,
        optim: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = SophiaG,
        optim_params: Dict[str, Any] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        gradient_clip: float = None,
        device: torch.device = None,
        epochs: int = 1,
        prior_weight: Optional[float] = None,
    ) -> List[float]:
        """
        Fitting method. Allows basic configuration of the optimizer and other
        fitting parameters.

        The per-batch loss is the maximum-a-posteriori objective normalised per
        data point::

            loss = mean_i(-log p(x_i)) - prior_weight * log_prior(theta)

        With the default ``prior_weight = 1 / len(data_train)`` one epoch of
        training optimises ``-(1/N) * [sum_i log p(x_i) + log_prior(theta)]``,
        i.e. the proper (data-size normalised) log posterior. Before v1.0 the
        prior was added with weight 1 to a *mean* NLL, i.e. it was weighted N
        times too strongly; pass ``prior_weight=1.0`` to restore that behaviour.

        Args:
            data_train: training data. Either a tensor of shape (N, *event_shape)
                or a dataset whose items (or fancy-indexed batches) are tensors
                or ``(x, y)`` tuples (only ``x`` is used).
            optim: optimizer class or an already constructed optimizer instance.
                Passing an instance keeps the optimizer state (e.g. momentum)
                across several calls of ``fit``.
            optim_params: keyword arguments for the optimizer class (ignored if
                ``optim`` is an instance).
            batch_size: number of samples per optimization step.
            shuffle: whether to shuffle the data in every epoch.
            gradient_clip: max. gradient norm (no clipping if None).
            device: training device. Defaults to mps/cuda/cpu (in that order).
            epochs: number of epochs.
            prior_weight: weight of the log prior in the loss (see above).

        Returns:
            Loss curve (mean loss per epoch).
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

        model = self.to(device)

        # Create the optimizer once (or reuse a given instance) instead of
        # re-creating it, which would reset its state.
        if isinstance(optim, torch.optim.Optimizer):
            optimizer = optim
        else:
            optimizer = optim(model.parameters(), **(optim_params or {}))

        N = len(data_train)
        if N == 0:
            raise ValueError("Training data is empty.")
        if prior_weight is None:
            prior_weight = 1.0 / N

        epoch_losses = []
        for _ in range(epochs):
            losses = []
            if shuffle:
                order = np.random.permutation(N)
            else:
                order = np.arange(N)

            for idx in range(0, N, batch_size):
                sample = _get_batch(data_train, order[idx : idx + batch_size])
                sample = sample.to(device)

                if self.soft_training:
                    noise = self.training_noise_prior.sample([sample.shape[0]]).to(device)
                    # Repeat noise for all data dimensions
                    sigma = noise.view(-1, *([1] * (sample.dim() - 1))).expand_as(sample)
                    e = torch.normal(torch.zeros_like(sample), sigma)
                    sample = sample + e
                    noise = noise.unsqueeze(-1).detach()
                    # scaling of noise for the conditioning recommended by SoftFlow paper
                    noise = noise * 2 / self.training_noise_prior.high
                else:
                    noise = None

                optimizer.zero_grad()
                loss = (
                    -model.log_prob(sample, context=noise).mean()
                    - prior_weight * model.log_prior()
                )
                loss.backward()
                losses.append(float(loss.detach()))
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                if not self.is_feasible():
                    raise RuntimeError("Model is not invertible")

                model.transform.clear_cache()
            epoch_losses.append(float(np.mean(losses)))

        return epoch_losses

    def to_onnx(self, path: str, export_mode: export_modes = "log_prob") -> None:
        """Saves the model as onnx file

        Args:
            path: path to save the model.
            export_mode: export mode. Can be "log_prob" or "sample".
        """
        mode_cache = self.export
        self.export = export_mode
        dummy_input = self.base_distribution.sample()
        torch.onnx.export(self, dummy_input, path, verbose=True)
        self.export = mode_cache

    def log_prob(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns the models log-densities for the given samples

        Args:
            x: sample tensor.
        """

        if context is None:
            context = self._default_context(x.shape[0], x.device)
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            if context is not None:
                y = layer.backward(x, context=context)
                log_det = log_det - layer.log_abs_det_jacobian(y, x, context=context)
            else:
                y = layer.backward(x)
                log_det = log_det - layer.log_abs_det_jacobian(y, x)
            x = y

        return self.base_distribution.log_prob(x) + log_det

    def sample(
        self, sample_shape: Iterable[int] = None, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns n_sample samples from the distribution

        Args:
            n_sample: sample shape.
        """
        if sample_shape is None:
            sample_shape = [1]

        y = self.base_distribution.sample(sample_shape)
        return self._forward(y, context=context)

    def to(self, device) -> None:
        """Moves the model to the given device"""
        self.device = device
        # self.layers = torch.nn.ModuleList([l.to(device) for l in self.layers])
        self.trainable_layers = torch.nn.ModuleList(
            [l.to(device) for l in self.trainable_layers]
        )
                    
        self._distribution_to(device)
        return super().to(device)

    def is_feasible(self) -> bool:
        """Checks is the model parameters meet all constraints"""
        return all(
            [l.is_feasible() for l in self.layers if isinstance(l, BaseTransform)]
        )

    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to meet non-zero constraints"""
        for l in self.layers:
            if isinstance(l, BaseTransform) and not l.is_feasible():
                l.add_jitter(jitter)

    def _distribution_to(self, device: str) -> None:
        """Moves the base distribution to the given device"""
        pass

    def calibrated_latent_radial_udl_profile(self, q: float, calibration_dataset: torch.Tensor, r_max: float = 10000, n_samples: int = 10000, cut_to_data_tail: bool = True) -> torch.Tensor:
        """
        Computes the radial_udl_profile of the base distribution that contains a q's fraction of the latent representations of the calibration set.
        Base distribution must be of type RadialDistribution in order for the method to be defined.

        Args:
            self: A USFlow with a base distribution of type RadialDistribution.
            q (float): The fraction of latent representations to be contained in the UDL profile.
            calibration_dataset (torch.Tensor): The calibration dataset.
            r_max (float): Maximum radius for the radial profile.
            n_samples (int): Number of samples for the radial profile.
            cut_to_data_tail (bool): Whether to cut the profile to the data tail.

        Returns:
            torch.Tensor: Upper density level set of the distribution given as tensor of shape n_intervals x 2.
                        Each row represents an interval [a,b] where a is the lower bound and b is the upper bound of the radial component.
        """
        if not isinstance(self.base_distribution, RadialDistribution):
            raise TypeError("The base distribution of the flow must be of type RadialDistribution.")

        # Get latent representations
        with torch.no_grad():
            latent_representations = self.backward(calibration_dataset)
            latent_log_probs = self.base_distribution.log_prob(latent_representations)
        
        latent_log_probs, _ = torch.sort(latent_log_probs, descending=True)
        threshold_idx = int(len(latent_log_probs) * q)
        threshold = latent_log_probs[threshold_idx]

        log_prob_max = latent_log_probs[0]

        # Compute radial UDL profile
        baseprofile = self.base_distribution.radial_udl_profile(threshold=threshold, r_max=r_max, n_samples=n_samples)


        def intersect_intervals(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """gets two unions of disjoint intervals as nx2 vectors and returns the intersection of these intervals
            
            Args:
                a: first union of disjoint intervals as nx2 vector.
                b: second union of disjoint intervals as nx2 vector.
            Returns:
                A tensor of shape (m, 2) where m is the number of intervals in the intersection.
            
            Example:
            >>> a = torch.tensor([[0, 2], [3, 5], [6, 8]])
            >>> b = torch.tensor([[1, 3], [5, 7]])
            >>> intersect_intervals(a, b)
            tensor([[1, 2],
                    [7, 8]])
            """
            # copy the intervals to avoid modifying the input tensors
            a = a.clone()
            b = b.clone()

            def sort_intervals(a, b) -> torch.Tensor:
                if a[0, 0] > b[0, 0]:
                    a, b = b, a 
                return a, b

            # ensure that the intervals are sorted by their start values
            a = a[a[:, 0].argsort()]
            b = b[b[:, 0].argsort()]
            a, b = sort_intervals(a, b)

            result = []
            while len(a) > 0 and len(b) > 0:
                a, b = sort_intervals(a, b)
                h = a[0,1]
                b_intersect = b[b[:, 0] <= h]
                b_intersect[:, 1] = torch.minimum(b_intersect[:, 1], h)

                result.append(b_intersect)
                a = a[1:]


            return torch.cat(result, dim=0)
        
        if cut_to_data_tail:
            tail = self.base_distribution.radial_ldl_profile(threshold=log_prob_max, r_max=r_max, n_samples=n_samples)
            profile = intersect_intervals(baseprofile, tail)
        else:
            profile = baseprofile

        return profile

class USFlow(Flow):
    """Implementation of a uniformly scaling flow architecture by using 
    bijective 1X1 convolutions (parametrized by LU decomposed weight matrices),
    additive coupling layers and a scale transform.
    The flow is trained in a maximum posterior fashion by adding a log-normal
    prior on the diagonal elements of the LU weight matrices.
    """
    MASKTYPE = Literal["checkerboard", "channel"]

    def __init__(
        self, 
        base_distribution: dist.Distribution,
        in_dims: List[int], 
        coupling_blocks: int,
        conditioner_cls: Type[torch.nn.Module],
        conditioner_args: Dict[str, Any], 
        soft_training = False, 
        prior_scale: Optional[float] = None,
        training_noise_prior=None,
        affine_conjugation: bool = False, 
        nonlinearity: Optional[torch.nn.Module] = None,
        lu_transform: int = 1,
        householder: int = 1,
        masktype: MASKTYPE = "checkerboard",
        exact_prior_constants: bool = False,
        *args,
        **kwargs
    ):
        
        layers = []
        self.coupling_blocks = coupling_blocks
        self.in_dims = in_dims
        self.soft_training = soft_training
        self.training_noise_prior = training_noise_prior
        self.conditioner_cls = conditioner_cls
        self.conditioner_args = conditioner_args
        self.prior_scale = prior_scale
        self.exact_prior_constants = exact_prior_constants
        #self.nonlinearity = nonlinearity
        if masktype == "checkerboard" :
            self.mask_Generator = USFlow.create_checkerboard_mask 
        elif masktype == "channel":
            self.mask_Generator = USFlow.create_channel_mask
        else:
            raise ValueError(f"Unknown mask type {masktype}")
        
        
        if lu_transform < 0:
            raise ValueError("Number of LU transforms must be non-negative")
        self.lu_transform = lu_transform
        if householder < 0:
            raise ValueError(
                "Number of Householder vectors transforms must be non-negative"
            )
        self.householder = householder
        
        mask = self.mask_Generator(in_dims)
        for i in range(coupling_blocks):
            
            affine_layers = []
            # LU layer
            for _ in range(lu_transform):
                lu_layer = LUTransform(in_dims[0], prior_scale)
                affine_layers.append(lu_layer)
            # Householder layer
            if householder > 0:
                householder_layer = HouseholderTransform(
                        dim=in_dims[0],
                        nvs=householder,
                        device=self.device
                )
                affine_layers.append(householder_layer)
            
            # Create block affine layer
            block_affine_layer = None
            if len(affine_layers) > 0:
                block_affine_layer = BlockAffineTransform(
                    in_dims,
                    SequentialAffineTransform(
                        affine_layers
                    )
                )
                layers.append(block_affine_layer)
            
            coupling_layer = MaskedCoupling(
                mask,
                conditioner_cls(**conditioner_args),
            )
            layers.append(coupling_layer)
            
            # Inverse affine transformation
            if affine_conjugation and block_affine_layer is not None:
                layers.append(InverseTransform(block_affine_layer))
            # alternate mask
            mask = 1 - mask
            
        # Scale layer
        lu_layer = LUTransform(in_dims[0], prior_scale) 
        block_affine_layer = BlockAffineTransform(
            in_dims,
            lu_layer
        )
        layers.append(block_affine_layer)       
        scale_layer = ScaleTransform(in_dims)
        layers.append(scale_layer)
        
        super().__init__(
            base_distribution,
            layers,
            soft_training=soft_training,
            training_noise_prior=training_noise_prior,
            *args,
            **kwargs
        )

        
    @classmethod
    def create_checkerboard_mask(
        cls, in_dims, invert: bool = False
    ) -> torch.Tensor:
        """Creates a checkerboard mask of size $(h,w)$.

        Args:
            h (_type_): height
            w (_type_): width
            invert (bool, optional): If True, inverts the mask. Defaults to False.
        Returns:
            Checkerboard mask of height $h$ and width $w$.
        """
        axes = [torch.arange(d, dtype=torch.int32) for d in in_dims]
        ax_idxs = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        
        mask = torch.fmod(ax_idxs.sum(dim=0), 2)
        mask = mask.to(torch.float32).view(1, *in_dims)
        if invert:
            mask = 1 - mask
        return mask
    
    @classmethod
    def create_channel_mask(
        cls, in_dims, invert: bool = False
    ) -> torch.Tensor:
        """Creates a checkerboard mask of size $(h,w)$.

        Args:
            h (_type_): height
            w (_type_): width
            invert (bool, optional): If True, inverts the mask. Defaults to False.
        Returns:
            Checkerboard mask of height $h$ and width $w$.
        """
        axes = [torch.arange(d, dtype=torch.int32) for d in in_dims]
        ax_idxs = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        
        mask = torch.fmod(ax_idxs[0], 2)
        mask = mask.to(torch.float32).view(1, *in_dims)
        if invert:
            mask = 1 - mask
        return mask
        
    def log_prior(self) -> torch.Tensor:
        """Returns the log prior of the model parameters. The model is trained in maximum posterior fashion, i.e.
        $$argmax_{\\theta} \\log p_{\\theta}(D) + \\log p_{prior}(\\theta)$$ By default, this ia the constant zero, which amounts
        to maximum likelihood training (improper uniform prior).
        """
        if self.prior_scale is not None:
            log_prior = 0
            for p in self.layers:
                try:
                    log_prior += p.log_prior(
                        include_constants=self.exact_prior_constants
                    )
                except TypeError:
                    # Backward-compatible fallback for transforms that do not
                    # expose include_constants.
                    log_prior += p.log_prior()
            return log_prior
        else:
            return 0
        
    def to(self, device) -> None:
        """Moves the model to the given device"""
        self.device = device
        # self.layers = torch.nn.ModuleList([l.to(device) for l in self.layers])
        self.trainable_layers = torch.nn.ModuleList(
            [l.to(device) for l in self.trainable_layers]
        )
                    
        self._distribution_to(device)
        return super().to(device)
    
    def simplify(self) -> Flow:
        """Simplifies the flow by removing LU/Householder layers and replacing 
        them with a PlaneBijectiveLinear layer"""
        layers = []
        for l in self.layers:
            layers.append(l.simplify())
        return Flow(self.base_distribution, layers)