from typing import Dict, Iterable, Optional, Union
from src.cjd_flows.transforms import Rotation, CompositeRotation
from src.cjd_flows.linalg import random_orthonormal_matrix
from src.cjd_flows.utils import inv_softplus
import pyro
import math

import torch
from torch.nn.functional import softplus
from torch.distributions import constraints, Distribution, Chi2, Independent as DIndependent
from torch.nn import Module, Parameter

import torch.nn as nn
import torch.distributions as dist
import torch.distributions.transforms as transforms
import pyro.distributions as pyro_dist


class RotatedLaplace(torch.distributions.Distribution):
    """Implements a Laplace distribution that is rotated so that the bounding
    box of the density contours is of minimal (Euclidean) volume."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_enumerate_support = False

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.dim = loc.shape[0]
        self.loc = loc
        self.scale = scale
        self.rotation = random_orthonormal_matrix(self.dim)
        self.laplace = pyro.distributions.Laplace(self.loc, self.scale)
        batch_shape = self.laplace.batch_shape
        if len(batch_shape) > 0:
            self.laplace = pyro.distributions.Independent(
                self.laplace, len(batch_shape)
            )
        self.shape = self.laplace.event_shape
        super().__init__(event_shape=(self.dim,), validate_args=False)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples n points from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)

        return torch.matmul(self.laplace.sample(sample_shape), self.rotation)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        return self.laplace.log_prob(torch.matmul(x, self.rotation.t()))


class Chi(Distribution):
    arg_constraints = {"df": constraints.positive}
    support = constraints.positive # This will be updated to include scale
    has_enumerate_support = False

    def __init__(self, df: int, scale: float = 1.0, validate_args=None):
        """
        Initialize the Chi distribution with degrees of freedom `df`.
        Args:
            df (Tensor): degrees of freedom.
            scale (float): scale parameter.
            validate_args (bool, optional): Whether to validate input parameters. Default: None.
        """
        self.chi2 = Chi2(df)
        self.df = df
        self.scale = scale
        super(Chi, self).__init__(
            self.chi2._batch_shape, self.chi2._event_shape, validate_args=validate_args # This will be updated to include scale
        )

    def sample(self, sample_shape=torch.Size()):
        """
        Generate samples from the Chi distribution.
        Args:
            sample_shape (torch.Size, optional): The size of the sample to draw. Default: torch.Size().
        Returns:
            Tensor: A sample of the specified shape.
        """
        return self.scale * torch.sqrt(self.chi2.sample(sample_shape))

    def log_prob(self, value):
        """
        Calculate the log probability of a given value.
        Args:
            value (Tensor): The value at which to evaluate the log probability.
            Returns: Tensor: The log probability of the value.
        """
        value = value / self.scale
        y = value**2
        return self.chi2.log_prob(y) + torch.log(value * 2) - torch.log(torch.tensor(self.scale))

    def cdf(self, value):
        """
        Calculate the cumulative distribution function (CDF) at a given value.
        Args:
            value (Tensor): The value at which to evaluate the CDF.
        Returns:
            Tensor: The CDF of the value.
        """ 
        value = value / self.scale
        y = value**2
        return self.chi2.cdf(y)

    def entropy(self):
        """
        Calculate the entropy of the distribution.
        Returns:
            Tensor: The entropy of the distribution.
        """
        return self.chi2.entropy() / 2 + torch.log(torch.tensor(2)) + torch.log(torch.tensor(self.scale))


class DistributionModule(Module):
    """Wrapper class to treat distributions as PyTorch modules with learnable parameters."""
    
    def __init__(
        self,
        distribution_class: type,
        n_batch_dims: int = 0,
    ):
        super().__init__()
        self.distribution_class = distribution_class
        self.n_batch_dims = n_batch_dims
        
    @property
    def distribution(self) -> Distribution:
        params = self._get_distribution_params()
        d = self.distribution_class(**params)
        
        # Handle batch dimensions
        nbatch_dims = len(d.batch_shape) - self.n_batch_dims
        if nbatch_dims > 0:
            d = DIndependent(d, nbatch_dims)
        return d

    def _get_distribution_params(self) -> Dict[str, torch.Tensor]:
        """Should be implemented by subclasses to return current parameters"""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        return self.distribution.sample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)

    @property
    def event_shape(self) -> torch.Size:
        return self.distribution.event_shape

    @property
    def batch_shape(self) -> torch.Size:
        return self.distribution.batch_shape
        

class Gamma(DistributionModule):
    def __init__(
        self,
        concentration: torch.Tensor,
        rate: torch.Tensor,
        device: str = "cpu",
    ):
        super().__init__(torch.distributions.Gamma)
        # Use unconstrained parameters + constraints
        self.concentration_unconstrained = Parameter(inv_softplus(concentration))
        self.rate_unconstrained = Parameter(inv_softplus(rate))
        self.to(device)

    def _get_distribution_params(self) -> Dict[str, torch.Tensor]:
        return {
            "concentration": softplus(self.concentration_unconstrained),
            "rate": softplus(self.rate_unconstrained)
        }

class LogNormal(DistributionModule):
    def __init__(
        self,
        loc: torch.Tensor, 
        scale: torch.Tensor, 
        device: str = "cpu",
    ):
        super().__init__(torch.distributions.LogNormal)
        self.loc = Parameter(loc)
        self.scale_unconstrained = Parameter(inv_softplus(scale))
        self.to(device)

    def _get_distribution_params(self) -> Dict[str, torch.Tensor]:
        return {
            "loc": self.loc,
            "scale": softplus(self.scale_unconstrained)
        }

class Laplace(DistributionModule):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        device: str = "cpu",
    ):
        super().__init__(torch.distributions.Laplace)
        self.loc = Parameter(loc)
        self.scale_unconstrained = Parameter(inv_softplus(scale))
        self.to(device)

    def _get_distribution_params(self) -> Dict[str, torch.Tensor]:
        return {
            "loc": self.loc,
            "scale": softplus(self.scale_unconstrained)
        }

class Normal(DistributionModule):
    def __init__(
        self, 
        loc: torch.Tensor, 
        scale: torch.Tensor,
        device: str = "cpu",
    ):
        super().__init__(torch.distributions.Normal)
        self.loc = Parameter(loc)
        self.scale_unconstrained = Parameter(inv_softplus(scale))
        self.to(device)

    def _get_distribution_params(self) -> Dict[str, torch.Tensor]:
        if self.scale_unconstrained.dim() == 0:
            # If scale is a scalar, we need to expand it to match loc's shape
            scale = softplus(self.scale_unconstrained).expand_as(self.loc)
        else:
            scale = softplus(self.scale_unconstrained)
        return {
            "loc": self.loc,
            "scale": scale
        }

class Categorical(DistributionModule):
    def __init__(
        self,
        logits: torch.Tensor,
        device: str = "cpu",
    ):
        super().__init__(torch.distributions.Categorical)
        self.logits = Parameter(logits)
        self.to(device)

    def _get_distribution_params(self) -> Dict[str, torch.Tensor]:
        return {"logits": self.logits}


class UniformUnitLpBall(torch.distributions.Distribution):
    """Implements a uniform distribution on the unit ball."""

    support = constraints.real
    has_enumerate_support = False

    def __init__(self, dim: int, p: float):
        self.p = p
        self.dim = dim
        if self.p == 1:
            self.log_surface_area_unit_ball = (
                (3 / 2) * math.log(self.dim)
                + math.log(2) * self.dim
                - torch.log(torch.arange(1, self.dim + 1)).sum()
            )
        elif self.p == 2:
            self.log_surface_area_unit_ball = (
                math.log(2)
                + (self.dim / 2) * math.log(math.pi)
                - math.lgamma(self.dim / 2)
            )
        elif self.p == math.inf:
            self.log_surface_area_unit_ball = math.log(2) * self.dim + math.log(
                self.dim
            )
        else:
            raise ValueError("p must be 1, 2, or inf.")
        super().__init__(event_shape=(dim,), validate_args=False)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)

        if self.p == 1:
            x = pyro.distributions.Dirichlet(torch.ones(self.dim)).sample(sample_shape)
            dims = (
                pyro.distributions.Categorical(probs=torch.ones(2) / 2).sample(
                    sample_shape + (self.dim,)
                )
                * 2
                - 1
            )
            x = x * dims
        elif self.p == 2:
            x = pyro.distributions.Normal(0, 1).sample(sample_shape + (self.dim,))
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.p == math.inf:
            extremal_dims = pyro.distributions.Categorical(
                torch.ones(self.dim) / self.dim
            ).sample(sample_shape + (1,))
            mask = (
                torch.ones(sample_shape + (self.dim,)).cumsum(dim=-1) - 1
                == extremal_dims
            )

            boundary = torch.ones(sample_shape + (self.dim,))
            hyperplane_distribution = pyro.distributions.Uniform(-boundary, boundary)
            x = hyperplane_distribution.sample()
            x[mask] = 1.0
        else:
            raise ValueError("p must be 1, 2, or inf.")

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""

        return -self.log_surface_area_unit_ball


class RadialDistribution(torch.nn.Module):
    """Implements radial distributions. More precisely, this class realizes
    Lp-radial distributions with specifiable redial distribution.

    Args:
        loc: Location of the distribution
        norm_distribution: Distribution of the radial component
        p: Exponent of the Lp norm used to define the distribution. Currently,
        p = 1, 2, and inf are supported.
    """

    arg_constraints = {"loc": constraints.real}
    support = constraints.real
    has_enumerate_support = False

    def __init__(
        self,
        loc: torch.Tensor,
        norm_distribution: DistributionModule,
        p: float,
        n_batch_dims: int = 0,
        device: str = "cpu",
    ):
        torch.nn.Module.__init__(self)
        self.norm_distribution = norm_distribution
        self.event_shape=loc.shape[n_batch_dims:]
        self.batch_shape=loc.shape[:n_batch_dims]
        
        if not isinstance(p, float):
            raise ValueError("p must be a float.")
        if p <= 0:
            raise ValueError("p must be positive.")

        self.device = device
        self.loc = torch.nn.Parameter(loc.to(device))
        self.p = p
        self.n_batch_dims = n_batch_dims
        self.dim = torch.prod(torch.tensor(loc.shape[self.n_batch_dims :]))
        self.shape = loc.shape
        self.unit_ball_distribution = UniformUnitLpBall(self.dim, p)
        self.to(self.device)
        
    
    def _merge_intervals(self, intervals: torch.Tensor) -> torch.Tensor:
            """returns start and end of consecutive intervals of indices"""
            if len(intervals) == 1:
                return torch.tensor([[intervals[0], intervals[0]]])
            
            intervals = intervals.sort().values
            merged = []
            start = intervals[0]
            end = intervals[0]
            for i in range(1, len(intervals)):
                if intervals[i] == end + 1:
                    end = intervals[i]
                else:
                    merged.append([start, end])
                    start = intervals[i]
                    end = intervals[i]
            merged.append([start, end])
            return torch.tensor(merged, device=self.device)
    

    def radial_udl_profile(self, q: Optional[float] = None, threshold: Optional[float] = None, r_max: float = 100000, n_samples: int = 10000) -> torch.Tensor:
        """Computes an approximate representation of the upper density level set of distribution as intervals radial intervals.
        
        Args:
            q: probability level.
        Returns:
            Tensor: Upper density level set of the distribution given as tensor of shape n_intervals x 2. Each row reresents an interval [a,b] where a is the lower bound and b is the upper bound of the radial component.
        """
        if q is not None and threshold is not None:
            raise ValueError("Only one of 'q' or 'threshold' can be provided.")
        if q is None and threshold is None:
            raise ValueError("Either 'q' or 'threshold' must be provided.")

        rs = torch.linspace(1e-20, r_max, n_samples, device=self.device).reshape(-1, 1)

        # radial profile function 
        profile = self.norm_distribution.log_prob(rs) - self.log_delta_volume(self.p, rs).flatten()
        
        # compute threshold
        if q is not None:
            sample = self.norm_distribution.sample((n_samples,)).to(self.device)
            logprob = self.norm_distribution.log_prob(sample) - self.log_delta_volume(self.p, sample).flatten()
            sorted_logprob, _ = torch.sort(logprob, descending=True)
            threshold_idx = int(n_samples * q)
            threshold = sorted_logprob[threshold_idx]

        # compute intervals
        indices = torch.arange(n_samples, device=self.device)
        indices = indices[profile > threshold]

    
        
        return rs.flatten()[self._merge_intervals(indices)]    
    
    def radial_ldl_profile(self, q: Optional[float] = None, threshold: Optional[float] = None, r_max: float = 100000, n_samples: int = 10000) -> torch.Tensor:
        """Computes an approximate representation of the lower density level set of distribution as intervals radial intervals.
        
        Args:
            q: probability level.
        Returns:
            Tensor: Lower density level set of the distribution given as tensor of shape n_intervals x 2. Each row reresents an interval [a,b] where a is the lower bound and b is the upper bound of the radial component.
        """
        if q is not None and threshold is not None:
            raise ValueError("Only one of 'q' or 'threshold' can be provided.")
        if q is None and threshold is None:
            raise ValueError("Either 'q' or 'threshold' must be provided.")

        rs = torch.linspace(1e-20, r_max, n_samples, device=self.device).reshape(-1, 1)

        # radial profile function 
        profile = self.norm_distribution.log_prob(rs) - self.log_delta_volume(self.p, rs).flatten()
        
        # compute threshold
        if q is not None:
            sample = self.norm_distribution.sample((n_samples,)).to(self.device)
            logprob = self.norm_distribution.log_prob(sample) - self.log_delta_volume(self.p, sample).flatten()
            sorted_logprob, _ = torch.sort(logprob, descending=False)
            threshold_idx = int(n_samples * q)
            threshold = sorted_logprob[threshold_idx]

        # compute intervals
        indices = torch.arange(n_samples, device=self.device)
        indices = indices[profile <= threshold]

        return rs.flatten()[self._merge_intervals(indices)]

    def r_profile(self, r: torch.Tensor) -> torch.Tensor:
        """Computes the radial profile of the distribution at radius r.
        
        Args:
            r: radius (batch)
        Returns:
            Tensor: Radial profile of the distribution at radius r.
        """
        if isinstance(r, torch.Tensor):
            r = r.to(self.device)
        else:
            r = torch.tensor(r, device=self.device)

        log_prob_norm = self.norm_distribution.log_prob(r.unsqueeze(-1)).squeeze(-1)
        log_dV = self.log_delta_volume(self.p, r)

        return log_prob_norm - log_dV      

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        peel = False
        if sample_shape is None:
            sample_shape = (1,)
            peel = True
        else:
            sample_shape = tuple(sample_shape)

        r = self.norm_distribution.sample(sample_shape).to(self.device)
        r = r.repeat(
            *[1 for _ in sample_shape],
            *[1 for _ in range(self.n_batch_dims)],
            *tuple(self.event_shape),
        )

        u = self.unit_ball_distribution.sample(
            sample_shape + tuple(self.batch_shape)
        ).to(self.device)
        u = u.reshape(*sample_shape, *self.shape)
        x = r * u

        if peel:
            x = x.squeeze(0)

        return x + self.loc

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        x = x - self.loc

        event_dims = tuple(range(x.dim() - len(self.event_shape), x.dim()))
        r = x.norm(p=self.p, dim=event_dims)

        log_prob_norm = self.norm_distribution.log_prob(r.unsqueeze(-1)).squeeze(-1)
        log_dV = self.log_delta_volume(self.p, r)

        return log_prob_norm - log_dV

    def log_delta_volume(
        self, p: int, r: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Computes the differential log-volume of an $L^p$ ball with radius r.
        Currently, $p=1,2,\text{ or }\infty$ is implemented

        Args:
            p: p norm
            r: radius (batch)
        Returns:
            Differential volume (batch)
        """
        if p == 1:
            # V_1^d'(r) = (2r)**(d-1) / (d-1)!
            log_denominator = sum([math.log(i) for i in range(1, self.dim)])
            log_dv = (
                math.log(2) * self.dim + torch.log(r) * (self.dim - 1) - log_denominator
            )
        elif p == 2:
            # V_2^d'(r) = d * (pi)^(d/2) * r^(d-1) / Gamma(d/2 + 1)
            log_numerator = (
                math.log(self.dim)
                + (self.dim / 2) * math.log(math.pi)
                + (self.dim - 1) * torch.log(r)
            )
            log_dv = log_numerator - math.lgamma((self.dim / 2) + 1)
        elif p == math.inf:
            # V_\infty^d'(r) = d * (pi)^(d/2) * r^(d-1) / Gamma(d/2 + 1)
            log_dv = (
                math.log(self.dim)
                + self.dim * math.log(2)
                + (self.dim - 1) * torch.log(r)
            )
        else:
            raise ValueError(f"p={p} not implemented. Use p=1,2, or infinity")

        return log_dv


class Categorical(DistributionModule, torch.distributions.Categorical):
    """Wrapper class for the Categorical distribution."""

    def __init__(
        self,
        logits: torch.Tensor,
        device: str = "cpu",
        *args,
        **kwargs
    ):
        """Initializes the Categorical distribution."""
        DistributionModule.__init__(self, torch.distributions.Categorical, *args, **kwargs)
        torch.distributions.Categorical.__init__(
            self,
            logits=logits,
            validate_args=False,
            *args,
            **kwargs
        )
        self._logits = torch.nn.Parameter(logits)
        self.register_generated_arg(
            "logits",
            lambda: self.logits
        )
        
        self.to(device)
        
    @property
    def logits(self) -> torch.Tensor:
        """Returns the logits of the distribution."""
        return self._logits
    @logits.setter
    def logits(self, value: torch.Tensor) -> None:
        self._logits = torch.nn.Parameter(value)
    
    @property
    def probs(self) -> torch.Tensor:
        """Returns the probabilities of the distribution."""
        return torch.nn.functional.softmax(self._logits, dim=-1)

class RadialMM(DistributionModule):

    def __init__(
        self,
        loc: torch.Tensor,
        norm_distribution: torch.distributions.Distribution,
        p: float,
        mixture_weights: torch.Tensor = None,
        n_batch_dims: int = 1,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """Builds a mixture of radial distributions.

        Args:
            loc: Location param. of (B,D)
            norm_distribution: Norm distributions of (B,D).
            p: Order of norm.
            device: Compute deivce.
        """
        
        super().__init__(
            torch.distributions.MixtureSameFamily,
            *args,
            **kwargs
        )
        self.radial_batch = RadialDistribution(
            loc, 
            norm_distribution, 
            p, 
            device=device, 
            n_batch_dims=n_batch_dims
        )

        if mixture_weights is None:
            mixture_weights = torch.ones(norm_distribution.batch_shape)
        else:
            assert isinstance(
                mixture_weights, torch.Tensor
            ), f"`mixture_weights` must be a tensor. Got {type(mixture_weights)}"

        # Move weights to the same device as the distribution
        mixture_weights = mixture_weights.to(device)
        self.mixture_distribution = Categorical(
            logits=mixture_weights
        )
        
        self.register_generated_arg(
            "mixture_distribution",
            lambda: self.mixture_distribution
        )
        self.register_generated_arg(
            "component_distribution",
            lambda: self.radial_batch
        )

        self.to(device)

   

class LMM(torch.nn.Module, torch.distributions.MixtureSameFamily):
    """Wrapper class for the Laplace Mixture Model (LMM) distribution."""

    def __init__(
        self, loc: torch.Tensor, scale: torch.Tensor, mixture_weights: torch.Tensor,
        device: str = "cpu", *args, **kwargs
    ):
        """Initializes the LMM distribution."""
        torch.nn.Module.__init__(self)
        self._laplace_batch = Laplace(loc, scale, n_batch_dims=1)
        self._mixture_distribution = Categorical(mixture_weights)
        torch.distributions.MixtureSameFamily.__init__(
            self,
            mixture_distribution=self._mixture_distribution,
            component_distribution=self._laplace_batch,
            validate_args=False,
            *args,
            **kwargs,
        )
        self.to(device)

class GammaMM(DistributionModule):
    def __init__(
        self, 
        concentration: torch.Tensor, 
        rate: torch.Tensor, 
        mixture_weights: torch.Tensor,
        device: str = "cpu",
    ):
        super().__init__(torch.distributions.MixtureSameFamily)
        # Store unconstrained parameters
        self.concentration_unconstrained = Parameter(inv_softplus(concentration))
        self.rate_unconstrained = Parameter(inv_softplus(rate))
        self.mixture_logits = Parameter(mixture_weights)
        self.to(device)

    def _get_distribution_params(self) -> Dict[str, Union[Distribution, torch.Tensor]]:
        # Apply constraints
        concentration = softplus(self.concentration_unconstrained)
        rate = softplus(self.rate_unconstrained)
        
        # Move component dimension to last position
        comp_dim = 0  # Component dimension is first
        permute_order = list(range(1, concentration.dim())) + [comp_dim]
        concentration_perm = concentration.permute(*permute_order)
        rate_perm = rate.permute(*permute_order)
        
        # Create distributions
        comp_dist = torch.distributions.Gamma(concentration_perm, rate_perm)
        mix_dist = torch.distributions.Categorical(logits=self.mixture_logits)
        
        return {
            "mixture_distribution": mix_dist,
            "component_distribution": comp_dist
        }

class Independent(torch.nn.Module, torch.distributions.Independent):
    """Wrapper class for the Independent distribution."""

    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        reinterpreted_batch_ndims: int = 0,
        *args,
        **kwargs
    ):
        """Initializes the Independent distribution."""
        torch.nn.Module.__init__(self)
        self._base_distribution = base_distribution
        torch.distributions.Independent.__init__(
            self,
            self._base_distribution,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            *args,
            **kwargs
        )

class MixtureModel(DistributionModule):
    """Base class for mixture models of distributions on R_>=0."""
    
    def __init__(
        self,
        distribution_class,
        param_names,
        param_constraints,
        *params,
        mixture_weights,
        device="cpu"
    ):
        super().__init__(distribution_class=dist.MixtureSameFamily)
        self.component_distribution_class = distribution_class
        self.param_names = param_names
        self.param_constraints = param_constraints
        
        # Store unconstrained parameters
        self.unconstrained_params = nn.ParameterList()
        for i, (name, param) in enumerate(zip(param_names, params)):
            constraint = param_constraints.get(name)
            if constraint == constraints.positive:
                self.unconstrained_params.append(nn.Parameter(inv_softplus(param)))
            else:
                self.unconstrained_params.append(nn.Parameter(param))
        
        # Mixture weights (logits)
        self.mixture_logits = nn.Parameter(mixture_weights)
        self.to(device)
    
    def _get_constrained_params(self):
        params = []
        for i, name in enumerate(self.param_names):
            constraint = self.param_constraints.get(name)
            param = self.unconstrained_params[i]
            if isinstance(constraint, type(constraints.positive)) and constraint.lower_bound == 0.0:
                params.append(softplus(param))
            else:
                params.append(param)
        return params
    
    def _get_distribution_params(self):
        # Apply constraints
        params = self._get_constrained_params()
        
        # Permute component dimension to last
        #params_perm = []
        #for i, name in enumerate(self.param_names):
        #    permute_order = list(range(1, params[i].dim())) + [0]
        #    params_perm.append(params[i].permute(permute_order))

        # Create component distribution
        comp_dist = self.component_distribution_class(**dict(zip(self.param_names, params)))
        
        # Permute mixture logits
        #logits_perm = self.mixture_logits.permute(permute_order)
        mix_dist = dist.Categorical(logits=self.mixture_logits)
        
        return {
            "mixture_distribution": mix_dist,
            "component_distribution": comp_dist
        }
    
    @property
    def distribution(self) -> Distribution:
        return super().distribution


class GMM(MixtureModel):
    def __init__(
        self, 
        loc: torch.Tensor, 
        covariance_matrix: torch.Tensor, 
        mixture_weights: torch.Tensor, 
        device: str = "cpu",
    ):
        param_constraints = {
            "loc": constraints.real,
            "covariance_matrix": constraints.positive_definite
        }

        super().__init__(
            dist.MultivariateNormal,
            ["loc", "covariance_matrix"],
            param_constraints,
            loc,
            covariance_matrix,
            mixture_weights=mixture_weights,
            device=device
        )

class LogNormalMM(MixtureModel):
    """Mixture of Log-Normal distributions."""
    def __init__(self, loc, scale, mixture_weights, device="cpu"):
        param_constraints = {"loc": None, "scale": constraints.positive}
        super().__init__(
            dist.LogNormal,
            ["loc", "scale"],
            param_constraints,
            loc,
            scale,
            mixture_weights=mixture_weights,
            device=device
        )

class WeibullMM(MixtureModel):
    """Mixture of Weibull distributions."""
    def __init__(self, scale, concentration, mixture_weights, device="cpu"):
        param_constraints = {
            "scale": constraints.positive,
            "concentration": constraints.positive
        }
        super().__init__(
            dist.Weibull,
            ["scale", "concentration"],
            param_constraints,
            scale,
            concentration,
            mixture_weights=mixture_weights,
            device=device
        )
