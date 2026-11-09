"""Lp-radial base distributions with learnable 1-D norm (radius) distributions.

A radial base ``p(z) = h(||z - loc||_p)`` is described by the law of its
radius ``R``; ``h(r) = p_R(r) / (d V_p r^{d-1})``. Learning the 1-D law of
``R`` closes the expressivity gap of volume-preserving / constant-Jacobian
flows (Draxler et al., ICML 2024), and monotone profiles make every density
level set an Lp ball (see :mod:`cjd_flows.radial.levelsets`).

Example::

    from cjd_flows.radial import RadialBase, KhinchineNorm
    base = RadialBase((784,), p=2.0)                       # learnable GG norm, init = N(0, I)
    base = RadialBase((784,), p=1.0, norm=KhinchineNorm(784, 1.0))  # monotone by construction
"""

from .base import RadialBase, RadialMixture
from .geometry import (
    log_sphere_measure,
    log_unit_ball_volume,
    lp_norm,
    reference_exponent,
    reference_log_radius,
    sample_unit_sphere,
)
from .levelsets import (
    density_threshold_radius,
    hdr_pvalue,
    udl_intervals,
    udl_log_threshold,
    udl_radius,
)
from .norms import (
    GammaMixture,
    GeneralizedGamma,
    GeneralizedGammaMixture,
    KhinchineNorm,
    LogNormalMixture,
    LogRadiusFlow,
    MonotoneProfileNorm,
    NormDistribution,
    SplineNorm,
    StudentTNorm,
    WeibullMixture,
    fit_norm_to_radii,
)
from .splines import rq_spline

__all__ = [
    "RadialBase",
    "RadialMixture",
    "NormDistribution",
    "GeneralizedGamma",
    "GeneralizedGammaMixture",
    "GammaMixture",
    "WeibullMixture",
    "LogNormalMixture",
    "StudentTNorm",
    "KhinchineNorm",
    "SplineNorm",
    "LogRadiusFlow",
    "MonotoneProfileNorm",
    "fit_norm_to_radii",
    "log_unit_ball_volume",
    "log_sphere_measure",
    "lp_norm",
    "sample_unit_sphere",
    "reference_exponent",
    "reference_log_radius",
    "udl_radius",
    "hdr_pvalue",
    "density_threshold_radius",
    "udl_log_threshold",
    "udl_intervals",
    "rq_spline",
]
