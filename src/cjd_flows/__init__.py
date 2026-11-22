"""CJD-Flows: normalizing flows with constant Jacobian determinant.

Quickstart::

    import torch
    from cjd_flows import USFlow, fit

    flow = USFlow((2,), blocks=8, base="ggmix")
    fit(flow, data, epochs=20)
    flow.log_prob(x); flow.sample((1000,))

The legacy pyro-based API (``cjd_flows.flows``, ``cjd_flows.transforms``,
``cjd_flows.distributions``) remains importable with the ``[legacy]`` extra.
"""

from cjd_flows import bijections, conditioners, radial
from cjd_flows.core import BaseDistribution, Bijection, Chain, Flow, StandardNormal
from cjd_flows.io import load, save
from cjd_flows.models import USFlow, VeriFlow, make_base
from cjd_flows.radial import RadialBase, RadialMixture
from cjd_flows.train import fit

__version__ = "1.0.0"

__all__ = [
    "Bijection", "Chain", "Flow", "BaseDistribution", "StandardNormal",
    "RadialBase", "RadialMixture", "USFlow", "VeriFlow", "make_base", "fit", "save", "load",
    "bijections", "conditioners", "radial", "__version__",
]
