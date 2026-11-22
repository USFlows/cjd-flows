"""Deprecated location of the radial latent-space diagnostics.

The evaluator moved to :mod:`cjd_flows.evaluation`, which supports both v1 and
0.9 flows; this module re-exports it for existing code.
"""

from scipy.stats import norm  # noqa: F401  (kept for backwards compatibility)

from cjd_flows.evaluation import RadialFlowEvaluator

__all__ = ["RadialFlowEvaluator"]
