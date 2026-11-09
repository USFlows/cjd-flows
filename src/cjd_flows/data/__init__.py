"""Datasets and preprocessing for CJD flows.

* :mod:`~cjd_flows.data.preprocess` -- dequantization, constant-det scalings,
  standardization, bits/dim conversion.
* :mod:`~cjd_flows.data.toy` -- 2-D toy datasets and an n-D asymmetric GMM.
* :mod:`~cjd_flows.data.tabular` -- MAF benchmarks (POWER, GAS, HEPMASS,
  MINIBOONE, BSDS300).
* :mod:`~cjd_flows.data.images` -- MNIST, Fashion-MNIST, CIFAR-10.

Downloads go to ``$CJD_FLOWS_DATA`` or ``~/.cache/cjd_flows``.
"""

from cjd_flows.data._download import default_root, download_url
from cjd_flows.data.images import DequantizedImageDataset, load_image_tensors, load_images
from cjd_flows.data.preprocess import (
    AffineScale,
    Compose,
    Logit,
    Preprocessing,
    Standardize,
    ToSymmetric,
    ToUnitInterval,
    UniformDequantize,
    bits_per_dim,
    to_symmetric,
    to_unit_interval,
)
from cjd_flows.data.tabular import TABULAR_DATASETS, load_tabular
from cjd_flows.data.toy import (
    TOY_DATASETS,
    GaussianMixture,
    asymmetric_gmm,
    asymmetric_gmm_data,
    checkerboard,
    circles,
    eight_gaussians,
    make_toy,
    pinwheel,
    rings,
    spirals,
    swiss_roll,
    toy_log_prob,
    two_moons,
)

__all__ = [
    "default_root",
    "download_url",
    "DequantizedImageDataset",
    "load_image_tensors",
    "load_images",
    "AffineScale",
    "Compose",
    "Logit",
    "Preprocessing",
    "Standardize",
    "ToSymmetric",
    "ToUnitInterval",
    "UniformDequantize",
    "bits_per_dim",
    "to_symmetric",
    "to_unit_interval",
    "TABULAR_DATASETS",
    "load_tabular",
    "TOY_DATASETS",
    "GaussianMixture",
    "asymmetric_gmm",
    "asymmetric_gmm_data",
    "checkerboard",
    "circles",
    "eight_gaussians",
    "make_toy",
    "pinwheel",
    "rings",
    "spirals",
    "swiss_roll",
    "toy_log_prob",
    "two_moons",
]
