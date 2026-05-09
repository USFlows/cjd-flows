# CJD-Flows: Constant-Jacobian-Density Flows

CJD-Flows is a Python library for building and evaluating flow-based density estimators with a guaranteed constant Jacobian determinant. In this repository, we use **US flows** (uniformly scaling flows) as the term for the broader model class, and **CJD-Flows** as the library name.

The library is designed for applications that benefit from stable training, exact likelihoods, and predictable latent level-set geometry.

## Used libraries

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Pyro](https://img.shields.io/badge/Pyro-Flow%20Modeling-4B8BBE)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)

## Citation

If you use this library, please cite the VeriFlow paper:

> Zaid, F. A., Neider, D., and Yalciner, M. (2026). VeriFlow: Modeling Distributions for Neural Network Verification. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(33), 28050-28058. https://doi.org/10.1609/aaai.v40i33.40030

GitHub citation support and downloadable formats:

- [`CITATION.cff`](CITATION.cff) (for GitHub's "Cite this repository" widget)
- [`CITATION.bib`](CITATION.bib) (downloadable BibTeX)

<details>
<summary>BibTeX</summary>

```bibtex
@article{Zaid_Neider_Yalciner_2026,
  title   = {VeriFlow: Modeling Distributions for Neural Network Verification},
  author  = {Zaid, Faried Abu and Neider, Daniel and Yal{\c{c}}iner, Mustafa},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {33},
  pages   = {28050--28058},
  year    = {2026},
  month   = mar,
  doi     = {10.1609/aaai.v40i33.40030},
  url     = {https://ojs.aaai.org/index.php/AAAI/article/view/40030}
}
```

</details>

## Core features

- Exact and efficient `log_prob` evaluation and sampling.
- Uniformly scaling architectures with constant Jacobian determinant.
- Piecewise-affine behavior for additive-coupling architectures with (leaky-)ReLU conditioners.
- UDL-preserving structure, enabling interpretable level-set mappings between latent and data spaces.
- ONNX export support for inference graphs (`log_prob` and sampling).

## Architecture overview

CJD-Flows provides a modular implementation of US-flow components:

- Flow models and building blocks in `src/usflows/flows.py`.
- Affine and coupling transforms in `src/usflows/transforms.py`.
- Flexible base distributions in `src/usflows/distributions.py`, including radial distributions for `L1`, `L2`, and `Linf` geometries.

A key component is the **learnable radial norm distribution** (including mixture families), which closes an important expressivity gap for uniformly scaling flows while keeping latent geometry controllable.

## Evaluation suite

The evaluation module in `src/usflows/explib/eval.py` is tailored to radial-base US flows and includes:

- norm-distribution diagnostics (KS, Wasserstein, PP/QQ/KDE),
- radiality diagnostics (sign symmetry and simplex-uniformity tests),
- calibration-oriented latent-space analysis,
- fine-grained mismatch diagnostics that help distinguish over- vs. under-approximation of density level sets.

## Installation

Assuming the package is published on PyPI:

```bash
pip install cjd-flows
```

## Applications

CJD-Flows is intended as a general-purpose density-modeling library. We are currently expanding application examples and will add more soon.

### Example: Neural network verification

One current use case is verification with distribution-aware input restrictions:

- train a US flow as a proxy density model of valid inputs,
- define latent upper density level sets (UDLs),
- map these sets through the flow to input-space constraints,
- verify properties (e.g., robustness/fairness) over likely inputs.

For verification workflows, the library also provides simplification methods that transform trained flow models into verifier-friendly fully connected or convolutional network forms to maximize compatibility with existing tools.

This turns flow density estimates into practically usable constraints for symbolic and abstract verification pipelines.

## Experiments

The repository includes a lightweight YAML-based experiment framework.

```bash
python scripts/run-experiment.py --config <config file> --report_dir <log dir>
```

## Acknowledgment

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logos/baiaa-logo.svg">
  <img src="docs/images/logos/baiaa-logo-black.svg">
</picture>

The Bavarian AI Act Accelerator is a two-year project funded by the Bavarian
State Ministry of Digital Affairs to support SMEs, start-ups, and the public
sector in Bavaria in complying with the EU AI Act. Under the leadership of the
appliedAI Institute for Europe and in collaboration with Ludwig Maximilian
University, the Technical University of Munich, and the Technical University of
Nuremberg, training, resources, and events are being offered. The project
objectives include reducing compliance costs, shortening the time to compliance,
and strengthening AI innovation. To achieve these objectives, the project is
divided into five work packages: project management, research, education, tools
and infrastructure, and community.
