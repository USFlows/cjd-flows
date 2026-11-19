# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [semantic versioning](https://semver.org/).

## [1.0.0] - unreleased

A new, pyro-free core. The constant Jacobian determinant is now part of the
API contract instead of a property of particular layer combinations. The
pyro-based API of 0.9 stays available as an optional extra.

### Added

- **Core** (`cjd_flows.core`): `Bijection` contract with `forward(x, context)`
  (data → latent), `inverse(z, context)` and `log_abs_det()`, which returns a
  constant 0-dim tensor; `Chain`, `InverseBijection` (`b.inv`), `Identity` and
  `Flow` with `log_prob`, `sample`, `rsample`, `transform`, `generate` and `fit`.
  Plain `nn.Module`s without hidden caches, so `.to()`, `.double()`,
  `state_dict` and `torch.compile` work as usual.
- **Builders** (`cjd_flows.models`): `USFlow(...)` and the verification preset
  `VeriFlow(...)` (piecewise affine, adjoint LU blocks, monotone L1-radial base).
- **Bijections** (`cjd_flows.bijections`): split and masked additive couplings,
  GIN couplings, shift-only autoregressive layers (MADE), LU (free or unit
  determinant), Householder, orthogonal (Cayley / matrix exponential), ActNorm,
  `GlobalScale`, `Shift` (data-initialized bias with unit determinant),
  permutations, Squeeze, Haar wavelet and Patchify; `merge_linear` for export.
- **Conditioners** (`cjd_flows.conditioners`): `MLP`, `ConvNet` (1–3 spatial
  dimensions, optionally gated), `MADE` and `JetConditioner` (ViT with adaLN-zero
  context). All zero-initialize their output; optional spectral normalization
  with `lipschitz_upper_bound`.
- **Radial base distributions** (`cjd_flows.radial`): Lp-radial bases for any
  `p > 0` with learnable norm distributions (generalized Gamma and mixtures,
  monotone variants, Khinchine construction, monotone profile, spline,
  Student-t, LogNormal mixture), multi-centre `RadialMixture`, and level-set
  utilities (`udl_radius`, `hdr_pvalue`, `density_threshold_radius`,
  `udl_intervals`).
- **Training** (`cjd_flows.train.fit`): AdamW with warmup and cosine schedule,
  EMA, gradient clipping, validation and early stopping, bf16/fp16, noise
  augmentation and soft training, a separate learning rate for scale-like
  parameters, data-dependent initialization before the first step.
- **Data** (`cjd_flows.data`): toy 2-D datasets with exact densities, the MAF
  tabular benchmarks, image datasets, uniform dequantization and bits/dim helpers.
- **Scores and export**: anomaly scores (NLL, latent norm, typicality, HDR
  p-value) with AUROC/AUPR/FPR@95 (`cjd_flows.scores`); ONNX export of
  `log_prob`, `sample`, `forward` and `inverse` graphs for all bases
  (`cjd_flows.export`).
- **Testing helpers** (`cjd_flows.testing`): `check_bijection`,
  `jacobian_log_abs_det` and `check_flow_normalization` for custom layers.
- Benchmark scripts in `benchmarks/` (density suite and a CIFAR-10 study).

### Changed

- Packaging moved from Poetry to [uv](https://docs.astral.sh/uv/) and hatchling.
  The core only depends on `torch` and `numpy`; everything else is an extra
  (`legacy`, `onnx`, `eval`, `vision`, `experiments`, `docs`).
- Requires Python ≥ 3.10.
- `pyro-ppl` is no longer a dependency of the core. The 0.9 API
  (`cjd_flows.flows`, `transforms`, `distributions`, `explib`) needs the
  `legacy` (or `experiments`) extra.
- `LULinear.log_prior` uses the exact prior scaling `σ / (h·w·√c)` for the
  diagonal so that the total log-determinant has standard deviation `σ`.

### Fixed (0.9 API)

- Training: `shuffle=False` raised a swallowed `NameError`; bare `except`
  hid batch-loading errors; optimizer instances lost their state between calls;
  the log prior was not weighted by `1/N`.
- Conditioners: `CondConvNet(c_out=-1)` returned one channel too many;
  `ConvNet2D` and `GatedConv` did not preserve the spatial shape.
- Soft training: `sample` and the transform directions ignored the default context.
- Radial distributions: the `p = ∞` sampler never produced points on the
  negative faces; wrong L1 normalizer; wrong sample shapes for multi-dimensional
  events; integer `p` was rejected; a broken duplicate `Categorical` shadowed
  the working one.
- Transforms: Householder and rotation inverses, per-sample LeakyReLU
  log-determinant, block-LU sign, dtype handling of the LU transform, and the
  coupling mask is now a non-persistent buffer.

## [0.9.0]

Last release of the pyro-based implementation.
