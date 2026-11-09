# CJD-Flows v1 design

This document fixes the architecture of the pyro-free v1 core. All new code
follows it.

## Contract

See `src/cjd_flows/core.py`:

* `Bijection(nn.Module)` with
  * `forward(x, context=None) -> z` — **normalizing direction** (data → latent),
  * `inverse(z, context=None) -> x` — generative direction,
  * `log_abs_det() -> Tensor` — a 0-dim tensor; constant in `x` and `context`.
  * Class flags: `piecewise_affine` and `volume_preserving`.
* `Chain`, `InverseBijection` (`b.inv`), `Identity`.
* `BaseDistribution(nn.Module)` with `event_shape`, `log_prob(z) -> (B,)` and `sample(sample_shape)`.
* `Flow(bijection, base, context_dim=0, default_context=0.0)` with `log_prob`, `sample`, `rsample`, `transform`, `generate`, `log_abs_det` and `fit`.

## Conventions

* Tensors are `(B, *event_shape)`. Vector data is `(B, D)`. Spatial data is `(B, C, *spatial)` (1–3 spatial dims). The **channel dimension is dim 1**; for vectors, D is the channel dimension.
* No hidden caches and no custom `.to()` overrides. Masks, permutations and constant matrices are **buffers**. Create every tensor with the `device`/`dtype` of the parameters or the input (`x.new_*`, `torch.eye(n, device=..., dtype=...)`).
* `log_abs_det()` must be differentiable w.r.t. parameters and must not depend on the input. Layers whose determinant depends on the spatial size receive `event_shape` at construction.
* Every bijection must pass `cjd_flows.testing.check_bijection` in float64.
* Conditioners are plain `nn.Module`s `(B, C, *spatial)[, context (B, ctx)] -> (B, C_out, *spatial)`. **Their last layer is zero-initialized**, so every coupling starts as the identity.
* Docstrings are Google style with type hints. No bare `except`. Do not import pyro anywhere in the new modules.
* Optional dependencies (scipy, sklearn, onnx, torchvision, matplotlib) are imported lazily inside functions.

## Package layout (new modules)

| Module | Content |
|---|---|
| `core.py` | Contract, `Chain`, `Flow`, `StandardNormal` |
| `testing.py` | `check_bijection`, `jacobian_log_abs_det`, `check_flow_normalization` |
| `bijections/linear.py` | LU (free / SL(n) unit-det), Householder (efficient), orthogonal (Cayley / matrix-exp), ActNorm (free or zero-sum log-scale), `GlobalScale`, permutations, all applied channel-wise ("one-star conv") |
| `bijections/coupling.py` | Additive masked coupling (checkerboard / channel / custom mask), split coupling (channel/spatial, Jet-style), GIN coupling (zero-sum log-scale, bounded) |
| `bijections/autoregressive.py` | Shift-only autoregressive layers (MADE for vectors), det = 1 |
| `bijections/reshape.py` | Squeeze (space-to-depth), Haar wavelet (orthonormal), Patchify |
| `conditioners.py` | `MLP`, `ConvNet` (gated, residual, ND), `MADE`, `JetConditioner` (ViT); optional spectral norm; zero-init output |
| `radial/` | Lp-radial base distributions with learnable 1-D norm distributions (`norms.py`), geometry (`geometry.py`), `RadialBase` (`base.py`), multi-centre radial mixtures, level-set / HDR utilities (`levelsets.py`) |
| `train.py` | `fit()` trainer: DataLoader, AdamW + warmup + cosine, EMA, grad clip, validation + early stopping, optional soft training / noise augmentation, AMP |
| `data/` | Preprocessing (uniform dequantization, affine scaling, bpd), toy 2-D datasets, MAF tabular benchmarks, image datasets |
| `scores.py` | Anomaly / OOD scores (NLL, latent norm, typicality, HDR p-value) and metrics (AUROC, AUPR, FPR@95) |
| `export.py` | ONNX export wrappers (log_prob / sample / forward / inverse), `to_affine_network` |
| `models.py` | High-level builders (`USFlow`, presets) |

Legacy pyro-based modules (`flows.py`, `transforms.py`, `distributions.py`,
`networks.py`, `explib/`) stay importable for backward compatibility. pyro
becomes an optional `[legacy]` extra.

## Theory notes for implementers

* Draxler et al. (ICML 2024): volume-preserving flows with a fixed base are
  not KL-universal. The invariant is the distribution of density values
  (`μ(t) = vol{p>t}`). A learnable 1-D radial (norm) distribution closes this
  gap. The choice of `p` in the Lp norm does not change expressivity.
* Radial profile: `h(r) = p_R(r) / (d V_p r^{d-1})`, with `V_p = (2Γ(1+1/p))^d / Γ(1+d/p)` and `V_∞ = 2^d`. Use `lgamma`; never form `r^{d-1}`.
* Volume coordinate `t = V_p r^d`: `p_T(t) = h(r)`. A radially *decreasing*
  base corresponds to a decreasing 1-D density in `t`.
* Monotonicity: Gamma(k) norms are monotone iff `k ≤ d` and bounded iff
  `k ≥ d`. LogNormal norms are always "shell" profiles (non-monotone). The
  Khinchine construction `R = C·U^{1/d}` gives monotone profiles for any
  law of `C`.
