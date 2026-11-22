# CJD-Flows

CJD-Flows is a PyTorch library of normalizing flows whose Jacobian determinant
is **constant**: $\log|\det J_f(x)|$ does not depend on $x$. We call this model
class **uniformly scaling flows (US flows)**.

A constant determinant gives three things:

- **Exact, cheap likelihoods.** $\log p(x) = \log p_Z(f(x)) + C$, where $C$ is
  a single learnable scalar.
- **Density level-set preservation.** Data-space density level sets are exactly
  the pre-images of latent level sets. With a radial base whose profile
  decreases with the radius (e.g. `"ggmix-monotone"`, `"monotone"`,
  `"khinchine"`), the latent norm ranks inputs by density, which gives faithful
  anomaly scores and verifiable density constraints. Non-monotone profiles can
  have "shell" level sets, where a small latent norm does not mean high density.
- **Constant volume change.** The flow stretches every region of the data space
  by the same factor, so the volume change is one learnable scalar. This says
  nothing about conditioning: an additive coupling can still be an
  ill-conditioned shear, so gradient magnitudes are not guaranteed to be benign.

The library's main ingredient is the **learnable radial base distribution** of
VeriFlow, first presented at AISoLA 2023 (Abu Zaid, Diekmann, Neider): an
Lp-radial base whose 1-D norm distribution is learned jointly with the flow. A
uniformly scaling flow changes the density values of its base only by one
global factor (its constant Jacobian determinant), so with a fixed base its
expressivity is limited (Draxler et al., ICML 2024, prove that
volume-preserving flows with a fixed base are not universal). The learnable
norm distribution removes exactly this limitation while keeping the latent
level-set geometry intact.

```python
from cjd_flows import USFlow, fit
from cjd_flows.data.toy import make_toy

x = make_toy("two_moons", 20_000)
flow = USFlow((2,), blocks=8, base="ggmix")
fit(flow, x, epochs=30, lr=1e-3, ema_decay=0.999)
flow.log_prob(x[:5])
```

```{toctree}
:maxdepth: 2
:caption: User guide

installation
concepts
tutorials/quickstart
tutorials/images
tutorials/veriflow_onnx
tutorials/anomaly_scores
tutorials/evaluation
tutorials/transformer_autoregressive
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
design
legacy
citation
```
