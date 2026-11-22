# CJD-Flows

CJD-Flows is a PyTorch library of normalizing flows whose Jacobian determinant
is **constant**: $\log|\det J_f(x)|$ does not depend on $x$. We call this model
class **uniformly scaling flows (US flows)**.

A constant determinant gives three things:

- **Exact, cheap likelihoods.** $\log p(x) = \log p_Z(f(x)) + C$, where $C$ is
  a single learnable scalar.
- **Density level-set preservation.** Data-space density level sets are exactly
  the pre-images of latent level sets. With a radial base, the latent norm
  ranks inputs by density, which gives faithful anomaly scores and verifiable
  density constraints.
- **Stable training.** Additive and volume-preserving layers do not suffer from
  exploding or vanishing Jacobians.

The library's main ingredient is the **learnable radial base distribution** of
VeriFlow, first presented at AISoLA 2023 (Abu Zaid, Diekmann, Neider): an
Lp-radial base whose 1-D norm distribution is learned jointly with the flow. A
uniformly scaling flow cannot change the distribution of density values of its
base, so with a fixed base its expressivity is limited (Draxler et al., ICML
2024, prove that volume-preserving flows with a fixed base are not universal).
The learnable norm distribution removes exactly this limitation while keeping
the latent level-set geometry intact.

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
