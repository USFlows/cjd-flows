# Quickstart: a 2-D toy flow

Train a uniformly scaling flow with a learnable radial base on the two-moons
dataset:

```python
import torch
from cjd_flows import USFlow, fit
from cjd_flows.data.toy import make_toy

x = make_toy("two_moons", 20_000, seed=0)
flow = USFlow((2,), blocks=8, base="ggmix")   # learnable radial base (generalized-gamma mixture)
result = fit(flow, x, epochs=30, lr=1e-3, ema_decay=0.999)

print(result.train_losses[-1])   # mean training NLL (nats/sample) of the last epoch
```

{func}`~cjd_flows.models.USFlow` returns a {class}`~cjd_flows.core.Flow`, a
plain `torch.nn.Module`. Its main methods:

```python
flow.eval()
with torch.no_grad():
    log_p = flow.log_prob(x[:5])     # exact log-density
    samples = flow.sample((1000,))   # samples from the model
    z = flow.transform(x[:5])        # data -> latent (forward direction)
    x_rec = flow.generate(z)         # latent -> data (inverse direction)

flow.log_abs_det()   # THE Jacobian log-determinant: one scalar for all inputs
```

Because the determinant is constant, `log_prob(x)` is the base log-density of
the latent plus one scalar, so the latent norm ranks inputs by density (for a
radial base with a monotone profile).

## Saving and loading

Flows built with `USFlow` / `VeriFlow` remember their builder arguments, so a
saved file is enough to rebuild them:

```python
from cjd_flows import load, save

save(flow, "flow.pt", metadata={"epochs": 30})
flow = load("flow.pt")   # rebuilt from the stored configuration, in eval mode
```

The file holds tensors, strings and numbers only and is read with
`torch.load(..., weights_only=True)`. A flow with your own architecture is
loaded into a model you construct, `load(path, model=my_flow)`; see
{mod}`cjd_flows.io` for registering your own builders.

## Choosing the base

`base` selects the base distribution (see {func}`~cjd_flows.models.make_base`):

- fixed bases: `"gaussian"` (standard normal) and `"laplace"` (the
  `p`-generalized normal, Laplace for `p = 1`);
- learned single-centre Lp-radial bases, whose 1-D norm distribution is learned
  jointly with the flow, for example `"ggmix"` (mixture of generalized gammas),
  `"ggmix-monotone"`, `"spline"`, `"monotone"` or `"khinchine"`;
- `"radial-mixture"`: several radial components with learnable centres (a
  multi-modal latent instead of a single radial profile).

The exponent of the norm is set with `p`:

```python
flow = USFlow((2,), blocks=8, base="ggmix-monotone", p=1.0)   # L1-radial, monotone profile
```

## Writing your own layer

Any {class}`~cjd_flows.core.Bijection` with a constant log-determinant can be
used in a flow. {func}`cjd_flows.testing.check_bijection` checks the contract
(inverse, log-determinant against autograd, constancy, finite gradients):

```python
from cjd_flows.bijections import SplitAdditiveCoupling
from cjd_flows.conditioners import MLP
from cjd_flows.testing import check_bijection

layer = SplitAdditiveCoupling((4,), lambda i, o: MLP(i, o, hidden=(16,))).double()
check_bijection(layer, torch.randn(8, 4, dtype=torch.float64))
```
