# VeriFlow and ONNX export

VeriFlow uses a uniformly scaling flow as a proxy for the data distribution in
neural network verification: train a piecewise-affine flow, take a latent upper
density level set, which is a norm ball, and verify properties over its image.

## A verification-friendly flow

{func}`~cjd_flows.models.VeriFlow` is a preset of
{func}`~cjd_flows.models.USFlow`: ReLU conditioners, masked additive couplings
in adjoint LU blocks and an L1-radial base with a *monotone* learnable profile.
The flow is piecewise affine, and the latent density level sets are L1 balls
(polytopes), so their pre-images can be encoded exactly in MILP/SMT.

```python
from cjd_flows import VeriFlow, fit
from cjd_flows.data.toy import make_toy

x = make_toy("two_moons", 20_000, seed=0)
flow = VeriFlow((2,))
fit(flow, x, epochs=30, lr=1e-3)
assert flow.is_piecewise_affine()
```

## Latent level sets

{func}`~cjd_flows.radial.levelsets.udl_radius` gives the radius `r_q` of the
latent norm ball that holds probability mass `q`. For a monotone profile this
ball is the upper density level set (highest-density region) of mass `q`:

```python
from cjd_flows.radial.levelsets import udl_radius

r = udl_radius(flow.base, 0.95)   # ||z - loc||_1 <= r holds 95 % of the latent mass
```

Because the flow is uniformly scaling, the image of this ball under the
generative direction is the upper density level set of the model in data
space.

## Export to ONNX

Verifiers usually take ONNX models.
{func}`~cjd_flows.export.to_onnx` exports one of four graphs:

| `mode` | input | output | direction |
|---|---|---|---|
| `"log_prob"` | `x` | `log p(x)` | data → density |
| `"forward"` | `x` | `z` | data → latent (normalizing direction) |
| `"inverse"` | `z` | `x` | latent → data (generative direction) |
| `"sample"` | `z` | `x` | generative direction with an externally drawn `z ~ p_Z` (no RNG in the graph) |

In this library `forward` is the normalizing direction (data → latent) used for
likelihoods. Verification queries over a latent level set ("every point in the
ball, mapped to data space, satisfies P") need the generative direction,
`inverse`:

```python
import torch
from cjd_flows.export import to_onnx, verify_onnx

path = to_onnx(flow, "veriflow_inverse.onnx", mode="inverse")
z = torch.randn(16, 2)
print(verify_onnx(flow, path, z))   # max deviation between onnxruntime and torch
```

The channel-linear layers (LU, orthogonal, Householder, ActNorm) are exported
as frozen matrices, and consecutive ones are merged
({func}`~cjd_flows.bijections.linear.merge_linear`). Export needs the `onnx`
extra.
