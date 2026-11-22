# Transformer-autoregressive layers

{class}`~cjd_flows.bijections.autoregressive.TransformerAutoregressive` is a
token-autoregressive layer with a causal transformer, the volume-preserving
counterpart of TarFlow (Zhai et al.). For a sequence of `N` tokens with `D`
features each (event shape `(N, D)`) it computes

$$
z_t = x_t + \mu_t(x_{<t}) \quad \text{(shift only, default)} \qquad
z_t = x_t \odot e^{s_t(x_{<t})} + \mu_t(x_{<t}) \quad (\texttt{scale=True}),
$$

where the log-scales $s_t$ are bounded and centred over the `D` features of
each token, so they sum to zero. In both cases $\log|\det J| = 0$. TarFlow's
per-token scales change the volume; here they do not.

## Building a flow

Patchify the images into tokens, then stack layers with alternating direction
so that every token is conditioned on every other one in some layer:

```python
import torch
from cjd_flows import Flow, RadialBase
from cjd_flows.bijections import GlobalScale, Patchify, TransformerAutoregressive

tokens = (64, 12)   # (3, 16, 16) images, patch size 2 -> 8 * 8 tokens of 2 * 2 * 3 features
layers = [Patchify((3, 16, 16), patch_size=2)]
for i in range(4):
    layers.append(TransformerAutoregressive(tokens, embed_dim=128, depth=2, reverse=i % 2 == 1))
layers.append(GlobalScale(tokens))

flow = Flow(layers, RadialBase(tokens, p=2.0))
x = torch.rand(8, 3, 16, 16) * 2 - 1
print(flow.log_prob(x).shape)   # (8,)
```

The output head of each layer is zero-initialized, so every layer starts as
the identity.

## Likelihood and sampling

- `forward` (data → latent, used by `log_prob` and anomaly scores) is a single
  parallel pass.
- `inverse` (sampling) decodes the tokens one by one and reuses the attention
  keys and values of earlier tokens. Without autograd (`torch.no_grad`,
  `torch.inference_mode`, ONNX export) the keys and values go into preallocated
  buffers; with autograd (e.g. `rsample`) the cache grows by concatenation so
  that gradients flow through the inverse.

```python
flow.eval()
with torch.no_grad():
    samples = flow.sample((4,))   # (4, 3, 16, 16)
```

## Numerics of the inverse

With `scale=False` the inverse is exact up to rounding. With `scale=True`,
rounding errors of earlier tokens are multiplied by up to `exp(|s|)` in later
ones, so large log-scales make the sequential inverse (sampling)
ill-conditioned over long sequences. `forward` and `log_prob` are not
affected. Check `inverse(forward(x))` on trained models, or lower `clamp`.

Under fp16/bf16 autocast the cached inverse rounds the shifts differently from
the parallel `forward`, so `inverse(forward(x))` is only exact to about half
precision. Train with autocast if you like, but sample and invert without it.
