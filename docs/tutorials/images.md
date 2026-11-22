# Image flows

Images use the same builder. For image event shapes `(C, H, W)`,
{func}`~cjd_flows.models.USFlow` uses ConvNet conditioners by default.

```python
import torch
from torch.utils.data import random_split

from cjd_flows import USFlow, fit
from cjd_flows.data import load_images

train_full = load_images("mnist", split="train")   # dequantized, scaled to [-1, 1)
test = load_images("mnist", split="test")
train, val = random_split(train_full, [55_000, 5_000], generator=torch.Generator().manual_seed(0))

flow = USFlow((1, 28, 28), blocks=8, hidden=(128, 128), base="ggmix")
fit(flow, train, val_data=val, epochs=20, batch_size=128, device="cuda")
```

`fit` uses the validation data for early stopping and restores the best
validation checkpoint, so the validation split is carved out of the training
data; the test split is only used for the final number below.

{func}`~cjd_flows.data.images.load_images` downloads MNIST, Fashion-MNIST or
CIFAR-10 to `$CJD_FLOWS_DATA` (default `~/.cache/cjd_flows`) and returns a
{class}`~cjd_flows.data.images.DequantizedImageDataset` that draws fresh
uniform dequantization noise for every batch. `classes=` keeps only some labels
(one-class setups), and `exclude_classes=True` keeps all others.

## Bits per dimension

Likelihoods are computed in model space. To report bits/dim in the dequantized
pixel space, add the log-determinant of the preprocessing (here the affine map
to `[-1, 1)`, which keeps the determinant constant):

```python
import torch
from cjd_flows.data import bits_per_dim
from cjd_flows.train import evaluate_nll

nll = evaluate_nll(flow, test)                  # mean NLL in nats per sample (model space)
bpd = bits_per_dim(-nll, test.dim, test.preprocessing_log_det)
```

Affine preprocessing keeps the constant determinant; a logit transform does
not.

## Transformer conditioners

`conditioner="jet"` uses a ViT conditioner
({class}`~cjd_flows.conditioners.JetConditioner`) instead of the ConvNet:

```python
flow = USFlow((3, 32, 32), blocks=8, conditioner="jet",
              conditioner_kwargs={"embed_dim": 256, "depth": 2}, base="ggmix")
```

For token-autoregressive flows with a causal transformer see
{doc}`transformer_autoregressive`.
