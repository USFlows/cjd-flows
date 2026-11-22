# Anomaly scores

With a radial base, a uniformly scaling flow ranks inputs by density through
the latent norm. {func}`~cjd_flows.scores.anomaly_scores` computes several
scores (higher = more anomalous) batch-wise:

| Score | Definition | Needs |
|---|---|---|
| `nll` | $-\log p(x)$ | any base |
| `latent_norm` | $\lVert f(x) - \mathrm{loc} \rVert_p$ | any base |
| `typicality` | $-\log p_R(\lVert z \rVert)$ | radial base (or a fitted {class}`~cjd_flows.scores.TypicalityScorer`) |
| `hdr` | mass of the latent region more likely than $z$; $F_R(\lVert z \rVert)$ for monotone profiles | radial base with a CDF |

```python
import torch
from cjd_flows import USFlow, fit
from cjd_flows.data.toy import make_toy
from cjd_flows.scores import anomaly_scores, detection_metrics

x_train = make_toy("two_moons", 20_000, seed=0)
x_in = make_toy("two_moons", 2_000, seed=1)       # normal test data
x_out = 4 * torch.rand(2_000, 2) - 2               # anomalies: uniform on a box

flow = USFlow((2,), blocks=8, base="ggmix")
fit(flow, x_train, epochs=30, lr=1e-3)

s_in = anomaly_scores(flow, x_in)
s_out = anomaly_scores(flow, x_out)
for kind in s_in:
    print(kind, detection_metrics(s_in[kind], s_out[kind]))   # AUROC, AUPR, FPR@95
```

{func}`~cjd_flows.scores.detection_metrics` treats the anomalies as the
positive class. {func}`~cjd_flows.scores.one_class_protocol` runs the "train on
one class, test against the rest" protocol for a list of classes.

Uniformly scaling flows for one-class classification are studied in Abu Zaid
et al. (2025), [arXiv:2510.09452](https://arxiv.org/abs/2510.09452).
