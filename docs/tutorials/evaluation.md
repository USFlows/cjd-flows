# Evaluation suite

{class}`~cjd_flows.evaluation.RadialFlowEvaluator` checks how well the latents
of a trained flow match the radial structure of its base. It needs the `eval`
extra.

- norm-distribution diagnostics (KS, Wasserstein, PP/QQ/KDE),
- radiality diagnostics (sign symmetry and simplex-uniformity tests),
- calibration-oriented latent-space analysis,
- fine-grained mismatch diagnostics that help distinguish over- vs.
  under-approximation of density level sets.

```python
from cjd_flows import USFlow, fit
from cjd_flows.data.toy import make_toy
from cjd_flows.evaluation import RadialFlowEvaluator

x = make_toy("two_moons", 20_000, seed=0)
x_test = make_toy("two_moons", 2_000, seed=1)
flow = USFlow((2,), blocks=8, base="ggmix-monotone", p=1.0)
fit(flow, x, epochs=30, lr=1e-3)

ev = RadialFlowEvaluator(flow, x_test)   # p, norm law and centre are read from the base
print(ev.ks_norm_statistic())            # latent norms vs. the base norm distribution
print(ev.wasserstein_norm_distance())
ev.pp_plot_norms()                       # returns a matplotlib Axes
print(ev.test_l1_radial_symmetry())      # L1 bases: sign symmetry + simplex uniformity
```

The evaluator works with v1 flows with a radial or Gaussian base (the Gaussian
is treated as L2-radial with a chi norm law) and with the 0.9 flows. For other
bases, pass `p`, `norm_distribution` and `loc` explicitly.

The binned uniformity test and the simplex tests take their p-values from
simulations (cone measure of the Lp sphere, uniform simplex samples). The
binned test is only meaningful for low dimensions. The Bhattacharyya variant
of the simplex test has little power against anisotropic latents; the energy
test (default) is preferable.

```{image} ../images/eval_suite.svg
:alt: Overview of the evaluation suite
```
