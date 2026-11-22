# Legacy API

The pyro-based API of version 0.9 (`cjd_flows.flows`, `cjd_flows.transforms`,
`cjd_flows.distributions`, together with `cjd_flows.networks`,
`cjd_flows.linalg`, `cjd_flows.sophia` and `cjd_flows.utils`) is still
importable with the `legacy` extra:

```bash
pip install "cjd-flows[legacy]"
```

Its known bugs are fixed (see the changelog), but it is not documented here and
receives no new features. New code should use the v1 API ({doc}`api/index`).

The YAML/Ray experiment framework of 0.9 (`cjd_flows.explib`, `scripts/`,
`experiments/`) was removed in 1.0. Experiments for papers live in their own
repositories; the last state of the framework is tagged
[`archive/explib-final`](https://github.com/USFlows/cjd-flows/tree/archive/explib-final).
The latent-space diagnostics of `explib.eval` moved to
{mod}`cjd_flows.evaluation`.
