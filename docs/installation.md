# Installation

CJD-Flows needs Python 3.11 or newer. The core only depends on `torch` and
`numpy`; everything else is an optional extra.

```bash
pip install cjd-flows                 # core: torch + numpy only
pip install "cjd-flows[eval,onnx]"    # + scipy/sklearn/matplotlib, ONNX export
pip install "cjd-flows[legacy]"       # + the pyro-based 0.9 API
```

| Extra | Contents | Needed for |
|---|---|---|
| `eval` | scipy, scikit-learn, matplotlib | {mod}`cjd_flows.evaluation` |
| `onnx` | onnx, onnxscript, onnxruntime | {func}`cjd_flows.export.to_onnx`, {func}`cjd_flows.export.verify_onnx` |
| `vision` | torchvision | image datasets via torchvision |
| `tabular` | pandas, h5py | first preprocessing of the MAF tabular benchmarks (GAS, HEPMASS, BSDS300) |
| `legacy` | pyro-ppl | the 0.9 API, see {doc}`legacy` |
| `docs` | Sphinx and extensions | building this documentation |

## Development

The repository uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # dev environment incl. pytest/ruff
uv run pytest -m "not slow"
```

Build the documentation locally with:

```bash
pip install -e ".[docs,eval,onnx]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```
