# USFlows: Flow Based Density Estimators for Neuro-Symbolic Verification
USFlows provides a stable and convenient library of flow based general purpose density models with flexibile base distributions, 
which are specifically tailored towards the use in neuro-symbolic verification procedures. The major goal is to
provide models that can represent reference distributions which are suitable for satisfiability based approaches, 
abstract interpretation, and hypothesis testing simultaneously.
The implemented layer are carefully designed to guarntee the following properties:

- Efficient computation of exact densities as well as efficient sampling.
- A piece-wise affine log-density function for all models with (leaky-)ReLU nonlinearity and Laplacian base distribution.
- UDL preserving layers map the upper density level sets of the data distribution to the upper density level sets
of the base Distribution.
- Direct onnx export of log_prob and sampling methods.

# Installation
1) Clone Project:
```bash
git clone git@github.com:aai-institute/VeriFlow.git
```
2) Install poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
3) Finally, within the veriflow project directory:
```bash
poetry install
```

# Experiments
Veriflow comes with a lightweigt experimentation library that allow effortless configuaration, e.g. of hyperparameter optimzation experiments via yaml config files.
Additionally, we define several benchmarking experiments.

## Run an experiment
Within the projects script folder you'll find a a script called **run_experiment.py**. You can use it to conduct an experiment from the a config file.
```bash
poetry run python run_experiment.py --config <config file> --report_dir <log dir>
```

## Acknowledgment

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logos/baiaa-logo.svg">
  <img src="docs/images/logos/baiaa-logo-black.svg">
</picture>

The Bavarian AI Act Accelerator is a two-year project funded by the Bavarian
State Ministry of Digital Affairs to support SMEs, start-ups, and the public
sector in Bavaria in complying with the EU AI Act. Under the leadership of the
appliedAI Institute for Europe and in collaboration with Ludwig Maximilian
University, the Technical University of Munich, and the Technical University of
Nuremberg, training, resources, and events are being offered. The project
objectives include reducing compliance costs, shortening the time to compliance,
and strengthening AI innovation. To achieve these objectives, the project is
divided into five work packages: project management, research, education, tools
and infrastructure, and community.
