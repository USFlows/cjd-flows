"""Skip the experiment-library tests unless the ``experiments`` extra is installed."""

import importlib.util

collect_ignore = []
if any(importlib.util.find_spec(m) is None for m in ("pyro", "ray", "yaml")):
    collect_ignore += ["hyperopt_test.py"]
