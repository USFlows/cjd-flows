"""Skip collection of legacy test modules whose imports need pyro.

pyro is an optional dependency (``cjd-flows[legacy]``); the core test job runs
without it.
"""

import importlib.util

collect_ignore = []
if importlib.util.find_spec("pyro") is None:
    collect_ignore += ["flows_test.py", "networks_test.py", "transforms_test.py"]
