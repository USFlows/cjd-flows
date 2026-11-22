"""``import cjd_flows`` must not pull in pyro or the optional dependencies."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

FORBIDDEN = ("pyro", "scipy", "sklearn", "torchvision", "onnx", "onnxruntime", "onnxscript", "matplotlib")

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.mark.parametrize(
    "stmt",
    [
        "import cjd_flows",
        "import cjd_flows, cjd_flows.data, cjd_flows.scores, cjd_flows.export, cjd_flows.testing, cjd_flows.train",
    ],
)
def test_import_does_not_load_optional_dependencies(stmt):
    code = (
        f"{stmt}\n"
        "import json, sys\n"
        f"print(json.dumps(sorted({{m.split('.')[0] for m in sys.modules}} & set({list(FORBIDDEN)!r}))))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True, timeout=300
    )
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded == [], f"`{stmt}` imported optional dependencies: {loaded}"
