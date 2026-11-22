"""Shared helpers for the benchmark scripts: run metadata, parameter counts, result files.

Every benchmark record carries ``meta = run_metadata()`` so that published
tables can state exactly which library version, commit and hardware produced
them. No host or user names are recorded.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def device_name(device: str = "cpu") -> str:
    """Human-readable name of the compute device (GPU name or CPU model)."""
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    return _cpu_name()


def run_metadata(device: str = "cpu") -> dict:
    """Library version, git commit, software versions and hardware of the current run."""
    import torch

    import cjd_flows

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "library_version": cjd_flows.__version__,
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": f"rocm {torch.version.hip}" if torch.version.hip else (f"cuda {torch.version.cuda}" if torch.version.cuda else "cpu"),
        "device": device_name(device),
        "os": f"{platform.system()} {platform.release()}",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def count_parameters(module) -> int:
    """Number of distinct trainable parameters (shared tensors counted once)."""
    seen = set()
    n = 0
    for p in module.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            n += p.numel()
    return n


def read_records(path) -> list:
    """Complete JSON records of a results file.

    Only newline-terminated lines count as complete: an unterminated last line
    (interrupted append) is ignored, so its run is redone. A malformed terminated
    line raises.
    """
    lines = Path(path).read_text().split("\n")[:-1]  # the last segment is "" or an unterminated tail
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError(f"{path}:{i + 1}: malformed record") from None
    return records


def drop_incomplete_tail(path) -> None:
    """Truncate an unterminated last line, so the next append starts a fresh record."""
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            path.write_bytes(data[: data.rfind(b"\n") + 1])
