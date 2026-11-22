"""Download / cache helpers shared by the dataset loaders (internal)."""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]

ENV_VAR = "CJD_FLOWS_DATA"
USER_AGENT = "cjd-flows-data-loader"


def default_root() -> Path:
    """Default dataset cache directory.

    ``$CJD_FLOWS_DATA`` if set, otherwise ``~/.cache/cjd_flows``.
    """
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "cjd_flows"


def resolve_root(root: Optional[PathLike]) -> Path:
    return Path(root).expanduser() if root is not None else default_root()


def md5sum(path: PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_url(
    url: str,
    dest: PathLike,
    md5: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 3,
    progress: bool = True,
) -> Path:
    """Downloads ``url`` to ``dest`` (atomically, resumable, with progress).

    Args:
        url: source URL (http/https only).
        dest: target file path. Nothing is done if it already exists (and
            matches ``md5`` if given).
        md5: optional expected MD5 checksum.
        timeout: socket timeout in seconds for each read.
        retries: number of attempts (partial downloads are resumed).
        progress: print progress to stderr.

    Returns:
        The destination path.
    """
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"refusing to download non-http URL {url!r}")
    dest = Path(dest)
    if dest.exists():
        if md5 is None or md5sum(dest) == md5:
            return dest
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            done = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            if done:
                req.add_header("Range", f"bytes={done}-")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if done and r.status != 206:  # server ignored the range request
                    done = 0
                total = r.headers.get("Content-Length")
                total = int(total) + done if total is not None else None
                mode = "ab" if done else "wb"
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        now = time.time()
                        if progress and now - last > 2.0:
                            last = now
                            rate = done / max(now - t0, 1e-6) / 2**20
                            tot = f"/{total / 2**20:.0f}" if total else ""
                            print(
                                f"\r[cjd_flows] {dest.name}: {done / 2**20:.0f}{tot} MiB ({rate:.1f} MiB/s)",
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )
            if progress:
                print(file=sys.stderr)
            if total is not None and done < total:
                raise OSError(f"incomplete download ({done} < {total} bytes)")
            if md5 is not None and md5sum(part) != md5:
                part.unlink()
                raise OSError(f"MD5 mismatch for {url}")
            part.replace(dest)
            return dest
        except OSError as e:  # urllib errors derive from OSError
            last_err = e
            if progress:
                print(f"\n[cjd_flows] download attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"could not download {url}: {last_err}")
