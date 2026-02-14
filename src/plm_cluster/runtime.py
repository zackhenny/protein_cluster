from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.metadata as importlib_metadata


REQUIRED_TOOLS = ["mmseqs", "hhmake"]


def setup_logging(log_dir: str | Path, name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def command_exists(tool: str) -> bool:
    return shutil.which(tool) is not None


def require_executables(tools: list[str], config_paths: dict[str, str] | None = None) -> dict[str, str]:
    resolved: dict[str, str] = {}
    config_paths = config_paths or {}
    for t in tools:
        if t in config_paths and config_paths[t]:
            p = Path(config_paths[t])
            if not p.exists():
                raise RuntimeError(f"Configured executable for {t} not found: {p}")
            resolved[t] = str(p)
        else:
            w = shutil.which(t)
            if not w:
                raise RuntimeError(
                    f"Required executable '{t}' not found in PATH. Install it or provide config.tools.{t}_path"
                )
            resolved[t] = w
    return resolved


def run_cmd(cmd: list[str], logger: logging.Logger, cwd: str | None = None) -> str:
    logger.info("CMD: %s", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if out.stdout:
        logger.info("STDOUT: %s", out.stdout.strip())
    if out.stderr:
        logger.info("STDERR: %s", out.stderr.strip())
    if out.returncode != 0:
        raise RuntimeError(f"Command failed ({out.returncode}): {' '.join(cmd)}")
    return out.stdout


def executable_version(tool_path: str, version_arg: str = "--version") -> str:
    try:
        out = subprocess.run([tool_path, version_arg], capture_output=True, text=True)
        txt = (out.stdout or out.stderr).strip().splitlines()
        return txt[0] if txt else "unknown"
    except Exception:
        return "unknown"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    manifest_path: str | Path,
    params: dict[str, Any],
    tool_paths: dict[str, str],
    inputs: list[str],
) -> None:
    libs = {}
    for pkg in ["numpy", "pandas", "torch", "fair-esm", "leidenalg", "igraph"]:
        try:
            libs[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            libs[pkg] = "not_installed"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "params": params,
        "tool_paths": tool_paths,
        "tool_versions": {k: executable_version(v) for k, v in tool_paths.items()},
        "python_lib_versions": libs,
        "inputs": [{"path": p, "sha256": file_sha256(p) if Path(p).exists() else None} for p in inputs],
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(payload, indent=2))
