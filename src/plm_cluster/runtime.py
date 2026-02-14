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


def require_executables(tools: list[str], config_paths: dict[str, str] | None = None) -> dict[str, str]:
    resolved: dict[str, str] = {}
    config_paths = config_paths or {}
    for tool in tools:
        conf_key = f"{tool}_path"
        configured = config_paths.get(conf_key, "") if conf_key in config_paths else config_paths.get(tool, "")
        if configured:
            p = Path(configured)
            if not p.exists():
                raise RuntimeError(f"Configured executable for {tool} not found: {p}")
            resolved[tool] = str(p)
        else:
            p = shutil.which(tool)
            if not p:
                raise RuntimeError(
                    f"Required executable '{tool}' not found in PATH. Install it or set tools.{tool}_path in config."
                )
            resolved[tool] = p
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


def executable_version(tool_path: str) -> str:
    for arg in ("--version", "-h"):
        try:
            out = subprocess.run([tool_path, arg], capture_output=True, text=True)
            txt = (out.stdout or out.stderr).strip().splitlines()
            if txt:
                return txt[0]
        except Exception:
            continue
    return "unknown"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def collect_lib_versions() -> dict[str, str]:
    libs = {}
    for pkg in ["numpy", "pandas", "torch", "fair-esm", "igraph", "leidenalg", "scikit-learn"]:
        try:
            libs[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            libs[pkg] = "not_installed"
    return libs


def write_manifest(
    manifest_path: str | Path,
    params: dict[str, Any],
    tool_paths: dict[str, str],
    inputs: list[str],
) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "git_commit": git_commit(),
        "params": params,
        "tool_paths": tool_paths,
        "tool_versions": {k: executable_version(v) for k, v in tool_paths.items()},
        "python_lib_versions": collect_lib_versions(),
        "inputs": [{"path": p, "sha256": file_sha256(p) if p and Path(p).exists() else None} for p in inputs],
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(payload, indent=2))
