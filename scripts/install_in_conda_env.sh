#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <conda_env_name> [repo_path] [extras]"
  echo "Example: $0 plm_cluster . test,embed"
  exit 1
fi

ENV_NAME="$1"
REPO_PATH="${2:-$(pwd)}"
EXTRAS="${3:-test,embed}"

export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

PKG_SPEC="${REPO_PATH}"
if [[ -n "${EXTRAS}" ]]; then
  PKG_SPEC="${REPO_PATH}[${EXTRAS}]"
fi

# Ensure pip/setuptools/wheel are inside target env (not system/home)
conda run -n "${ENV_NAME}" python -m pip install --no-user --upgrade pip setuptools wheel

# Install test runner support packages explicitly for pytest invocations in HPC envs
conda run -n "${ENV_NAME}" python -m pip install --no-user iniconfig pygments

# Install this project into the target env
conda run -n "${ENV_NAME}" python -m pip install --no-user --no-build-isolation "${PKG_SPEC}"

# Verify install locations and entrypoints are in env
conda run -n "${ENV_NAME}" python - <<'PY'
import shutil, site, sys
print('python:', sys.executable)
print('prefix:', sys.prefix)
print('usersite:', site.getusersitepackages())
print('esm-extract:', shutil.which('esm-extract'))
print('plm_cluster:', shutil.which('plm_cluster'))
PY
