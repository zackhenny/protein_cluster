#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <conda_env_name> [repo_path]"
  exit 1
fi

ENV_NAME="$1"
REPO_PATH="${2:-$(pwd)}"

# Ensure pip never defaults to user site during install
export PYTHONNOUSERSITE=1
export PIP_USER=0

conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation --no-user "${REPO_PATH}"

# Quick verification: pip/python path must resolve inside target env
conda run -n "${ENV_NAME}" python - <<'PY'
import site, sys
print('python:', sys.executable)
print('prefix:', sys.prefix)
print('usersite:', site.getusersitepackages())
PY
