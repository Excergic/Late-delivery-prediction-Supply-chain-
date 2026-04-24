#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BIN="${PROJECT_ROOT}/.python_user/bin"
if [[ -d "${LOCAL_BIN}" ]]; then
  export PATH="${LOCAL_BIN}:${PATH}"
  export PYTHONUSERBASE="${PROJECT_ROOT}/.python_user"
  export PYTHONPATH="${PROJECT_ROOT}/.python_user/lib/python3.11/site-packages:${PYTHONPATH:-}"
fi

echo "[quality-gate] Running pytest..."
pytest --strict-markers -x

echo "[quality-gate] Running ruff..."
python -m ruff check .

echo "[quality-gate] Running mypy..."
mypy .

echo "[quality-gate] All gates passed."
