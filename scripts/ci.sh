#!/usr/bin/env bash
# Local equivalent of .github/workflows/ci.yml (decisions.md D-35).
#   bash scripts/ci.sh          # fast job + model job
#   bash scripts/ci.sh fast     # lint, types, offline tests only
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check .
ruff format --check .
mypy contextlens project.py train.py evaluate.py
pytest -q -m "not model"
if [[ "${1:-all}" != "fast" ]]; then
  pytest -q -m model
fi
echo "ci: all checks passed"
