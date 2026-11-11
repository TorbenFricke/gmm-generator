#!/usr/bin/env bash
# Run all local quality checks: lint, formatting, type checks and tests.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${VENV_BIN:-.venv/bin}
"$BIN/ruff" check src tests run_simulation.py
"$BIN/ruff" format --check src tests run_simulation.py
"$BIN/mypy"
"$BIN/pytest" -q
