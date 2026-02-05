#!/usr/bin/env bash
set -euo pipefail

PROJECT=${1:-"lean_projects/DividendCoveredCall"}
PARAMS=${2:-"configs/lean/covered_call_params.yaml"}
TRIALS=${TRIALS:-20}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: run_local_end_to_end.sh [project] [params]"
  echo "Env: TRIALS=<int>"
  exit 0
fi

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

PY=".venv/bin/python"
$PY -m pip install --upgrade pip
$PY -m pip install -e .[dev]

$PY -m tools.imctl backtest --project "$PROJECT" --params "$PARAMS"
$PY -m tools.imctl optimize --project "$PROJECT" --study "local-opt" --n-trials "$TRIALS" --sampler tpe --constraints configs/lean/constraints.yaml
$PY -m tools.imctl report --run-id latest
