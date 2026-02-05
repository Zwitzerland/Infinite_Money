#!/usr/bin/env bash
set -euo pipefail

PROJECT=${1:?"Usage: ralph_loop.sh <project>"}
TRIALS=${TRIALS:-20}

echo "Running optimization loop for project: $PROJECT"
./scripts/run_local_optuna.sh "$PROJECT"

if grep -q "DONE: yes" STATUS.md; then
  echo "DONE criteria satisfied."
else
  echo "DONE criteria not satisfied. See STATUS.md for blockers."
  exit 1
fi
