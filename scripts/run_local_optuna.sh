#!/usr/bin/env bash
set -euo pipefail

PROJECT=${1:?"Usage: run_local_optuna.sh <project>"}
TRIALS=${TRIALS:-20}
SEARCH_SPACE=${SEARCH_SPACE:-optimizer/search_space.yaml}

python -m optimizer.study_optuna \
  --project "$PROJECT" \
  --trials "$TRIALS" \
  --search-space "$SEARCH_SPACE" \
  --download-data
