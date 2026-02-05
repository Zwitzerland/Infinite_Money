#!/usr/bin/env bash
set -euo pipefail

PROJECT=${1:?"Usage: run_cloud_optuna.sh <project>"}
TARGET=${TARGET:-"Sharpe Ratio"}
DIRECTION=${DIRECTION:-"max"}
NODE=${NODE:-"O2-8"}
PARALLEL_NODES=${PARALLEL_NODES:-2}

lean cloud optimize "$PROJECT" \
  --target "$TARGET" \
  --target-direction "$DIRECTION" \
  --parameter lookback 10 60 5 \
  --parameter delta 0.1 0.5 0.05 \
  --parameter dte_min 7 21 7 \
  --parameter dte_max 21 42 7 \
  --constraint "Drawdown < 0.25" \
  --constraint "Compounding Annual Return > 0" \
  --node "$NODE" \
  --parallel-nodes "$PARALLEL_NODES" \
  --push
