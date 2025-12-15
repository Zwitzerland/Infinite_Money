#!/usr/bin/env bash
set -eo pipefail

MODE="smoke"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python scripts/ingest_context.py
python scripts/build_knowledge.py
python scripts/run_backtests.py --baseline
python scripts/run_robustness.py --baseline
python scripts/run_stress.py --baseline
python scripts/propose_patch.py
python scripts/open_pr.py

echo "Run cycle complete (mode=${MODE})."
