#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make all
echo ">>> Bootstrap complete. Artifacts in ./artifacts"
