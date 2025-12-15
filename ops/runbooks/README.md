# Runbooks

- **Startup**: `docker-compose -f ops/docker-compose.yml up -d` to launch the Lean base container.
- **Ingestion**: `python scripts/ingest_context.py` to index PDFs and chats.
- **Knowledge build**: `python scripts/build_knowledge.py` after ingestion.
- **Backtest**: `python scripts/run_backtests.py --baseline` writes artifacts under `research/experiments/`.
- **Robustness**: `python scripts/run_robustness.py --baseline` for smoke perturbations.
- **Stress**: `python scripts/run_stress.py --baseline` for simple shocks.
- **PR Factory**: `bash scripts/run_cycle.sh --mode=smoke` exercises the workflow without external credentials.
