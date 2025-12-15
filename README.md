# Infinite Money R&D Stack

PR-based quant research and backtesting OS with reproducible ingestion, knowledge building, evaluation, and constraint enforcement. All workflows are deterministic, Docker-friendly, and avoid autonomous live deployment.

## Repository layout

- `context/`: PDF and chat corpus for grounding.
- `knowledge/`: Corpus map, summaries, evidence objects, and citations policy.
- `lean/`: QuantConnect Lean project placeholder for deterministic backtests.
- `eval/`: Metrics, leakage/robustness/stress placeholders, and evaluation gates.
- `research/`: Hypotheses, experiments, and reports.
- `constraints/`: Declarative risk and exposure policy with compiler stub.
- `ops/`: Docker Compose, environment template, and runbooks.
- `scripts/`: CLI utilities for ingestion, knowledge building, backtests, and PR drafting.
- `hedge_fund/`: Event-driven trading skeleton modules.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Smoke workflow:

```bash
bash scripts/run_cycle.sh --mode=smoke
```

Individual steps:

```bash
python scripts/ingest_context.py
python scripts/build_knowledge.py
python scripts/run_backtests.py --baseline
python scripts/run_robustness.py --baseline
python scripts/run_stress.py --baseline
python scripts/propose_patch.py
python scripts/open_pr.py
```

## Determinism and safety

- No autonomous live trading deployments; all changes flow through PRs and manual gating.
- Record dataset hashes, seeds, and manifests for every experiment under `research/experiments/`.
- Keep secrets out of the repo; copy `ops/.env.template` to `.env` and configure locally or via CI secrets.
- Docker Compose (`ops/docker-compose.yml`) pins runtime image for reproducible Lean interactions.

## Testing

```bash
ruff check .
mypy hedge_fund scripts eval constraints
pytest -q
```
