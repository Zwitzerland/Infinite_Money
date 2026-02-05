# Infinite Money

Event-driven trading research platform with LEAN orchestration, backtest tools,
and an AI signal pipeline. The repo is organized around reproducible workflows
and strict promotion gates (no direct live trading).

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -e .[dev]
make doctor
make test
```

Prefer a guided setup? See `docs/LOCAL_SETUP.md`.

## Core workflows

Backtest (synthetic G2MAX-X example):

```bash
python -m hedge_fund.backtest.runner
```

LEAN backtest (local):

```bash
imctl backtest --project lean_projects/DividendCoveredCall --params configs/lean/covered_call_params.yaml
```

Optimization (Optuna + LEAN):

```bash
imctl optimize --project lean_projects/DividendCoveredCall --study local-opt --n-trials 20
```

AI pipeline:

```bash
python -m hedge_fund.ai.cli --config hedge_fund/conf/ai_stack.yaml
```

Signals → LEAN export:

```bash
python -m hedge_fund.ai.integration.lean_export --config hedge_fund/conf/ai_stack.yaml
```

IBKR connectivity smoke test:

```bash
python -m hedge_fund.exec.ibkr_smoke_test --host 127.0.0.1 --port 7497 --client-id 1
```

QuantConnect CLI:

```bash
python -m control_plane.quantconnect_cli compile --project-id 123456 --name "smoke-compile"
```

## Configuration

- `hedge_fund/conf/` holds platform defaults (backtest, contracts, AI stack).
- Use `configs/` for LEAN project parameterization and optimization constraints.

## Repo layout

- `hedge_fund/`: core research, backtest, execution, AI, utilities.
- `control_plane/`: QuantConnect orchestration and connectors.
- `optimizer/`: Optuna + LEAN optimization tooling.
- `tools/`: repo CLIs (`imctl`).
- `lean_projects/`: LEAN templates.
- `gates/`: promotion gate schema + rules.
- `docs/`: documentation and doctrine.

## Documentation

- `docs/INDEX.md` (start here)
- `docs/LOCAL_SETUP.md` (setup walkthrough)
- `docs/STRUCTURE.md` (repo organization)
- `docs/runbook.md` (operational commands)

## Policy

Live trading is not wired here. All live deployment must flow through promotion
gates and control-plane automation.
