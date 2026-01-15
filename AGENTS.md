# AGENTS Instructions

## Setup

```bash
pip install -e .[dev]
```

## Checks

```bash
ruff check .
mypy hedge_fund tests
PYTHONPATH=. pytest -q
```

## Backtests

```bash
PYTHONPATH=. python -m hedge_fund.backtest.runner --config-path conf --config-name backtest
```

## Rules

- Do not deploy live trading directly. All live deployment must flow through
  promotion gates and QuantConnect automation.
- Use Model Context Protocol integrations for external systems.
- Keep changes typed, deterministic, and testable.
