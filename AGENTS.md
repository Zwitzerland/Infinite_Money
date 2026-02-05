# AGENTS Instructions

## Setup

```bash
pip install -e .[dev]
```

## Checks

```bash
ruff check .
mypy hedge_fund tests
pytest
```

## Backtests

```bash
python -m hedge_fund.backtest.runner
```

## Rules

- Do not deploy live trading directly. All live deployment must flow through
  promotion gates and QuantConnect automation.
- Use Model Context Protocol integrations for external systems.
- Keep changes typed, deterministic, and testable.
