# AlphaQuanta - Production Trading Agent Framework

A production-grade agent framework for algorithmic trading using QuantConnect Lean engine and Interactive Brokers, built on OpenAI's "Practical Guide to Building Agents".

## Architecture Overview

**LeanCoreAgent**: Single omni-agent that ingests Lean data, generates trading signals using ensemble mean-reversion + momentum strategies, and executes trades via ib_insync (live) or QuantConnect backtester (paper).

## Core Components

- **Agent Core**: Model + Tools + Instructions pattern
- **Data Tools**: QuantConnect API integration for historical/paper/live feeds
- **Action Tools**: ib_insync OrderRouter, Lean BacktestRunner, PositionSizer
- **Risk Guardrails**: Notional limits, stop hierarchy, drawdown kill-switch
- **Telemetry**: Prometheus ACU meter + PnL dashboards

## Quick Start

```bash
# Install dependencies
poetry install

# Paper trading backtest
python runner.py --paper --symbol SPY --start 2018-01-01 --end 2024-12-31

# Live paper session
python runner.py --live --paper --acu-cap 20

# Run tests
pytest tests/ -v
```

## Acceptance Criteria

- ✅ Paper-trade backtest (SPY 2018-2024) reproducible, Sharpe > 1.6
- ✅ Live paper session pushes orders to IB within 250ms median latency
- ✅ Full run ≤20 ACUs; P95 latency ≤300ms default path
- ✅ All guardrails fire on injected "BUY GME 10000 @ MKT" jailbreak

## Project Structure

```
alphaquanta/
├── alphaquanta/           # Core package
│   ├── agents/           # LeanCoreAgent implementation
│   ├── tools/            # Data/Action/Orchestration tools
│   ├── guardrails/       # Risk management system
│   ├── strategies/       # Trading strategies
│   └── telemetry/        # ACU and PnL monitoring
├── tests/                # Pytest suite with zero-ACU mocks
├── docs/                 # Architecture documentation
├── docker-compose.yaml   # Lean engine + IB Gateway
├── runner.py             # CLI interface
└── pyproject.toml        # Poetry configuration
```

## License

MIT License
