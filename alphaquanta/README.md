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

### One-Line Install
```bash
curl -sSL https://raw.githubusercontent.com/<ORG>/alphaquanta_q/main/install.sh | bash
```

### Manual Setup
```bash
# Clone repository
git clone https://github.com/<ORG>/alphaquanta_q.git
cd alphaquanta_q

# Install dependencies
poetry install

# Set up environment (edit with your tokens)
cp .env.example .env
nano .env

# Start services
docker-compose up -d

# Run quantum-hybrid backtest
python runner.py --mode backtest --quantum on --symbol SPY --start 2018-01-01 --end 2024-12-31

# Paper trading with quantum alpha discovery
python runner.py --mode paper --quantum on --acu-cap 20

# Classical-only mode (no QPU usage)
python runner.py --mode paper --quantum off --symbol SPY

# Run tests
pytest tests/ -v
```

### Environment Setup
Set these environment variables in `.env`:
```bash
# Quantum Computing (required for quantum features)
IBM_QUANTUM_TOKEN=your_ibm_token_here
DWAVE_API_TOKEN=your_dwave_token_here

# Trading APIs
QC_API_TOKEN=your_quantconnect_token
IB_USERNAME=your_ib_username
IB_PASSWORD=your_ib_password

# Optional: Monitoring credentials
GRAFANA_PASSWORD=your_grafana_password
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
