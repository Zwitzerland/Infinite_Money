# Infinite Money Engine

A QuantConnect LEAN Python algorithmic trading system with full local and cloud parity.

## Features

- Statistical arbitrage strategies
- Multi-factor alpha models
- Regime detection and adaptation
- Kelly criterion position sizing
- Risk management and stress testing
- Alternative data integration
- AI-powered sentiment analysis
- Reinforcement learning allocation

## Quick Start

```bash
# Bootstrap the environment
make bootstrap

# Run backtest
make backtest

# Paper trading (requires IBKR credentials)
make paper
```

## Architecture

- `src/InfiniteMoneyEngine.py` - Main strategy class
- `src/alpha_models/` - Alpha signal generation
- `src/portfolio/` - Portfolio optimization and sizing
- `src/risk/` - Risk management and limits
- `src/exec/` - Execution and cost modeling
- `src/data/` - Data sources and utilities
- `src/ai/` - AI/ML components

## Requirements

- Python 3.10
- QuantConnect LEAN CLI
- Docker (for local development)
- Interactive Brokers (for live trading)

## License

MIT
