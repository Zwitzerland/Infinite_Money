# Infinite_Money

A comprehensive algorithmic trading system combining quantum computing, machine learning, and traditional quantitative finance.

## Engine

The **Infinite Money Engine** is a QuantConnect LEAN-based algorithmic trading system located in `/engine/`. It provides:

- **Statistical Arbitrage**: Mean reversion and pairs trading strategies
- **Multi-Factor Alpha Models**: Momentum, value, and quality factors
- **Regime Detection**: Market state identification and adaptation
- **Kelly Criterion**: Optimal position sizing
- **Risk Management**: Comprehensive risk controls and stress testing
- **Alternative Data**: Insider trading and sentiment analysis
- **AI Integration**: OpenAI sentiment analysis and RL allocation

### Quick Start

```bash
# Setup the engine
make bootstrap

# Run a backtest
make backtest

# Start paper trading
make paper
```

### Engine Documentation

- [Engine README](engine/README.md) - Detailed setup and usage
- [Engine Configuration](engine/lean.json) - LEAN configuration
- [Engine Tests](engine/tests/) - Test suite

### Tools

- [Fetch OpenInsider Data](tools/fetch_openinsider.py) - Download Form 4 insider trading data