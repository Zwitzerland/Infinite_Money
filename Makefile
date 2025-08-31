# Root Makefile - Delegates to engine/
.PHONY: help bootstrap data backtest optimize paper live logs test clean

help:
	@echo "Infinite Money - Root Makefile"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  bootstrap  - Setup engine environment"
	@echo "  data       - Download market data"
	@echo "  backtest   - Run backtest"
	@echo "  optimize   - Run optimization"
	@echo "  paper      - Start paper trading"
	@echo "  live       - Start live trading"
	@echo "  logs       - View logs"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean build artifacts"
	@echo ""
	@echo "All targets delegate to engine/Makefile"

bootstrap:
	@echo "Setting up Infinite Money Engine..."
	cd engine && make bootstrap

data:
	@echo "Downloading market data..."
	cd engine && make data

backtest:
	@echo "Running backtest..."
	cd engine && make backtest

optimize:
	@echo "Running optimization..."
	cd engine && make optimize

paper:
	@echo "Starting paper trading..."
	cd engine && make paper

live:
	@echo "Starting live trading..."
	cd engine && make live

logs:
	@echo "Viewing logs..."
	cd engine && make logs

test:
	@echo "Running tests..."
	cd engine && make test

clean:
	@echo "Cleaning build artifacts..."
	cd engine && make clean

