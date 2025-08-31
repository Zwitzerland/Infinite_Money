.PHONY: bootstrap backtest paper clean test

# Default target
all: bootstrap

# Bootstrap the environment
bootstrap:
	@echo "Bootstrapping Infinite Money Engine..."
	lean init
	lean config set environment local
	lean config set data-provider local
	@echo "Environment bootstrapped successfully!"

# Run backtest
backtest:
	@echo "Running backtest..."
	lean backtest "InfiniteMoneyEngine.py" --output results/

# Paper trading
paper:
	@echo "Starting paper trading..."
	lean live "InfiniteMoneyEngine.py" --environment paper

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf results/
	rm -rf logs/
	rm -rf data/cache/
	@echo "Clean complete!"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Update dependencies
update:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Lint code
lint:
	@echo "Linting code..."
	flake8 src/ tests/
	mypy src/

# Security check
security:
	@echo "Running security checks..."
	bandit -r src/
	safety check

# Full CI pipeline
ci: install lint test security
	@echo "CI pipeline completed successfully!"

# Help
help:
	@echo "Available targets:"
	@echo "  bootstrap  - Initialize LEAN environment"
	@echo "  backtest   - Run backtest"
	@echo "  paper      - Start paper trading"
	@echo "  clean      - Clean build artifacts"
	@echo "  test       - Run tests"
	@echo "  install    - Install dependencies"
	@echo "  update     - Update dependencies"
	@echo "  format     - Format code"
	@echo "  lint       - Lint code"
	@echo "  security   - Security checks"
	@echo "  ci         - Full CI pipeline"
	@echo "  help       - Show this help"
