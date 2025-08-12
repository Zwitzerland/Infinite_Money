.PHONY: setup lint test smoke train experiment

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check .

test:
	PYTHONPATH=alphaquanta $(PYTHON) -m pytest -q

smoke:
	PYTHONPATH=alphaquanta:. $(PYTHON) scripts/smoke_backtest.py

train:
	PYTHONPATH=alphaquanta $(PYTHON) alphaquanta/runner.py --mode backtest --quantum off --symbol SPY --start 2021-01-01 --end 2021-03-01 -v

experiment:
	@echo "No experiments defined; add your experimental command here."

