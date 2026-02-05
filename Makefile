.PHONY: doctor format lint type test smoke optimize report knowledge

PY := python

doctor:
	$(PY) -m tools.imctl doctor

format:
	$(PY) -m ruff format .

lint:
	$(PY) -m ruff check .

type:
	$(PY) -m mypy hedge_fund optimizer tools

test:
	$(PY) -m pytest

smoke:
	$(PY) -m tools.imctl backtest --project lean_projects/DividendCoveredCall --params configs/lean/covered_call_params.yaml

optimize:
	$(PY) -m tools.imctl optimize --project lean_projects/DividendCoveredCall --study local-opt --n-trials 20 --sampler tpe --constraints configs/lean/constraints.yaml

report:
	$(PY) -m tools.imctl report --run-id latest

knowledge:
	$(PY) -m tools.imctl knowledge sync
