SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup train backtest select quantum schedule all

setup:
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate && pip install -r requirements.txt

train:
	source $(VENV)/bin/activate && $(PY) -m alpha_hive.rl_forge --config configs/config.yaml --timesteps 50000

backtest:
	source $(VENV)/bin/activate && $(PY) -m alpha_hive.simulator --config configs/config.yaml --model_path artifacts/ppo_last.zip

select:
	source $(VENV)/bin/activate && $(PY) -m alpha_hive.selector --stats_path artifacts/backtest_stats.json

quantum:
	source $(VENV)/bin/activate && $(PY) -m alpha_hive.quantum_opt --csv data/prices.csv --out artifacts/quantum_weights.json

schedule:
	source $(VENV)/bin/activate && $(PY) -m alpha_hive.schedule --config configs/config.yaml

all: setup train backtest select quantum
