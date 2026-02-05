# LEAN Optimization Guide

This guide describes how to run leakage-robust, reproducible optimizations
against QuantConnect LEAN using the tooling in `optimizer/`.

## Prereqs

1. Install the LEAN CLI and authenticate:

```bash
python -m pip install lean
lean login
```

2. Initialize a LEAN workspace if you do not already have one:

```bash
lean init
```

3. Ensure Docker is running (required for local backtests/optimizations).

## Parameterization (required)

All parameters to optimize must be read via `QCAlgorithm.get_parameter`.

Python example:

```python
lookback = int(self.get_parameter("lookback", 20))
```

The parameter name must match `optimizer/search_space.yaml`.

Example project in this repo:

- `lean_projects/DividendCoveredCall`

## Search space

Edit `optimizer/search_space.yaml` to define parameters, objective, and
constraints. The example uses:

- `lookback` (int, 10..60 step 5)
- objective: `Sharpe Ratio` (maximize)
- constraints: `Drawdown < 0.25`, `Compounding Annual Return > 0`

## Local backtest

Run a single backtest with explicit parameters:

```bash
lean backtest "lean_projects/DividendCoveredCall" \
  --parameter lookback 20 \
  --parameter delta 0.30 \
  --parameter dte_min 7 \
  --parameter dte_max 30 \
  --parameter max_drawdown 0.25 \
  --download-data
```

Lean stores output in `<project>/backtests/<timestamp>/` or the directory you
pass with `--output`.

## Local optimization (grid/euler)

Non-interactive local optimization using the LEAN CLI:

```bash
lean optimize "lean_projects/DividendCoveredCall" \
  --strategy "Grid Search" \
  --target "Sharpe Ratio" \
  --target-direction "max" \
  --parameter lookback 10 60 5 \
  --parameter delta 0.1 0.5 0.05 \
  --parameter dte_min 7 21 7 \
  --parameter dte_max 21 42 7 \
  --constraint "Drawdown < 0.25" \
  --constraint "Compounding Annual Return > 0" \
  --download-data
```

Results are stored under `<project>/optimizations/<timestamp>/` or in the
directory supplied with `--output`.

## Cloud optimization

Run a cloud optimization (non-interactive) with constraints and parallel nodes:

```bash
lean cloud optimize "lean_projects/DividendCoveredCall" \
  --target "Sharpe Ratio" \
  --target-direction "max" \
  --parameter lookback 10 60 5 \
  --parameter delta 0.1 0.5 0.05 \
  --parameter dte_min 7 21 7 \
  --parameter dte_max 21 42 7 \
  --constraint "Drawdown < 0.25" \
  --constraint "Compounding Annual Return > 0" \
  --node O2-8 \
  --parallel-nodes 2 \
  --push
```

Use `scripts/run_cloud_optuna.sh` for a one-command wrapper.

## AI-driven Optuna search

The Optuna runner shells out to `lean backtest` for each trial and saves a
reproducible run directory under `artifacts/`.

```bash
python -m optimizer.study_optuna --project "lean_projects/DividendCoveredCall" \
  --trials 20 --download-data
```

Outputs:

- `artifacts/opt_<timestamp>/summary.json`
- `artifacts/opt_<timestamp>/report.md`
- `artifacts/opt_<timestamp>/trial_*/` (per-trial outputs)

## Earnings volatility optimization (non-LEAN)

The earnings volatility strategy has a lightweight Optuna search that runs on
CSV inputs without LEAN:

```bash
python -m optimizer.earnings_vol_optuna --trials 120
```

Outputs:

- `artifacts/earnings_vol_opt_<timestamp>/summary.json`
- `artifacts/earnings_vol_opt_<timestamp>/trials.json`

## Validation protocol

The validation utilities enforce purged splits and embargo to avoid leakage:

- `optimizer/validation/purged_cv.py`: Purged K-fold with embargo
- `optimizer/validation/cpcv.py`: CPCV path generation + robust score summary

Use these utilities when training ML models or when evaluating multiple trials
to avoid information leakage and inflated metrics.

## Common failure modes

- Missing `lean` CLI: install with `python -m pip install lean` and run
  `lean login`.
- No Docker: local backtests/optimizations require Docker.
- Parameters not wired to `QCAlgorithm.get_parameter`: optimization will have
  no effect.
- Constraints failing: check stats names match LEAN output (e.g. `Drawdown`).
