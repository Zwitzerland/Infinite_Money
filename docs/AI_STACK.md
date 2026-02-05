# AI Stack (Bare Bones)

This is a minimal, modular scaffold for combining transformers, ML ensembles,
news ingestion, agent orchestration, and AWS Braket quantum optimization. It is
not a guarantee of superior performance. Markets are noisy and non-stationary;
expect degradation without continual validation.

## Goals

- Modular data, model, and optimization layers.
- Leakage-safe validation (purged splits + embargo + CPCV).
- Reproducible runs with immutable artifacts.

## Install (optional extras)

```bash
python -m pip install -e .[dev,ai,news,quantum]
```

Optional extras:

- `ai`: transformers + core ML stack
- `news`: news ingestion helpers
- `quantum`: AWS Braket SDK + Ocean plugin

## Configuration

Edit `hedge_fund/conf/ai_stack.yaml` to enable features. All steps default to
`enabled: false` to avoid accidental API calls.

Market data sources:

- `source: csv` (default) using `data/market_data.csv`
- `source: ibkr` (requires IB Gateway/TWS running)

Training is off by default; set `training.enabled: true` to train GBDT/linear
models on the generated features.

Required environment variables (examples):

- `NEWS_API_KEY` for NewsAPI
- `AWS_PROFILE`, `AWS_REGION`, and an S3 bucket for Braket

## Run the bare bones pipeline

```bash
python -m hedge_fund.ai.cli --config hedge_fund/conf/ai_stack.yaml
```

## Export signals for LEAN

```bash
python -m hedge_fund.ai.integration.lean_export --config hedge_fund/conf/ai_stack.yaml
```

This writes `data/custom/ai_signals.csv`, which the LEAN project
`lean_projects/AISignalTrader` consumes.

Supported export modes (configure in `hedge_fund/conf/ai_stack.yaml` under
`signal_export.method`):
- `rule`: SMA directional baseline
- `model`: walk-forward linear/GBDT prediction
- `g2max`: growth x guardrails exposure
- `sr_barrier`: algorithmic support/resistance barrier exposure (see `docs/SR_BARRIER_RULE.md`)

Artifacts are written to `artifacts/ai_runs/<run_id>/`.

## Suggested model stack (practical)

- Transformer encoder (price + text embeddings)
- GBDT (LightGBM/XGBoost) for tabular features
- Linear or elastic net baseline
- Weighted ensemble + regime-based routing

## Advanced methods to add (modular)

- Temporal Fusion Transformer or PatchTST for price sequences.
- Cross-attention fusion between price features and news embeddings.
- Probabilistic heads (quantile regression) for risk-aware forecasts.
- Conformal prediction for calibrated intervals (`hedge_fund/ai/models/calibration.py`).
- Regime-based model routing using `hedge_fund/ai/features/regime.py`.
- Risk parity or mean-variance portfolio construction (`hedge_fund/ai/portfolio/allocator.py`).
- G2MAX compounding rule (`hedge_fund/ai/portfolio/g2max.py`).
- QUBO portfolio selection on AWS Braket (`hedge_fund/ai/quantum/braket.py`).

## News ingestion flow

1. Ingest and normalize raw headlines.
2. Deduplicate via content hash.
3. Embed texts with sentence-transformers.
4. Align news timestamps to market sessions.

## Filings ingestion flow (legal, public)

1. Pull official filings (EDGAR, House/Senate) or licensed vendor feeds.
2. Normalize identifiers with OpenFIGI.
3. Track discovered_at vs filing_time for latency features.
4. Treat filings as research features only.

## Quantum portfolio optimization (AWS Braket)

Use `hedge_fund/ai/quantum/braket.py` to build a QUBO from expected returns and
covariance, then submit to Braket. Start with classical/Hybrid solvers first.

See `docs/QUANTUM_FINANCE.md` for advanced quantum workflows.

## Leakage-safe validation

Use `optimizer/validation` (purged CV + embargo + CPCV) and summarize scores in
`hedge_fund/ai/validation.py`. Prefer probabilistic/deflated Sharpe and avoid
single-pass tuning.

## Practical limits

- Accuracy is bounded by noisy, non-stationary markets.
- Combine model outputs with risk controls and regime filters.
- Always paper trade before any live deployment.

See `docs/AGENT_AUTOMATION.md` for perpetual optimization loop design.

## Recommended next steps

1. Wire market data ingestion to QuantConnect cloud exports.
2. Add regime filters and volatility targeting before any live deployment.
3. Build a staged optimization: coarse search then refine.
4. Gate promotion with CPCV and risk diagnostics.

## Training workflow (enabled)

The training workflow runs purged CV and evaluates models with MSE, MAE, IC,
directional accuracy, and Sharpe. Results are written to
`artifacts/ai_runs/<run_id>/training_summary.json`.
