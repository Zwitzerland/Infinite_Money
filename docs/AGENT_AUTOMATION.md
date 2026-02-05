# Agent Automation Blueprint

This document describes how to run AI agents in a perpetual backtest/optimize
loop with strict validation, cost control, and promotion gates.

## Agent roles

- **Data Agent**: ingests, validates, and versions datasets.
- **Feature Agent**: builds feature snapshots and checks leakage.
- **Model Agent**: trains models and emits predictions + uncertainty.
- **Backtest Agent**: runs LEAN backtests and collects diagnostics.
- **Risk Agent**: enforces drawdown/leverage constraints and promotion gates.
- **Optimization Agent**: tunes parameters and orchestrates Optuna/LEAN.
- **Quantum Agent**: runs Braket portfolio selection experiments.
- **Execution Agent**: paper‑trade only; live requires manual approval.

## Perpetual loop

1. Ingest and validate new data window.
2. Update features + labels.
3. Train / refresh models.
4. Backtest and evaluate with CPCV + embargo.
5. Optimize parameters under constraints.
6. Risk gate + human approval.
7. Export signals and paper‑trade.
8. Monitor drift and repeat.

## Controls

- Hard caps on cost (QCC, Braket, AWS Batch).
- Promotion gates on deflated Sharpe and drawdown.
- Kill switch for data issues or unexpected drawdown.

## Implementation hooks

- `hedge_fund/ai/agents/orchestrator.py`
- `hedge_fund/ai/agents/loop.py` (CLI entry point)
- `hedge_fund/conf/agent_loop.yaml`
- `optimizer/` for Optuna and validation
- `docs/AWS_PIPELINE.md` for Step Functions orchestration
