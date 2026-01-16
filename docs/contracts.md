# Immutable Contracts

This document defines the baseline contracts enforced by automation. The
canonical source is `hedge_fund.utils.contracts` and the contract bundle CLI.

## Data contract

- Required event schemas: `bars`, `trades`, `corporate_actions`, `orders`,
  `fills`.
- Partitioning by `symbol`, `date`, and `event_type`.
- Point-in-time joins enforced to avoid leakage.
- Checksums recorded for raw → curated → features lineage.
- Event-time ingestion uses the EventRecord schema for provenance.

## Backtest contract

- Minimum history window, embargo, and CPCV splits.
- Required metrics: Sharpe, Sortino, Calmar, drawdown, turnover.
- Artifact schema: run ID, params, metrics, seed, environment.

## Promotion contract

- Deflated Sharpe and probability-of-overfitting gates.
- Minimum number of out-of-sample windows.
- Minimum paper trading days.
- Regime and stress tests required.

## Execution contract

- IBKR pacing limits enforced.
- Maximum orders per second, turnover, leverage, and notional per order.
