# Operating Doctrine (Initial)

This document captures the initial operating doctrine for the Infinite Money
platform. It encodes the first-pass system goals, reality constraints, and
promotion gates to prevent unbounded self-modification.

## Core thesis

We do not seek failproof profits. We build a cloud-hosted, audit-grade,
multi-agent research → validation → deployment factory around QuantConnect LEAN
and Interactive Brokers, where only statistically-defensible and
execution-feasible changes are promoted. Quantum computation is used as an
asynchronous optimization kernel for discrete decisions, not as a low-latency
trading brain.

## System goal

1. Continuously ingest market + reference data.
2. Generate strategy hypotheses and parameterizations.
3. Implement changes as version-controlled code.
4. Backtest + validate with anti-overfitting gates.
5. Paper trade and canary trade with risk governors.
6. Promote only after all gates pass.
7. Monitor live trading and rollback on anomalies.
8. Use quantum computation in batch mode for discrete optimization problems.

## Non-negotiable constraints

1. Interactive Brokers request pacing and order-rate limits are enforced in
   execution and risk contracts.
2. QuantConnect LEAN is the canonical runtime for strategy execution.
3. Learning is gated (shadow evaluation + canary exposure); no uncontrolled
   self-modification.

## Phase 1 baseline deliverables

1. Repository layout per the `hedge_fund/` package structure.
2. Immutable contracts for data, backtest, promotion, execution.
3. Standardized contract bundle CLI for automation.
4. Runbooks and architecture diagrams in `docs/`.

## Risks and caveats

- **Data risk:** corporate actions, timezones, and point-in-time joins must be
  enforced to prevent leakage and survivorship bias.
- **Microstructure risk:** pacing, slippage, and order-rate limits constrain
  feasible execution and must be encoded in execution contracts.
- **Overfitting risk:** promotion gates must deflate performance and enforce
  out-of-sample windows before any deployment.

## Alpha fabric ingestion

The ingestion layer normalizes all sources into a single EventRecord schema
with explicit event-time, publish-time, and ingestion-time fields. This enables
latency-aware backtesting and provenance tracking across macro, regulatory, and
alternative data sources.

Initial connectors cover free public sources (FRED, BLS, Treasury, USGS, GDELT,
OFAC SDN, PatentsView) with a source registry tracking licensing and rate
limits.
