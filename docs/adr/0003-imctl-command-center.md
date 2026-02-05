# ADR 0003: imctl Command Center

Date: 2026-01-22

## Status

Accepted

## Context

We need a single CLI to validate environments, run backtests/optimizations, and
manage knowledge synchronization and reporting.

## Decision

Create `imctl` under `tools/imctl/` with subcommands for doctor, backtest,
optimize, report, and knowledge sync.

## Consequences

- All experiments write to a standardized ledger.
- A single command can run the local pipeline end-to-end.

## References

- PDFs: none (no PDF references registered yet).
