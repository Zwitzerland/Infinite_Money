# ADR 0001: LEAN Project Folder as Source of Truth

Date: 2026-01-22

## Status

Accepted

## Context

We need a reproducible, auditable system where research, backtests, and
optimizations are derived from a single source of truth.

## Decision

Treat the LEAN project folders under `lean_projects/` as the canonical source
for strategy code and parameters. All backtests and optimizations must reference
these folders.

## Consequences

- All automation must accept a LEAN project path.
- Artifacts capture the project and parameters for traceability.

## References

- PDFs: none (no PDF references registered yet).
