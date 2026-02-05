# ADR 0002: PDF Knowledge Index

Date: 2026-01-22

## Status

Accepted

## Context

Reference PDFs (papers, specs, policies) must be primary sources and mapped to
code decisions for auditability.

## Decision

Add `knowledge/` with indexed PDFs, extracted text, and traceability mapping in
`docs/TRACEABILITY.md`.

## Consequences

- PDFs are hashed and extracted via `imctl knowledge sync`.
- The mapping file `knowledge/refs.yaml` becomes the traceability source.

## References

- PDFs: none (no PDF references registered yet).
