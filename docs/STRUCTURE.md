# Suggested Structure (Current + Target)

## Keep (core code)
- `hedge_fund/`
- `control_plane/`
- `optimizer/`
- `tools/`
- `gates/`
- `mcp_servers/`
- `knowledge/`

## Keep (docs + scripts)
- `docs/`
- `scripts/`

## Keep (data + artifacts)
- `data/` (local data, ignored by Git)
- `artifacts/` (generated outputs; keep README only)

## Keep (Lean + infra)
- `lean_projects/`
- `infra/`

## Research bundles (local, ignored by Git)
- `opencode-quant-platform/`
- `Trading_Research_Bundle/`

## Cleanup targets (optional)
- `.venv/`
- `**/__pycache__/`
- `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`
- old `backtests/` folders (Lean outputs)

See `docs/CLEANUP_CANDIDATES.md` for details and confirmation checklist.
