# Cleanup Candidates (Review Before Delete)

These are typically safe to remove, but confirm before deleting.

## Local environments
- `.venv/` (recreate with `python -m venv .venv`)

## Python caches
- `**/__pycache__/`
- `**/*.pyc`
- `.pytest_cache/`
- `.ruff_cache/`
- `.mypy_cache/`

## Generated outputs
- `lean_projects/*/backtests/` (LEAN output; keep if you need history)
- `artifacts/` (keep only `artifacts/README.md`)
- `outputs/`
- `storage/`

## Knowledge outputs
- `knowledge/pdfs/` (raw PDFs)
- `knowledge/extracted/` (derived text)
- `knowledge/index.json`

## Local datasets
- `data/` (sample + export data)

## Agent workspaces
- `.ralphy-sandboxes/`
- `.ralphy-worktrees/`

If you want me to delete these, tell me which sections to remove.
