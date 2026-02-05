# Repository Map

Root: `./`

Total files: varies (run `imctl doctor` to regenerate)

## Top-level directories
- `artifacts/` (generated output; keep README only)
- `configs/` (LEAN + optimization configs)
- `control_plane/` (QuantConnect orchestration)
- `data/` (local datasets, ignored by Git)
- `docs/` (documentation)
- `gates/` (promotion gates)
- `hedge_fund/` (core research + execution stack)
- `infra/` (infrastructure scaffolding)
- `knowledge/` (refs tracked; PDFs/extracted ignored)
- `lean_projects/` (LEAN templates; backtests ignored)
- `mcp_servers/` (MCP integrations)
- `optimizer/` (Optuna + LEAN optimization tooling)
- `scripts/` (automation scripts)
- `storage/` (local storage, ignored by Git)
- `tests/` (unit tests)
- `tools/` (repo CLIs, including `imctl`)

## Notable items
- `lean_projects/`: LEAN strategy projects
- `optimizer/`: Optuna + LEAN optimization tooling
- `hedge_fund/`: core AI/exec/backtest modules
- `knowledge/`: PDF knowledge index
- `tools/imctl/`: repo command center

## Possible duplicates / templates
- `lean_projects/QC_Template` (template)

## Optional local bundles (ignored by Git)
- `opencode-quant-platform/`
- `Trading_Research_Bundle/`
