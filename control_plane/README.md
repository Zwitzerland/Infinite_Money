# Control Plane

The control plane orchestrates research → validation → promotion workflows and
is the only system authorized to trigger QuantConnect compile/backtest/live
operations. Tool integrations should be surfaced via Model Context Protocol
servers.

## Quickstart

```bash
python -m hedge_fund.utils.contracts_cli
```

## Configuration

- `INFINITE_MONEY_QUANTCONNECT_API_TOKEN` (required for QuantConnect calls)
- `INFINITE_MONEY_QUANTCONNECT_BASE_URL` (optional, defaults to QC v2 API)

Contract bundle defaults live in `hedge_fund/conf/contracts.yaml`.

## QuantConnect CLI

Set the API token as an environment variable (do not hardcode it):

```bash
setx INFINITE_MONEY_QUANTCONNECT_API_TOKEN "<TOKEN>"
```

Compile a project:

```bash
python -m control_plane.quantconnect_cli compile --project-id 123456 --name "smoke-compile"
```

Create a backtest:

```bash
python -m control_plane.quantconnect_cli backtest --project-id 123456 --name "smoke-backtest"
```

Upload a file to a project:

```bash
python -m control_plane.quantconnect_cli upload-file --project-id 123456 --remote-name "main.py" --path "./my_algo.py"
```

Live deployments are intentionally not wired here; they must flow through
promotion gates and control-plane automation.
