# Control Plane

The control plane orchestrates research → validation → promotion workflows and
is the only system authorized to trigger QuantConnect compile/backtest/live
operations. Tool integrations should be surfaced via Model Context Protocol
servers.

## Quickstart

```bash
python -m hedge_fund.utils.contracts_cli --config-path conf --config-name contracts
```
