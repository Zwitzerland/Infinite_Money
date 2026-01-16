# Infinite Money

Minimal skeleton for an event-driven trading research platform.

## Quickstart

```bash
pip install -e .[dev]
ruff check .
mypy hedge_fund tests
PYTHONPATH=. pytest -q
```

## Repository layout

- `hedge_fund/`: core research, backtest, execution, risk, and utilities.
- `control_plane/`: orchestration service for compile/backtest/live automation.
- `mcp_servers/`: Model Context Protocol tool integrations.
- `gates/`: promotion gate schemas and logic.
- `artifacts/`: lightweight examples of generated artifacts.
- `docs/`: operating doctrine, architecture, contracts, and runbooks.
- `docs/mcp_servers.md`: MCP server surface map.

## Example

```python
from hedge_fund.backtest.events import OrderEvent, OrderSide
from hedge_fund.exec.simulated import SimulatedExecutionHandler

handler = SimulatedExecutionHandler({"AAPL": 100.0})
order = OrderEvent("AAPL", OrderSide.BUY, 100, 1_692_000_000)
fill = handler.place_order(order)
print(fill)
```

## Contracts and doctrine

```bash
python -m hedge_fund.utils.contracts_cli --config-path conf --config-name contracts
```

See `docs/` for the operating doctrine, architecture diagram, contracts, and
runbook.

## Ingestion quickstart

```bash
python -m hedge_fund.data.ingest.cli --config-path conf --config-name ingest
```

## Available public sources (initial)

See `docs/alpha_fabric_sources.md` for the full source map, including OFR repo,
CFTC COT, BIS bulk data, OFAC sanctions, PatentsView, and FOIA/CIA reading room
entries for declassified corpora.

## News subsystem quickstart

```bash
python -m hedge_fund.data.news.cli --config-path conf --config-name news_ingest
```

See `docs/news_pipeline.md` for the event/analog/forecast pipeline.
