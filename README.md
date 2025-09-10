# Infinite Money

Minimal skeleton for an event-driven trading research platform.

## Quickstart

```bash
pip install -e .[dev]
ruff check .
mypy hedge_fund tests
PYTHONPATH=. pytest -q
```

## Example

```python
from hedge_fund.backtest.events import OrderEvent, OrderSide
from hedge_fund.exec.simulated import SimulatedExecutionHandler

handler = SimulatedExecutionHandler({"AAPL": 100.0})
order = OrderEvent("AAPL", OrderSide.BUY, 100, 1_692_000_000)
fill = handler.place_order(order)
print(fill)
```
