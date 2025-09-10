"""Tests for backtest events."""
from __future__ import annotations

from hedge_fund.backtest.events import (
    OrderEvent,
    OrderSide,
    FillEvent,
)


def test_order_event_fields() -> None:
    event = OrderEvent("AAPL", OrderSide.BUY, 100, 1_692_000_000)
    assert event.symbol == "AAPL"
    assert event.side is OrderSide.BUY
    assert event.order_type == "market"


def test_fill_event() -> None:
    fill = FillEvent("AAPL", OrderSide.SELL, 50, 150.0, 1_692_000_000, 1.0)
    assert fill.price == 150.0
    assert fill.commission == 1.0
