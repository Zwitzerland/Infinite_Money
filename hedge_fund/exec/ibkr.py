"""Interactive Brokers execution adapter using :mod:`ib_insync`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from hedge_fund.backtest.events import FillEvent, OrderEvent

from .base import ExecutionHandler

if TYPE_CHECKING:  # pragma: no cover - hints only
    from ib_insync import IB


@dataclass
class IBKRExecutionHandler(ExecutionHandler):
    """Place orders via an ``ib_insync.IB`` connection.

    Notes
    -----
    This adapter requires a running TWS or IB Gateway instance and is
    intentionally minimal. It returns a placeholder :class:`FillEvent`
    because full order life-cycle management requires asynchronous
    callbacks which are beyond this skeleton.
    """

    ib: "IB"

    def place_order(self, order: OrderEvent) -> FillEvent:  # pragma: no cover - network
        from ib_insync import MarketOrder, Stock

        contract = Stock(order.symbol, "SMART", "USD")
        if order.order_type == "market":
            ib_order = MarketOrder(order.side.value, order.quantity)
        else:
            raise NotImplementedError("limit orders not implemented")

        trade = self.ib.placeOrder(contract, ib_order)
        # In real usage, we'd await fills; here we emit a naive fill at last price.
        price: Optional[float] = getattr(trade.orderStatus, "avgFillPrice", None)
        if price is None:
            price = 0.0
        return FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=order.timestamp,
            commission=0.0,
        )
