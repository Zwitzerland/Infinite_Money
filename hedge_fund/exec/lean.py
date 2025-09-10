"""QuantConnect LEAN execution adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hedge_fund.backtest.events import FillEvent, OrderEvent

from .base import ExecutionHandler

if TYPE_CHECKING:  # pragma: no cover - hints only
    from QuantConnect.Algorithm import QCAlgorithm  # type: ignore[import-not-found]


@dataclass
class LeanExecutionHandler(ExecutionHandler):
    """Delegate orders to a running QCAlgorithm instance."""

    algorithm: "QCAlgorithm"

    def place_order(self, order: OrderEvent) -> FillEvent:  # pragma: no cover - framework
        if order.order_type == "market":
            ticket = self.algorithm.MarketOrder(order.symbol, order.quantity)
        else:
            ticket = self.algorithm.LimitOrder(order.symbol, order.quantity, 0.0)
        order_event = ticket.OrderEvents[-1]
        return FillEvent(
            symbol=order_event.Symbol.Value,
            side=order.side,
            quantity=order_event.FillQuantity,
            price=order_event.FillPrice,
            timestamp=int(order_event.UtcTime.timestamp()),
            commission=order_event.OrderFee.Value.Amount,
        )
