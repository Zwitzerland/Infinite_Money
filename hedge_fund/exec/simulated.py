"""Simple simulated execution handler for testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

from hedge_fund.backtest.events import FillEvent, OrderEvent

from .base import ExecutionHandler


@dataclass
class SimulatedExecutionHandler(ExecutionHandler):
    """Fill orders immediately at provided prices with optional liquidity limits.

    Parameters
    ----------
    prices:
        Mapping of symbol to fill price.
    liquidity:
        Available shares per symbol for a single call. If the order
        quantity exceeds this amount a partial fill occurs and the
        remaining liquidity is depleted.
    commission:
        Flat commission per trade.
    """

    prices: Mapping[str, float]
    liquidity: MutableMapping[str, int] = field(default_factory=dict)
    commission: float = 0.0

    def place_order(self, order: OrderEvent) -> FillEvent:
        price = self.prices[order.symbol]
        available = self.liquidity.get(order.symbol, order.quantity)
        fill_qty = min(order.quantity, available)
        self.liquidity[order.symbol] = available - fill_qty
        return FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=price,
            timestamp=order.timestamp,
            commission=self.commission,
        )
