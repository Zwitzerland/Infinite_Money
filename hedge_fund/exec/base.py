"""Execution handler interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hedge_fund.backtest.events import FillEvent, OrderEvent


class ExecutionHandler(ABC):
    """Abstract execution handler.

    Concrete implementations interact with a broker or simulation to
    place orders and return the resulting :class:`~hedge_fund.backtest.events.FillEvent`.
    """

    @abstractmethod
    def place_order(self, order: OrderEvent) -> FillEvent:
        """Submit *order* and return the resulting fill event."""
        raise NotImplementedError
