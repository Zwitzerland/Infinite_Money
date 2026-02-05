"""Base protocol for market data sources."""
from __future__ import annotations

from typing import Protocol, Sequence

from ..market import MarketBar


class MarketDataSource(Protocol):
    """Protocol for market data sources."""

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | None = None,
        end: str | None = None,
    ) -> list[MarketBar]:
        ...
