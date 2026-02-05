"""Market data ingestion helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class MarketBar:
    """OHLCV bar for model training."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_market_data(config: Mapping[str, Any]) -> list[MarketBar]:
    """Fetch market data for training."""
    from .ingest import load_market_data

    return load_market_data(config)
