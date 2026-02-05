"""CSV-backed market data source."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, cast

import pandas as pd

from ..market import MarketBar


@dataclass(frozen=True)
class CSVMarketDataSource:
    """Load OHLCV data from CSV files."""

    path: str
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"
    tz: str | None = "UTC"

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | None = None,
        end: str | None = None,
    ) -> list[MarketBar]:
        df: Any = cast(Any, pd.read_csv(Path(self.path)))
        if self.symbol_column in df.columns:
            df = cast(Any, df)[df[self.symbol_column].isin(symbols)]
        if self.timestamp_column not in df.columns:
            raise ValueError("timestamp column missing in CSV")

        timestamps = pd.to_datetime(df[self.timestamp_column], utc=True, errors="coerce")
        df = cast(Any, df).assign(timestamp=timestamps)
        if start:
            df = cast(Any, df)[df["timestamp"] >= pd.to_datetime(start, utc=True)]
        if end:
            df = cast(Any, df)[df["timestamp"] <= pd.to_datetime(end, utc=True)]

        bars: list[MarketBar] = []
        records: list[dict[str, Any]] = cast(Any, df).to_dict("records")
        for record in records:
            ts = pd.Timestamp(record["timestamp"])
            if pd.isna(ts):
                continue
            timestamp = datetime.fromisoformat(str(ts))
            bars.append(
                MarketBar(
                    symbol=str(record.get(self.symbol_column, symbols[0])),
                    timestamp=timestamp,
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=float(record.get("volume", 0.0) or 0.0),
                )
            )
        return bars
