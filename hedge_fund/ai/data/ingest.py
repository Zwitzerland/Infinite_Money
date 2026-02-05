"""Market data ingestion orchestrator."""
from __future__ import annotations

from typing import Any, Mapping

from .market import MarketBar
from .sources import CSVMarketDataSource, IBKRMarketDataSource


def load_market_data(config: Mapping[str, Any]) -> list[MarketBar]:
    """Load market data based on configuration."""
    source = str(config.get("source", "csv")).lower()
    symbols = config.get("symbols", [])
    if not symbols:
        raise ValueError("market_data.symbols must be non-empty")

    if source == "csv":
        csv_cfg = config.get("csv", {})
        path = str(csv_cfg.get("path", "data/market_data.csv"))
        reader = CSVMarketDataSource(
            path=path,
            timestamp_column=str(csv_cfg.get("timestamp_column", "timestamp")),
            symbol_column=str(csv_cfg.get("symbol_column", "symbol")),
            tz=str(csv_cfg.get("tz", "UTC")),
        )
        return reader.fetch(symbols)

    if source == "ibkr":
        ibkr_cfg = config.get("ibkr", {})
        reader = IBKRMarketDataSource(
            host=str(ibkr_cfg.get("host", "127.0.0.1")),
            port=int(ibkr_cfg.get("port", 7497)),
            client_id=int(ibkr_cfg.get("client_id", 7)),
            duration=str(ibkr_cfg.get("duration", "2 Y")),
            bar_size=str(ibkr_cfg.get("bar_size", "1 day")),
            use_rth=bool(ibkr_cfg.get("use_rth", True)),
        )
        return reader.fetch(symbols)

    if source == "quantconnect":
        raise NotImplementedError(
            "QuantConnect export ingestion not implemented. "
            "Use CSV export or QC object store export first."
        )

    raise ValueError(f"Unsupported market_data.source: {source}")
