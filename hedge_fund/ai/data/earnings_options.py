"""Earnings-related options snapshot ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd


@dataclass(frozen=True)
class EarningsOptionSnapshot:
    """Snapshot of option volatility inputs around earnings."""

    symbol: str
    asof_date: datetime
    earnings_date: datetime
    underlying_price: float
    iv_front: float
    iv_back: float
    rv_20d: float
    front_dte: int
    back_dte: int
    option_volume: float
    open_interest: float
    bid_ask_spread: float


def _required_columns() -> Sequence[str]:
    return (
        "symbol",
        "asof_date",
        "earnings_date",
        "underlying_price",
        "iv_front",
        "iv_back",
        "rv_20d",
        "front_dte",
        "back_dte",
    )


def load_earnings_option_snapshots(
    path: str,
    tz: str | None = "UTC",
) -> list[EarningsOptionSnapshot]:
    """Load earnings volatility snapshots from CSV."""
    df: Any = cast(Any, pd.read_csv(Path(path)))
    missing = [col for col in _required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    asof = pd.to_datetime(df["asof_date"], utc=True, errors="coerce")
    earnings = pd.to_datetime(df["earnings_date"], utc=True, errors="coerce")
    df = cast(Any, df).assign(asof_date=asof, earnings_date=earnings)

    snapshots: list[EarningsOptionSnapshot] = []
    records: list[dict[str, Any]] = cast(Any, df).to_dict("records")
    for record in records:
        if pd.isna(record["asof_date"]) or pd.isna(record["earnings_date"]):
            continue
        snapshots.append(
            EarningsOptionSnapshot(
                symbol=str(record["symbol"]),
                asof_date=datetime.fromisoformat(str(record["asof_date"])),
                earnings_date=datetime.fromisoformat(str(record["earnings_date"])),
                underlying_price=float(record["underlying_price"]),
                iv_front=float(record["iv_front"]),
                iv_back=float(record["iv_back"]),
                rv_20d=float(record["rv_20d"]),
                front_dte=int(record["front_dte"]),
                back_dte=int(record["back_dte"]),
                option_volume=float(record.get("option_volume", 0.0) or 0.0),
                open_interest=float(record.get("open_interest", 0.0) or 0.0),
                bid_ask_spread=float(record.get("bid_ask_spread", 0.0) or 0.0),
            )
        )
    return snapshots


def fetch_earnings_option_snapshots(config: Mapping[str, Any]) -> list[EarningsOptionSnapshot]:
    """Fetch earnings option data based on configuration."""
    path = str(config.get("path", "data/options/earnings_snapshots.csv"))
    tz = cast(str | None, config.get("tz", "UTC"))
    return load_earnings_option_snapshots(path, tz=tz)
