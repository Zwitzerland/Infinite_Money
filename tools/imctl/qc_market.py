"""Utilities for working with QuantConnect (LEAN) datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import pandas as pd


@dataclass(frozen=True)
class QCDailyEquityExport:
    """Export settings for QuantConnect daily equity OHLCV."""

    zip_path: Path
    symbol: str
    output_csv: Path
    price_scale: float = 10000.0
    start: str | None = None
    end: str | None = None


def export_qc_daily_equity_to_market_csv(settings: QCDailyEquityExport) -> Path:
    """Convert QC `equity/usa/daily/<symbol>.zip` into the AI market CSV format."""

    zip_path = settings.zip_path
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if not names:
            raise ValueError(f"Empty zip file: {zip_path}")

        # Typical QC layout is a single `<symbol>.csv` inside the zip.
        with archive.open(names[0]) as handle:
            raw = pd.read_csv(
                handle,
                header=None,
                names=["date", "open", "high", "low", "close", "volume"],
            )

    timestamp = pd.to_datetime(raw["date"], format="%Y%m%d %H:%M", utc=True, errors="coerce")
    df = raw.assign(timestamp=timestamp).dropna(subset=["timestamp"])
    if settings.start:
        df = df[df["timestamp"] >= pd.to_datetime(settings.start, utc=True)]
    if settings.end:
        df = df[df["timestamp"] <= pd.to_datetime(settings.end, utc=True)]

    scale = float(settings.price_scale)
    df = df.assign(
        symbol=str(settings.symbol),
        open=df["open"].astype(float) / scale,
        high=df["high"].astype(float) / scale,
        low=df["low"].astype(float) / scale,
        close=df["close"].astype(float) / scale,
        volume=df["volume"].astype(float),
    )

    output_path = settings.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].to_csv(
        output_path,
        index=False,
    )
    return output_path
