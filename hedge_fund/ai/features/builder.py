"""Feature frame builder for market bars."""
from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from ..data.market import MarketBar
from .technical import atr, rsi, rolling_vol, sma
from .regime import classify_regime


def build_feature_frame(
    bars: Iterable[MarketBar],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Build a feature frame from OHLCV bars."""
    df = pd.DataFrame(
        [
            {
                "symbol": bar.symbol,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )

    def _build_symbol(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values("timestamp").set_index("timestamp")
        close = pd.Series(frame["close"].to_numpy(), index=frame.index, dtype=float)
        windows_cfg = config.get("windows")
        if isinstance(windows_cfg, list):
            windows = [int(value) for value in windows_cfg]
        else:
            windows = [10, 20, 50]
        for window in windows:
            frame[f"sma_{window}"] = sma(close, int(window))
            frame[f"rsi_{window}"] = rsi(close, int(window))
            frame[f"vol_{window}"] = rolling_vol(close, int(window))

        high = pd.Series(frame["high"].to_numpy(), index=frame.index, dtype=float)
        low = pd.Series(frame["low"].to_numpy(), index=frame.index, dtype=float)
        frame["atr_14"] = atr(high, low, close, window=14)
        frame["regime"] = classify_regime(close)
        return frame.dropna()

    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    grouped = [_build_symbol(group) for _, group in df.groupby("symbol")]
    return pd.concat(grouped).reset_index().set_index(["symbol", "timestamp"])
