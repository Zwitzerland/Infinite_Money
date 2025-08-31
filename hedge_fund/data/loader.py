"""
Simple data loader that tries network (yfinance) and falls back to synthetic data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List


def _synthetic_series(symbol: str, start: str, end: str) -> List[Dict[str, Any]]:
    start_dt = datetime.fromisoformat(str(start))
    end_dt = datetime.fromisoformat(str(end))
    days = max(1, (end_dt - start_dt).days)
    price = 100.0
    rows: List[Dict[str, Any]] = []
    for i in range(days):
        price *= 1.0 + (0.001 if i % 5 != 0 else -0.002)
        rows.append(
            {
                "date": (start_dt + timedelta(days=i)).isoformat(),
                "symbol": symbol,
                "open": price * 0.995,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": 1_000_000,
            }
        )
    return rows


def load_ohlcv(symbol: str, start: str, end: str) -> List[Dict[str, Any]]:
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end)
        if hist is not None and not hist.empty:
            out: List[Dict[str, Any]] = []
            for date, row in hist.iterrows():
                out.append(
                    {
                        "date": date.isoformat(),
                        "symbol": symbol,
                        "open": float(row.get("Open", row.get("open", 0)) or 0),
                        "high": float(row.get("High", row.get("high", 0)) or 0),
                        "low": float(row.get("Low", row.get("low", 0)) or 0),
                        "close": float(row.get("Close", row.get("close", 0)) or 0),
                        "volume": int(row.get("Volume", row.get("volume", 0)) or 0),
                    }
                )
            return out
    except Exception:
        pass

    return _synthetic_series(symbol, start, end)





