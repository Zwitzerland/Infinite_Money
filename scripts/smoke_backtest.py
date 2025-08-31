#!/usr/bin/env python3
"""
Smoke backtest running the hardened engine with data loaded via hedge_fund/data/loader.py.
Falls back to synthetic data if network is blocked.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

from hedge_fund.data.loader import load_ohlcv

from alphaquanta.agents.lean_core_agent import LeanCoreAgent
from alphaquanta.telemetry.acu_tracker import ACUTracker


def _to_engine_format(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    prev_close = None
    for row in rows:
        price = float(row.get("close", row.get("price", 100.0)))
        formatted.append(
            {
                "date": row.get("date") or row.get("timestamp") or datetime.utcnow().isoformat(),
                "symbol": row.get("symbol", "SPY"),
                "open": float(row.get("open", price)),
                "high": float(row.get("high", price)),
                "low": float(row.get("low", price)),
                "close": price,
                "volume": int(row.get("volume", 1000000)),
                "price": price,
                "prev_price": prev_close if prev_close is not None else float(row.get("open", price)),
            }
        )
        prev_close = price
    return formatted


async def main():
    symbol = "SPY"
    end = datetime.utcnow().date()
    start = end - timedelta(days=90)

    rows = load_ohlcv(symbol=symbol, start=str(start), end=str(end))
    data = _to_engine_format(rows)

    acu_tracker = ACUTracker(budget=5)
    agent = LeanCoreAgent(
        mode="backtest",
        quantum_enabled=False,
        config={
            "risk": {},
            "position_sizing": {},
            "lean": {},
            "quantum": {"enabled": False},
        },
        acu_tracker=acu_tracker,
        qpu_tracker=None,
    )

    # Monkeypatch agent to use preloaded data instead of network
    async def _get_hist(symbol: str, start_date: str, end_date: str):
        return data

    agent.data_tool.get_historical_data = _get_hist  # type: ignore[attr-defined]

    result = await agent.run_backtest(symbol=symbol, start_date=str(start), end_date=str(end))

    print(
        f"SMOKE: trades={result.total_trades}, sharpe={result.sharpe_ratio:.3f}, return={result.total_return:.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())





