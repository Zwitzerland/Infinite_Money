from datetime import datetime, timedelta

import pytest

from alphaquanta.agents.lean_core_agent import LeanCoreAgent
from alphaquanta.telemetry.acu_tracker import ACUTracker


@pytest.mark.asyncio
async def test_backtest_with_synthetic_data():
    symbol = "SPY"
    end = datetime.utcnow().date()
    start = end - timedelta(days=60)

    synthetic = []
    price = 100.0
    prev = None
    for i in range(60):
        price *= 1.0 + (0.001 if i % 6 else -0.002)
        synthetic.append(
            {
                "date": str(start + timedelta(days=i)),
                "symbol": symbol,
                "open": price * 0.995,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": 1000000,
                "price": price,
                "prev_price": prev if prev is not None else price,
            }
        )
        prev = price

    acu_tracker = ACUTracker(budget=3)
    agent = LeanCoreAgent(
        mode="backtest",
        quantum_enabled=False,
        config={"risk": {}, "position_sizing": {}, "lean": {}, "quantum": {"enabled": False}},
        acu_tracker=acu_tracker,
    )

    async def _hist(symbol: str, start_date: str, end_date: str):  # noqa: ARG001
        return synthetic

    agent.data_tool.get_historical_data = _hist  # type: ignore[attr-defined]

    result = await agent.run_backtest(symbol, str(start), str(end))
    assert result.total_trades >= 0
    assert -1.0 <= result.total_return <= 2.0





