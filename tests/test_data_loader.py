from datetime import datetime, timedelta

from hedge_fund.data.loader import load_ohlcv


def test_loader_returns_data():
    end = datetime.utcnow().date()
    start = end - timedelta(days=10)
    rows = load_ohlcv("SPY", str(start), str(end))
    assert isinstance(rows, list)
    assert len(rows) > 0
    first = rows[0]
    for key in ["date", "symbol", "open", "high", "low", "close", "volume"]:
        assert key in first





