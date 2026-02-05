from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from hedge_fund.alpha.sr import SRBarrierParams, compute_sr_barrier_exposure


def _make_feature_frame(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: float = 1.0,
    symbol: str = "SPY",
) -> pd.DataFrame:
    start = datetime(2020, 1, 1)
    index = [start + timedelta(days=int(i)) for i in range(close.shape[0])]
    mi = pd.MultiIndex.from_product([[symbol], index], names=["symbol", "timestamp"])
    frame = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "atr_14": np.full_like(close, atr, dtype=float),
            "regime": np.zeros_like(close, dtype=int),
        },
        index=mi,
    )
    return frame


def test_sr_barrier_zero_when_cost_impossible() -> None:
    rng = np.random.default_rng(123)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 1, n)).astype(float)
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    features = _make_feature_frame(close=close, high=high, low=low, atr=1.0)

    # cost_atr=1 makes p* = (1+1)/(1+1) = 1, so edge is impossible.
    params = SRBarrierParams(
        pivot_lookback=2,
        train_window=100,
        horizon=5,
        zone_atr=0.5,
        tp_atr=1.0,
        sl_atr=1.0,
        cost_atr=1.0,
        min_resolved_events=5,
    )
    exposure = compute_sr_barrier_exposure(features, params)
    assert float(exposure.abs().max()) == 0.0


def test_sr_barrier_detects_synthetic_bounce_edge() -> None:
    # Deterministic oscillation: support at 100 gets hit and price reliably bounces to 102.
    cycle = np.array([100.0, 101.0, 102.0, 101.0], dtype=float)
    close = np.tile(cycle, 80)
    high = close + 0.1
    low = close - 0.1
    features = _make_feature_frame(close=close, high=high, low=low, atr=1.0)

    params = SRBarrierParams(
        pivot_lookback=1,
        train_window=60,
        horizon=3,
        zone_atr=0.2,
        tp_atr=1.0,
        sl_atr=1.0,
        cost_atr=0.0,
        min_resolved_events=10,
        confidence=0.80,
        kelly_fraction=0.5,
        max_exposure=1.0,
        use_regime_filter=False,
    )
    exposure = compute_sr_barrier_exposure(features, params)
    assert float(exposure.abs().max()) <= params.max_exposure + 1e-9
    # Should emit some non-zero exposures once enough events resolve.
    assert float((exposure.abs() > 0).sum()) > 0


def test_sr_barrier_requires_single_symbol() -> None:
    close = np.array([100.0, 101.0, 100.0, 101.0, 100.0], dtype=float)
    high = close + 0.1
    low = close - 0.1

    a = _make_feature_frame(close=close, high=high, low=low, symbol="A")
    b = _make_feature_frame(close=close, high=high, low=low, symbol="B")
    features = pd.concat([a, b]).sort_index()

    params = SRBarrierParams(pivot_lookback=1, train_window=10, horizon=2, min_resolved_events=1)
    try:
        compute_sr_barrier_exposure(features, params)
    except ValueError as exc:
        assert "exactly 1 symbol" in str(exc)
    else:
        raise AssertionError("expected ValueError")
