"""Regime classification helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regime(close: pd.Series, trend_window: int = 50, vol_window: int = 20) -> pd.Series:
    """Classify market regime based on trend and volatility.

    Returns labels: 1 (trend), 0 (range), -1 (high-vol drawdown).
    """
    trend = close / close.rolling(trend_window, min_periods=trend_window).mean() - 1
    vol = np.log(close).diff().rolling(vol_window, min_periods=vol_window).std()
    trend_flag = trend > 0
    vol_flag = vol > vol.median()
    regime = pd.Series(0, index=close.index, dtype=int)
    regime[trend_flag & ~vol_flag] = 1
    regime[~trend_flag & vol_flag] = -1
    return regime
