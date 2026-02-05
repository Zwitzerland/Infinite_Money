"""Label generation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, cast


def _to_series(close: pd.Series | Sequence[float]) -> pd.Series:
    if isinstance(close, pd.Series):
        return close
    return cast(pd.Series, pd.Series(list(close)))


def forward_returns(
    close: pd.Series | Sequence[float],
    horizon: int = 5,
) -> pd.Series | pd.DataFrame:
    """Compute forward returns for a given horizon."""
    series = _to_series(close)
    return cast(pd.Series, series.pct_change(periods=horizon).shift(-horizon))


def volatility_adjusted_returns(
    close: pd.Series | Sequence[float],
    horizon: int = 5,
    vol_window: int = 20,
) -> pd.Series | pd.DataFrame:
    """Return normalized by rolling volatility."""
    series = _to_series(close)
    returns = forward_returns(series, horizon=horizon)
    log_values = np.log(series.to_numpy(dtype=float))
    log_returns = pd.Series(log_values, index=series.index).diff()
    vol = log_returns.rolling(vol_window, min_periods=vol_window).std()
    return returns / vol.shift(-horizon)


def direction_label(
    close: pd.Series | Sequence[float],
    horizon: int = 5,
) -> pd.Series | pd.DataFrame:
    """Binary direction label (1 for up, 0 for down)."""
    series = _to_series(close)
    return (forward_returns(series, horizon=horizon) > 0).astype(int)
