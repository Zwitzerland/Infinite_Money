"""Simple real-data alpha sleeves (research-only)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def build_alpha_signal(close: pd.Series, windows: Sequence[int]) -> pd.Series:
    returns = close.pct_change().dropna()
    win = list(windows) if windows else [20, 60, 120]
    while len(win) < 3:
        win.append(win[-1])

    w_fast, w_mid, w_slow = win[:3]

    mom = np.tanh(returns.rolling(w_fast).mean().fillna(0.0) * 400) * returns
    mean_rev = -returns.diff().fillna(0.0) * 0.5
    breakout = np.sign(returns) * returns.rolling(w_mid).std().fillna(0.0)

    sleeves = pd.DataFrame({"momo": mom, "revert": mean_rev, "breakout": breakout})
    sharpe = sleeves.rolling(w_slow).mean() / (sleeves.rolling(w_slow).std() + 1e-9)
    sharpe = sharpe.clip(lower=0.0)
    weights = sharpe.div(sharpe.sum(axis=1), axis=0).fillna(1.0 / sleeves.shape[1])
    alpha = (weights * sleeves).sum(axis=1)
    return alpha
