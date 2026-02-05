"""Risk overlays for model-driven positions."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def apply_vol_target(positions: Sequence[float], volatility: Sequence[float], target: float = 0.15) -> list[float]:
    """Scale positions to target volatility."""
    pos = np.asarray(positions, dtype=float)
    vol = np.asarray(volatility, dtype=float)
    scale = np.where(vol > 0, target / vol, 0.0)
    return (pos * scale).tolist()


def apply_drawdown_stop(equity_curve: Sequence[float], positions: Sequence[float], max_drawdown: float) -> list[float]:
    """Zero positions when drawdown exceeds a threshold."""
    equity = np.asarray(equity_curve, dtype=float)
    pos = np.asarray(positions, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    stopped = np.where(drawdown > max_drawdown, 0.0, pos)
    return stopped.tolist()
