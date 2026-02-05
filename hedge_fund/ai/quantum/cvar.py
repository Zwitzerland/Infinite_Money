"""CVaR helpers for portfolio objectives."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def cvar(returns: Sequence[float], alpha: float = 0.05) -> float:
    """Compute Conditional Value at Risk (CVaR)."""
    values = np.sort(np.asarray(returns, dtype=float))
    if values.size == 0:
        raise ValueError("returns must be non-empty")
    cutoff = max(1, int(alpha * len(values)))
    return float(values[:cutoff].mean())
