"""Calibration helpers for predictive intervals."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def conformal_interval(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    alpha: float = 0.1,
) -> float:
    """Compute a symmetric conformal radius from residuals."""
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    if residuals.size == 0:
        raise ValueError("Residuals must be non-empty")
    return float(np.quantile(residuals, 1 - alpha))
