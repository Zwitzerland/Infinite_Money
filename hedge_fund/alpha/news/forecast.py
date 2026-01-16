"""Impact forecasting utilities with conformal-style calibration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hedge_fund.data.news.models import MarketImpactObservation


@dataclass(frozen=True)
class ImpactForecast:
    """Forecast distribution for event impacts."""

    median: float
    lower: float
    upper: float


def conformal_interval(
    observations: Iterable[MarketImpactObservation], alpha: float = 0.1
) -> ImpactForecast:
    """Compute a simple conformal-like interval from historical impacts."""
    values = np.array([obs.returns for obs in observations], dtype=float)
    if values.size == 0:
        return ImpactForecast(median=0.0, lower=0.0, upper=0.0)
    median = float(np.median(values))
    lower = float(np.quantile(values, alpha / 2))
    upper = float(np.quantile(values, 1 - alpha / 2))
    return ImpactForecast(median=median, lower=lower, upper=upper)
