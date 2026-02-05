"""Evaluation metrics for forecasts and strategies."""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Sequence

import numpy as np


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(diff**2))


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    diff = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.mean(diff))


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    direction_true = np.sign(np.asarray(y_true))
    direction_pred = np.sign(np.asarray(y_pred))
    return float(np.mean(direction_true == direction_pred))


def information_coefficient(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) < 2:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def sharpe_ratio(returns: Sequence[float], periods: int = 252) -> float:
    series = np.asarray(returns)
    mean = series.mean()
    std = series.std(ddof=1)
    if std == 0:
        return 0.0
    return float(mean / std * sqrt(periods))


def sortino_ratio(returns: Sequence[float], periods: int = 252) -> float:
    series = np.asarray(returns)
    downside = series[series < 0]
    if downside.size == 0:
        return 0.0
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return float(series.mean() / downside_std * sqrt(periods))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    values = np.asarray(equity_curve)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return float(np.max(drawdown))


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute PSR from Bailey et al. (approx)."""
    if n <= 1:
        return 0.0
    numerator = (sharpe - benchmark) * sqrt(n - 1)
    denominator = sqrt(1 - skew * sharpe + 0.25 * (kurtosis - 1) * sharpe**2)
    if denominator == 0:
        return 0.0
    return float(NormalDist().cdf(numerator / denominator))


def deflated_sharpe_ratio(
    sharpe: float,
    n: int,
    trials: int = 1,
) -> float:
    """Compute deflated Sharpe ratio to adjust for multiple testing."""
    if n <= 1:
        return 0.0
    sr_std = sqrt((1.0 + 0.5 * sharpe**2) / n)
    z = NormalDist().inv_cdf(1 - 1 / trials) if trials > 1 else 0.0
    return sharpe - z * sr_std
