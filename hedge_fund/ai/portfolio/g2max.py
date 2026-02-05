"""G2MAX compounding utilities.

This module defines a conservative, research-only compounding rule that blends
fractional Kelly sizing, volatility targeting, and drawdown throttles. It does
not place orders or provide trading advice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class G2MaxParams:
    """Parameters for the G2MAX compounding rule."""

    phi_base: float = 0.4
    vol_target: float = 0.14
    drawdown_soft: float = 0.10
    drawdown_hard: float = 0.20
    leverage: float = 2.5
    ewma_lambda: float = 0.94
    lookback: int = 60


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    out = np.full_like(x, 0.0, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(x[start : i + 1]))
    return out


def _rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.full_like(x, 0.0, dtype=float)
    out = np.full_like(x, 0.0, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = float(np.var(x[start : i + 1]))
    return out


def _ewma_variance(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    prev = 0.0
    for i, r in enumerate(x):
        prev = lam * prev + (1.0 - lam) * (r ** 2)
        out[i] = prev
    return out


def g2max_exposure(
    returns: np.ndarray,
    params: G2MaxParams,
    alpha_returns: np.ndarray | None = None,
    alpha_weight: float = 0.4,
) -> np.ndarray:
    """Compute G2MAX exposure series for a return stream.

    If alpha_returns is supplied, exposures are derived from a blended return
    stream while equity updates use the raw returns.
    """

    if alpha_returns is not None:
        if len(alpha_returns) != len(returns):
            raise ValueError("alpha_returns length must match returns")
        effective_returns = (1.0 - alpha_weight) * returns + alpha_weight * alpha_returns
    else:
        effective_returns = returns

    mu_hat = _rolling_mean(effective_returns, params.lookback)
    var_hat = _rolling_var(effective_returns, params.lookback)
    var_hat = np.where(var_hat <= 0.0, np.nan, var_hat)
    kelly = np.nan_to_num(mu_hat / var_hat, nan=0.0)
    kelly = np.clip(kelly, -params.leverage, params.leverage)

    ewvar = _ewma_variance(effective_returns, params.ewma_lambda)
    vol_ann = np.sqrt(ewvar) * np.sqrt(252.0)
    vol_scale = np.clip(params.vol_target / (vol_ann + 1e-12), 0.0, params.leverage)

    exposure = np.zeros_like(returns, dtype=float)
    equity = 1.0
    peak = 1.0
    for i, r in enumerate(returns):
        peak = max(peak, equity)
        mdd = (peak - equity) / peak if peak > 0 else 0.0
        if mdd >= params.drawdown_hard:
            gate = 0.25
        elif mdd >= params.drawdown_soft:
            gate = 0.5
        else:
            gate = 1.0
        expo = params.phi_base * kelly[i] * vol_scale[i] * gate
        exposure[i] = float(np.clip(expo, -params.leverage, params.leverage))
        equity = max(equity * (1.0 + exposure[i] * r), 1e-12)
    return exposure


def g2max_equity_curve(returns: np.ndarray, params: G2MaxParams) -> np.ndarray:
    """Compute equity curve using G2MAX exposure sizing."""

    exposure = g2max_exposure(returns, params)
    equity = np.ones_like(returns, dtype=float)
    value = 1.0
    for i, r in enumerate(returns):
        value = max(value * (1.0 + exposure[i] * r), 1e-12)
        equity[i] = value
    return equity
