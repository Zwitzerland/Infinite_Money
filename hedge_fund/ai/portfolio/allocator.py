"""Portfolio allocation helpers."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_variance_weights(
    expected_returns: Sequence[float],
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """Compute mean-variance weights (unconstrained)."""
    mu = np.asarray(expected_returns, dtype=float)
    inv_cov = np.linalg.pinv(covariance)
    raw = inv_cov @ mu
    weights = raw / (raw.sum() if raw.sum() != 0 else 1.0)
    return weights * (1.0 / risk_aversion)


def risk_parity_weights(covariance: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Compute simple risk parity weights via iterative scaling."""
    n = covariance.shape[0]
    weights = np.ones(n) / n
    for _ in range(max_iter):
        portfolio_var = weights @ covariance @ weights
        marginal = covariance @ weights
        risk_contrib = weights * marginal / (portfolio_var if portfolio_var != 0 else 1.0)
        target = np.ones(n) / n
        weights *= target / np.maximum(risk_contrib, 1e-8)
        weights /= weights.sum()
    return weights
