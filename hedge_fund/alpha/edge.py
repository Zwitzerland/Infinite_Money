"""Edge discovery model using Lasso with purged cross-validation.

This module trains a sparse linear model to uncover trading edges while
mitigating look-ahead bias through purged K-fold splits. It also exposes
utility methods to evaluate Sharpe and deflated Sharpe ratios of the
resulting strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import LassoCV


def _purged_kfold(
    n: int, n_splits: int, purge: int
) -> Iterable[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Yield train/test indices with a purge window between sets."""
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test: NDArray[np.int_] = np.arange(start, stop, dtype=int)
        train: NDArray[np.int_] = np.concatenate(
            [
                np.arange(0, max(0, start - purge), dtype=int),
                np.arange(min(n, stop + purge), n, dtype=int),
            ]
        )
        yield train, test
        current = stop


@dataclass
class EdgeModel:
    """Lasso-based edge discovery with purged cross-validation."""

    n_splits: int = 3
    purge_pct: float = 0.1
    alphas: Sequence[float] | None = None
    random_state: int = 0
    model: LassoCV | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Fit the Lasso model using purged K-fold CV."""
        n = len(x)
        purge = int(self.purge_pct * n)
        cv = list(_purged_kfold(n, self.n_splits, purge))
        self.model = LassoCV(alphas=self.alphas, cv=cv, random_state=self.random_state)
        self.model.fit(x, y)

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Return model predictions."""
        if self.model is None:
            msg = "Model must be fit before prediction"
            raise ValueError(msg)
        preds = self.model.predict(x)
        return pd.Series(preds, index=x.index, name="prediction")

    @staticmethod
    def _strategy_returns(y_true: pd.Series, y_pred: pd.Series) -> NDArray[np.float64]:
        signal: NDArray[np.float64] = np.sign(y_pred).to_numpy()
        result: NDArray[np.float64] = signal * y_true.to_numpy(dtype=float)
        return result

    def sharpe_ratio(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute realized Sharpe ratio of sign strategy."""
        returns = self._strategy_returns(y_true, y_pred)
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        if std == 0:
            return 0.0
        return mean / std

    def deflated_sharpe_ratio(
        self, y_true: pd.Series, y_pred: pd.Series, trials: int = 1
    ) -> float:
        """Sharpe ratio adjusted for multiple testing.

        Parameters
        ----------
        y_true:
            Realized returns.
        y_pred:
            Predicted returns from the model.
        trials:
            Number of trials or explored strategies.
        """
        sr = self.sharpe_ratio(y_true, y_pred)
        n = len(y_true)
        sr_std = sqrt((1.0 + 0.5 * sr**2) / n)
        z = NormalDist().inv_cdf(1 - 1 / trials) if trials > 1 else 0.0
        return sr - z * sr_std

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Model coefficients after fitting."""
        if self.model is None:
            msg = "Model has not been fit"
            raise ValueError(msg)
        coef: NDArray[np.float64] = self.model.coef_
        return coef
