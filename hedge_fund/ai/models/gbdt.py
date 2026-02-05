"""Gradient-boosted tree model wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class GBDTForecaster:
    """LightGBM regressor wrapper."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    model: object | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "lightgbm not installed. Install with `pip install -e .[ai]`."
            ) from exc

        self.model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
        )
        self.model.fit(features, targets)

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        if self.model is None:
            raise ValueError("Model not fit")
        preds = self.model.predict(np.asarray(features))
        return preds.tolist()
