"""Linear model wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import ElasticNet


@dataclass
class LinearForecaster:
    """ElasticNet forecaster for tabular features."""

    alpha: float = 0.1
    l1_ratio: float = 0.5
    model: ElasticNet | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        self.model.fit(features, targets)

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        if self.model is None:
            raise ValueError("Model not fit")
        preds = self.model.predict(np.asarray(features))
        return preds.tolist()
