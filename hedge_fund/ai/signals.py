"""Signal post-processing helpers."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def score_to_positions(scores: Sequence[float], threshold: float = 0.0) -> list[int]:
    """Convert model scores to long/short/flat positions."""
    positions = []
    for score in scores:
        if score > threshold:
            positions.append(1)
        elif score < -threshold:
            positions.append(-1)
        else:
            positions.append(0)
    return positions


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Normalize scores to z-scores."""
    values = np.asarray(scores, dtype=float)
    mean = values.mean()
    std = values.std()
    if std == 0:
        return values.tolist()
    return ((values - mean) / std).tolist()
