"""Simple ensemble helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class Predictor(Protocol):
    """Minimal predictor protocol."""

    def predict(self, features: Sequence[Sequence[float]]) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class EnsembleMember:
    """Member of a weighted ensemble."""

    name: str
    weight: float
    predictor: Predictor


class WeightedEnsemble:
    """Weighted average ensemble."""

    def __init__(self, members: Sequence[EnsembleMember]) -> None:
        if not members:
            raise ValueError("members must be non-empty")
        self._members = members

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        predictions = [member.predict(features) for member in self._members]
        length = len(predictions[0])
        for series in predictions:
            if len(series) != length:
                raise ValueError("Ensemble members returned mismatched lengths")

        total_weight = sum(member.weight for member in self._members)
        if total_weight == 0:
            raise ValueError("Total ensemble weight must be > 0")

        combined: list[float] = []
        for idx in range(length):
            score = 0.0
            for member, series in zip(self._members, predictions):
                score += member.weight * float(series[idx])
            combined.append(score / total_weight)
        return combined
