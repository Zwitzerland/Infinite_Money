"""Validation helpers for ML experiments."""
from __future__ import annotations

from typing import Iterable

from optimizer.validation import cpcv_indices, purged_kfold_indices, summarize_cpcv_scores


def summarize_scores(scores: Iterable[float]) -> dict[str, float]:
    """Summarize CPCV scores into a compact dict."""
    summary = summarize_cpcv_scores(scores)
    return {
        "mean": summary.mean,
        "median": summary.median,
        "p10": summary.p10,
        "p90": summary.p90,
    }


__all__ = [
    "cpcv_indices",
    "purged_kfold_indices",
    "summarize_scores",
]
