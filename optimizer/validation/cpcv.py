"""Combinatorial purged cross-validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, Tuple

import numpy as np

from .purged_cv import purged_kfold_indices


@dataclass(frozen=True)
class CPCVScoreSummary:
    """Summary of CPCV path scores."""

    mean: float
    median: float
    p10: float
    p90: float


def cpcv_indices(
    n_samples: int,
    n_splits: int,
    n_test_folds: int,
    purge: int,
    embargo: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices for CPCV paths.

    Each path is a combination of test folds. Training indices are purged and
    embargoed around every test fold in the path.
    """
    if n_test_folds < 1 or n_test_folds >= n_splits:
        raise ValueError("n_test_folds must be in [1, n_splits - 1]")

    folds = list(purged_kfold_indices(n_samples, n_splits, 0, 0))
    fold_tests = [test for _, test in folds]

    for test_combo in combinations(range(n_splits), n_test_folds):
        test_idx = np.concatenate([fold_tests[i] for i in test_combo])
        test_idx.sort()

        train_mask = np.ones(n_samples, dtype=bool)
        for idx in test_combo:
            test = fold_tests[idx]
            start, stop = test[0], test[-1] + 1
            purge_start = max(0, start - purge)
            embargo_stop = min(n_samples, stop + embargo)
            train_mask[purge_start:embargo_stop] = False
        train_idx = np.arange(n_samples)[train_mask]
        yield train_idx, test_idx


def summarize_cpcv_scores(scores: Iterable[float]) -> CPCVScoreSummary:
    """Summarize CPCV path scores into robust percentiles."""
    values = np.array(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("scores must be non-empty")
    return CPCVScoreSummary(
        mean=float(values.mean()),
        median=float(np.median(values)),
        p10=float(np.percentile(values, 10)),
        p90=float(np.percentile(values, 90)),
    )
