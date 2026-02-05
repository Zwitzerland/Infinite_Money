"""Purged K-fold split utilities with embargo handling."""
from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np


def purged_kfold_indices(
    n_samples: int,
    n_splits: int,
    purge: int,
    embargo: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices with purge + embargo windows.

    Parameters
    ----------
    n_samples:
        Total number of samples.
    n_splits:
        Number of folds.
    purge:
        Number of samples to remove before the test fold.
    embargo:
        Number of samples to remove after the test fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_samples <= n_splits:
        raise ValueError("n_samples must exceed n_splits")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    indices = np.arange(n_samples)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        test = indices[start:stop]
        train_mask = np.ones(n_samples, dtype=bool)
        purge_start = max(0, start - purge)
        embargo_stop = min(n_samples, stop + embargo)
        train_mask[purge_start:embargo_stop] = False
        train = indices[train_mask]
        yield train, test
        current = stop


def purged_kfold_split(
    xs: Sequence[object],
    n_splits: int,
    purge: int,
    embargo: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Convenience wrapper for sequences."""
    return purged_kfold_indices(len(xs), n_splits, purge, embargo)
