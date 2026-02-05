"""Training workflow for AI stack models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import numpy as np
import pandas as pd

from optimizer.validation import purged_kfold_indices

from ..evaluation import (
    directional_accuracy,
    information_coefficient,
    max_drawdown,
    mean_absolute_error,
    mean_squared_error,
    sharpe_ratio,
    simulate_strategy,
)
from ..models import GBDTForecaster, LinearForecaster


@dataclass
class ModelResult:
    name: str
    metrics: Mapping[str, float]


def _prepare_dataset(features: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = features.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])

    df = df.join(labels.rename("label"), how="inner")
    df = df.dropna()
    y = df.pop("label").to_numpy(dtype=float)
    x = df.to_numpy(dtype=float)
    return x, y, df


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    positions = np.sign(y_pred)
    equity = simulate_strategy(y_true, positions)
    returns = np.asarray(y_true)
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "ic": information_coefficient(y_true, y_pred),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
    }


def run_training(
    features: pd.DataFrame,
    labels: pd.Series,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Mapping[str, Any]:
    """Train enabled models and evaluate with purged CV."""
    x, y, aligned = _prepare_dataset(features, labels)
    split_cfg = config.get("split", {})
    n_splits = int(split_cfg.get("n_splits", 3))
    purge = int(split_cfg.get("purge", 5))
    embargo = int(split_cfg.get("embargo", 5))

    folds = list(purged_kfold_indices(len(x), n_splits, purge, embargo))
    model_cfg = config.get("models", {})
    ensemble_cfg = config.get("ensemble", {})
    weights = ensemble_cfg.get("weights", {})

    results: dict[str, list[Mapping[str, float]]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        fold_predictions: dict[str, np.ndarray] = {}

        if model_cfg.get("linear", {}).get("enabled", False):
            model = LinearForecaster()
            model.fit(x_train, y_train)
            preds = np.asarray(model.predict(x_test))
            fold_predictions["linear"] = preds
            results.setdefault("linear", []).append(_evaluate(y_test, preds))

        if model_cfg.get("gbdt", {}).get("enabled", False):
            model = GBDTForecaster()
            model.fit(x_train, y_train)
            preds = np.asarray(model.predict(x_test))
            fold_predictions["gbdt"] = preds
            results.setdefault("gbdt", []).append(_evaluate(y_test, preds))

        if fold_predictions and weights:
            total_weight = sum(float(weights.get(name, 0.0)) for name in fold_predictions)
            if total_weight > 0:
                ensemble_pred = np.zeros_like(next(iter(fold_predictions.values())))
                for name, preds in fold_predictions.items():
                    ensemble_pred += float(weights.get(name, 0.0)) * preds
                ensemble_pred = ensemble_pred / total_weight
                results.setdefault("ensemble", []).append(_evaluate(y_test, ensemble_pred))

    summary = {"folds": len(folds), "results": results}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
