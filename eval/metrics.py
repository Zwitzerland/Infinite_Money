"""Metrics and manifest helpers for experiment artifacts."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import json
import statistics


@dataclass
class PerformanceMetrics:
    """Lightweight performance container for smoke tests."""

    returns: List[float]

    def sharpe_proxy(self) -> float:
        if not self.returns:
            return 0.0
        mean_ret = statistics.fmean(self.returns)
        std_ret = statistics.pstdev(self.returns) or 1e-9
        return mean_ret / std_ret

    def max_drawdown(self) -> float:
        peak = 0.0
        trough = 0.0
        cumulative = 0.0
        for value in self.returns:
            cumulative += value
            peak = max(peak, cumulative)
            trough = min(trough, cumulative)
        drawdown = peak - trough
        return drawdown if peak else 0.0


def save_metrics(metrics: PerformanceMetrics, output_path: Path) -> None:
    """Persist metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2))


def load_metrics(path: Path) -> PerformanceMetrics:
    """Load metrics from disk."""
    data = json.loads(path.read_text())
    return PerformanceMetrics(**data)


def evaluate_gates(metrics: PerformanceMetrics, gates: Dict[str, float]) -> Dict[str, bool]:
    """Evaluate metrics against thresholds.

    Parameters
    ----------
    metrics: PerformanceMetrics
        Metrics instance to evaluate.
    gates: Dict[str, float]
        Thresholds keyed by name.

    Returns
    -------
    Dict[str, bool]
        Pass/fail flags keyed by gate.
    """
    return {
        "sharpe_proxy": metrics.sharpe_proxy() >= gates.get("min_sharpe_proxy", 0.0),
        "max_drawdown": metrics.max_drawdown() <= gates.get("max_drawdown", float("inf")),
    }
