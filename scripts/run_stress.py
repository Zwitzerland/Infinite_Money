"""Run simple stress scenarios."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from eval.metrics import PerformanceMetrics

ROOT = Path(__file__).resolve().parents[1]
STRESS_DIR = ROOT / "research" / "experiments"


def generate_returns(seed: int, n: int = 10) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.001, scale=0.01, size=n).tolist()


def stress_scenarios(seed: int) -> Dict[str, Dict[str, float]]:
    shocks = {"drawdown": -0.05, "vol_spike": 0.02}
    results: Dict[str, Dict[str, float]] = {}
    base_returns = generate_returns(seed)
    for name, shock in shocks.items():
        stressed = [r + shock for r in base_returns]
        metrics = PerformanceMetrics(returns=stressed)
        results[name] = {"sharpe_proxy": metrics.sharpe_proxy(), "max_drawdown": metrics.max_drawdown()}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stress tests")
    parser.add_argument("--baseline", action="store_true", help="Stress the baseline strategy")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()
    label = "baseline" if args.baseline else "candidate"
    results = stress_scenarios(args.seed)
    report_dir = STRESS_DIR / f"stress-{label}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "stress.json"
    path.write_text(json.dumps({"label": label, "results": results}, indent=2))
    print(f"Saved stress report to {path}")


if __name__ == "__main__":
    main()
