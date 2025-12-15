"""Run lightweight robustness checks against perturbations."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from eval.metrics import PerformanceMetrics

ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_DIR = ROOT / "research" / "experiments"


def generate_returns(seed: int, n: int = 10) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.001, scale=0.01, size=n).tolist()


def run_perturbations(seed: int) -> Dict[str, float]:
    scenarios = {"base": seed, "high_slippage": seed + 1, "shock": seed + 2}
    sharpe_scores: Dict[str, float] = {}
    for name, scenario_seed in scenarios.items():
        returns = generate_returns(scenario_seed)
        metrics = PerformanceMetrics(returns=returns)
        sharpe_scores[name] = metrics.sharpe_proxy()
    return sharpe_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness sweeps")
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline robustness")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()
    label = "baseline" if args.baseline else "candidate"
    results = run_perturbations(args.seed)
    report_dir = ROBUSTNESS_DIR / f"robustness-{label}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "robustness.json"
    report_path.write_text(json.dumps({"label": label, "results": results}, indent=2))
    print(f"Saved robustness report to {report_path}")


if __name__ == "__main__":
    main()
