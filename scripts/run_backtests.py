"""Run baseline or candidate backtests (smoke version)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from eval.metrics import PerformanceMetrics, save_metrics

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "research" / "experiments"


def generate_returns(seed: int, n: int = 10) -> List[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.001, scale=0.01, size=n).tolist()


def write_manifest(experiment_dir: Path, seed: int, label: str) -> None:
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "label": label,
        "code_commit": "untracked",
        "dataset_hash": "placeholder",
        "docker_image": "lean-local",
    }
    experiment_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2))


def run_backtest(label: str, seed: int) -> Path:
    experiment_dir = EXPERIMENTS_DIR / datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    returns = generate_returns(seed)
    metrics = PerformanceMetrics(returns=returns)
    save_metrics(metrics, experiment_dir / "metrics.json")
    write_manifest(experiment_dir, seed, label)
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests")
    parser.add_argument("--baseline", action="store_true", help="Run the baseline strategy")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()
    label = "baseline" if args.baseline else "candidate"
    path = run_backtest(label, args.seed)
    print(f"Saved {label} backtest to {path}")


if __name__ == "__main__":
    main()
