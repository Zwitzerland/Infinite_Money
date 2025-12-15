"""Score experiment outputs against evaluation gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from eval.metrics import evaluate_gates, load_metrics

ROOT = Path(__file__).resolve().parents[1]
GATES_PATH = ROOT / "eval" / "gates.json"
DEFAULT_GATES = {"min_sharpe_proxy": 0.5, "max_drawdown": 0.15}


def load_gates() -> Dict[str, float]:
    if GATES_PATH.exists():
        return json.loads(GATES_PATH.read_text())
    return DEFAULT_GATES


def main() -> None:
    parser = argparse.ArgumentParser(description="Score experiment results")
    parser.add_argument("metrics", type=Path, help="Path to metrics.json")
    args = parser.parse_args()
    metrics = load_metrics(args.metrics)
    gates = load_gates()
    scores = evaluate_gates(metrics, gates)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
