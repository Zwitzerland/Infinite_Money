"""Optuna optimization for the G2MAX-X synthetic backtest."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import optuna

from g2max_x_lab import run_simulation


def _series_metrics(series: np.ndarray) -> Mapping[str, float]:
    cagr = series[-1] ** (252 / len(series)) - 1
    peak = np.maximum.accumulate(series)
    max_drawdown = float(np.max((peak - series) / peak))
    returns = np.diff(series) / series[:-1]
    vol = float(np.std(returns) * np.sqrt(252.0)) if len(returns) else 0.0
    sharpe = float(cagr / vol) if vol > 0 else 0.0
    calmar = float(cagr / max_drawdown) if max_drawdown > 0 else float(cagr * 10.0)
    return {
        "final_equity": float(series[-1]),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "volatility": vol,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize G2MAX-X parameters")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--max-drawdown", type=float, default=0.18)
    parser.add_argument("--output-root", default="artifacts")
    return parser.parse_args()


def _suggest_params(trial: optuna.Trial) -> Mapping[str, Any]:
    return {
        "phi_base": trial.suggest_float("phi_base", 0.15, 0.65, step=0.05),
        "vol_target": trial.suggest_float("vol_target", 0.08, 0.24, step=0.02),
        "d1": trial.suggest_float("d1", 0.05, 0.20, step=0.01),
        "d2": trial.suggest_float("d2", 0.15, 0.35, step=0.01),
        "leverage": trial.suggest_float("leverage", 1.0, 3.0, step=0.25),
        "lookback": trial.suggest_int("lookback", 30, 120, step=10),
        "ewma_lambda": trial.suggest_float("ewma_lambda", 0.85, 0.99, step=0.01),
    }


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            seeds.append(int(value))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _aggregate(metrics_list: list[Mapping[str, float]]) -> Mapping[str, float]:
    keys = metrics_list[0].keys()
    avg = {
        key: float(np.mean([metrics[key] for metrics in metrics_list]))
        for key in keys
    }
    worst_drawdown = float(max(metrics["max_drawdown"] for metrics in metrics_list))
    avg["worst_drawdown"] = worst_drawdown
    return avg


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now(timezone.utc).strftime("g2max_opt_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)

    baseline_metrics_list = []
    for seed in seeds:
        baseline_eq, _ = run_simulation(seed=seed, years=args.years)
        baseline_metrics_list.append(_series_metrics(baseline_eq.to_numpy()))
    baseline_metrics = _aggregate(baseline_metrics_list)

    trial_summaries: list[Mapping[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        metrics_list = []
        for seed in seeds:
            eq, _ = run_simulation(seed=seed, years=args.years, **params)
            metrics_list.append(_series_metrics(eq.to_numpy()))
        aggregate = _aggregate(metrics_list)
        trial.set_user_attr("metrics", aggregate)
        trial.set_user_attr("params", params)
        trial.set_user_attr("per_seed", metrics_list)
        trial_summaries.append({"params": params, "metrics": aggregate})

        if aggregate["worst_drawdown"] > args.max_drawdown:
            return float("-inf")
        if aggregate["cagr"] < baseline_metrics["cagr"]:
            return float("-inf")
        return aggregate["calmar"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    best_params = study.best_params
    best_metrics_list = []
    best_series = []
    for seed in seeds:
        best_eq, _ = run_simulation(seed=seed, years=args.years, **best_params)
        best_metrics_list.append(_series_metrics(best_eq.to_numpy()))
        best_series.append(best_eq)
    best_metrics = _aggregate(best_metrics_list)

    summary = {
        "seeds": seeds,
        "years": args.years,
        "trials": args.trials,
        "constraint_max_drawdown": args.max_drawdown,
        "baseline": baseline_metrics,
        "best": best_metrics,
        "best_params": best_params,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    comparison = best_series[0].to_frame(name="best")
    comparison.to_csv(output_dir / "equity_comparison.csv", index_label="date")

    plt.figure(figsize=(10, 5))
    plt.plot(best_series[0].index, best_series[0], label="best (seed 1)")
    plt.yscale("log")
    plt.title("G2MAX-X Optimization (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "equity_comparison.png", dpi=150)

    print(output_dir)


if __name__ == "__main__":
    main()
