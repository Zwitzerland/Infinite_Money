"""Optuna optimization for the earnings volatility strategy."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna

from hedge_fund.ai.data.earnings_options import load_earnings_option_snapshots
from hedge_fund.ai.strategies.earnings_volatility import (
    EarningsVolParams,
    build_earnings_vol_signal,
)


@dataclass(frozen=True)
class ExecutionParams:
    starting_equity: float
    risk_per_trade: float
    trade_cost: float
    short_straddle_scale: float
    calendar_scale: float
    pnl_cap: float
    pnl_floor: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize earnings volatility params")
    parser.add_argument("--data-path", default="data/options/earnings_snapshots.csv")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--min-trades", type=int, default=25)
    parser.add_argument("--max-drawdown", type=float, default=0.35)
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--risk-per-trade", type=float, default=0.02)
    parser.add_argument("--trade-cost", type=float, default=0.002)
    return parser.parse_args()


def _pnl_proxy(snapshot: Any, action: str, params: ExecutionParams) -> float:
    if action == "short_straddle":
        edge = snapshot.iv_front - snapshot.rv_20d
        move_scale = float(np.sqrt(snapshot.front_dte / 252.0))
        gross = edge * move_scale * params.short_straddle_scale
    elif action == "calendar_spread":
        term_slope = (snapshot.iv_back - snapshot.iv_front) / snapshot.iv_front
        move_scale = float(np.sqrt(snapshot.back_dte / 252.0))
        gross = term_slope * move_scale * params.calendar_scale
    else:
        return 0.0

    capped = max(min(gross, params.pnl_cap), -params.pnl_floor)
    return capped - params.trade_cost


def _series_metrics(values: list[float]) -> Mapping[str, float]:
    if not values:
        return {
            "final_equity": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    cagr = arr[-1] ** (252 / len(arr)) - 1
    peak = np.maximum.accumulate(arr)
    max_drawdown = float(np.max((peak - arr) / peak))
    calmar = float(cagr / max_drawdown) if max_drawdown > 0 else float(cagr * 10.0)
    return {
        "final_equity": float(arr[-1]),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def _run_backtest(
    snapshots: list[Any],
    strategy: EarningsVolParams,
    execution: ExecutionParams,
) -> tuple[list[float], int]:
    equity = []
    eq = execution.starting_equity
    trades = 0
    snapshots = sorted(snapshots, key=lambda item: item.asof_date)
    for snapshot in snapshots:
        decision = build_earnings_vol_signal(snapshot, strategy)
        if decision.action == "skip":
            continue
        trade_return = _pnl_proxy(snapshot, decision.action, execution)
        eq *= 1 + execution.risk_per_trade * trade_return
        equity.append(eq)
        trades += 1
    return equity, trades


def main() -> None:
    args = _parse_args()
    output_dir = (
        Path(args.output_root)
        / datetime.now(timezone.utc).strftime("earnings_vol_opt_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = load_earnings_option_snapshots(args.data_path)
    execution = ExecutionParams(
        starting_equity=1.0,
        risk_per_trade=args.risk_per_trade,
        trade_cost=args.trade_cost,
        short_straddle_scale=1.0,
        calendar_scale=0.6,
        pnl_cap=0.5,
        pnl_floor=0.5,
    )

    trial_summaries: list[Mapping[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = EarningsVolParams(
            min_days_to_earnings=trial.suggest_int("min_days_to_earnings", 3, 12),
            max_days_to_earnings=trial.suggest_int("max_days_to_earnings", 12, 25),
            min_iv_rv=trial.suggest_float("min_iv_rv", 1.05, 1.6, step=0.05),
            min_volume=trial.suggest_float("min_volume", 100, 600, step=50),
            min_open_interest=trial.suggest_float("min_open_interest", 200, 1200, step=100),
            max_bid_ask_spread=trial.suggest_float("max_bid_ask_spread", 0.15, 0.6, step=0.05),
            min_front_dte=trial.suggest_int("min_front_dte", 5, 12),
            max_front_dte=trial.suggest_int("max_front_dte", 18, 45, step=3),
            min_back_dte=trial.suggest_int("min_back_dte", 30, 60, step=5),
            min_term_inversion=trial.suggest_float("min_term_inversion", 0.02, 0.12, step=0.01),
            min_term_contango=trial.suggest_float("min_term_contango", 0.02, 0.12, step=0.01),
        )
        equity, trades = _run_backtest(snapshots, params, execution)
        metrics = _series_metrics(equity)
        metrics["trades"] = float(trades)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params", params.__dict__)
        trial_summaries.append({"params": params.__dict__, "metrics": metrics})

        if trades < args.min_trades:
            return float("-inf")
        if metrics["max_drawdown"] > args.max_drawdown:
            return float("-inf")
        return metrics["calmar"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    summary = {
        "data_path": args.data_path,
        "trials": args.trials,
        "min_trades": args.min_trades,
        "max_drawdown": args.max_drawdown,
        "best_params": study.best_params,
        "best_metrics": study.best_trial.user_attrs.get("metrics"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "trials.json").write_text(json.dumps(trial_summaries, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
