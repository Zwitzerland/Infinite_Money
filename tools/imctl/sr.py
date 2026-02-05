"""Support/Resistance (SR) research workflows for imctl."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json

import optuna
import pandas as pd
import yaml
from omegaconf import OmegaConf

from hedge_fund.ai.config import load_config
from hedge_fund.ai.data import fetch_market_data
from hedge_fund.ai.evaluation.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from hedge_fund.ai.features import build_feature_frame
from hedge_fund.alpha.sr import SRBarrierParams, compute_sr_barrier_result
from optimizer.validation.cpcv import cpcv_indices, summarize_cpcv_scores

from .ledger import LedgerRun, create_run, record_latest, write_run_config


@dataclass(frozen=True)
class SRBacktestMetrics:
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    turnover_daily: float
    turnover_annual: float
    nonzero_days: int
    total_days: int


def _drop_symbol_index(features: pd.DataFrame) -> pd.DataFrame:
    if isinstance(features.index, pd.MultiIndex):
        symbols = features.index.get_level_values(0).unique()
        if len(symbols) != 1:
            raise ValueError("sr workflows require exactly 1 symbol")
        return features.droplevel(0)
    return features


def _strategy_returns(
    close: pd.Series,
    exposure: pd.Series,
    cost_per_turnover: float,
) -> pd.Series:
    forward_returns = close.pct_change().shift(-1)
    forward_returns = forward_returns.dropna()
    position = exposure.reindex(forward_returns.index).astype(float)
    pnl = position * forward_returns
    turnover = position.diff().abs().fillna(position.abs())
    return pnl - float(cost_per_turnover) * turnover


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _cagr(equity: pd.Series, periods: int = 252) -> float:
    if len(equity) == 0:
        return 0.0
    return float(equity.iloc[-1] ** (periods / len(equity)) - 1.0)


def _apply_window(index: pd.Index, start: str | None, end: str | None) -> pd.Index:
    if start is None and end is None:
        return index
    start_ts = pd.to_datetime(start, utc=True) if start else index.min()
    end_ts = pd.to_datetime(end, utc=True) if end else index.max()
    return index[(index >= start_ts) & (index <= end_ts)]


def run_sr_report(
    *,
    config_path: Path,
    artifacts_root: Path,
    cost_per_turnover: float,
    n_splits: int,
    n_test_folds: int,
    purge: int,
    embargo: int,
    start: str | None,
    end: str | None,
) -> Path:
    """Run SR analysis and write an artifacts report."""

    run: LedgerRun = create_run(artifacts_root)
    write_run_config(
        run,
        {
            "command": "sr report",
            "config": str(config_path),
            "cost_per_turnover": cost_per_turnover,
            "window": {"start": start, "end": end},
            "cpcv": {
                "n_splits": n_splits,
                "n_test_folds": n_test_folds,
                "purge": purge,
                "embargo": embargo,
            },
        },
    )

    cfg = load_config(config_path)
    market_cfg: Mapping[str, Any] = cfg.get("market_data", {})
    bars = fetch_market_data(market_cfg)
    features = build_feature_frame(bars, cfg.get("features", {}))

    sr_cfg: Mapping[str, Any] = cfg.get("signal_export", {}).get("sr_barrier", {})
    params = SRBarrierParams(
        pivot_lookback=int(sr_cfg.get("pivot_lookback", 5)),
        train_window=int(sr_cfg.get("train_window", 252)),
        horizon=int(sr_cfg.get("horizon", 10)),
        zone_atr=float(sr_cfg.get("zone_atr", 0.6)),
        tp_atr=float(sr_cfg.get("tp_atr", 1.0)),
        sl_atr=float(sr_cfg.get("sl_atr", 1.0)),
        cost_atr=float(sr_cfg.get("cost_atr", 0.05)),
        level_source=str(sr_cfg.get("level_source", "pivots")),
        round_atr_mult=float(sr_cfg.get("round_atr_mult", 4.0)),
        min_resolved_events=int(sr_cfg.get("min_resolved_events", 25)),
        confidence=float(sr_cfg.get("confidence", 0.95)),
        kelly_fraction=float(sr_cfg.get("kelly_fraction", 0.25)),
        max_exposure=float(sr_cfg.get("max_exposure", 1.0)),
        use_regime_filter=bool(sr_cfg.get("use_regime_filter", True)),
    )

    result = compute_sr_barrier_result(features, params)
    frame = _drop_symbol_index(features).sort_index()
    close = pd.Series(frame["close"].to_numpy(), index=frame.index, dtype=float)

    returns_all = _strategy_returns(close, result.exposure, cost_per_turnover)
    window_index = _apply_window(returns_all.index, start, end)
    returns = returns_all.reindex(window_index)
    equity = _equity_curve(returns)

    bh_returns = close.pct_change().shift(-1).reindex(window_index)
    bh_equity = _equity_curve(bh_returns)

    metrics = SRBacktestMetrics(
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        cagr=_cagr(equity),
        max_drawdown=max_drawdown(equity),
        turnover_daily=float(result.exposure.diff().abs().mean()),
        turnover_annual=float(result.exposure.diff().abs().mean() * 252.0),
        nonzero_days=int((result.exposure.abs() > 1e-12).sum()),
        total_days=int(len(result.exposure)),
    )
    bh_metrics = SRBacktestMetrics(
        sharpe=sharpe_ratio(bh_returns.fillna(0.0)),
        sortino=sortino_ratio(bh_returns.fillna(0.0)),
        cagr=_cagr(bh_equity),
        max_drawdown=max_drawdown(bh_equity),
        turnover_daily=0.0,
        turnover_annual=0.0,
        nonzero_days=int(len(bh_returns.dropna())),
        total_days=int(len(bh_returns.dropna())),
    )

    cpcv_scores: list[float] = []
    arr = returns.to_numpy(dtype=float)
    for _, test_idx in cpcv_indices(
        n_samples=len(arr),
        n_splits=n_splits,
        n_test_folds=n_test_folds,
        purge=purge,
        embargo=embargo,
    ):
        score = sharpe_ratio(arr[test_idx])
        cpcv_scores.append(float(score))
    cpcv_summary = summarize_cpcv_scores(cpcv_scores)

    n_support_entries = int(result.entry_support.sum())
    n_resistance_entries = int(result.entry_resistance.sum())
    n_support_resolved = float(result.resolved_entries_support.sum())
    n_support_wins = float(result.resolved_wins_support.sum())
    n_resistance_resolved = float(result.resolved_entries_resistance.sum())
    n_resistance_wins = float(result.resolved_wins_resistance.sum())

    report: dict[str, Any] = {
        "data": {
            "start": str(result.exposure.index.min()),
            "end": str(result.exposure.index.max()),
            "rows": int(len(result.exposure)),
            "eval_start": str(window_index.min()) if len(window_index) else None,
            "eval_end": str(window_index.max()) if len(window_index) else None,
            "eval_rows": int(len(window_index)),
        },
        "sr": {
            "params": {
                "pivot_lookback": params.pivot_lookback,
                "train_window": params.train_window,
                "horizon": params.horizon,
                "zone_atr": params.zone_atr,
                "tp_atr": params.tp_atr,
                "sl_atr": params.sl_atr,
                "cost_atr": params.cost_atr,
                "level_source": params.level_source,
                "round_atr_mult": params.round_atr_mult,
                "min_resolved_events": params.min_resolved_events,
                "confidence": params.confidence,
                "kelly_fraction": params.kelly_fraction,
                "max_exposure": params.max_exposure,
                "use_regime_filter": params.use_regime_filter,
            },
            "thresholds": {
                "p0_martingale": params.p0_martingale,
                "p_break_even": params.p_break_even,
                "reward_risk": params.reward_risk,
            },
            "events": {
                "support_entries": n_support_entries,
                "support_resolved": n_support_resolved,
                "support_wins": n_support_wins,
                "support_win_rate": float(n_support_wins / n_support_resolved)
                if n_support_resolved > 0
                else float("nan"),
                "resistance_entries": n_resistance_entries,
                "resistance_resolved": n_resistance_resolved,
                "resistance_wins": n_resistance_wins,
                "resistance_win_rate": float(n_resistance_wins / n_resistance_resolved)
                if n_resistance_resolved > 0
                else float("nan"),
            },
        },
        "backtest": {
            "cost_per_turnover": cost_per_turnover,
            "strategy": metrics.__dict__,
            "buy_hold": bh_metrics.__dict__,
        },
        "cpcv": {
            "n_splits": n_splits,
            "n_test_folds": n_test_folds,
            "purge": purge,
            "embargo": embargo,
            "scores": cpcv_scores,
            "summary": cpcv_summary.__dict__,
        },
    }

    (run.root / "sr_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(
        {
            "return": returns,
            "equity": equity,
            "exposure": result.exposure.reindex(window_index),
        }
    ).to_csv(run.root / "sr_equity.csv", index_label="timestamp")

    record_latest(run, artifacts_root)
    return run.root


def _sampler(name: str) -> optuna.samplers.BaseSampler:
    if name.lower() == "cmaes":
        return optuna.samplers.CmaEsSampler()
    return optuna.samplers.TPESampler()


def _sample_params(trial: optuna.Trial, space: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        ptype = str(spec.get("type", "int")).lower()
        if ptype == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["min"]),
                int(spec["max"]),
                step=int(spec.get("step", 1)),
            )
        elif ptype == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["min"]),
                float(spec["max"]),
                step=float(spec.get("step", 0.1)),
            )
        else:
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"categorical param '{name}' requires non-empty choices")
            params[name] = trial.suggest_categorical(name, choices)
    return params


def _objective_value(metrics: Mapping[str, float], target: str) -> float:
    if target not in metrics:
        raise ValueError(f"Objective '{target}' not found. Available: {sorted(metrics)}")
    return float(metrics[target])


def _parse_constraint(expression: str) -> tuple[str, str, float]:
    operators = [">=", "<=", "==", ">", "<"]
    for op in operators:
        if op in expression:
            lhs, rhs = expression.split(op, maxsplit=1)
            return lhs.strip(), op, float(rhs.strip())
    raise ValueError(f"Invalid constraint expression: {expression}")


def _compare(actual: float, op: str, target: float) -> bool:
    if op == ">=":
        return actual >= target
    if op == "<=":
        return actual <= target
    if op == ">":
        return actual > target
    if op == "<":
        return actual < target
    if op == "==":
        return actual == target
    raise ValueError(f"Unsupported operator: {op}")


def run_sr_sweep(
    *,
    config_path: Path,
    search_space_path: Path,
    artifacts_root: Path,
    n_trials: int,
    sampler_name: str,
    cost_per_turnover: float,
    n_splits: int,
    n_test_folds: int,
    purge: int,
    embargo: int,
    start: str | None,
    end: str | None,
) -> Path:
    """Run an Optuna sweep over SR barrier params using CPCV median Sharpe."""

    run: LedgerRun = create_run(artifacts_root)
    write_run_config(
        run,
        {
            "command": "sr sweep",
            "config": str(config_path),
            "search_space": str(search_space_path),
            "n_trials": n_trials,
            "sampler": sampler_name,
            "cost_per_turnover": cost_per_turnover,
            "window": {"start": start, "end": end},
            "cpcv": {
                "n_splits": n_splits,
                "n_test_folds": n_test_folds,
                "purge": purge,
                "embargo": embargo,
            },
        },
    )

    cfg = load_config(config_path)
    market_cfg: Mapping[str, Any] = cfg.get("market_data", {})
    bars = fetch_market_data(market_cfg)
    features = build_feature_frame(bars, cfg.get("features", {}))
    frame = _drop_symbol_index(features).sort_index()
    close = pd.Series(frame["close"].to_numpy(), index=frame.index, dtype=float)

    eval_index = _apply_window(close.index, start, end)

    space = yaml.safe_load(search_space_path.read_text()) or {}
    (run.root / "search_space.yaml").write_text(yaml.safe_dump(space))
    params_space = space.get("parameters", {})
    objective_cfg: Mapping[str, Any] = space.get("objective", {})
    target = str(objective_cfg.get("target", "cpcv_median_sharpe"))
    direction = str(objective_cfg.get("direction", "max")).lower()
    maximize = direction.startswith("max")
    constraints = space.get("constraints", [])

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        overrides = _sample_params(trial, params_space)

        base: Mapping[str, Any] = cfg.get("signal_export", {}).get("sr_barrier", {})
        merged = {**dict(base), **overrides}
        params = SRBarrierParams(
            pivot_lookback=int(merged.get("pivot_lookback", 5)),
            train_window=int(merged.get("train_window", 252)),
            horizon=int(merged.get("horizon", 10)),
            zone_atr=float(merged.get("zone_atr", 0.6)),
            tp_atr=float(merged.get("tp_atr", 1.0)),
            sl_atr=float(merged.get("sl_atr", 1.0)),
            cost_atr=float(merged.get("cost_atr", 0.05)),
            level_source=str(merged.get("level_source", "pivots")),
            round_atr_mult=float(merged.get("round_atr_mult", 4.0)),
            min_resolved_events=int(merged.get("min_resolved_events", 25)),
            confidence=float(merged.get("confidence", 0.95)),
            kelly_fraction=float(merged.get("kelly_fraction", 0.25)),
            max_exposure=float(merged.get("max_exposure", 1.0)),
            use_regime_filter=bool(merged.get("use_regime_filter", True)),
        )

        result = compute_sr_barrier_result(features, params)
        returns_all = _strategy_returns(close, result.exposure, cost_per_turnover)
        returns = returns_all.reindex(eval_index)
        arr = returns.to_numpy(dtype=float)
        scores: list[float] = []
        for _, test_idx in cpcv_indices(
            n_samples=len(arr),
            n_splits=n_splits,
            n_test_folds=n_test_folds,
            purge=purge,
            embargo=embargo,
        ):
            scores.append(float(sharpe_ratio(arr[test_idx])))
        cpcv_summary = summarize_cpcv_scores(scores)
        overall_sharpe = sharpe_ratio(arr)

        ex_eval = result.exposure.reindex(eval_index)

        metrics = {
            "cpcv_mean_sharpe": float(cpcv_summary.mean),
            "cpcv_median_sharpe": float(cpcv_summary.median),
            "cpcv_p10_sharpe": float(cpcv_summary.p10),
            "cpcv_p90_sharpe": float(cpcv_summary.p90),
            "sharpe": float(overall_sharpe),
            "deflated_sharpe": float(deflated_sharpe_ratio(overall_sharpe, len(arr), n_trials)),
            "turnover_daily": float(ex_eval.diff().abs().mean()),
            "nonzero_days": int((ex_eval.abs() > 1e-12).sum()),
        }

        constraint_results: list[dict[str, Any]] = []
        constraints_passed = True
        if isinstance(constraints, list):
            for expr in constraints:
                metric, op, threshold = _parse_constraint(str(expr))
                actual = metrics.get(metric)
                passed = actual is not None and _compare(float(actual), op, float(threshold))
                constraint_results.append(
                    {
                        "expression": str(expr),
                        "passed": bool(passed),
                        "actual": actual,
                    }
                )
                if not passed:
                    constraints_passed = False

        record = {
            "trial": trial.number,
            "objective": float(_objective_value(metrics, target)),
            "constraints_passed": constraints_passed,
            "constraint_results": constraint_results,
            "params": overrides,
            "metrics": metrics,
        }
        trial_rows.append(record)
        (run.root / f"trial_{trial.number:04d}.json").write_text(json.dumps(record, indent=2))

        value = float(_objective_value(metrics, target))
        if not constraints_passed:
            return float("-inf") if maximize else float("inf")
        return value

    study = optuna.create_study(
        study_name=f"sr-sweep-{run.run_id}",
        direction="maximize" if maximize else "minimize",
        sampler=_sampler(sampler_name),
    )
    study.optimize(objective, n_trials=n_trials)

    df = pd.DataFrame(trial_rows)
    df.to_parquet(run.root / "trials.parquet", index=False)
    best = study.best_trial
    best_params = best.params
    (run.root / "best_params.yaml").write_text(yaml.safe_dump(best_params))

    tuned = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if isinstance(tuned, dict):
        export_cfg = tuned.setdefault("signal_export", {})
        if isinstance(export_cfg, dict):
            sr_cfg = export_cfg.setdefault("sr_barrier", {})
            if isinstance(sr_cfg, dict):
                sr_cfg.update(best_params)
        (run.root / "tuned_config.yaml").write_text(yaml.safe_dump(tuned))

    report = {
        "target": target,
        "direction": direction,
        "n_trials": n_trials,
        "best_value": best.value,
        "best_params": best_params,
    }
    (run.root / "summary.json").write_text(json.dumps(report, indent=2))
    record_latest(run, artifacts_root)
    return run.root
