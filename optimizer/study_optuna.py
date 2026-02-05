"""Optuna-driven optimization that shells out to LEAN backtests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import optuna
from omegaconf import OmegaConf

from optimizer.objective import ObjectiveConfig, evaluate_objective
from optimizer.reports import create_run_context, write_report
from optimizer.runner_lean import LeanRunConfig, load_result, run_backtest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna optimization")
    parser.add_argument("--project", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--search-space",
        default="optimizer/search_space.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts",
        help="Root directory for run artifacts.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Use QuantConnect data downloader for LEAN backtests.",
    )
    return parser.parse_args()


def _load_search_space(path: Path) -> Mapping[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _suggest_params(trial: optuna.Trial, parameters: Mapping[str, Any]) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    for name, spec in parameters.items():
        param_type = spec.get("type", "int")
        if param_type == "int":
            suggestions[name] = trial.suggest_int(
                name,
                int(spec["min"]),
                int(spec["max"]),
                step=int(spec.get("step", 1)),
            )
        elif param_type == "float":
            suggestions[name] = trial.suggest_float(
                name,
                float(spec["min"]),
                float(spec["max"]),
                step=float(spec.get("step", 0.1)),
            )
        else:
            choices = spec.get("choices")
            if not choices:
                raise ValueError(f"choices required for parameter '{name}'")
            suggestions[name] = trial.suggest_categorical(name, choices)
    return suggestions


def _objective_config(space: Mapping[str, Any]) -> ObjectiveConfig:
    objective = space.get("objective", {})
    constraints = space.get("constraints", [])
    return ObjectiveConfig(
        target=str(objective.get("target", "Sharpe Ratio")),
        direction=str(objective.get("direction", "max")),
        constraints=tuple(constraints),
    )


def _study_direction(direction: str) -> str:
    return "maximize" if direction.lower().startswith("max") else "minimize"


def _extract_data_window(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for start_key, end_key in (
        ("StartDate", "EndDate"),
        ("start", "end"),
        ("Start", "End"),
    ):
        start = result.get(start_key)
        end = result.get(end_key)
        if start and end:
            return {"start": start, "end": end}

    stats = result.get("Statistics") or result.get("statistics")
    if isinstance(stats, Mapping):
        start = stats.get("Start Date") or stats.get("StartDate")
        end = stats.get("End Date") or stats.get("EndDate")
        if start and end:
            return {"start": start, "end": end}
    return None


def main() -> None:
    args = _parse_args()
    repo_root = Path.cwd()
    output_root = Path(args.output_root)
    space = _load_search_space(Path(args.search_space))
    objective_cfg = _objective_config(space)

    context = create_run_context(repo_root, output_root)
    (context.output_dir / "search_space.json").write_text(
        json.dumps(space, indent=2, sort_keys=True)
    )

    trial_summaries: list[Mapping[str, Any]] = []

    def run_trial(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, space.get("parameters", {}))
        trial_dir = context.output_dir / f"trial_{trial.number:04d}"
        config = LeanRunConfig(
            project=args.project,
            output_dir=trial_dir,
            backtest_name=f"optuna-trial-{trial.number}",
            parameters=params,
            download_data=args.download_data,
        )
        result = run_backtest(config)
        if result.return_code != 0:
            raise RuntimeError(f"LEAN backtest failed, see {result.stderr_path}")

        raw_result = load_result(trial_dir)
        data_window = _extract_data_window(raw_result)
        evaluation = evaluate_objective(raw_result, objective_cfg)
        trial.set_user_attr("metrics", evaluation.metrics)
        trial.set_user_attr("constraints_passed", evaluation.constraints_passed)

        trial_summary = {
            "number": trial.number,
            "objective": evaluation.objective,
            "params": params,
            "constraints_passed": evaluation.constraints_passed,
            "metrics": evaluation.metrics,
            "data_window": data_window,
        }
        (trial_dir / "trial_summary.json").write_text(
            json.dumps(trial_summary, indent=2, sort_keys=True)
        )
        trial_summaries.append(trial_summary)

        if not evaluation.constraints_passed:
            return float("-inf") if objective_cfg.direction.startswith("max") else float(
                "inf"
            )
        return evaluation.objective

    study = optuna.create_study(direction=_study_direction(objective_cfg.direction))
    study.optimize(run_trial, n_trials=args.trials)

    best = study.best_trial
    summary = {
        "trials": args.trials,
        "best_value": best.value,
        "best_params": best.params,
        "objective": objective_cfg.target,
        "direction": objective_cfg.direction,
    }

    for trial in trial_summaries:
        if trial.get("data_window"):
            summary["data_window"] = trial["data_window"]
            break

    reverse = objective_cfg.direction.startswith("max")
    trial_summaries.sort(
        key=lambda item: item.get("objective", float("-inf")),
        reverse=reverse,
    )
    write_report(context, space, trial_summaries, summary)
    print(f"Run artifacts: {context.output_dir}")


if __name__ == "__main__":
    main()
