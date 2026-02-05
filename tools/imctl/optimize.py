"""LEAN optimization runner for imctl."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json
import yaml
import optuna
import pandas as pd

from optimizer.objective import ObjectiveConfig, evaluate_objective
from optimizer.runner_lean import LeanRunConfig, load_result, run_backtest

from .ledger import create_run, record_latest, write_run_config


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _sample_params(trial: optuna.Trial, space: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        ptype = spec.get("type", "int")
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
            params[name] = trial.suggest_categorical(name, spec.get("choices", []))
    return params


def _sampler(name: str) -> optuna.samplers.BaseSampler:
    if name.lower() == "cmaes":
        return optuna.samplers.CmaEsSampler()
    return optuna.samplers.TPESampler()


def run_imctl_optimize(
    project: str,
    search_space_path: Path,
    constraints_path: Path | None,
    study_name: str,
    n_trials: int,
    sampler_name: str,
    artifacts_root: Path,
) -> Path:
    search_space = _load_yaml(search_space_path)
    params_space = search_space.get("parameters", {})
    objective_cfg = search_space.get("objective", {})
    constraints = search_space.get("constraints", [])
    if constraints_path:
        constraints = _load_yaml(constraints_path).get("constraints", constraints)

    config = ObjectiveConfig(
        target=str(objective_cfg.get("target", "Sharpe Ratio")),
        direction=str(objective_cfg.get("direction", "max")),
        constraints=tuple(constraints),
    )

    run = create_run(artifacts_root)
    write_run_config(
        run,
        {
            "command": "optimize",
            "project": project,
            "search_space": str(search_space_path),
            "constraints": constraints,
            "study": study_name,
            "n_trials": n_trials,
            "sampler": sampler_name,
        },
    )
    (run.root / "search_space.yaml").write_text(yaml.safe_dump(search_space))

    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, params_space)
        trial_dir = run.root / f"trial_{trial.number:04d}"
        config_run = LeanRunConfig(
            project=project,
            output_dir=trial_dir,
            backtest_name=f"{study_name}-{trial.number}",
            parameters=params,
        )
        result = run_backtest(config_run)
        if result.return_code != 0:
            raise RuntimeError(f"LEAN backtest failed: {result.stderr_path}")

        raw = load_result(trial_dir)
        evaluation = evaluate_objective(raw, config)
        record = {
            "trial": trial.number,
            "objective": evaluation.objective,
            "constraints_passed": evaluation.constraints_passed,
            "params": params,
            "metrics": evaluation.metrics,
        }
        (trial_dir / "trial_summary.json").write_text(json.dumps(record, indent=2))
        trial_records.append(record)
        if not evaluation.constraints_passed:
            return float("-inf") if config.direction.startswith("max") else float("inf")
        return evaluation.objective

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize" if config.direction.startswith("max") else "minimize",
        sampler=_sampler(sampler_name),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_params = best.params
    (run.root / "params_best.yaml").write_text(yaml.safe_dump(best_params))

    df = pd.DataFrame(trial_records)
    df.to_parquet(run.root / "trials.parquet", index=False)

    metrics = {
        "best_value": best.value,
        "best_params": best_params,
        "objective": config.target,
        "direction": config.direction,
    }
    (run.root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report_lines = [
        "# Optimization Report",
        "",
        f"Run ID: `{run.run_id}`",
        f"Study: `{study_name}`",
        "",
        "## Best",
        f"- Objective: {metrics['objective']}",
        f"- Best value: {metrics['best_value']}",
        f"- Best params: {metrics['best_params']}",
        "",
        "## Trials",
    ]
    for record in trial_records[:5]:
        report_lines.append(
            f"- trial={record['trial']} objective={record['objective']} "
            f"constraints={record['constraints_passed']} params={record['params']}"
        )
    (run.root / "report.md").write_text("\n".join(report_lines) + "\n")
    (run.root / "summary.json").write_text(json.dumps(metrics, indent=2))
    record_latest(run, artifacts_root)
    return run.root
