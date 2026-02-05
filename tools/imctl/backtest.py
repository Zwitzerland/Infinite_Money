"""LEAN backtest runner for imctl."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json
import yaml

from optimizer.runner_lean import LeanRunConfig, load_result, run_backtest

from .ledger import create_run, record_latest, write_run_config
from .charts import render_lean_equity_chart


def _load_params(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    return payload


def run_imctl_backtest(project: str, params_path: Path, artifacts_root: Path) -> Path:
    params = _load_params(params_path)
    run = create_run(artifacts_root)
    write_run_config(
        run,
        {
            "command": "backtest",
            "project": project,
            "params_path": str(params_path),
        },
    )

    config = LeanRunConfig(
        project=project,
        output_dir=run.root,
        backtest_name=f"imctl-{run.run_id}",
        parameters=params,
    )
    result = run_backtest(config)
    if result.return_code != 0:
        raise RuntimeError(f"LEAN backtest failed. See {result.stderr_path}")

    output = load_result(run.root)
    metrics = {
        "Statistics": output.get("Statistics") or output.get("statistics") or {},
        "TotalPerformance": output.get("TotalPerformance")
        or output.get("totalPerformance")
        or {},
    }
    (run.root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run.root / "params_best.yaml").write_text(yaml.safe_dump(params))

    try:
        chart_path = render_lean_equity_chart(run.root)
        (run.root / "equity_chart.txt").write_text(str(chart_path))
    except Exception as exc:
        (run.root / "equity_chart_error.txt").write_text(str(exc))

    record_latest(run, artifacts_root)
    return run.root
