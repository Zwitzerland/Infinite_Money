"""Report builder for experiment runs."""
from __future__ import annotations

from pathlib import Path

import json
import yaml


def build_report(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "params_best.yaml"
    report_path = run_dir / "report.md"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    params = yaml.safe_load(params_path.read_text()) if params_path.exists() else {}

    lines = [
        "# Run Report",
        "",
        f"Run ID: `{run_dir.name}`",
        "",
        "## Metrics",
        json.dumps(metrics, indent=2),
        "",
        "## Parameters",
        json.dumps(params, indent=2),
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return report_path
