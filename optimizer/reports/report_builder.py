"""Build Markdown and JSON reports for optimization runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import subprocess


@dataclass(frozen=True)
class RunContext:
    """Metadata for a reproducible run."""

    run_id: str
    created_at: str
    git_hash: str
    output_dir: Path


def _git_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def create_run_context(root: Path, output_root: Path) -> RunContext:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"opt_{timestamp}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_hash=_git_hash(root),
        output_dir=output_dir,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def write_report(
    context: RunContext,
    config: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write JSON + Markdown report artifacts for a run."""
    payload = {
        "context": {
            **asdict(context),
            "output_dir": str(context.output_dir),
        },
        "config": config,
        "summary": summary,
        "trials": trials,
    }
    _write_json(context.output_dir / "summary.json", payload)

    lines = [
        "# Optimization Report",
        "",
        f"Run ID: `{context.run_id}`",
        f"Created: `{context.created_at}`",
        f"Git hash: `{context.git_hash}`",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Top Trials")
    for trial in trials[:5]:
        params = ", ".join(f"{k}={v}" for k, v in trial.get("params", {}).items())
        lines.append(
            f"- trial={trial.get('number')} objective={trial.get('objective')} "
            f"constraints={trial.get('constraints_passed')} params=({params})"
        )
    _write_markdown(context.output_dir / "report.md", lines)
