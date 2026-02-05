"""Experiment ledger helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import json
import platform
import subprocess
import sys


@dataclass(frozen=True)
class LedgerRun:
    run_id: str
    root: Path


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _lean_version() -> str:
    try:
        result = subprocess.run(
            ["lean", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def create_run(root: Path) -> LedgerRun:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_root = root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    return LedgerRun(run_id=run_id, root=run_root)


def write_run_config(run: LedgerRun, extra: Mapping[str, Any] | None = None) -> None:
    payload = {
        "run_id": run.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python": sys.version,
        "platform": platform.platform(),
        "lean_version": _lean_version(),
    }
    if extra:
        payload.update(extra)
    (run.root / "run_config.json").write_text(json.dumps(payload, indent=2))


def record_latest(run: LedgerRun, artifacts_root: Path) -> None:
    (artifacts_root / "latest_run.txt").write_text(run.run_id)
