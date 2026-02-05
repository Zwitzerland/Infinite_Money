"""Verification suite runner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import json
import subprocess
import sys
import time


@dataclass(frozen=True)
class CheckResult:
    name: str
    command: str
    return_code: int
    duration_seconds: float


def _run_command(name: str, command: list[str], cwd: Path) -> CheckResult:
    start = time.time()
    result = subprocess.run(command, cwd=cwd)
    return CheckResult(
        name=name,
        command=" ".join(command),
        return_code=result.returncode,
        duration_seconds=round(time.time() - start, 2),
    )


def run_checks(root: Path, output_path: Path) -> Mapping[str, object]:
    python = sys.executable
    checks = [
        ("lint", [python, "-m", "ruff", "check", "."]),
        ("unit", [python, "-m", "pytest"]),
        ("smoke_backtest", [python, "-m", "hedge_fund.backtest.runner"]),
    ]
    results = [_run_command(name, cmd, root) for name, cmd in checks]
    payload = {
        "checks": [result.__dict__ for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload
