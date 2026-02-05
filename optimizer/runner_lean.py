"""LEAN CLI runner helpers for backtests and optimizations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
import sys

import json
import shutil
import subprocess


@dataclass(frozen=True)
class ParameterRange:
    """Parameter range definition for LEAN optimizations."""

    name: str
    min_value: float
    max_value: float
    step: float


@dataclass(frozen=True)
class LeanRunConfig:
    """Configuration for a LEAN run."""

    project: str
    output_dir: Path
    backtest_name: str | None = None
    parameters: Mapping[str, float | int | str] = field(default_factory=dict)
    parameter_ranges: Sequence[ParameterRange] = field(default_factory=tuple)
    target: str | None = None
    target_direction: str | None = None
    strategy: str | None = None
    constraints: Sequence[str] = field(default_factory=tuple)
    download_data: bool = False
    extra_args: Sequence[str] = field(default_factory=tuple)
    node: str | None = None
    parallel_nodes: int | None = None
    push: bool = False


@dataclass(frozen=True)
class LeanRunResult:
    """Result metadata for a LEAN run."""

    config: LeanRunConfig
    command: Sequence[str]
    output_dir: Path
    stdout_path: Path
    stderr_path: Path
    return_code: int


def _lean_executable() -> str:
    lean = shutil.which("lean")
    if lean:
        return lean
    candidate = Path(sys.executable).with_name("lean.exe" if sys.platform == "win32" else "lean")
    if candidate.exists():
        return str(candidate)
    raise RuntimeError(
        "lean CLI not found. Install via 'pip install lean' and run 'lean login'."
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_run_config(config: LeanRunConfig) -> None:
    payload = {
        "project": config.project,
        "backtest_name": config.backtest_name,
        "parameters": config.parameters,
        "parameter_ranges": [
            {
                "name": item.name,
                "min_value": item.min_value,
                "max_value": item.max_value,
                "step": item.step,
            }
            for item in config.parameter_ranges
        ],
        "target": config.target,
        "target_direction": config.target_direction,
        "strategy": config.strategy,
        "constraints": list(config.constraints),
        "download_data": config.download_data,
        "extra_args": list(config.extra_args),
        "node": config.node,
        "parallel_nodes": config.parallel_nodes,
        "push": config.push,
        "timestamp": _timestamp(),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def _run_command(config: LeanRunConfig, command: Sequence[str]) -> LeanRunResult:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "lean_stdout.log"
    stderr_path = output_dir / "lean_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    return LeanRunResult(
        config=config,
        command=command,
        output_dir=output_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        return_code=result.returncode,
    )


def _backtest_command(config: LeanRunConfig, lean_exe: str) -> list[str]:
    command = [lean_exe, "backtest", config.project, "--output", str(config.output_dir)]
    if config.backtest_name:
        command += ["--backtest-name", config.backtest_name]
    if config.download_data:
        command.append("--download-data")
    for name, value in config.parameters.items():
        command += ["--parameter", name, str(value)]
    command += list(config.extra_args)
    return command


def _optimize_command(config: LeanRunConfig, lean_exe: str) -> list[str]:
    if not config.strategy or not config.target or not config.target_direction:
        raise ValueError("strategy, target, and target_direction are required")
    command = [
        lean_exe,
        "optimize",
        config.project,
        "--output",
        str(config.output_dir),
        "--strategy",
        config.strategy,
        "--target",
        config.target,
        "--target-direction",
        config.target_direction,
    ]
    for param in config.parameter_ranges:
        command += [
            "--parameter",
            param.name,
            str(param.min_value),
            str(param.max_value),
            str(param.step),
        ]
    for constraint in config.constraints:
        command += ["--constraint", constraint]
    if config.download_data:
        command.append("--download-data")
    command += list(config.extra_args)
    return command


def _cloud_optimize_command(config: LeanRunConfig, lean_exe: str) -> list[str]:
    if not config.target or not config.target_direction:
        raise ValueError("target and target_direction are required")
    if not config.node or not config.parallel_nodes:
        raise ValueError("node and parallel_nodes are required")
    command = [
        lean_exe,
        "cloud",
        "optimize",
        config.project,
        "--target",
        config.target,
        "--target-direction",
        config.target_direction,
        "--node",
        config.node,
        "--parallel-nodes",
        str(config.parallel_nodes),
    ]
    for param in config.parameter_ranges:
        command += [
            "--parameter",
            param.name,
            str(param.min_value),
            str(param.max_value),
            str(param.step),
        ]
    for constraint in config.constraints:
        command += ["--constraint", constraint]
    if config.push:
        command.append("--push")
    command += list(config.extra_args)
    return command


def run_backtest(config: LeanRunConfig) -> LeanRunResult:
    """Run a LEAN backtest using the CLI."""
    lean_exe = _lean_executable()
    _write_run_config(config)
    command = _backtest_command(config, lean_exe)
    return _run_command(config, command)


def run_optimize(config: LeanRunConfig) -> LeanRunResult:
    """Run a local LEAN optimization using the CLI."""
    lean_exe = _lean_executable()
    _write_run_config(config)
    command = _optimize_command(config, lean_exe)
    return _run_command(config, command)


def run_cloud_optimize(config: LeanRunConfig) -> LeanRunResult:
    """Run a cloud LEAN optimization using the CLI."""
    lean_exe = _lean_executable()
    _write_run_config(config)
    command = _cloud_optimize_command(config, lean_exe)
    return _run_command(config, command)


def find_result_json(output_dir: Path) -> Path:
    """Find the most likely LEAN result JSON file in an output directory."""
    candidates = [
        output_dir / "backtest.json",
        output_dir / "results.json",
        output_dir / "result.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for path in output_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue

        # LEAN outputs differ by engine/config. Accept both legacy PascalCase
        # and the newer camelCase keys.
        if (
            "Statistics" in payload
            or "TotalPerformance" in payload
            or "statistics" in payload
            or "totalPerformance" in payload
        ):
            return path
    raise FileNotFoundError(f"No LEAN result JSON found in {output_dir}")


def load_result(output_dir: Path) -> Mapping[str, object]:
    path = find_result_json(output_dir)
    return json.loads(path.read_text(encoding="utf-8"))
