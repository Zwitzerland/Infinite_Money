"""Environment checks and repo docs generator."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import subprocess
import tomllib

from .repo_scan import render_dependency_graph, render_repo_map


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "missing"


def _dependencies(pyproject_path: Path) -> Iterable[str]:
    data = tomllib.loads(pyproject_path.read_text())
    deps = data.get("project", {}).get("dependencies", [])
    return deps


def run_doctor(root: Path) -> dict[str, str]:
    status = {
        "python": _command_version(["python", "--version"]),
        "lean": _command_version(["lean", "--version"]),
        "docker": _command_version(["docker", "--version"]),
    }

    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    repo_map = render_repo_map(root)
    (docs_dir / "REPO_MAP.md").write_text(repo_map)

    deps = _dependencies(root / "pyproject.toml")
    dep_graph = render_dependency_graph(root, deps)
    (docs_dir / "DEPENDENCY_GRAPH.md").write_text(dep_graph)
    return status
