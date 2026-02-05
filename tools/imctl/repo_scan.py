"""Repository scanning utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "outputs",
}


def list_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def render_repo_map(root: Path) -> str:
    files = list_files(root)
    sections = ["# Repository Map", "", f"Root: `{root}`", ""]
    sections.append(f"Total files: {len(files)}")
    sections.append("")
    sections.append("## Top-level directories")
    for path in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in EXCLUDED_DIRS]):
        count = len([p for p in path.rglob("*") if p.is_file() and not any(part in EXCLUDED_DIRS for part in p.parts)])
        sections.append(f"- `{path.name}/` ({count} files)")

    sections.append("")
    sections.append("## Notable items")
    sections.append("- `lean_projects/`: LEAN strategy projects")
    sections.append("- `optimizer/`: Optuna + LEAN optimization tooling")
    sections.append("- `hedge_fund/`: core AI/exec/backtest modules")
    sections.append("- `knowledge/`: PDF knowledge index")
    sections.append("- `tools/imctl/`: repo command center")
    sections.append("")
    sections.append("## Possible duplicates / templates")
    sections.append("- `lean_projects/QC_Template` (template)")
    return "\n".join(sections) + "\n"


def render_dependency_graph(root: Path, dependencies: Iterable[str]) -> str:
    sections = ["# Dependency Graph", "", "## Python dependencies"]
    for dep in sorted(dependencies):
        sections.append(f"- {dep}")
    sections.append("")
    sections.append("## Internal modules")
    modules = [
        "hedge_fund",
        "optimizer",
        "control_plane",
        "gates",
        "mcp_servers",
        "tools.imctl",
        "lean_projects",
    ]
    for module in modules:
        sections.append(f"- {module}")
    return "\n".join(sections) + "\n"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))
