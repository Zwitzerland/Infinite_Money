"""Constraint compiler that validates declarative policies before execution.

This stub enforces structural checks and prepares diagnostics for downstream components.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class ConstraintDiagnostics:
    """Diagnostics emitted during constraint compilation."""

    errors: List[str]
    warnings: List[str]


def load_policy(policy_path: Path) -> Dict[str, float]:
    """Load the YAML constraint policy.

    Parameters
    ----------
    policy_path: Path
        Location of the YAML policy file.

    Returns
    -------
    Dict[str, float]
        Parsed constraint values keyed by name.
    """
    policy_data = yaml.safe_load(policy_path.read_text())
    return policy_data.get("constraints", {})


def compile_constraints(policy: Dict[str, float]) -> ConstraintDiagnostics:
    """Compile constraints into diagnostics.

    Parameters
    ----------
    policy: Dict[str, float]
        Constraint values keyed by policy name.

    Returns
    -------
    ConstraintDiagnostics
        Diagnostics summarizing constraint coverage and potential issues.
    """
    errors: List[str] = []
    warnings: List[str] = []
    required = [
        "max_gross_leverage",
        "max_net_exposure",
        "max_single_name_exposure",
        "max_sector_exposure",
        "max_daily_turnover",
        "max_drawdown",
        "data_freshness_max_lag_minutes",
    ]
    missing = [key for key in required if key not in policy]
    if missing:
        errors.append(f"Missing constraints: {', '.join(missing)}")

    for key, value in policy.items():
        if value is None:
            errors.append(f"Constraint {key} is not set.")
        elif value <= 0:
            warnings.append(f"Constraint {key} is non-positive; verify assumptions.")

    return ConstraintDiagnostics(errors=errors, warnings=warnings)


def validate_policy(path: Path) -> ConstraintDiagnostics:
    """Load and validate a policy file."""
    policy = load_policy(path)
    return compile_constraints(policy)


if __name__ == "__main__":
    diagnostics = validate_policy(Path("constraints/policy.yml"))
    if diagnostics.errors:
        print("Errors:\n" + "\n".join(diagnostics.errors))
    if diagnostics.warnings:
        print("Warnings:\n" + "\n".join(diagnostics.warnings))
    if not diagnostics.errors and not diagnostics.warnings:
        print("Policy validated successfully.")
