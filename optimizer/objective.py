"""Objective and constraint evaluation for backtest outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for objective evaluation."""

    target: str
    direction: str
    constraints: Sequence[str]


@dataclass(frozen=True)
class ConstraintResult:
    """Result of a single constraint check."""

    expression: str
    passed: bool
    actual: float | None


@dataclass(frozen=True)
class ObjectiveResult:
    """Objective evaluation result."""

    objective: float
    direction: str
    constraints_passed: bool
    constraint_results: Sequence[ConstraintResult]
    metrics: Mapping[str, float]


def _normalize_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _parse_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    stats = result.get("Statistics") or result.get("statistics")
    if isinstance(stats, Mapping):
        for key, value in stats.items():
            parsed = _parse_value(value)
            if parsed is not None:
                metrics[_normalize_key(str(key))] = parsed

    total_perf = result.get("TotalPerformance") or result.get("totalPerformance")
    if isinstance(total_perf, Mapping):
        portfolio = total_perf.get("PortfolioStatistics") or total_perf.get(
            "portfolioStatistics"
        )
        if isinstance(portfolio, Mapping):
            for key, value in portfolio.items():
                parsed = _parse_value(value)
                if parsed is not None:
                    metrics[_normalize_key(str(key))] = parsed
    return metrics


def _parse_constraint(expression: str) -> tuple[str, str, float]:
    operators = [">=", "<=", "==", ">", "<"]
    for op in operators:
        if op in expression:
            lhs, rhs = expression.split(op, maxsplit=1)
            return lhs.strip(), op, float(rhs.strip())
    raise ValueError(f"Invalid constraint expression: {expression}")


def _compare(actual: float, op: str, target: float) -> bool:
    if op == ">=":
        return actual >= target
    if op == "<=":
        return actual <= target
    if op == ">":
        return actual > target
    if op == "<":
        return actual < target
    if op == "==":
        return actual == target
    raise ValueError(f"Unsupported operator: {op}")


def evaluate_objective(
    result: Mapping[str, Any],
    config: ObjectiveConfig,
) -> ObjectiveResult:
    """Evaluate objective and constraints for a LEAN backtest result."""
    metrics = _extract_metrics(result)
    target_key = _normalize_key(config.target)
    objective_value = metrics.get(target_key)
    if objective_value is None:
        raise ValueError(f"Target metric '{config.target}' not found in results")

    constraint_results: list[ConstraintResult] = []
    for expression in config.constraints:
        metric, op, threshold = _parse_constraint(expression)
        metric_key = _normalize_key(metric)
        actual = metrics.get(metric_key)
        passed = actual is not None and _compare(actual, op, threshold)
        constraint_results.append(
            ConstraintResult(expression=expression, passed=passed, actual=actual)
        )

    constraints_passed = all(item.passed for item in constraint_results)
    return ObjectiveResult(
        objective=objective_value,
        direction=config.direction,
        constraints_passed=constraints_passed,
        constraint_results=tuple(constraint_results),
        metrics=metrics,
    )
