"""Hybrid quantum/classical optimization loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .braket import submit_qubo_job


@dataclass
class HybridResult:
    solution: Mapping[int, int]
    objective: float
    penalty: float


def solve_qubo_hybrid(
    qubo: dict[tuple[int, int], float],
    device_arn: str,
    s3_bucket: str,
    s3_prefix: str,
    shots: int = 100,
) -> HybridResult:
    """Run a QUBO solve and return the best sample (annealer)."""
    sampler: Any = submit_qubo_job(
        qubo,
        device_arn=device_arn,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        shots=shots,
    )
    sample_set: Any = getattr(sampler, "first")
    best = sample_set.sample
    energy = sample_set.energy
    return HybridResult(solution=best, objective=float(-energy), penalty=0.0)
