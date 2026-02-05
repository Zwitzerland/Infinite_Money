"""AWS Braket QUBO scaffolding."""
from __future__ import annotations

from typing import Iterable


def build_qubo(
    expected_returns: Iterable[float],
    covariance: list[list[float]],
    budget: int,
    risk_aversion: float,
    penalty: float,
) -> dict[tuple[int, int], float]:
    """Build a basic QUBO for portfolio selection.

    Uses a binary selection vector with a budget constraint.
    """
    mu = list(expected_returns)
    n = len(mu)
    if n == 0:
        raise ValueError("expected_returns must be non-empty")
    if len(covariance) != n:
        raise ValueError("covariance matrix shape mismatch")

    qubo: dict[tuple[int, int], float] = {}
    for i in range(n):
        qubo[(i, i)] = risk_aversion * covariance[i][i] - mu[i] + penalty * (
            1 - 2 * budget
        )
        for j in range(i + 1, n):
            qubo[(i, j)] = 2 * risk_aversion * covariance[i][j] + 2 * penalty
    return qubo


def submit_qubo_job(
    qubo: dict[tuple[int, int], float],
    device_arn: str,
    s3_bucket: str,
    s3_prefix: str,
    shots: int = 100,
) -> object:
    """Submit a QUBO to AWS Braket (requires Braket SDK + Ocean plugin)."""
    try:
        from braket.ocean_plugin import BraketDWaveSampler
    except ImportError as exc:
        raise RuntimeError(
            "Braket SDK not installed. Use `pip install -e .[quantum]`."
        ) from exc

    sampler = BraketDWaveSampler(
        s3_destination_folder=(s3_bucket, s3_prefix),
        device_arn=device_arn,
    )
    return sampler.sample_qubo(qubo, num_reads=shots)
