"""Support/resistance (SR) barrier models.

This package provides an algorithmic (non-discretionary) definition of SR levels
and a leakage-aware evaluation pipeline built around two-barrier (take-profit / stop)
first-hitting outcomes.
"""

from __future__ import annotations

from .barrier import SRBarrierParams, SRBarrierResult, compute_sr_barrier_exposure, compute_sr_barrier_result

__all__ = [
    "SRBarrierParams",
    "SRBarrierResult",
    "compute_sr_barrier_exposure",
    "compute_sr_barrier_result",
]
