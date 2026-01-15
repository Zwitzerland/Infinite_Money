"""Promotion packet schema for research → validation → deployment."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from pydantic import BaseModel, Field


class PromotionPacket(BaseModel):
    """Structured promotion packet for gate evaluation.

    Parameters
    ----------
    run_id
        Unique identifier for the experiment run.
    created_at
        UTC timestamp for packet creation.
    metrics
        Key performance and risk metrics.
    parameters
        Parameter set evaluated in the backtest.
    passed
        Whether the candidate passed promotion gates.
    """

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    metrics: Mapping[str, Any]
    parameters: Mapping[str, Any]
    passed: bool
