"""EventRecord schema for the alpha fabric ingestion layer."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


Domain = Literal[
    "macro",
    "rates",
    "sec",
    "pol",
    "defense",
    "weather",
    "energy",
    "geo",
    "flight",
    "ship",
    "market",
    "news",
]
License = Literal["open", "restricted", "paid"]


class GeoPoint(BaseModel):
    """Geographic metadata."""

    iso3: str | None = None
    lat: float | None = None
    lon: float | None = None
    admin1: str | None = None
    admin2: str | None = None


class EventRecord(BaseModel):
    """Normalized event-time record with provenance.

    Parameters
    ----------
    event_id
        UUIDv7-compatible identifier.
    domain
        Domain label for the event.
    source
        Source system name.
    publisher_uri
        Canonical URL of the source.
    symbol_id
        Identifier tuple (FIGI/ISIN/CUSIP/CIK/NA).
    event_time
        When the event occurred (UTC).
    publish_time
        When the event was published (UTC).
    ingest_time
        When the event was ingested (UTC).
    value
        Scalar/vector/json payload.
    units
        Units for the value field.
    license
        License tier for the source.
    confidence
        Confidence score in [0, 1].
    checksum
        SHA256 checksum over the raw payload.
    """

    event_id: UUID = Field(default_factory=uuid4)
    domain: Domain
    source: str
    publisher_uri: str
    symbol_id: tuple[str, ...] = Field(default_factory=tuple)
    geo: GeoPoint | None = None
    event_time: datetime
    publish_time: datetime
    ingest_time: datetime
    value: Any
    units: str
    rev_id: str | None = None
    license: License
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    checksum: str
