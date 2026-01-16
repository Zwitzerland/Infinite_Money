"""Base utilities for ingestion connectors."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from hedge_fund.data.schema.event_record import EventRecord, GeoPoint, Domain, License
from hedge_fund.data.utils.checksums import sha256_digest


@dataclass(frozen=True)
class IngestResult:
    """Container for ingested EventRecords and raw payload."""

    records: tuple[EventRecord, ...]
    raw_payload: bytes


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=timezone.utc)


def normalize_records(
    *,
    domain: Domain,
    source: str,
    publisher_uri: str,
    event_time: datetime,
    publish_time: datetime,
    ingest_time: datetime,
    value: Any,
    units: str,
    license_tier: License,
    confidence: float,
    symbol_id: tuple[str, ...] = (),
    geo: GeoPoint | None = None,
    rev_id: str | None = None,
    raw_payload: bytes,
) -> EventRecord:
    """Create an EventRecord from source data."""
    checksum = sha256_digest(raw_payload)
    return EventRecord(
        domain=domain,
        source=source,
        publisher_uri=publisher_uri,
        symbol_id=symbol_id,
        geo=geo,
        event_time=event_time,
        publish_time=publish_time,
        ingest_time=ingest_time,
        value=value,
        units=units,
        rev_id=rev_id,
        license=license_tier,
        confidence=confidence,
        checksum=checksum,
    )


def records_to_frame(records: tuple[EventRecord, ...]) -> pd.DataFrame:
    """Convert EventRecords to a DataFrame."""
    return pd.DataFrame([record.model_dump() for record in records])


def ensure_response_ok(response: Any, context: str) -> Mapping[str, Any]:
    """Validate HTTP response and return parsed JSON."""
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: expected JSON object response.")
    return payload
