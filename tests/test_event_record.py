from __future__ import annotations

from datetime import datetime, timezone

from hedge_fund.data.schema.event_record import EventRecord


def test_event_record_serializes() -> None:
    now = datetime.now(tz=timezone.utc)
    record = EventRecord(
        domain="macro",
        source="FRED",
        publisher_uri="https://example.test",
        symbol_id=("CPI",),
        event_time=now,
        publish_time=now,
        ingest_time=now,
        value=1.0,
        units="index",
        license="open",
        confidence=0.9,
        checksum="abc123",
    )
    payload = record.model_dump()
    assert payload["domain"] == "macro"
    assert payload["checksum"] == "abc123"
