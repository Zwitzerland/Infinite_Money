"""USGS earthquake feed connector."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from hedge_fund.data.ingest.base import IngestResult, normalize_records, utc_now
from hedge_fund.data.schema.event_record import GeoPoint


@dataclass(frozen=True)
class UsgsConfig:
    """Configuration for USGS ingestion."""

    feed_url: str = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"


def fetch_usgs_feed(config: UsgsConfig) -> IngestResult:
    """Fetch USGS earthquake data and normalize to EventRecord."""
    response = requests.get(config.feed_url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    ingest_time = utc_now()
    records = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [])
        event_time = pd.Timestamp(props.get("time", 0), unit="ms", tz="UTC").to_pydatetime()
        geo = None
        if len(coords) >= 2:
            geo = GeoPoint(lat=coords[1], lon=coords[0])
        record = normalize_records(
            domain="geo",
            source="USGS",
            publisher_uri=config.feed_url,
            event_time=event_time,
            publish_time=ingest_time,
            ingest_time=ingest_time,
            value=props,
            units="mw",
            license_tier="open",
            confidence=1.0,
            symbol_id=(),
            geo=geo,
            raw_payload=str(feature).encode(),
        )
        records.append(record)
    return IngestResult(records=tuple(records), raw_payload=str(payload).encode())
