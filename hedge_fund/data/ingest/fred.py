"""FRED ingestion connector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

from hedge_fund.data.ingest.base import IngestResult, normalize_records, utc_now


@dataclass(frozen=True)
class FredConfig:
    """Configuration for FRED ingestion."""

    api_key: str
    series_id: str


def fetch_fred_series(config: FredConfig) -> IngestResult:
    """Fetch a FRED series and normalize to EventRecord."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": config.series_id,
        "api_key": config.api_key,
        "file_type": "json",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    observations: Iterable[dict[str, str]] = payload.get("observations", [])
    ingest_time = utc_now()
    records = []
    for obs in observations:
        event_time = pd.Timestamp(obs["date"], tz="UTC").to_pydatetime()
        value = obs["value"]
        record = normalize_records(
            domain="macro",
            source="FRED",
            publisher_uri=url,
            event_time=event_time,
            publish_time=ingest_time,
            ingest_time=ingest_time,
            value=value,
            units="index",
            license_tier="open",
            confidence=1.0,
            symbol_id=(config.series_id,),
            raw_payload=str(obs).encode(),
        )
        records.append(record)
    return IngestResult(records=tuple(records), raw_payload=str(payload).encode())
