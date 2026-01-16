"""GDELT event feed connector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

from hedge_fund.data.ingest.base import IngestResult, normalize_records, utc_now


@dataclass(frozen=True)
class GdeltConfig:
    """Configuration for GDELT ingestion."""

    query: str
    max_records: int = 50


def fetch_gdelt_events(config: GdeltConfig) -> IngestResult:
    """Fetch GDELT events using the DOC API."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": config.query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(config.max_records),
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    articles: Iterable[dict[str, str]] = payload.get("articles", [])
    ingest_time = utc_now()
    records = []
    for article in articles:
        publish_time = pd.Timestamp(article.get("seendate"), tz="UTC").to_pydatetime()
        record = normalize_records(
            domain="news",
            source="GDELT",
            publisher_uri=url,
            event_time=publish_time,
            publish_time=publish_time,
            ingest_time=ingest_time,
            value=article,
            units="count",
            license_tier="open",
            confidence=0.8,
            symbol_id=(),
            raw_payload=str(article).encode(),
        )
        records.append(record)
    return IngestResult(records=tuple(records), raw_payload=str(payload).encode())
