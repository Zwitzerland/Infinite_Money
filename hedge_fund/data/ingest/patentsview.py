"""PatentsView ingestion connector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from hedge_fund.data.ingest.base import IngestResult, normalize_records, utc_now


@dataclass(frozen=True)
class PatentsViewConfig:
    """Configuration for PatentsView ingestion."""

    query: dict[str, Any]
    fields: list[str]
    limit: int = 50


def fetch_patentsview(config: PatentsViewConfig) -> IngestResult:
    """Fetch PatentsView results and normalize to EventRecord."""
    url = "https://search.patentsview.org/api/v1/patents/query"
    payload = {"q": config.query, "f": config.fields, "o": {"per_page": config.limit}}
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    ingest_time = utc_now()
    records = []
    for patent in data.get("patents", []):
        date_str = patent.get("patent_date", None)
        event_time = (
            pd.Timestamp(date_str, tz="UTC").to_pydatetime()
            if date_str
            else ingest_time
        )
        record = normalize_records(
            domain="macro",
            source="PATENTSVIEW",
            publisher_uri=url,
            event_time=event_time,
            publish_time=ingest_time,
            ingest_time=ingest_time,
            value=patent,
            units="count",
            license_tier="open",
            confidence=0.8,
            symbol_id=(),
            raw_payload=str(patent).encode(),
        )
        records.append(record)
    return IngestResult(records=tuple(records), raw_payload=str(data).encode())
