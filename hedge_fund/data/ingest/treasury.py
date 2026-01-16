"""Treasury FiscalData ingestion connector."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from hedge_fund.data.ingest.base import IngestResult, normalize_records, utc_now


@dataclass(frozen=True)
class TreasuryConfig:
    """Configuration for Treasury FiscalData ingestion."""

    endpoint: str
    page_size: int = 100


def fetch_treasury_dataset(config: TreasuryConfig) -> IngestResult:
    """Fetch Treasury FiscalData dataset and normalize to EventRecord."""
    url = f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/{config.endpoint}"
    params = {"page[size]": config.page_size}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    ingest_time = utc_now()
    records = []
    for item in data:
        date_value = item.get("record_date") or item.get("date") or item.get("record_date")
        if date_value is None:
            continue
        event_time = pd.Timestamp(date_value, tz="UTC").to_pydatetime()
        record = normalize_records(
            domain="rates",
            source="TREASURY",
            publisher_uri=url,
            event_time=event_time,
            publish_time=ingest_time,
            ingest_time=ingest_time,
            value=item,
            units="mixed",
            license_tier="open",
            confidence=1.0,
            symbol_id=(),
            raw_payload=str(item).encode(),
        )
        records.append(record)
    return IngestResult(records=tuple(records), raw_payload=str(payload).encode())
