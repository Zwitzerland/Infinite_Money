"""BLS ingestion connector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

from hedge_fund.data.ingest.base import IngestResult, normalize_records, utc_now


@dataclass(frozen=True)
class BlsConfig:
    """Configuration for BLS ingestion."""

    series_id: str
    start_year: int
    end_year: int
    api_key: str | None = None


def fetch_bls_series(config: BlsConfig) -> IngestResult:
    """Fetch a BLS series and normalize to EventRecord."""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": [config.series_id],
        "startyear": str(config.start_year),
        "endyear": str(config.end_year),
    }
    if config.api_key:
        payload["registrationKey"] = config.api_key
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    series_list: Iterable[dict[str, object]] = data.get("Results", {}).get("series", [])
    ingest_time = utc_now()
    records = []
    for series in series_list:
        for item in series.get("data", []):
            year = item["year"]
            period = item["period"]
            if not period.startswith("M"):
                continue
            month = period[1:]
            event_time = pd.Timestamp(f"{year}-{month}-01", tz="UTC").to_pydatetime()
            record = normalize_records(
                domain="macro",
                source="BLS",
                publisher_uri=url,
                event_time=event_time,
                publish_time=ingest_time,
                ingest_time=ingest_time,
                value=item["value"],
                units=item.get("footnotes", "") or "index",
                license_tier="open",
                confidence=1.0,
                symbol_id=(config.series_id,),
                raw_payload=str(item).encode(),
            )
            records.append(record)
    return IngestResult(records=tuple(records), raw_payload=str(data).encode())
