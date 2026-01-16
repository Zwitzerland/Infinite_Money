"""OFAC SDN ingestion connector."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from io import StringIO
import requests

from hedge_fund.data.ingest.base import IngestResult, normalize_records, utc_now


@dataclass(frozen=True)
class OfacConfig:
    """Configuration for OFAC ingestion."""

    sdn_url: str = "https://www.treasury.gov/ofac/downloads/sdn.csv"


def fetch_ofac_sdn(config: OfacConfig) -> IngestResult:
    """Fetch OFAC SDN list and normalize to EventRecord."""
    response = requests.get(config.sdn_url, timeout=30)
    response.raise_for_status()
    payload = response.text
    frame = pd.read_csv(StringIO(payload), header=None, dtype=str)
    ingest_time = utc_now()
    records = []
    for _, row in frame.iterrows():
        value = row.to_dict()
        record = normalize_records(
            domain="sec",
            source="OFAC",
            publisher_uri=config.sdn_url,
            event_time=ingest_time,
            publish_time=ingest_time,
            ingest_time=ingest_time,
            value=value,
            units="list",
            license_tier="open",
            confidence=1.0,
            symbol_id=(),
            raw_payload=str(value).encode(),
        )
        records.append(record)
    return IngestResult(records=tuple(records), raw_payload=payload.encode())
