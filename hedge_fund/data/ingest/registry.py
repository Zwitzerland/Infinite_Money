"""Data source registry for ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AuthMethod = Literal["none", "api_key", "oauth", "token"]


@dataclass(frozen=True)
class DataSource:
    """Metadata for a data source."""

    name: str
    domain: str
    description: str
    publisher_uri: str
    license_tier: str
    auth_method: AuthMethod
    rate_limit_per_minute: int | None = None


SOURCE_REGISTRY: tuple[DataSource, ...] = (
    DataSource(
        name="FRED",
        domain="macro",
        description="FRED macro series.",
        publisher_uri="https://api.stlouisfed.org/fred/series/observations",
        license_tier="open",
        auth_method="api_key",
    ),
    DataSource(
        name="BLS",
        domain="macro",
        description="BLS public time series API.",
        publisher_uri="https://api.bls.gov/publicAPI/v2/timeseries/data/",
        license_tier="open",
        auth_method="api_key",
    ),
    DataSource(
        name="TREASURY",
        domain="rates",
        description="Treasury FiscalData API.",
        publisher_uri="https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/",
        license_tier="open",
        auth_method="none",
    ),
    DataSource(
        name="USGS",
        domain="geo",
        description="USGS earthquake feeds.",
        publisher_uri="https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        license_tier="open",
        auth_method="none",
    ),
    DataSource(
        name="GDELT",
        domain="news",
        description="GDELT DOC API.",
        publisher_uri="https://api.gdeltproject.org/api/v2/doc/doc",
        license_tier="open",
        auth_method="none",
    ),
    DataSource(
        name="OFAC_SDN",
        domain="sec",
        description="OFAC SDN sanctions list.",
        publisher_uri="https://www.treasury.gov/ofac/downloads/sdn.csv",
        license_tier="open",
        auth_method="none",
    ),
    DataSource(
        name="PATENTSVIEW",
        domain="macro",
        description="PatentsView search API.",
        publisher_uri="https://search.patentsview.org/api/v1/patents/query",
        license_tier="open",
        auth_method="none",
    ),
)
