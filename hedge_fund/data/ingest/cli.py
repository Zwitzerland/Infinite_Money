"""CLI for ingestion connectors."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra

from hedge_fund.data.ingest.bls import BlsConfig, fetch_bls_series
from hedge_fund.data.ingest.fred import FredConfig, fetch_fred_series
from hedge_fund.data.ingest.gdelt import GdeltConfig, fetch_gdelt_events
from hedge_fund.data.ingest.ofac import OfacConfig, fetch_ofac_sdn
from hedge_fund.data.ingest.patentsview import PatentsViewConfig, fetch_patentsview
from hedge_fund.data.ingest.treasury import TreasuryConfig, fetch_treasury_dataset
from hedge_fund.data.ingest.usgs import UsgsConfig, fetch_usgs_feed
from hedge_fund.data.ingest.base import records_to_frame
from hedge_fund.utils.settings import PlatformSettings


@dataclass(frozen=True)
class IngestCliConfig:
    """Configuration for ingestion CLI."""

    source: Literal["fred", "bls", "treasury", "usgs", "gdelt", "ofac", "patentsview"]
    output_path: str
    series_id: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    treasury_endpoint: str | None = None
    gdelt_query: str | None = None
    patentsview_query: str | None = None


@hydra.main(config_path="../../../conf", config_name="ingest", version_base=None)
def main(cfg: IngestCliConfig) -> None:
    settings = PlatformSettings()
    output = Path(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if cfg.source == "fred":
        if cfg.series_id is None:
            raise ValueError("series_id required for FRED.")
        key = settings.fred_api_key
        if key is None:
            raise ValueError("FRED API key missing; inject via secrets manager.")
        result = fetch_fred_series(
            FredConfig(api_key=key.get_secret_value(), series_id=cfg.series_id)
        )
    elif cfg.source == "bls":
        if cfg.series_id is None or cfg.start_year is None or cfg.end_year is None:
            raise ValueError("series_id, start_year, end_year required for BLS.")
        key = settings.bls_api_key
        result = fetch_bls_series(
            BlsConfig(
                series_id=cfg.series_id,
                start_year=cfg.start_year,
                end_year=cfg.end_year,
                api_key=key.get_secret_value() if key else None,
            )
        )
    elif cfg.source == "treasury":
        if cfg.treasury_endpoint is None:
            raise ValueError("treasury_endpoint required for Treasury.")
        result = fetch_treasury_dataset(TreasuryConfig(endpoint=cfg.treasury_endpoint))
    elif cfg.source == "usgs":
        result = fetch_usgs_feed(UsgsConfig())
    elif cfg.source == "gdelt":
        if cfg.gdelt_query is None:
            raise ValueError("gdelt_query required for GDELT.")
        result = fetch_gdelt_events(GdeltConfig(query=cfg.gdelt_query))
    elif cfg.source == "ofac":
        result = fetch_ofac_sdn(OfacConfig())
    elif cfg.source == "patentsview":
        if cfg.patentsview_query is None:
            raise ValueError("patentsview_query required for PatentsView.")
        result = fetch_patentsview(
            PatentsViewConfig(
                query=json.loads(cfg.patentsview_query),
                fields=["patent_title", "patent_date", "patent_number"],
            )
        )
    else:
        raise ValueError(f"Unknown source: {cfg.source}")

    frame = records_to_frame(result.records)
    frame.to_parquet(output, index=False)
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps({"record_count": len(result.records)}, indent=2))


if __name__ == "__main__":
    main()
