"""CLI for news ingestion."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra

from hedge_fund.data.news.providers import (
    CurrentsConfig,
    EventRegistryConfig,
    NewsApiConfig,
    fetch_currents,
    fetch_event_registry,
    fetch_gdelt_gkg,
    fetch_newsapi,
    normalize_articles,
)
from hedge_fund.utils.settings import PlatformSettings


@dataclass(frozen=True)
class NewsCliConfig:
    """Configuration for news ingestion CLI."""

    source: Literal["gdelt", "newsapi", "currents", "eventregistry"]
    output_path: str
    query: str
    gdelt_gkg_url: str | None = None


@hydra.main(config_path="../../../conf", config_name="news_ingest", version_base=None)
def main(cfg: NewsCliConfig) -> None:
    settings = PlatformSettings()
    output = Path(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if cfg.source == "gdelt":
        if cfg.gdelt_gkg_url is None:
            raise ValueError("gdelt_gkg_url required for GDELT.")
        raw = fetch_gdelt_gkg(cfg.gdelt_gkg_url)
    elif cfg.source == "newsapi":
        if settings.newsapi_key is None:
            raise ValueError("NewsAPI key missing; inject via secrets manager.")
        raw = fetch_newsapi(
            NewsApiConfig(api_key=settings.newsapi_key.get_secret_value(), query=cfg.query)
        )
    elif cfg.source == "currents":
        if settings.currents_api_key is None:
            raise ValueError("Currents API key missing; inject via secrets manager.")
        raw = fetch_currents(
            CurrentsConfig(
                api_key=settings.currents_api_key.get_secret_value(), query=cfg.query
            )
        )
    elif cfg.source == "eventregistry":
        if settings.eventregistry_api_key is None:
            raise ValueError("Event Registry API key missing; inject via secrets manager.")
        raw = fetch_event_registry(
            EventRegistryConfig(
                api_key=settings.eventregistry_api_key.get_secret_value(), query=cfg.query
            )
        )
    else:
        raise ValueError(f"Unknown source: {cfg.source}")

    articles = normalize_articles(raw)
    payload = [article.model_dump() for article in articles]
    output.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
