"""News providers ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import requests

from hedge_fund.data.news.normalize import RawArticle, normalize_article
from hedge_fund.data.news.models import Article


@dataclass(frozen=True)
class NewsApiConfig:
    """Config for NewsAPI."""

    api_key: str
    query: str


@dataclass(frozen=True)
class CurrentsConfig:
    """Config for Currents API."""

    api_key: str
    query: str


@dataclass(frozen=True)
class EventRegistryConfig:
    """Config for Event Registry API."""

    api_key: str
    query: str


def _parse_datetime(value: str) -> datetime:
    return pd.Timestamp(value, tz="UTC").to_pydatetime()


def fetch_newsapi(config: NewsApiConfig) -> tuple[RawArticle, ...]:
    """Fetch articles from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    response = requests.get(
        url, params={"q": config.query, "apiKey": config.api_key}, timeout=30
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    articles: Iterable[dict[str, Any]] = payload.get("articles", [])
    raw = []
    for article in articles:
        raw.append(
            RawArticle(
                source="NewsAPI",
                url=article["url"],
                published_at=_parse_datetime(article["publishedAt"]),
                language=article.get("language", "en"),
                title=article.get("title", ""),
                snippet=article.get("description"),
                payload=article,
            )
        )
    return tuple(raw)


def fetch_currents(config: CurrentsConfig) -> tuple[RawArticle, ...]:
    """Fetch articles from Currents."""
    url = "https://api.currentsapi.services/v1/latest-news"
    response = requests.get(
        url, params={"keywords": config.query, "apiKey": config.api_key}, timeout=30
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    news: Iterable[dict[str, Any]] = payload.get("news", [])
    raw = []
    for article in news:
        raw.append(
            RawArticle(
                source="Currents",
                url=article["url"],
                published_at=_parse_datetime(article["published"]),
                language=article.get("language", "en"),
                title=article.get("title", ""),
                snippet=article.get("description"),
                payload=article,
            )
        )
    return tuple(raw)


def fetch_event_registry(config: EventRegistryConfig) -> tuple[RawArticle, ...]:
    """Fetch articles from Event Registry."""
    url = "https://eventregistry.org/api/v1/article/getArticles"
    response = requests.post(
        url,
        json={
            "apiKey": config.api_key,
            "keyword": config.query,
            "resultType": "articles",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    articles = payload.get("articles", {}).get("results", [])
    raw = []
    for article in articles:
        raw.append(
            RawArticle(
                source="EventRegistry",
                url=article["url"],
                published_at=_parse_datetime(article["dateTimePub"]),
                language=article.get("lang", "en"),
                title=article.get("title", ""),
                snippet=article.get("body"),
                payload=article,
            )
        )
    return tuple(raw)


def fetch_gdelt_gkg(url: str) -> tuple[RawArticle, ...]:
    """Fetch GDELT GKG JSON feed and normalize to RawArticle."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    raw = []
    for entry in payload.get("articles", []):
        raw.append(
            RawArticle(
                source="GDELT",
                url=entry["url"],
                published_at=_parse_datetime(entry["seendate"]),
                language=entry.get("language", "en"),
                title=entry.get("title", ""),
                snippet=entry.get("snippet"),
                payload=entry,
            )
        )
    return tuple(raw)


def normalize_articles(raw_articles: Iterable[RawArticle]) -> tuple[Article, ...]:
    """Normalize raw articles into Article models."""
    return tuple(normalize_article(article) for article in raw_articles)
