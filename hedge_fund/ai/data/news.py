"""News ingestion and normalization helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any, Iterable, Mapping

import os

import requests


@dataclass(frozen=True)
class NewsArticle:
    """Canonical news article record."""

    id: str
    title: str
    body: str
    source: str
    published_at: datetime
    url: str | None = None


def _hash_article(title: str, body: str, source: str) -> str:
    digest = sha1(f"{title}|{body}|{source}".encode("utf-8")).hexdigest()
    return digest


def dedupe_articles(articles: Iterable[NewsArticle]) -> list[NewsArticle]:
    """Remove exact duplicates based on hash of title/body/source."""
    seen: set[str] = set()
    output: list[NewsArticle] = []
    for article in articles:
        if article.id in seen:
            continue
        seen.add(article.id)
        output.append(article)
    return output


def fetch_news(config: Mapping[str, Any]) -> list[NewsArticle]:
    """Fetch news using a minimal API integration.

    Currently supports NewsAPI-compatible endpoints.
    """
    provider = str(config.get("provider", "newsapi")).lower()
    if provider != "newsapi":
        raise ValueError(f"Unsupported news provider: {provider}")

    api_key_env = str(config.get("api_key_env", "NEWS_API_KEY"))
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in ${api_key_env}")

    query = str(config.get("query", "equities OR stocks OR markets"))
    lookback_days = int(config.get("lookback_days", 7))
    max_results = int(config.get("max_results", 50))
    language = str(config.get("language", "en"))

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "from": start.date().isoformat(),
            "to": end.date().isoformat(),
            "language": language,
            "pageSize": max_results,
        },
        headers={"X-Api-Key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    articles: list[NewsArticle] = []
    for item in payload.get("articles", []):
        title = str(item.get("title") or "")
        body = str(item.get("description") or "")
        source = str((item.get("source") or {}).get("name") or "unknown")
        published_raw = item.get("publishedAt")
        published = (
            datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            if published_raw
            else end
        )
        article_id = _hash_article(title, body, source)
        articles.append(
            NewsArticle(
                id=article_id,
                title=title,
                body=body,
                source=source,
                published_at=published,
                url=item.get("url"),
            )
        )
    return articles
