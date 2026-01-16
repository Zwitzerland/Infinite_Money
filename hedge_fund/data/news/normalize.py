"""Normalization helpers for news ingestion."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urlunparse

from hedge_fund.data.news.models import Article


@dataclass(frozen=True)
class RawArticle:
    """Raw article payload from providers."""

    source: str
    url: str
    published_at: datetime
    language: str
    title: str
    snippet: str | None
    payload: dict[str, Any]


def canonicalize_url(url: str) -> str:
    """Normalize URL by stripping query/fragment."""
    parts = urlparse(url)
    return urlunparse((parts.scheme, parts.netloc, parts.path, "", "", ""))


def hash_article(url: str, title: str) -> str:
    """Hash article URL + title for stable ID."""
    digest = hashlib.sha256(f"{url}|{title}".encode()).hexdigest()
    return digest


def normalize_article(raw: RawArticle) -> Article:
    """Normalize raw article payload into Article."""
    canonical_url = canonicalize_url(raw.url)
    article_id = hash_article(canonical_url, raw.title)
    return Article(
        article_id=article_id,
        source=raw.source,
        url=canonical_url,
        published_at=raw.published_at,
        ingested_at=datetime.now(tz=timezone.utc),
        language=raw.language,
        title=raw.title,
        snippet=raw.snippet,
    )
