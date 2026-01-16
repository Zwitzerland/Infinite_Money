"""News and event schemas for the alpha fabric."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class Article(BaseModel):
    """Normalized news article."""

    article_id: str
    source: str
    url: HttpUrl
    published_at: datetime
    ingested_at: datetime
    language: str
    title: str
    snippet: str | None = None
    duplicate_cluster_id: str | None = None


class EntityMention(BaseModel):
    """Entity mention extracted from an article."""

    article_id: str
    entity_type: str
    entity_id: str
    surface_form: str
    confidence: float = Field(ge=0.0, le=1.0)


class Event(BaseModel):
    """Normalized news event."""

    event_id: str
    event_time: datetime
    event_type_code: str
    actors: tuple[str, ...]
    locations: tuple[str, ...]
    tone: float | None = None
    supporting_article_ids: tuple[str, ...]


class AssetLink(BaseModel):
    """Mapping from entity to asset identifier."""

    entity_id: str
    asset_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    effective_from: datetime
    effective_to: datetime | None = None


class MarketImpactObservation(BaseModel):
    """Observed market response to an event."""

    event_id: str
    asset_id: str
    window: str
    returns: float
    volatility: float
    volume: float
    drawdown: float
    recovery_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)
