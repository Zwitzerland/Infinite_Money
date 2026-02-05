"""Legal filings ingestion (research-only)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import json
import requests


@dataclass(frozen=True)
class FilingEvent:
    """Minimal normalized filing event metadata."""

    source: str
    identifier: str
    issuer: str | None
    fetched_at: str
    raw_excerpt: str | None = None


def _headers(config: Mapping[str, Any]) -> dict[str, str]:
    user_agent_env = str(config.get("user_agent_env", "SEC_USER_AGENT"))
    user_agent = str(config.get(user_agent_env, ""))
    if not user_agent:
        user_agent = "InfiniteMoney/1.0 (research-only; contact: you@example.com)"
    return {"User-Agent": user_agent}


def fetch_sec_filings(config: Mapping[str, Any]) -> list[FilingEvent]:
    if not config.get("enabled", False):
        return []
    cik_list = config.get("cik_list", [])
    events: list[FilingEvent] = []
    headers = _headers(config)
    for cik in cik_list:
        cik_str = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        issuer = payload.get("entityName") if isinstance(payload, dict) else None
        excerpt = json.dumps(payload)[:1000]
        events.append(
            FilingEvent(
                source="sec_edgar",
                identifier=f"CIK{cik_str}",
                issuer=issuer,
                fetched_at=datetime.utcnow().isoformat(),
                raw_excerpt=excerpt,
            )
        )
    return events


def fetch_house_ptr(config: Mapping[str, Any]) -> list[FilingEvent]:
    if not config.get("enabled", False):
        return []
    url = str(config.get("download_url", "")).strip()
    if not url:
        return []
    resp = requests.get(url, headers=_headers(config), timeout=30)
    resp.raise_for_status()
    excerpt = resp.text[:1000]
    return [
        FilingEvent(
            source="house_ptr",
            identifier=url,
            issuer=None,
            fetched_at=datetime.utcnow().isoformat(),
            raw_excerpt=excerpt,
        )
    ]


def fetch_senate_ptr(config: Mapping[str, Any]) -> list[FilingEvent]:
    if not config.get("enabled", False):
        return []
    url = str(config.get("download_url", "")).strip()
    if not url:
        return []
    resp = requests.get(url, headers=_headers(config), timeout=30)
    resp.raise_for_status()
    excerpt = resp.text[:1000]
    return [
        FilingEvent(
            source="senate_ptr",
            identifier=url,
            issuer=None,
            fetched_at=datetime.utcnow().isoformat(),
            raw_excerpt=excerpt,
        )
    ]


def _merge_user_agent(root: Mapping[str, Any], sub: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(sub)
    if "user_agent_env" not in merged and root.get("user_agent_env"):
        merged["user_agent_env"] = root.get("user_agent_env")
    return merged


def fetch_filings(config: Mapping[str, Any]) -> list[FilingEvent]:
    """Fetch filings from enabled sources."""

    if not config.get("enabled", False):
        return []
    events: list[FilingEvent] = []
    events.extend(fetch_sec_filings(_merge_user_agent(config, config.get("sec_edgar", {}))))
    events.extend(fetch_house_ptr(_merge_user_agent(config, config.get("house", {}))))
    events.extend(fetch_senate_ptr(_merge_user_agent(config, config.get("senate", {}))))
    return events
