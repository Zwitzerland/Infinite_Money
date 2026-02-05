"""AWS client helpers."""
from __future__ import annotations

from typing import Any

import boto3


def get_session(region: str | None = None, profile: str | None = None) -> boto3.Session:
    """Create a boto3 session."""
    return boto3.Session(region_name=region, profile_name=profile)


def get_client(service: str, region: str | None = None, profile: str | None = None) -> Any:
    """Get a boto3 client for the given service."""
    session = get_session(region=region, profile=profile)
    return session.client(service)
