"""Secrets Manager helpers."""
from __future__ import annotations

import json
from typing import Any

from .clients import get_client


def get_secret_value(
    secret_id: str,
    region: str | None = None,
    profile: str | None = None,
) -> Any:
    """Fetch and parse a secret value from AWS Secrets Manager."""
    client = get_client("secretsmanager", region=region, profile=profile)
    response = client.get_secret_value(SecretId=secret_id)
    secret = response.get("SecretString")
    if secret is None:
        raise ValueError(f"Secret {secret_id} has no SecretString")
    try:
        return json.loads(secret)
    except json.JSONDecodeError:
        return secret
