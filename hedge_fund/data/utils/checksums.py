"""Checksum utilities."""
from __future__ import annotations

import hashlib


def sha256_digest(payload: bytes) -> str:
    """Return a SHA256 checksum for a payload."""
    return hashlib.sha256(payload).hexdigest()
