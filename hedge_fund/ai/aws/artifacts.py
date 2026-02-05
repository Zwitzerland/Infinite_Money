"""S3 artifact uploader."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import os

from .clients import get_client


def _iter_files(path: Path) -> Iterable[Path]:
    for root, _, files in os.walk(path):
        for file in files:
            yield Path(root) / file


def upload_directory(
    local_dir: Path,
    bucket: str,
    prefix: str,
    region: str | None = None,
    profile: str | None = None,
) -> list[str]:
    """Upload a directory to S3."""
    client = get_client("s3", region=region, profile=profile)
    uploaded: list[str] = []
    for file_path in _iter_files(local_dir):
        rel = file_path.relative_to(local_dir)
        key = f"{prefix.rstrip('/')}/{rel.as_posix()}"
        client.upload_file(str(file_path), bucket, key)
        uploaded.append(key)
    return uploaded
