"""Configuration helpers for the AI stack."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load an OmegaConf config from disk."""
    return OmegaConf.load(Path(path))


def config_to_dict(cfg: DictConfig) -> Mapping[str, Any]:
    """Convert config to a standard dict for serialization."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
