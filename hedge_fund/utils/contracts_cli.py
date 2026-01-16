"""CLI to materialize contract bundles for the platform."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
from omegaconf import DictConfig, OmegaConf

from hedge_fund.utils.contracts import default_contract_bundle
from hedge_fund.utils.settings import PlatformSettings


@dataclass(frozen=True)
class ContractCliConfig:
    """Configuration for contract bundle output."""

    format: Literal["json"] = "json"
    output_path: str | None = None


def _serialize_bundle(bundle: DictConfig) -> str:
    return json.dumps(OmegaConf.to_container(bundle, resolve=True), indent=2)


@hydra.main(
    config_path="../../conf",
    config_name="contracts",
    version_base=None,
)  # type: ignore[misc]
def main(cfg: ContractCliConfig) -> None:
    settings = PlatformSettings()
    bundle = default_contract_bundle(settings).model_dump()
    bundle_cfg = OmegaConf.create(bundle)
    if cfg.format != "json":
        raise ValueError(f"Unsupported format: {cfg.format}")
    payload = _serialize_bundle(bundle_cfg)
    if cfg.output_path:
        path = Path(cfg.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return
    print(payload)


if __name__ == "__main__":
    main()
