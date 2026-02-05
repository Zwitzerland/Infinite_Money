"""AWS pipeline orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from ..integration.lean_export import export_signals
from ..pipeline import run_pipeline
from .artifacts import upload_directory
from .step_functions import start_execution


@dataclass(frozen=True)
class AwsPipelineConfig:
    """Configuration for AWS pipeline runs."""

    config_path: str
    output_root: str
    s3_bucket: str | None
    s3_prefix: str | None
    state_machine_arn: str | None
    region: str | None
    profile: str | None


def _load_config(path: str) -> Mapping[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def run_local_pipeline(config: AwsPipelineConfig) -> Path:
    """Run pipeline locally and optionally upload artifacts to S3."""
    output_dir = run_pipeline(config_path=config.config_path, output_root=config.output_root)
    export_signals(config_path=config.config_path, output_root=".")

    if config.s3_bucket and config.s3_prefix:
        upload_directory(
            output_dir,
            bucket=config.s3_bucket,
            prefix=f"{config.s3_prefix.rstrip('/')}/{output_dir.name}",
            region=config.region,
            profile=config.profile,
        )
    return output_dir


def launch_step_functions(config: AwsPipelineConfig) -> str:
    """Launch a Step Functions pipeline execution."""
    if not config.state_machine_arn:
        raise ValueError("state_machine_arn is required")

    payload = {
        "config_path": config.config_path,
        "output_root": config.output_root,
        "s3_bucket": config.s3_bucket,
        "s3_prefix": config.s3_prefix,
    }
    return start_execution(
        config.state_machine_arn,
        payload,
        region=config.region,
        profile=config.profile,
    )


def load_pipeline_config(path: str) -> AwsPipelineConfig:
    cfg = _load_config(path)
    return AwsPipelineConfig(
        config_path=str(cfg.get("config_path", "hedge_fund/conf/ai_stack.yaml")),
        output_root=str(cfg.get("output_root", "artifacts/ai_runs")),
        s3_bucket=cfg.get("s3_bucket"),
        s3_prefix=cfg.get("s3_prefix"),
        state_machine_arn=cfg.get("state_machine_arn"),
        region=cfg.get("region"),
        profile=cfg.get("profile"),
    )
