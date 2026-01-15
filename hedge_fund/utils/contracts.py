"""Contract definitions for data, backtests, promotion, and execution."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from hedge_fund.utils.settings import PlatformSettings


class DataContract(BaseModel):
    """Defines required schemas and data hygiene constraints."""

    schema_version: str = Field(default="1.0.0")
    event_schemas: tuple[str, ...] = Field(
        default=("bars", "trades", "corporate_actions", "orders", "fills"),
    )
    required_partitions: tuple[str, ...] = Field(
        default=("symbol", "date", "event_type"),
    )
    timezone: str = Field(default="UTC")
    enforce_point_in_time: bool = Field(default=True)
    checksum_algorithm: Literal["sha256", "blake2"] = Field(default="sha256")


class BacktestContract(BaseModel):
    """Defines the minimum standard for a valid backtest artifact."""

    min_history_days: int = Field(default=365, ge=30)
    embargo_days: int = Field(default=5, ge=0)
    cpcv_splits: int = Field(default=5, ge=2)
    max_drawdown: float = Field(default=0.25, ge=0.0, le=1.0)
    required_metrics: tuple[str, ...] = Field(
        default=("sharpe", "sortino", "calmar", "drawdown", "turnover"),
    )
    artifact_fields: tuple[str, ...] = Field(
        default=(
            "run_id",
            "start",
            "end",
            "params",
            "metrics",
            "seed",
            "environment",
        ),
    )


class PromotionContract(BaseModel):
    """Promotion gates for research -> paper -> live."""

    min_deflated_sharpe: float = Field(default=1.0)
    max_overfit_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    min_oos_windows: int = Field(default=3, ge=1)
    min_paper_days: int = Field(default=30, ge=1)
    require_regime_tests: bool = Field(default=True)
    require_stress_tests: bool = Field(default=True)


class ExecutionContract(BaseModel):
    """Execution pacing and risk constraints."""

    max_orders_per_sec: int = Field(default=50, ge=1)
    max_position_turnover: float = Field(default=1.5, ge=0.0)
    max_leverage: float = Field(default=2.0, ge=0.0)
    max_notional_per_order: float = Field(default=1_000_000.0, ge=0.0)
    ibkr_web_api_rps_limit: int = Field(default=50, ge=1)
    ibkr_client_portal_rps_limit: int = Field(default=10, ge=1)
    ibkr_tws_messages_per_sec: int = Field(default=50, ge=1)


class ContractBundle(BaseModel):
    """Full set of immutable contracts."""

    data: DataContract
    backtest: BacktestContract
    promotion: PromotionContract
    execution: ExecutionContract


def default_contract_bundle(settings: PlatformSettings) -> ContractBundle:
    """Create contracts using platform settings."""
    return ContractBundle(
        data=DataContract(),
        backtest=BacktestContract(),
        promotion=PromotionContract(
            min_deflated_sharpe=settings.min_deflated_sharpe,
            max_overfit_probability=settings.max_overfit_probability,
            min_oos_windows=settings.min_oos_windows,
            min_paper_days=settings.min_paper_days,
        ),
        execution=ExecutionContract(
            max_orders_per_sec=settings.max_order_rate_per_sec,
            max_position_turnover=settings.max_position_turnover,
            max_leverage=settings.max_leverage,
            max_notional_per_order=settings.max_notional_per_order,
            ibkr_web_api_rps_limit=settings.ibkr_web_api_rps_limit,
            ibkr_client_portal_rps_limit=settings.ibkr_client_portal_rps_limit,
            ibkr_tws_messages_per_sec=settings.ibkr_tws_messages_per_sec,
        ),
    )
