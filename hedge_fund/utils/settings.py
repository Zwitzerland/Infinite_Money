"""Platform settings loaded from environment variables."""
from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    """Core platform settings with safe defaults."""

    model_config = SettingsConfigDict(env_prefix="INFINITE_MONEY_")

    ibkr_web_api_rps_limit: int = Field(
        default=50,
        ge=1,
        description="Interactive Brokers Web API global request limit per user.",
    )
    ibkr_client_portal_rps_limit: int = Field(
        default=10,
        ge=1,
        description="Interactive Brokers Client Portal Gateway request limit.",
    )
    ibkr_tws_messages_per_sec: int = Field(
        default=50,
        ge=1,
        description="TWS API message pacing limit.",
    )
    max_order_rate_per_sec: int = Field(
        default=50,
        ge=1,
        description="Max orders per second allowed by execution contract.",
    )
    max_position_turnover: float = Field(
        default=1.5,
        ge=0.0,
        description="Max daily turnover fraction of portfolio.",
    )
    max_leverage: float = Field(
        default=2.0,
        ge=0.0,
        description="Max portfolio leverage.",
    )
    max_notional_per_order: float = Field(
        default=1_000_000.0,
        ge=0.0,
        description="Absolute max notional per order.",
    )
    min_deflated_sharpe: float = Field(
        default=1.0,
        description="Minimum deflated Sharpe for promotion.",
    )
    max_overfit_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Max probability of overfitting.",
    )
    min_oos_windows: int = Field(
        default=3,
        ge=1,
        description="Minimum number of out-of-sample windows.",
    )
    min_paper_days: int = Field(
        default=30,
        ge=1,
        description="Minimum paper trading days before promotion.",
    )
    quantconnect_base_url: str = Field(
        default="https://www.quantconnect.com/api/v2",
        description="Base URL for QuantConnect REST API.",
    )
    quantconnect_api_token: SecretStr | None = Field(
        default=None,
        description="QuantConnect API token (inject via secrets manager).",
    )
