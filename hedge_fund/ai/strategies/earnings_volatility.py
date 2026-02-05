"""Rule-based earnings volatility strategy signals."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..data.earnings_options import EarningsOptionSnapshot


@dataclass(frozen=True)
class EarningsVolParams:
    """Thresholds for selecting earnings volatility trades."""

    min_days_to_earnings: int = 5
    max_days_to_earnings: int = 20
    min_iv_rv: float = 1.25
    min_volume: float = 250.0
    min_open_interest: float = 500.0
    max_bid_ask_spread: float = 0.25
    min_front_dte: int = 7
    max_front_dte: int = 30
    min_back_dte: int = 30
    min_term_inversion: float = 0.05
    min_term_contango: float = 0.05


@dataclass(frozen=True)
class EarningsVolDecision:
    """Decision output for the earnings volatility strategy."""

    symbol: str
    asof_date: datetime
    earnings_date: datetime
    action: str
    reason: str
    iv_rv_ratio: float
    term_slope: float
    days_to_earnings: int


def _days_between(start: datetime, end: datetime) -> int:
    return int((end - start).days)


def _iv_rv_ratio(iv_front: float, rv_20d: float) -> float:
    if rv_20d <= 0:
        return 0.0
    return iv_front / rv_20d


def _term_slope(iv_front: float, iv_back: float) -> float:
    if iv_front == 0:
        return 0.0
    return (iv_back - iv_front) / iv_front


def build_earnings_vol_signal(
    snapshot: EarningsOptionSnapshot,
    params: EarningsVolParams,
) -> EarningsVolDecision:
    """Build a strategy decision from a single earnings snapshot."""
    days_to_earnings = _days_between(snapshot.asof_date, snapshot.earnings_date)
    iv_rv = _iv_rv_ratio(snapshot.iv_front, snapshot.rv_20d)
    term_slope = _term_slope(snapshot.iv_front, snapshot.iv_back)

    if days_to_earnings < params.min_days_to_earnings:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "too_close_to_earnings",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if days_to_earnings > params.max_days_to_earnings:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "too_far_from_earnings",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if snapshot.front_dte < params.min_front_dte:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "front_dte_too_short",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if snapshot.front_dte > params.max_front_dte:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "front_dte_too_long",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if snapshot.back_dte < params.min_back_dte:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "back_dte_too_short",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if snapshot.option_volume < params.min_volume:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "volume_too_low",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if snapshot.open_interest < params.min_open_interest:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "open_interest_too_low",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if snapshot.bid_ask_spread > params.max_bid_ask_spread:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "spread_too_wide",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if iv_rv < params.min_iv_rv:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "skip",
            "iv_rv_too_low",
            iv_rv,
            term_slope,
            days_to_earnings,
        )

    if term_slope <= -params.min_term_inversion:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "short_straddle",
            "front_iv_inverted",
            iv_rv,
            term_slope,
            days_to_earnings,
        )
    if term_slope >= params.min_term_contango:
        return EarningsVolDecision(
            snapshot.symbol,
            snapshot.asof_date,
            snapshot.earnings_date,
            "calendar_spread",
            "back_iv_premium",
            iv_rv,
            term_slope,
            days_to_earnings,
        )

    return EarningsVolDecision(
        snapshot.symbol,
        snapshot.asof_date,
        snapshot.earnings_date,
        "skip",
        "term_structure_flat",
        iv_rv,
        term_slope,
        days_to_earnings,
    )
