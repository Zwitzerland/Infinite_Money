"""Support/Resistance barrier (SRB) signal generator.

This module is intentionally *not* a promise of profitability.

It encodes the core SR thesis in a testable way:
- Define SR levels algorithmically (confirmed pivots).
- Define a bracket (TP/SL) around the level.
- Estimate the conditional first-hitting win rate p(a).
- Only emit exposure when the *conservative* lower confidence bound on p(a)
  exceeds the break-even threshold implied by TP/SL and costs.

Key design choice
-----------------
Outcomes are recorded at the *resolution time* (first time TP or SL is hit),
not at the entry time. This prevents leakage where future outcomes would be
counted before they are observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log10, sqrt
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


Side = Literal["long", "short"]
LevelSource = Literal["pivots", "rounds", "avwap"]


@dataclass(frozen=True)
class SRBarrierParams:
    """Parameters for the SR barrier strategy.

    All distances are expressed as multiples of ATR so that the break-even
    probability and reward/risk ratio are invariant to the price scale.
    """

    pivot_lookback: int = 5
    train_window: int = 252
    horizon: int = 10
    zone_atr: float = 0.6
    tp_atr: float = 1.0
    sl_atr: float = 1.0
    cost_atr: float = 0.05
    level_source: LevelSource = "pivots"
    round_atr_mult: float = 4.0
    min_resolved_events: int = 25
    confidence: float = 0.95
    kelly_fraction: float = 0.25
    max_exposure: float = 1.0
    use_regime_filter: bool = True

    def __post_init__(self) -> None:
        if self.pivot_lookback < 1:
            raise ValueError("pivot_lookback must be >= 1")
        if self.train_window < 10:
            raise ValueError("train_window must be >= 10")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.zone_atr <= 0:
            raise ValueError("zone_atr must be > 0")
        if self.tp_atr <= 0 or self.sl_atr <= 0:
            raise ValueError("tp_atr and sl_atr must be > 0")
        if self.cost_atr < 0:
            raise ValueError("cost_atr must be >= 0")
        if self.round_atr_mult <= 0:
            raise ValueError("round_atr_mult must be > 0")
        if not (0 < self.confidence < 1):
            raise ValueError("confidence must be in (0, 1)")
        if not (0 <= self.kelly_fraction <= 1):
            raise ValueError("kelly_fraction must be in [0, 1]")
        if self.max_exposure <= 0:
            raise ValueError("max_exposure must be > 0")

    @property
    def reward_risk(self) -> float:
        return float(self.tp_atr / self.sl_atr)

    @property
    def p0_martingale(self) -> float:
        """Martingale baseline for two-barrier hitting (driftless diffusion)."""
        return float(self.sl_atr / (self.tp_atr + self.sl_atr))

    @property
    def p_break_even(self) -> float:
        """Break-even probability for the TP/SL/cost bracket.

        p* = (l + c) / (u + l)
        with u = tp_atr * ATR, l = sl_atr * ATR, c = cost_atr * ATR.

        ATR cancels, so p* depends only on ATR multiples.
        """
        return float((self.sl_atr + self.cost_atr) / (self.tp_atr + self.sl_atr))


@dataclass(frozen=True)
class SRBarrierResult:
    """Full SR barrier output, including diagnostics useful for research."""

    exposure: pd.Series
    support: pd.Series
    resistance: pd.Series
    support_stats: pd.DataFrame
    resistance_stats: pd.DataFrame
    entry_support: pd.Series
    entry_resistance: pd.Series
    resolved_entries_support: pd.Series
    resolved_wins_support: pd.Series
    resolved_entries_resistance: pd.Series
    resolved_wins_resistance: pd.Series


def _wilson_lower_bound(k: float, n: float, confidence: float) -> float:
    """Wilson score interval lower bound for a binomial proportion."""
    if n <= 0:
        return float("nan")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = k / n
    denom = 1.0 + (z**2) / n
    center = phat + (z**2) / (2.0 * n)
    margin = z * sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * n)) / n)
    return float((center - margin) / denom)


def _confirmed_pivots(
    high: NDArray[np.floating[Any]],
    low: NDArray[np.floating[Any]],
    lookback: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return pivot-high / pivot-low prices confirmed with delay.

    A pivot at index `p` is confirmed at time `t = p + lookback` once we have
    observed the forward window. This makes the pivot definition causal with
    a fixed confirmation delay.
    """
    n = high.shape[0]
    piv_hi = np.full(n, np.nan, dtype=float)
    piv_lo = np.full(n, np.nan, dtype=float)

    window = 2 * lookback + 1
    if n < window:
        return piv_hi, piv_lo

    for t in range(window - 1, n):
        pivot = t - lookback
        start = t - (window - 1)
        stop = t + 1
        hi_win = high[start:stop]
        lo_win = low[start:stop]
        pivot_hi = high[pivot]
        pivot_lo = low[pivot]
        if np.isfinite(pivot_hi) and pivot_hi == float(np.nanmax(hi_win)):
            piv_hi[t] = float(pivot_hi)
        if np.isfinite(pivot_lo) and pivot_lo == float(np.nanmin(lo_win)):
            piv_lo[t] = float(pivot_lo)
    return piv_hi, piv_lo


def _support_resistance_series(
    close: NDArray[np.floating[Any]],
    high: NDArray[np.floating[Any]],
    low: NDArray[np.floating[Any]],
    lookback: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    piv_hi, piv_lo = _confirmed_pivots(high, low, lookback)
    n = close.shape[0]

    support = np.full(n, np.nan, dtype=float)
    resistance = np.full(n, np.nan, dtype=float)
    last_support = float("nan")
    last_resistance = float("nan")
    for t in range(n):
        if np.isfinite(piv_lo[t]):
            last_support = float(piv_lo[t])
        if np.isfinite(piv_hi[t]):
            last_resistance = float(piv_hi[t])
        support[t] = last_support
        resistance[t] = last_resistance
    return support, resistance


def _nice_step(value: float) -> float:
    """Return a 1/2/5 * 10^k step close to `value`."""
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    exponent = floor(log10(value))
    fraction = value / (10**exponent)
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10**exponent))


def _round_number_levels(
    close: NDArray[np.floating[Any]],
    atr: NDArray[np.floating[Any]],
    atr_mult: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute pseudo-round support/resistance levels using a nice-number grid."""
    n = close.shape[0]
    support = np.full(n, np.nan, dtype=float)
    resistance = np.full(n, np.nan, dtype=float)
    for t in range(n):
        step = _nice_step(float(atr[t]) * float(atr_mult))
        if not np.isfinite(step) or step <= 0 or not np.isfinite(close[t]):
            continue
        base = floor(float(close[t]) / step) * step
        support[t] = float(base)
        resistance[t] = float(base + step)
    return support, resistance


def _anchored_vwap_levels(
    close: NDArray[np.floating[Any]],
    high: NDArray[np.floating[Any]],
    low: NDArray[np.floating[Any]],
    volume: NDArray[np.floating[Any]],
    lookback: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute anchored VWAP levels (support anchored to pivot lows, resistance to pivot highs)."""

    piv_hi, piv_lo = _confirmed_pivots(high, low, lookback)
    typical = (high + low + close) / 3.0
    n = close.shape[0]

    support = np.full(n, np.nan, dtype=float)
    resistance = np.full(n, np.nan, dtype=float)

    active_s = False
    active_r = False
    sum_pv_s = 0.0
    sum_v_s = 0.0
    sum_pv_r = 0.0
    sum_v_r = 0.0

    for t in range(n):
        if np.isfinite(piv_lo[t]):
            active_s = True
            sum_pv_s = 0.0
            sum_v_s = 0.0
        if np.isfinite(piv_hi[t]):
            active_r = True
            sum_pv_r = 0.0
            sum_v_r = 0.0

        px = float(typical[t])
        vol = float(volume[t])
        if not np.isfinite(px) or not np.isfinite(vol) or vol <= 0:
            continue

        if active_s:
            sum_pv_s += px * vol
            sum_v_s += vol
            support[t] = float(sum_pv_s / sum_v_s) if sum_v_s > 0 else float("nan")
        if active_r:
            sum_pv_r += px * vol
            sum_v_r += vol
            resistance[t] = float(sum_pv_r / sum_v_r) if sum_v_r > 0 else float("nan")

    return support, resistance


def _entry_mask(
    price: NDArray[np.floating[Any]],
    level: NDArray[np.floating[Any]],
    zone: NDArray[np.floating[Any]],
) -> NDArray[np.bool_]:
    in_zone = np.isfinite(level) & np.isfinite(zone) & (np.abs(price - level) <= zone)
    prev = np.concatenate(([False], in_zone[:-1]))
    return np.asarray(in_zone & ~prev, dtype=bool)


def _resolve_brackets(
    *,
    entries: NDArray[np.bool_],
    level: NDArray[np.floating[Any]],
    high: NDArray[np.floating[Any]],
    low: NDArray[np.floating[Any]],
    atr: NDArray[np.floating[Any]],
    params: SRBarrierParams,
    side: Side,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return arrays of resolved_entries/resolved_wins at resolution times."""
    n = high.shape[0]
    resolved_entries = np.zeros(n, dtype=float)
    resolved_wins = np.zeros(n, dtype=float)

    entry_idx = np.flatnonzero(entries)
    for i in entry_idx:
        lvl = float(level[i])
        atr_i = float(atr[i])
        if not np.isfinite(lvl) or not np.isfinite(atr_i) or atr_i <= 0:
            continue

        u = params.tp_atr * atr_i
        sl_dist = params.sl_atr * atr_i
        if side == "long":
            tp = lvl + u
            sl = lvl - sl_dist
            for j in range(i + 1, min(i + params.horizon, n - 1) + 1):
                stop_hit = low[j] <= sl
                tp_hit = high[j] >= tp
                if not (stop_hit or tp_hit):
                    continue
                resolved_entries[j] += 1.0
                # Conservative: if both hit in same bar -> treat as stop.
                resolved_wins[j] += 1.0 if (tp_hit and not stop_hit) else 0.0
                break
        else:
            tp = lvl - u
            sl = lvl + sl_dist
            for j in range(i + 1, min(i + params.horizon, n - 1) + 1):
                stop_hit = high[j] >= sl
                tp_hit = low[j] <= tp
                if not (stop_hit or tp_hit):
                    continue
                resolved_entries[j] += 1.0
                resolved_wins[j] += 1.0 if (tp_hit and not stop_hit) else 0.0
                break

    return resolved_entries, resolved_wins


def _rolling_probabilities(
    *,
    resolved_entries: pd.Series,
    resolved_wins: pd.Series,
    params: SRBarrierParams,
) -> pd.DataFrame:
    roll_n = resolved_entries.rolling(params.train_window, min_periods=1).sum().shift(1)
    roll_k = resolved_wins.rolling(params.train_window, min_periods=1).sum().shift(1)

    def _lb(row: pd.Series) -> float:
        k = float(row["k"])
        n = float(row["n"])
        if n <= 0:
            return float("nan")
        return _wilson_lower_bound(k, n, params.confidence)

    frame = pd.DataFrame({"n": roll_n, "k": roll_k})
    frame["p_hat"] = frame["k"] / frame["n"].replace(0.0, np.nan)
    frame["p_low"] = frame.apply(_lb, axis=1)
    return frame


def _kelly_fraction(p: float, b: float) -> float:
    """Kelly fraction for a binary payoff: +b on win, -1 on loss."""
    if b <= 0:
        raise ValueError("b must be > 0")
    return float((p * (b + 1.0) - 1.0) / b)


def compute_sr_barrier_exposure(features: pd.DataFrame, params: SRBarrierParams) -> pd.Series:
    return compute_sr_barrier_result(features, params).exposure


def compute_sr_barrier_result(features: pd.DataFrame, params: SRBarrierParams) -> SRBarrierResult:
    """Compute SR barrier exposure plus intermediate diagnostics.

    The input must contain `close`, `high`, `low`, and `atr_14`.
    """

    required = {"close", "high", "low", "atr_14"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature frame missing required columns: {sorted(missing)}")

    # Support a MultiIndex (symbol, timestamp) but require a single symbol.
    if isinstance(features.index, pd.MultiIndex):
        symbols = features.index.get_level_values(0).unique()
        if len(symbols) != 1:
            raise ValueError("sr_barrier requires market_data.symbols to contain exactly 1 symbol")
        frame = features.droplevel(0)
    else:
        frame = features

    frame = frame.sort_index()

    close = frame["close"].astype(float).to_numpy()
    high = frame["high"].astype(float).to_numpy()
    low = frame["low"].astype(float).to_numpy()
    atr = frame["atr_14"].astype(float).to_numpy()

    if params.level_source == "pivots":
        support_arr, resistance_arr = _support_resistance_series(
            close,
            high,
            low,
            params.pivot_lookback,
        )
    elif params.level_source == "rounds":
        support_arr, resistance_arr = _round_number_levels(close, atr, params.round_atr_mult)
    else:  # "avwap"
        if "volume" not in frame.columns:
            raise ValueError("sr_barrier level_source='avwap' requires a volume column")
        volume = frame["volume"].astype(float).to_numpy()
        support_arr, resistance_arr = _anchored_vwap_levels(
            close,
            high,
            low,
            volume,
            params.pivot_lookback,
        )
    zone = params.zone_atr * atr
    entry_support_arr = _entry_mask(close, support_arr, zone)
    entry_resistance_arr = _entry_mask(close, resistance_arr, zone)

    res_n_s, res_k_s = _resolve_brackets(
        entries=entry_support_arr,
        level=support_arr,
        high=high,
        low=low,
        atr=atr,
        params=params,
        side="long",
    )
    res_n_r, res_k_r = _resolve_brackets(
        entries=entry_resistance_arr,
        level=resistance_arr,
        high=high,
        low=low,
        atr=atr,
        params=params,
        side="short",
    )

    idx = frame.index
    resolved_entries_support = pd.Series(res_n_s, index=idx, name="resolved_entries_support")
    resolved_wins_support = pd.Series(res_k_s, index=idx, name="resolved_wins_support")
    resolved_entries_resistance = pd.Series(res_n_r, index=idx, name="resolved_entries_resistance")
    resolved_wins_resistance = pd.Series(res_k_r, index=idx, name="resolved_wins_resistance")

    support_stats = _rolling_probabilities(
        resolved_entries=resolved_entries_support,
        resolved_wins=resolved_wins_support,
        params=params,
    )
    resistance_stats = _rolling_probabilities(
        resolved_entries=resolved_entries_resistance,
        resolved_wins=resolved_wins_resistance,
        params=params,
    )

    p_star = params.p_break_even
    b = params.reward_risk

    exposure = pd.Series(0.0, index=idx, name="sr_exposure")

    regime = frame.get("regime")
    if regime is not None:
        regime_series = regime.astype(int)
    else:
        regime_series = pd.Series(0, index=idx, dtype=int)

    near_support = np.isfinite(support_arr) & np.isfinite(zone) & (np.abs(close - support_arr) <= zone)
    near_resistance = np.isfinite(resistance_arr) & np.isfinite(zone) & (np.abs(close - resistance_arr) <= zone)

    for t in range(len(idx)):
        if params.use_regime_filter and int(regime_series.iloc[t]) != 0:
            continue

        # Long bounce at support
        if bool(near_support[t]):
            n = float(support_stats.iloc[t]["n"])
            p_low = float(support_stats.iloc[t]["p_low"])
            if n >= params.min_resolved_events and np.isfinite(p_low) and p_low > p_star:
                f = params.kelly_fraction * max(
                    0.0,
                    _kelly_fraction(float(support_stats.iloc[t]["p_hat"]), b),
                )
                exposure.iloc[t] = float(min(params.max_exposure, max(-params.max_exposure, f)))
            continue

        # Short bounce at resistance
        if bool(near_resistance[t]):
            n = float(resistance_stats.iloc[t]["n"])
            p_low = float(resistance_stats.iloc[t]["p_low"])
            if n >= params.min_resolved_events and np.isfinite(p_low) and p_low > p_star:
                f = params.kelly_fraction * max(
                    0.0,
                    _kelly_fraction(float(resistance_stats.iloc[t]["p_hat"]), b),
                )
                exposure.iloc[t] = -float(min(params.max_exposure, max(-params.max_exposure, f)))
            continue

    return SRBarrierResult(
        exposure=exposure,
        support=pd.Series(support_arr, index=idx, name="support"),
        resistance=pd.Series(resistance_arr, index=idx, name="resistance"),
        support_stats=support_stats,
        resistance_stats=resistance_stats,
        entry_support=pd.Series(entry_support_arr, index=idx, name="entry_support"),
        entry_resistance=pd.Series(entry_resistance_arr, index=idx, name="entry_resistance"),
        resolved_entries_support=resolved_entries_support,
        resolved_wins_support=resolved_wins_support,
        resolved_entries_resistance=resolved_entries_resistance,
        resolved_wins_resistance=resolved_wins_resistance,
    )
