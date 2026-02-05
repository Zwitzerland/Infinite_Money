"""Backtest runner for the earnings volatility strategy."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from hedge_fund.ai.data.earnings_options import (
    EarningsOptionSnapshot,
    fetch_earnings_option_snapshots,
)
from hedge_fund.ai.strategies.earnings_volatility import (
    EarningsVolParams,
    build_earnings_vol_signal,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf"


@dataclass(frozen=True)
class BacktestOutputPaths:
    report_path: str
    equity_curve_path: str
    trades_path: str


@dataclass(frozen=True)
class ExecutionParams:
    starting_equity: float
    risk_per_trade: float
    trade_cost: float
    short_straddle_scale: float
    calendar_scale: float
    pnl_cap: float
    pnl_floor: float


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _series_metrics(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy()
    if len(values) == 0:
        return {
            "final_equity": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
        }
    cagr = values[-1] ** (252 / len(values)) - 1
    peak = np.maximum.accumulate(values)
    max_drawdown = float(np.max((peak - values) / peak))
    return {
        "final_equity": float(values[-1]),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
    }


def _pnl_proxy(
    snapshot: EarningsOptionSnapshot,
    action: str,
    params: ExecutionParams,
) -> float:
    if action == "short_straddle":
        edge = snapshot.iv_front - snapshot.rv_20d
        move_scale = float(np.sqrt(snapshot.front_dte / 252.0))
        gross = edge * move_scale * params.short_straddle_scale
    elif action == "calendar_spread":
        term_slope = (snapshot.iv_back - snapshot.iv_front) / snapshot.iv_front
        move_scale = float(np.sqrt(snapshot.back_dte / 252.0))
        gross = term_slope * move_scale * params.calendar_scale
    else:
        return 0.0

    capped = max(min(gross, params.pnl_cap), -params.pnl_floor)
    return capped - params.trade_cost


def _iter_trades(
    snapshots: Iterable[EarningsOptionSnapshot],
    strategy: EarningsVolParams,
) -> list[tuple[EarningsOptionSnapshot, str]]:
    decisions = []
    for snapshot in snapshots:
        decision = build_earnings_vol_signal(snapshot, strategy)
        if decision.action != "skip":
            decisions.append((snapshot, decision.action))
    decisions.sort(key=lambda item: item[0].asof_date)
    return decisions


def _write_json(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="earnings_volatility",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    output = BacktestOutputPaths(
        report_path=str(cfg.output.report_path),
        equity_curve_path=str(cfg.output.equity_curve_path),
        trades_path=str(cfg.output.trades_path),
    )
    execution = ExecutionParams(
        starting_equity=float(cfg.execution.starting_equity),
        risk_per_trade=float(cfg.execution.risk_per_trade),
        trade_cost=float(cfg.execution.trade_cost),
        short_straddle_scale=float(cfg.execution.short_straddle_scale),
        calendar_scale=float(cfg.execution.calendar_scale),
        pnl_cap=float(cfg.execution.pnl_cap),
        pnl_floor=float(cfg.execution.pnl_floor),
    )
    strategy = EarningsVolParams(
        min_days_to_earnings=int(cfg.strategy.min_days_to_earnings),
        max_days_to_earnings=int(cfg.strategy.max_days_to_earnings),
        min_iv_rv=float(cfg.strategy.min_iv_rv),
        min_volume=float(cfg.strategy.min_volume),
        min_open_interest=float(cfg.strategy.min_open_interest),
        max_bid_ask_spread=float(cfg.strategy.max_bid_ask_spread),
        min_front_dte=int(cfg.strategy.min_front_dte),
        max_front_dte=int(cfg.strategy.max_front_dte),
        min_back_dte=int(cfg.strategy.min_back_dte),
        min_term_inversion=float(cfg.strategy.min_term_inversion),
        min_term_contango=float(cfg.strategy.min_term_contango),
    )

    snapshots = fetch_earnings_option_snapshots(cfg.data)
    trades = _iter_trades(snapshots, strategy)
    if not trades:
        raise ValueError("No qualifying trades found for the given filters")

    equity = []
    trade_rows = []
    eq = execution.starting_equity
    for snapshot, action in trades:
        trade_return = _pnl_proxy(snapshot, action, execution)
        eq *= 1 + execution.risk_per_trade * trade_return
        equity.append(eq)
        trade_rows.append(
            {
                "asof_date": snapshot.asof_date.isoformat(),
                "symbol": snapshot.symbol,
                "action": action,
                "iv_front": snapshot.iv_front,
                "iv_back": snapshot.iv_back,
                "rv_20d": snapshot.rv_20d,
                "front_dte": snapshot.front_dte,
                "back_dte": snapshot.back_dte,
                "trade_return": trade_return,
                "equity": eq,
            }
        )

    equity_series = pd.Series(equity, index=[row[0].asof_date for row in trades])
    metrics = _series_metrics(equity_series)
    report = {
        "run_id": f"earnings-vol-{int(datetime.now(timezone.utc).timestamp())}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "trades": len(trades),
    }

    report_path = _resolve_path(output.report_path)
    equity_path = _resolve_path(output.equity_curve_path)
    trades_path = _resolve_path(output.trades_path)
    _write_json(report, report_path)

    equity_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"equity": equity_series}).to_csv(
        equity_path,
        index_label="date",
    )
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trade_rows).to_csv(trades_path, index=False)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
