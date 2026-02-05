"""Backtest runner entry point.

Notes
-----
This is a lightweight example runner that uses the G2MAX-X synthetic
simulation as a placeholder for a real event-driven backtest engine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from g2max_x_lab import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf"


@dataclass(frozen=True)
class BacktestConfig:
    """Typed view of the example backtest configuration."""

    seed: int
    years: int
    output_path: str
    equity_curve_path: str
    export_equity_curve: bool


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _series_metrics(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy()
    cagr = values[-1] ** (252 / len(values)) - 1
    peak = np.maximum.accumulate(values)
    max_drawdown = float(np.max((peak - values) / peak))
    return {
        "final_equity": float(values[-1]),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
    }


def _write_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="backtest",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    config = BacktestConfig(
        seed=int(cfg.seed),
        years=int(cfg.years),
        output_path=str(cfg.output_path),
        equity_curve_path=str(cfg.equity_curve_path),
        export_equity_curve=bool(cfg.export_equity_curve),
    )
    eq, bh = run_simulation(seed=config.seed, years=config.years)

    report = {
        "run_id": f"g2maxx-{config.seed}-{int(datetime.now(timezone.utc).timestamp())}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": pd.Timestamp(eq.index[0]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(eq.index[-1]).strftime("%Y-%m-%d"),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": {
            "g2max_x": _series_metrics(eq),
            "buy_hold": _series_metrics(bh),
        },
    }

    output_path = _resolve_path(config.output_path)
    _write_report(report, output_path)

    equity_curve_path = _resolve_path(config.equity_curve_path)
    if config.export_equity_curve:
        equity_curve_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"G2MAX_X": eq, "BuyHold": bh}).to_csv(
            equity_curve_path,
            index_label="date",
        )

    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    if config.export_equity_curve:
        print(f"Equity curve: {equity_curve_path}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
