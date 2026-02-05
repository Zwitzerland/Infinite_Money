"""Chart rendering utilities for imctl."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class EquityCurve:
    timestamps: pd.DatetimeIndex
    values: pd.Series


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_backtest_result_json(output_dir: Path) -> Path:
    candidates = [path for path in output_dir.glob("*.json") if path.stem.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No numeric LEAN result JSON found in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_size)


def _extract_equity_curve(payload: dict) -> EquityCurve:
    charts = payload.get("Charts") or payload.get("charts") or {}
    chart = charts.get("Strategy Equity")
    if not isinstance(chart, dict):
        raise KeyError("Missing 'Strategy Equity' chart")

    series = chart.get("Series") or chart.get("series") or {}
    equity = series.get("Equity")
    if not isinstance(equity, dict):
        raise KeyError("Missing 'Equity' series")

    values = equity.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("Empty equity series")

    ts = [int(row[0]) for row in values if isinstance(row, list) and len(row) >= 2]
    val = [float(row[-1]) for row in values if isinstance(row, list) and len(row) >= 2]
    index = pd.to_datetime(ts, unit="s", utc=True)
    series_out = pd.Series(val, index=index).sort_index()
    return EquityCurve(timestamps=series_out.index, values=series_out)


def _extract_benchmark_curve(payload: dict, start_equity: float) -> EquityCurve | None:
    charts = payload.get("Charts") or payload.get("charts") or {}
    chart = charts.get("Benchmark")
    if not isinstance(chart, dict):
        return None
    series = chart.get("Series") or chart.get("series") or {}
    benchmark = series.get("Benchmark")
    if not isinstance(benchmark, dict):
        return None
    values = benchmark.get("values")
    if not isinstance(values, list) or not values:
        return None

    ts = [int(row[0]) for row in values if isinstance(row, list) and len(row) >= 2]
    px = [float(row[1]) for row in values if isinstance(row, list) and len(row) >= 2]
    idx = pd.to_datetime(ts, unit="s", utc=True)
    series_px = pd.Series(px, index=idx).sort_index()
    base = float(series_px.iloc[0])
    if base <= 0:
        return None
    equity = start_equity * (series_px / base)
    return EquityCurve(timestamps=equity.index, values=equity)


def render_lean_equity_chart(output_dir: Path) -> Path:
    """Render strategy vs benchmark equity curve from LEAN output JSON."""

    result_path = _find_backtest_result_json(output_dir)
    payload = _load_json(result_path)

    strategy = _extract_equity_curve(payload)
    start_equity = float(strategy.values.iloc[0])
    benchmark = _extract_benchmark_curve(payload, start_equity)

    plt.figure(figsize=(10, 5))
    plt.plot(strategy.timestamps.to_pydatetime(), strategy.values.to_numpy(), label="Strategy")
    if benchmark is not None:
        plt.plot(benchmark.timestamps.to_pydatetime(), benchmark.values.to_numpy(), label="Benchmark")

    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = output_dir / "equity_chart.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path
