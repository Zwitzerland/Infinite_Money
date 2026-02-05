"""Export AI signals for LEAN custom data ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import csv
import pandas as pd

from ..config import load_config
from ..data import fetch_market_data
from ..features import build_feature_frame
from ..labels import forward_returns
from ..models import GBDTForecaster, LinearForecaster
from ..portfolio.alpha_sleeves import build_alpha_signal
from ..portfolio.g2max import G2MaxParams, g2max_exposure
from ...alpha.sr import SRBarrierParams, compute_sr_barrier_exposure


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for exporting signals."""

    method: str
    output_path: Path
    min_train: int = 252
    retrain_every: int = 20
    rule_window: int = 20
    model: str = "linear"
    threshold: float = 0.0


def _rule_based_signal(features: pd.DataFrame, window: int) -> pd.Series:
    close = pd.Series(
        features["close"].to_numpy(),
        index=features.index,
        dtype=float,
    )
    sma = close.rolling(window=window, min_periods=window).mean()
    signal = (close > sma).astype(int).replace({0: -1})
    return signal.dropna()


def _walkforward_predict(
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    min_train: int,
    retrain_every: int,
) -> pd.Series:
    if len(labels) < min_train:
        raise ValueError("Not enough data for walk-forward training")

    feature_matrix = features.to_numpy(dtype=float)
    label_array = labels.to_numpy(dtype=float)
    predictions: list[float] = []
    index = []

    model: LinearForecaster | GBDTForecaster | None = None
    for idx in range(min_train, len(labels)):
        if model is None or (idx - min_train) % retrain_every == 0:
            if model_name == "gbdt":
                model = GBDTForecaster()
            else:
                model = LinearForecaster()
            model.fit(feature_matrix[:idx], label_array[:idx])

        assert model is not None

        pred = model.predict([feature_matrix[idx]])[0]
        predictions.append(float(pred))
        index.append(labels.index[idx])

    return pd.Series(predictions, index=index)


def _g2max_signal(
    features: pd.DataFrame,
    params: G2MaxParams,
    alpha_mode: str,
    alpha_weight: float,
    sleeve_windows: list[int],
) -> pd.Series:
    close = pd.Series(
        features["close"].to_numpy(),
        index=features.index,
        dtype=float,
    )
    returns = close.pct_change().dropna()
    alpha_returns = None
    if alpha_mode == "sleeves":
        alpha_series = build_alpha_signal(close, sleeve_windows)
        alpha_series = alpha_series.reindex(returns.index).fillna(0.0)
        alpha_returns = alpha_series.to_numpy()
    exposures = g2max_exposure(
        returns.to_numpy(),
        params,
        alpha_returns=alpha_returns,
        alpha_weight=alpha_weight,
    )
    return pd.Series(exposures, index=returns.index)


def export_signals(config_path: str, output_root: str | Path) -> Path:
    cfg = load_config(config_path)
    export_cfg: Mapping[str, Any] = cfg.get("signal_export", {})
    method = str(export_cfg.get("method", "rule"))
    output_path = Path(export_cfg.get("output_path", "data/custom/ai_signals.csv"))
    output_path = (Path(output_root) / output_path).resolve() if not output_path.is_absolute() else output_path

    market_cfg: Mapping[str, Any] = cfg.get("market_data", {})
    bars = fetch_market_data(market_cfg)
    features = build_feature_frame(bars, cfg.get("features", {}))

    if "close" not in features.columns:
        raise ValueError("Feature frame missing close column")

    if method == "rule":
        window = int(export_cfg.get("rule_window", 20))
        signals = _rule_based_signal(features, window)
    elif method == "model":
        horizon = int(cfg.get("labels", {}).get("horizon", 5))
        close = pd.Series(
            features["close"].to_numpy(),
            index=features.index,
            dtype=float,
        )
        labels = pd.Series(forward_returns(close, horizon=horizon)).dropna()
        aligned_features = features.loc[labels.index]
        signals = _walkforward_predict(
            aligned_features,
            labels,
            model_name=str(export_cfg.get("model", "linear")),
            min_train=int(export_cfg.get("min_train", 252)),
            retrain_every=int(export_cfg.get("retrain_every", 20)),
        )
    elif method == "g2max":
        g2max_cfg: Mapping[str, Any] = export_cfg.get("g2max", {})
        g2max_params = G2MaxParams(
            phi_base=float(g2max_cfg.get("phi_base", 0.4)),
            vol_target=float(g2max_cfg.get("vol_target", 0.14)),
            drawdown_soft=float(g2max_cfg.get("drawdown_soft", 0.10)),
            drawdown_hard=float(g2max_cfg.get("drawdown_hard", 0.20)),
            leverage=float(g2max_cfg.get("leverage", 2.5)),
            ewma_lambda=float(g2max_cfg.get("ewma_lambda", 0.94)),
            lookback=int(g2max_cfg.get("lookback", 60)),
        )
        alpha_mode = str(g2max_cfg.get("alpha_mode", "none")).lower()
        alpha_weight = float(g2max_cfg.get("alpha_weight", 0.4))
        sleeve_windows = [int(x) for x in g2max_cfg.get("sleeve_windows", [20, 60, 120])]
        signals = _g2max_signal(features, g2max_params, alpha_mode, alpha_weight, sleeve_windows)
    elif method == "sr_barrier":
        sr_cfg: Mapping[str, Any] = export_cfg.get("sr_barrier", {})
        sr_params = SRBarrierParams(
            pivot_lookback=int(sr_cfg.get("pivot_lookback", 5)),
            train_window=int(sr_cfg.get("train_window", 252)),
            horizon=int(sr_cfg.get("horizon", 10)),
            zone_atr=float(sr_cfg.get("zone_atr", 0.6)),
            tp_atr=float(sr_cfg.get("tp_atr", 1.0)),
            sl_atr=float(sr_cfg.get("sl_atr", 1.0)),
            cost_atr=float(sr_cfg.get("cost_atr", 0.05)),
            level_source=str(sr_cfg.get("level_source", "pivots")),
            round_atr_mult=float(sr_cfg.get("round_atr_mult", 4.0)),
            min_resolved_events=int(sr_cfg.get("min_resolved_events", 25)),
            confidence=float(sr_cfg.get("confidence", 0.95)),
            kelly_fraction=float(sr_cfg.get("kelly_fraction", 0.25)),
            max_exposure=float(sr_cfg.get("max_exposure", 1.0)),
            use_regime_filter=bool(sr_cfg.get("use_regime_filter", True)),
        )
        signals = compute_sr_barrier_exposure(features, sr_params)
    else:
        raise ValueError(f"Unsupported signal_export.method: {method}")

    threshold = float(export_cfg.get("threshold", 0.0))
    signals = signals.apply(lambda value: 0 if abs(value) < threshold else value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = market_cfg.get("symbols", [])
    symbol = str(export_cfg.get("symbol") or (symbols[0] if symbols else "SPY"))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "symbol", "signal"])
        export_df = signals.to_frame(name="signal").reset_index()
        index_col = export_df.columns[0]
        export_df["timestamp"] = export_df[index_col].astype(str)
        export_df["symbol"] = symbol
        for row in export_df[["timestamp", "signal"]].itertuples(index=False, name=None):
            writer.writerow([row[0], symbol, float(row[1])])
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export AI signals to CSV")
    parser.add_argument(
        "--config",
        default="hedge_fund/conf/ai_stack.yaml",
        help="Path to the AI stack config file.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Root directory for relative output paths.",
    )
    args = parser.parse_args()
    path = export_signals(args.config, args.output_root)
    print(f"Signals written to {path}")


if __name__ == "__main__":
    main()
