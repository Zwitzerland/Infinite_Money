"""Bare-bones AI pipeline wiring."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import json
import logging

import pandas as pd

from omegaconf import OmegaConf

from .config import load_config
from .data.market import fetch_market_data
from .data.filings import fetch_filings
from .data.news import dedupe_articles, fetch_news
from .features import build_feature_frame
from .features.text_embeddings import embed_texts
from .labels import direction_label, forward_returns, volatility_adjusted_returns
from .training import run_training
from .portfolio.g2max import G2MaxParams, g2max_exposure

logger = logging.getLogger(__name__)


def _create_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("ai_%Y%m%d_%H%M%S")
    output_dir = root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _summarize(step: str, summary: dict[str, Any]) -> None:
    summary.setdefault("steps", []).append(step)


def run_pipeline(
    config_path: str = "hedge_fund/conf/ai_stack.yaml",
    output_root: str = "artifacts/ai_runs",
) -> Path:
    """Run the minimal AI pipeline and write artifacts."""
    cfg = load_config(config_path)
    output_dir = _create_run_dir(output_root)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    summary: dict[str, Any] = {
        "run_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "status": "stub",
    }

    news_cfg: Mapping[str, Any] = cfg.get("news", {})
    if news_cfg.get("enabled", False):
        try:
            articles = dedupe_articles(fetch_news(news_cfg))
            summary["news_articles"] = len(articles)
            _summarize("news_ingest", summary)
            embed_cfg: Mapping[str, Any] = cfg.get("embeddings", {})
            if embed_cfg.get("enabled", False) and articles:
                texts = [f"{item.title}\n{item.body}" for item in articles]
                embeddings = embed_texts(
                    texts,
                    model_name=str(embed_cfg.get("model_name")),
                )
                summary["news_embeddings"] = len(embeddings)
                _summarize("text_embeddings", summary)
        except Exception as exc:  # pragma: no cover - external I/O
            logger.exception("News ingest failed")
            summary["news_error"] = str(exc)

    filings_cfg: Mapping[str, Any] = cfg.get("filings", {})
    if filings_cfg.get("enabled", False):
        try:
            filings = fetch_filings(filings_cfg)
            summary["filings_count"] = len(filings)
            _summarize("filings_ingest", summary)
            if filings:
                sample = [event.__dict__ for event in filings[:5]]
                (output_dir / "filings_sample.json").write_text(
                    json.dumps(sample, indent=2)
                )
        except Exception as exc:  # pragma: no cover - external I/O
            logger.exception("Filings ingest failed")
            summary["filings_error"] = str(exc)

    market_cfg: Mapping[str, Any] = cfg.get("market_data", {})
    if market_cfg.get("enabled", False):
        try:
            bars = fetch_market_data(market_cfg)
            summary["market_bars"] = len(bars)
            _summarize("market_data", summary)

            feature_cfg: Mapping[str, Any] = cfg.get("features", {})
            feature_frame = build_feature_frame(bars, feature_cfg)
            summary["feature_rows"] = len(feature_frame)
            _summarize("features", summary)

            label_cfg: Mapping[str, Any] = cfg.get("labels", {})
            horizon = int(label_cfg.get("horizon", 5))
            close: pd.Series = pd.Series(
                feature_frame["close"].to_numpy(),
                index=feature_frame.index,
                dtype=float,
            )
            summary["label_horizon"] = horizon
            summary["labels_forward"] = forward_returns(
                close, horizon=horizon
            ).dropna().shape[0]
            summary["labels_vol_adjusted"] = volatility_adjusted_returns(
                close,
                horizon=horizon,
                vol_window=int(label_cfg.get("vol_window", 20)),
            ).dropna().shape[0]
            summary["labels_direction"] = direction_label(
                close, horizon=horizon
            ).dropna().shape[0]
            _summarize("labels", summary)

            signal_cfg: Mapping[str, Any] = cfg.get("signal_export", {})
            if str(signal_cfg.get("method", "rule")) == "g2max":
                g2max_cfg: Mapping[str, Any] = signal_cfg.get("g2max", {})
                params = G2MaxParams(
                    phi_base=float(g2max_cfg.get("phi_base", 0.4)),
                    vol_target=float(g2max_cfg.get("vol_target", 0.14)),
                    drawdown_soft=float(g2max_cfg.get("drawdown_soft", 0.10)),
                    drawdown_hard=float(g2max_cfg.get("drawdown_hard", 0.20)),
                    leverage=float(g2max_cfg.get("leverage", 2.5)),
                    ewma_lambda=float(g2max_cfg.get("ewma_lambda", 0.94)),
                    lookback=int(g2max_cfg.get("lookback", 60)),
                )
                returns = close.pct_change().dropna()
                exposures = g2max_exposure(returns.to_numpy(), params)
                exposure_series = pd.Series(exposures, index=returns.index, name="exposure")
                exposure_series.to_csv(output_dir / "g2max_exposure.csv")
                summary["g2max_exposures"] = len(exposure_series)
                _summarize("g2max_exposure", summary)

            training_cfg: Mapping[str, Any] = cfg.get("training", {})
            if training_cfg.get("enabled", False):
                labels = cast(pd.Series, pd.Series(forward_returns(close, horizon=horizon)))
                training_summary = run_training(
                    feature_frame,
                    labels.dropna(),
                    training_cfg,
                    output_dir,
                )
                summary["training"] = training_summary
                _summarize("training", summary)
        except Exception as exc:  # pragma: no cover - external I/O
            logger.exception("Market data ingest failed")
            summary["market_error"] = str(exc)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return output_dir
