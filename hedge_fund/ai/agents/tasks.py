"""Built-in agent tasks for the AI loop."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import json
import pandas as pd

from ..data import fetch_market_data
from ..features import build_feature_frame
from ..integration.lean_export import export_signals
from ..labels import forward_returns
from ..training import run_training
from .orchestrator import AgentContext


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


@dataclass
class DataAgent:
    name: str = "data_agent"

    def run(self, context: AgentContext) -> None:
        cfg = context.config.get("market_data", {})
        summary = {"agent": self.name, "enabled": cfg.get("enabled", False)}
        if not cfg.get("enabled", False):
            _write_json(context.artifacts_dir / "data_agent.json", summary)
            return
        bars = fetch_market_data(cfg)
        summary["bar_count"] = len(bars)
        summary["timestamp"] = datetime.now(timezone.utc).isoformat()
        _write_json(context.artifacts_dir / "data_agent.json", summary)


@dataclass
class FeatureAgent:
    name: str = "feature_agent"

    def run(self, context: AgentContext) -> None:
        cfg = context.config.get("market_data", {})
        summary = {"agent": self.name, "enabled": cfg.get("enabled", False)}
        if not cfg.get("enabled", False):
            _write_json(context.artifacts_dir / "feature_agent.json", summary)
            return
        bars = fetch_market_data(cfg)
        features = build_feature_frame(bars, context.config.get("features", {}))
        summary["feature_rows"] = len(features)
        summary["feature_cols"] = len(features.columns)
        _write_json(context.artifacts_dir / "feature_agent.json", summary)


@dataclass
class ModelAgent:
    name: str = "model_agent"

    def run(self, context: AgentContext) -> None:
        train_cfg = context.config.get("training", {})
        summary = {"agent": self.name, "enabled": train_cfg.get("enabled", False)}
        if not train_cfg.get("enabled", False):
            _write_json(context.artifacts_dir / "model_agent.json", summary)
            return
        bars = fetch_market_data(context.config.get("market_data", {}))
        features = build_feature_frame(bars, context.config.get("features", {}))
        close = pd.Series(features["close"].to_numpy(), index=features.index)
        horizon = int(context.config.get("labels", {}).get("horizon", 5))
        labels = pd.Series(forward_returns(close, horizon=horizon)).dropna()
        summary["training"] = run_training(features, labels, train_cfg, context.artifacts_dir)
        _write_json(context.artifacts_dir / "model_agent.json", summary)


@dataclass
class BacktestAgent:
    name: str = "backtest_agent"

    def run(self, context: AgentContext) -> None:
        summary = {"agent": self.name, "enabled": True}
        path = export_signals(
            config_path=str(context.config.get("config_path", "hedge_fund/conf/ai_stack.yaml")),
            output_root=".",
        )
        summary["signals_path"] = str(path)
        _write_json(context.artifacts_dir / "backtest_agent.json", summary)


@dataclass
class OptimizationAgent:
    name: str = "optimization_agent"

    def run(self, context: AgentContext) -> None:
        summary = {
            "agent": self.name,
            "enabled": False,
            "note": "Use optimizer/study_optuna.py or LEAN optimize manually.",
        }
        _write_json(context.artifacts_dir / "optimization_agent.json", summary)


@dataclass
class QuantumAgent:
    name: str = "quantum_agent"

    def run(self, context: AgentContext) -> None:
        cfg = context.config.get("quantum", {})
        summary = {"agent": self.name, "enabled": cfg.get("enabled", False)}
        summary["note"] = "Quantum runs are disabled by default to avoid cost."
        _write_json(context.artifacts_dir / "quantum_agent.json", summary)


@dataclass
class ExecutionAgent:
    name: str = "execution_agent"

    def run(self, context: AgentContext) -> None:
        summary = {
            "agent": self.name,
            "enabled": False,
            "note": "Execution disabled. Enable only after paper trading validation.",
        }
        _write_json(context.artifacts_dir / "execution_agent.json", summary)
