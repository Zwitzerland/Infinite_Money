"""Agent loop entry point."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from .orchestrator import AgentContext, AgentOrchestrator
from .tasks import (
    BacktestAgent,
    DataAgent,
    ExecutionAgent,
    FeatureAgent,
    ModelAgent,
    OptimizationAgent,
    QuantumAgent,
)


def run_agent_loop(config_path: str, output_root: str) -> Path:
    cfg = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    run_id = datetime.now(timezone.utc).strftime("agents_%Y%m%d_%H%M%S")
    artifacts_dir = Path(output_root) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "agent_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    agent_cfg = cfg.get("agents", {})
    agents = []
    if agent_cfg.get("data", True):
        agents.append(DataAgent())
    if agent_cfg.get("features", True):
        agents.append(FeatureAgent())
    if agent_cfg.get("model", True):
        agents.append(ModelAgent())
    if agent_cfg.get("backtest", True):
        agents.append(BacktestAgent())
    if agent_cfg.get("optimization", False):
        agents.append(OptimizationAgent())
    if agent_cfg.get("quantum", False):
        agents.append(QuantumAgent())
    if agent_cfg.get("execution", False):
        agents.append(ExecutionAgent())

    context = AgentContext(run_id=run_id, artifacts_dir=artifacts_dir, config=config_dict)
    orchestrator = AgentOrchestrator(agents)
    orchestrator.run(context)
    return artifacts_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the AI agent loop")
    parser.add_argument(
        "--config",
        default="hedge_fund/conf/agent_loop.yaml",
        help="Path to agent loop config.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/agent_runs",
        help="Root directory for agent artifacts.",
    )
    args = parser.parse_args()
    path = run_agent_loop(args.config, args.output_root)
    print(f"Agent loop artifacts: {path}")


if __name__ == "__main__":
    main()
