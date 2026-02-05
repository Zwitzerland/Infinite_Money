"""Lightweight agent orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentContext:
    """Shared context passed to agents."""

    run_id: str
    artifacts_dir: Path
    config: Mapping[str, object]


class Agent(Protocol):
    """Protocol for AI agents."""

    name: str

    def run(self, context: AgentContext) -> None:
        ...


class AgentOrchestrator:
    """Runs a list of agents sequentially."""

    def __init__(self, agents: Sequence[Agent]) -> None:
        self._agents = agents

    def run(self, context: AgentContext) -> None:
        for agent in self._agents:
            logger.info("Running agent: %s", agent.name)
            agent.run(context)
