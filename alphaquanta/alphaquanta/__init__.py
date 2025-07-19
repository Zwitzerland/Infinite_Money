"""
AlphaQuanta - Production-grade agent framework for algorithmic trading.

Core Components:
- LeanCoreAgent: Single omni-agent for trading signals and execution
- Tools: QuantConnect API, ib_insync integration, backtesting
- Guardrails: Risk management and safety checks
- Telemetry: ACU tracking and PnL monitoring
"""

from .agents.lean_core_agent import LeanCoreAgent
from .tools.data_tools import QuantConnectDataTool, MarketDataTool
from .tools.action_tools import IBOrderRouter, LeanBacktestRunner, PositionSizer
from .guardrails.risk_guardrails import RiskGuardrailEngine, NotionalLimitGuardrail
from .telemetry.acu_tracker import ACUTracker
from .telemetry.pnl_monitor import PnLMonitor

__version__ = "0.1.0"
__all__ = [
    "LeanCoreAgent",
    "QuantConnectDataTool",
    "MarketDataTool", 
    "IBOrderRouter",
    "LeanBacktestRunner",
    "PositionSizer",
    "RiskGuardrailEngine",
    "NotionalLimitGuardrail",
    "ACUTracker",
    "PnLMonitor",
]
