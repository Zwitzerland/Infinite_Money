"""
Infinite_Money: Autonomous Quantum Trading System

A next-generation autonomous trading system featuring multi-agent architecture,
quantum computing integration, and continuous self-improvement capabilities.
"""

__version__ = "1.0.0"
__author__ = "Infinite_Money Team"
__email__ = "team@infinite-money.com"

from .main import AutonomousTradingSystem
from .agents import (
    ChiefArchitectAgent,
    DataEngineerAgent,
    AlphaResearcherAgent,
    PortfolioManagerAgent,
    ExecutionTraderAgent,
    ComplianceOfficerAgent,
)
from .quantum import QuantumEngine
from .backtest import BacktestEngine
from .utils.config import Config

__all__ = [
    "AutonomousTradingSystem",
    "ChiefArchitectAgent",
    "DataEngineerAgent", 
    "AlphaResearcherAgent",
    "PortfolioManagerAgent",
    "ExecutionTraderAgent",
    "ComplianceOfficerAgent",
    "QuantumEngine",
    "BacktestEngine",
    "Config",
]