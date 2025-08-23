"""
Infinite_Money: Autonomous Quantum Trading System with Apex Stack v5

A next-generation autonomous trading system featuring multi-agent architecture,
quantum computing integration, Apex Stack v5 components, and continuous self-improvement capabilities.
"""

__version__ = "2.0.0"
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

# Apex Stack v5 Components
from .optimization.wasserstein_kelly import WassersteinKellyOptimizer, KellyConstraints
from .quantum.risk_acceleration import QuantumRiskAccelerator, QAEConfig
from .risk.martingale_optimal_transport import MartingaleOptimalTransport, MOTConfig, PriceBand, HedgeBand
from .ml.microstructure_models import MicrostructureEdge, DeepLOB, HLOB, TLOB

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
    # Apex Stack v5
    "WassersteinKellyOptimizer",
    "KellyConstraints",
    "QuantumRiskAccelerator", 
    "QAEConfig",
    "MartingaleOptimalTransport",
    "MOTConfig",
    "PriceBand",
    "HedgeBand",
    "MicrostructureEdge",
    "DeepLOB",
    "HLOB", 
    "TLOB"
]