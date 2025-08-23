"""
Backtest configuration for Infinite_Money trading system.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    initial_capital: float
    symbols: List[str]
    strategy_name: str
    commission: float = 0.001
    slippage: float = 0.0005
    benchmark: Optional[str] = None
    risk_free_rate: float = 0.02