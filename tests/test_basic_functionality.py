"""Basic functionality tests for the Infinite_Money project."""

import pytest
from hedge_fund.data.loader import load_ohlcv


def test_data_loader_basic():
    """Test that the data loader returns data."""
    from datetime import datetime, timedelta
    
    end = datetime.now().date()
    start = end - timedelta(days=10)
    
    rows = load_ohlcv("SPY", str(start), str(end))
    assert isinstance(rows, list)
    assert len(rows) > 0
    
    first = rows[0]
    for key in ["date", "symbol", "open", "high", "low", "close", "volume"]:
        assert key in first


def test_alphaquanta_imports():
    """Test that alphaquanta modules can be imported."""
    from alphaquanta.models import TradeSignal, OrderSide, OrderType
    from alphaquanta.telemetry.acu_tracker import ACUTracker
    
    # Test model creation
    signal = TradeSignal(
        symbol="SPY",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        confidence=0.8,
        strategy="test"
    )
    
    assert signal.symbol == "SPY"
    assert signal.side == OrderSide.BUY
    
    # Test ACU tracker
    tracker = ACUTracker(budget=10)
    assert tracker.budget == 10
    assert tracker.total_used == 0


def test_smoke_backtest_components():
    """Test that smoke backtest components are available."""
    from alphaquanta.agents.lean_core_agent import LeanCoreAgent
    from alphaquanta.telemetry.acu_tracker import ACUTracker
    
    # Test agent creation
    acu_tracker = ACUTracker(budget=5)
    agent = LeanCoreAgent(
        mode="backtest",
        quantum_enabled=False,
        config={
            "risk": {},
            "position_sizing": {},
            "lean": {},
            "quantum": {"enabled": False},
        },
        acu_tracker=acu_tracker,
        qpu_tracker=None,
    )
    
    assert agent.mode.value == "backtest"
    assert agent.quantum_enabled is False
