"""Simple signal backtest simulator."""
from __future__ import annotations

from typing import Sequence


def simulate_strategy(
    returns: Sequence[float],
    positions: Sequence[int],
    trading_cost: float = 0.0,
) -> list[float]:
    """Simulate an equity curve from returns and positions."""
    if len(returns) != len(positions):
        raise ValueError("returns and positions must have the same length")
    equity = [1.0]
    for ret, pos in zip(returns, positions):
        net = pos * ret - trading_cost * abs(pos)
        equity.append(equity[-1] * (1 + net))
    return equity[1:]
