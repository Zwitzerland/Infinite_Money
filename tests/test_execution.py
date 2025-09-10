from hedge_fund.backtest.events import OrderEvent, OrderSide
from hedge_fund.exec.simulated import SimulatedExecutionHandler


def test_simulated_execution_returns_fill() -> None:
    handler = SimulatedExecutionHandler({"AAPL": 100.0}, commission=1.0)
    order = OrderEvent("AAPL", OrderSide.BUY, 10, 1)
    fill = handler.place_order(order)
    assert fill.symbol == "AAPL"
    assert fill.quantity == 10
    assert fill.price == 100.0
    assert fill.commission == 1.0


def test_simulated_partial_fill() -> None:
    handler = SimulatedExecutionHandler({"AAPL": 100.0}, liquidity={"AAPL": 5})
    order = OrderEvent("AAPL", OrderSide.BUY, 10, 1)
    fill = handler.place_order(order)
    assert fill.quantity == 5
    assert handler.liquidity["AAPL"] == 0
