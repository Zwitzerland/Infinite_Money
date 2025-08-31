import pytest

torch = pytest.importorskip("torch")

from alphaquanta.rl.ppo_executor import PPOExecutor  # noqa: E402


@pytest.mark.asyncio
async def test_ppo_executor_execute_trade():
    execu = PPOExecutor(config={"rl": {"training_enabled": False}})
    signal = {"symbol": "SPY", "side": "BUY", "quantity": 10, "confidence": 0.6, "price": 100.0}
    market = {"price": 100.0, "volume": 1000, "volatility": 0.02}
    result = await execu.execute_trade(signal, market)
    assert result["success"] is True
    assert "execution_result" in result





