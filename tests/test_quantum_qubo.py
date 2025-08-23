import pytest

from alphaquanta.quantum.qaoa_optimizer import QAOABasketOptimizer


class DummyQPU:
    def start_quantum_operation(self, *args, **kwargs):
        return "op"

    def end_quantum_operation(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_qaoa_optimizer_simple():
    opt = QAOABasketOptimizer(config={"algorithms": {"qaoa": {"max_layers": 2}}}, qpu_tracker=DummyQPU())
    weights = await opt.optimize_basket(["SPY", "QQQ"])
    assert isinstance(weights, dict)
    assert set(weights.keys()).issubset({"SPY", "QQQ"})





