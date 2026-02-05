import numpy as np

from hedge_fund.ai.portfolio.g2max import G2MaxParams, g2max_equity_curve, g2max_exposure


def test_g2max_exposure_bounds() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, 252)
    params = G2MaxParams(leverage=2.0)
    exposure = g2max_exposure(returns, params)
    assert exposure.shape[0] == returns.shape[0]
    assert np.all(np.abs(exposure) <= params.leverage + 1e-9)


def test_g2max_equity_positive() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0001, 0.012, 252)
    params = G2MaxParams(leverage=2.0)
    equity = g2max_equity_curve(returns, params)
    assert equity.shape[0] == returns.shape[0]
    assert np.all(equity > 0)
