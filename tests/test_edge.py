from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from hedge_fund.alpha import EdgeModel


def test_edge_model_deflated_sharpe() -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    coeff = np.array([0.05, -0.02, 0.03])
    y = pd.Series(x.to_numpy() @ coeff + rng.normal(scale=0.01, size=60))
    model = EdgeModel(n_splits=3, purge_pct=0.1, random_state=0)
    model.fit(x, y)
    preds = model.predict(x)
    ds = model.deflated_sharpe_ratio(y, preds, trials=10)
    assert np.isfinite(ds)
    assert model.coef_.shape[0] == x.shape[1]
