from __future__ import annotations

from hedge_fund.utils.contracts import default_contract_bundle
from hedge_fund.utils.settings import PlatformSettings


def test_default_contract_bundle_uses_settings() -> None:
    settings = PlatformSettings(
        max_order_rate_per_sec=12,
        min_deflated_sharpe=1.25,
        max_overfit_probability=0.1,
        min_oos_windows=4,
        min_paper_days=40,
    )
    bundle = default_contract_bundle(settings)

    assert bundle.execution.max_orders_per_sec == 12
    assert bundle.promotion.min_deflated_sharpe == 1.25
    assert bundle.promotion.max_overfit_probability == 0.1
    assert bundle.promotion.min_oos_windows == 4
    assert bundle.promotion.min_paper_days == 40
