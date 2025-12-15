# Quant Canon

This living document captures non-negotiable rules for research, backtesting, and execution within this repository. Update with citations when grounded evidence is available.

- Prevent leakage through proper train/test separation, purged cross-validation, and point-in-time data handling.
- Control overfitting using walk-forward validation, deflated performance metrics, and penalty terms for complexity.
- Model execution with realistic slippage, fees, borrow availability, and latency.
- Enforce risk budgets: drawdown governors, CVaR/ES limits, turnover caps, and exposure constraints.
- Require reproducibility: fix random seeds, record dataset hashes, and capture environment metadata for every experiment.
- Avoid live deployment automation; all promotion decisions flow through PRs and manual gates.
