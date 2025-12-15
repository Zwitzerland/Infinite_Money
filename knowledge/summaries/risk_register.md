# Risk Register

- Data leakage from misaligned labels or lookahead bias.
- Regime shifts causing model breakdown; monitor volatility and liquidity proxies.
- Survivorship bias and stale corporate actions; prefer point-in-time datasets.
- Transaction cost underestimation including slippage, fees, borrow, and latency.
- Tail risk and drawdown clusters; implement governors and scenario stress tests.
- Infrastructure failures or missing secrets; include guardrails and clear error handling.
