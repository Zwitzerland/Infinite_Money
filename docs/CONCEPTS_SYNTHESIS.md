# Concepts Synthesis (Cross-Book Summary)

This is a synthesis of the most repeated, empirically grounded ideas across the
advanced reading list. It is research guidance, not a performance promise.

## Core Compounding Mechanics
- Log-utility (Kelly) optimizes long-run growth but is unstable; fractional Kelly
  reduces drawdown risk.
- Volatility targeting stabilizes compounding paths across regimes.
- Drawdown throttles prevent pathological overbetting.
- Diversification reduces estimation error and tail risk.
- Transaction costs and turnover can erase compounding gains.

## Edge Discovery Principles
- Edge is fragile: measure decay, slippage sensitivity, and turnover impact.
- Use purged CV + embargo to avoid leakage in backtests.
- Separate signal discovery from execution to isolate edge sources.

## Portfolio Construction
- Combine multiple weak signals with risk parity or risk budgeting.
- Shrink alpha and covariance estimates to avoid overconfidence.
- Stress-test over crisis and high-vol regimes.
- Maintain leverage caps linked to volatility regimes.

## Execution and Microstructure
- Execution cost and market impact can erase small alphas.
- Model fill probability and slippage with microstructure-aware simulations.
- Avoid overfitting on high-frequency noise.

## Model Governance
- Promotion gates with minimum deflated Sharpe and drawdown constraints.
- Always compare against simple baselines.
- Keep an audit trail of data, parameters, and outputs.
- Require adversarial/regime stress tests before promotion.

## Synthesized Strategy Concept
"G2MAX" (Growth x Guardrails) combines:
1) Fractional Kelly sizing for growth.
2) Volatility targeting for stability.
3) Drawdown throttles for risk control.
4) Risk parity allocation to avoid single-factor fragility.
5) Leakage-safe validation and regime-aware gating.

See `docs/G2MAX_RULE.md` for the formal rule and `g2max_x_lab.py` for a
simulation example.
