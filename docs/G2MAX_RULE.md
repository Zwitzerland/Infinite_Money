# G2MAX Rule (Growth x Guardrails)

This is a research-only compounding rule designed to balance growth with risk
controls. It does not guarantee returns.

## Core Rule
Let r_t be the portfolio return at time t.

1) Estimate drift/variance on a rolling window:
   mu_t = mean(r_{t-L+1:t})
   var_t = var(r_{t-L+1:t})

2) Kelly fraction with shrinkage and cap:
   k_t = clamp(mu_t / var_t, -L, L)

3) Volatility targeting scale:
   s_t = clamp(target_vol / ewma_vol_t, 0, L)

4) Drawdown gate:
   g_t = 1.0 (normal)
   g_t = 0.5 if drawdown > dd_soft
   g_t = 0.25 if drawdown > dd_hard

5) Exposure:
   exposure_t = clip(phi_base * k_t * s_t * g_t, -L, L)

6) Equity update:
   equity_{t+1} = equity_t * (1 + exposure_t * r_t)

## Parameter Notes
- phi_base: fractional Kelly control (typical 0.2–0.5)
- L: max leverage cap (hard limit)
- dd_soft/dd_hard: drawdown thresholds for throttling
- target_vol: volatility target for risk normalization

## Files
- Rule implementation: `hedge_fund/ai/portfolio/g2max.py`
- Simulation lab: `g2max_x_lab.py`
- Interactive chart: `tools/g2max_interactive.py`
- Signal export: `hedge_fund/ai/integration/lean_export.py` (method `g2max`)
