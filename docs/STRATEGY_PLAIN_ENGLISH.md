# Strategy in Plain English (G2MAX)

This is a research-only strategy designed to control risk while trying to grow
capital steadily. It does **not** guarantee profits.

## What it does
1) Looks at recent price moves and estimates whether returns are improving.
2) Calculates a "safe" exposure using fractional Kelly sizing.
3) Scales exposure so volatility stays near a target level.
4) Cuts exposure if the strategy is in a drawdown.
5) Caps leverage at a hard maximum.

## How it makes decisions
- **Drift estimate:** uses a rolling average of returns.
- **Risk estimate:** uses rolling variance + EWMA volatility.
- **Position size:** proportional to drift / variance, scaled down.
- **Drawdown throttle:** cuts risk when losses stack up.

## Optional alpha sleeves (real data)
When enabled, it adds a simple "alpha" signal built from:
- Momentum
- Mean reversion
- Breakout/volatility

These sleeves are blended with market returns to compute exposure.

## Key parameters you can tune
- `phi_base`: aggressiveness of sizing
- `vol_target`: risk target
- `drawdown_soft`/`drawdown_hard`: how fast it throttles
- `leverage`: hard cap
- `lookback`: sensitivity to recent data
- `ewma_lambda`: volatility smoothing

## Safety defaults
- Paper trading only
- Hard caps on order size and position size
- No trading unless IBKR is running on paper port 7497
