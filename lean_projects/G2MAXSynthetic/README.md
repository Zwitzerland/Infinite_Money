# G2MAX Synthetic (LEAN)

This project reproduces the synthetic G2MAX-X backtest inside QuantConnect.
It plots the same synthetic equity curves used in the local runner.

## How to run (QuantConnect)
1) Create a new project and paste `main.py`.
2) Set parameters to match the latest best run:
   - `seed=7`, `years=10`
   - `phi_base=0.6`, `vol_target=0.22`
   - `d1=0.07`, `d2=0.31`
   - `leverage=3.0`, `lookback=30`, `ewma_lambda=0.92`

## Notes
- This is a synthetic backtest (no real market data).
- If you set `trade_real=true`, it will apply exposures to SPY for a
  *non-identical* real-data experiment.
