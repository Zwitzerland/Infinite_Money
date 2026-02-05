# Dividend Covered Call (LEAN)

This project runs a covered-call strategy on a single equity. The intent is to
provide a parameterized LEAN algorithm that can be optimized via `lean optimize`
or Optuna.

## Parameters

- `symbol` (string, default `SPY`): underlying equity symbol.
- `lookback` (int, default `20`): SMA lookback for the trend filter.
- `delta` (float, default `0.30`): target option delta for the call sell.
- `dte_min` (int, default `7`): minimum days to expiration.
- `dte_max` (int, default `30`): maximum days to expiration.
- `max_drawdown` (float, default `0.25`): hard drawdown stop.

All parameters are read via `QCAlgorithm.get_parameter`.

## Run locally

```bash
lean backtest "lean_projects/DividendCoveredCall" \
  --parameter symbol SPY \
  --parameter lookback 20 \
  --parameter delta 0.30 \
  --parameter dte_min 7 \
  --parameter dte_max 30 \
  --parameter max_drawdown 0.25 \
  --download-data
```

Options data requires a compatible QuantConnect data subscription. Use
`--download-data` to fetch missing data from QuantConnect.
