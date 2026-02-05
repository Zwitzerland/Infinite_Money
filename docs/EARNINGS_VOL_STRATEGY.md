# Earnings Volatility Strategy (Research-Only)

This strategy focuses on volatility term-structure signals around earnings. It is
research-only and uses a simplified PnL proxy to rank trades; it is not a
production options pricer.

## Data inputs (CSV)

CSV path defaults to `data/options/earnings_snapshots.csv`.

Required columns:

| Column | Description |
| --- | --- |
| `symbol` | Underlying ticker |
| `asof_date` | Signal date (UTC ISO8601) |
| `earnings_date` | Earnings date (UTC ISO8601) |
| `underlying_price` | Spot price |
| `iv_front` | Front expiry IV (decimal, e.g. 0.45) |
| `iv_back` | Back expiry IV (decimal) |
| `rv_20d` | 20-day realized vol (decimal) |
| `front_dte` | Days-to-expiry for front contract |
| `back_dte` | Days-to-expiry for back contract |

Optional columns:

| Column | Description |
| --- | --- |
| `option_volume` | Aggregate option volume |
| `open_interest` | Aggregate open interest |
| `bid_ask_spread` | Bid/ask spread (decimal of underlying) |

## Signal logic

1) Filter by days-to-earnings window.
2) Filter by DTE bounds for front/back expiries.
3) Filter by liquidity (volume/open interest/spread).
4) Require IV/RV premium.
5) Choose trade by term-structure slope:
   - Inversion (front IV > back IV) → `short_straddle`
   - Contango (back IV > front IV) → `calendar_spread`

Term slope uses `(iv_back - iv_front) / iv_front`.

## Backtest

```bash
python -m hedge_fund.backtest.earnings_volatility
```

Outputs (defaults):
- `artifacts/earnings_vol_report.json`
- `artifacts/earnings_vol_equity.csv`
- `artifacts/earnings_vol_trades.csv`

## Optimization

```bash
python -m optimizer.earnings_vol_optuna --trials 120
```

PowerShell helper:

```powershell
scripts\run_earnings_vol_optimization.ps1
```

Outputs:
- `artifacts/earnings_vol_opt_<timestamp>/summary.json`
- `artifacts/earnings_vol_opt_<timestamp>/trials.json`

## Notes

- The PnL proxy scales with IV/RV edge and term-structure slope; it does not
  model full option greeks or discrete earnings jumps.
- Paper-only. No live execution automation is included.
