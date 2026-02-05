# AI Signal Trader (LEAN)

This project consumes AI signals exported to `data/custom/ai_signals.csv` and
trades the underlying equity (default SPY).

## Export signals

```bash
python -m hedge_fund.ai.integration.lean_export --config hedge_fund/conf/ai_stack.yaml
```

## Backtest (cloud)

```bash
lean cloud push --project "lean_projects/AISignalTrader"
lean cloud backtest "lean_projects/AISignalTrader" \
  --name "ai-signal-baseline" \
  --parameter symbol SPY \
  --parameter signal_threshold 0.0 \
  --parameter max_drawdown 0.25 \
  --parameter signal_mode directional \
  --parameter max_exposure 1.0
```

## Notes

- The custom data file must be present at `data/custom/ai_signals.csv`.
- Cloud runs require uploading the CSV to the project or object store.
- Set `signal_mode=exposure` to use continuous exposure signals (e.g., G2MAX).
