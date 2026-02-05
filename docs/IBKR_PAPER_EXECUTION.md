# IBKR Paper Execution (G2MAX Signals)

This flow uses the latest signal in `data/custom/ai_signals.csv` and places a
paper trade in IBKR. It is paper-only and requires explicit confirmation.

## 1) Export G2MAX signals
```
python -m hedge_fund.ai.integration.lean_export --config hedge_fund/conf/ai_stack.yaml
```

## 2) Start IBKR paper (TWS/IB Gateway)
- API enabled, port 7497

## 3) Dry run
```
python -m hedge_fund.exec.ibkr_signal_executor --csv data/custom/ai_signals.csv --symbol SPY
```

## 4) Execute (paper only)
```
python -m hedge_fund.exec.ibkr_signal_executor --csv data/custom/ai_signals.csv --symbol SPY --execute --confirm --approve
```

## 5) Daily automation (paper only)

Run once manually:

```
powershell -ExecutionPolicy Bypass -File scripts/run_daily_paper_g2max.ps1
```

Schedule it daily:

```
powershell -ExecutionPolicy Bypass -File scripts/register_g2max_task.ps1 -Time "09:35"
```

Enable paper execution:

```
setx INFINITE_MONEY_EXECUTE 1
setx INFINITE_MONEY_CONFIRM 1
```

Without these env vars, the task runs in dry-run mode.

When running manually, `--approve` will prompt for confirmation before
submitting the paper order.

## Safety gates
- Max order notional: 100,000
- Max position notional: 200,000
- Min cash buffer: 500
- PDT guard: blocks exits for 2 trading days after entry

Adjust using flags:
```
--max-order-notional 50000 --max-position-notional 100000 --min-cash-buffer 1000
--min-hold-days 2
```

## Notes
- This script refuses non-paper ports.
- No live trading support is included.
