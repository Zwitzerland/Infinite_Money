# Runbook (Initial)

Current as of 2026-01-21.

## Contract bundle export

```bash
python -m hedge_fund.utils.contracts_cli
```

Defaults live in `hedge_fund/conf/contracts.yaml`.

## Backtest runner (example)

```bash
python -m hedge_fund.backtest.runner
```

Outputs:

- `artifacts/backtest_report.json`
- `artifacts/backtest_equity.csv`

Configuration: `hedge_fund/conf/backtest.yaml`.

## Promotion gates

1. Validate data partitions and checksums.
2. Run backtests with CPCV + embargo.
3. Compute deflated Sharpe and probability-of-overfitting.
4. Require paper trading days to elapse.
5. Promote only after all checks pass.

## Incident response

1. Trigger kill switch on data quality or risk breaches.
2. Liquidate or reduce exposure per execution contract.
3. Capture telemetry, event logs, and run IDs for audit.
