# AWS Pipeline (AI + Quantum + IBKR)

This document describes a production‑oriented AWS pipeline that perpetually
ingests data, trains models, backtests in LEAN, runs quantum portfolio
optimization on AWS Braket, and publishes signals for IBKR execution.

This is a template: you must provision AWS resources and credentials before
running. Live trading should remain disabled until paper trading is validated.

## Architecture (high level)

1. **EventBridge** schedule triggers a **Step Functions** state machine.
2. **AWS Batch/ECS** jobs run each pipeline step:
   - data ingest + feature build
   - model training + evaluation
   - signal export
   - LEAN cloud backtest + optimization
   - quantum portfolio optimization (Braket)
   - IBKR execution (paper by default)
3. **S3** stores artifacts and backtest outputs.
4. **Secrets Manager** stores API keys (News API, IBKR, QuantConnect).
5. **CloudWatch** captures logs and alerts.

## Required AWS resources

- S3 bucket for artifacts
- ECR repository for container images
- AWS Batch compute environment + job queues
- Step Functions state machine (pipeline orchestration)
- EventBridge rule for scheduling (e.g., nightly)
- Secrets Manager for credentials
- Braket access (optional)

## Config files

- `hedge_fund/conf/ai_stack.yaml`: AI pipeline config
- `hedge_fund/conf/aws_pipeline.yaml`: AWS pipeline config

Account-specific defaults are set for:

- Account ID: `484907500947`
- Region: `us-east-1`
- Bucket: `smaze-quantum-bucket`

## Local pipeline run (uploads artifacts to S3)

```bash
python -m hedge_fund.ai.aws_cli --config hedge_fund/conf/aws_pipeline.yaml --mode local
```

## Step Functions launch

```bash
python -m hedge_fund.ai.aws_cli --config hedge_fund/conf/aws_pipeline.yaml --mode step-functions
```

## Secrets Manager schema (recommended)

Store a JSON secret named `infinite-money/ibkr`:

```json
{
  "host": "127.0.0.1",
  "port": 7497,
  "client_id": 7,
  "account": "DU123456"
}
```

Store a JSON secret named `infinite-money/news` with API keys.

## LEAN integration

- Use `hedge_fund/ai/integration/lean_export.py` to export signals to
  `data/custom/ai_signals.csv`.
- The LEAN project `lean_projects/AISignalTrader` consumes the CSV in cloud
  backtests. Upload the CSV as a project file or place it in object storage.

## Quantum optimization (Braket)

Use `hedge_fund/ai/quantum/braket.py` to build a QUBO from expected returns and
covariance, then submit to Braket. Start with hybrid solvers for stability.

## Safety defaults

- Do **not** enable live trading by default.
- Require manual promotion gates and drawdown limits before live deployment.
- Always run a paper‑trading burn‑in.

## Advanced pipeline definition

See `infra/aws/step_functions/ai_pipeline_advanced.asl.json` for a gated
workflow with data validation, risk checks, and quantum optimization.

## Cost controls (recommended)

- Braket job concurrency caps and shot limits.
- Batch Spot + budget alarms.
- Step Functions timeouts and retries with caps.
- CloudWatch alerts for spend spikes.
