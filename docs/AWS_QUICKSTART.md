# AWS Quickstart (Easy Mode)

This quickstart gathers all the AWS IDs needed to configure the pipeline and
exports them into a single JSON file.

## 1) Make sure AWS CLI is logged in

```powershell
aws sts get-caller-identity
```

If it fails, run `aws configure` or `aws configure sso`.

## 2) Run the collector

```powershell
cd "C:\Users\Brendon Perez\Documents\Infinite_Money"
powershell -ExecutionPolicy Bypass -File scripts\aws_collect.ps1
```

Outputs:

- `artifacts/aws_inventory.json`

## 3) Send me the inventory

Share the contents of `artifacts/aws_inventory.json` (it contains only IDs and
ARNs, no secrets).

## 4) I’ll wire the pipeline for you

Once I have the inventory, I’ll generate exact AWS commands and update
`hedge_fund/conf/aws_pipeline.yaml` with the correct role and resource ARNs.
