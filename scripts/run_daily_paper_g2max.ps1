$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $root "..")
$venv = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
  python -m venv $venv
  & "$python" -m pip install -e "$projectRoot"
}

& "$python" -m hedge_fund.ai.integration.lean_export --config "$projectRoot\hedge_fund\conf\ai_stack.yaml"

$execute = $env:INFINITE_MONEY_EXECUTE -eq "1"
$confirm = $env:INFINITE_MONEY_CONFIRM -eq "1"

if ($execute -and $confirm) {
  & "$python" -m hedge_fund.exec.ibkr_signal_executor --csv "$projectRoot\data\custom\ai_signals.csv" --symbol SPY --execute --confirm --min-hold-days 2 --approve
} else {
  & "$python" -m hedge_fund.exec.ibkr_signal_executor --csv "$projectRoot\data\custom\ai_signals.csv" --symbol SPY --min-hold-days 2
  Write-Output "Dry run only. Set INFINITE_MONEY_EXECUTE=1 and INFINITE_MONEY_CONFIRM=1 to place paper orders."
}
