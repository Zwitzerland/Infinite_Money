$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $root "..")
$venv = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
  python -m venv $venv
  & "$python" -m pip install -e "$projectRoot"
}

& "$python" -m hedge_fund.backtest.runner
& "$python" "$projectRoot\scripts\render_backtest_chart.py"

$chartPath = Join-Path $projectRoot "artifacts\backtest_equity.png"
Start-Process -FilePath $chartPath
