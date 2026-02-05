$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $root "..")
$venv = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
  python -m venv $venv
  & "$python" -m pip install -e "$projectRoot"
}

& "$python" -m optimizer.earnings_vol_optuna --trials 120 --min-trades 25 --max-drawdown 0.35

$optDir = Get-ChildItem "$projectRoot\artifacts" -Directory -Filter "earnings_vol_opt_*" |
  Sort-Object Name -Descending | Select-Object -First 1

if ($optDir) {
  Write-Output "Summary: $(Join-Path $optDir.FullName 'summary.json')"
  Write-Output "Trials: $(Join-Path $optDir.FullName 'trials.json')"
}
