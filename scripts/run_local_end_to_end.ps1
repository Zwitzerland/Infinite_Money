param(
    [string]$Project = "lean_projects/DividendCoveredCall",
    [string]$Params = "configs/lean/covered_call_params.yaml",
    [int]$Trials = 20
)

if ($args -contains "-h" -or $args -contains "--help") {
    Write-Host "Usage: run_local_end_to_end.ps1 [-Project <path>] [-Params <path>] [-Trials <int>]"
    exit 0
}

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$python = ".venv\Scripts\python"
& $python -m pip install --upgrade pip
& $python -m pip install -e .[dev]

& $python -m tools.imctl backtest --project $Project --params $Params
& $python -m tools.imctl optimize --project $Project --study "local-opt" --n-trials $Trials --sampler tpe --constraints configs/lean/constraints.yaml
& $python -m tools.imctl report --run-id latest
