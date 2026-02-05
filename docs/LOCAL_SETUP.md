# Local Setup

This guide describes the local setup required to run the Infinite Money stack.

## 1) System prerequisites

- Python 3.11
- Git
- Docker Desktop (required for LEAN local backtests)
- QuantConnect LEAN CLI

## 2) Create Python environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
. .venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## 3) Install LEAN CLI

```bash
python -m pip install lean
lean login
lean init --organization "<ORG_NAME>" --language python
```

## 4) Repo project discovery

- LEAN projects live in `lean_projects/<name>`.
- `imctl` and `Makefile` targets reference these paths directly.

## 5) Parameter injection

All LEAN algorithms must read parameters via `QCAlgorithm.get_parameter`.
Example:

```python
lookback = int(self.get_parameter("lookback", 20))
```

## 6) Run the standard commands

```bash
make doctor
make smoke
make optimize
make report
```

## 7) Artifacts

- All experiment outputs live under `artifacts/<run_id>/`.
- `run_config.json` captures config + versions.
- `checks.json` captures lint/test/smoke results.
- `report.md` summarizes best trial.

## 8) Common failures

- `lean` missing: install `lean` in the venv and log in.
- Docker not running: start Docker Desktop before `make smoke`.
- Data terms/subscriptions missing: accept QC terms and ensure data access.
