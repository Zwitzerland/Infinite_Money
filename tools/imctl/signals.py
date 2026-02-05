"""Signal export helpers."""

from __future__ import annotations

from pathlib import Path

from hedge_fund.ai.integration.lean_export import export_signals


def run_signal_export(config_path: Path, output_root: Path) -> Path:
    return export_signals(str(config_path), output_root)
