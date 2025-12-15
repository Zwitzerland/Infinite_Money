"""Placeholder PR factory that emits candidate descriptions without code changes."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "research" / "reports"
STRATEGY_LEDGER = ROOT / "knowledge" / "summaries" / "strategy_ledger.md"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    proposal = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hypothesis": "Placeholder hypothesis from strategy ledger",
        "citations": [],
        "plan": "Small, reversible change placeholder.",
    }
    path = REPORTS_DIR / f"proposal-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(proposal, indent=2))
    print(f"Saved proposal to {path}")
    if not STRATEGY_LEDGER.exists():
        print("Strategy ledger missing; populate knowledge base for grounded proposals.")


if __name__ == "__main__":
    main()
