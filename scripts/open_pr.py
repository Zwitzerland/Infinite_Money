"""Emit PR draft text bundles for manual submission."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT_DIR = ROOT / "research" / "reports" / "pr_drafts"


def main() -> None:
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    draft = {
        "title": "Placeholder PR draft",
        "body": "Describe hypothesis, changes, metrics, robustness, and rollback steps.",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    path = DRAFT_DIR / f"draft-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(draft, indent=2))
    print(f"Saved PR draft to {path}")


if __name__ == "__main__":
    main()
