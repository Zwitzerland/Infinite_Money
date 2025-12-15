"""Query the knowledge base for quick answers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "knowledge" / "indexes" / "evidence_objects.json"


def load_evidence() -> List[dict]:
    if not EVIDENCE_PATH.exists():
        return []
    return json.loads(EVIDENCE_PATH.read_text())


def query(text: str) -> List[dict]:
    evidence = load_evidence()
    # Simple keyword filter for smoke testing
    return [item for item in evidence if text.lower() in json.dumps(item).lower()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query knowledge base")
    parser.add_argument("text", type=str, help="Search text")
    args = parser.parse_args()
    results = query(args.text)
    for item in results:
        print(json.dumps(item, indent=2))
    if not results:
        print("No evidence objects matched. Populate context to improve answers.")


if __name__ == "__main__":
    main()
