"""Build knowledge artifacts from ingested context."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
CORPUS_MAP = ROOT / "knowledge" / "corpus_map.json"
PDF_CARD_DIR = ROOT / "knowledge" / "summaries" / "pdf_cards"
CHAT_CARD_DIR = ROOT / "knowledge" / "summaries" / "chat_cards"
EVIDENCE_PATH = ROOT / "knowledge" / "indexes" / "evidence_objects.json"


@dataclass
class EvidenceObject:
    method_id: str
    description: str
    citations: List[str]
    suggested_modules: List[str]
    tests: List[str]


def load_corpus() -> List[dict]:
    if not CORPUS_MAP.exists():
        return []
    return json.loads(CORPUS_MAP.read_text())


def write_pdf_card(entry: dict) -> None:
    PDF_CARD_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_CARD_DIR / f"{Path(entry['path']).stem}.md"
    path.write_text(
        "\n".join(
            [
                f"# {Path(entry['path']).name}",
                "",
                "Structured summary (UN-GROUNDED PLACEHOLDER). Add citations when corpus is populated.",
                "- Key methods: TODO",
                "- Assumptions: TODO",
                "- Pitfalls: TODO",
                "- Implementable checklist: TODO",
            ]
        )
    )


def write_chat_card(entry: dict) -> None:
    CHAT_CARD_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAT_CARD_DIR / f"{Path(entry['path']).stem}.md"
    path.write_text(
        "\n".join(
            [
                f"# {Path(entry['path']).name}",
                "",
                "Chat summary (UN-GROUNDED PLACEHOLDER). Add citations when corpus is populated.",
                "- Constraints: TODO",
                "- Preferences: TODO",
                "- Decisions: TODO",
                "- Requested features: TODO",
            ]
        )
    )


def write_evidence_objects(entries: List[dict]) -> None:
    evidence: List[EvidenceObject] = []
    for entry in entries:
        evidence.append(
            EvidenceObject(
                method_id=f"{Path(entry['path']).stem}-baseline",
                description="Placeholder evidence object awaiting grounded methods.",
                citations=[],
                suggested_modules=["hedge_fund.alpha", "hedge_fund.risk"],
                tests=["pytest -q"],
            )
        )
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps([asdict(item) for item in evidence], indent=2))


def main() -> None:
    entries = load_corpus()
    for entry in entries:
        if entry.get("source_type") == "pdf":
            write_pdf_card(entry)
        if entry.get("source_type") == "chat":
            write_chat_card(entry)
    write_evidence_objects(entries)
    print(f"Processed {len(entries)} entries into knowledge base.")


if __name__ == "__main__":
    main()
