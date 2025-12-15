"""Ingest PDFs and chat transcripts into structured manifests.

This script is designed to be deterministic and side-effect free beyond writing
artifacts to the `knowledge/` directory. It computes SHA256 hashes for
reproducibility and stores chunked text for downstream indexing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "context" / "pdfs"
CHAT_DIR = ROOT / "context" / "chats"
KNOWLEDGE_DIR = ROOT / "knowledge"
INDEX_DIR = KNOWLEDGE_DIR / "indexes"
CORPUS_MAP = KNOWLEDGE_DIR / "corpus_map.json"


@dataclass
class CorpusEntry:
    """Metadata for a single source document."""

    path: str
    sha256: str
    source_type: str
    count: int


@dataclass
class EvidenceObject:
    """Structured evidence capturing chunks and their provenance."""

    method_id: str
    description: str
    citations: List[str]
    suggested_modules: List[str]
    tests: List[str]


def sha256_file(path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def ingest_pdfs() -> List[CorpusEntry]:
    entries: List[CorpusEntry] = []
    pdf_chunks: Dict[str, List[Dict[str, str]]] = {}
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        page_texts: List[Dict[str, str]] = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            chunk_id = f"{pdf_path.name}-p{idx + 1}"
            page_texts.append({"chunk_id": chunk_id, "page": idx + 1, "text": text})
        pdf_chunks[pdf_path.name] = page_texts
        entry = CorpusEntry(
            path=str(pdf_path.relative_to(ROOT)),
            sha256=sha256_file(pdf_path),
            source_type="pdf",
            count=len(page_texts),
        )
        entries.append(entry)
    if pdf_chunks:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        (INDEX_DIR / "pdf_chunks.json").write_text(json.dumps(pdf_chunks, indent=2))
    return entries


def ingest_chats() -> List[CorpusEntry]:
    entries: List[CorpusEntry] = []
    chat_chunks: Dict[str, List[Dict[str, str]]] = {}
    for chat_path in sorted(list(CHAT_DIR.glob("*.md")) + list(CHAT_DIR.glob("*.txt")) + list(CHAT_DIR.glob("*.json"))):
        content = chat_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        chunks: List[Dict[str, str]] = []
        for idx, line in enumerate(lines):
            chunk_id = f"{chat_path.name}-t{idx + 1}"
            chunks.append({"chunk_id": chunk_id, "turn": idx + 1, "text": line})
        chat_chunks[chat_path.name] = chunks
        entry = CorpusEntry(
            path=str(chat_path.relative_to(ROOT)),
            sha256=sha256_file(chat_path),
            source_type="chat",
            count=len(chunks),
        )
        entries.append(entry)
    if chat_chunks:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        (INDEX_DIR / "chat_chunks.json").write_text(json.dumps(chat_chunks, indent=2))
    return entries


def save_corpus_map(entries: List[CorpusEntry]) -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_MAP.write_text(json.dumps([asdict(entry) for entry in entries], indent=2))


def ensure_context_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if not (PDF_DIR / "README.md").exists():
        (PDF_DIR / "README.md").write_text(
            "Place grounding PDFs here. Files are hashed during ingestion."
        )
    if not (CHAT_DIR / "README.md").exists():
        (CHAT_DIR / "README.md").write_text(
            "Place chat transcripts here. Files are hashed during ingestion."
        )


def main() -> None:
    ensure_context_dirs()
    entries: List[CorpusEntry] = []
    entries.extend(ingest_pdfs())
    entries.extend(ingest_chats())
    save_corpus_map(entries)
    print(f"Indexed {len(entries)} sources into {CORPUS_MAP}")


if __name__ == "__main__":
    main()
