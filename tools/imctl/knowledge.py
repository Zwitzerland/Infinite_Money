"""PDF knowledge index utilities."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import json
import time

import yaml


@dataclass(frozen=True)
class PdfRecord:
    path: str
    sha256: str
    title: str
    extracted_path: str


def _hash_file(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _extract_pdf(path: Path, output_path: Path) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf not installed; install with `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    output_path.write_text("\n\n".join(pages))


def sync_knowledge(root: Path) -> Mapping[str, Any]:
    excluded = {"knowledge", ".venv", ".git", "artifacts", "node_modules", "lean_projects"}
    pdfs = [
        path
        for path in root.rglob("*.pdf")
        if not any(part in excluded for part in path.parts)
    ]
    knowledge_root = root / "knowledge"
    pdf_dir = knowledge_root / "pdfs"
    extracted_dir = knowledge_root / "extracted"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    records: list[PdfRecord] = []
    for pdf in pdfs:
        digest = _hash_file(pdf)
        extracted_path = extracted_dir / f"{digest}.md"
        if not extracted_path.exists():
            _extract_pdf(pdf, extracted_path)
        records.append(
            PdfRecord(
                path=str(pdf.relative_to(root)),
                sha256=digest,
                title=pdf.stem,
                extracted_path=str(extracted_path.relative_to(root)),
            )
        )
        # Copy to knowledge/pdfs for reference
        target = pdf_dir / pdf.name
        if not target.exists():
            target.write_bytes(pdf.read_bytes())

    index_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(records),
        "pdfs": [record.__dict__ for record in records],
    }
    (knowledge_root / "index.json").write_text(json.dumps(index_payload, indent=2))

    refs_path = knowledge_root / "refs.yaml"
    if not refs_path.exists():
        refs_path.write_text("mappings: []\n")
    return index_payload


def render_traceability(root: Path) -> str:
    refs_path = root / "knowledge" / "refs.yaml"
    if not refs_path.exists():
        return "# Traceability\n\nNo PDF references defined.\n"
    refs = yaml.safe_load(refs_path.read_text()) or {}
    mappings = refs.get("mappings", [])
    lines = ["# Traceability", "", "| Module/Decision | PDF | Rationale |", "| --- | --- | --- |"]
    if not mappings:
        lines.append("| (none) | (none) | No PDFs registered yet. |")
        return "\n".join(lines) + "\n"
    for mapping in mappings:
        module = mapping.get("module", "")
        pdfs = ", ".join(mapping.get("pdfs", []))
        rationale = mapping.get("rationale", "")
        lines.append(f"| {module} | {pdfs} | {rationale} |")
    return "\n".join(lines) + "\n"
