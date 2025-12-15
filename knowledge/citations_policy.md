# Citations Policy

This repository is grounded in the PDF and chat corpus under `context/`. The following rules govern when and how to cite sources:

## When to cite
- Any substantive quantitative claim, modeling choice, or algorithm description derived from the corpus must include a citation.
- Risk, evaluation, and constraint logic must include citations to supporting evidence where applicable.
- Summaries, runbooks, and backtesting rules must reference the originating documents.
- Code docstrings that implement published methods must cite the corresponding PDF page.

## Format
- Use `[PDF: filename p.X]` for PDF references, where `filename` is the exact file name and `X` is the page number.
- Use `[CHAT: filename turn Y]` for chat transcripts, where `filename` is the chat file name and `Y` is the turn or timestamp identifier.
- Multiple citations can be comma-separated.

## Prohibited practices
- No uncited “best practice” assertions in gating logic or constraints.
- No references to undocumented external sources.
- Do not include secrets or personally identifiable information in citations.

## Auditing
- Keep `knowledge/corpus_map.json` up to date with file hashes and page counts to support reproducibility.
- Ensure generated summaries and evidence objects include citations according to this policy.
