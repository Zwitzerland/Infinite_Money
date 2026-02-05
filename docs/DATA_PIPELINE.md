# Data Pipeline (Legal Sources)

## Ingestion Sources
- EDGAR (SEC) via official API
- House/Senate disclosures (official portals)
- News feeds (GDELT/Event Registry/NewsAPI)
- Social feeds only via official APIs
- Options snapshots via CSV exports (research-only)

## Pipeline Steps
1) Ingest raw sources with rate-limit compliance.
2) Normalize identifiers with OpenFIGI.
3) Deduplicate by content hash.
4) Track discovered_at vs event_time latency.
5) Build features for backtests only.

## Integration Points
- `hedge_fund/ai/data/filings.py`
- `hedge_fund/ai/data/news.py`
- `hedge_fund/ai/data/earnings_options.py`
- `hedge_fund/ai/pipeline.py`
