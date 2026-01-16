# News → Events → Forecasts Pipeline

This document outlines the news subsystem that powers event normalization,
historical analog retrieval, and calibrated impact forecasts.

## Components

1. **Ingestion**: GDELT (canonical), NewsAPI/Currents/Event Registry (redundant).
2. **Normalization**: article schemas, URL canonicalization, dedupe hashing.
3. **Events**: event coding (CAMEO or internal taxonomy) with supporting articles.
4. **Analog retrieval**: TF-IDF similarity for initial retrieval (replaceable).
5. **Forecasts**: conformal-style intervals over historical impacts.

## CLI

```bash
python -m hedge_fund.data.news.cli --config-path conf --config-name news_ingest
```

## Guardrails

- Agents consume only features/forecasts, never raw news.
- All forecasts are calibrated with empirical intervals.
