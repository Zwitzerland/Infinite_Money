# Alpha Fabric Sources

This document maps the initial public sources and their intended ingestion
connectors.

## Macro and rates

- FRED/ALFRED series (macro, financial).
- BLS time series (CPI, payrolls, JOLTS).
- Treasury FiscalData APIs (rates, auctions).
- OFR repo datasets (short-term funding monitor).
- CFTC Commitments of Traders (positioning).
- BIS bulk downloads (global banking/derivatives).

## Regulatory disclosures

- SEC EDGAR submissions (10-K/Q, 8-K, 13F/13D, XBRL).
- SEC Form N-MFP datasets (money market funds).

## Politician disclosures

- House and Senate disclosures + PTRs (primary portals).
- Quiver Quantitative API for convenience (cross-check against primary).

## Defense and procurement

- SAM.gov opportunities + award data.
- Federal Register arms sales notices.

## Weather, geo, transport

- NOAA/NWS weather feeds.
- USGS earthquake GeoJSON feeds.
- ADS-B Exchange flight data.
- Global Fishing Watch AIS presence.

## Market data

- IEX Cloud, Polygon.io, Tiingo, Nasdaq Data Link (license dependent).

## News and attention

- GDELT 2.0 Events/DOC APIs.
- Cloudflare Radar metrics (license dependent).
- NewsAPI, Currents, Event Registry (redundant headline feeds).
- OFAC sanctions list updates.
- PatentsView API (innovation signals).
- CIA Reading Room + FOIA.gov (declassified corpus).

## EventRecord contract

All sources normalize into the `EventRecord` schema with event-time,
publish-time, ingestion-time, and checksum-based provenance.
