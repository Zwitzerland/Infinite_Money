# Sprint Alpha - Q-v2

- **Duration**: 2025-07-19 to 2025-08-02 (14 days)
- **Total Budget**: 5.5 ACU + 0.5 QPU-min

- Deploy UAT enclave with live EDGAR + Polygon feeds
- Implement QPU throttling and billing hooks
- Add Diffusion-Forecast AlphaModule for volatility plays
- Enable quantum Monte-Carlo VaR with kill-switch
- Set up comprehensive Grafana monitoring dashboards
- Complete end-to-end soak testing with HITL drills

## Task Matrix

| Day Range | Owner | Deliverable | Compute Limit | KPI | Status |
|-----------|-------|-------------|---------------|-----|--------|
| D1-D2 | Infra | UAT cluster IaC (Terraform) | 0.8 ACU | Green CI | Not Started |
| D3-D4 | Data | Live feed plumbing (EDGAR, Polygon) | 1.2 ACU | 0 pkt loss | Not Started |
| D5-D6 | Quantum | QPU throttle & billing hooks | 0.6 ACU / 0.5 QPU | Alert ≤ 80% quota | Not Started |
| D7-D8 | Strat | Diffusion module PoC | 1.4 ACU | ΔSharpe +0.05 | Not Started |
| D9-D10 | Risk | VaR + kill-switch | 0.7 ACU | <30 ms trigger | Not Started |
| D11-D12 | Ops | Grafana dashboards | 0.3 ACU | P95 ≤ 1s | Not Started |
| D13-D14 | QA | End-to-end soak, HITL drills | 0.5 ACU | Pass 100% | Not Started |

## Compute Budget by Owner

- **Infra**: 0.8 ACU + 0.0 QPU-min
- **Data**: 1.2 ACU + 0.0 QPU-min
- **Quantum**: 0.6 ACU + 0.5 QPU-min
- **Strat**: 1.4 ACU + 0.0 QPU-min
- **Risk**: 0.7 ACU + 0.0 QPU-min
- **Ops**: 0.3 ACU + 0.0 QPU-min
- **QA**: 0.5 ACU + 0.0 QPU-min

## Milestones

- **UAT Environment Ready** (D4): Isolated UAT enclave operational with live feeds
- **Quantum Integration Complete** (D8): QPU throttling and Diffusion module deployed
- **Production Readiness** (D14): All systems validated and ready for go-live
