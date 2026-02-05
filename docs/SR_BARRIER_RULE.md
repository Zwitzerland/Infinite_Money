# Support/Resistance Barrier Rule (SRB)

This is research-only. It does **not** guarantee profits.

## Core thesis
Support/resistance (SR) can have edge only if the price process is **not** a martingale
*at those levels*. In practice SR levels can proxy state-dependent liquidity / order-flow
"barriers" that shift next-hitting probabilities away from the martingale baseline.

If markets were perfectly frictionless and information-efficient (discounted prices are
martingales under an equivalent measure), then any SR rule has zero expected edge before
costs, and negative edge after costs.

## Minimal math model (bracket / two-barrier hitting)
Let `a` be a level. Enter when price enters a small zone around `a`.

Define:
- take-profit distance `u > 0`
- stop distance `l > 0`
- total execution cost `c >= 0` (spread + fees + slippage), expressed in the same units

Let `p(a)` be the conditional probability that price hits `a+u` before `a-l`.

Expected P&L per trade in this stylized bracket model:

`E[Pi | a] = p(a)*u - (1 - p(a))*l - c`

Edge condition:

`E[Pi | a] > 0  <=>  p(a) > (l + c) / (u + l)`

### Martingale baseline
In a driftless martingale diffusion (random-walk idealization), the two-barrier
probability is:

`p0 = l / (u + l)`

Plugging into the bracket EV yields:

`E[Pi] = 0 - c <= 0`

So in a strict martingale world, SR has no positive expected edge.

## Why SR can still work in reality (mechanism)
Markets are not homogeneous in state. Even if innovations are “random”, microstructure
creates localized frictions:

- Orders cluster around salient prices (prior highs/lows, round numbers, option strikes).
- Liquidity concentration can create temporary barrier behavior (reversals at levels).
- Stops clustered beyond levels can create “break acceleration” once levels are breached.

These mechanisms imply `p(a) != p0` locally.

## What counts as “proof” in applied finance
You cannot have a theorem that SR is always profitable in real markets without
empirical premises.

What you *can* do (and what this repo implements) is:
1) A mathematically explicit edge inequality (`p(a) > (l+c)/(u+l)`), and
2) Leakage-safe, out-of-sample evidence that your SR definition produces `p_hat(a)`
   above the break-even threshold after realistic costs.

## Validation protocol (non-negotiable)
- SR levels must be defined **algorithmically** (no discretionary drawing).
- Pivots must be **confirmed with delay** to avoid look-ahead bias.
- Entry/outcome labeling must be causal: only outcomes fully observable before “now” are
  included in estimation.
- Use conservative confidence bounds for `p(a)` (Wilson lower bound) rather than point
  estimates.
- Any parameter search requires multiple-testing controls:
  - deflated Sharpe / trial-count penalties
  - purged CV + embargo / CPCV for time series

## Implementation (repo)
- SR barrier engine: `hedge_fund/alpha/sr/*`
- Signal export to LEAN: `hedge_fund/ai/integration/lean_export.py` (method `sr_barrier`)
- Strategy backtests should flow through LEAN + promotion gates before paper execution.

SR engine notes:
- `level_source`: `pivots` (default) | `rounds` (nice-number grid) | `avwap` (anchored VWAP on confirmed pivots)

imctl workflows:
- `imctl sr report` writes a leakage-safe diagnostics + CPCV report under `artifacts/`.
- `imctl sr sweep` runs an Optuna sweep (CPCV median Sharpe objective) and emits `best_params.yaml` + `tuned_config.yaml`.

## References (starting point)
- Carol Osler (FRBNY), “Support for Resistance: Technical Analysis and Intraday Exchange Rates”, 2000.
- Andrew Lo, Harry Mamaysky, Jiang Wang, “Foundations of Technical Analysis”, 2000.
- Marcos Lopez de Prado, “Advances in Financial Machine Learning” (leakage-safe validation).
