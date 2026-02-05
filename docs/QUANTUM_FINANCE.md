# Quantum Finance Playbook

This document outlines cutting‑edge, practical quantum workflows for portfolio
optimization and signal engineering. Use these as **experiments**, not as
guarantees of alpha.

## Core use cases

1. **Portfolio selection (QUBO/Ising)**
   - Encode mean‑variance or CVaR objectives with budget and sector constraints.
   - Solve with annealing or hybrid solvers on AWS Braket.
   - Compare against classical baselines (MIP/QP/heuristics).

2. **Quantum‑inspired feature engineering**
   - Quantum kernel methods with fidelity kernels for nonlinear patterns.
   - Feature maps over returns/volatility/spreads (see `hedge_fund/ai/quantum/feature_maps.py`).

3. **Variational circuits (QAOA/QNN)**
   - Small‑scale experiments using QAOA or variational regression.
   - Use simulators first; constrain depth to stay within NISQ limits.

## Practical workflow on AWS Braket

1. Build QUBO (`hedge_fund/ai/quantum/braket.py`).
2. Submit to Braket (annealer or hybrid) with a limited read budget.
3. Decode best bitstring → portfolio weights.
4. Evaluate under CPCV + risk constraints.
5. Repeat with penalty tuning.

## Recommended defaults

- Start with hybrid solvers + small portfolios (20‑50 assets).
- Scale coefficients to similar magnitudes to avoid solver bias.
- Use multiple runs; evaluate distributions, not single samples.

## Code entry points

- QUBO builder: `hedge_fund/ai/quantum/braket.py`
- QAOA circuit scaffolding: `hedge_fund/ai/quantum/qaoa.py`
- Hybrid solver wrapper: `hedge_fund/ai/quantum/hybrid.py`
- CVaR helper: `hedge_fund/ai/quantum/cvar.py`

## Caveats

- Quantum advantage is not guaranteed and is currently problem‑specific.
- Always benchmark against classical baselines.
- Treat quantum results as **one input** to a broader ensemble.
