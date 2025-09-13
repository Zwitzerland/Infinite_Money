"""G²-MAX-X compounding simulator.

This lab stitches together synthetic alpha sleeves, an EXP3 meta-allocator,
robust Kelly sizing, drawdown throttling, and volatility targeting.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Tuple

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import pandas as pd


@dataclass
class Regime:
    """Return regime parameters."""

    mu: float  # annualised drift
    vol: float  # annualised volatility
    length: int  # trading days


def regime_returns(regimes: Iterable[Regime], rng: np.random.Generator) -> np.ndarray:
    """Generate concatenated daily returns for the supplied regimes."""

    blocks: list[np.ndarray] = []
    for r in regimes:
        mu_d = r.mu / 252.0
        vol_d = r.vol / np.sqrt(252.0)
        blocks.append(rng.normal(mu_d, vol_d, r.length))
    return np.concatenate(blocks)


def ewma(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted moving average."""

    out: list[float] = []
    prev = 0.0
    for v in x:
        prev = lam * prev + (1.0 - lam) * v
        out.append(prev)
    return np.array(out)


def exp3_alloc(returns_matrix: np.ndarray, gamma: float = 0.08) -> np.ndarray:
    """Risk-scaled EXP3 bandit allocator."""

    t, k = returns_matrix.shape
    w = np.ones(k)
    history = np.zeros((t, k))
    for i in range(t):
        p = (1 - gamma) * w / w.sum() + gamma / k
        history[i] = p
        vol = returns_matrix[:i].std(axis=0) + 1e-6 if i > 20 else np.ones(k)
        reward = np.clip(returns_matrix[i] / vol, -0.01, 0.01)
        est_gain = reward / p
        w *= np.exp(gamma * est_gain / k)
    return history


def robust_kelly_series(
    r_mkt: pd.Series,
    r_alpha: pd.Series,
    phi_base: float = 0.4,
    vol_target: float = 0.14,
    d1: float = 0.10,
    d2: float = 0.20,
    leverage: float = 2.5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return equity, exposure and buy-hold curves."""

    r_core = 0.6 * r_mkt.to_numpy() + 0.4 * r_alpha.to_numpy()
    s = pd.Series(r_core, index=r_mkt.index)
    look = 60
    mu_hat = s.rolling(look).mean()
    var_hat = s.rolling(look).var().replace(0.0, np.nan)
    mu_shrunk = 0.5 * mu_hat.fillna(0.0)
    f_kelly = (mu_shrunk / var_hat).clip(-leverage, leverage).fillna(0.0)

    roll_total = (1.0 + r_mkt).rolling(252).apply(lambda x: np.prod(x) - 1.0, raw=True)
    crash_mask = (roll_total < 0).astype(float).fillna(0.0)

    lam = 0.94
    ewvar: list[float] = []
    prev = 0.0
    for r in s.values:
        prev = lam * prev + (1 - lam) * (r ** 2)
        ewvar.append(prev)
    vol_ann = np.sqrt(np.array(ewvar)) * np.sqrt(252.0)
    scale = np.clip(vol_target / (vol_ann + 1e-9), 0.0, leverage)

    equity = [1.0]
    exposures: list[float] = []
    peak = 1.0
    for ri, fk, sc, cm in zip(s.values, f_kelly.values, scale, crash_mask.values):
        cur = equity[-1]
        peak = max(peak, cur)
        mdd = (peak - cur) / peak if peak > 0 else 0.0
        phi_dyn = phi_base * 0.25 if mdd >= d2 else phi_base * 0.5 if mdd >= d1 else phi_base
        crash_scale = 0.5 if cm > 0 else 1.0
        expo = np.clip(phi_dyn * fk * sc * crash_scale, -leverage, leverage)
        exposures.append(expo)
        equity.append(max(cur * (1 + expo * ri), 1e-12))

    eq = pd.Series(equity[1:], index=s.index, name="G2MAX_X")
    bh: pd.Series = r_mkt.add(1.0).cumprod()
    bh.name = "BuyHold"
    return eq, pd.Series(exposures, index=s.index, name="Exposure"), bh


def run_simulation(seed: int = 7) -> Tuple[pd.Series, pd.Series]:
    """Run the full G²-MAX-X simulation."""

    rng = np.random.default_rng(seed)
    years = 10
    n = 252 * years
    dates = pd.bdate_range(end=datetime.today().date(), periods=n)
    regs = [
        Regime(0.12, 0.18, int(n * 0.28)),
        Regime(0.02, 0.24, int(n * 0.20)),
        Regime(-0.30, 0.50, int(n * 0.10)),
        Regime(0.16, 0.26, int(n * 0.14)),
        Regime(0.08, 0.18, n - int(n * 0.28) - int(n * 0.20) - int(n * 0.10) - int(n * 0.14)),
    ]
    r_mkt = pd.Series(regime_returns(regs, rng), index=dates, name="mkt")

    roll_mean = r_mkt.rolling(126).mean().fillna(0.0).to_numpy()
    expo_momo = np.tanh(roll_mean * 400)
    r_momo = expo_momo * r_mkt.to_numpy() + rng.normal(0, 0.0005, n)

    lam_events = 0.03
    events = (rng.random(n) < lam_events).astype(float)
    kernel = np.exp(-np.arange(0, 30) / 6.0)
    drift = np.convolve(events, kernel, mode="full")[:n] * 0.0006
    r_rag = drift + rng.normal(0, 0.002, n)

    mean_rev = -0.3 * np.concatenate([[0.0], np.diff(r_mkt.to_numpy())])
    state_boost = (ewma((r_mkt.to_numpy() > 0).astype(float), 0.9) - 0.5) * 0.002
    r_kg = mean_rev + state_boost + rng.normal(0, 0.0012, n)

    p = np.array([[0.96, 0.04], [0.08, 0.92]])
    state = 0
    swm: list[int] = []
    for _ in range(n):
        state = 0 if rng.random() < p[state, 0] else 1
        swm.append(state)
    swm_arr = np.array(swm)
    r_swm = (swm_arr * 2 - 1) * 0.0009 + rng.normal(0, 0.0010, n)

    liq = ewma(np.abs(r_mkt.to_numpy()), 0.97)
    impact_edge = (liq.mean() - liq) * 0.3 * 0.001
    r_api = impact_edge + rng.normal(0, 0.0008, n)

    sleeve_df = pd.DataFrame(
        {"momo": r_momo, "rag": r_rag, "kg": r_kg, "swm": r_swm, "api": r_api},
        index=dates,
    )

    bandit_w = pd.DataFrame(exp3_alloc(sleeve_df.values), index=dates, columns=sleeve_df.columns)
    r_alpha = (bandit_w * sleeve_df).sum(axis=1)

    eq, _, bh = robust_kelly_series(r_mkt, r_alpha)
    return eq, bh


def main() -> None:
    """Render log-equity chart for the simulation."""

    eq, bh = run_simulation()
    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq.values, label="G²-MAX-X")
    plt.plot(bh.index, bh.values, label="Buy & Hold")
    plt.yscale("log")
    plt.xlabel("Date")
    plt.ylabel("Equity (log scale)")
    plt.title("G²-MAX-X vs Buy & Hold")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover - manual visualisation
    main()
