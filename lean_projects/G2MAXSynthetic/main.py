# ruff: noqa: F403, F405
from AlgorithmImports import *
from dataclasses import dataclass
import numpy as np


@dataclass
class Regime:
    mu: float
    vol: float
    length: int


def regime_returns(regimes, rng):
    blocks = []
    for r in regimes:
        mu_d = r.mu / 252.0
        vol_d = r.vol / np.sqrt(252.0)
        blocks.append(rng.normal(mu_d, vol_d, r.length))
    return np.concatenate(blocks)


def ewma(x, lam=0.94):
    out = []
    prev = 0.0
    for v in x:
        prev = lam * prev + (1.0 - lam) * v
        out.append(prev)
    return np.array(out)


def exp3_alloc(returns_matrix, gamma=0.08):
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
    r_mkt,
    r_alpha,
    phi_base=0.4,
    vol_target=0.14,
    d1=0.10,
    d2=0.20,
    leverage=2.5,
    lookback=60,
    ewma_lambda=0.94,
):
    r_core = 0.6 * r_mkt + 0.4 * r_alpha
    mu_hat = np.convolve(r_core, np.ones(lookback) / lookback, mode="same")
    var_hat = np.convolve((r_core - mu_hat) ** 2, np.ones(lookback) / lookback, mode="same")
    var_hat = np.where(var_hat == 0.0, np.nan, var_hat)
    mu_shrunk = 0.5 * np.nan_to_num(mu_hat, nan=0.0)
    f_kelly = np.nan_to_num(mu_shrunk / var_hat, nan=0.0)
    f_kelly = np.clip(f_kelly, -leverage, leverage)

    roll_total = np.convolve(1 + r_mkt, np.ones(252), mode="same") / 252.0 - 1.0
    crash_mask = (roll_total < 0).astype(float)

    ewvar = []
    prev = 0.0
    for r in r_core:
        prev = ewma_lambda * prev + (1 - ewma_lambda) * (r ** 2)
        ewvar.append(prev)
    vol_ann = np.sqrt(np.array(ewvar)) * np.sqrt(252.0)
    scale = np.clip(vol_target / (vol_ann + 1e-9), 0.0, leverage)

    equity = [1.0]
    exposures = []
    peak = 1.0
    for ri, fk, sc, cm in zip(r_core, f_kelly, scale, crash_mask):
        cur = equity[-1]
        peak = max(peak, cur)
        mdd = (peak - cur) / peak if peak > 0 else 0.0
        phi_dyn = phi_base * 0.25 if mdd >= d2 else phi_base * 0.5 if mdd >= d1 else phi_base
        crash_scale = 0.5 if cm > 0 else 1.0
        expo = np.clip(phi_dyn * fk * sc * crash_scale, -leverage, leverage)
        exposures.append(expo)
        equity.append(max(cur * (1 + expo * ri), 1e-12))

    eq = np.array(equity[1:])
    bh = np.cumprod(1.0 + r_mkt)
    return eq, np.array(exposures), bh


class G2MAXSynthetic(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2014, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        self.seed = int(self.get_parameter("seed", 7))
        self.years = int(self.get_parameter("years", 10))
        self.phi_base = float(self.get_parameter("phi_base", 0.6))
        self.vol_target = float(self.get_parameter("vol_target", 0.22))
        self.d1 = float(self.get_parameter("d1", 0.07))
        self.d2 = float(self.get_parameter("d2", 0.31))
        self.leverage = float(self.get_parameter("leverage", 3.0))
        self.lookback = int(self.get_parameter("lookback", 30))
        self.ewma_lambda = float(self.get_parameter("ewma_lambda", 0.92))

        self.trade_real = str(self.get_parameter("trade_real", "false")).lower() == "true"
        self.max_exposure = float(self.get_parameter("max_exposure", 1.0))

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.SetBenchmark(self.symbol)

        rng = np.random.default_rng(self.seed)
        n = 252 * self.years
        regs = [
            Regime(0.12, 0.18, int(n * 0.28)),
            Regime(0.02, 0.24, int(n * 0.20)),
            Regime(-0.30, 0.50, int(n * 0.10)),
            Regime(0.16, 0.26, int(n * 0.14)),
            Regime(0.08, 0.18, n - int(n * 0.28) - int(n * 0.20) - int(n * 0.10) - int(n * 0.14)),
        ]
        r_mkt = regime_returns(regs, rng)

        roll_mean = ewma(r_mkt, 0.9)
        expo_momo = np.tanh(roll_mean * 400)
        r_momo = expo_momo * r_mkt + rng.normal(0, 0.0005, n)

        lam_events = 0.03
        events = (rng.random(n) < lam_events).astype(float)
        kernel = np.exp(-np.arange(0, 30) / 6.0)
        drift = np.convolve(events, kernel, mode="full")[:n] * 0.0006
        r_rag = drift + rng.normal(0, 0.002, n)

        mean_rev = -0.3 * np.concatenate([[0.0], np.diff(r_mkt)])
        state_boost = (ewma((r_mkt > 0).astype(float), 0.9) - 0.5) * 0.002
        r_kg = mean_rev + state_boost + rng.normal(0, 0.0012, n)

        p = np.array([[0.96, 0.04], [0.08, 0.92]])
        state = 0
        swm = []
        for _ in range(n):
            state = 0 if rng.random() < p[state, 0] else 1
            swm.append(state)
        swm_arr = np.array(swm)
        r_swm = (swm_arr * 2 - 1) * 0.0009 + rng.normal(0, 0.0010, n)

        liq = ewma(np.abs(r_mkt), 0.97)
        impact_edge = (liq.mean() - liq) * 0.3 * 0.001
        r_api = impact_edge + rng.normal(0, 0.0008, n)

        sleeve = np.vstack([r_momo, r_rag, r_kg, r_swm, r_api]).T
        bandit_w = exp3_alloc(sleeve)
        r_alpha = (bandit_w * sleeve).sum(axis=1)

        self.eq, self.expo, self.bh = robust_kelly_series(
            r_mkt,
            r_alpha,
            phi_base=self.phi_base,
            vol_target=self.vol_target,
            d1=self.d1,
            d2=self.d2,
            leverage=self.leverage,
            lookback=self.lookback,
            ewma_lambda=self.ewma_lambda,
        )
        self.i = 0

        self.Plot("Equity", "G2MAX_X", float(self.eq[0]))
        self.Plot("Equity", "BuyHold", float(self.bh[0]))

    def OnData(self, data):
        if self.i >= len(self.eq):
            return

        self.Plot("Equity", "G2MAX_X", float(self.eq[self.i]))
        self.Plot("Equity", "BuyHold", float(self.bh[self.i]))

        if self.trade_real:
            exposure = max(-self.max_exposure, min(self.max_exposure, float(self.expo[self.i])))
            self.SetHoldings(self.symbol, exposure)

        if self.i == len(self.eq) - 1:
            self.Debug(f"Final equity (G2MAX_X): {self.eq[-1]:.4f}")
            self.Debug(f"Final equity (BuyHold): {self.bh[-1]:.4f}")
        self.i += 1
