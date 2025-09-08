import gymnasium as gym
import numpy as np, pandas as pd

class SimplePortfolioEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, symbols, cost_bps=2.0, window=64):
        super().__init__()
        self.df = df[df["symbol"].isin(symbols)].copy()
        self.symbols = list(symbols)
        self.cost = cost_bps/1e4; self.window = window
        pivot_close = self.df.pivot(index="date", columns="symbol", values="close").ffill()
        pivot_ret   = self.df.pivot(index="date", columns="symbol", values="ret").fillna(0.0)
        pivot_ma    = self.df.pivot(index="date", columns="symbol", values="ma20").ffill()
        pivot_rsi   = self.df.pivot(index="date", columns="symbol", values="rsi14").fillna(50.0)
        self.features = np.stack([pivot_ret.values, (pivot_close/pivot_ma).values, (pivot_rsi/100.0).values, np.sign(pivot_ret.values)], axis=-1)
        self.dates = pivot_close.index.values; self.n = self.features.shape[0]; self.m = len(self.symbols)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.m,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window, self.m*4), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window; self.w = np.zeros(self.m, dtype=np.float32); self.nav = 1.0
        obs = self.features[self.t-self.window:self.t].reshape(self.window, -1).astype(np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        tw = action / 2.0; tw = tw - tw.mean()
        if np.sum(np.abs(tw)) > 0: tw = tw / np.sum(np.abs(tw))
        turnover = np.sum(np.abs(tw - self.w)); self.w = tw
        rets = self.features[self.t, :, 0]
        gross = float(np.dot(self.w, rets)); net = gross - turnover*self.cost
        self.nav *= (1.0 + net); self.t += 1
        done = (self.t >= self.n-1)
        obs = self.features[self.t-self.window:self.t].reshape(self.window, -1).astype(np.float32)
        return obs, net, done, False, {"nav": self.nav, "ret": net}
