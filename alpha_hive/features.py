import pandas as pd, numpy as np

def rsi(close: pd.Series, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(com=n-1, adjust=False).mean()
    ma_down = down.ewm(com=n-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in df.groupby("symbol"):
        g = g.copy()
        g["ret"] = g["close"].pct_change().fillna(0.0)
        g["ma20"] = g["close"].rolling(20).mean().bfill()
        g["rsi14"] = rsi(g["close"], 14).fillna(50.0)
        g["zvol"] = (g["volume"] - g["volume"].rolling(20).mean().bfill()) / (g["volume"].rolling(20).std().bfill() + 1e-9)
        out.append(g)
    return pd.concat(out).reset_index(drop=True)
