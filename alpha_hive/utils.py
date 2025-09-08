import numpy as np, json, os, random

def set_seed(seed: int = 1337):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def sharpe(returns: np.ndarray, eps=1e-9):
    if len(returns) == 0: return 0.0
    return np.sqrt(252)*(np.mean(returns)/(np.std(returns)+eps))

def drawdown(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / (peak + 1e-9)

def max_drawdown(equity: np.ndarray):
    return float(np.min(drawdown(equity)))

def cvar(returns: np.ndarray, alpha=0.95):
    if len(returns) == 0: return 0.0
    q = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= q]
    return float(np.mean(tail)) if len(tail) else 0.0

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
