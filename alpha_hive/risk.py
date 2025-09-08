import numpy as np
from .utils import cvar, max_drawdown, sharpe

def audit(returns: np.ndarray, equity: np.ndarray, dd_limit=0.08):
    stats = {"sharpe": float(sharpe(returns)), "max_drawdown": float(abs(max_drawdown(equity))), "cvar_95": float(abs(cvar(returns, 0.95)))}
    stats["risk_off"] = (stats["max_drawdown"] > dd_limit)
    return stats
