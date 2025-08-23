"""
Distributionally-Robust Kelly Optimization
Implements Wasserstein-Kelly to maximize worst-case log-growth inside an ambiguity ball.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any, Tuple, Optional
from scipy.stats import norm
from dataclasses import dataclass

from ..utils.logger import get_logger


@dataclass
class KellyConstraints:
    """Kelly optimization constraints."""
    cdar_budget: float  # CDaR constraint
    l_var_budget: float  # L-VaR constraint
    max_leverage: float  # Maximum leverage cap
    ambiguity_radius: float  # Wasserstein ball radius


class WassersteinKellyOptimizer:
    """
    Distributionally-Robust Kelly Optimizer
    
    Solves: max_b inf_{Q∈B_W(P,δ)} E_Q[log(1 + b^T R)]
    s.t. CDaR_α(wealth) ≤ D*, L-VaR ≤ L*
    """
    
    def __init__(self, constraints: KellyConstraints):
        """Initialize the optimizer."""
        self.constraints = constraints
        self.logger = get_logger("WassersteinKelly")
        
    def optimize_weights(self, 
                        returns: np.ndarray,
                        current_wealth: float,
                        market_conditions: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize portfolio weights using distributionally-robust Kelly.
        
        Args:
            returns: Historical returns matrix (T x N)
            current_wealth: Current portfolio value
            market_conditions: Current market regime indicators
            
        Returns:
            optimal_weights: Optimal portfolio weights
            optimization_info: Optimization details and diagnostics
        """
        try:
            T, N = returns.shape
            
            # 1. Estimate empirical distribution
            empirical_mean = np.mean(returns, axis=0)
            empirical_cov = np.cov(returns.T)
            
            # 2. Set up the optimization problem
            b = cp.Variable(N)  # Portfolio weights
            
            # Wasserstein ambiguity set
            delta = self.constraints.ambiguity_radius
            
            # Worst-case expected log return
            worst_case_return = cp.sum(cp.multiply(b, empirical_mean)) - \
                               delta * cp.norm(cp.sqrt(empirical_cov) @ b, 2)
            
            # CDaR constraint (Conditional Drawdown at Risk)
            cdar_constraint = self._compute_cdar_constraint(b, returns, current_wealth)
            
            # L-VaR constraint (Liquidity-adjusted Value at Risk)
            l_var_constraint = self._compute_l_var_constraint(b, returns, market_conditions)
            
            # Leverage constraint
            leverage_constraint = cp.norm(b, 1) <= self.constraints.max_leverage
            
            # Objective: maximize worst-case log growth
            objective = cp.Maximize(worst_case_return)
            
            # Constraints
            constraints = [
                cdar_constraint <= self.constraints.cdar_budget,
                l_var_constraint <= self.constraints.l_var_budget,
                leverage_constraint,
                b >= -1,  # No more than 100% short
                b <= 2    # No more than 200% long
            ]
            
            # Solve the problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = b.value
                
                # Compute diagnostics
                diagnostics = {
                    "status": "optimal",
                    "objective_value": problem.value,
                    "cdar_actual": self._compute_actual_cdar(optimal_weights, returns, current_wealth),
                    "l_var_actual": self._compute_actual_l_var(optimal_weights, returns, market_conditions),
                    "leverage": np.sum(np.abs(optimal_weights)),
                    "worst_case_return": worst_case_return.value,
                    "ambiguity_radius": delta
                }
                
                self.logger.info(f"Kelly optimization completed. Leverage: {diagnostics['leverage']:.3f}")
                return optimal_weights, diagnostics
                
            else:
                self.logger.warning(f"Kelly optimization failed: {problem.status}")
                return np.zeros(N), {"status": "failed", "reason": problem.status}
                
        except Exception as e:
            self.logger.error(f"Error in Kelly optimization: {str(e)}")
            return np.zeros(N), {"status": "error", "message": str(e)}
    
    def _compute_cdar_constraint(self, b: cp.Variable, returns: np.ndarray, wealth: float) -> cp.Expression:
        """Compute CDaR constraint expression."""
        # Simplified CDaR approximation using CVaR of drawdowns
        alpha = 0.05  # 95% confidence level
        
        # Portfolio returns
        portfolio_returns = returns @ b
        
        # Cumulative wealth path
        wealth_path = wealth * cp.cumprod(1 + portfolio_returns)
        
        # Drawdowns
        peak = cp.maximum.accumulate(wealth_path)
        drawdowns = (peak - wealth_path) / peak
        
        # CDaR as CVaR of drawdowns
        sorted_drawdowns = cp.sort(drawdowns, axis=0)
        k = int(alpha * len(drawdowns))
        cdar = cp.sum(sorted_drawdowns[k:]) / (len(drawdowns) - k)
        
        return cdar
    
    def _compute_l_var_constraint(self, b: cp.Variable, returns: np.ndarray, 
                                 market_conditions: Dict[str, Any]) -> cp.Expression:
        """Compute L-VaR constraint expression."""
        # Portfolio returns
        portfolio_returns = returns @ b
        
        # Base VaR
        var_quantile = 0.05
        var = cp.quantile(portfolio_returns, var_quantile)
        
        # Liquidity adjustment based on market conditions
        liquidity_factor = market_conditions.get("liquidity_factor", 1.0)
        volatility_factor = market_conditions.get("volatility_factor", 1.0)
        
        # L-VaR = VaR * liquidity_adjustment
        l_var = var * liquidity_factor * volatility_factor
        
        return -l_var  # Negative because we want to minimize risk
    
    def _compute_actual_cdar(self, weights: np.ndarray, returns: np.ndarray, wealth: float) -> float:
        """Compute actual CDaR for given weights."""
        portfolio_returns = returns @ weights
        wealth_path = wealth * np.cumprod(1 + portfolio_returns)
        
        peak = np.maximum.accumulate(wealth_path)
        drawdowns = (peak - wealth_path) / peak
        
        alpha = 0.05
        sorted_drawdowns = np.sort(drawdowns)
        k = int(alpha * len(drawdowns))
        
        return np.mean(sorted_drawdowns[k:])
    
    def _compute_actual_l_var(self, weights: np.ndarray, returns: np.ndarray,
                             market_conditions: Dict[str, Any]) -> float:
        """Compute actual L-VaR for given weights."""
        portfolio_returns = returns @ weights
        var = np.percentile(portfolio_returns, 5)
        
        liquidity_factor = market_conditions.get("liquidity_factor", 1.0)
        volatility_factor = market_conditions.get("volatility_factor", 1.0)
        
        return -var * liquidity_factor * volatility_factor
    
    def compute_leverage_throttle(self, 
                                optimal_weights: np.ndarray,
                                forecast_dispersion: float,
                                regime_change_prob: float) -> float:
        """
        Compute leverage throttle: λ(t) = min{λ_cap, c(t)||b*||}
        where c(t) ↓ as forecast dispersion ↑ or regime change-points fire
        """
        base_leverage = np.sum(np.abs(optimal_weights))
        
        # Throttle factor based on uncertainty
        dispersion_factor = 1.0 / (1.0 + forecast_dispersion)
        regime_factor = 1.0 / (1.0 + regime_change_prob)
        
        throttle_factor = min(dispersion_factor, regime_factor)
        
        final_leverage = min(
            self.constraints.max_leverage,
            throttle_factor * base_leverage
        )
        
        self.logger.info(f"Leverage throttle: {final_leverage:.3f} (base: {base_leverage:.3f})")
        
        return final_leverage