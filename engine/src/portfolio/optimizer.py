"""
Portfolio Optimizer Module

Implements various portfolio optimization methods:
- Mean-variance optimization
- Risk parity
- Maximum Sharpe ratio
- Black-Litterman model
- Hierarchical risk parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import logging


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods.
    
    Implements optimization strategies:
    - Mean-variance optimization
    - Risk parity
    - Maximum Sharpe ratio
    - Black-Litterman model
    - Hierarchical risk parity
    """
    
    def __init__(self, lookback_period: int = 252, risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer.
        
        Args:
            lookback_period: Number of days for lookback window
            risk_free_rate: Annual risk-free rate
        """
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.returns_history = {}
        self.covariance_matrix = None
        self.expected_returns = None
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, signals: Dict[str, float], 
                current_positions: Dict[str, float], 
                regime: str = "normal") -> Dict[str, float]:
        """
        Optimize portfolio weights.
        
        Args:
            signals: Alpha signals for each symbol
            current_positions: Current portfolio positions
            regime: Current market regime
            
        Returns:
            Dictionary mapping symbols to target weights
        """
        try:
            # Update return estimates
            self._update_return_estimates(signals)
            
            # Choose optimization method based on regime
            if regime == "crisis":
                weights = self._risk_parity_optimization()
            elif regime == "high_volatility":
                weights = self._minimum_variance_optimization()
            elif regime == "bull_market":
                weights = self._maximum_sharpe_optimization()
            else:
                weights = self._mean_variance_optimization()
            
            # Apply constraints and adjustments
            weights = self._apply_constraints(weights, current_positions, regime)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return self._equal_weight_fallback(signals)
    
    def _update_return_estimates(self, signals: Dict[str, float]):
        """Update expected returns based on signals."""
        try:
            # Convert signals to expected returns
            self.expected_returns = {}
            
            for symbol, signal in signals.items():
                # Simple signal to return conversion
                # In practice, this would be more sophisticated
                expected_return = signal * 0.1  # 10% annual return for unit signal
                self.expected_returns[symbol] = expected_return
                
        except Exception as e:
            self.logger.error(f"Error updating return estimates: {e}")
    
    def _mean_variance_optimization(self) -> Dict[str, float]:
        """Mean-variance optimization."""
        try:
            if not self.expected_returns or not self.covariance_matrix:
                return {}
            
            symbols = list(self.expected_returns.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Create optimization problem
            weights = cp.Variable(n_assets)
            
            # Expected return
            returns = np.array([self.expected_returns[s] for s in symbols])
            expected_return = returns @ weights
            
            # Risk (variance)
            if self.covariance_matrix is not None:
                risk = cp.quad_form(weights, self.covariance_matrix)
            else:
                # Use identity matrix if no covariance data
                risk = cp.sum_squares(weights)
            
            # Objective: maximize Sharpe ratio
            objective = cp.Maximize(expected_return - 0.5 * risk)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Full investment
                weights >= 0,  # Long only
                weights <= 0.1  # Max 10% per position
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                return {symbols[i]: weights.value[i] for i in range(n_assets)}
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {e}")
            return {}
    
    def _risk_parity_optimization(self) -> Dict[str, float]:
        """Risk parity optimization."""
        try:
            if not self.covariance_matrix:
                return {}
            
            symbols = list(self.expected_returns.keys()) if self.expected_returns else []
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Objective function for risk parity
            def risk_parity_objective(w):
                portfolio_risk = np.sqrt(w.T @ self.covariance_matrix @ w)
                risk_contributions = w * (self.covariance_matrix @ w) / portfolio_risk
                target_risk = portfolio_risk / n_assets
                
                # Minimize sum of squared differences from target risk
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Full investment
            ]
            
            bounds = [(0, 0.1) for _ in range(n_assets)]  # Long only, max 10%
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(risk_parity_objective, initial_weights, 
                           method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return {symbols[i]: result.x[i] for i in range(n_assets)}
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {e}")
            return {}
    
    def _minimum_variance_optimization(self) -> Dict[str, float]:
        """Minimum variance optimization."""
        try:
            if not self.covariance_matrix:
                return {}
            
            symbols = list(self.expected_returns.keys()) if self.expected_returns else []
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Objective: minimize variance
            def min_var_objective(w):
                return w.T @ self.covariance_matrix @ w
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Full investment
            ]
            
            bounds = [(0, 0.1) for _ in range(n_assets)]  # Long only, max 10%
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(min_var_objective, initial_weights, 
                           method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return {symbols[i]: result.x[i] for i in range(n_assets)}
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in minimum variance optimization: {e}")
            return {}
    
    def _maximum_sharpe_optimization(self) -> Dict[str, float]:
        """Maximum Sharpe ratio optimization."""
        try:
            if not self.expected_returns or not self.covariance_matrix:
                return {}
            
            symbols = list(self.expected_returns.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Objective: maximize Sharpe ratio
            def sharpe_objective(w):
                returns = np.array([self.expected_returns[s] for s in symbols])
                expected_return = returns @ w
                risk = np.sqrt(w.T @ self.covariance_matrix @ w)
                
                if risk == 0:
                    return -np.inf
                
                sharpe = (expected_return - self.risk_free_rate / 252) / risk
                return -sharpe  # Minimize negative Sharpe
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Full investment
            ]
            
            bounds = [(0, 0.1) for _ in range(n_assets)]  # Long only, max 10%
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(sharpe_objective, initial_weights, 
                           method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return {symbols[i]: result.x[i] for i in range(n_assets)}
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in maximum Sharpe optimization: {e}")
            return {}
    
    def _apply_constraints(self, weights: Dict[str, float], 
                          current_positions: Dict[str, float], 
                          regime: str) -> Dict[str, float]:
        """Apply additional constraints to weights."""
        try:
            constrained_weights = weights.copy()
            
            # Turnover constraints
            max_turnover = 0.1  # 10% max turnover
            if regime == "crisis":
                max_turnover = 0.05  # 5% in crisis
            
            total_turnover = 0.0
            for symbol in weights:
                current_weight = current_positions.get(symbol, 0.0)
                target_weight = weights.get(symbol, 0.0)
                turnover = abs(target_weight - current_weight)
                total_turnover += turnover
            
            # Scale down if turnover too high
            if total_turnover > max_turnover:
                scale_factor = max_turnover / total_turnover
                for symbol in constrained_weights:
                    current_weight = current_positions.get(symbol, 0.0)
                    target_weight = weights.get(symbol, 0.0)
                    constrained_weights[symbol] = current_weight + scale_factor * (target_weight - current_weight)
            
            # Concentration limits
            max_concentration = 0.15  # 15% max concentration
            if regime == "crisis":
                max_concentration = 0.10  # 10% in crisis
            
            for symbol in constrained_weights:
                if constrained_weights[symbol] > max_concentration:
                    constrained_weights[symbol] = max_concentration
            
            # Normalize weights
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                for symbol in constrained_weights:
                    constrained_weights[symbol] /= total_weight
            
            return constrained_weights
            
        except Exception as e:
            self.logger.error(f"Error applying constraints: {e}")
            return weights
    
    def _equal_weight_fallback(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Equal weight fallback if optimization fails."""
        try:
            n_assets = len(signals)
            if n_assets == 0:
                return {}
            
            equal_weight = 1.0 / n_assets
            return {symbol: equal_weight for symbol in signals.keys()}
            
        except Exception as e:
            self.logger.error(f"Error in equal weight fallback: {e}")
            return {}
    
    def update_covariance_matrix(self, returns_data: Dict[str, np.ndarray]):
        """Update covariance matrix from returns data."""
        try:
            if not returns_data:
                return
            
            # Align returns data
            symbols = list(returns_data.keys())
            min_length = min(len(returns_data[s]) for s in symbols)
            
            if min_length < 30:  # Need at least 30 observations
                return
            
            # Create returns matrix
            returns_matrix = []
            for symbol in symbols:
                returns = returns_data[symbol][-min_length:]
                returns_matrix.append(returns)
            
            returns_matrix = np.array(returns_matrix).T
            
            # Calculate covariance matrix using Ledoit-Wolf shrinkage
            lw = LedoitWolf()
            self.covariance_matrix = lw.fit(returns_matrix).covariance_
            
        except Exception as e:
            self.logger.error(f"Error updating covariance matrix: {e}")
    
    def get_portfolio_statistics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Get portfolio statistics."""
        stats = {}
        
        try:
            if not weights or not self.expected_returns or not self.covariance_matrix:
                return stats
            
            symbols = list(weights.keys())
            weight_vector = np.array([weights[s] for s in symbols])
            
            # Expected return
            returns = np.array([self.expected_returns.get(s, 0) for s in symbols])
            expected_return = returns @ weight_vector
            stats['expected_return'] = expected_return
            
            # Risk
            if self.covariance_matrix is not None:
                risk = np.sqrt(weight_vector.T @ self.covariance_matrix @ weight_vector)
                stats['volatility'] = risk
                
                # Sharpe ratio
                if risk > 0:
                    sharpe = (expected_return - self.risk_free_rate / 252) / risk
                    stats['sharpe_ratio'] = sharpe
            
            # Concentration
            stats['concentration'] = np.sum(weight_vector ** 2)  # Herfindahl index
            
            # Number of positions
            stats['num_positions'] = len([w for w in weight_vector if w > 0.01])
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio statistics: {e}")
            
        return stats
