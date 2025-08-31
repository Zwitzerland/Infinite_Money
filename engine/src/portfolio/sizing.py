"""
Kelly Criterion Sizing Module

Implements Kelly criterion and related position sizing methods:
- Full Kelly criterion
- Fractional Kelly
- Risk-adjusted Kelly
- Regime-based Kelly
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
import logging


class KellyCriterion:
    """
    Kelly criterion position sizing.
    
    Implements various Kelly criterion methods:
    - Full Kelly criterion
    - Fractional Kelly
    - Risk-adjusted Kelly
    - Regime-based Kelly
    """
    
    def __init__(self, fractional_kelly: float = 0.25, max_leverage: float = 2.0):
        """
        Initialize the Kelly criterion model.
        
        Args:
            fractional_kelly: Fraction of full Kelly to use (0-1)
            max_leverage: Maximum leverage allowed
        """
        self.fractional_kelly = fractional_kelly
        self.max_leverage = max_leverage
        self.return_history = {}
        self.volatility_history = {}
        self.logger = logging.getLogger(__name__)
        
    def calculate_sizes(self, target_weights: Dict[str, float], 
                       signals: Dict[str, float], 
                       regime: str = "normal") -> Dict[str, float]:
        """
        Calculate Kelly-optimal position sizes.
        
        Args:
            target_weights: Target portfolio weights
            signals: Alpha signals for each symbol
            regime: Current market regime
            
        Returns:
            Dictionary mapping symbols to Kelly-optimal weights
        """
        try:
            kelly_weights = {}
            
            for symbol in target_weights.keys():
                if symbol in signals:
                    # Calculate Kelly fraction for this symbol
                    kelly_fraction = self._calculate_kelly_fraction(symbol, signals[symbol], regime)
                    
                    # Apply fractional Kelly
                    kelly_weight = target_weights[symbol] * kelly_fraction * self.fractional_kelly
                    
                    # Apply leverage constraints
                    kelly_weight = self._apply_leverage_constraints(kelly_weight, regime)
                    
                    kelly_weights[symbol] = kelly_weight
                else:
                    kelly_weights[symbol] = target_weights[symbol]
            
            # Normalize weights
            kelly_weights = self._normalize_weights(kelly_weights)
            
            return kelly_weights
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly sizes: {e}")
            return target_weights
    
    def _calculate_kelly_fraction(self, symbol: str, signal: float, regime: str) -> float:
        """Calculate Kelly fraction for a symbol."""
        try:
            # Get historical data for this symbol
            returns = self.return_history.get(symbol, [])
            volatility = self.volatility_history.get(symbol, 0.2)  # Default 20% vol
            
            if len(returns) < 30:
                # Use signal-based Kelly if insufficient data
                return self._signal_based_kelly(signal, regime)
            
            # Calculate Kelly using historical returns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Full Kelly formula: f = (μ - r) / σ²
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            full_kelly = (mean_return - risk_free_rate) / (std_return ** 2)
            
            # Apply regime adjustments
            full_kelly = self._adjust_kelly_for_regime(full_kelly, regime)
            
            # Clip to reasonable bounds
            full_kelly = np.clip(full_kelly, -1.0, 1.0)
            
            return full_kelly
            
        except Exception as e:
            self.logger.debug(f"Error calculating Kelly fraction for {symbol}: {e}")
            return self._signal_based_kelly(signal, regime)
    
    def _signal_based_kelly(self, signal: float, regime: str) -> float:
        """Calculate Kelly fraction based on signal strength."""
        try:
            # Convert signal to Kelly fraction
            # Strong positive signal -> higher Kelly
            # Strong negative signal -> lower Kelly
            
            # Base Kelly from signal
            base_kelly = signal * 0.5  # Scale signal to Kelly range
            
            # Regime adjustments
            if regime == "crisis":
                base_kelly *= 0.5  # Reduce position sizes in crisis
            elif regime == "high_volatility":
                base_kelly *= 0.7  # Reduce position sizes in high vol
            elif regime == "bull_market":
                base_kelly *= 1.2  # Increase position sizes in bull market
            elif regime == "bear_market":
                base_kelly *= 0.6  # Reduce position sizes in bear market
            
            # Clip to bounds
            base_kelly = np.clip(base_kelly, -1.0, 1.0)
            
            return base_kelly
            
        except Exception as e:
            self.logger.debug(f"Error in signal-based Kelly: {e}")
            return 0.0
    
    def _adjust_kelly_for_regime(self, kelly_fraction: float, regime: str) -> float:
        """Adjust Kelly fraction based on market regime."""
        try:
            adjusted_kelly = kelly_fraction
            
            if regime == "crisis":
                # Reduce position sizes in crisis
                adjusted_kelly *= 0.3
            elif regime == "high_volatility":
                # Reduce position sizes in high volatility
                adjusted_kelly *= 0.6
            elif regime == "bear_market":
                # Reduce position sizes in bear market
                adjusted_kelly *= 0.7
            elif regime == "bull_market":
                # Slightly increase position sizes in bull market
                adjusted_kelly *= 1.1
            elif regime == "sideways":
                # Normal position sizes in sideways market
                adjusted_kelly *= 0.9
            
            return adjusted_kelly
            
        except Exception as e:
            self.logger.debug(f"Error adjusting Kelly for regime: {e}")
            return kelly_fraction
    
    def _apply_leverage_constraints(self, kelly_weight: float, regime: str) -> float:
        """Apply leverage constraints to Kelly weight."""
        try:
            # Get regime-specific leverage limits
            max_leverage = self._get_regime_leverage_limit(regime)
            
            # Apply leverage constraint
            if abs(kelly_weight) > max_leverage:
                kelly_weight = np.sign(kelly_weight) * max_leverage
            
            return kelly_weight
            
        except Exception as e:
            self.logger.debug(f"Error applying leverage constraints: {e}")
            return kelly_weight
    
    def _get_regime_leverage_limit(self, regime: str) -> float:
        """Get leverage limit for current regime."""
        try:
            if regime == "crisis":
                return 0.5  # Very low leverage in crisis
            elif regime == "high_volatility":
                return 0.8  # Low leverage in high vol
            elif regime == "bear_market":
                return 1.0  # Moderate leverage in bear market
            elif regime == "sideways":
                return 1.5  # Normal leverage in sideways market
            elif regime == "bull_market":
                return self.max_leverage  # Full leverage in bull market
            else:
                return 1.0  # Default leverage
                
        except Exception as e:
            self.logger.debug(f"Error getting regime leverage limit: {e}")
            return 1.0
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        try:
            total_weight = sum(abs(w) for w in weights.values())
            
            if total_weight > 0:
                normalized_weights = {}
                for symbol, weight in weights.items():
                    normalized_weights[symbol] = weight / total_weight
                return normalized_weights
            else:
                return weights
                
        except Exception as e:
            self.logger.debug(f"Error normalizing weights: {e}")
            return weights
    
    def update_return_history(self, symbol: str, returns: List[float]):
        """Update return history for a symbol."""
        try:
            if symbol not in self.return_history:
                self.return_history[symbol] = []
            
            self.return_history[symbol].extend(returns)
            
            # Keep only recent history
            if len(self.return_history[symbol]) > 252:
                self.return_history[symbol] = self.return_history[symbol][-252:]
            
            # Update volatility
            if len(self.return_history[symbol]) > 30:
                self.volatility_history[symbol] = np.std(self.return_history[symbol][-30:])
                
        except Exception as e:
            self.logger.debug(f"Error updating return history for {symbol}: {e}")
    
    def calculate_optimal_kelly(self, returns: np.ndarray, 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate optimal Kelly fraction using numerical optimization."""
        try:
            if len(returns) < 30:
                return 0.0
            
            # Objective function: maximize geometric mean return
            def objective(kelly_fraction):
                # Apply Kelly fraction to returns
                kelly_returns = kelly_fraction * returns
                
                # Calculate geometric mean
                geometric_mean = np.exp(np.mean(np.log(1 + kelly_returns)))
                
                return -geometric_mean  # Minimize negative geometric mean
            
            # Constraints
            constraints = [
                {'type': 'ineq', 'fun': lambda x: 1 - x},  # Kelly <= 1
                {'type': 'ineq', 'fun': lambda x: x + 1}   # Kelly >= -1
            ]
            
            # Initial guess
            initial_kelly = 0.1
            
            # Optimize
            result = minimize(objective, initial_kelly, 
                           method='SLSQP', constraints=constraints,
                           bounds=[(-1.0, 1.0)])
            
            if result.success:
                return result.x[0]
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Error calculating optimal Kelly: {e}")
            return 0.0
    
    def calculate_kelly_statistics(self, symbol: str) -> Dict[str, float]:
        """Calculate Kelly-related statistics for a symbol."""
        stats = {}
        
        try:
            returns = self.return_history.get(symbol, [])
            
            if len(returns) < 30:
                return stats
            
            returns = np.array(returns)
            
            # Basic statistics
            stats['mean_return'] = np.mean(returns)
            stats['volatility'] = np.std(returns)
            stats['sharpe_ratio'] = stats['mean_return'] / stats['volatility'] if stats['volatility'] > 0 else 0
            
            # Kelly statistics
            risk_free_rate = 0.02 / 252
            full_kelly = (stats['mean_return'] - risk_free_rate) / (stats['volatility'] ** 2)
            stats['full_kelly'] = full_kelly
            stats['fractional_kelly'] = full_kelly * self.fractional_kelly
            
            # Risk metrics
            stats['var_95'] = np.percentile(returns, 5)
            stats['max_drawdown'] = self._calculate_max_drawdown(returns)
            
            # Win rate
            stats['win_rate'] = np.mean(returns > 0)
            
        except Exception as e:
            self.logger.debug(f"Error calculating Kelly statistics for {symbol}: {e}")
            
        return stats
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            return np.min(drawdown)
        except Exception:
            return 0.0
    
    def get_portfolio_kelly_stats(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Get portfolio-level Kelly statistics."""
        stats = {}
        
        try:
            total_kelly = 0.0
            weighted_sharpe = 0.0
            weighted_vol = 0.0
            
            for symbol, weight in weights.items():
                symbol_stats = self.calculate_kelly_statistics(symbol)
                
                if symbol_stats:
                    total_kelly += abs(weight * symbol_stats.get('fractional_kelly', 0))
                    weighted_sharpe += weight * symbol_stats.get('sharpe_ratio', 0)
                    weighted_vol += weight * symbol_stats.get('volatility', 0)
            
            stats['total_kelly_exposure'] = total_kelly
            stats['weighted_sharpe'] = weighted_sharpe
            stats['weighted_volatility'] = weighted_vol
            
            # Portfolio Kelly ratio
            if weighted_vol > 0:
                stats['portfolio_kelly_ratio'] = weighted_sharpe / weighted_vol
            else:
                stats['portfolio_kelly_ratio'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Kelly stats: {e}")
            
        return stats
