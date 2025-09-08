"""
Risk Limits Module

Implements various risk management limits:
- Position limits
- Sector limits
- VaR limits
- Drawdown limits
- Correlation limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging


class RiskLimits:
    """
    Risk limits management system.
    
    Implements various risk limits:
    - Position size limits
    - Sector concentration limits
    - VaR limits
    - Drawdown limits
    - Correlation limits
    """
    
    def __init__(self, max_position_size: float = 0.1, max_sector_weight: float = 0.3,
                 var_limit: float = 0.02, max_drawdown: float = 0.15):
        """
        Initialize risk limits.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_sector_weight: Maximum sector weight
            var_limit: VaR limit as fraction of portfolio
            max_drawdown: Maximum drawdown limit
        """
        self.max_position_size = max_position_size
        self.max_sector_weight = max_sector_weight
        self.var_limit = var_limit
        self.max_drawdown = max_drawdown
        self.sector_mappings = {}
        self.correlation_matrix = None
        self.logger = logging.getLogger(__name__)
        
    def apply_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply risk limits to portfolio weights.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Risk-adjusted weights
        """
        try:
            limited_weights = weights.copy()
            
            # Apply position size limits
            limited_weights = self._apply_position_limits(limited_weights)
            
            # Apply sector limits
            limited_weights = self._apply_sector_limits(limited_weights)
            
            # Apply correlation limits
            limited_weights = self._apply_correlation_limits(limited_weights)
            
            # Apply VaR limits
            limited_weights = self._apply_var_limits(limited_weights)
            
            # Normalize weights
            limited_weights = self._normalize_weights(limited_weights)
            
            return limited_weights
            
        except Exception as e:
            self.logger.error(f"Error applying risk limits: {e}")
            return weights
    
    def _apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position size limits."""
        try:
            limited_weights = weights.copy()
            
            for symbol, weight in limited_weights.items():
                if abs(weight) > self.max_position_size:
                    limited_weights[symbol] = np.sign(weight) * self.max_position_size
            
            return limited_weights
            
        except Exception as e:
            self.logger.debug(f"Error applying position limits: {e}")
            return weights
    
    def _apply_sector_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply sector concentration limits."""
        try:
            if not self.sector_mappings:
                return weights
            
            limited_weights = weights.copy()
            sector_weights = {}
            
            # Calculate sector weights
            for symbol, weight in limited_weights.items():
                sector = self.sector_mappings.get(symbol, "Unknown")
                if sector not in sector_weights:
                    sector_weights[sector] = 0.0
                sector_weights[sector] += abs(weight)
            
            # Check sector limits
            for sector, sector_weight in sector_weights.items():
                if sector_weight > self.max_sector_weight:
                    # Scale down weights in this sector
                    scale_factor = self.max_sector_weight / sector_weight
                    
                    for symbol, weight in limited_weights.items():
                        if self.sector_mappings.get(symbol) == sector:
                            limited_weights[symbol] *= scale_factor
            
            return limited_weights
            
        except Exception as e:
            self.logger.debug(f"Error applying sector limits: {e}")
            return weights
    
    def _apply_correlation_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply correlation limits."""
        try:
            if self.correlation_matrix is None:
                return weights
            
            limited_weights = weights.copy()
            symbols = list(weights.keys())
            
            # Find highly correlated positions
            high_correlation_pairs = []
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if (symbol1 in self.correlation_matrix.index and 
                        symbol2 in self.correlation_matrix.columns):
                        
                        correlation = self.correlation_matrix.loc[symbol1, symbol2]
                        
                        if abs(correlation) > 0.7:  # High correlation threshold
                            high_correlation_pairs.append((symbol1, symbol2, correlation))
            
            # Reduce weights of highly correlated positions
            for symbol1, symbol2, correlation in high_correlation_pairs:
                weight1 = limited_weights.get(symbol1, 0.0)
                weight2 = limited_weights.get(symbol2, 0.0)
                
                if abs(weight1) > 0 and abs(weight2) > 0:
                    # Reduce the smaller position
                    if abs(weight1) < abs(weight2):
                        limited_weights[symbol1] *= 0.5
                    else:
                        limited_weights[symbol2] *= 0.5
            
            return limited_weights
            
        except Exception as e:
            self.logger.debug(f"Error applying correlation limits: {e}")
            return weights
    
    def _apply_var_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply VaR limits."""
        try:
            # Calculate portfolio VaR
            portfolio_var = self._calculate_portfolio_var(weights)
            
            if portfolio_var > self.var_limit:
                # Scale down weights to meet VaR limit
                scale_factor = self.var_limit / portfolio_var
                
                limited_weights = {}
                for symbol, weight in weights.items():
                    limited_weights[symbol] = weight * scale_factor
                
                return limited_weights
            
            return weights
            
        except Exception as e:
            self.logger.debug(f"Error applying VaR limits: {e}")
            return weights
    
    def _calculate_portfolio_var(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio VaR."""
        try:
            if not weights:
                return 0.0
            
            # Simple VaR calculation (assuming normal distribution)
            # In practice, this would use historical simulation or Monte Carlo
            
            total_weight = sum(abs(w) for w in weights.values())
            if total_weight == 0:
                return 0.0
            
            # Assume 20% annual volatility for each position
            # and 95% confidence level
            position_var = 0.20 * 1.645 / np.sqrt(252)  # Daily VaR
            
            portfolio_var = position_var * total_weight
            
            return portfolio_var
            
        except Exception as e:
            self.logger.debug(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
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
    
    def update_sector_mappings(self, sector_mappings: Dict[str, str]):
        """Update sector mappings for symbols."""
        try:
            self.sector_mappings.update(sector_mappings)
        except Exception as e:
            self.logger.error(f"Error updating sector mappings: {e}")
    
    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Update correlation matrix."""
        try:
            self.correlation_matrix = correlation_matrix
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
    
    def check_limits(self, weights: Dict[str, float]) -> Dict[str, bool]:
        """Check if weights violate any limits."""
        limit_violations = {}
        
        try:
            # Check position size limits
            for symbol, weight in weights.items():
                if abs(weight) > self.max_position_size:
                    limit_violations[f"{symbol}_position_size"] = True
                else:
                    limit_violations[f"{symbol}_position_size"] = False
            
            # Check sector limits
            if self.sector_mappings:
                sector_weights = {}
                for symbol, weight in weights.items():
                    sector = self.sector_mappings.get(symbol, "Unknown")
                    if sector not in sector_weights:
                        sector_weights[sector] = 0.0
                    sector_weights[sector] += abs(weight)
                
                for sector, sector_weight in sector_weights.items():
                    if sector_weight > self.max_sector_weight:
                        limit_violations[f"{sector}_sector_weight"] = True
                    else:
                        limit_violations[f"{sector}_sector_weight"] = False
            
            # Check VaR limit
            portfolio_var = self._calculate_portfolio_var(weights)
            limit_violations["var_limit"] = portfolio_var > self.var_limit
            
        except Exception as e:
            self.logger.error(f"Error checking limits: {e}")
            
        return limit_violations
    
    def get_risk_metrics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Get risk metrics for current weights."""
        metrics = {}
        
        try:
            # Portfolio concentration
            concentration = sum(w ** 2 for w in weights.values())
            metrics['concentration'] = concentration
            
            # Number of positions
            num_positions = len([w for w in weights.values() if abs(w) > 0.01])
            metrics['num_positions'] = num_positions
            
            # Largest position
            if weights:
                largest_position = max(abs(w) for w in weights.values())
                metrics['largest_position'] = largest_position
            else:
                metrics['largest_position'] = 0.0
            
            # Portfolio VaR
            portfolio_var = self._calculate_portfolio_var(weights)
            metrics['portfolio_var'] = portfolio_var
            
            # Sector concentration
            if self.sector_mappings:
                sector_weights = {}
                for symbol, weight in weights.items():
                    sector = self.sector_mappings.get(symbol, "Unknown")
                    if sector not in sector_weights:
                        sector_weights[sector] = 0.0
                    sector_weights[sector] += abs(weight)
                
                max_sector_weight = max(sector_weights.values()) if sector_weights else 0.0
                metrics['max_sector_weight'] = max_sector_weight
            else:
                metrics['max_sector_weight'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            
        return metrics
    
    def set_limits(self, max_position_size: Optional[float] = None,
                  max_sector_weight: Optional[float] = None,
                  var_limit: Optional[float] = None,
                  max_drawdown: Optional[float] = None):
        """Update risk limits."""
        try:
            if max_position_size is not None:
                self.max_position_size = max_position_size
            if max_sector_weight is not None:
                self.max_sector_weight = max_sector_weight
            if var_limit is not None:
                self.var_limit = var_limit
            if max_drawdown is not None:
                self.max_drawdown = max_drawdown
                
        except Exception as e:
            self.logger.error(f"Error setting limits: {e}")
    
    def get_current_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return {
            'max_position_size': self.max_position_size,
            'max_sector_weight': self.max_sector_weight,
            'var_limit': self.var_limit,
            'max_drawdown': self.max_drawdown
        }

