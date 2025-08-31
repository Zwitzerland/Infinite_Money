"""
Cost Model Module

Implements trading cost models:
- Commission models
- Slippage models
- Market impact models
- Opportunity cost models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging


class CostModel:
    """
    Trading cost modeling system.
    
    Implements various cost models:
    - Commission models
    - Slippage models
    - Market impact models
    - Opportunity cost models
    """
    
    def __init__(self, commission_rate: float = 0.001, slippage_model: str = "constant"):
        """
        Initialize cost model.
        
        Args:
            commission_rate: Commission rate as fraction of trade value
            slippage_model: Type of slippage model to use
        """
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_params = {}
        self.market_impact_params = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize slippage parameters
        self._initialize_slippage_params()
        
    def _initialize_slippage_params(self):
        """Initialize slippage model parameters."""
        try:
            # Constant slippage model
            self.slippage_params['constant'] = {
                'bps': 5.0  # 5 basis points
            }
            
            # Linear slippage model
            self.slippage_params['linear'] = {
                'base_bps': 2.0,
                'slope': 0.1  # bps per $1M traded
            }
            
            # Square root slippage model
            self.slippage_params['sqrt'] = {
                'base_bps': 1.0,
                'scale': 0.5  # bps per sqrt($1M traded)
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing slippage parameters: {e}")
    
    def apply_costs(self, trade_value: float, symbol: str) -> float:
        """
        Apply trading costs to trade value.
        
        Args:
            trade_value: Original trade value
            symbol: Trading symbol
            
        Returns:
            Trade value after costs
        """
        try:
            # Calculate various costs
            commission_cost = self._calculate_commission(trade_value)
            slippage_cost = self._calculate_slippage(trade_value, symbol)
            market_impact_cost = self._calculate_market_impact(trade_value, symbol)
            
            # Total cost
            total_cost = commission_cost + slippage_cost + market_impact_cost
            
            # Apply cost to trade value
            adjusted_value = trade_value - total_cost
            
            return adjusted_value
            
        except Exception as e:
            self.logger.error(f"Error applying costs: {e}")
            return trade_value
    
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission cost."""
        try:
            return abs(trade_value) * self.commission_rate
        except Exception as e:
            self.logger.debug(f"Error calculating commission: {e}")
            return 0.0
    
    def _calculate_slippage(self, trade_value: float, symbol: str) -> float:
        """Calculate slippage cost."""
        try:
            if self.slippage_model == "constant":
                return self._constant_slippage(trade_value)
            elif self.slippage_model == "linear":
                return self._linear_slippage(trade_value)
            elif self.slippage_model == "sqrt":
                return self._sqrt_slippage(trade_value)
            else:
                return self._constant_slippage(trade_value)
                
        except Exception as e:
            self.logger.debug(f"Error calculating slippage: {e}")
            return 0.0
    
    def _constant_slippage(self, trade_value: float) -> float:
        """Calculate constant slippage."""
        try:
            bps = self.slippage_params['constant']['bps']
            return abs(trade_value) * (bps / 10000)
        except Exception as e:
            self.logger.debug(f"Error calculating constant slippage: {e}")
            return 0.0
    
    def _linear_slippage(self, trade_value: float) -> float:
        """Calculate linear slippage."""
        try:
            params = self.slippage_params['linear']
            base_bps = params['base_bps']
            slope = params['slope']
            
            # Convert trade value to millions
            trade_millions = abs(trade_value) / 1000000
            
            # Linear slippage: base + slope * trade_size
            total_bps = base_bps + slope * trade_millions
            
            return abs(trade_value) * (total_bps / 10000)
            
        except Exception as e:
            self.logger.debug(f"Error calculating linear slippage: {e}")
            return 0.0
    
    def _sqrt_slippage(self, trade_value: float) -> float:
        """Calculate square root slippage."""
        try:
            params = self.slippage_params['sqrt']
            base_bps = params['base_bps']
            scale = params['scale']
            
            # Convert trade value to millions
            trade_millions = abs(trade_value) / 1000000
            
            # Square root slippage: base + scale * sqrt(trade_size)
            total_bps = base_bps + scale * np.sqrt(trade_millions)
            
            return abs(trade_value) * (total_bps / 10000)
            
        except Exception as e:
            self.logger.debug(f"Error calculating sqrt slippage: {e}")
            return 0.0
    
    def _calculate_market_impact(self, trade_value: float, symbol: str) -> float:
        """Calculate market impact cost."""
        try:
            # Get market impact parameters for symbol
            params = self.market_impact_params.get(symbol, {})
            
            # Default parameters
            alpha = params.get('alpha', 0.1)  # Impact coefficient
            beta = params.get('beta', 0.5)   # Impact exponent
            
            # Market impact = alpha * (trade_size)^beta
            trade_size = abs(trade_value)
            market_impact = alpha * (trade_size ** beta)
            
            return market_impact
            
        except Exception as e:
            self.logger.debug(f"Error calculating market impact: {e}")
            return 0.0
    
    def calculate_total_cost(self, trade_value: float, symbol: str) -> Dict[str, float]:
        """Calculate breakdown of all trading costs."""
        try:
            costs = {}
            
            # Commission
            costs['commission'] = self._calculate_commission(trade_value)
            
            # Slippage
            costs['slippage'] = self._calculate_slippage(trade_value, symbol)
            
            # Market impact
            costs['market_impact'] = self._calculate_market_impact(trade_value, symbol)
            
            # Total cost
            costs['total'] = costs['commission'] + costs['slippage'] + costs['market_impact']
            
            # Cost as percentage of trade value
            if trade_value != 0:
                costs['total_pct'] = costs['total'] / abs(trade_value)
            else:
                costs['total_pct'] = 0.0
            
            return costs
            
        except Exception as e:
            self.logger.error(f"Error calculating total cost: {e}")
            return {}
    
    def update_slippage_params(self, model: str, params: Dict):
        """Update slippage model parameters."""
        try:
            if model in self.slippage_params:
                self.slippage_params[model].update(params)
            else:
                self.slippage_params[model] = params
                
        except Exception as e:
            self.logger.error(f"Error updating slippage parameters: {e}")
    
    def update_market_impact_params(self, symbol: str, params: Dict):
        """Update market impact parameters for a symbol."""
        try:
            self.market_impact_params[symbol] = params
        except Exception as e:
            self.logger.error(f"Error updating market impact parameters: {e}")
    
    def set_commission_rate(self, commission_rate: float):
        """Set commission rate."""
        try:
            self.commission_rate = commission_rate
        except Exception as e:
            self.logger.error(f"Error setting commission rate: {e}")
    
    def set_slippage_model(self, model: str):
        """Set slippage model type."""
        try:
            if model in self.slippage_params:
                self.slippage_model = model
            else:
                self.logger.warning(f"Unknown slippage model: {model}")
                
        except Exception as e:
            self.logger.error(f"Error setting slippage model: {e}")
    
    def get_cost_statistics(self, trades: List[Dict]) -> Dict[str, float]:
        """Get cost statistics from trade history."""
        try:
            stats = {}
            
            if not trades:
                return stats
            
            total_commission = 0.0
            total_slippage = 0.0
            total_market_impact = 0.0
            total_cost = 0.0
            total_value = 0.0
            
            for trade in trades:
                trade_value = trade.get('value', 0.0)
                symbol = trade.get('symbol', '')
                
                costs = self.calculate_total_cost(trade_value, symbol)
                
                total_commission += costs.get('commission', 0.0)
                total_slippage += costs.get('slippage', 0.0)
                total_market_impact += costs.get('market_impact', 0.0)
                total_cost += costs.get('total', 0.0)
                total_value += abs(trade_value)
            
            # Calculate statistics
            stats['total_commission'] = total_commission
            stats['total_slippage'] = total_slippage
            stats['total_market_impact'] = total_market_impact
            stats['total_cost'] = total_cost
            
            if total_value > 0:
                stats['avg_commission_bps'] = (total_commission / total_value) * 10000
                stats['avg_slippage_bps'] = (total_slippage / total_value) * 10000
                stats['avg_market_impact_bps'] = (total_market_impact / total_value) * 10000
                stats['avg_total_cost_bps'] = (total_cost / total_value) * 10000
            else:
                stats['avg_commission_bps'] = 0.0
                stats['avg_slippage_bps'] = 0.0
                stats['avg_market_impact_bps'] = 0.0
                stats['avg_total_cost_bps'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating cost statistics: {e}")
            return {}
    
    def optimize_trade_size(self, target_value: float, symbol: str) -> float:
        """Optimize trade size to minimize costs."""
        try:
            # Simple optimization: find trade size that minimizes cost per dollar
            best_size = target_value
            min_cost_ratio = float('inf')
            
            # Test different trade sizes
            for size_factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
                test_size = target_value * size_factor
                costs = self.calculate_total_cost(test_size, symbol)
                
                cost_ratio = costs.get('total_pct', 0.0)
                if cost_ratio < min_cost_ratio:
                    min_cost_ratio = cost_ratio
                    best_size = test_size
            
            return best_size
            
        except Exception as e:
            self.logger.error(f"Error optimizing trade size: {e}")
            return target_value
