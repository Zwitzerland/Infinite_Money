"""
Execution Router Module

Implements execution routing and order management:
- Smart order routing
- Liquidity analysis
- Market impact modeling
- Order splitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging


class ExecutionRouter:
    """
    Smart order routing and execution management.
    
    Implements execution strategies:
    - Smart order routing
    - Liquidity analysis
    - Market impact modeling
    - Order splitting
    """
    
    def __init__(self, max_order_size: float = 0.1, min_order_size: float = 0.001):
        """
        Initialize execution router.
        
        Args:
            max_order_size: Maximum order size as fraction of portfolio
            min_order_size: Minimum order size as fraction of portfolio
        """
        self.max_order_size = max_order_size
        self.min_order_size = min_order_size
        self.liquidity_data = {}
        self.market_impact_model = {}
        self.logger = logging.getLogger(__name__)
        
    def route_orders(self, target_weights: Dict[str, float], 
                    current_weights: Dict[str, float],
                    portfolio_value: float) -> List[Dict]:
        """
        Route orders for execution.
        
        Args:
            target_weights: Target portfolio weights
            current_weights: Current portfolio weights
            portfolio_value: Total portfolio value
            
        Returns:
            List of order instructions
        """
        try:
            orders = []
            
            for symbol in set(target_weights.keys()) | set(current_weights.keys()):
                target_weight = target_weights.get(symbol, 0.0)
                current_weight = current_weights.get(symbol, 0.0)
                
                # Calculate required trade
                trade_weight = target_weight - current_weight
                
                if abs(trade_weight) > self.min_order_size:
                    # Split large orders
                    order_splits = self._split_order(symbol, trade_weight, portfolio_value)
                    
                    for split in order_splits:
                        orders.append({
                            'symbol': symbol,
                            'weight': split,
                            'value': split * portfolio_value,
                            'side': 'buy' if split > 0 else 'sell',
                            'urgency': self._determine_urgency(symbol, abs(split))
                        })
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error routing orders: {e}")
            return []
    
    def _split_order(self, symbol: str, trade_weight: float, 
                    portfolio_value: float) -> List[float]:
        """Split large orders into smaller pieces."""
        try:
            splits = []
            remaining_weight = trade_weight
            
            while abs(remaining_weight) > self.min_order_size:
                # Determine split size based on liquidity
                max_split = self._get_max_split_size(symbol, portfolio_value)
                split_size = min(abs(remaining_weight), max_split, self.max_order_size)
                
                # Apply sign
                split_size = np.sign(remaining_weight) * split_size
                
                splits.append(split_size)
                remaining_weight -= split_size
            
            return splits
            
        except Exception as e:
            self.logger.debug(f"Error splitting order for {symbol}: {e}")
            return [trade_weight]
    
    def _get_max_split_size(self, symbol: str, portfolio_value: float) -> float:
        """Get maximum split size based on liquidity."""
        try:
            # Get liquidity data for symbol
            liquidity = self.liquidity_data.get(symbol, {})
            
            # Default liquidity assumptions
            avg_volume = liquidity.get('avg_volume', 1000000)  # Default 1M shares
            avg_price = liquidity.get('avg_price', 50.0)  # Default $50
            
            # Calculate maximum trade size (1% of average volume)
            max_trade_value = avg_volume * avg_price * 0.01
            max_trade_weight = max_trade_value / portfolio_value
            
            return min(max_trade_weight, self.max_order_size)
            
        except Exception as e:
            self.logger.debug(f"Error getting max split size for {symbol}: {e}")
            return self.max_order_size
    
    def _determine_urgency(self, symbol: str, order_size: float) -> str:
        """Determine order urgency level."""
        try:
            # Get market impact estimate
            market_impact = self._estimate_market_impact(symbol, order_size)
            
            if market_impact > 0.05:  # 5% market impact
                return 'low'  # Execute slowly
            elif market_impact > 0.02:  # 2% market impact
                return 'medium'  # Normal execution
            else:
                return 'high'  # Execute quickly
                
        except Exception as e:
            self.logger.debug(f"Error determining urgency for {symbol}: {e}")
            return 'medium'
    
    def _estimate_market_impact(self, symbol: str, order_size: float) -> float:
        """Estimate market impact of order."""
        try:
            # Get market impact model parameters
            model = self.market_impact_model.get(symbol, {})
            
            # Default parameters (square-root model)
            alpha = model.get('alpha', 0.1)  # Impact coefficient
            beta = model.get('beta', 0.5)   # Impact exponent
            
            # Market impact = alpha * (order_size)^beta
            market_impact = alpha * (order_size ** beta)
            
            return min(market_impact, 0.1)  # Cap at 10%
            
        except Exception as e:
            self.logger.debug(f"Error estimating market impact for {symbol}: {e}")
            return 0.02  # Default 2% impact
    
    def update_liquidity_data(self, symbol: str, liquidity_data: Dict):
        """Update liquidity data for a symbol."""
        try:
            self.liquidity_data[symbol] = liquidity_data
        except Exception as e:
            self.logger.error(f"Error updating liquidity data for {symbol}: {e}")
    
    def update_market_impact_model(self, symbol: str, model_params: Dict):
        """Update market impact model for a symbol."""
        try:
            self.market_impact_model[symbol] = model_params
        except Exception as e:
            self.logger.error(f"Error updating market impact model for {symbol}: {e}")
    
    def get_execution_statistics(self, orders: List[Dict]) -> Dict[str, float]:
        """Get execution statistics."""
        try:
            stats = {}
            
            if not orders:
                return stats
            
            # Order statistics
            total_orders = len(orders)
            buy_orders = len([o for o in orders if o['side'] == 'buy'])
            sell_orders = len([o for o in orders if o['side'] == 'sell'])
            
            stats['total_orders'] = total_orders
            stats['buy_orders'] = buy_orders
            stats['sell_orders'] = sell_orders
            
            # Value statistics
            total_value = sum(abs(o['value']) for o in orders)
            avg_order_value = total_value / total_orders if total_orders > 0 else 0
            
            stats['total_value'] = total_value
            stats['avg_order_value'] = avg_order_value
            
            # Urgency distribution
            urgency_counts = {}
            for order in orders:
                urgency = order['urgency']
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            for urgency, count in urgency_counts.items():
                stats[f'{urgency}_urgency_pct'] = count / total_orders
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating execution statistics: {e}")
            return {}
    
    def optimize_execution(self, orders: List[Dict], 
                          market_conditions: Dict) -> List[Dict]:
        """Optimize execution based on market conditions."""
        try:
            optimized_orders = []
            
            for order in orders:
                optimized_order = order.copy()
                
                # Adjust based on market conditions
                volatility = market_conditions.get('volatility', 0.2)
                spread = market_conditions.get('spread', 0.001)
                
                # Adjust urgency based on market conditions
                if volatility > 0.3:  # High volatility
                    optimized_order['urgency'] = 'low'
                elif spread > 0.005:  # Wide spreads
                    optimized_order['urgency'] = 'low'
                elif volatility < 0.1:  # Low volatility
                    optimized_order['urgency'] = 'high'
                
                optimized_orders.append(optimized_order)
            
            return optimized_orders
            
        except Exception as e:
            self.logger.error(f"Error optimizing execution: {e}")
            return orders

