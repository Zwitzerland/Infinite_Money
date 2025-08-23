"""
L-VaR (Liquidity-adjusted Value at Risk) Implementation
Gates positions so gross leverage collapses when depth thins - no mark-to-mid fantasies.

References:
- "Liquidity-adjusted Intraday Value at Risk modeling and risk management" (ScienceDirect)
- LI-VaR: Liquidity-Inclusive VaR that accounts for market microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from scipy.stats import norm, t as t_dist
from scipy.optimize import minimize_scalar

from ...utils.logger import get_logger


@dataclass
class LVaRConfig:
    """Configuration for L-VaR calculations."""
    confidence_level: float = 0.95     # VaR confidence level
    holding_period: int = 1            # Holding period in days
    liquidity_horizon: int = 5         # Liquidity horizon in days
    bid_ask_spread_threshold: float = 0.005  # 50bps spread threshold
    volume_lookback: int = 30          # Days for volume calculation
    depth_levels: int = 5              # Number of order book levels
    min_market_cap: float = 1e9        # Minimum market cap for liquid assets
    
    # Liquidity adjustment factors
    spread_penalty: float = 2.0        # Penalty factor for wide spreads
    volume_penalty: float = 1.5        # Penalty factor for low volume
    depth_penalty: float = 3.0         # Penalty factor for shallow depth
    
    # Emergency thresholds
    critical_spread: float = 0.02      # 200bps - critical spread level
    critical_volume_ratio: float = 0.1 # 10% of normal volume
    depth_collapse_threshold: float = 0.05  # 5% depth collapse


class LiquidityMetrics:
    """Container for liquidity metrics."""
    
    def __init__(self):
        self.bid_ask_spread = 0.0
        self.relative_spread = 0.0
        self.volume_ratio = 1.0
        self.depth_ratio = 1.0
        self.market_impact = 0.0
        self.turnover_ratio = 0.0
        self.amihud_ratio = 0.0
        self.liquidity_score = 1.0


class LVaRCalculator:
    """
    Liquidity-adjusted Value at Risk (L-VaR) Calculator
    
    L-VaR = VaR * √(1 + λ * LC)
    where λ is liquidity cost and LC is liquidity cost factor.
    """
    
    def __init__(self, config: LVaRConfig):
        """Initialize L-VaR calculator."""
        self.config = config
        self.logger = get_logger("LVaRCalculator")
        
    def compute_lvar(self, 
                    returns: np.ndarray,
                    weights: np.ndarray,
                    liquidity_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Liquidity-adjusted VaR for portfolio.
        
        Args:
            returns: Historical returns matrix (T x N)
            weights: Portfolio weights (N,)
            liquidity_data: Liquidity metrics for each asset
            
        Returns:
            lvar_value: L-VaR value
            diagnostics: Detailed liquidity diagnostics
        """
        try:
            # Compute base VaR
            base_var = self._compute_base_var(returns, weights)
            
            # Compute liquidity adjustment factors
            liquidity_metrics = self._compute_liquidity_metrics(liquidity_data)
            
            # Aggregate portfolio liquidity adjustment
            portfolio_adjustment = self._compute_portfolio_liquidity_adjustment(
                weights, liquidity_metrics
            )
            
            # Apply liquidity adjustment to VaR
            lvar_value = base_var * portfolio_adjustment
            
            # Detailed diagnostics
            diagnostics = {
                "base_var": base_var,
                "liquidity_adjustment": portfolio_adjustment,
                "lvar": lvar_value,
                "individual_metrics": liquidity_metrics,
                "portfolio_spread": self._compute_portfolio_spread(weights, liquidity_metrics),
                "portfolio_depth": self._compute_portfolio_depth(weights, liquidity_metrics),
                "liquidity_warning": portfolio_adjustment > 1.5
            }
            
            self.logger.debug(f"L-VaR: {lvar_value:.4f} (Base VaR: {base_var:.4f}, Adjustment: {portfolio_adjustment:.2f})")
            
            return lvar_value, diagnostics
            
        except Exception as e:
            self.logger.error(f"Error computing L-VaR: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def _compute_base_var(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Compute base VaR without liquidity adjustment."""
        # Portfolio returns
        portfolio_returns = returns @ weights
        
        # Parametric VaR assuming normal distribution
        mean_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        
        # Adjust for holding period
        volatility_adj = volatility * np.sqrt(self.config.holding_period)
        
        # VaR calculation
        z_score = norm.ppf(1 - self.config.confidence_level)
        var_value = -(mean_return * self.config.holding_period + z_score * volatility_adj)
        
        return max(var_value, 0.0)
    
    def _compute_liquidity_metrics(self, liquidity_data: Dict[str, Any]) -> List[LiquidityMetrics]:
        """Compute liquidity metrics for each asset."""
        metrics_list = []
        
        for asset_data in liquidity_data.get("assets", []):
            metrics = LiquidityMetrics()
            
            # Bid-ask spread metrics
            metrics.bid_ask_spread = asset_data.get("bid_ask_spread", 0.001)
            metrics.relative_spread = metrics.bid_ask_spread / asset_data.get("mid_price", 1.0)
            
            # Volume metrics
            current_volume = asset_data.get("volume", 0)
            avg_volume = asset_data.get("avg_volume", 1)
            metrics.volume_ratio = current_volume / (avg_volume + 1e-8)
            
            # Market depth metrics
            total_depth = asset_data.get("bid_depth", 0) + asset_data.get("ask_depth", 0)
            normal_depth = asset_data.get("normal_depth", 1)
            metrics.depth_ratio = total_depth / (normal_depth + 1e-8)
            
            # Market impact estimation
            metrics.market_impact = self._estimate_market_impact(asset_data)
            
            # Turnover ratio
            market_cap = asset_data.get("market_cap", 1e9)
            metrics.turnover_ratio = current_volume / (market_cap + 1e-8)
            
            # Amihud illiquidity ratio
            if current_volume > 0:
                price_change = asset_data.get("price_change", 0)
                metrics.amihud_ratio = abs(price_change) / current_volume
            else:
                metrics.amihud_ratio = 1.0  # High illiquidity
            
            # Overall liquidity score
            metrics.liquidity_score = self._compute_liquidity_score(metrics, asset_data)
            
            metrics_list.append(metrics)
        
        return metrics_list
    
    def _estimate_market_impact(self, asset_data: Dict[str, Any]) -> float:
        """Estimate market impact using square root law."""
        volume = asset_data.get("volume", 1)
        avg_volume = asset_data.get("avg_volume", 1)
        volatility = asset_data.get("volatility", 0.01)
        
        # Participation rate
        participation_rate = 0.1  # Assume 10% participation rate
        
        # Square root market impact model
        market_impact = 0.1 * volatility * np.sqrt(participation_rate * avg_volume / (volume + 1e-8))
        
        return market_impact
    
    def _compute_liquidity_score(self, metrics: LiquidityMetrics, asset_data: Dict[str, Any]) -> float:
        """Compute overall liquidity score (0-1, higher is more liquid)."""
        # Spread component (inverse relationship)
        spread_score = 1.0 / (1.0 + metrics.relative_spread * 100)
        
        # Volume component
        volume_score = min(1.0, metrics.volume_ratio)
        
        # Depth component
        depth_score = min(1.0, metrics.depth_ratio)
        
        # Market cap component
        market_cap = asset_data.get("market_cap", 0)
        cap_score = min(1.0, market_cap / self.config.min_market_cap)
        
        # Amihud component (inverse relationship)
        amihud_score = 1.0 / (1.0 + metrics.amihud_ratio * 1000)
        
        # Weighted average
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Spread, volume, depth, cap, amihud
        scores = [spread_score, volume_score, depth_score, cap_score, amihud_score]
        
        liquidity_score = sum(w * s for w, s in zip(weights, scores))
        
        return liquidity_score
    
    def _compute_portfolio_liquidity_adjustment(self, 
                                              weights: np.ndarray,
                                              liquidity_metrics: List[LiquidityMetrics]) -> float:
        """Compute portfolio-level liquidity adjustment factor."""
        if len(liquidity_metrics) != len(weights):
            return 1.0
        
        total_adjustment = 0.0
        
        for i, (weight, metrics) in enumerate(zip(weights, liquidity_metrics)):
            asset_weight = abs(weight)  # Use absolute weight for risk
            
            # Individual asset adjustments
            spread_adj = 1.0 + self.config.spread_penalty * metrics.relative_spread
            volume_adj = 1.0 + self.config.volume_penalty * max(0, 1 - metrics.volume_ratio)
            depth_adj = 1.0 + self.config.depth_penalty * max(0, 1 - metrics.depth_ratio)
            
            # Combined adjustment for this asset
            asset_adjustment = spread_adj * volume_adj * depth_adj
            
            # Weight by portfolio allocation
            total_adjustment += asset_weight * asset_adjustment
        
        # Ensure minimum adjustment of 1.0
        return max(total_adjustment, 1.0)
    
    def _compute_portfolio_spread(self, 
                                weights: np.ndarray,
                                liquidity_metrics: List[LiquidityMetrics]) -> float:
        """Compute portfolio-weighted bid-ask spread."""
        if len(liquidity_metrics) != len(weights):
            return 0.0
        
        weighted_spread = 0.0
        total_weight = 0.0
        
        for weight, metrics in zip(weights, liquidity_metrics):
            abs_weight = abs(weight)
            weighted_spread += abs_weight * metrics.bid_ask_spread
            total_weight += abs_weight
        
        return weighted_spread / (total_weight + 1e-8)
    
    def _compute_portfolio_depth(self, 
                               weights: np.ndarray,
                               liquidity_metrics: List[LiquidityMetrics]) -> float:
        """Compute portfolio-weighted depth ratio."""
        if len(liquidity_metrics) != len(weights):
            return 1.0
        
        weighted_depth = 0.0
        total_weight = 0.0
        
        for weight, metrics in zip(weights, liquidity_metrics):
            abs_weight = abs(weight)
            weighted_depth += abs_weight * metrics.depth_ratio
            total_weight += abs_weight
        
        return weighted_depth / (total_weight + 1e-8)
    
    def check_liquidity_constraints(self, 
                                  weights: np.ndarray,
                                  liquidity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if portfolio satisfies liquidity constraints."""
        liquidity_metrics = self._compute_liquidity_metrics(liquidity_data)
        
        # Portfolio metrics
        portfolio_spread = self._compute_portfolio_spread(weights, liquidity_metrics)
        portfolio_depth = self._compute_portfolio_depth(weights, liquidity_metrics)
        
        # Constraint checks
        spread_ok = portfolio_spread <= self.config.bid_ask_spread_threshold
        depth_ok = portfolio_depth >= self.config.depth_collapse_threshold
        
        # Emergency conditions
        spread_critical = portfolio_spread >= self.config.critical_spread
        depth_critical = portfolio_depth <= self.config.depth_collapse_threshold
        
        # Volume checks
        volume_issues = []
        for i, metrics in enumerate(liquidity_metrics):
            if metrics.volume_ratio < self.config.critical_volume_ratio:
                volume_issues.append({
                    "asset_index": i,
                    "volume_ratio": metrics.volume_ratio,
                    "weight": weights[i] if i < len(weights) else 0.0
                })
        
        constraint_check = {
            "spread_constraint": spread_ok,
            "depth_constraint": depth_ok,
            "overall_satisfied": spread_ok and depth_ok,
            "portfolio_spread": portfolio_spread,
            "portfolio_depth": portfolio_depth,
            "spread_critical": spread_critical,
            "depth_critical": depth_critical,
            "volume_issues": volume_issues,
            "emergency_conditions": spread_critical or depth_critical or len(volume_issues) > 0
        }
        
        if not constraint_check["overall_satisfied"]:
            self.logger.warning(f"Liquidity constraints violated. Spread: {portfolio_spread:.4f}, Depth: {portfolio_depth:.4f}")
        
        if constraint_check["emergency_conditions"]:
            self.logger.critical("Emergency liquidity conditions detected!")
        
        return constraint_check


class LVaRRiskController:
    """
    L-VaR based risk controller for dynamic position sizing.
    
    Gates gross leverage when liquidity conditions deteriorate.
    """
    
    def __init__(self, config: LVaRConfig):
        """Initialize L-VaR risk controller."""
        self.config = config
        self.logger = get_logger("LVaRRiskController")
        self.calculator = LVaRCalculator(config)
        
        # State tracking
        self.liquidity_history = []
        self.lvar_history = []
        
    def update_liquidity_state(self, liquidity_data: Dict[str, Any]):
        """Update controller state with new liquidity observations."""
        self.liquidity_history.append(liquidity_data)
        
        # Keep only recent history
        max_history = self.config.volume_lookback * 2
        if len(self.liquidity_history) > max_history:
            self.liquidity_history = self.liquidity_history[-max_history:]
    
    def compute_leverage_cap(self, 
                           current_weights: np.ndarray,
                           liquidity_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute dynamic leverage cap based on liquidity conditions.
        
        Args:
            current_weights: Current portfolio weights
            liquidity_data: Current liquidity metrics
            
        Returns:
            leverage_cap: Maximum allowed leverage
            diagnostics: Detailed diagnostics
        """
        try:
            # Base leverage cap
            base_leverage_cap = 2.0
            
            # Compute liquidity metrics
            liquidity_metrics = self.calculator._compute_liquidity_metrics(liquidity_data)
            
            # Check constraint violations
            constraint_check = self.calculator.check_liquidity_constraints(
                current_weights, liquidity_data
            )
            
            # Compute liquidity-adjusted leverage cap
            if constraint_check["emergency_conditions"]:
                # Emergency: drastically reduce leverage
                liquidity_factor = 0.1
            elif not constraint_check["overall_satisfied"]:
                # Constraints violated: reduce leverage
                spread_factor = min(1.0, self.config.bid_ask_spread_threshold / 
                                  (constraint_check["portfolio_spread"] + 1e-8))
                depth_factor = min(1.0, constraint_check["portfolio_depth"] / 
                                 self.config.depth_collapse_threshold)
                liquidity_factor = min(spread_factor, depth_factor)
            else:
                # Normal conditions: apply gradual adjustment
                liquidity_factor = self._compute_gradual_adjustment(liquidity_metrics)
            
            # Apply leverage cap
            leverage_cap = base_leverage_cap * liquidity_factor
            
            # Minimum leverage cap
            leverage_cap = max(leverage_cap, 0.1)
            
            diagnostics = {
                "base_leverage_cap": base_leverage_cap,
                "liquidity_factor": liquidity_factor,
                "final_leverage_cap": leverage_cap,
                "constraint_check": constraint_check,
                "liquidity_metrics": liquidity_metrics
            }
            
            self.logger.info(f"Dynamic leverage cap: {leverage_cap:.2f} (factor: {liquidity_factor:.2f})")
            
            return leverage_cap, diagnostics
            
        except Exception as e:
            self.logger.error(f"Error computing leverage cap: {str(e)}")
            return 0.1, {"error": str(e)}  # Conservative fallback
    
    def _compute_gradual_adjustment(self, liquidity_metrics: List[LiquidityMetrics]) -> float:
        """Compute gradual liquidity adjustment factor."""
        if not liquidity_metrics:
            return 0.5
        
        # Average liquidity score across assets
        avg_liquidity_score = np.mean([m.liquidity_score for m in liquidity_metrics])
        
        # Exponential decay based on liquidity score
        liquidity_factor = avg_liquidity_score ** 2  # Quadratic penalty
        
        return max(liquidity_factor, 0.1)
    
    def get_position_size_limits(self, 
                               liquidity_data: Dict[str, Any]) -> Dict[int, float]:
        """
        Get individual position size limits based on liquidity.
        
        Args:
            liquidity_data: Current liquidity metrics
            
        Returns:
            position_limits: Dictionary mapping asset index to max position size
        """
        liquidity_metrics = self.calculator._compute_liquidity_metrics(liquidity_data)
        position_limits = {}
        
        for i, metrics in enumerate(liquidity_metrics):
            # Base position limit
            base_limit = 0.2  # 20% max position
            
            # Adjust based on liquidity
            if metrics.liquidity_score > 0.8:
                # High liquidity: allow full position
                limit = base_limit
            elif metrics.liquidity_score > 0.6:
                # Medium liquidity: reduce position
                limit = base_limit * 0.7
            elif metrics.liquidity_score > 0.4:
                # Low liquidity: significantly reduce
                limit = base_limit * 0.4
            else:
                # Very low liquidity: minimal position
                limit = base_limit * 0.1
            
            # Additional constraints for critical conditions
            if metrics.relative_spread > self.config.critical_spread:
                limit *= 0.5
            
            if metrics.volume_ratio < self.config.critical_volume_ratio:
                limit *= 0.3
            
            if metrics.depth_ratio < self.config.depth_collapse_threshold:
                limit *= 0.2
            
            position_limits[i] = max(limit, 0.01)  # Minimum 1% position
        
        return position_limits
    
    def get_emergency_actions(self, 
                            current_weights: np.ndarray,
                            liquidity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get emergency actions when liquidity deteriorates."""
        constraint_check = self.calculator.check_liquidity_constraints(
            current_weights, liquidity_data
        )
        
        emergency_actions = {
            "actions_required": constraint_check["emergency_conditions"],
            "actions": []
        }
        
        if constraint_check["spread_critical"]:
            emergency_actions["actions"].append({
                "type": "REDUCE_SPREAD_SENSITIVE_POSITIONS",
                "severity": "CRITICAL",
                "target_reduction": 0.8,
                "reason": f"Portfolio spread {constraint_check['portfolio_spread']:.4f} exceeds critical level {self.config.critical_spread:.4f}"
            })
        
        if constraint_check["depth_critical"]:
            emergency_actions["actions"].append({
                "type": "FLATTEN_ILLIQUID_POSITIONS",
                "severity": "CRITICAL",
                "target_reduction": 0.9,
                "reason": f"Portfolio depth {constraint_check['portfolio_depth']:.4f} below critical level {self.config.depth_collapse_threshold:.4f}"
            })
        
        if constraint_check["volume_issues"]:
            for issue in constraint_check["volume_issues"]:
                emergency_actions["actions"].append({
                    "type": "REDUCE_LOW_VOLUME_POSITION",
                    "severity": "HIGH",
                    "asset_index": issue["asset_index"],
                    "current_weight": issue["weight"],
                    "target_reduction": 0.7,
                    "reason": f"Asset volume ratio {issue['volume_ratio']:.2f} below critical level {self.config.critical_volume_ratio:.2f}"
                })
        
        if emergency_actions["actions_required"]:
            self.logger.critical(f"L-VaR emergency actions required: {len(emergency_actions['actions'])} actions")
        
        return emergency_actions
    
    def monitor_liquidity_regime(self) -> Dict[str, Any]:
        """Monitor liquidity regime changes."""
        if len(self.liquidity_history) < 5:
            return {"regime": "UNKNOWN", "insufficient_data": True}
        
        # Analyze recent liquidity trends
        recent_data = self.liquidity_history[-5:]
        
        # Extract key metrics
        spreads = []
        volumes = []
        depths = []
        
        for data in recent_data:
            for asset_data in data.get("assets", []):
                spreads.append(asset_data.get("bid_ask_spread", 0))
                volumes.append(asset_data.get("volume", 0))
                depths.append(asset_data.get("bid_depth", 0) + asset_data.get("ask_depth", 0))
        
        if not spreads:
            return {"regime": "UNKNOWN", "no_data": True}
        
        # Regime classification
        avg_spread = np.mean(spreads)
        avg_volume = np.mean(volumes)
        avg_depth = np.mean(depths)
        
        # Compare to thresholds
        if avg_spread > self.config.critical_spread:
            regime = "CRISIS"
        elif avg_spread > self.config.bid_ask_spread_threshold:
            regime = "STRESSED"
        elif avg_volume < np.percentile(volumes, 25):  # Bottom quartile
            regime = "LOW_VOLUME"
        else:
            regime = "NORMAL"
        
        regime_info = {
            "regime": regime,
            "avg_spread": avg_spread,
            "avg_volume": avg_volume,
            "avg_depth": avg_depth,
            "observations": len(spreads),
            "trend": self._detect_trend(spreads)
        }
        
        return regime_info
    
    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in liquidity metrics."""
        if len(values) < 3:
            return "UNKNOWN"
        
        # Simple trend detection
        recent_values = values[-3:]
        
        if all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
            return "IMPROVING"
        elif all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
            return "DETERIORATING"
        else:
            return "STABLE"