"""
Portfolio Manager Agent with Apex Stack v5 Integration
Responsible for portfolio optimization, position management, and rebalancing.
Now integrates Apex Stack v5 components: Wasserstein-Kelly, MOT superhedging, and quantum risk acceleration.
"""

import numpy as np
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent, Task
from ..utils.logger import get_logger
from ..utils.config import Config
from ..optimization.wasserstein_kelly import WassersteinKellyOptimizer, KellyConstraints
from ..quantum.risk_acceleration import QuantumRiskAccelerator, QAEConfig
from ..risk.martingale_optimal_transport import MartingaleOptimalTransport, MOTConfig, PriceBand, HedgeBand


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result with Apex Stack v5 enhancements."""
    weights: np.ndarray
    expected_return: float
    risk: float
    sharpe_ratio: float
    timestamp: datetime
    kelly_leverage: float
    mot_admissible: bool
    quantum_risk_metrics: Dict[str, float]


class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Manager Agent with Apex Stack v5 Integration
    
    Responsibilities:
    - Distributionally-robust Kelly optimization
    - MOT superhedging constraints
    - Quantum risk acceleration
    - Position management and rebalancing
    - Risk allocation and diversification
    - Performance monitoring and adjustment
    """
    
    def __init__(self, config: Config, agent_config: Dict[str, Any] = None):
        """Initialize Portfolio Manager Agent with Apex Stack v5 components."""
        super().__init__("PortfolioManager", config, agent_config)
        
        # Portfolio state
        self.current_allocation = None
        self.target_allocation = None
        self.rebalance_threshold = agent_config.get("rebalance_threshold", 0.05)
        
        # Apex Stack v5 components
        self._initialize_apex_stack_components(agent_config)
        
        # Optimization parameters
        self.risk_free_rate = agent_config.get("risk_free_rate", 0.02)
        self.max_leverage = agent_config.get("max_leverage", 2.0)
        self.min_weight = agent_config.get("min_weight", 0.01)
        self.max_weight = agent_config.get("max_weight", 0.3)
        
        self.logger.info("Portfolio Manager Agent with Apex Stack v5 initialized")
        
    def _initialize_apex_stack_components(self, agent_config: Dict[str, Any]):
        """Initialize Apex Stack v5 components."""
        # Wasserstein-Kelly optimizer
        kelly_config = KellyConstraints(
            cdar_budget=agent_config.get("cdar_budget", 0.05),
            l_var_budget=agent_config.get("l_var_budget", 0.03),
            max_leverage=agent_config.get("max_leverage", 2.0),
            ambiguity_radius=agent_config.get("ambiguity_radius", 0.1)
        )
        self.kelly_optimizer = WassersteinKellyOptimizer(kelly_config)
        
        # Quantum risk accelerator
        qae_config = QAEConfig(
            max_iterations=agent_config.get("qae_max_iterations", 100),
            epsilon=agent_config.get("qae_epsilon", 0.01),
            alpha=agent_config.get("qae_alpha", 0.05),
            error_mitigation=agent_config.get("qae_error_mitigation", True),
            auto_fallback=agent_config.get("qae_auto_fallback", True)
        )
        self.quantum_risk = QuantumRiskAccelerator(qae_config)
        
        # MOT superhedging
        mot_config = MOTConfig(
            num_time_steps=agent_config.get("mot_time_steps", 10),
            num_scenarios=agent_config.get("mot_scenarios", 1000),
            confidence_level=agent_config.get("mot_confidence", 0.95),
            transaction_cost=agent_config.get("mot_transaction_cost", 0.001)
        )
        self.mot_superhedging = MartingaleOptimalTransport(mot_config)
        
        self.logger.info("Apex Stack v5 components initialized")
        
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute portfolio management task."""
        try:
            if task.task_type == "optimize_portfolio":
                return await self._optimize_portfolio_apex(task.data)
            elif task.task_type == "rebalance_portfolio":
                return await self._rebalance_portfolio(task.data)
            elif task.task_type == "manage_risk":
                return await self._manage_risk_apex(task.data)
            elif task.task_type == "check_mot_admissibility":
                return await self._check_mot_admissibility(task.data)
            elif task.task_type == "manage_portfolio":
                return await self._manage_portfolio(task.data)
            else:
                self.logger.warning(f"Unknown task type: {task.task_type}")
                return {"status": "error", "message": f"Unknown task type: {task.task_type}"}
                
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_portfolio_apex(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio using Apex Stack v5 components."""
        try:
            returns = data.get("returns")
            current_wealth = data.get("current_wealth", 1000000)
            market_conditions = data.get("market_conditions", {})
            
            if returns is None:
                return {"status": "error", "message": "No returns data provided"}
            
            # Convert to numpy array if needed
            if isinstance(returns, list):
                returns = np.array(returns)
            
            # 1. Wasserstein-Kelly optimization
            optimal_weights, kelly_diagnostics = self.kelly_optimizer.optimize_weights(
                returns, current_wealth, market_conditions
            )
            
            # 2. Compute leverage throttle
            forecast_dispersion = market_conditions.get("forecast_dispersion", 0.1)
            regime_change_prob = market_conditions.get("regime_change_prob", 0.05)
            
            leverage_throttle = self.kelly_optimizer.compute_leverage_throttle(
                optimal_weights, forecast_dispersion, regime_change_prob
            )
            
            # 3. Quantum risk estimation
            quantum_var, var_diagnostics = self.quantum_risk.estimate_var(
                returns, optimal_weights, confidence_level=0.95
            )
            
            quantum_cvar, cvar_diagnostics = self.quantum_risk.estimate_cvar(
                returns, optimal_weights, confidence_level=0.95
            )
            
            quantum_pfe, pfe_diagnostics = self.quantum_risk.estimate_pfe(
                returns, optimal_weights, time_horizon=10
            )
            
            # 4. MOT superhedging constraints
            price_band = self.mot_superhedging.compute_price_bands(
                returns, self._simple_payoff_function, market_conditions
            )
            
            hedge_band = self.mot_superhedging.compute_hedge_bands(
                returns, current_wealth * 0.1, market_conditions  # 10% target payoff
            )
            
            # 5. Check MOT admissibility
            mot_admissibility = self.mot_superhedging.check_superhedge_admissibility(
                optimal_weights, price_band, hedge_band
            )
            
            # 6. Apply MOT constraints if needed
            if not mot_admissibility["admissible"]:
                self.logger.warning("Portfolio not MOT admissible, applying constraints")
                optimal_weights = self._apply_mot_constraints(optimal_weights, price_band, hedge_band)
            
            # 7. Create enhanced allocation result
            allocation = PortfolioAllocation(
                weights=optimal_weights,
                expected_return=self._calculate_expected_return(returns, optimal_weights),
                risk=self._calculate_risk(returns, optimal_weights),
                sharpe_ratio=self._calculate_sharpe_ratio(returns, optimal_weights),
                timestamp=datetime.now(),
                kelly_leverage=leverage_throttle,
                mot_admissible=mot_admissibility["admissible"],
                quantum_risk_metrics={
                    "var": quantum_var,
                    "cvar": quantum_cvar,
                    "pfe": quantum_pfe,
                    "var_method": var_diagnostics.get("method", "unknown"),
                    "cvar_method": cvar_diagnostics.get("method", "unknown")
                }
            )
            
            self.target_allocation = allocation
            
            self.logger.info(f"Apex Stack portfolio optimized. Kelly leverage: {leverage_throttle:.3f}, MOT admissible: {mot_admissibility['admissible']}")
            
            return {
                "status": "success",
                "allocation": allocation,
                "kelly_diagnostics": kelly_diagnostics,
                "mot_admissibility": mot_admissibility,
                "quantum_diagnostics": {
                    "var": var_diagnostics,
                    "cvar": cvar_diagnostics,
                    "pfe": pfe_diagnostics
                },
                "message": "Apex Stack portfolio optimization completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error in Apex Stack portfolio optimization: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _manage_risk_apex(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage portfolio risk using Apex Stack v5 components."""
        try:
            current_weights = data.get("current_weights")
            returns = data.get("returns")
            market_conditions = data.get("market_conditions", {})
            
            if current_weights is None or returns is None:
                return {"status": "error", "message": "Missing weight or returns data"}
            
            # Convert to numpy arrays if needed
            if isinstance(current_weights, list):
                current_weights = np.array(current_weights)
            if isinstance(returns, list):
                returns = np.array(returns)
            
            # 1. Quantum risk metrics
            quantum_var, _ = self.quantum_risk.estimate_var(returns, current_weights)
            quantum_cvar, _ = self.quantum_risk.estimate_cvar(returns, current_weights)
            quantum_pfe, _ = self.quantum_risk.estimate_pfe(returns, current_weights)
            
            # 2. MOT admissibility check
            price_band = self.mot_superhedging.compute_price_bands(
                returns, self._simple_payoff_function, market_conditions
            )
            
            hedge_band = self.mot_superhedging.compute_hedge_bands(
                returns, np.sum(current_weights) * 0.1, market_conditions
            )
            
            mot_admissibility = self.mot_superhedging.check_superhedge_admissibility(
                current_weights, price_band, hedge_band
            )
            
            # 3. Risk limits and adjustments
            risk_limits = self._check_apex_risk_limits({
                "var": quantum_var,
                "cvar": quantum_cvar,
                "pfe": quantum_pfe,
                "mot_admissible": mot_admissibility["admissible"]
            })
            
            adjustments = self._generate_apex_risk_adjustments(risk_limits, mot_admissibility)
            
            self.logger.info(f"Apex Stack risk management completed. Quantum VaR: {quantum_var:.4f}, MOT admissible: {mot_admissibility['admissible']}")
            
            return {
                "status": "success",
                "quantum_risk_metrics": {
                    "var": quantum_var,
                    "cvar": quantum_cvar,
                    "pfe": quantum_pfe
                },
                "mot_admissibility": mot_admissibility,
                "risk_limits": risk_limits,
                "adjustments": adjustments,
                "message": "Apex Stack risk management completed"
            }
            
        except Exception as e:
            self.logger.error(f"Error in Apex Stack risk management: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _check_mot_admissibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check MOT superhedging admissibility."""
        try:
            portfolio_weights = data.get("portfolio_weights")
            returns = data.get("returns")
            market_conditions = data.get("market_conditions", {})
            
            if portfolio_weights is None or returns is None:
                return {"status": "error", "message": "Missing portfolio or returns data"}
            
            # Convert to numpy arrays if needed
            if isinstance(portfolio_weights, list):
                portfolio_weights = np.array(portfolio_weights)
            if isinstance(returns, list):
                returns = np.array(returns)
            
            # Compute MOT bands
            price_band = self.mot_superhedging.compute_price_bands(
                returns, self._simple_payoff_function, market_conditions
            )
            
            hedge_band = self.mot_superhedging.compute_hedge_bands(
                returns, np.sum(portfolio_weights) * 0.1, market_conditions
            )
            
            # Check admissibility
            admissibility = self.mot_superhedging.check_superhedge_admissibility(
                portfolio_weights, price_band, hedge_band
            )
            
            return {
                "status": "success",
                "admissible": admissibility["admissible"],
                "price_band": price_band,
                "hedge_band": hedge_band,
                "details": admissibility,
                "message": f"Portfolio MOT admissible: {admissibility['admissible']}"
            }
            
        except Exception as e:
            self.logger.error(f"Error in MOT admissibility check: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_portfolio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy portfolio optimization (fallback)."""
        signals = data.get("signals", {})
        current_positions = data.get("current_positions", {})
        market_data = data.get("market_data", {})
        
        self.logger.info("Using legacy portfolio optimization")
        
        try:
            # Simple portfolio optimization (placeholder)
            optimized_weights = {}
            
            for symbol, signal in signals.items():
                if signal.get("signal", 0) > 0:
                    optimized_weights[symbol] = min(signal.get("confidence", 0.5), self.max_weight)
                elif signal.get("signal", 0) < 0:
                    optimized_weights[symbol] = -min(signal.get("confidence", 0.5), self.max_weight)
                else:
                    optimized_weights[symbol] = 0.0
            
            return {
                "status": "success",
                "optimized_weights": optimized_weights,
                "optimization_method": "legacy"
            }
            
        except Exception as e:
            self.logger.error(f"Error in legacy portfolio optimization: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _manage_portfolio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage portfolio positions."""
        rebalance = data.get("rebalance", False)
        
        self.logger.info("Managing portfolio")
        
        try:
            # Portfolio management logic (placeholder)
            management_result = {
                "positions_managed": 0,
                "rebalancing_needed": rebalance,
                "risk_level": "medium"
            }
            
            return {
                "status": "success",
                "management_result": management_result
            }
            
        except Exception as e:
            self.logger.error(f"Error managing portfolio: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _rebalance_portfolio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rebalance portfolio to target allocation."""
        try:
            current_weights = data.get("current_weights")
            target_weights = data.get("target_weights")
            
            if current_weights is None or target_weights is None:
                return {"status": "error", "message": "Missing weight data"}
            
            # Convert to numpy arrays if needed
            if isinstance(current_weights, list):
                current_weights = np.array(current_weights)
            if isinstance(target_weights, list):
                target_weights = np.array(target_weights)
            
            # Calculate rebalancing trades
            trades = target_weights - current_weights
            
            # Check if rebalancing is needed
            total_deviation = np.sum(np.abs(trades))
            
            if total_deviation < self.rebalance_threshold:
                return {
                    "status": "success",
                    "trades": np.zeros_like(trades),
                    "message": "No rebalancing needed"
                }
            
            # Apply constraints
            trades = self._apply_trading_constraints(trades)
            
            self.logger.info(f"Rebalancing trades calculated. Total deviation: {total_deviation:.4f}")
            
            return {
                "status": "success",
                "trades": trades,
                "message": "Rebalancing trades calculated"
            }
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _simple_payoff_function(self, prices: np.ndarray) -> float:
        """Simple payoff function for MOT."""
        if len(prices) == 0:
            return 0.0
        return prices[-1] - prices[0]  # Simple price difference
    
    def _apply_mot_constraints(self, weights: np.ndarray, price_band: PriceBand, hedge_band: HedgeBand) -> np.ndarray:
        """Apply MOT constraints to portfolio weights."""
        # Simple constraint application
        # In practice, this would be a more sophisticated optimization
        
        # Scale weights to fit within price band
        portfolio_value = np.sum(weights)
        if portfolio_value > price_band.upper_bound:
            scale_factor = price_band.upper_bound / portfolio_value
            weights = weights * scale_factor
        elif portfolio_value < price_band.lower_bound:
            scale_factor = price_band.lower_bound / portfolio_value
            weights = weights * scale_factor
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def _check_apex_risk_limits(self, risk_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check Apex Stack risk limits."""
        limits = {
            "var_limit": 0.05,
            "cvar_limit": 0.07,
            "pfe_limit": 0.10,
            "mot_admissible": True
        }
        
        return {
            "var_ok": risk_metrics["var"] <= limits["var_limit"],
            "cvar_ok": risk_metrics["cvar"] <= limits["cvar_limit"],
            "pfe_ok": risk_metrics["pfe"] <= limits["pfe_limit"],
            "mot_ok": risk_metrics["mot_admissible"]
        }
    
    def _generate_apex_risk_adjustments(self, risk_limits: Dict[str, bool], mot_admissibility: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Apex Stack risk adjustments."""
        adjustments = {}
        
        # Reduce exposure if quantum VaR is too high
        if not risk_limits["var_ok"]:
            adjustments["reduce_exposure"] = 0.15
        
        # Increase hedging if quantum CVaR is too high
        if not risk_limits["cvar_ok"]:
            adjustments["increase_hedging"] = 0.25
        
        # Flatten if MOT not admissible
        if not risk_limits["mot_ok"]:
            adjustments["flatten_portfolio"] = True
            adjustments["migrate_to_carry"] = True
        
        return adjustments
    
    def _calculate_expected_return(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate expected portfolio return."""
        mean_returns = np.mean(returns, axis=0)
        return np.sum(weights * mean_returns)
    
    def _calculate_risk(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate portfolio risk (volatility)."""
        cov_matrix = np.cov(returns.T)
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        expected_return = self._calculate_expected_return(returns, weights)
        risk = self._calculate_risk(returns, weights)
        
        if risk == 0:
            return 0.0
        
        return (expected_return - self.risk_free_rate) / risk
    
    def _apply_trading_constraints(self, trades: np.ndarray) -> np.ndarray:
        """Apply trading constraints."""
        # Minimum trade size
        min_trade = 0.001
        trades[np.abs(trades) < min_trade] = 0
        
        # Leverage constraint
        total_exposure = np.sum(np.abs(trades))
        if total_exposure > self.max_leverage:
            trades = trades * (self.max_leverage / total_exposure)
        
        return trades
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent-specific metrics."""
        return {
            "optimization_method": "apex_stack_v5",
            "kelly_leverage": getattr(self.target_allocation, 'kelly_leverage', 0.0),
            "mot_admissible": getattr(self.target_allocation, 'mot_admissible', False),
            "max_leverage": self.max_leverage,
            "rebalance_threshold": self.rebalance_threshold
        }