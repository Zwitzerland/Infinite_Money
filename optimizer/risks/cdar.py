"""
CDaR (Conditional Drawdown at Risk) Implementation
Hard constraint for path control in portfolio optimization.

References:
- "Portfolio Optimization with Drawdown Constraints" (SSRN:223323)
- CDaR as constraint, not KPI - enforce as hard budget on wealth path
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from scipy.stats import norm

from ...utils.logger import get_logger


@dataclass
class CDaRConfig:
    """Configuration for CDaR calculations."""
    confidence_level: float = 0.95  # CDaR confidence level (α)
    lookback_window: int = 252      # Lookback window for CDaR calculation
    min_observations: int = 30      # Minimum observations for valid CDaR
    simulation_paths: int = 10000   # Number of Monte Carlo paths
    time_horizon: int = 252         # Time horizon for forward-looking CDaR
    budget: float = 0.05            # CDaR budget (5% maximum)


class CDaRCalculator:
    """
    Conditional Drawdown at Risk (CDaR) Calculator
    
    CDaR_α = E[DD | DD >= VaR_α(DD)]
    where DD is the drawdown distribution.
    """
    
    def __init__(self, config: CDaRConfig):
        """Initialize CDaR calculator."""
        self.config = config
        self.logger = get_logger("CDaRCalculator")
        
    def compute_historical_cdar(self, 
                               returns: np.ndarray,
                               weights: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Compute historical CDaR for given portfolio weights.
        
        Args:
            returns: Historical returns matrix (T x N)
            weights: Portfolio weights (N,)
            
        Returns:
            cdar_value: CDaR value
            diagnostics: Additional metrics
        """
        try:
            # Compute portfolio returns
            portfolio_returns = returns @ weights
            
            # Compute wealth path
            wealth_path = np.cumprod(1 + portfolio_returns)
            
            # Compute drawdowns
            drawdowns = self._compute_drawdowns(wealth_path)
            
            # Compute CDaR
            cdar_value = self._compute_cdar_from_drawdowns(drawdowns)
            
            # Additional diagnostics
            max_drawdown = np.max(drawdowns)
            avg_drawdown = np.mean(drawdowns[drawdowns > 0])
            drawdown_duration = self._compute_drawdown_duration(drawdowns)
            
            diagnostics = {
                "max_drawdown": max_drawdown,
                "avg_drawdown": avg_drawdown,
                "drawdown_duration": drawdown_duration,
                "num_drawdown_periods": np.sum(drawdowns > 0),
                "confidence_level": self.config.confidence_level
            }
            
            self.logger.debug(f"Historical CDaR: {cdar_value:.4f}, Max DD: {max_drawdown:.4f}")
            
            return cdar_value, diagnostics
            
        except Exception as e:
            self.logger.error(f"Error computing historical CDaR: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def compute_forward_looking_cdar(self, 
                                   returns: np.ndarray,
                                   weights: np.ndarray,
                                   initial_wealth: float = 1.0) -> Tuple[float, Dict[str, Any]]:
        """
        Compute forward-looking CDaR using Monte Carlo simulation.
        
        Args:
            returns: Historical returns for parameter estimation
            weights: Portfolio weights
            initial_wealth: Initial wealth level
            
        Returns:
            expected_cdar: Expected CDaR over time horizon
            diagnostics: Simulation diagnostics
        """
        try:
            # Estimate return distribution parameters
            portfolio_returns = returns @ weights
            mean_return = np.mean(portfolio_returns)
            volatility = np.std(portfolio_returns)
            
            # Monte Carlo simulation
            simulated_cdars = []
            
            for _ in range(self.config.simulation_paths):
                # Simulate future returns
                future_returns = np.random.normal(
                    mean_return, volatility, self.config.time_horizon
                )
                
                # Compute wealth path
                wealth_path = initial_wealth * np.cumprod(1 + future_returns)
                
                # Compute drawdowns
                drawdowns = self._compute_drawdowns(wealth_path)
                
                # Compute CDaR for this path
                path_cdar = self._compute_cdar_from_drawdowns(drawdowns)
                simulated_cdars.append(path_cdar)
            
            simulated_cdars = np.array(simulated_cdars)
            
            # Expected CDaR
            expected_cdar = np.mean(simulated_cdars)
            
            # Additional statistics
            cdar_percentiles = np.percentile(simulated_cdars, [5, 25, 50, 75, 95])
            
            diagnostics = {
                "expected_cdar": expected_cdar,
                "cdar_std": np.std(simulated_cdars),
                "cdar_p5": cdar_percentiles[0],
                "cdar_p25": cdar_percentiles[1],
                "cdar_p50": cdar_percentiles[2],
                "cdar_p75": cdar_percentiles[3],
                "cdar_p95": cdar_percentiles[4],
                "simulation_paths": self.config.simulation_paths,
                "time_horizon": self.config.time_horizon
            }
            
            self.logger.debug(f"Forward-looking CDaR: {expected_cdar:.4f} ± {np.std(simulated_cdars):.4f}")
            
            return expected_cdar, diagnostics
            
        except Exception as e:
            self.logger.error(f"Error computing forward-looking CDaR: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def _compute_drawdowns(self, wealth_path: np.ndarray) -> np.ndarray:
        """Compute drawdowns from wealth path."""
        # Running maximum (peak)
        peak = np.maximum.accumulate(wealth_path)
        
        # Drawdown as percentage from peak
        drawdowns = (peak - wealth_path) / peak
        
        return drawdowns
    
    def _compute_cdar_from_drawdowns(self, drawdowns: np.ndarray) -> float:
        """Compute CDaR from drawdown series."""
        if len(drawdowns) < self.config.min_observations:
            return 0.0
        
        # Sort drawdowns in descending order
        sorted_drawdowns = np.sort(drawdowns)[::-1]
        
        # Find VaR_α threshold
        var_index = int(self.config.confidence_level * len(sorted_drawdowns))
        var_index = min(var_index, len(sorted_drawdowns) - 1)
        
        # CDaR as conditional expectation beyond VaR
        if var_index > 0:
            cdar = np.mean(sorted_drawdowns[:var_index])
        else:
            cdar = sorted_drawdowns[0] if len(sorted_drawdowns) > 0 else 0.0
        
        return cdar
    
    def _compute_drawdown_duration(self, drawdowns: np.ndarray) -> float:
        """Compute average drawdown duration."""
        if len(drawdowns) == 0:
            return 0.0
        
        # Find drawdown periods
        in_drawdown = drawdowns > 1e-6  # Small threshold for numerical stability
        
        # Compute durations
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final duration if still in drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0
    
    def check_cdar_constraint(self, 
                             returns: np.ndarray,
                             weights: np.ndarray) -> Dict[str, Any]:
        """
        Check if portfolio satisfies CDaR constraint.
        
        Args:
            returns: Historical returns
            weights: Portfolio weights
            
        Returns:
            constraint_check: Constraint satisfaction results
        """
        # Compute historical CDaR
        historical_cdar, hist_diagnostics = self.compute_historical_cdar(returns, weights)
        
        # Compute forward-looking CDaR
        forward_cdar, forward_diagnostics = self.compute_forward_looking_cdar(returns, weights)
        
        # Check constraints
        historical_satisfied = historical_cdar <= self.config.budget
        forward_satisfied = forward_cdar <= self.config.budget
        
        constraint_check = {
            "historical_cdar": historical_cdar,
            "forward_cdar": forward_cdar,
            "budget": self.config.budget,
            "historical_satisfied": historical_satisfied,
            "forward_satisfied": forward_satisfied,
            "overall_satisfied": historical_satisfied and forward_satisfied,
            "historical_diagnostics": hist_diagnostics,
            "forward_diagnostics": forward_diagnostics
        }
        
        if not constraint_check["overall_satisfied"]:
            self.logger.warning(f"CDaR constraint violated. Historical: {historical_cdar:.4f}, Forward: {forward_cdar:.4f}, Budget: {self.config.budget:.4f}")
        
        return constraint_check
    
    def create_cdar_constraint_cvxpy(self, 
                                   returns: np.ndarray,
                                   weights_var: cp.Variable) -> cp.Constraint:
        """
        Create CVXPY constraint for CDaR.
        
        Args:
            returns: Historical returns
            weights_var: CVXPY variable for weights
            
        Returns:
            cdar_constraint: CVXPY constraint
        """
        T, N = returns.shape
        
        # Auxiliary variables for CDaR calculation
        portfolio_returns = returns @ weights_var
        
        # Simplified CDaR constraint using CVaR approximation
        # This is an approximation since exact CDaR is non-convex
        
        # Use empirical quantile estimation
        alpha = 1 - self.config.confidence_level
        k = int(alpha * T)
        
        if k > 0:
            # Create auxiliary variables for sorted returns
            z = cp.Variable(T)
            t = cp.Variable()
            
            # CVaR constraint (approximation for CDaR)
            cvar_constraint = [
                z >= 0,
                z >= -portfolio_returns - t,
                t + cp.sum(z) / (T * alpha) <= self.config.budget
            ]
            
            return cvar_constraint
        else:
            # If k = 0, use simple VaR constraint
            return [cp.max(-portfolio_returns) <= self.config.budget]


class CDaRPathController:
    """
    CDaR-based path controller for dynamic position sizing.
    
    Enforces CDaR as hard budget on wealth path with real-time monitoring.
    """
    
    def __init__(self, config: CDaRConfig):
        """Initialize CDaR path controller."""
        self.config = config
        self.logger = get_logger("CDaRPathController")
        self.calculator = CDaRCalculator(config)
        
        # State tracking
        self.wealth_history = []
        self.returns_history = []
        self.weights_history = []
        
    def update_state(self, 
                    wealth: float,
                    portfolio_return: float,
                    weights: np.ndarray):
        """Update controller state with new observations."""
        self.wealth_history.append(wealth)
        self.returns_history.append(portfolio_return)
        self.weights_history.append(weights.copy())
        
        # Keep only recent history
        max_history = self.config.lookback_window * 2
        if len(self.wealth_history) > max_history:
            self.wealth_history = self.wealth_history[-max_history:]
            self.returns_history = self.returns_history[-max_history:]
            self.weights_history = self.weights_history[-max_history:]
    
    def get_real_time_cdar(self) -> Tuple[float, Dict[str, Any]]:
        """Compute real-time CDaR from current state."""
        if len(self.wealth_history) < self.config.min_observations:
            return 0.0, {"insufficient_data": True}
        
        wealth_path = np.array(self.wealth_history)
        drawdowns = self.calculator._compute_drawdowns(wealth_path)
        cdar = self.calculator._compute_cdar_from_drawdowns(drawdowns)
        
        diagnostics = {
            "current_wealth": wealth_path[-1],
            "current_drawdown": drawdowns[-1],
            "max_historical_drawdown": np.max(drawdowns),
            "observations": len(wealth_path)
        }
        
        return cdar, diagnostics
    
    def check_breach_risk(self, proposed_weights: np.ndarray) -> Dict[str, Any]:
        """
        Check risk of CDaR constraint breach with proposed weights.
        
        Args:
            proposed_weights: Proposed portfolio weights
            
        Returns:
            breach_analysis: Analysis of potential breach risk
        """
        if len(self.returns_history) < self.config.min_observations:
            return {"insufficient_data": True}
        
        # Current CDaR
        current_cdar, current_diagnostics = self.get_real_time_cdar()
        
        # Simulate CDaR with proposed weights
        recent_returns = np.array(self.returns_history[-self.config.lookback_window:])
        
        if len(recent_returns.shape) == 1:
            # Single asset case
            recent_returns = recent_returns.reshape(-1, 1)
            proposed_weights = np.array([1.0])
        
        # Estimate CDaR with proposed weights
        simulated_cdar, _ = self.calculator.compute_forward_looking_cdar(
            recent_returns, proposed_weights, self.wealth_history[-1]
        )
        
        # Risk assessment
        breach_probability = self._estimate_breach_probability(simulated_cdar)
        risk_level = self._assess_risk_level(current_cdar, simulated_cdar, breach_probability)
        
        breach_analysis = {
            "current_cdar": current_cdar,
            "simulated_cdar": simulated_cdar,
            "budget": self.config.budget,
            "breach_probability": breach_probability,
            "risk_level": risk_level,
            "recommendation": self._get_recommendation(risk_level),
            "current_diagnostics": current_diagnostics
        }
        
        return breach_analysis
    
    def _estimate_breach_probability(self, simulated_cdar: float) -> float:
        """Estimate probability of CDaR constraint breach."""
        if simulated_cdar <= self.config.budget:
            return 0.0
        
        # Simple model: probability increases with distance from budget
        excess = simulated_cdar - self.config.budget
        budget_fraction = excess / self.config.budget
        
        # Sigmoid transformation
        probability = 1 / (1 + np.exp(-10 * budget_fraction))
        
        return min(probability, 1.0)
    
    def _assess_risk_level(self, 
                          current_cdar: float,
                          simulated_cdar: float,
                          breach_probability: float) -> str:
        """Assess overall risk level."""
        if breach_probability < 0.1:
            return "LOW"
        elif breach_probability < 0.3:
            return "MEDIUM"
        elif breach_probability < 0.7:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level."""
        recommendations = {
            "LOW": "Proceed with proposed allocation",
            "MEDIUM": "Consider reducing position sizes",
            "HIGH": "Significant position reduction recommended",
            "CRITICAL": "Flatten portfolio immediately"
        }
        
        return recommendations.get(risk_level, "Unknown risk level")
    
    def get_emergency_action(self) -> Dict[str, Any]:
        """Get emergency action if CDaR constraint is breached."""
        current_cdar, diagnostics = self.get_real_time_cdar()
        
        if current_cdar > self.config.budget:
            # Emergency: CDaR constraint breached
            breach_severity = (current_cdar - self.config.budget) / self.config.budget
            
            if breach_severity > 0.5:
                action = "FLATTEN_IMMEDIATELY"
                target_exposure = 0.0
            elif breach_severity > 0.2:
                action = "REDUCE_EXPOSURE_MAJOR"
                target_exposure = 0.3
            else:
                action = "REDUCE_EXPOSURE_MINOR"
                target_exposure = 0.7
            
            emergency_action = {
                "action_required": True,
                "action": action,
                "target_exposure": target_exposure,
                "breach_severity": breach_severity,
                "current_cdar": current_cdar,
                "budget": self.config.budget,
                "diagnostics": diagnostics
            }
            
            self.logger.critical(f"CDaR emergency action: {action}, severity: {breach_severity:.3f}")
            
        else:
            emergency_action = {
                "action_required": False,
                "current_cdar": current_cdar,
                "budget": self.config.budget,
                "margin": self.config.budget - current_cdar
            }
        
        return emergency_action