"""
Wasserstein-DRO Kelly Optimizer
Maximizes worst-case log-growth inside a Wasserstein ball around the empirical distribution.

References:
- "Wasserstein-Kelly Portfolios: A Robust Data-Driven Solution to Optimize Portfolio Growth" (arXiv:2302.13979)
- Modern Kelly criterion for uncertainty with distributional robustness
"""

import numpy as np
import cvxpy as cp
import torch
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import ot  # POT library for optimal transport

from ...utils.logger import get_logger


@dataclass
class DROKellyConfig:
    """Configuration for Wasserstein-DRO Kelly optimization."""
    # Wasserstein ball parameters
    wasserstein_radius: float = 0.1  # Radius of Wasserstein ball
    transport_cost: str = "euclidean"  # Transport cost function
    regularization: float = 1e-3  # Entropy regularization for OT
    
    # Kelly parameters
    risk_aversion: float = 1.0  # Risk aversion parameter
    leverage_cap: float = 2.0   # Maximum leverage
    min_weight: float = -1.0    # Minimum weight (short limit)
    max_weight: float = 2.0     # Maximum weight (long limit)
    
    # Optimization parameters
    solver: str = "ECOS"        # CVXPY solver
    max_iterations: int = 1000  # Maximum solver iterations
    tolerance: float = 1e-6     # Convergence tolerance
    
    # Robustness
    confidence_level: float = 0.95  # Confidence level for worst-case
    sample_size: int = 1000     # Sample size for empirical distribution


class WassersteinDROKellyOptimizer:
    """
    Wasserstein Distributionally Robust Kelly Optimizer
    
    Solves: max_w inf_{Q ∈ B_W(P,δ)} E_Q[log(1 + w^T r)]
    where B_W(P,δ) is the Wasserstein ball of radius δ around empirical distribution P.
    """
    
    def __init__(self, config: DROKellyConfig):
        """Initialize the Wasserstein-DRO Kelly optimizer."""
        self.config = config
        self.logger = get_logger("WassersteinDROKelly")
        
        # Cache for optimal transport computations
        self._ot_cache = {}
        
    def optimize(self, 
                returns: np.ndarray,
                market_conditions: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize portfolio weights using Wasserstein-DRO Kelly criterion.
        
        Args:
            returns: Historical returns matrix (T x N)
            market_conditions: Market condition indicators
            
        Returns:
            optimal_weights: Optimal portfolio weights
            diagnostics: Optimization diagnostics
        """
        try:
            T, N = returns.shape
            
            # Adaptive Wasserstein radius based on market conditions
            radius = self._adaptive_radius(returns, market_conditions)
            
            # Solve DRO Kelly problem
            if self.config.solver == "cvxpy":
                weights, diagnostics = self._solve_cvxpy(returns, radius)
            else:
                weights, diagnostics = self._solve_dual_formulation(returns, radius)
            
            # Apply leverage cap
            weights = self._apply_leverage_cap(weights)
            
            # Compute additional diagnostics
            diagnostics.update({
                "wasserstein_radius": radius,
                "leverage": np.sum(np.abs(weights)),
                "worst_case_growth": self._compute_worst_case_growth(weights, returns, radius),
                "empirical_growth": np.mean(np.log(1 + returns @ weights))
            })
            
            self.logger.info(f"DRO Kelly optimization completed. Leverage: {diagnostics['leverage']:.3f}")
            
            return weights, diagnostics
            
        except Exception as e:
            self.logger.error(f"Error in DRO Kelly optimization: {str(e)}")
            # Return equal weights as fallback
            equal_weights = np.ones(N) / N
            return equal_weights, {"status": "error", "message": str(e)}
    
    def _adaptive_radius(self, returns: np.ndarray, market_conditions: Dict[str, Any]) -> float:
        """Compute adaptive Wasserstein radius based on market conditions."""
        base_radius = self.config.wasserstein_radius
        
        # Adjust based on forecast dispersion
        forecast_dispersion = market_conditions.get("forecast_dispersion", 0.1)
        dispersion_factor = 1.0 + forecast_dispersion
        
        # Adjust based on regime change probability
        regime_change_prob = market_conditions.get("regime_change_prob", 0.05)
        regime_factor = 1.0 + regime_change_prob * 2
        
        # Adjust based on market volatility
        current_volatility = np.std(returns[-20:])  # Last 20 periods
        historical_volatility = np.std(returns)
        volatility_factor = current_volatility / (historical_volatility + 1e-8)
        
        adaptive_radius = base_radius * dispersion_factor * regime_factor * volatility_factor
        
        # Cap the radius to reasonable bounds
        adaptive_radius = np.clip(adaptive_radius, 0.01, 1.0)
        
        self.logger.debug(f"Adaptive Wasserstein radius: {adaptive_radius:.4f}")
        
        return adaptive_radius
    
    def _solve_cvxpy(self, returns: np.ndarray, radius: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve using CVXPY with approximation."""
        T, N = returns.shape
        
        # Decision variables
        w = cp.Variable(N)  # Portfolio weights
        
        # Approximation: Use scenarios from Wasserstein ball
        scenarios = self._generate_wasserstein_scenarios(returns, radius, n_scenarios=100)
        
        # Worst-case objective (use minimum over scenarios)
        scenario_objectives = []
        for scenario in scenarios:
            portfolio_returns = scenario @ w
            # Approximation: E[log(1 + r)] ≈ r - r²/2 for small r
            log_returns = portfolio_returns - cp.square(portfolio_returns) / 2
            scenario_objectives.append(cp.sum(log_returns) / len(scenario))
        
        worst_case_objective = cp.minimum(*scenario_objectives)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= self.config.min_weight,
            w <= self.config.max_weight,
            cp.norm(w, 1) <= self.config.leverage_cap  # Leverage constraint
        ]
        
        # Solve
        problem = cp.Problem(cp.Maximize(worst_case_objective), constraints)
        problem.solve(solver=getattr(cp, self.config.solver, cp.ECOS), verbose=False)
        
        if problem.status == cp.OPTIMAL:
            return w.value, {"status": "optimal", "objective": problem.value}
        else:
            raise ValueError(f"Optimization failed with status: {problem.status}")
    
    def _solve_dual_formulation(self, returns: np.ndarray, radius: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve using dual formulation of Wasserstein DRO.
        
        The dual problem is:
        min_λ λ * radius + (1/T) * Σ_i φ(λ, returns_i, w)
        """
        T, N = returns.shape
        
        # Use scipy optimization for the dual problem
        def dual_objective(params):
            """Dual objective function."""
            lambda_dual = params[0]
            w = params[1:N+1]
            
            # Normalize weights
            w = w / np.sum(w) if np.sum(w) != 0 else w
            
            # Compute dual function value
            dual_value = lambda_dual * radius
            
            # Add expectation term
            for t in range(T):
                r_t = returns[t]
                portfolio_return = np.dot(w, r_t)
                log_return = np.log(1 + portfolio_return) if portfolio_return > -1 else -np.inf
                
                # Dual function φ(λ, r_t, w)
                phi_value = log_return + lambda_dual * np.linalg.norm(r_t - np.mean(returns, axis=0))
                dual_value += phi_value / T
            
            return -dual_value  # Minimize negative (maximize)
        
        # Initial guess
        initial_lambda = 1.0
        initial_w = np.ones(N) / N
        initial_params = np.concatenate([[initial_lambda], initial_w])
        
        # Constraints
        def weight_sum_constraint(params):
            return np.sum(params[1:]) - 1.0
        
        def leverage_constraint(params):
            return self.config.leverage_cap - np.sum(np.abs(params[1:]))
        
        constraints = [
            {"type": "eq", "fun": weight_sum_constraint},
            {"type": "ineq", "fun": leverage_constraint}
        ]
        
        # Bounds
        bounds = [(0, None)]  # lambda >= 0
        for _ in range(N):
            bounds.append((self.config.min_weight, self.config.max_weight))
        
        # Optimize
        result = minimize(
            dual_objective,
            initial_params,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.config.max_iterations, "ftol": self.config.tolerance}
        )
        
        if result.success:
            optimal_lambda = result.x[0]
            optimal_weights = result.x[1:N+1]
            optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
            
            return optimal_weights, {
                "status": "optimal",
                "objective": -result.fun,
                "dual_lambda": optimal_lambda,
                "iterations": result.nit
            }
        else:
            raise ValueError(f"Dual optimization failed: {result.message}")
    
    def _generate_wasserstein_scenarios(self, 
                                      returns: np.ndarray, 
                                      radius: float, 
                                      n_scenarios: int = 100) -> List[np.ndarray]:
        """Generate scenarios from Wasserstein ball around empirical distribution."""
        T, N = returns.shape
        
        scenarios = []
        
        # Base empirical distribution
        empirical_samples = returns
        
        for _ in range(n_scenarios):
            # Perturb empirical samples within Wasserstein distance
            perturbed_samples = []
            
            for t in range(T):
                # Add noise bounded by transport cost
                noise = np.random.normal(0, radius / np.sqrt(N), N)
                
                # Ensure transport cost constraint
                if np.linalg.norm(noise) > radius:
                    noise = noise * radius / np.linalg.norm(noise)
                
                perturbed_sample = empirical_samples[t] + noise
                perturbed_samples.append(perturbed_sample)
            
            scenarios.append(np.array(perturbed_samples))
        
        return scenarios
    
    def _apply_leverage_cap(self, weights: np.ndarray) -> np.ndarray:
        """Apply leverage cap to weights."""
        current_leverage = np.sum(np.abs(weights))
        
        if current_leverage > self.config.leverage_cap:
            # Scale down weights to meet leverage cap
            scale_factor = self.config.leverage_cap / current_leverage
            weights = weights * scale_factor
        
        return weights
    
    def _compute_worst_case_growth(self, 
                                 weights: np.ndarray, 
                                 returns: np.ndarray, 
                                 radius: float) -> float:
        """Compute worst-case growth rate for given weights."""
        # Generate adversarial scenarios
        scenarios = self._generate_wasserstein_scenarios(returns, radius, n_scenarios=50)
        
        growth_rates = []
        for scenario in scenarios:
            portfolio_returns = scenario @ weights
            # Compute geometric mean return
            geometric_mean = np.exp(np.mean(np.log(1 + portfolio_returns))) - 1
            growth_rates.append(geometric_mean)
        
        # Return worst-case (minimum) growth rate
        return np.min(growth_rates)
    
    def compute_leverage_throttle(self, 
                                optimal_weights: np.ndarray,
                                forecast_dispersion: float,
                                regime_change_prob: float) -> float:
        """
        Compute leverage throttle: λ(t) = min{λ_cap, c(t)||w*||}
        where c(t) ↓ as forecast dispersion ↑ or regime change-points fire
        """
        base_leverage = np.sum(np.abs(optimal_weights))
        
        # Throttle factor based on uncertainty
        dispersion_factor = 1.0 / (1.0 + forecast_dispersion)
        regime_factor = 1.0 / (1.0 + regime_change_prob)
        
        throttle_factor = min(dispersion_factor, regime_factor)
        
        final_leverage = min(
            self.config.leverage_cap,
            throttle_factor * base_leverage
        )
        
        self.logger.info(f"Leverage throttle: {final_leverage:.3f} (base: {base_leverage:.3f})")
        
        return final_leverage
    
    def check_robustness(self, 
                        weights: np.ndarray, 
                        returns: np.ndarray) -> Dict[str, float]:
        """Check robustness of portfolio weights."""
        T, N = returns.shape
        
        # Compute various robustness metrics
        empirical_growth = np.mean(np.log(1 + returns @ weights))
        empirical_volatility = np.std(returns @ weights)
        empirical_sharpe = empirical_growth / (empirical_volatility + 1e-8)
        
        # Worst-case metrics with different radii
        robustness_metrics = {
            "empirical_growth": empirical_growth,
            "empirical_volatility": empirical_volatility,
            "empirical_sharpe": empirical_sharpe
        }
        
        for radius in [0.05, 0.1, 0.2]:
            worst_case_growth = self._compute_worst_case_growth(weights, returns, radius)
            robustness_metrics[f"worst_case_growth_r{radius}"] = worst_case_growth
        
        return robustness_metrics


class DROKellyEnsemble:
    """Ensemble of DRO Kelly optimizers with different configurations."""
    
    def __init__(self, configs: List[DROKellyConfig]):
        """Initialize ensemble of optimizers."""
        self.optimizers = [WassersteinDROKellyOptimizer(config) for config in configs]
        self.logger = get_logger("DROKellyEnsemble")
        
    def optimize_ensemble(self, 
                         returns: np.ndarray,
                         market_conditions: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize using ensemble of DRO Kelly optimizers."""
        T, N = returns.shape
        
        # Collect results from all optimizers
        ensemble_weights = []
        ensemble_diagnostics = []
        
        for i, optimizer in enumerate(self.optimizers):
            try:
                weights, diagnostics = optimizer.optimize(returns, market_conditions)
                ensemble_weights.append(weights)
                ensemble_diagnostics.append(diagnostics)
            except Exception as e:
                self.logger.warning(f"Optimizer {i} failed: {str(e)}")
                continue
        
        if not ensemble_weights:
            # All optimizers failed, return equal weights
            equal_weights = np.ones(N) / N
            return equal_weights, {"status": "all_failed"}
        
        # Ensemble combination strategies
        
        # 1. Simple average
        avg_weights = np.mean(ensemble_weights, axis=0)
        
        # 2. Weighted by worst-case growth
        growth_weights = []
        for i, weights in enumerate(ensemble_weights):
            if "worst_case_growth" in ensemble_diagnostics[i]:
                growth_weights.append(ensemble_diagnostics[i]["worst_case_growth"])
            else:
                growth_weights.append(0.0)
        
        growth_weights = np.array(growth_weights)
        growth_weights = np.exp(growth_weights)  # Convert to positive weights
        growth_weights = growth_weights / np.sum(growth_weights)
        
        weighted_avg_weights = np.average(ensemble_weights, axis=0, weights=growth_weights)
        
        # 3. Median (robust aggregation)
        median_weights = np.median(ensemble_weights, axis=0)
        
        # Select best combination based on robustness
        candidates = {
            "average": avg_weights,
            "growth_weighted": weighted_avg_weights,
            "median": median_weights
        }
        
        best_weights = avg_weights  # Default
        best_score = -np.inf
        
        for name, weights in candidates.items():
            # Evaluate robustness
            score = self._evaluate_robustness(weights, returns)
            if score > best_score:
                best_score = score
                best_weights = weights
        
        ensemble_diagnostics_summary = {
            "ensemble_size": len(ensemble_weights),
            "best_combination": name,
            "robustness_score": best_score,
            "individual_diagnostics": ensemble_diagnostics
        }
        
        return best_weights, ensemble_diagnostics_summary
    
    def _evaluate_robustness(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Evaluate robustness of weights (higher is better)."""
        # Simple robustness score based on worst-case Sharpe ratio
        portfolio_returns = returns @ weights
        
        # Compute rolling statistics
        window = min(50, len(portfolio_returns) // 4)
        rolling_means = []
        rolling_stds = []
        
        for i in range(window, len(portfolio_returns)):
            window_returns = portfolio_returns[i-window:i]
            rolling_means.append(np.mean(window_returns))
            rolling_stds.append(np.std(window_returns))
        
        if not rolling_means:
            return 0.0
        
        # Worst-case Sharpe ratio
        worst_case_mean = np.min(rolling_means)
        worst_case_std = np.max(rolling_stds)
        
        robustness_score = worst_case_mean / (worst_case_std + 1e-8)
        
        return robustness_score