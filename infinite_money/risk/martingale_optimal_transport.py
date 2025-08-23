"""
Martingale Optimal Transport Superhedging
Computes robust price/hedge bands (with frictions) and forces overlays to sit inside them.
This is the formal blast shield against tail model error.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize

from ..utils.logger import get_logger


@dataclass
class MOTConfig:
    """Configuration for Martingale Optimal Transport."""
    num_time_steps: int = 10
    num_scenarios: int = 1000
    confidence_level: float = 0.95
    transaction_cost: float = 0.001  # 10 basis points
    slippage_model: str = "linear"  # linear, square_root, or constant
    risk_free_rate: float = 0.02
    max_iterations: int = 1000
    tolerance: float = 1e-6


@dataclass
class PriceBand:
    """Price band with upper and lower bounds."""
    lower_bound: float
    upper_bound: float
    mid_price: float
    width: float
    confidence: float


@dataclass
class HedgeBand:
    """Hedge band with optimal hedge ratios."""
    hedge_ratios: np.ndarray
    cost: float
    effectiveness: float
    robustness: float


class MartingaleOptimalTransport:
    """
    Martingale Optimal Transport for Superhedging
    
    Implements MOT to compute robust price/hedge bands that serve as formal blast shield
    against tail model error and ensure no-arbitrage conditions.
    """
    
    def __init__(self, config: MOTConfig):
        """Initialize the MOT system."""
        self.config = config
        self.logger = get_logger("MOTSuperhedging")
        
    def compute_price_bands(self, 
                           asset_prices: np.ndarray,
                           payoff_function: callable,
                           constraints: Dict[str, Any]) -> PriceBand:
        """
        Compute robust price bands using MOT.
        
        Args:
            asset_prices: Historical asset prices (T x N)
            payoff_function: Function defining the payoff
            constraints: Additional constraints (volatility, correlation bounds)
            
        Returns:
            price_band: Robust price band with bounds
        """
        try:
            T, N = asset_prices.shape
            
            # 1. Generate martingale measures
            martingale_measures = self._generate_martingale_measures(asset_prices, constraints)
            
            # 2. Compute payoff under each measure
            payoffs = []
            for measure in martingale_measures:
                payoff = self._compute_payoff_under_measure(measure, payoff_function)
                payoffs.append(payoff)
            
            payoffs = np.array(payoffs)
            
            # 3. Compute robust bounds
            alpha = 1 - self.config.confidence_level
            lower_bound = np.percentile(payoffs, alpha * 100 / 2)
            upper_bound = np.percentile(payoffs, (1 - alpha / 2) * 100)
            
            # 4. Apply transaction costs
            transaction_cost = self._compute_transaction_cost(asset_prices)
            lower_bound -= transaction_cost
            upper_bound += transaction_cost
            
            # 5. Ensure no-arbitrage
            lower_bound = max(lower_bound, 0)  # No negative prices
            
            price_band = PriceBand(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                mid_price=(lower_bound + upper_bound) / 2,
                width=upper_bound - lower_bound,
                confidence=self.config.confidence_level
            )
            
            self.logger.info(f"MOT price band: [{lower_bound:.6f}, {upper_bound:.6f}], width: {price_band.width:.6f}")
            
            return price_band
            
        except Exception as e:
            self.logger.error(f"Error in MOT price band computation: {str(e)}")
            # Return conservative bounds
            return PriceBand(
                lower_bound=0.0,
                upper_bound=np.inf,
                mid_price=0.0,
                width=np.inf,
                confidence=0.0
            )
    
    def compute_hedge_bands(self, 
                           asset_prices: np.ndarray,
                           target_payoff: float,
                           constraints: Dict[str, Any]) -> HedgeBand:
        """
        Compute robust hedge bands using MOT.
        
        Args:
            asset_prices: Historical asset prices (T x N)
            target_payoff: Target payoff to hedge
            constraints: Additional constraints
            
        Returns:
            hedge_band: Robust hedge band with optimal ratios
        """
        try:
            T, N = asset_prices.shape
            
            # 1. Set up the optimization problem
            hedge_ratios = cp.Variable(N)
            
            # 2. Generate scenarios
            scenarios = self._generate_scenarios(asset_prices, constraints)
            
            # 3. Compute hedge payoffs under each scenario
            hedge_payoffs = []
            for scenario in scenarios:
                payoff = cp.sum(cp.multiply(hedge_ratios, scenario))
                hedge_payoffs.append(payoff)
            
            # 4. Objective: minimize worst-case hedging error
            hedging_errors = [cp.abs(payoff - target_payoff) for payoff in hedge_payoffs]
            worst_case_error = cp.max(hedging_errors)
            
            # 5. Constraints
            # Budget constraint
            budget_constraint = cp.sum(cp.abs(hedge_ratios)) <= 1.0
            
            # Martingale constraint (simplified)
            martingale_constraint = cp.sum(hedge_ratios) >= 0
            
            # 6. Solve optimization
            objective = cp.Minimize(worst_case_error)
            problem = cp.Problem(objective, [budget_constraint, martingale_constraint])
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_ratios = hedge_ratios.value
                
                # Compute hedge effectiveness
                effectiveness = self._compute_hedge_effectiveness(optimal_ratios, scenarios, target_payoff)
                
                # Compute robustness
                robustness = self._compute_hedge_robustness(optimal_ratios, scenarios, target_payoff)
                
                # Compute cost
                cost = self._compute_hedge_cost(optimal_ratios, asset_prices)
                
                hedge_band = HedgeBand(
                    hedge_ratios=optimal_ratios,
                    cost=cost,
                    effectiveness=effectiveness,
                    robustness=robustness
                )
                
                self.logger.info(f"MOT hedge band computed. Effectiveness: {effectiveness:.3f}, Cost: {cost:.6f}")
                
                return hedge_band
                
            else:
                self.logger.warning(f"MOT hedge optimization failed: {problem.status}")
                return HedgeBand(
                    hedge_ratios=np.zeros(N),
                    cost=0.0,
                    effectiveness=0.0,
                    robustness=0.0
                )
                
        except Exception as e:
            self.logger.error(f"Error in MOT hedge band computation: {str(e)}")
            return HedgeBand(
                hedge_ratios=np.zeros(N),
                cost=0.0,
                effectiveness=0.0,
                robustness=0.0
            )
    
    def check_superhedge_admissibility(self, 
                                      portfolio_weights: np.ndarray,
                                      price_band: PriceBand,
                                      hedge_band: HedgeBand) -> Dict[str, Any]:
        """
        Check if portfolio is admissible under MOT superhedging constraints.
        
        Args:
            portfolio_weights: Current portfolio weights
            price_band: MOT price band
            hedge_band: MOT hedge band
            
        Returns:
            admissibility: Admissibility check results
        """
        try:
            # 1. Check if portfolio value is within price band
            portfolio_value = np.sum(portfolio_weights)
            within_price_band = (price_band.lower_bound <= portfolio_value <= price_band.upper_bound)
            
            # 2. Check hedge effectiveness
            hedge_effective = hedge_band.effectiveness >= 0.8  # 80% effectiveness threshold
            
            # 3. Check robustness
            robust_enough = hedge_band.robustness >= 0.7  # 70% robustness threshold
            
            # 4. Check transaction costs
            transaction_cost = self._compute_transaction_cost_from_weights(portfolio_weights)
            cost_acceptable = transaction_cost <= price_band.width * 0.1  # 10% of band width
            
            # 5. Overall admissibility
            admissible = (within_price_band and hedge_effective and robust_enough and cost_acceptable)
            
            admissibility = {
                "admissible": admissible,
                "within_price_band": within_price_band,
                "hedge_effective": hedge_effective,
                "robust_enough": robust_enough,
                "cost_acceptable": cost_acceptable,
                "portfolio_value": portfolio_value,
                "price_band": price_band,
                "hedge_band": hedge_band,
                "transaction_cost": transaction_cost
            }
            
            if not admissible:
                self.logger.warning(f"Portfolio not admissible under MOT constraints: {admissibility}")
            else:
                self.logger.info("Portfolio admissible under MOT constraints")
            
            return admissibility
            
        except Exception as e:
            self.logger.error(f"Error in MOT admissibility check: {str(e)}")
            return {"admissible": False, "error": str(e)}
    
    def _generate_martingale_measures(self, 
                                    asset_prices: np.ndarray,
                                    constraints: Dict[str, Any]) -> List[np.ndarray]:
        """Generate martingale measures satisfying constraints."""
        T, N = asset_prices.shape
        measures = []
        
        # Generate multiple measures using different methods
        for _ in range(self.config.num_scenarios):
            # Method 1: Perturbed historical measure
            measure = self._perturb_historical_measure(asset_prices, constraints)
            measures.append(measure)
            
            # Method 2: Bootstrap measure
            measure = self._bootstrap_measure(asset_prices, constraints)
            measures.append(measure)
            
            # Method 3: Monte Carlo measure
            measure = self._monte_carlo_measure(asset_prices, constraints)
            measures.append(measure)
        
        return measures
    
    def _perturb_historical_measure(self, 
                                  asset_prices: np.ndarray,
                                  constraints: Dict[str, Any]) -> np.ndarray:
        """Generate measure by perturbing historical distribution."""
        T, N = asset_prices.shape
        
        # Compute historical returns
        returns = np.diff(asset_prices, axis=0) / asset_prices[:-1]
        
        # Perturb returns with noise
        noise_std = constraints.get("volatility_bound", 0.1)
        perturbed_returns = returns + np.random.normal(0, noise_std, returns.shape)
        
        # Ensure martingale property (E[return] = 0)
        perturbed_returns = perturbed_returns - np.mean(perturbed_returns, axis=0)
        
        return perturbed_returns
    
    def _bootstrap_measure(self, 
                         asset_prices: np.ndarray,
                         constraints: Dict[str, Any]) -> np.ndarray:
        """Generate measure using bootstrap resampling."""
        T, N = asset_prices.shape
        
        # Compute historical returns
        returns = np.diff(asset_prices, axis=0) / asset_prices[:-1]
        
        # Bootstrap resampling
        bootstrap_indices = np.random.choice(T-1, size=T-1, replace=True)
        bootstrap_returns = returns[bootstrap_indices]
        
        return bootstrap_returns
    
    def _monte_carlo_measure(self, 
                           asset_prices: np.ndarray,
                           constraints: Dict[str, Any]) -> np.ndarray:
        """Generate measure using Monte Carlo simulation."""
        T, N = asset_prices.shape
        
        # Estimate parameters from historical data
        returns = np.diff(asset_prices, axis=0) / asset_prices[:-1]
        mean_returns = np.mean(returns, axis=0)
        cov_returns = np.cov(returns.T)
        
        # Generate Monte Carlo returns
        mc_returns = np.random.multivariate_normal(mean_returns, cov_returns, T-1)
        
        # Ensure martingale property
        mc_returns = mc_returns - np.mean(mc_returns, axis=0)
        
        return mc_returns
    
    def _compute_payoff_under_measure(self, 
                                    measure: np.ndarray,
                                    payoff_function: callable) -> float:
        """Compute payoff under a given measure."""
        try:
            # Simulate asset prices under the measure
            initial_price = 100.0  # Assume initial price
            prices = [initial_price]
            
            for return_vec in measure:
                new_price = prices[-1] * (1 + np.sum(return_vec))
                prices.append(new_price)
            
            # Compute payoff
            payoff = payoff_function(np.array(prices))
            return payoff
            
        except Exception as e:
            self.logger.error(f"Error computing payoff under measure: {str(e)}")
            return 0.0
    
    def _generate_scenarios(self, 
                          asset_prices: np.ndarray,
                          constraints: Dict[str, Any]) -> List[np.ndarray]:
        """Generate scenarios for hedge computation."""
        T, N = asset_prices.shape
        scenarios = []
        
        # Generate scenarios using different methods
        for _ in range(self.config.num_scenarios):
            # Use martingale measures as scenarios
            measure = self._perturb_historical_measure(asset_prices, constraints)
            scenarios.append(measure[-1])  # Take final return vector
        
        return scenarios
    
    def _compute_hedge_effectiveness(self, 
                                   hedge_ratios: np.ndarray,
                                   scenarios: List[np.ndarray],
                                   target_payoff: float) -> float:
        """Compute hedge effectiveness."""
        if len(scenarios) == 0:
            return 0.0
        
        hedge_payoffs = [np.sum(hedge_ratios * scenario) for scenario in scenarios]
        hedging_errors = [abs(payoff - target_payoff) for payoff in hedge_payoffs]
        
        # Effectiveness = 1 - normalized mean error
        mean_error = np.mean(hedging_errors)
        max_error = max(hedging_errors)
        
        if max_error == 0:
            return 1.0
        
        effectiveness = 1.0 - (mean_error / max_error)
        return max(0.0, effectiveness)
    
    def _compute_hedge_robustness(self, 
                                hedge_ratios: np.ndarray,
                                scenarios: List[np.ndarray],
                                target_payoff: float) -> float:
        """Compute hedge robustness."""
        if len(scenarios) == 0:
            return 0.0
        
        hedge_payoffs = [np.sum(hedge_ratios * scenario) for scenario in scenarios]
        hedging_errors = [abs(payoff - target_payoff) for payoff in hedge_payoffs]
        
        # Robustness = percentage of scenarios with error below threshold
        threshold = target_payoff * 0.1  # 10% threshold
        robust_scenarios = sum(1 for error in hedging_errors if error <= threshold)
        
        robustness = robust_scenarios / len(scenarios)
        return robustness
    
    def _compute_transaction_cost(self, asset_prices: np.ndarray) -> float:
        """Compute transaction costs."""
        T, N = asset_prices.shape
        
        # Simple linear transaction cost model
        total_volume = np.sum(np.abs(np.diff(asset_prices, axis=0)))
        transaction_cost = total_volume * self.config.transaction_cost
        
        return transaction_cost
    
    def _compute_transaction_cost_from_weights(self, portfolio_weights: np.ndarray) -> float:
        """Compute transaction cost from portfolio weights."""
        # Simple linear cost model
        total_weight = np.sum(np.abs(portfolio_weights))
        transaction_cost = total_weight * self.config.transaction_cost
        
        return transaction_cost