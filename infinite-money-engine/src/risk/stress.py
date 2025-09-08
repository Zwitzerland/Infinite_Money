"""
Stress Testing Module

Implements various stress testing scenarios:
- Historical scenarios
- Hypothetical scenarios
- Monte Carlo stress tests
- Sensitivity analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging


class StressTester:
    """
    Portfolio stress testing system.
    
    Implements various stress testing methods:
    - Historical scenario analysis
    - Hypothetical scenarios
    - Monte Carlo stress tests
    - Sensitivity analysis
    """
    
    def __init__(self, confidence_level: float = 0.95, num_simulations: int = 10000):
        """
        Initialize stress tester.
        
        Args:
            confidence_level: Confidence level for stress tests
            num_simulations: Number of Monte Carlo simulations
        """
        self.confidence_level = confidence_level
        self.num_simulations = num_simulations
        self.historical_scenarios = {}
        self.hypothetical_scenarios = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize historical scenarios
        self._initialize_historical_scenarios()
        
    def _initialize_historical_scenarios(self):
        """Initialize historical stress scenarios."""
        try:
            # 2008 Financial Crisis
            self.historical_scenarios['2008_crisis'] = {
                'description': '2008 Financial Crisis',
                'market_shock': -0.40,  # 40% market decline
                'volatility_shock': 2.0,  # 2x volatility increase
                'correlation_shock': 0.8,  # High correlation
                'liquidity_shock': 0.5,  # 50% liquidity reduction
                'duration': 252  # 1 year
            }
            
            # 2020 COVID-19 Crisis
            self.historical_scenarios['2020_covid'] = {
                'description': '2020 COVID-19 Crisis',
                'market_shock': -0.30,  # 30% market decline
                'volatility_shock': 3.0,  # 3x volatility increase
                'correlation_shock': 0.9,  # Very high correlation
                'liquidity_shock': 0.7,  # 70% liquidity reduction
                'duration': 60  # 2 months
            }
            
            # 2000 Dot-com Bubble
            self.historical_scenarios['2000_dotcom'] = {
                'description': '2000 Dot-com Bubble Burst',
                'market_shock': -0.50,  # 50% market decline
                'volatility_shock': 1.5,  # 1.5x volatility increase
                'correlation_shock': 0.6,  # Moderate correlation
                'liquidity_shock': 0.3,  # 30% liquidity reduction
                'duration': 504  # 2 years
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing historical scenarios: {e}")
    
    def run_stress_test(self, weights: Dict[str, float], 
                       scenario_name: str = "custom") -> Dict[str, float]:
        """
        Run stress test on portfolio.
        
        Args:
            weights: Portfolio weights
            scenario_name: Name of stress scenario
            
        Returns:
            Stress test results
        """
        try:
            if scenario_name in self.historical_scenarios:
                return self._run_historical_scenario(weights, scenario_name)
            elif scenario_name in self.hypothetical_scenarios:
                return self._run_hypothetical_scenario(weights, scenario_name)
            else:
                return self._run_monte_carlo_stress(weights)
                
        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            return {}
    
    def _run_historical_scenario(self, weights: Dict[str, float], 
                               scenario_name: str) -> Dict[str, float]:
        """Run historical scenario stress test."""
        try:
            scenario = self.historical_scenarios[scenario_name]
            results = {}
            
            # Calculate portfolio impact
            market_shock = scenario['market_shock']
            volatility_shock = scenario['volatility_shock']
            
            # Portfolio return impact
            portfolio_return = sum(weights.values()) * market_shock
            results['portfolio_return'] = portfolio_return
            
            # Portfolio volatility impact
            base_volatility = 0.20  # Assume 20% base volatility
            stressed_volatility = base_volatility * volatility_shock
            results['stressed_volatility'] = stressed_volatility
            
            # VaR impact
            var_shock = stats.norm.ppf(1 - self.confidence_level) * stressed_volatility
            results['stressed_var'] = abs(var_shock)
            
            # Maximum drawdown estimate
            max_drawdown = self._estimate_max_drawdown(portfolio_return, stressed_volatility)
            results['max_drawdown'] = max_drawdown
            
            # Liquidity impact
            liquidity_shock = scenario['liquidity_shock']
            results['liquidity_impact'] = liquidity_shock
            
            # Correlation impact
            correlation_shock = scenario['correlation_shock']
            results['correlation_impact'] = correlation_shock
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running historical scenario: {e}")
            return {}
    
    def _run_hypothetical_scenario(self, weights: Dict[str, float], 
                                 scenario_name: str) -> Dict[str, float]:
        """Run hypothetical scenario stress test."""
        try:
            scenario = self.hypothetical_scenarios[scenario_name]
            results = {}
            
            # Apply scenario shocks
            market_shock = scenario.get('market_shock', 0.0)
            volatility_shock = scenario.get('volatility_shock', 1.0)
            
            # Calculate impacts
            portfolio_return = sum(weights.values()) * market_shock
            results['portfolio_return'] = portfolio_return
            
            base_volatility = 0.20
            stressed_volatility = base_volatility * volatility_shock
            results['stressed_volatility'] = stressed_volatility
            
            var_shock = stats.norm.ppf(1 - self.confidence_level) * stressed_volatility
            results['stressed_var'] = abs(var_shock)
            
            max_drawdown = self._estimate_max_drawdown(portfolio_return, stressed_volatility)
            results['max_drawdown'] = max_drawdown
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running hypothetical scenario: {e}")
            return {}
    
    def _run_monte_carlo_stress(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Run Monte Carlo stress test."""
        try:
            results = {}
            
            # Generate random market shocks
            market_shocks = np.random.normal(-0.1, 0.3, self.num_simulations)  # Mean -10%, std 30%
            volatility_shocks = np.random.lognormal(0, 0.5, self.num_simulations)  # Log-normal distribution
            
            # Calculate portfolio impacts
            portfolio_returns = []
            portfolio_vars = []
            max_drawdowns = []
            
            for i in range(self.num_simulations):
                market_shock = market_shocks[i]
                vol_shock = volatility_shocks[i]
                
                # Portfolio return
                portfolio_return = sum(weights.values()) * market_shock
                portfolio_returns.append(portfolio_return)
                
                # Portfolio VaR
                base_vol = 0.20
                stressed_vol = base_vol * vol_shock
                var_shock = stats.norm.ppf(1 - self.confidence_level) * stressed_vol
                portfolio_vars.append(abs(var_shock))
                
                # Max drawdown
                max_dd = self._estimate_max_drawdown(portfolio_return, stressed_vol)
                max_drawdowns.append(max_dd)
            
            # Calculate statistics
            results['worst_case_return'] = np.percentile(portfolio_returns, 5)
            results['expected_return'] = np.mean(portfolio_returns)
            results['return_volatility'] = np.std(portfolio_returns)
            
            results['worst_case_var'] = np.percentile(portfolio_vars, 95)
            results['expected_var'] = np.mean(portfolio_vars)
            
            results['worst_case_drawdown'] = np.percentile(max_drawdowns, 95)
            results['expected_drawdown'] = np.mean(max_drawdowns)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running Monte Carlo stress test: {e}")
            return {}
    
    def _estimate_max_drawdown(self, return_shock: float, volatility: float) -> float:
        """Estimate maximum drawdown from return and volatility shocks."""
        try:
            # Simple drawdown estimation
            # In practice, this would be more sophisticated
            
            if return_shock < 0:
                # For negative returns, estimate drawdown
                drawdown = abs(return_shock) * (1 + volatility)
            else:
                # For positive returns, minimal drawdown
                drawdown = volatility * 0.1
            
            return min(drawdown, 1.0)  # Cap at 100%
            
        except Exception as e:
            self.logger.debug(f"Error estimating max drawdown: {e}")
            return 0.0
    
    def add_hypothetical_scenario(self, scenario_name: str, scenario_params: Dict):
        """Add a hypothetical stress scenario."""
        try:
            self.hypothetical_scenarios[scenario_name] = scenario_params
        except Exception as e:
            self.logger.error(f"Error adding hypothetical scenario: {e}")
    
    def run_sensitivity_analysis(self, weights: Dict[str, float], 
                               parameter: str, 
                               range_values: List[float]) -> Dict[str, List[float]]:
        """Run sensitivity analysis on a parameter."""
        try:
            results = {
                'parameter_values': range_values,
                'portfolio_returns': [],
                'portfolio_vars': [],
                'max_drawdowns': []
            }
            
            for value in range_values:
                # Create scenario with parameter value
                scenario = {
                    'market_shock': -0.2,  # Base market shock
                    'volatility_shock': 1.5,  # Base volatility shock
                }
                
                # Adjust scenario based on parameter
                if parameter == 'market_shock':
                    scenario['market_shock'] = value
                elif parameter == 'volatility_shock':
                    scenario['volatility_shock'] = value
                
                # Run stress test
                stress_results = self._run_hypothetical_scenario(weights, 'sensitivity')
                
                results['portfolio_returns'].append(stress_results.get('portfolio_return', 0.0))
                results['portfolio_vars'].append(stress_results.get('stressed_var', 0.0))
                results['max_drawdowns'].append(stress_results.get('max_drawdown', 0.0))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running sensitivity analysis: {e}")
            return {}
    
    def get_stress_test_summary(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Get summary of all stress tests."""
        try:
            summary = {}
            
            # Run all historical scenarios
            for scenario_name in self.historical_scenarios.keys():
                results = self._run_historical_scenario(weights, scenario_name)
                for metric, value in results.items():
                    summary[f"{scenario_name}_{metric}"] = value
            
            # Run Monte Carlo stress test
            mc_results = self._run_monte_carlo_stress(weights)
            for metric, value in mc_results.items():
                summary[f"monte_carlo_{metric}"] = value
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting stress test summary: {e}")
            return {}
    
    def get_available_scenarios(self) -> Dict[str, List[str]]:
        """Get list of available stress scenarios."""
        return {
            'historical': list(self.historical_scenarios.keys()),
            'hypothetical': list(self.hypothetical_scenarios.keys())
        }

