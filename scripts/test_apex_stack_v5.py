#!/usr/bin/env python3
"""
Test script for Apex Stack v5 components
Validates Wasserstein-Kelly, MOT superhedging, quantum risk acceleration, and microstructure models.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from infinite_money.optimization.wasserstein_kelly import WassersteinKellyOptimizer, KellyConstraints
from infinite_money.quantum.risk_acceleration import QuantumRiskAccelerator, QAEConfig
from infinite_money.risk.martingale_optimal_transport import MartingaleOptimalTransport, MOTConfig, PriceBand, HedgeBand
from infinite_money.ml.microstructure_models import MicrostructureEdge, MicrostructureConfig, LOBFeatures
from infinite_money.utils.logger import setup_logger


class ApexStackV5Tester:
    """Test suite for Apex Stack v5 components."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.logger = setup_logger("ApexStackV5Tester")
        self.test_results = {}
        
    def generate_test_data(self):
        """Generate synthetic test data."""
        np.random.seed(42)
        
        # Generate synthetic returns data
        n_assets = 10
        n_periods = 252  # One year of daily data
        
        # Generate correlated returns
        correlation_matrix = np.eye(n_assets) * 0.8 + np.ones((n_assets, n_assets)) * 0.2
        correlation_matrix = correlation_matrix / correlation_matrix.sum(axis=1, keepdims=True)
        
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=correlation_matrix * 0.02**2,  # 2% daily volatility
            size=n_periods
        )
        
        # Generate LOB data
        lob_data = []
        for i in range(100):
            bid_prices = np.random.uniform(100, 110, 10)
            ask_prices = bid_prices + np.random.uniform(0.1, 1.0, 10)
            bid_sizes = np.random.uniform(100, 1000, 10)
            ask_sizes = np.random.uniform(100, 1000, 10)
            order_flow = np.random.normal(0, 100, 20)
            
            lob = LOBFeatures(
                bid_prices=bid_prices,
                ask_prices=ask_prices,
                bid_sizes=bid_sizes,
                ask_sizes=ask_sizes,
                timestamp=datetime.now() + timedelta(seconds=i),
                spread=np.mean(ask_prices - bid_prices),
                mid_price=np.mean((bid_prices + ask_prices) / 2),
                order_flow=order_flow
            )
            lob_data.append(lob)
        
        return returns, lob_data
    
    def test_wasserstein_kelly(self):
        """Test Wasserstein-Kelly optimization."""
        self.logger.info("Testing Wasserstein-Kelly optimization...")
        
        try:
            # Generate test data
            returns, _ = self.generate_test_data()
            
            # Initialize optimizer
            constraints = KellyConstraints(
                cdar_budget=0.05,
                l_var_budget=0.03,
                max_leverage=2.0,
                ambiguity_radius=0.1
            )
            optimizer = WassersteinKellyOptimizer(constraints)
            
            # Test optimization
            current_wealth = 1000000
            market_conditions = {
                "liquidity_factor": 1.0,
                "volatility_factor": 1.0,
                "forecast_dispersion": 0.1,
                "regime_change_prob": 0.05
            }
            
            optimal_weights, diagnostics = optimizer.optimize_weights(
                returns, current_wealth, market_conditions
            )
            
            # Test leverage throttle
            leverage_throttle = optimizer.compute_leverage_throttle(
                optimal_weights, 0.1, 0.05
            )
            
            # Validate results
            assert optimal_weights is not None
            assert len(optimal_weights) == returns.shape[1]
            assert np.allclose(np.sum(optimal_weights), 1.0, atol=1e-3)
            assert leverage_throttle >= 0
            assert leverage_throttle <= constraints.max_leverage
            
            self.test_results["wasserstein_kelly"] = {
                "status": "PASS",
                "optimal_weights": optimal_weights,
                "leverage_throttle": leverage_throttle,
                "diagnostics": diagnostics
            }
            
            self.logger.info("Wasserstein-Kelly test PASSED")
            
        except Exception as e:
            self.test_results["wasserstein_kelly"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.logger.error(f"Wasserstein-Kelly test FAILED: {str(e)}")
    
    def test_quantum_risk_acceleration(self):
        """Test quantum risk acceleration."""
        self.logger.info("Testing quantum risk acceleration...")
        
        try:
            # Generate test data
            returns, _ = self.generate_test_data()
            
            # Initialize quantum risk accelerator
            config = QAEConfig(
                max_iterations=50,
                epsilon=0.01,
                alpha=0.05,
                error_mitigation=True,
                auto_fallback=True
            )
            quantum_risk = QuantumRiskAccelerator(config)
            
            # Test risk estimation
            portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]  # Equal weights
            
            var_estimate, var_diagnostics = quantum_risk.estimate_var(
                returns, portfolio_weights, confidence_level=0.95
            )
            
            cvar_estimate, cvar_diagnostics = quantum_risk.estimate_cvar(
                returns, portfolio_weights, confidence_level=0.95
            )
            
            pfe_estimate, pfe_diagnostics = quantum_risk.estimate_pfe(
                returns, portfolio_weights, time_horizon=10
            )
            
            # Validate results
            assert var_estimate is not None
            assert cvar_estimate is not None
            assert pfe_estimate is not None
            assert var_estimate <= 0  # VaR should be negative (loss)
            assert cvar_estimate <= var_estimate  # CVaR should be <= VaR
            
            self.test_results["quantum_risk_acceleration"] = {
                "status": "PASS",
                "var_estimate": var_estimate,
                "cvar_estimate": cvar_estimate,
                "pfe_estimate": pfe_estimate,
                "var_diagnostics": var_diagnostics,
                "cvar_diagnostics": cvar_diagnostics,
                "pfe_diagnostics": pfe_diagnostics
            }
            
            self.logger.info("Quantum risk acceleration test PASSED")
            
        except Exception as e:
            self.test_results["quantum_risk_acceleration"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.logger.error(f"Quantum risk acceleration test FAILED: {str(e)}")
    
    def test_martingale_optimal_transport(self):
        """Test MOT superhedging."""
        self.logger.info("Testing Martingale Optimal Transport...")
        
        try:
            # Generate test data
            returns, _ = self.generate_test_data()
            
            # Initialize MOT
            config = MOTConfig(
                num_time_steps=10,
                num_scenarios=100,
                confidence_level=0.95,
                transaction_cost=0.001
            )
            mot = MartingaleOptimalTransport(config)
            
            # Define simple payoff function
            def payoff_function(prices):
                return prices[-1] - prices[0] if len(prices) > 0 else 0.0
            
            # Test price bands
            constraints = {"volatility_bound": 0.1}
            price_band = mot.compute_price_bands(returns, payoff_function, constraints)
            
            # Test hedge bands
            target_payoff = 1000.0
            hedge_band = mot.compute_hedge_bands(returns, target_payoff, constraints)
            
            # Test admissibility
            portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]
            admissibility = mot.check_superhedge_admissibility(
                portfolio_weights, price_band, hedge_band
            )
            
            # Validate results
            assert price_band.lower_bound <= price_band.upper_bound
            assert price_band.confidence == config.confidence_level
            assert hedge_band.hedge_ratios is not None
            assert len(hedge_band.hedge_ratios) == returns.shape[1]
            assert "admissible" in admissibility
            
            self.test_results["martingale_optimal_transport"] = {
                "status": "PASS",
                "price_band": price_band,
                "hedge_band": hedge_band,
                "admissibility": admissibility
            }
            
            self.logger.info("Martingale Optimal Transport test PASSED")
            
        except Exception as e:
            self.test_results["martingale_optimal_transport"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.logger.error(f"Martingale Optimal Transport test FAILED: {str(e)}")
    
    def test_microstructure_models(self):
        """Test microstructure models."""
        self.logger.info("Testing microstructure models...")
        
        try:
            # Generate test data
            _, lob_data = self.generate_test_data()
            
            # Initialize microstructure edge
            config = MicrostructureConfig(
                sequence_length=50,
                num_levels=10,
                hidden_size=64,
                num_layers=2,
                dropout=0.1,
                learning_rate=1e-4,
                batch_size=16,
                device="cpu"  # Use CPU for testing
            )
            microstructure = MicrostructureEdge(config)
            
            # Test microstructure prediction
            if len(lob_data) > 0:
                predictions = microstructure.predict_microstructure(lob_data[0])
                
                # Test short-horizon signals
                signals = microstructure.get_short_horizon_signals(predictions)
                
                # Validate results
                assert "ensemble" in predictions
                assert "toxicity" in predictions
                assert "confidence" in predictions
                assert "direction" in signals
                assert "strength" in signals
                
                self.test_results["microstructure_models"] = {
                    "status": "PASS",
                    "predictions": predictions,
                    "signals": signals
                }
                
                self.logger.info("Microstructure models test PASSED")
            else:
                raise ValueError("No LOB data available for testing")
            
        except Exception as e:
            self.test_results["microstructure_models"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.logger.error(f"Microstructure models test FAILED: {str(e)}")
    
    def test_integration(self):
        """Test integration of all Apex Stack v5 components."""
        self.logger.info("Testing Apex Stack v5 integration...")
        
        try:
            # Generate test data
            returns, lob_data = self.generate_test_data()
            
            # Test portfolio optimization with all components
            from infinite_money.agents.portfolio_manager import PortfolioManagerAgent
            from infinite_money.utils.config import Config
            
            # Create mock config
            config = Config()
            agent_config = {
                "cdar_budget": 0.05,
                "l_var_budget": 0.03,
                "max_leverage": 2.0,
                "ambiguity_radius": 0.1,
                "qae_max_iterations": 50,
                "mot_scenarios": 100
            }
            
            # Initialize portfolio manager
            portfolio_manager = PortfolioManagerAgent(config, agent_config)
            
            # Test Apex Stack optimization
            data = {
                "returns": returns,
                "current_wealth": 1000000,
                "market_conditions": {
                    "liquidity_factor": 1.0,
                    "volatility_factor": 1.0,
                    "forecast_dispersion": 0.1,
                    "regime_change_prob": 0.05
                }
            }
            
            result = await portfolio_manager._optimize_portfolio_apex(data)
            
            # Validate integration results
            assert result["status"] == "success"
            assert "allocation" in result
            assert "kelly_diagnostics" in result
            assert "mot_admissibility" in result
            assert "quantum_diagnostics" in result
            
            self.test_results["integration"] = {
                "status": "PASS",
                "result": result
            }
            
            self.logger.info("Apex Stack v5 integration test PASSED")
            
        except Exception as e:
            self.test_results["integration"] = {
                "status": "FAIL",
                "error": str(e)
            }
            self.logger.error(f"Apex Stack v5 integration test FAILED: {str(e)}")
    
    def run_all_tests(self):
        """Run all Apex Stack v5 tests."""
        self.logger.info("Starting Apex Stack v5 comprehensive testing...")
        
        # Run individual component tests
        self.test_wasserstein_kelly()
        self.test_quantum_risk_acceleration()
        self.test_martingale_optimal_transport()
        self.test_microstructure_models()
        self.test_integration()
        
        # Generate summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate test summary report."""
        self.logger.info("Generating test summary...")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("APEX STACK V5 TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        print("="*60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result["status"] == "PASS" else "❌ FAIL"
            print(f"{test_name:.<40} {status}")
            if result["status"] == "FAIL":
                print(f"  Error: {result['error']}")
        
        print("="*60)
        
        if failed_tests == 0:
            self.logger.info("All Apex Stack v5 tests PASSED! 🎉")
            return True
        else:
            self.logger.error(f"{failed_tests} Apex Stack v5 tests FAILED!")
            return False


async def main():
    """Main test function."""
    tester = ApexStackV5Tester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All Apex Stack v5 components are working correctly!")
        print("The system is ready for advanced quantum trading tasks.")
    else:
        print("\n⚠️  Some Apex Stack v5 components need attention.")
        print("Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)