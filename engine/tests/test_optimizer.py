"""
Test Portfolio Optimizer Module

Tests for portfolio optimization functionality.
"""

import unittest
import numpy as np
import pandas as pd
from src.portfolio.optimizer import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PortfolioOptimizer(lookback_period=252, risk_free_rate=0.02)
        
    def test_initialization(self):
        """Test PortfolioOptimizer initialization."""
        self.assertEqual(self.optimizer.lookback_period, 252)
        self.assertEqual(self.optimizer.risk_free_rate, 0.02)
        self.assertIsInstance(self.optimizer.returns_history, dict)
        self.assertIsNone(self.optimizer.covariance_matrix)
        self.assertIsNone(self.optimizer.expected_returns)
    
    def test_update_return_estimates(self):
        """Test return estimates update."""
        signals = {'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': -0.2}
        
        self.optimizer._update_return_estimates(signals)
        
        self.assertIsInstance(self.optimizer.expected_returns, dict)
        self.assertEqual(len(self.optimizer.expected_returns), 3)
        self.assertIn('AAPL', self.optimizer.expected_returns)
        self.assertIn('MSFT', self.optimizer.expected_returns)
        self.assertIn('GOOGL', self.optimizer.expected_returns)
        
        # Check that expected returns are reasonable
        for symbol, expected_return in self.optimizer.expected_returns.items():
            self.assertIsInstance(expected_return, float)
            self.assertGreaterEqual(expected_return, -0.1)  # Max 10% negative return
            self.assertLessEqual(expected_return, 0.1)      # Max 10% positive return
    
    def test_mean_variance_optimization(self):
        """Test mean-variance optimization."""
        # Set up expected returns and covariance matrix
        self.optimizer.expected_returns = {'AAPL': 0.05, 'MSFT': 0.04, 'GOOGL': 0.06}
        
        # Create a simple covariance matrix
        cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.optimizer.covariance_matrix = cov_matrix
        
        weights = self.optimizer._mean_variance_optimization()
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check that all weights are non-negative (long-only constraint)
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0.0)
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        # Set up covariance matrix
        cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.optimizer.covariance_matrix = cov_matrix
        
        weights = self.optimizer._risk_parity_optimization()
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check that all weights are non-negative
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0.0)
    
    def test_minimum_variance_optimization(self):
        """Test minimum variance optimization."""
        # Set up covariance matrix
        cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.optimizer.covariance_matrix = cov_matrix
        
        weights = self.optimizer._minimum_variance_optimization()
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check that all weights are non-negative
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0.0)
    
    def test_maximum_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        # Set up expected returns and covariance matrix
        self.optimizer.expected_returns = {'AAPL': 0.05, 'MSFT': 0.04, 'GOOGL': 0.06}
        
        cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.optimizer.covariance_matrix = cov_matrix
        
        weights = self.optimizer._maximum_sharpe_optimization()
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check that all weights are non-negative
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0.0)
    
    def test_apply_constraints(self):
        """Test constraint application."""
        weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        current_positions = {'AAPL': 0.35, 'MSFT': 0.35, 'GOOGL': 0.3}
        regime = "normal"
        
        constrained_weights = self.optimizer._apply_constraints(weights, current_positions, regime)
        
        self.assertIsInstance(constrained_weights, dict)
        self.assertEqual(len(constrained_weights), 3)
        
        # Check that weights sum to approximately 1
        total_weight = sum(constrained_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check that all weights are non-negative
        for weight in constrained_weights.values():
            self.assertGreaterEqual(weight, 0.0)
    
    def test_equal_weight_fallback(self):
        """Test equal weight fallback."""
        signals = {'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': -0.2}
        
        weights = self.optimizer._equal_weight_fallback(signals)
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        
        # Check that all weights are equal
        expected_weight = 1.0 / 3
        for weight in weights.values():
            self.assertAlmostEqual(weight, expected_weight, places=6)
    
    def test_update_covariance_matrix(self):
        """Test covariance matrix update."""
        returns_data = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.025, 100),
            'GOOGL': np.random.normal(0.001, 0.03, 100)
        }
        
        self.optimizer.update_covariance_matrix(returns_data)
        
        self.assertIsNotNone(self.optimizer.covariance_matrix)
        self.assertIsInstance(self.optimizer.covariance_matrix, np.ndarray)
        self.assertEqual(self.optimizer.covariance_matrix.shape, (3, 3))
        
        # Check that covariance matrix is symmetric
        np.testing.assert_array_almost_equal(
            self.optimizer.covariance_matrix,
            self.optimizer.covariance_matrix.T
        )
    
    def test_get_portfolio_statistics(self):
        """Test portfolio statistics calculation."""
        weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        
        # Set up expected returns and covariance matrix
        self.optimizer.expected_returns = {'AAPL': 0.05, 'MSFT': 0.04, 'GOOGL': 0.06}
        
        cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.optimizer.covariance_matrix = cov_matrix
        
        stats = self.optimizer.get_portfolio_statistics(weights)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('expected_return', stats)
        self.assertIn('volatility', stats)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('concentration', stats)
        self.assertIn('num_positions', stats)
        
        # Check that statistics are reasonable
        self.assertIsInstance(stats['expected_return'], float)
        self.assertIsInstance(stats['volatility'], float)
        self.assertIsInstance(stats['sharpe_ratio'], float)
        self.assertIsInstance(stats['concentration'], float)
        self.assertIsInstance(stats['num_positions'], int)
        
        self.assertGreaterEqual(stats['volatility'], 0.0)
        self.assertGreaterEqual(stats['concentration'], 0.0)
        self.assertLessEqual(stats['concentration'], 1.0)
        self.assertGreaterEqual(stats['num_positions'], 0)
    
    def test_optimize_method(self):
        """Test main optimize method."""
        signals = {'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': -0.2}
        current_positions = {'AAPL': 0.35, 'MSFT': 0.35, 'GOOGL': 0.3}
        regime = "normal"
        
        # Set up covariance matrix
        returns_data = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.025, 100),
            'GOOGL': np.random.normal(0.001, 0.03, 100)
        }
        self.optimizer.update_covariance_matrix(returns_data)
        
        weights = self.optimizer.optimize(signals, current_positions, regime)
        
        self.assertIsInstance(weights, dict)
        self.assertEqual(len(weights), 3)
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # Check that all weights are non-negative
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0.0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty signals
        weights = self.optimizer.optimize({}, {}, "normal")
        self.assertEqual(weights, {})
        
        # Test with single symbol
        signals = {'AAPL': 0.5}
        current_positions = {'AAPL': 0.5}
        weights = self.optimizer.optimize(signals, current_positions, "normal")
        self.assertIsInstance(weights, dict)
        
        # Test with missing covariance matrix
        signals = {'AAPL': 0.5, 'MSFT': 0.3}
        current_positions = {'AAPL': 0.5, 'MSFT': 0.5}
        weights = self.optimizer.optimize(signals, current_positions, "normal")
        self.assertIsInstance(weights, dict)


if __name__ == '__main__':
    unittest.main()
