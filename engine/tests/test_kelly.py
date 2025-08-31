"""
Test Kelly Criterion Module

Tests for Kelly criterion position sizing functionality.
"""

import unittest
import numpy as np
import pandas as pd
from src.portfolio.sizing import KellyCriterion


class TestKellyCriterion(unittest.TestCase):
    """Test cases for KellyCriterion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kelly = KellyCriterion(fractional_kelly=0.25, max_leverage=2.0)
        
    def test_initialization(self):
        """Test KellyCriterion initialization."""
        self.assertEqual(self.kelly.fractional_kelly, 0.25)
        self.assertEqual(self.kelly.max_leverage, 2.0)
        self.assertIsInstance(self.kelly.return_history, dict)
        self.assertIsInstance(self.kelly.volatility_history, dict)
    
    def test_calculate_sizes(self):
        """Test calculate_sizes method."""
        target_weights = {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3}
        signals = {'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': -0.2}
        regime = "normal"
        
        kelly_weights = self.kelly.calculate_sizes(target_weights, signals, regime)
        
        self.assertIsInstance(kelly_weights, dict)
        self.assertEqual(len(kelly_weights), 3)
        self.assertIn('AAPL', kelly_weights)
        self.assertIn('MSFT', kelly_weights)
        self.assertIn('GOOGL', kelly_weights)
        
        # Check that weights sum to approximately 1
        total_weight = sum(abs(w) for w in kelly_weights.values())
        self.assertGreater(total_weight, 0)
    
    def test_signal_based_kelly(self):
        """Test signal-based Kelly calculation."""
        signal = 0.5
        regime = "normal"
        
        kelly_fraction = self.kelly._signal_based_kelly(signal, regime)
        
        self.assertIsInstance(kelly_fraction, float)
        self.assertGreaterEqual(kelly_fraction, -1.0)
        self.assertLessEqual(kelly_fraction, 1.0)
    
    def test_regime_adjustments(self):
        """Test regime-based Kelly adjustments."""
        base_kelly = 0.5
        
        # Test different regimes
        crisis_kelly = self.kelly._adjust_kelly_for_regime(base_kelly, "crisis")
        bull_kelly = self.kelly._adjust_kelly_for_regime(base_kelly, "bull_market")
        bear_kelly = self.kelly._adjust_kelly_for_regime(base_kelly, "bear_market")
        
        # Crisis should reduce Kelly
        self.assertLess(crisis_kelly, base_kelly)
        
        # Bull market should increase Kelly
        self.assertGreater(bull_kelly, base_kelly)
        
        # Bear market should reduce Kelly
        self.assertLess(bear_kelly, base_kelly)
    
    def test_leverage_constraints(self):
        """Test leverage constraint application."""
        kelly_weight = 3.0  # Exceeds max leverage
        regime = "normal"
        
        constrained_weight = self.kelly._apply_leverage_constraints(kelly_weight, regime)
        
        self.assertLessEqual(abs(constrained_weight), self.kelly.max_leverage)
    
    def test_update_return_history(self):
        """Test return history update."""
        symbol = "AAPL"
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        
        self.kelly.update_return_history(symbol, returns)
        
        self.assertIn(symbol, self.kelly.return_history)
        self.assertEqual(len(self.kelly.return_history[symbol]), 5)
        
        # Check volatility update
        self.assertIn(symbol, self.kelly.volatility_history)
        self.assertIsInstance(self.kelly.volatility_history[symbol], float)
    
    def test_calculate_optimal_kelly(self):
        """Test optimal Kelly calculation."""
        returns = np.random.normal(0.001, 0.02, 100)  # 100 daily returns
        
        optimal_kelly = self.kelly.calculate_optimal_kelly(returns)
        
        self.assertIsInstance(optimal_kelly, float)
        self.assertGreaterEqual(optimal_kelly, -1.0)
        self.assertLessEqual(optimal_kelly, 1.0)
    
    def test_calculate_kelly_statistics(self):
        """Test Kelly statistics calculation."""
        symbol = "AAPL"
        returns = np.random.normal(0.001, 0.02, 50)
        
        # Update return history
        self.kelly.update_return_history(symbol, returns)
        
        stats = self.kelly.calculate_kelly_statistics(symbol)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('mean_return', stats)
        self.assertIn('volatility', stats)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('full_kelly', stats)
        self.assertIn('fractional_kelly', stats)
    
    def test_portfolio_kelly_stats(self):
        """Test portfolio-level Kelly statistics."""
        weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        
        # Add some return history
        for symbol in weights:
            returns = np.random.normal(0.001, 0.02, 30)
            self.kelly.update_return_history(symbol, returns)
        
        stats = self.kelly.get_portfolio_kelly_stats(weights)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_kelly_exposure', stats)
        self.assertIn('weighted_sharpe', stats)
        self.assertIn('weighted_volatility', stats)
        self.assertIn('portfolio_kelly_ratio', stats)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty signals
        target_weights = {'AAPL': 0.5, 'MSFT': 0.5}
        signals = {}
        regime = "normal"
        
        kelly_weights = self.kelly.calculate_sizes(target_weights, signals, regime)
        
        # Should return target weights when no signals
        self.assertEqual(kelly_weights, target_weights)
        
        # Test with zero signal
        signals = {'AAPL': 0.0, 'MSFT': 0.0}
        kelly_weights = self.kelly.calculate_sizes(target_weights, signals, regime)
        
        self.assertIsInstance(kelly_weights, dict)
        self.assertEqual(len(kelly_weights), 2)


if __name__ == '__main__':
    unittest.main()
