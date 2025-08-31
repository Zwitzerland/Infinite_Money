"""
Test Cointegration Module

Tests for cointegration analysis functionality.
"""

import unittest
import numpy as np
import pandas as pd
from src.alpha_models.stat_arb import StatisticalArbitrage


class TestCointegration(unittest.TestCase):
    """Test cases for cointegration analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stat_arb = StatisticalArbitrage(lookback_period=252, z_threshold=2.0)
        
    def test_initialization(self):
        """Test StatisticalArbitrage initialization."""
        self.assertEqual(self.stat_arb.lookback_period, 252)
        self.assertEqual(self.stat_arb.z_threshold, 2.0)
        self.assertIsInstance(self.stat_arb.pairs, dict)
        self.assertIsInstance(self.stat_arb.spread_history, dict)
    
    def test_calculate_zscore(self):
        """Test z-score calculation."""
        # Test with normal distribution
        series = np.random.normal(0, 1, 100)
        z_score = self.stat_arb._calculate_zscore(series)
        
        self.assertIsInstance(z_score, float)
        self.assertGreaterEqual(z_score, -10)  # Reasonable bounds
        self.assertLessEqual(z_score, 10)
        
        # Test with constant series
        constant_series = np.ones(10)
        z_score = self.stat_arb._calculate_zscore(constant_series)
        self.assertEqual(z_score, 0.0)
        
        # Test with single value
        single_series = np.array([1.0])
        z_score = self.stat_arb._calculate_zscore(single_series)
        self.assertEqual(z_score, 0.0)
    
    def test_find_cointegrated_pairs(self):
        """Test cointegrated pairs detection."""
        # Create test data
        data = {
            'AAPL': np.random.normal(0, 1, 100),
            'MSFT': np.random.normal(0, 1, 100),
            'GOOGL': np.random.normal(0, 1, 100)
        }
        
        pairs = self.stat_arb._find_cointegrated_pairs(list(data.keys()), data)
        
        self.assertIsInstance(pairs, list)
        # Each pair should be a tuple of two symbols
        for pair in pairs:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair[0], str)
            self.assertIsInstance(pair[1], str)
    
    def test_calculate_spread(self):
        """Test spread calculation."""
        # Create correlated price series
        np.random.seed(42)
        n = 100
        common_factor = np.random.normal(0, 1, n)
        
        # Create two correlated series
        prices1 = common_factor + np.random.normal(0, 0.1, n)
        prices2 = 2 * common_factor + np.random.normal(0, 0.1, n)
        
        data = {
            'AAPL': prices1,
            'MSFT': prices2
        }
        
        spread = self.stat_arb._calculate_spread('AAPL', 'MSFT', data)
        
        self.assertIsInstance(spread, np.ndarray)
        self.assertEqual(len(spread), n)
        
        # Test with None data
        spread = self.stat_arb._calculate_spread('AAPL', 'INVALID', data)
        self.assertIsNone(spread)
    
    def test_generate_pairs_signals(self):
        """Test pairs trading signal generation."""
        # Create test data with some cointegration
        np.random.seed(42)
        n = 100
        common_factor = np.random.normal(0, 1, n)
        
        data = {
            'AAPL': common_factor + np.random.normal(0, 0.1, n),
            'MSFT': 2 * common_factor + np.random.normal(0, 0.1, n),
            'GOOGL': np.random.normal(0, 1, n)  # Independent
        }
        
        signals = self.stat_arb._generate_pairs_signals(data)
        
        self.assertIsInstance(signals, dict)
        # Should contain signals for symbols in pairs
        for symbol in signals:
            self.assertIn(symbol, data.keys())
            self.assertIsInstance(signals[symbol], float)
            self.assertGreaterEqual(signals[symbol], -1.0)
            self.assertLessEqual(signals[symbol], 1.0)
    
    def test_generate_mean_reversion_signals(self):
        """Test mean reversion signal generation."""
        # Create test data
        data = {
            'AAPL': np.random.normal(100, 10, 300),  # 300 days of data
            'MSFT': np.random.normal(200, 20, 300)
        }
        
        signals = self.stat_arb._generate_mean_reversion_signals(data)
        
        self.assertIsInstance(signals, dict)
        for symbol in signals:
            self.assertIn(symbol, data.keys())
            self.assertIsInstance(signals[symbol], float)
    
    def test_generate_momentum_reversal_signals(self):
        """Test momentum reversal signal generation."""
        # Create test data with trends
        n = 50
        data = {
            'AAPL': np.linspace(100, 150, n) + np.random.normal(0, 2, n),
            'MSFT': np.linspace(200, 180, n) + np.random.normal(0, 3, n)
        }
        
        signals = self.stat_arb._generate_momentum_reversal_signals(data)
        
        self.assertIsInstance(signals, dict)
        for symbol in signals:
            self.assertIn(symbol, data.keys())
            self.assertIsInstance(signals[symbol], float)
    
    def test_generate_volatility_signals(self):
        """Test volatility-based signal generation."""
        # Create test data with varying volatility
        n = 50
        data = {
            'AAPL': np.random.normal(0, 0.02, n),  # Low volatility
            'MSFT': np.random.normal(0, 0.05, n)   # High volatility
        }
        
        signals = self.stat_arb._generate_volatility_signals(data)
        
        self.assertIsInstance(signals, dict)
        for symbol in signals:
            self.assertIn(symbol, data.keys())
            self.assertIsInstance(signals[symbol], float)
    
    def test_get_price_series(self):
        """Test price series extraction."""
        # Test with different data formats
        data_with_close = {'AAPL': type('obj', (object,), {'close': np.array([1, 2, 3])})}
        data_with_price = {'MSFT': type('obj', (object,), {'Price': np.array([4, 5, 6])})}
        data_invalid = {'GOOGL': type('obj', (object,), {})}
        
        # Test close price extraction
        prices = self.stat_arb._get_price_series('AAPL', data_with_close)
        self.assertIsInstance(prices, np.ndarray)
        self.assertEqual(len(prices), 3)
        
        # Test Price extraction
        prices = self.stat_arb._get_price_series('MSFT', data_with_price)
        self.assertIsInstance(prices, np.ndarray)
        self.assertEqual(len(prices), 3)
        
        # Test invalid data
        prices = self.stat_arb._get_price_series('GOOGL', data_invalid)
        self.assertIsNone(prices)
    
    def test_update_pairs(self):
        """Test pairs update functionality."""
        new_pairs = [('AAPL', 'MSFT'), ('GOOGL', 'TSLA')]
        
        self.stat_arb.update_pairs(new_pairs)
        
        self.assertEqual(self.stat_arb.pairs, new_pairs)
    
    def test_get_pair_statistics(self):
        """Test pair statistics calculation."""
        # Add some spread history
        self.stat_arb.spread_history['AAPL-MSFT'] = np.random.normal(0, 1, 100)
        self.stat_arb.spread_history['GOOGL-TSLA'] = np.random.normal(0, 2, 50)
        
        # Update pairs
        self.stat_arb.pairs = [('AAPL', 'MSFT'), ('GOOGL', 'TSLA')]
        
        stats = self.stat_arb.get_pair_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('AAPL-MSFT', stats)
        self.assertIn('GOOGL-TSLA', stats)
        
        # Check statistics structure
        for pair_key, pair_stats in stats.items():
            self.assertIn('mean', pair_stats)
            self.assertIn('std', pair_stats)
            self.assertIn('current_zscore', pair_stats)
            self.assertIn('length', pair_stats)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty data
        signals = self.stat_arb.generate_signals({})
        self.assertEqual(signals, {})
        
        # Test with single symbol
        data = {'AAPL': np.random.normal(0, 1, 10)}
        signals = self.stat_arb.generate_signals(data)
        self.assertIsInstance(signals, dict)
        
        # Test with insufficient data
        data = {'AAPL': np.random.normal(0, 1, 5)}  # Less than lookback period
        signals = self.stat_arb.generate_signals(data)
        self.assertIsInstance(signals, dict)


if __name__ == '__main__':
    unittest.main()
