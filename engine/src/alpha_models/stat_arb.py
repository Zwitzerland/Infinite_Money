"""
Statistical Arbitrage Module

Implements various statistical arbitrage strategies:
- Pairs trading
- Mean reversion
- Cointegration-based strategies
- Momentum reversal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import logging


class StatisticalArbitrage:
    """
    Statistical arbitrage signal generator.
    
    Implements various statistical arbitrage strategies including:
    - Pairs trading based on cointegration
    - Mean reversion signals
    - Momentum reversal
    - Volatility-based signals
    """
    
    def __init__(self, lookback_period: int = 252, z_threshold: float = 2.0):
        """
        Initialize the statistical arbitrage model.
        
        Args:
            lookback_period: Number of days for lookback window
            z_threshold: Z-score threshold for signal generation
        """
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.pairs = {}
        self.spread_history = {}
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, data) -> Dict[str, float]:
        """
        Generate statistical arbitrage signals.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary mapping symbols to signal strengths
        """
        signals = {}
        
        try:
            # Generate pairs trading signals
            pairs_signals = self._generate_pairs_signals(data)
            signals.update(pairs_signals)
            
            # Generate mean reversion signals
            mean_rev_signals = self._generate_mean_reversion_signals(data)
            signals.update(mean_rev_signals)
            
            # Generate momentum reversal signals
            momentum_signals = self._generate_momentum_reversal_signals(data)
            signals.update(momentum_signals)
            
            # Generate volatility signals
            vol_signals = self._generate_volatility_signals(data)
            signals.update(vol_signals)
            
        except Exception as e:
            self.logger.error(f"Error generating stat arb signals: {e}")
            
        return signals
    
    def _generate_pairs_signals(self, data) -> Dict[str, float]:
        """Generate pairs trading signals based on cointegration."""
        signals = {}
        
        # Find cointegrated pairs
        symbols = list(data.keys())
        pairs = self._find_cointegrated_pairs(symbols, data)
        
        for pair in pairs:
            symbol1, symbol2 = pair
            
            # Calculate spread
            spread = self._calculate_spread(symbol1, symbol2, data)
            
            if spread is not None:
                # Calculate z-score
                z_score = self._calculate_zscore(spread)
                
                # Generate signals based on z-score
                if z_score > self.z_threshold:
                    signals[symbol1] = -0.5  # Short signal for symbol1
                    signals[symbol2] = 0.5   # Long signal for symbol2
                elif z_score < -self.z_threshold:
                    signals[symbol1] = 0.5   # Long signal for symbol1
                    signals[symbol2] = -0.5  # Short signal for symbol2
        
        return signals
    
    def _find_cointegrated_pairs(self, symbols: List[str], data) -> List[Tuple[str, str]]:
        """Find cointegrated pairs using Engle-Granger test."""
        pairs = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                try:
                    # Get price data
                    prices1 = self._get_price_series(symbol1, data)
                    prices2 = self._get_price_series(symbol2, data)
                    
                    if prices1 is not None and prices2 is not None:
                        # Test for cointegration
                        score, pvalue, _ = coint(prices1, prices2)
                        
                        if pvalue < 0.05:  # 95% confidence level
                            pairs.append((symbol1, symbol2))
                            
                except Exception as e:
                    self.logger.debug(f"Error testing cointegration for {symbol1}-{symbol2}: {e}")
                    continue
        
        return pairs
    
    def _calculate_spread(self, symbol1: str, symbol2: str, data) -> Optional[np.ndarray]:
        """Calculate the spread between two symbols."""
        try:
            prices1 = self._get_price_series(symbol1, data)
            prices2 = self._get_price_series(symbol2, data)
            
            if prices1 is None or prices2 is None:
                return None
            
            # Calculate hedge ratio using OLS
            model = OLS(prices1, prices2).fit()
            hedge_ratio = model.params[0]
            
            # Calculate spread
            spread = prices1 - hedge_ratio * prices2
            
            return spread
            
        except Exception as e:
            self.logger.debug(f"Error calculating spread for {symbol1}-{symbol2}: {e}")
            return None
    
    def _calculate_zscore(self, series: np.ndarray) -> float:
        """Calculate z-score of a series."""
        if len(series) < 2:
            return 0.0
        
        mean = np.mean(series)
        std = np.std(series)
        
        if std == 0:
            return 0.0
        
        return (series[-1] - mean) / std
    
    def _generate_mean_reversion_signals(self, data) -> Dict[str, float]:
        """Generate mean reversion signals based on price deviations."""
        signals = {}
        
        for symbol in data.keys():
            try:
                prices = self._get_price_series(symbol, data)
                
                if prices is not None and len(prices) > self.lookback_period:
                    # Calculate moving average
                    ma = np.mean(prices[-self.lookback_period:])
                    current_price = prices[-1]
                    
                    # Calculate deviation
                    deviation = (current_price - ma) / ma
                    
                    # Generate signal based on deviation
                    if deviation > 0.1:  # 10% above mean
                        signals[symbol] = -0.3  # Short signal
                    elif deviation < -0.1:  # 10% below mean
                        signals[symbol] = 0.3   # Long signal
                        
            except Exception as e:
                self.logger.debug(f"Error generating mean reversion signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _generate_momentum_reversal_signals(self, data) -> Dict[str, float]:
        """Generate momentum reversal signals."""
        signals = {}
        
        for symbol in data.keys():
            try:
                prices = self._get_price_series(symbol, data)
                
                if prices is not None and len(prices) > 20:
                    # Calculate short and long momentum
                    short_momentum = (prices[-1] - prices[-5]) / prices[-5]
                    long_momentum = (prices[-1] - prices[-20]) / prices[-20]
                    
                    # Signal when short momentum diverges from long momentum
                    if short_momentum > 0.05 and long_momentum < -0.05:
                        signals[symbol] = -0.2  # Reversal short signal
                    elif short_momentum < -0.05 and long_momentum > 0.05:
                        signals[symbol] = 0.2   # Reversal long signal
                        
            except Exception as e:
                self.logger.debug(f"Error generating momentum reversal signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _generate_volatility_signals(self, data) -> Dict[str, float]:
        """Generate volatility-based signals."""
        signals = {}
        
        for symbol in data.keys():
            try:
                prices = self._get_price_series(symbol, data)
                
                if prices is not None and len(prices) > 20:
                    # Calculate returns
                    returns = np.diff(np.log(prices))
                    
                    # Calculate volatility
                    vol = np.std(returns[-20:])
                    vol_ma = np.mean([np.std(returns[i:i+20]) for i in range(len(returns)-40, len(returns)-20)])
                    
                    # Signal when volatility is mean-reverting
                    if vol > vol_ma * 1.5:
                        signals[symbol] = -0.1  # Volatility short signal
                    elif vol < vol_ma * 0.5:
                        signals[symbol] = 0.1   # Volatility long signal
                        
            except Exception as e:
                self.logger.debug(f"Error generating volatility signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _get_price_series(self, symbol: str, data) -> Optional[np.ndarray]:
        """Extract price series for a symbol from data."""
        try:
            if symbol in data and hasattr(data[symbol], 'close'):
                return np.array(data[symbol].close)
            elif symbol in data and hasattr(data[symbol], 'Price'):
                return np.array(data[symbol].Price)
            else:
                return None
        except Exception:
            return None
    
    def update_pairs(self, new_pairs: List[Tuple[str, str]]):
        """Update the list of trading pairs."""
        self.pairs = new_pairs
    
    def get_pair_statistics(self) -> Dict[str, Dict]:
        """Get statistics for current pairs."""
        stats = {}
        
        for pair in self.pairs:
            symbol1, symbol2 = pair
            pair_key = f"{symbol1}-{symbol2}"
            
            if pair_key in self.spread_history:
                spread = self.spread_history[pair_key]
                stats[pair_key] = {
                    'mean': np.mean(spread),
                    'std': np.std(spread),
                    'current_zscore': self._calculate_zscore(spread),
                    'length': len(spread)
                }
        
        return stats
