"""
Factor Models Module

Implements various factor models for alpha generation:
- Fama-French factors
- Momentum factors
- Quality factors
- Value factors
- Size factors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging


class FactorModel:
    """
    Multi-factor model for alpha signal generation.
    
    Implements various factor models including:
    - Fama-French three-factor model
    - Momentum factors
    - Quality factors
    - Value factors
    - Size factors
    """
    
    def __init__(self, lookback_period: int = 252):
        """
        Initialize the factor model.
        
        Args:
            lookback_period: Number of days for lookback window
        """
        self.lookback_period = lookback_period
        self.factors = {}
        self.factor_loadings = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, data) -> Dict[str, float]:
        """
        Generate factor-based alpha signals.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary mapping symbols to signal strengths
        """
        signals = {}
        
        try:
            # Calculate factor returns
            factor_returns = self._calculate_factor_returns(data)
            
            # Calculate factor loadings
            factor_loadings = self._calculate_factor_loadings(data)
            
            # Generate signals based on factor exposures
            signals = self._generate_factor_signals(factor_returns, factor_loadings)
            
        except Exception as e:
            self.logger.error(f"Error generating factor signals: {e}")
            
        return signals
    
    def _calculate_factor_returns(self, data) -> Dict[str, np.ndarray]:
        """Calculate factor returns."""
        factor_returns = {}
        
        try:
            # Market factor (excess return)
            market_returns = self._calculate_market_returns(data)
            factor_returns['market'] = market_returns
            
            # Size factor (SMB - Small Minus Big)
            smb_returns = self._calculate_size_factor(data)
            factor_returns['size'] = smb_returns
            
            # Value factor (HML - High Minus Low)
            hml_returns = self._calculate_value_factor(data)
            factor_returns['value'] = hml_returns
            
            # Momentum factor
            momentum_returns = self._calculate_momentum_factor(data)
            factor_returns['momentum'] = momentum_returns
            
            # Quality factor
            quality_returns = self._calculate_quality_factor(data)
            factor_returns['quality'] = quality_returns
            
            # Volatility factor
            volatility_returns = self._calculate_volatility_factor(data)
            factor_returns['volatility'] = volatility_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating factor returns: {e}")
            
        return factor_returns
    
    def _calculate_market_returns(self, data) -> np.ndarray:
        """Calculate market excess returns."""
        try:
            # Use SPY as market proxy
            if 'SPY' in data:
                spy_prices = self._get_price_series('SPY', data)
                if spy_prices is not None:
                    returns = np.diff(np.log(spy_prices))
                    # Assume risk-free rate of 2%
                    risk_free_rate = 0.02 / 252
                    excess_returns = returns - risk_free_rate
                    return excess_returns
        except Exception as e:
            self.logger.debug(f"Error calculating market returns: {e}")
            
        return np.zeros(self.lookback_period)
    
    def _calculate_size_factor(self, data) -> np.ndarray:
        """Calculate size factor (SMB) returns."""
        try:
            # Divide stocks into small and big based on market cap
            small_cap_returns = []
            big_cap_returns = []
            
            for symbol in data.keys():
                if symbol != 'SPY':
                    prices = self._get_price_series(symbol, data)
                    if prices is not None and len(prices) > 1:
                        returns = np.diff(np.log(prices))
                        
                        # Simple size classification (in practice, use market cap)
                        if len(returns) > 0:
                            if np.random.random() > 0.5:  # Placeholder logic
                                small_cap_returns.append(returns)
                            else:
                                big_cap_returns.append(returns)
            
            if small_cap_returns and big_cap_returns:
                small_avg = np.mean(small_cap_returns, axis=0)
                big_avg = np.mean(big_cap_returns, axis=0)
                smb = small_avg - big_avg
                return smb
                
        except Exception as e:
            self.logger.debug(f"Error calculating size factor: {e}")
            
        return np.zeros(self.lookback_period)
    
    def _calculate_value_factor(self, data) -> np.ndarray:
        """Calculate value factor (HML) returns."""
        try:
            # Divide stocks into high and low book-to-market
            high_bm_returns = []
            low_bm_returns = []
            
            for symbol in data.keys():
                if symbol != 'SPY':
                    prices = self._get_price_series(symbol, data)
                    if prices is not None and len(prices) > 1:
                        returns = np.diff(np.log(prices))
                        
                        # Simple value classification (in practice, use B/M ratio)
                        if len(returns) > 0:
                            if np.random.random() > 0.5:  # Placeholder logic
                                high_bm_returns.append(returns)
                            else:
                                low_bm_returns.append(returns)
            
            if high_bm_returns and low_bm_returns:
                high_avg = np.mean(high_bm_returns, axis=0)
                low_avg = np.mean(low_bm_returns, axis=0)
                hml = high_avg - low_avg
                return hml
                
        except Exception as e:
            self.logger.debug(f"Error calculating value factor: {e}")
            
        return np.zeros(self.lookback_period)
    
    def _calculate_momentum_factor(self, data) -> np.ndarray:
        """Calculate momentum factor returns."""
        try:
            # Calculate momentum for each stock
            momentum_returns = []
            
            for symbol in data.keys():
                if symbol != 'SPY':
                    prices = self._get_price_series(symbol, data)
                    if prices is not None and len(prices) > 20:
                        # 20-day momentum
                        momentum = (prices[-1] - prices[-20]) / prices[-20]
                        returns = np.diff(np.log(prices))
                        
                        if len(returns) > 0:
                            momentum_returns.append(returns)
            
            if momentum_returns:
                # Sort by momentum and take long-short
                momentum_scores = [np.mean(ret) for ret in momentum_returns]
                sorted_indices = np.argsort(momentum_scores)
                
                n = len(momentum_returns)
                winner_returns = np.mean([momentum_returns[i] for i in sorted_indices[-n//3:]], axis=0)
                loser_returns = np.mean([momentum_returns[i] for i in sorted_indices[:n//3]], axis=0)
                
                return winner_returns - loser_returns
                
        except Exception as e:
            self.logger.debug(f"Error calculating momentum factor: {e}")
            
        return np.zeros(self.lookback_period)
    
    def _calculate_quality_factor(self, data) -> np.ndarray:
        """Calculate quality factor returns."""
        try:
            # Quality based on volatility (lower vol = higher quality)
            quality_returns = []
            
            for symbol in data.keys():
                if symbol != 'SPY':
                    prices = self._get_price_series(symbol, data)
                    if prices is not None and len(prices) > 20:
                        returns = np.diff(np.log(prices))
                        volatility = np.std(returns[-20:])
                        
                        if len(returns) > 0:
                            quality_returns.append((returns, volatility))
            
            if quality_returns:
                # Sort by quality (volatility) and take long-short
                quality_returns.sort(key=lambda x: x[1])
                n = len(quality_returns)
                
                high_quality = np.mean([ret[0] for ret in quality_returns[:n//3]], axis=0)
                low_quality = np.mean([ret[0] for ret in quality_returns[-n//3:]], axis=0)
                
                return high_quality - low_quality
                
        except Exception as e:
            self.logger.debug(f"Error calculating quality factor: {e}")
            
        return np.zeros(self.lookback_period)
    
    def _calculate_volatility_factor(self, data) -> np.ndarray:
        """Calculate volatility factor returns."""
        try:
            # Volatility factor (high vol - low vol)
            vol_returns = []
            
            for symbol in data.keys():
                if symbol != 'SPY':
                    prices = self._get_price_series(symbol, data)
                    if prices is not None and len(prices) > 20:
                        returns = np.diff(np.log(prices))
                        volatility = np.std(returns[-20:])
                        
                        if len(returns) > 0:
                            vol_returns.append((returns, volatility))
            
            if vol_returns:
                # Sort by volatility and take long-short
                vol_returns.sort(key=lambda x: x[1])
                n = len(vol_returns)
                
                high_vol = np.mean([ret[0] for ret in vol_returns[-n//3:]], axis=0)
                low_vol = np.mean([ret[0] for ret in vol_returns[:n//3]], axis=0)
                
                return high_vol - low_vol
                
        except Exception as e:
            self.logger.debug(f"Error calculating volatility factor: {e}")
            
        return np.zeros(self.lookback_period)
    
    def _calculate_factor_loadings(self, data) -> Dict[str, Dict[str, float]]:
        """Calculate factor loadings for each stock."""
        factor_loadings = {}
        
        try:
            for symbol in data.keys():
                if symbol != 'SPY':
                    prices = self._get_price_series(symbol, data)
                    if prices is not None and len(prices) > 1:
                        returns = np.diff(np.log(prices))
                        
                        # Calculate factor loadings using regression
                        loadings = self._regress_factors(returns)
                        factor_loadings[symbol] = loadings
                        
        except Exception as e:
            self.logger.error(f"Error calculating factor loadings: {e}")
            
        return factor_loadings
    
    def _regress_factors(self, returns: np.ndarray) -> Dict[str, float]:
        """Regress stock returns on factors to get loadings."""
        loadings = {}
        
        try:
            # Get factor returns
            factor_returns = self._calculate_factor_returns({})
            
            if len(returns) > 0 and factor_returns:
                # Align lengths
                min_length = min(len(returns), min(len(factor_returns[f]) for f in factor_returns))
                
                if min_length > 10:
                    y = returns[-min_length:]
                    
                    for factor_name, factor_ret in factor_returns.items():
                        if len(factor_ret) >= min_length:
                            x = factor_ret[-min_length:]
                            
                            # Simple regression
                            correlation = np.corrcoef(y, x)[0, 1]
                            if not np.isnan(correlation):
                                loadings[factor_name] = correlation
                            else:
                                loadings[factor_name] = 0.0
                        else:
                            loadings[factor_name] = 0.0
                            
        except Exception as e:
            self.logger.debug(f"Error in factor regression: {e}")
            
        return loadings
    
    def _generate_factor_signals(self, factor_returns: Dict[str, np.ndarray], 
                               factor_loadings: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Generate signals based on factor exposures."""
        signals = {}
        
        try:
            for symbol, loadings in factor_loadings.items():
                signal = 0.0
                
                for factor_name, loading in loadings.items():
                    if factor_name in factor_returns and len(factor_returns[factor_name]) > 0:
                        # Recent factor performance
                        recent_return = factor_returns[factor_name][-1] if len(factor_returns[factor_name]) > 0 else 0
                        
                        # Signal = loading * recent factor return
                        signal += loading * recent_return
                
                # Normalize signal
                if abs(signal) > 0.01:  # Minimum threshold
                    signals[symbol] = np.clip(signal, -1.0, 1.0)
                    
        except Exception as e:
            self.logger.error(f"Error generating factor signals: {e}")
            
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
    
    def get_factor_exposures(self) -> Dict[str, Dict[str, float]]:
        """Get current factor exposures for all stocks."""
        return self.factor_loadings
    
    def get_factor_performance(self) -> Dict[str, float]:
        """Get recent factor performance."""
        performance = {}
        
        try:
            factor_returns = self._calculate_factor_returns({})
            
            for factor_name, returns in factor_returns.items():
                if len(returns) > 0:
                    # Recent performance (last 20 days)
                    recent_returns = returns[-20:] if len(returns) >= 20 else returns
                    performance[factor_name] = np.mean(recent_returns)
                    
        except Exception as e:
            self.logger.error(f"Error calculating factor performance: {e}")
            
        return performance
