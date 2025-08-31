"""
Regime Detection Module

Implements market regime detection algorithms:
- Hidden Markov Models
- Volatility regimes
- Trend regimes
- Correlation regimes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging


class RegimeDetector:
    """
    Market regime detection using various methods.
    
    Detects different market regimes:
    - Bull/Bear markets
    - High/Low volatility periods
    - Trending/Sideways markets
    - Crisis/Normal periods
    """
    
    def __init__(self, lookback_period: int = 252, n_regimes: int = 3):
        """
        Initialize the regime detector.
        
        Args:
            lookback_period: Number of days for lookback window
            n_regimes: Number of regimes to detect
        """
        self.lookback_period = lookback_period
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.regime_history = []
        self.current_regime = "normal"
        self.logger = logging.getLogger(__name__)
        
    def detect_regime(self, data) -> str:
        """
        Detect current market regime.
        
        Args:
            data: Market data
            
        Returns:
            String indicating current regime
        """
        try:
            # Extract market features
            features = self._extract_features(data)
            
            if features is not None and len(features) > 0:
                # Detect regime using multiple methods
                volatility_regime = self._detect_volatility_regime(features)
                trend_regime = self._detect_trend_regime(features)
                correlation_regime = self._detect_correlation_regime(data)
                
                # Combine regime signals
                combined_regime = self._combine_regimes(volatility_regime, trend_regime, correlation_regime)
                
                # Update regime history
                self.regime_history.append(combined_regime)
                self.current_regime = combined_regime
                
                return combined_regime
                
        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            
        return self.current_regime
    
    def _extract_features(self, data) -> Optional[np.ndarray]:
        """Extract features for regime detection."""
        try:
            features = []
            
            # Get market data (use SPY as proxy)
            if 'SPY' in data:
                spy_prices = self._get_price_series('SPY', data)
                if spy_prices is not None and len(spy_prices) > self.lookback_period:
                    returns = np.diff(np.log(spy_prices))
                    
                    # Volatility features
                    vol_20 = np.std(returns[-20:])
                    vol_60 = np.std(returns[-60:])
                    vol_252 = np.std(returns[-252:]) if len(returns) >= 252 else np.std(returns)
                    
                    # Return features
                    ret_20 = np.mean(returns[-20:])
                    ret_60 = np.mean(returns[-60:])
                    ret_252 = np.mean(returns[-252:]) if len(returns) >= 252 else np.mean(returns)
                    
                    # Drawdown features
                    cumulative_returns = np.cumprod(1 + returns)
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown = np.min(drawdown[-252:]) if len(drawdown) >= 252 else np.min(drawdown)
                    
                    # Momentum features
                    momentum_20 = (spy_prices[-1] - spy_prices[-20]) / spy_prices[-20]
                    momentum_60 = (spy_prices[-1] - spy_prices[-60]) / spy_prices[-60]
                    
                    # Skewness and kurtosis
                    skewness = stats.skew(returns[-252:]) if len(returns) >= 252 else stats.skew(returns)
                    kurtosis = stats.kurtosis(returns[-252:]) if len(returns) >= 252 else stats.kurtosis(returns)
                    
                    features = [
                        vol_20, vol_60, vol_252,
                        ret_20, ret_60, ret_252,
                        max_drawdown,
                        momentum_20, momentum_60,
                        skewness, kurtosis
                    ]
                    
                    return np.array(features)
                    
        except Exception as e:
            self.logger.debug(f"Error extracting features: {e}")
            
        return None
    
    def _detect_volatility_regime(self, features: np.ndarray) -> str:
        """Detect volatility-based regime."""
        try:
            vol_20 = features[0]
            vol_60 = features[1]
            vol_252 = features[2]
            
            # Compare current volatility to historical
            vol_ratio_20 = vol_20 / vol_252
            vol_ratio_60 = vol_60 / vol_252
            
            if vol_ratio_20 > 1.5 or vol_ratio_60 > 1.3:
                return "high_volatility"
            elif vol_ratio_20 < 0.7 or vol_ratio_60 < 0.8:
                return "low_volatility"
            else:
                return "normal_volatility"
                
        except Exception as e:
            self.logger.debug(f"Error detecting volatility regime: {e}")
            return "normal_volatility"
    
    def _detect_trend_regime(self, features: np.ndarray) -> str:
        """Detect trend-based regime."""
        try:
            ret_20 = features[3]
            ret_60 = features[4]
            ret_252 = features[5]
            momentum_20 = features[8]
            momentum_60 = features[9]
            
            # Strong uptrend
            if (ret_20 > 0.02 and ret_60 > 0.01 and 
                momentum_20 > 0.05 and momentum_60 > 0.1):
                return "strong_uptrend"
            
            # Strong downtrend
            elif (ret_20 < -0.02 and ret_60 < -0.01 and 
                  momentum_20 < -0.05 and momentum_60 < -0.1):
                return "strong_downtrend"
            
            # Weak uptrend
            elif ret_20 > 0.005 and ret_60 > 0.002:
                return "weak_uptrend"
            
            # Weak downtrend
            elif ret_20 < -0.005 and ret_60 < -0.002:
                return "weak_downtrend"
            
            # Sideways
            else:
                return "sideways"
                
        except Exception as e:
            self.logger.debug(f"Error detecting trend regime: {e}")
            return "sideways"
    
    def _detect_correlation_regime(self, data) -> str:
        """Detect correlation-based regime."""
        try:
            # Calculate correlation matrix for stocks
            returns_matrix = []
            symbols = list(data.keys())
            
            for symbol in symbols[:20]:  # Use top 20 stocks
                if symbol != 'SPY':
                    prices = self._get_price_series(symbol, data)
                    if prices is not None and len(prices) > 60:
                        returns = np.diff(np.log(prices))
                        if len(returns) >= 60:
                            returns_matrix.append(returns[-60:])
            
            if len(returns_matrix) > 5:
                returns_df = pd.DataFrame(returns_matrix).T
                correlation_matrix = returns_df.corr()
                
                # Calculate average correlation
                avg_correlation = np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)])
                
                if avg_correlation > 0.7:
                    return "high_correlation"
                elif avg_correlation < 0.3:
                    return "low_correlation"
                else:
                    return "normal_correlation"
                    
        except Exception as e:
            self.logger.debug(f"Error detecting correlation regime: {e}")
            
        return "normal_correlation"
    
    def _combine_regimes(self, vol_regime: str, trend_regime: str, corr_regime: str) -> str:
        """Combine different regime signals into final regime."""
        try:
            # Crisis regime
            if (vol_regime == "high_volatility" and 
                (trend_regime == "strong_downtrend" or trend_regime == "weak_downtrend") and
                corr_regime == "high_correlation"):
                return "crisis"
            
            # Bull market
            elif (vol_regime in ["normal_volatility", "low_volatility"] and
                  trend_regime in ["strong_uptrend", "weak_uptrend"] and
                  corr_regime in ["normal_correlation", "low_correlation"]):
                return "bull_market"
            
            # Bear market
            elif (vol_regime == "high_volatility" and
                  trend_regime in ["strong_downtrend", "weak_downtrend"]):
                return "bear_market"
            
            # Sideways market
            elif (vol_regime in ["normal_volatility", "low_volatility"] and
                  trend_regime == "sideways"):
                return "sideways"
            
            # High volatility period
            elif vol_regime == "high_volatility":
                return "high_volatility"
            
            # Default
            else:
                return "normal"
                
        except Exception as e:
            self.logger.debug(f"Error combining regimes: {e}")
            return "normal"
    
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
    
    def get_regime_probabilities(self) -> Dict[str, float]:
        """Get probabilities for each regime."""
        if len(self.regime_history) == 0:
            return {"normal": 1.0}
        
        # Count regime occurrences
        regime_counts = {}
        total = len(self.regime_history)
        
        for regime in self.regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calculate probabilities
        probabilities = {}
        for regime, count in regime_counts.items():
            probabilities[regime] = count / total
        
        return probabilities
    
    def get_regime_transitions(self) -> Dict[str, Dict[str, int]]:
        """Get regime transition matrix."""
        if len(self.regime_history) < 2:
            return {}
        
        transitions = {}
        
        for i in range(len(self.regime_history) - 1):
            current_regime = self.regime_history[i]
            next_regime = self.regime_history[i + 1]
            
            if current_regime not in transitions:
                transitions[current_regime] = {}
            
            if next_regime not in transitions[current_regime]:
                transitions[current_regime][next_regime] = 0
            
            transitions[current_regime][next_regime] += 1
        
        return transitions
    
    def get_regime_duration(self) -> Dict[str, float]:
        """Get average duration of each regime."""
        if len(self.regime_history) == 0:
            return {}
        
        durations = {}
        current_regime = self.regime_history[0]
        current_duration = 1
        regime_durations = []
        
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append((current_regime, current_duration))
                current_regime = self.regime_history[i]
                current_duration = 1
        
        # Add last regime
        regime_durations.append((current_regime, current_duration))
        
        # Calculate average durations
        regime_avg_durations = {}
        for regime, duration in regime_durations:
            if regime not in regime_avg_durations:
                regime_avg_durations[regime] = []
            regime_avg_durations[regime].append(duration)
        
        # Calculate averages
        avg_durations = {}
        for regime, durations_list in regime_avg_durations.items():
            avg_durations[regime] = np.mean(durations_list)
        
        return avg_durations
