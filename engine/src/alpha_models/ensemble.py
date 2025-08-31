"""
Ensemble Models Module

Implements ensemble methods for combining alpha signals:
- Weighted averaging
- Machine learning ensembles
- Dynamic weighting
- Regime-based weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import logging


class EnsembleModel:
    """
    Ensemble model for combining multiple alpha signals.
    
    Implements various ensemble methods:
    - Simple weighted averaging
    - Machine learning-based combination
    - Dynamic weighting based on performance
    - Regime-based weighting
    """
    
    def __init__(self, lookback_period: int = 252):
        """
        Initialize the ensemble model.
        
        Args:
            lookback_period: Number of days for lookback window
        """
        self.lookback_period = lookback_period
        self.model_performance = {}
        self.weights = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ensemble methods
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ensemble models."""
        try:
            # Simple ensemble models
            self.models['linear'] = LinearRegression()
            self.models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Initialize weights
            self.weights = {
                'stat_arb': 0.25,
                'factor': 0.25,
                'ensemble': 0.25,
                'sentiment': 0.15,
                'insider': 0.10
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing ensemble models: {e}")
    
    def generate_signals(self, data, regime: str = "normal") -> Dict[str, float]:
        """
        Generate ensemble alpha signals.
        
        Args:
            data: Market data
            regime: Current market regime
            
        Returns:
            Dictionary mapping symbols to signal strengths
        """
        signals = {}
        
        try:
            # Get individual model signals
            model_signals = self._get_model_signals(data)
            
            # Combine signals using ensemble methods
            signals = self._combine_signals(model_signals, regime)
            
            # Update model performance
            self._update_performance(model_signals)
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble signals: {e}")
            
        return signals
    
    def _get_model_signals(self, data) -> Dict[str, Dict[str, float]]:
        """Get signals from individual models."""
        model_signals = {}
        
        try:
            # This would typically get signals from other alpha models
            # For now, we'll create placeholder signals
            symbols = list(data.keys()) if hasattr(data, 'keys') else []
            
            # Placeholder signals for demonstration
            for symbol in symbols[:10]:  # Limit to 10 symbols
                model_signals['stat_arb'] = {symbol: np.random.normal(0, 0.1)}
                model_signals['factor'] = {symbol: np.random.normal(0, 0.1)}
                model_signals['sentiment'] = {symbol: np.random.normal(0, 0.05)}
                model_signals['insider'] = {symbol: np.random.normal(0, 0.03)}
                
        except Exception as e:
            self.logger.debug(f"Error getting model signals: {e}")
            
        return model_signals
    
    def _combine_signals(self, model_signals: Dict[str, Dict[str, float]], 
                        regime: str) -> Dict[str, float]:
        """Combine signals using ensemble methods."""
        combined_signals = {}
        
        try:
            # Get regime-adjusted weights
            regime_weights = self._get_regime_weights(regime)
            
            # Get all unique symbols
            all_symbols = set()
            for model_name, signals in model_signals.items():
                all_symbols.update(signals.keys())
            
            # Combine signals for each symbol
            for symbol in all_symbols:
                signal = 0.0
                total_weight = 0.0
                
                for model_name, signals in model_signals.items():
                    if symbol in signals:
                        weight = regime_weights.get(model_name, self.weights.get(model_name, 0.0))
                        signal += weight * signals[symbol]
                        total_weight += weight
                
                # Normalize by total weight
                if total_weight > 0:
                    combined_signals[symbol] = signal / total_weight
                else:
                    combined_signals[symbol] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error combining signals: {e}")
            
        return combined_signals
    
    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get regime-adjusted weights for different models."""
        base_weights = self.weights.copy()
        
        try:
            if regime == "crisis":
                # In crisis, favor defensive models
                base_weights['stat_arb'] *= 1.5
                base_weights['factor'] *= 0.8
                base_weights['sentiment'] *= 0.5
                base_weights['insider'] *= 1.2
                
            elif regime == "bull_market":
                # In bull market, favor momentum models
                base_weights['factor'] *= 1.3
                base_weights['sentiment'] *= 1.2
                base_weights['stat_arb'] *= 0.8
                
            elif regime == "bear_market":
                # In bear market, favor defensive models
                base_weights['stat_arb'] *= 1.4
                base_weights['factor'] *= 0.7
                base_weights['sentiment'] *= 0.6
                
            elif regime == "high_volatility":
                # In high volatility, reduce all weights
                for key in base_weights:
                    base_weights[key] *= 0.8
                    
        except Exception as e:
            self.logger.debug(f"Error adjusting regime weights: {e}")
            
        return base_weights
    
    def _update_performance(self, model_signals: Dict[str, Dict[str, float]]):
        """Update model performance tracking."""
        try:
            # This would typically track actual performance
            # For now, we'll use placeholder logic
            for model_name in model_signals.keys():
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = []
                
                # Placeholder performance metric
                performance = np.random.normal(0.5, 0.1)
                self.model_performance[model_name].append(performance)
                
                # Keep only recent performance
                if len(self.model_performance[model_name]) > self.lookback_period:
                    self.model_performance[model_name] = self.model_performance[model_name][-self.lookback_period:]
                    
        except Exception as e:
            self.logger.debug(f"Error updating performance: {e}")
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights."""
        try:
            # Validate weights
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                # Normalize weights
                for key in new_weights:
                    new_weights[key] /= total_weight
            
            self.weights.update(new_weights)
            
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}")
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get recent performance of each model."""
        performance = {}
        
        try:
            for model_name, perf_history in self.model_performance.items():
                if len(perf_history) > 0:
                    performance[model_name] = np.mean(perf_history[-20:])  # Last 20 periods
                else:
                    performance[model_name] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            
        return performance
    
    def optimize_weights(self, historical_signals: Dict[str, Dict[str, float]], 
                        historical_returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Optimize ensemble weights using historical data."""
        try:
            # Prepare training data
            X = []
            y = []
            
            for symbol in historical_returns.keys():
                if symbol in historical_signals:
                    # Get signal features
                    features = []
                    for model_name, signals in historical_signals.items():
                        if symbol in signals:
                            features.append(signals[symbol])
                        else:
                            features.append(0.0)
                    
                    X.append(features)
                    
                    # Get target (future returns)
                    if len(historical_returns[symbol]) > 1:
                        future_return = historical_returns[symbol][1]  # Next period return
                        y.append(future_return)
            
            if len(X) > 10 and len(y) > 10:
                X = np.array(X)
                y = np.array(y)
                
                # Train ensemble model
                self.models['linear'].fit(X, y)
                
                # Extract weights from linear model
                weights = self.models['linear'].coef_
                
                # Normalize weights
                weights = np.abs(weights)
                weights = weights / np.sum(weights)
                
                # Update weights
                model_names = list(historical_signals.keys())
                new_weights = {}
                for i, model_name in enumerate(model_names):
                    if i < len(weights):
                        new_weights[model_name] = weights[i]
                
                self.update_weights(new_weights)
                
        except Exception as e:
            self.logger.error(f"Error optimizing weights: {e}")
            
        return self.weights
    
    def get_ensemble_statistics(self) -> Dict[str, float]:
        """Get ensemble statistics."""
        stats = {}
        
        try:
            # Weight statistics
            stats['weight_entropy'] = self._calculate_weight_entropy()
            stats['weight_concentration'] = self._calculate_weight_concentration()
            
            # Performance statistics
            performance = self.get_model_performance()
            if performance:
                stats['avg_performance'] = np.mean(list(performance.values()))
                stats['performance_std'] = np.std(list(performance.values()))
                stats['best_model'] = max(performance, key=performance.get)
                stats['worst_model'] = min(performance, key=performance.get)
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble statistics: {e}")
            
        return stats
    
    def _calculate_weight_entropy(self) -> float:
        """Calculate entropy of current weights."""
        try:
            weights = list(self.weights.values())
            weights = [w for w in weights if w > 0]
            
            if len(weights) == 0:
                return 0.0
            
            entropy = -np.sum(weights * np.log(weights))
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_weight_concentration(self) -> float:
        """Calculate concentration of weights (Herfindahl index)."""
        try:
            weights = list(self.weights.values())
            concentration = np.sum(np.array(weights) ** 2)
            return concentration
            
        except Exception:
            return 1.0
