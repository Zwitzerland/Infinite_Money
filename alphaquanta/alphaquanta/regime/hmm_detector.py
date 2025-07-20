"""
Hidden Markov Model regime detection for quantum-hybrid trading system.
Implements Baum-Welch training and Viterbi decoding for market regime identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import asyncio
import time
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)


class HiddenMarkovModel:
    """Hidden Markov Model for market regime detection."""
    
    def __init__(self, n_states: int = 3, n_features: int = 5, random_state: int = 42):
        self.n_states = n_states
        self.n_features = n_features
        self.random_state = random_state
        
        self.transition_matrix = None  # A[i,j] = P(s_t+1 = j | s_t = i)
        self.emission_means = None     # μ[i] = mean of observations in state i
        self.emission_covs = None      # Σ[i] = covariance of observations in state i
        self.initial_probs = None      # π[i] = P(s_0 = i)
        
        self.is_fitted = False
        self.log_likelihood_history = []
        self.convergence_threshold = 1e-6
        self.max_iterations = 100
        
        self.regime_labels = {
            0: "low_volatility",
            1: "normal_market", 
            2: "high_volatility"
        }
        
        np.random.seed(random_state)
        logger.info(f"HMM initialized with {n_states} states, {n_features} features")
    
    def _initialize_parameters(self, observations: np.ndarray):
        """Initialize HMM parameters using K-means clustering."""
        from sklearn.cluster import KMeans
        
        n_obs, n_features = observations.shape
        
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(observations)
        
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.transition_matrix += np.random.normal(0, 0.01, (self.n_states, self.n_states))
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        self.emission_means = np.zeros((self.n_states, n_features))
        self.emission_covs = np.zeros((self.n_states, n_features, n_features))
        
        for state in range(self.n_states):
            state_obs = observations[cluster_labels == state]
            if len(state_obs) > 0:
                self.emission_means[state] = np.mean(state_obs, axis=0)
                self.emission_covs[state] = np.cov(state_obs.T) + np.eye(n_features) * 1e-6
            else:
                self.emission_means[state] = np.random.normal(0, 1, n_features)
                self.emission_covs[state] = np.eye(n_features)
        
        unique, counts = np.unique(cluster_labels, return_counts=True)
        self.initial_probs = np.ones(self.n_states) / self.n_states
        for i, count in zip(unique, counts):
            self.initial_probs[i] = count / len(cluster_labels)
        
        logger.info("HMM parameters initialized using K-means clustering")
    
    def _compute_emission_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Compute emission probabilities for all states and observations."""
        n_obs = len(observations)
        emission_probs = np.zeros((n_obs, self.n_states))
        
        for state in range(self.n_states):
            try:
                rv = multivariate_normal(
                    mean=self.emission_means[state],
                    cov=self.emission_covs[state],
                    allow_singular=True
                )
                emission_probs[:, state] = rv.pdf(observations)
            except Exception as e:
                logger.warning(f"Emission probability computation failed for state {state}: {e}")
                emission_probs[:, state] = 1.0 / self.n_states
        
        emission_probs = np.maximum(emission_probs, 1e-10)
        
        return emission_probs
    
    def _forward_algorithm(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm for computing forward probabilities."""
        n_obs = len(observations)
        emission_probs = self._compute_emission_probabilities(observations)
        
        alpha = np.zeros((n_obs, self.n_states))
        alpha[0] = self.initial_probs * emission_probs[0]
        
        for t in range(1, n_obs):
            for j in range(self.n_states):
                alpha[t, j] = emission_probs[t, j] * np.sum(
                    alpha[t-1] * self.transition_matrix[:, j]
                )
        
        log_likelihood = np.log(np.sum(alpha[-1]))
        
        return alpha, log_likelihood
    
    def _backward_algorithm(self, observations: np.ndarray) -> np.ndarray:
        """Backward algorithm for computing backward probabilities."""
        n_obs = len(observations)
        emission_probs = self._compute_emission_probabilities(observations)
        
        beta = np.zeros((n_obs, self.n_states))
        beta[-1] = 1.0
        
        for t in range(n_obs - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i] * emission_probs[t+1] * beta[t+1]
                )
        
        return beta
    
    def _baum_welch_step(self, observations: np.ndarray) -> float:
        """Single step of Baum-Welch algorithm."""
        n_obs = len(observations)
        emission_probs = self._compute_emission_probabilities(observations)
        
        alpha, log_likelihood = self._forward_algorithm(observations)
        beta = self._backward_algorithm(observations)
        
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        xi = np.zeros((n_obs - 1, self.n_states, self.n_states))
        for t in range(n_obs - 1):
            denominator = np.sum(alpha[t] * beta[t])
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.transition_matrix[i, j] * 
                                  emission_probs[t+1, j] * beta[t+1, j]) / denominator
        
        self.initial_probs = gamma[0]
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        
        for state in range(self.n_states):
            weights = gamma[:, state]
            total_weight = np.sum(weights)
            
            if total_weight > 1e-10:
                self.emission_means[state] = np.average(observations, axis=0, weights=weights)
                
                diff = observations - self.emission_means[state]
                self.emission_covs[state] = np.average(
                    np.array([np.outer(d, d) for d in diff]), 
                    axis=0, 
                    weights=weights
                ) + np.eye(self.n_features) * 1e-6
        
        return log_likelihood
    
    def fit(self, observations: np.ndarray) -> Dict[str, Any]:
        """Fit HMM using Baum-Welch algorithm."""
        start_time = time.time()
        
        if len(observations.shape) != 2:
            raise ValueError("Observations must be 2D array (n_samples, n_features)")
        
        n_obs, n_features = observations.shape
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")
        
        logger.info(f"Training HMM on {n_obs} observations")
        
        self._initialize_parameters(observations)
        
        self.log_likelihood_history = []
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            log_likelihood = self._baum_welch_step(observations)
            self.log_likelihood_history.append(log_likelihood)
            
            improvement = log_likelihood - prev_log_likelihood
            if iteration > 0 and improvement < self.convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = log_likelihood
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.4f}")
        
        self.is_fitted = True
        training_time = time.time() - start_time
        
        final_log_likelihood = self.log_likelihood_history[-1]
        logger.info(f"HMM training completed in {training_time:.2f}s, "
                   f"final log-likelihood: {final_log_likelihood:.4f}")
        
        return {
            'final_log_likelihood': final_log_likelihood,
            'iterations': len(self.log_likelihood_history),
            'training_time': training_time,
            'converged': improvement < self.convergence_threshold
        }
    
    def viterbi_decode(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Viterbi algorithm for finding most likely state sequence."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
        
        n_obs = len(observations)
        emission_probs = self._compute_emission_probabilities(observations)
        
        viterbi_prob = np.zeros((n_obs, self.n_states))
        viterbi_path = np.zeros((n_obs, self.n_states), dtype=int)
        
        viterbi_prob[0] = np.log(self.initial_probs) + np.log(emission_probs[0])
        
        for t in range(1, n_obs):
            for j in range(self.n_states):
                transition_scores = viterbi_prob[t-1] + np.log(self.transition_matrix[:, j])
                viterbi_path[t, j] = np.argmax(transition_scores)
                viterbi_prob[t, j] = np.max(transition_scores) + np.log(emission_probs[t, j])
        
        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(viterbi_prob[-1])
        
        for t in range(n_obs - 2, -1, -1):
            states[t] = viterbi_path[t + 1, states[t + 1]]
        
        sequence_prob = np.max(viterbi_prob[-1])
        
        return states, sequence_prob
    
    def predict_state_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Predict state probabilities for new observations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        alpha, _ = self._forward_algorithm(observations)
        beta = self._backward_algorithm(observations)
        
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        return gamma
    
    def get_regime_info(self, state_probs: np.ndarray) -> Dict[str, Any]:
        """Get regime information from state probabilities."""
        current_state = np.argmax(state_probs[-1])
        current_regime = self.regime_labels.get(current_state, f"state_{current_state}")
        
        states = np.argmax(state_probs, axis=1)
        regime_changes = np.where(np.diff(states) != 0)[0]
        
        if len(regime_changes) > 0:
            stability = len(states) - regime_changes[-1] - 1
        else:
            stability = len(states)
        
        confidence = state_probs[-1, current_state]
        
        return {
            'current_regime': current_regime,
            'current_state': int(current_state),
            'confidence': float(confidence),
            'stability': int(stability),
            'state_probabilities': state_probs[-1].tolist(),
            'regime_labels': self.regime_labels
        }


class MarketRegimeDetector:
    """Market regime detector using Hidden Markov Models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hmm_config = config.get('regime_detection', {})
        
        self.n_states = self.hmm_config.get('n_states', 3)
        self.lookback_days = self.hmm_config.get('lookback_days', 252)
        self.retrain_frequency = self.hmm_config.get('retrain_frequency', 'daily')
        self.min_observations = self.hmm_config.get('min_observations', 50)
        
        self.feature_config = self.hmm_config.get('features', {})
        self.use_returns = self.feature_config.get('use_returns', True)
        self.use_volatility = self.feature_config.get('use_volatility', True)
        self.use_volume = self.feature_config.get('use_volume', True)
        self.use_momentum = self.feature_config.get('use_momentum', True)
        self.use_mean_reversion = self.feature_config.get('use_mean_reversion', True)
        
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.last_training_date = None
        self.current_regime = None
        
        self.regime_history = []
        self.training_metrics = {}
        
        logger.info(f"Market regime detector initialized with {self.n_states} states")
    
    def extract_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Extract features for HMM from price data."""
        features = []
        
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        prices = price_data['close'].values
        
        if self.use_returns:
            returns = np.diff(np.log(prices))
            returns = np.pad(returns, (1, 0), mode='constant', constant_values=0)
            features.append(returns)
        
        if self.use_volatility:
            window = min(20, len(prices) // 4)
            if window > 1:
                volatility = pd.Series(prices).pct_change().rolling(window).std().fillna(0).values
            else:
                volatility = np.zeros(len(prices))
            features.append(volatility)
        
        if self.use_volume and 'volume' in price_data.columns:
            volume = price_data['volume'].values
            volume_ma = pd.Series(volume).rolling(20, min_periods=1).mean().values
            volume_ratio = volume / (volume_ma + 1e-10)
            features.append(np.log1p(volume_ratio))
        
        if self.use_momentum:
            window = min(20, len(prices) // 4)
            if window > 1:
                momentum = (prices / pd.Series(prices).rolling(window, min_periods=1).mean().values) - 1
            else:
                momentum = np.zeros(len(prices))
            features.append(momentum)
        
        if self.use_mean_reversion:
            window = min(10, len(prices) // 8)
            if window > 1:
                sma = pd.Series(prices).rolling(window, min_periods=1).mean().values
                mean_reversion = (prices - sma) / (sma + 1e-10)
            else:
                mean_reversion = np.zeros(len(prices))
            features.append(mean_reversion)
        
        feature_matrix = np.column_stack(features)
        
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_matrix
    
    async def train_model(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Train HMM model on historical price data."""
        start_time = time.time()
        
        if len(price_data) < self.min_observations:
            raise ValueError(f"Insufficient data: need at least {self.min_observations} observations")
        
        logger.info(f"Training HMM model on {len(price_data)} observations")
        
        features = self.extract_features(price_data)
        
        features_scaled = self.scaler.fit_transform(features)
        
        n_features = features_scaled.shape[1]
        self.hmm_model = HiddenMarkovModel(
            n_states=self.n_states,
            n_features=n_features,
            random_state=42
        )
        
        training_result = self.hmm_model.fit(features_scaled)
        
        self.last_training_date = datetime.now()
        self.training_metrics = training_result
        
        training_time = time.time() - start_time
        logger.info(f"HMM training completed in {training_time:.2f}s")
        
        return {
            'training_result': training_result,
            'n_features': n_features,
            'n_observations': len(price_data),
            'training_time': training_time,
            'model_params': {
                'n_states': self.n_states,
                'transition_matrix': self.hmm_model.transition_matrix.tolist(),
                'regime_labels': self.hmm_model.regime_labels
            }
        }
    
    async def detect_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime."""
        if self.hmm_model is None or not self.hmm_model.is_fitted:
            raise ValueError("Model must be trained before regime detection")
        
        start_time = time.time()
        
        features = self.extract_features(price_data)
        features_scaled = self.scaler.transform(features)
        
        state_probs = self.hmm_model.predict_state_probabilities(features_scaled)
        
        regime_info = self.hmm_model.get_regime_info(state_probs)
        
        self.current_regime = regime_info
        
        self.regime_history.append({
            'timestamp': datetime.now().isoformat(),
            'regime_info': regime_info,
            'detection_time': time.time() - start_time
        })
        
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        detection_time = time.time() - start_time
        
        return {
            'regime_info': regime_info,
            'detection_time': detection_time,
            'model_confidence': regime_info['confidence'],
            'regime_stability': regime_info['stability']
        }
    
    def get_regime_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust portfolio weights based on current regime."""
        if self.current_regime is None:
            return base_weights
        
        current_state = self.current_regime['current_state']
        confidence = self.current_regime['confidence']
        
        regime_adjustments = {
            0: 0.8,   # Low volatility: reduce position sizes
            1: 1.0,   # Normal market: no adjustment
            2: 0.6    # High volatility: significantly reduce positions
        }
        
        adjustment_factor = regime_adjustments.get(current_state, 1.0)
        
        final_adjustment = 1.0 + (adjustment_factor - 1.0) * confidence
        
        adjusted_weights = {}
        for symbol, weight in base_weights.items():
            adjusted_weights[symbol] = weight * final_adjustment
        
        return adjusted_weights
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self.last_training_date is None:
            return True
        
        if self.retrain_frequency == 'daily':
            return (datetime.now() - self.last_training_date).days >= 1
        elif self.retrain_frequency == 'weekly':
            return (datetime.now() - self.last_training_date).days >= 7
        elif self.retrain_frequency == 'monthly':
            return (datetime.now() - self.last_training_date).days >= 30
        
        return False
    
    def get_prometheus_metrics(self) -> Dict[str, float]:
        """Get metrics for Prometheus monitoring."""
        if self.current_regime is None:
            return {}
        
        metrics = {}
        
        for state, prob in enumerate(self.current_regime['state_probabilities']):
            metrics[f'hmm_state_prob_state_{state}'] = prob
        
        metrics['hmm_current_state'] = float(self.current_regime['current_state'])
        metrics['hmm_confidence'] = self.current_regime['confidence']
        metrics['hmm_stability'] = float(self.current_regime['stability'])
        
        if self.training_metrics:
            metrics['hmm_log_likelihood'] = self.training_metrics.get('final_log_likelihood', 0)
            metrics['hmm_training_iterations'] = float(self.training_metrics.get('iterations', 0))
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if self.hmm_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'config': self.config,
            'n_states': self.n_states,
            'transition_matrix': self.hmm_model.transition_matrix.tolist(),
            'emission_means': self.hmm_model.emission_means.tolist(),
            'emission_covs': self.hmm_model.emission_covs.tolist(),
            'initial_probs': self.hmm_model.initial_probs.tolist(),
            'regime_labels': self.hmm_model.regime_labels,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
            'training_metrics': self.training_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"HMM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.n_states = model_data['n_states']
        
        n_features = len(model_data['emission_means'][0])
        self.hmm_model = HiddenMarkovModel(
            n_states=self.n_states,
            n_features=n_features
        )
        
        self.hmm_model.transition_matrix = np.array(model_data['transition_matrix'])
        self.hmm_model.emission_means = np.array(model_data['emission_means'])
        self.hmm_model.emission_covs = np.array(model_data['emission_covs'])
        self.hmm_model.initial_probs = np.array(model_data['initial_probs'])
        self.hmm_model.regime_labels = model_data['regime_labels']
        self.hmm_model.is_fitted = True
        
        self.scaler.mean_ = np.array(model_data['scaler_mean'])
        self.scaler.scale_ = np.array(model_data['scaler_scale'])
        
        if model_data['last_training_date']:
            self.last_training_date = datetime.fromisoformat(model_data['last_training_date'])
        self.training_metrics = model_data.get('training_metrics', {})
        
        logger.info(f"HMM model loaded from {filepath}")
