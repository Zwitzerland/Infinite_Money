"""
Bayesian meta-learner for hyperparameter optimization of quantum-hybrid trading system.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import asyncio
from scipy.optimize import minimize
from scipy.stats import norm
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GaussianProcess:
    """Gaussian Process for Bayesian optimization."""
    
    def __init__(self, kernel_type: str = 'rbf', noise_level: float = 1e-6):
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.kernel_params = {'length_scale': 1.0, 'signal_variance': 1.0}
        
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Radial Basis Function kernel."""
        length_scale = self.kernel_params['length_scale']
        signal_variance = self.kernel_params['signal_variance']
        
        X1_expanded = X1[:, np.newaxis, :]
        X2_expanded = X2[np.newaxis, :, :]
        sq_dists = np.sum((X1_expanded - X2_expanded) ** 2, axis=2)
        
        K = signal_variance * np.exp(-0.5 * sq_dists / (length_scale ** 2))
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Process to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        self._optimize_hyperparameters()
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at test points."""
        if self.X_train is None:
            raise ValueError("Model not fitted yet")
        
        K_train = self.rbf_kernel(self.X_train, self.X_train)
        K_train += self.noise_level * np.eye(len(self.X_train))
        K_test = self.rbf_kernel(self.X_train, X_test)
        K_test_test = self.rbf_kernel(X_test, X_test)
        
        try:
            L = np.linalg.cholesky(K_train)
        except np.linalg.LinAlgError:
            K_train += 1e-3 * np.eye(len(self.X_train))
            L = np.linalg.cholesky(K_train)
        
        alpha = np.linalg.solve(L, self.y_train)
        alpha = np.linalg.solve(L.T, alpha)
        
        mu = K_test.T @ alpha
        
        v = np.linalg.solve(L, K_test)
        var = np.diag(K_test_test) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)  # Ensure positive variance
        
        return mu, var
    
    def _optimize_hyperparameters(self):
        """Optimize kernel hyperparameters using maximum likelihood."""
        def neg_log_likelihood(params):
            length_scale, signal_variance = np.exp(params)
            self.kernel_params = {'length_scale': length_scale, 'signal_variance': signal_variance}
            
            K = self.rbf_kernel(self.X_train, self.X_train)
            K += self.noise_level * np.eye(len(self.X_train))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, self.y_train)
                
                log_likelihood = -0.5 * np.sum(alpha ** 2) - np.sum(np.log(np.diag(L))) - 0.5 * len(self.X_train) * np.log(2 * np.pi)
                return -log_likelihood
            except np.linalg.LinAlgError:
                return 1e10
        
        initial_params = np.log([self.kernel_params['length_scale'], self.kernel_params['signal_variance']])
        
        result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B')
        
        if result.success:
            length_scale, signal_variance = np.exp(result.x)
            self.kernel_params = {'length_scale': length_scale, 'signal_variance': signal_variance}


class AcquisitionFunction:
    """Acquisition functions for Bayesian optimization."""
    
    @staticmethod
    def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-10)
        improvement = mu - f_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei
    
    @staticmethod
    def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        return mu + beta * np.sqrt(sigma)
    
    @staticmethod
    def probability_of_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.01) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-10)
        improvement = mu - f_best - xi
        Z = improvement / sigma
        return norm.cdf(Z)


class BayesianOptimizer:
    """Bayesian optimizer for quantum-hybrid trading system hyperparameters."""
    
    def __init__(self, config: Dict[str, Any], qpu_tracker=None):
        self.config = config
        self.meta_config = config.get('meta_learning', {})
        self.qpu_tracker = qpu_tracker
        
        self.acquisition_function = self.meta_config.get('acquisition_function', 'expected_improvement')
        self.n_initial_points = self.meta_config.get('n_initial_points', 5)
        self.n_iterations = self.meta_config.get('n_iterations', 50)
        self.xi = self.meta_config.get('xi', 0.01)
        self.beta = self.meta_config.get('beta', 2.0)
        
        self.gp = GaussianProcess()
        
        self.parameter_space = self._define_parameter_space()
        self.parameter_bounds = self._get_parameter_bounds()
        
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        
        self.optimization_history = []
        
        logger.info(f"Bayesian optimizer initialized with {len(self.parameter_space)} parameters")
    
    def _define_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """Define the hyperparameter space for optimization."""
        return {
            'qaoa_depth': {'type': 'int', 'bounds': [1, 6], 'default': 3},
            'qaoa_shots': {'type': 'int', 'bounds': [512, 4096], 'default': 1024},
            'qaoa_optimizer': {'type': 'categorical', 'choices': ['COBYLA', 'SPSA', 'ADAM'], 'default': 'COBYLA'},
            'qaoa_warm_start': {'type': 'bool', 'default': True},
            
            'diffusion_timesteps': {'type': 'int', 'bounds': [50, 500], 'default': 100},
            'diffusion_noise_schedule': {'type': 'categorical', 'choices': ['linear', 'cosine', 'sigmoid'], 'default': 'cosine'},
            'diffusion_learning_rate': {'type': 'float', 'bounds': [1e-5, 1e-2], 'default': 1e-3},
            'diffusion_batch_size': {'type': 'int', 'bounds': [16, 128], 'default': 32},
            
            'var_confidence_levels': {'type': 'categorical', 'choices': [[0.95], [0.99], [0.95, 0.99]], 'default': [0.95, 0.99]},
            'var_lookback_days': {'type': 'int', 'bounds': [20, 500], 'default': 252},
            'var_num_qubits': {'type': 'int', 'bounds': [4, 12], 'default': 8},
            
            'rl_learning_rate': {'type': 'float', 'bounds': [1e-5, 1e-2], 'default': 3e-4},
            'rl_gamma': {'type': 'float', 'bounds': [0.9, 0.999], 'default': 0.99},
            'rl_clip_epsilon': {'type': 'float', 'bounds': [0.1, 0.3], 'default': 0.2},
            'rl_entropy_coef': {'type': 'float', 'bounds': [0.001, 0.1], 'default': 0.01},
            
            'risk_var_threshold': {'type': 'float', 'bounds': [0.01, 0.1], 'default': 0.05},
            'risk_concentration_limit': {'type': 'float', 'bounds': [0.1, 0.9], 'default': 0.3},
            'risk_max_drawdown': {'type': 'float', 'bounds': [0.05, 0.3], 'default': 0.15},
            
            'signal_confidence_threshold': {'type': 'float', 'bounds': [0.1, 0.9], 'default': 0.5},
            'signal_quantum_weight': {'type': 'float', 'bounds': [0.0, 1.0], 'default': 0.7},
            'signal_classical_weight': {'type': 'float', 'bounds': [0.0, 1.0], 'default': 0.3}
        }
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for param_name, param_config in self.parameter_space.items():
            if param_config['type'] in ['int', 'float']:
                bounds.append(tuple(param_config['bounds']))
            elif param_config['type'] == 'bool':
                bounds.append((0.0, 1.0))
            elif param_config['type'] == 'categorical':
                bounds.append((0.0, float(len(param_config['choices']) - 1)))
        return bounds
    
    def _encode_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameters to continuous space for GP."""
        encoded = []
        for param_name, param_config in self.parameter_space.items():
            value = params.get(param_name, param_config['default'])
            
            if param_config['type'] == 'int':
                bounds = param_config['bounds']
                normalized = (value - bounds[0]) / (bounds[1] - bounds[0])
                encoded.append(normalized)
            elif param_config['type'] == 'float':
                bounds = param_config['bounds']
                normalized = (value - bounds[0]) / (bounds[1] - bounds[0])
                encoded.append(normalized)
            elif param_config['type'] == 'bool':
                encoded.append(float(value))
            elif param_config['type'] == 'categorical':
                choices = param_config['choices']
                if value in choices:
                    index = choices.index(value)
                else:
                    index = 0
                encoded.append(float(index) / (len(choices) - 1))
        
        return np.array(encoded)
    
    def _decode_parameters(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode parameters from continuous space."""
        params = {}
        for i, (param_name, param_config) in enumerate(self.parameter_space.items()):
            value = encoded[i]
            
            if param_config['type'] == 'int':
                bounds = param_config['bounds']
                decoded = int(bounds[0] + value * (bounds[1] - bounds[0]))
                params[param_name] = np.clip(decoded, bounds[0], bounds[1])
            elif param_config['type'] == 'float':
                bounds = param_config['bounds']
                decoded = bounds[0] + value * (bounds[1] - bounds[0])
                params[param_name] = np.clip(decoded, bounds[0], bounds[1])
            elif param_config['type'] == 'bool':
                params[param_name] = value > 0.5
            elif param_config['type'] == 'categorical':
                choices = param_config['choices']
                index = int(np.round(value * (len(choices) - 1)))
                index = np.clip(index, 0, len(choices) - 1)
                params[param_name] = choices[index]
        
        return params
    
    async def optimize(self, objective_function: Callable[[Dict[str, Any]], float], 
                      max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Run Bayesian optimization to find best hyperparameters."""
        if max_iterations is None:
            max_iterations = self.n_iterations
        
        logger.info(f"Starting Bayesian optimization for {max_iterations} iterations")
        
        await self._initialize_random_points(objective_function)
        
        for iteration in range(max_iterations):
            start_time = time.time()
            
            if len(self.X_observed) > 0:
                X_array = np.array(self.X_observed)
                y_array = np.array(self.y_observed)
                self.gp.fit(X_array, y_array)
            
            next_point = self._find_next_point()
            next_params = self._decode_parameters(next_point)
            
            logger.info(f"Iteration {iteration + 1}: Evaluating {next_params}")
            score = await objective_function(next_params)
            
            self.X_observed.append(next_point)
            self.y_observed.append(score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = next_params.copy()
                logger.info(f"New best score: {self.best_score:.4f}")
            
            iteration_time = time.time() - start_time
            self.optimization_history.append({
                'iteration': iteration + 1,
                'parameters': next_params,
                'score': score,
                'best_score': self.best_score,
                'time': iteration_time,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s, score: {score:.4f}")
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        
        return {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'total_evaluations': len(self.y_observed)
        }
    
    async def _initialize_random_points(self, objective_function: Callable[[Dict[str, Any]], float]):
        """Initialize with random parameter evaluations."""
        logger.info(f"Initializing with {self.n_initial_points} random points")
        
        for i in range(self.n_initial_points):
            random_encoded = np.random.random(len(self.parameter_space))
            random_params = self._decode_parameters(random_encoded)
            
            score = await objective_function(random_params)
            
            self.X_observed.append(random_encoded)
            self.y_observed.append(score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = random_params.copy()
            
            logger.info(f"Random init {i + 1}/{self.n_initial_points}: score={score:.4f}")
    
    def _find_next_point(self) -> np.ndarray:
        """Find next point to evaluate using acquisition function."""
        if len(self.X_observed) == 0:
            return np.random.random(len(self.parameter_space))
        
        n_candidates = 1000
        candidates = np.random.random((n_candidates, len(self.parameter_space)))
        
        mu, var = self.gp.predict(candidates)
        sigma = np.sqrt(var)
        
        if self.acquisition_function == 'expected_improvement':
            acquisition_values = AcquisitionFunction.expected_improvement(
                mu, sigma, self.best_score, self.xi
            )
        elif self.acquisition_function == 'upper_confidence_bound':
            acquisition_values = AcquisitionFunction.upper_confidence_bound(
                mu, sigma, self.beta
            )
        elif self.acquisition_function == 'probability_of_improvement':
            acquisition_values = AcquisitionFunction.probability_of_improvement(
                mu, sigma, self.best_score, self.xi
            )
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
        
        best_idx = np.argmax(acquisition_values)
        return candidates[best_idx]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.optimization_history:
            return {}
        
        scores = [entry['score'] for entry in self.optimization_history]
        times = [entry['time'] for entry in self.optimization_history]
        
        return {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'total_iterations': len(self.optimization_history),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_iteration_time': np.mean(times),
            'total_optimization_time': sum(times),
            'improvement_over_random': self.best_score - np.mean(scores[:self.n_initial_points]) if len(scores) > self.n_initial_points else 0
        }
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file."""
        results = {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'parameter_space': self.parameter_space,
            'summary': self.get_optimization_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str):
        """Load optimization results from file."""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.best_params = results.get('best_parameters')
        self.best_score = results.get('best_score', -np.inf)
        self.optimization_history = results.get('optimization_history', [])
        
        self.X_observed = []
        self.y_observed = []
        for entry in self.optimization_history:
            encoded = self._encode_parameters(entry['parameters'])
            self.X_observed.append(encoded)
            self.y_observed.append(entry['score'])
        
        logger.info(f"Optimization results loaded from {filepath}")


class QuantumHyperparameterObjective:
    """Objective function for quantum-hybrid trading system optimization."""
    
    def __init__(self, config: Dict[str, Any], qpu_tracker=None):
        self.config = config
        self.qpu_tracker = qpu_tracker
        self.base_config = config.copy()
        
        self.eval_config = config.get('meta_learning', {}).get('evaluation', {})
        self.backtest_period = self.eval_config.get('backtest_period', 90)  # days
        self.min_trades = self.eval_config.get('min_trades', 10)
        self.sharpe_weight = self.eval_config.get('sharpe_weight', 0.7)
        self.return_weight = self.eval_config.get('return_weight', 0.2)
        self.efficiency_weight = self.eval_config.get('efficiency_weight', 0.1)
        
        logger.info("Quantum hyperparameter objective initialized")
    
    async def __call__(self, parameters: Dict[str, Any]) -> float:
        """Evaluate hyperparameters and return objective score."""
        start_time = time.time()
        
        try:
            updated_config = self._update_config_with_parameters(parameters)
            
            backtest_result = await self._run_parameter_backtest(updated_config)
            
            score = self._calculate_objective_score(backtest_result, parameters)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Parameter evaluation completed in {elapsed_time:.2f}s, score: {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Parameter evaluation failed: {e}")
            return -1000.0  # Large penalty for failed evaluations
    
    def _update_config_with_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with new hyperparameters."""
        config = self.base_config.copy()
        
        if 'quantum' not in config:
            config['quantum'] = {}
        
        quantum_config = config['quantum']
        
        if 'algorithms' not in quantum_config:
            quantum_config['algorithms'] = {}
        if 'qaoa' not in quantum_config['algorithms']:
            quantum_config['algorithms']['qaoa'] = {}
        
        qaoa_config = quantum_config['algorithms']['qaoa']
        qaoa_config['max_layers'] = parameters.get('qaoa_depth', 3)
        qaoa_config['shots'] = parameters.get('qaoa_shots', 1024)
        qaoa_config['optimizer'] = parameters.get('qaoa_optimizer', 'COBYLA')
        qaoa_config['warm_start'] = parameters.get('qaoa_warm_start', True)
        
        if 'diffusion' not in quantum_config:
            quantum_config['diffusion'] = {}
        
        diffusion_config = quantum_config['diffusion']
        diffusion_config['timesteps'] = parameters.get('diffusion_timesteps', 100)
        diffusion_config['noise_schedule'] = parameters.get('diffusion_noise_schedule', 'cosine')
        diffusion_config['learning_rate'] = parameters.get('diffusion_learning_rate', 1e-3)
        diffusion_config['batch_size'] = parameters.get('diffusion_batch_size', 32)
        
        if 'var' not in quantum_config:
            quantum_config['var'] = {}
        
        var_config = quantum_config['var']
        var_config['confidence_levels'] = parameters.get('var_confidence_levels', [0.95, 0.99])
        var_config['lookback_days'] = parameters.get('var_lookback_days', 252)
        var_config['num_qubits'] = parameters.get('var_num_qubits', 8)
        
        if 'rl' not in config:
            config['rl'] = {}
        
        rl_config = config['rl']
        rl_config['learning_rate'] = parameters.get('rl_learning_rate', 3e-4)
        rl_config['gamma'] = parameters.get('rl_gamma', 0.99)
        rl_config['clip_epsilon'] = parameters.get('rl_clip_epsilon', 0.2)
        rl_config['entropy_coef'] = parameters.get('rl_entropy_coef', 0.01)
        
        if 'risk' not in config:
            config['risk'] = {}
        
        risk_config = config['risk']
        risk_config['var_threshold'] = parameters.get('risk_var_threshold', 0.05)
        risk_config['concentration_limit'] = parameters.get('risk_concentration_limit', 0.3)
        risk_config['max_drawdown'] = parameters.get('risk_max_drawdown', 0.15)
        
        return config
    
    async def _run_parameter_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest with given parameters."""
        from ..agents.lean_core_agent import LeanCoreAgent
        from ..telemetry.acu_tracker import ACUTracker
        from ..telemetry.qpu_tracker import QPUTracker
        
        acu_tracker = ACUTracker(budget=20)
        qpu_tracker = QPUTracker(budget_minutes=10) if config.get('quantum', {}).get('enabled', False) else None
        
        agent = LeanCoreAgent(
            mode='backtest',
            quantum_enabled=config.get('quantum', {}).get('enabled', False),
            config=config,
            acu_tracker=acu_tracker,
            qpu_tracker=qpu_tracker
        )
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.backtest_period)).strftime('%Y-%m-%d')
        
        backtest_result = await agent.run_backtest(
            symbol='SPY',
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            'sharpe_ratio': backtest_result.sharpe_ratio,
            'total_return': backtest_result.total_return,
            'max_drawdown': backtest_result.max_drawdown,
            'win_rate': backtest_result.win_rate,
            'total_trades': backtest_result.total_trades,
            'acu_used': acu_tracker.total_used,
            'qpu_used': qpu_tracker.total_used if qpu_tracker else 0
        }
    
    def _calculate_objective_score(self, backtest_result: Dict[str, Any], parameters: Dict[str, Any]) -> float:
        """Calculate objective score from backtest results."""
        sharpe_ratio = backtest_result.get('sharpe_ratio', 0)
        total_return = backtest_result.get('total_return', 0)
        max_drawdown = abs(backtest_result.get('max_drawdown', 0))
        win_rate = backtest_result.get('win_rate', 0)
        total_trades = backtest_result.get('total_trades', 0)
        acu_used = backtest_result.get('acu_used', 20)
        qpu_used = backtest_result.get('qpu_used', 0)
        
        sharpe_score = sharpe_ratio * self.sharpe_weight
        
        return_score = np.tanh(total_return * 5) * self.return_weight
        
        acu_efficiency = max(0, (20 - acu_used) / 20)
        qpu_efficiency = max(0, (10 - qpu_used) / 10) if qpu_used > 0 else 1.0
        efficiency_score = (acu_efficiency + qpu_efficiency) / 2 * self.efficiency_weight
        
        drawdown_penalty = max(0, max_drawdown - 0.15) * 10  # Penalty for >15% drawdown
        trade_penalty = max(0, self.min_trades - total_trades) * 0.1  # Penalty for too few trades
        
        total_score = sharpe_score + return_score + efficiency_score - drawdown_penalty - trade_penalty
        
        if win_rate > 0.6:
            total_score += (win_rate - 0.6) * 2
        
        return total_score


async def optimize_quantum_hyperparameters(config: Dict[str, Any], qpu_tracker=None) -> Dict[str, Any]:
    """Main function to optimize quantum-hybrid trading system hyperparameters."""
    logger.info("Starting quantum hyperparameter optimization")
    
    optimizer = BayesianOptimizer(config, qpu_tracker)
    objective = QuantumHyperparameterObjective(config, qpu_tracker)
    
    results = await optimizer.optimize(objective)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"optimization_results_{timestamp}.json"
    optimizer.save_optimization_results(results_file)
    
    logger.info(f"Hyperparameter optimization completed. Results saved to {results_file}")
    
    return results
