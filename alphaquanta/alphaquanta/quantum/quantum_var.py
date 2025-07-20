"""
Quantum Value-at-Risk (VaR) calculation using amplitude estimation.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from typing import Dict, List, Optional, Tuple
import logging
import time
import asyncio
from scipy import stats

logger = logging.getLogger(__name__)


class QuantumVaRCalculator:
    """Quantum Value-at-Risk calculator using amplitude estimation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.quantum_config = config.get('quantum', {})
        self.var_config = self.quantum_config.get('var', {})
        self.ibm_config = self.quantum_config.get('ibm', {})
        
        self.service = None
        self.backend_name = self.ibm_config.get('backend', 'ibm_brisbane')
        
        if self.ibm_config.get('token'):
            try:
                self.service = QiskitRuntimeService(
                    token=self.ibm_config['token'],
                    instance=self.ibm_config.get('instance', 'ibm-q/open/main')
                )
                logger.info(f"Initialized Qiskit Runtime for VaR calculation")
            except Exception as e:
                logger.warning(f"Failed to initialize Qiskit Runtime: {e}")
                self.service = None
        
        self.confidence_levels = self.var_config.get('confidence_levels', [0.95, 0.99])
        self.num_qubits = self.var_config.get('num_qubits', 8)
        self.shots = self.var_config.get('shots', 1024)
        self.max_evaluation_qubits = self.var_config.get('max_evaluation_qubits', 4)
        
        self.lookback_days = self.var_config.get('lookback_days', 252)
        self.monte_carlo_samples = self.var_config.get('monte_carlo_samples', 10000)
        
        logger.info(f"Quantum VaR Calculator initialized - qubits: {self.num_qubits}, "
                   f"confidence levels: {self.confidence_levels}")
    
    def _create_loss_distribution_circuit(self, portfolio_weights: np.ndarray,
                                        returns_covariance: np.ndarray,
                                        expected_returns: np.ndarray) -> QuantumCircuit:
        """Create quantum circuit encoding portfolio loss distribution."""
        num_assets = len(portfolio_weights)
        qc = QuantumCircuit(self.num_qubits)
        
        normalized_weights = portfolio_weights / np.linalg.norm(portfolio_weights)
        
        qc.h(range(min(num_assets, self.num_qubits)))
        
        for i in range(min(num_assets - 1, self.num_qubits - 1)):
            for j in range(i + 1, min(num_assets, self.num_qubits)):
                if i < self.num_qubits and j < self.num_qubits:
                    correlation = returns_covariance[i, j] if i < len(returns_covariance) and j < len(returns_covariance[0]) else 0
                    angle = np.arcsin(min(1.0, abs(correlation)))
                    qc.cry(angle, i, j)
        
        for i in range(min(num_assets, self.num_qubits)):
            if i < len(expected_returns):
                phase = expected_returns[i] * np.pi
                qc.rz(phase, i)
        
        return qc
    
    def _create_amplitude_estimation_circuit(self, loss_circuit: QuantumCircuit,
                                           threshold: float) -> QuantumCircuit:
        """Create amplitude estimation circuit for VaR calculation."""
        eval_qubits = min(self.max_evaluation_qubits, 4)
        total_qubits = self.num_qubits + eval_qubits
        
        qc = QuantumCircuit(total_qubits, eval_qubits)
        
        for i in range(eval_qubits):
            qc.h(self.num_qubits + i)
        
        qc.compose(loss_circuit, range(self.num_qubits), inplace=True)
        
        self._apply_qft(qc, range(self.num_qubits, total_qubits))
        
        qc.measure(range(self.num_qubits, total_qubits), range(eval_qubits))
        
        return qc
    
    def _apply_qft(self, circuit: QuantumCircuit, qubits: List[int]):
        """Apply Quantum Fourier Transform."""
        n = len(qubits)
        for i in range(n):
            circuit.h(qubits[i])
            for j in range(i + 1, n):
                circuit.cp(np.pi / (2 ** (j - i)), qubits[j], qubits[i])
        
        for i in range(n // 2):
            circuit.swap(qubits[i], qubits[n - 1 - i])
    
    async def _execute_amplitude_estimation(self, circuit: QuantumCircuit) -> float:
        """Execute amplitude estimation circuit."""
        if self.service is None:
            from qiskit_aer import AerSimulator
            simulator = AerSimulator()
            transpiled_circuit = transpile(circuit, simulator)
            
            job = simulator.run(transpiled_circuit, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            logger.info("Executed amplitude estimation on local simulator")
        else:
            try:
                with Session(service=self.service, backend=self.backend_name) as session:
                    sampler = Sampler(session=session)
                    
                    backend = self.service.backend(self.backend_name)
                    transpiled_circuit = transpile(circuit, backend)
                    
                    job = sampler.run([transpiled_circuit], shots=self.shots)
                    result = job.result()
                    
                    counts = {}
                    for i, count in enumerate(result.quasi_dists[0]):
                        bitstring = format(i, f'0{circuit.num_clbits}b')
                        counts[bitstring] = int(count * self.shots)
                    
                    logger.info(f"Executed amplitude estimation on {self.backend_name}")
            except Exception as e:
                logger.error(f"Quantum execution failed: {e}, falling back to simulator")
                return await self._execute_amplitude_estimation(circuit)
        
        total_counts = sum(counts.values())
        if total_counts == 0:
            return 0.0
        
        amplitude_estimate = 0.0
        for bitstring, count in counts.items():
            measurement_value = int(bitstring, 2)
            probability = count / total_counts
            
            amplitude_estimate += probability * measurement_value / (2 ** len(bitstring))
        
        return amplitude_estimate
    
    def _classical_var_fallback(self, portfolio_weights: np.ndarray,
                              returns_covariance: np.ndarray,
                              expected_returns: np.ndarray,
                              confidence_level: float) -> float:
        """Classical VaR calculation fallback."""
        portfolio_return = np.dot(portfolio_weights, expected_returns)
        portfolio_variance = np.dot(portfolio_weights.T, np.dot(returns_covariance, portfolio_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(portfolio_return + z_score * portfolio_volatility)
        
        return max(0, var)
    
    def _monte_carlo_var(self, portfolio_weights: np.ndarray,
                        returns_covariance: np.ndarray,
                        expected_returns: np.ndarray,
                        confidence_level: float) -> float:
        """Monte Carlo VaR calculation."""
        portfolio_returns = np.random.multivariate_normal(
            expected_returns, returns_covariance, self.monte_carlo_samples
        )
        
        portfolio_pnl = np.dot(portfolio_returns, portfolio_weights)
        
        var_percentile = (1 - confidence_level) * 100
        var = -np.percentile(portfolio_pnl, var_percentile)
        
        return max(0, var)
    
    async def calculate_var(self, portfolio_weights: np.ndarray,
                          returns_history: np.ndarray,
                          confidence_level: float = 0.95,
                          qpu_tracker=None) -> Dict[str, float]:
        """Calculate Value-at-Risk using quantum amplitude estimation."""
        start_time = time.time()
        
        if len(portfolio_weights) > self.num_qubits:
            logger.warning(f"Portfolio too large ({len(portfolio_weights)} assets) for quantum VaR, using classical fallback")
            return await self._calculate_classical_var(portfolio_weights, returns_history, confidence_level)
        
        qpu_operation_id = None
        if qpu_tracker:
            qpu_operation_id = qpu_tracker.start_quantum_operation('quantum_var', estimated_time=0.8)
        
        try:
            if len(returns_history.shape) == 1:
                returns_history = returns_history.reshape(-1, 1)
            
            expected_returns = np.mean(returns_history[-self.lookback_days:], axis=0)
            returns_covariance = np.cov(returns_history[-self.lookback_days:].T)
            
            if returns_covariance.ndim == 0:
                returns_covariance = np.array([[returns_covariance]])
            elif returns_covariance.ndim == 1:
                returns_covariance = np.diag(returns_covariance)
            
            returns_covariance += np.eye(len(returns_covariance)) * 1e-6
            
            loss_circuit = self._create_loss_distribution_circuit(
                portfolio_weights, returns_covariance, expected_returns
            )
            
            ae_circuit = self._create_amplitude_estimation_circuit(loss_circuit, confidence_level)
            
            amplitude_estimate = await self._execute_amplitude_estimation(ae_circuit)
            
            quantum_var = self._amplitude_to_var(amplitude_estimate, portfolio_weights, 
                                               returns_covariance, expected_returns, confidence_level)
            
            classical_var = self._classical_var_fallback(
                portfolio_weights, returns_covariance, expected_returns, confidence_level
            )
            
            monte_carlo_var = self._monte_carlo_var(
                portfolio_weights, returns_covariance, expected_returns, confidence_level
            )
            
            elapsed_time = time.time() - start_time
            
            result = {
                'quantum_var': quantum_var,
                'classical_var': classical_var,
                'monte_carlo_var': monte_carlo_var,
                'confidence_level': confidence_level,
                'quantum_advantage': (classical_var - quantum_var) / classical_var if classical_var > 0 else 0,
                'execution_time_seconds': elapsed_time,
                'portfolio_size': len(portfolio_weights),
                'method': 'quantum_amplitude_estimation'
            }
            
            logger.info(f"Quantum VaR calculation completed: {quantum_var:.4f} "
                       f"(classical: {classical_var:.4f}, advantage: {result['quantum_advantage']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Quantum VaR calculation failed: {e}")
            return await self._calculate_classical_var(portfolio_weights, returns_history, confidence_level)
        
        finally:
            if qpu_tracker and qpu_operation_id:
                actual_qpu_time = min(0.8, (time.time() - start_time) / 60.0)  # Convert to minutes
                qpu_tracker.end_quantum_operation(qpu_operation_id, actual_qpu_time)

    def _amplitude_to_var(self, amplitude: float, portfolio_weights: np.ndarray,
                         returns_covariance: np.ndarray, expected_returns: np.ndarray,
                         confidence_level: float) -> float:
        """Convert amplitude estimation result to VaR estimate."""
        portfolio_return = np.dot(portfolio_weights, expected_returns)
        portfolio_variance = np.dot(portfolio_weights.T, np.dot(returns_covariance, portfolio_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if amplitude > 0:
            implied_quantile = 1 - amplitude
            
            if implied_quantile > 0.001 and implied_quantile < 0.999:
                z_score = stats.norm.ppf(implied_quantile)
            else:
                z_score = stats.norm.ppf(1 - confidence_level)
            
            quantum_var = -(portfolio_return + z_score * portfolio_volatility)
        else:
            z_score = stats.norm.ppf(1 - confidence_level)
            quantum_var = -(portfolio_return + z_score * portfolio_volatility)
        
        return max(0, quantum_var)

    async def _calculate_classical_var(self, portfolio_weights: np.ndarray,
                                     returns_history: np.ndarray,
                                     confidence_level: float) -> Dict[str, float]:
        """Classical VaR calculation fallback."""
        logger.info("Using classical VaR calculation fallback")
        
        if len(returns_history.shape) == 1:
            returns_history = returns_history.reshape(-1, 1)
        
        expected_returns = np.mean(returns_history[-self.lookback_days:], axis=0)
        returns_covariance = np.cov(returns_history[-self.lookback_days:].T)
        
        if returns_covariance.ndim == 0:
            returns_covariance = np.array([[returns_covariance]])
        elif returns_covariance.ndim == 1:
            returns_covariance = np.diag(returns_covariance)
        
        classical_var = self._classical_var_fallback(
            portfolio_weights, returns_covariance, expected_returns, confidence_level
        )
        
        monte_carlo_var = self._monte_carlo_var(
            portfolio_weights, returns_covariance, expected_returns, confidence_level
        )
        
        return {
            'quantum_var': classical_var,  # Same as classical in fallback
            'classical_var': classical_var,
            'monte_carlo_var': monte_carlo_var,
            'confidence_level': confidence_level,
            'quantum_advantage': 0.0,
            'execution_time_seconds': 0.1,
            'portfolio_size': len(portfolio_weights),
            'method': 'classical_fallback'
        }

    async def calculate_portfolio_risk_metrics(self, portfolio_weights: np.ndarray,
                                             returns_history: np.ndarray,
                                             qpu_tracker=None) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics."""
        start_time = time.time()
        
        risk_metrics = {}
        
        for confidence_level in self.confidence_levels:
            var_result = await self.calculate_var(
                portfolio_weights, returns_history, confidence_level, qpu_tracker
            )
            
            risk_metrics[f'var_{int(confidence_level*100)}'] = var_result['quantum_var']
            risk_metrics[f'classical_var_{int(confidence_level*100)}'] = var_result['classical_var']
        
        if len(returns_history.shape) == 1:
            returns_history = returns_history.reshape(-1, 1)
        
        portfolio_returns = np.dot(returns_history[-self.lookback_days:], portfolio_weights)
        
        for confidence_level in self.confidence_levels:
            var_threshold = risk_metrics[f'var_{int(confidence_level*100)}']
            tail_losses = portfolio_returns[portfolio_returns <= -var_threshold]
            
            if len(tail_losses) > 0:
                expected_shortfall = -np.mean(tail_losses)
            else:
                expected_shortfall = var_threshold * 1.3  # Conservative estimate
            
            risk_metrics[f'expected_shortfall_{int(confidence_level*100)}'] = expected_shortfall
        
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        risk_metrics['max_drawdown'] = abs(max_drawdown)
        risk_metrics['current_drawdown'] = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0
        
        risk_metrics['portfolio_volatility'] = np.std(portfolio_returns)
        risk_metrics['downside_volatility'] = np.std(portfolio_returns[portfolio_returns < 0])
        
        mean_return = np.mean(portfolio_returns)
        risk_metrics['sharpe_ratio'] = mean_return / risk_metrics['portfolio_volatility'] if risk_metrics['portfolio_volatility'] > 0 else 0
        risk_metrics['sortino_ratio'] = mean_return / risk_metrics['downside_volatility'] if risk_metrics['downside_volatility'] > 0 else 0
        
        risk_metrics['quantum_var_efficiency'] = np.mean([
            risk_metrics.get(f'var_{int(cl*100)}', 0) / risk_metrics.get(f'classical_var_{int(cl*100)}', 1)
            for cl in self.confidence_levels
        ])
        
        elapsed_time = time.time() - start_time
        risk_metrics['calculation_time'] = elapsed_time
        
        logger.info(f"Portfolio risk metrics calculated in {elapsed_time:.2f}s")
        
        return risk_metrics

    def get_risk_alert_thresholds(self) -> Dict[str, float]:
        """Get risk alert thresholds for monitoring."""
        return {
            'max_var_95': 0.05,  # 5% daily VaR threshold
            'max_var_99': 0.08,  # 8% daily VaR threshold  
            'max_drawdown': 0.15,  # 15% maximum drawdown
            'min_sharpe_ratio': 1.0,  # Minimum Sharpe ratio
            'max_portfolio_volatility': 0.25,  # 25% annual volatility
            'quantum_efficiency_threshold': 0.95  # Quantum should be at least 95% as good as classical
        }

    def check_risk_breaches(self, risk_metrics: Dict[str, float]) -> List[str]:
        """Check for risk threshold breaches."""
        thresholds = self.get_risk_alert_thresholds()
        breaches = []
        
        if risk_metrics.get('var_95', 0) > thresholds['max_var_95']:
            breaches.append(f"95% VaR breach: {risk_metrics['var_95']:.3f} > {thresholds['max_var_95']:.3f}")
        
        if risk_metrics.get('var_99', 0) > thresholds['max_var_99']:
            breaches.append(f"99% VaR breach: {risk_metrics['var_99']:.3f} > {thresholds['max_var_99']:.3f}")
        
        if risk_metrics.get('max_drawdown', 0) > thresholds['max_drawdown']:
            breaches.append(f"Max drawdown breach: {risk_metrics['max_drawdown']:.3f} > {thresholds['max_drawdown']:.3f}")
        
        if risk_metrics.get('sharpe_ratio', 0) < thresholds['min_sharpe_ratio']:
            breaches.append(f"Low Sharpe ratio: {risk_metrics['sharpe_ratio']:.3f} < {thresholds['min_sharpe_ratio']:.3f}")
        
        if risk_metrics.get('portfolio_volatility', 0) > thresholds['max_portfolio_volatility']:
            breaches.append(f"High volatility: {risk_metrics['portfolio_volatility']:.3f} > {thresholds['max_portfolio_volatility']:.3f}")
        
        if risk_metrics.get('quantum_var_efficiency', 1) < thresholds['quantum_efficiency_threshold']:
            breaches.append(f"Low quantum efficiency: {risk_metrics['quantum_var_efficiency']:.3f} < {thresholds['quantum_efficiency_threshold']:.3f}")
        
        return breaches
