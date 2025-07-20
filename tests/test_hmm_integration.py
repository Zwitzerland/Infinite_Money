"""
Comprehensive HMM integration tests for Quantum Apex Money Engine 2.0.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock

from alphaquanta.alphaquanta.regime.hmm_detector import MarketRegimeDetector
from alphaquanta.alphaquanta.quantum.qaoa_optimizer import QAOABasketOptimizer
from alphaquanta.alphaquanta.agents.lean_core_agent import LeanCoreAgent
from alphaquanta.alphaquanta.guardrails.risk_guardrails import HMMRegimeGuardrail
from alphaquanta.alphaquanta.telemetry.pnl_monitor import PnLMonitor
from alphaquanta.alphaquanta.telemetry.qpu_tracker import QPUTracker
from alphaquanta.alphaquanta.models import TradeSignal, OrderSide, OrderType


class TestHMMIntegration:
    """Test HMM integration with quantum components."""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'regime_detection': {
                'n_states': 3,
                'lookback_days': 252,
                'min_observations': 50
            },
            'quantum': {
                'algorithms': {
                    'qaoa': {'max_layers': 2}
                }
            }
        }
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data for testing."""
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        return pd.DataFrame({'date': dates, 'price': prices})
    
    @pytest.mark.asyncio
    async def test_hmm_detector_training(self, mock_config, sample_price_data):
        """Test HMM detector training and regime detection."""
        detector = MarketRegimeDetector(mock_config)
        
        result = await detector.train_model(sample_price_data)
        assert result['success'] is True
        assert result['log_likelihood'] < 0  # Log likelihood should be negative
        assert detector.hmm_model is not None
        
        regime_info = await detector.detect_regime(sample_price_data.tail(50))
        assert 'current_regime' in regime_info
        assert 'confidence' in regime_info
        assert 'state_probabilities' in regime_info
        assert len(regime_info['state_probabilities']) == 3
        assert 0 <= regime_info['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_qaoa_hmm_modulation(self, mock_config):
        """Test QAOA optimizer with HMM state modulation."""
        qpu_tracker = QPUTracker(mock_config.get('quantum', {}))
        optimizer = QAOABasketOptimizer(mock_config.get('quantum', {}), qpu_tracker)
        
        hmm_state_probs = np.array([0.2, 0.6, 0.2])
        modulation_factor = optimizer._calculate_regime_modulation(hmm_state_probs)
        
        assert isinstance(modulation_factor, float)
        assert 0.8 <= modulation_factor <= 1.2  # Should be within expected range
        
        symbols = ['SPY']
        weights = await optimizer.optimize_basket(
            symbols,
            hmm_state_probs=hmm_state_probs
        )
        
        assert len(weights) > 0
        assert 'SPY' in weights
        assert isinstance(weights['SPY'], float)
    
    @pytest.mark.asyncio
    async def test_lean_core_agent_hmm_integration(self, mock_config):
        """Test LeanCoreAgent with HMM integration."""
        acu_tracker = Mock()
        acu_tracker.start_operation = Mock()
        acu_tracker.end_operation = Mock()
        
        qpu_tracker = QPUTracker(mock_config.get('quantum', {}))
        
        agent = LeanCoreAgent(
            mode='paper',
            quantum_enabled=True,
            config=mock_config,
            acu_tracker=acu_tracker,
            qpu_tracker=qpu_tracker
        )
        
        assert agent.regime_detector is not None
        assert isinstance(agent.regime_detector, MarketRegimeDetector)
        
        mock_market_data = {
            'price': 150.0,
            'sma_20': 148.0,
            'rsi': 45.0,
            'historical_data': [
                {'price': 145.0 + i * 0.5} for i in range(60)
            ]
        }
        
        agent.market_data.get_current_data = AsyncMock(return_value=mock_market_data)
        
        signals = await agent.generate_signals(['SPY'])
        assert len(signals) == 1
        
        signal = signals[0]
        assert signal.symbol == 'SPY'
        assert 'quantum_hybrid' in signal.strategy
        assert signal.metadata.get('quantum_enhanced') is True
    
    def test_hmm_regime_guardrail(self):
        """Test HMM-aware risk guardrail."""
        mock_detector = Mock()
        mock_detector.current_regime = {
            'current_state': 2,  # High risk regime
            'confidence': 0.85
        }
        
        guardrail = HMMRegimeGuardrail(regime_detector=mock_detector)
        
        signal = TradeSignal(
            symbol='GME',
            side=OrderSide.BUY,
            quantity=10000,  # Large position
            order_type=OrderType.MARKET,
            confidence=0.8,
            strategy='test'
        )
        
        result = asyncio.run(guardrail.validate(signal))
        
        assert result['approved'] is False  # Should be blocked
        assert result['requires_hitl'] is True  # Should require human intervention
        assert result['metadata']['hmm_regime_validated'] is True
        assert result['metadata']['current_regime'] == 2
    
    def test_pnl_monitor_hmm_metrics(self):
        """Test PnL monitor with HMM metrics integration."""
        monitor = PnLMonitor(initial_capital=100000.0)
        
        mock_detector = Mock()
        mock_detector.current_regime = {
            'current_regime': 'low_volatility_bullish',
            'current_state': 0,
            'confidence': 0.82,
            'stability': 0.75,
            'state_probabilities': [0.8, 0.15, 0.05]
        }
        
        hmm_metrics = monitor.add_hmm_metrics(mock_detector)
        
        assert hmm_metrics['hmm_current_regime'] == 'low_volatility_bullish'
        assert hmm_metrics['hmm_current_state'] == 0
        assert hmm_metrics['hmm_confidence'] == 0.82
        assert hmm_metrics['hmm_stability'] == 0.75
        assert len(hmm_metrics['hmm_state_distribution']) == 3
        
        digest_json = monitor.export_executive_digest(mock_detector)
        assert 'hmm_state_distribution' in digest_json
        assert 'regime_info' in digest_json
    
    @pytest.mark.asyncio
    async def test_end_to_end_hmm_workflow(self, mock_config, sample_price_data):
        """Test complete end-to-end HMM workflow."""
        detector = MarketRegimeDetector(mock_config)
        qpu_tracker = QPUTracker(mock_config.get('quantum', {}))
        optimizer = QAOABasketOptimizer(mock_config.get('quantum', {}), qpu_tracker)
        
        training_result = await detector.train_model(sample_price_data)
        assert training_result['success'] is True
        
        regime_info = await detector.detect_regime(sample_price_data.tail(50))
        hmm_state_probs = np.array(regime_info['state_probabilities'])
        
        weights = await optimizer.optimize_basket(
            ['SPY', 'QQQ'],
            hmm_state_probs=hmm_state_probs
        )
        
        assert len(weights) > 0
        assert all(isinstance(w, float) for w in weights.values())
        
        mock_detector_obj = Mock()
        mock_detector_obj.current_regime = {
            'current_state': regime_info['current_regime'],
            'confidence': regime_info['confidence']
        }
        
        guardrail = HMMRegimeGuardrail(regime_detector=mock_detector_obj)
        
        test_signal = TradeSignal(
            symbol='SPY',
            side=OrderSide.BUY,
            quantity=1000,
            order_type=OrderType.MARKET,
            confidence=0.7,
            strategy='hmm_enhanced'
        )
        
        validation_result = await guardrail.validate(test_signal)
        assert 'hmm_regime_validated' in validation_result['metadata']
        
        print("✅ End-to-end HMM workflow test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
