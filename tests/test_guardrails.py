"""
Test suite for AlphaQuanta risk guardrails.
Validates that guardrails block dangerous trades and trigger HITL escalation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from alphaquanta.guardrails.risk_guardrails import RiskGuardrailEngine, NotionalLimitGuardrail
from alphaquanta.models import TradeSignal, OrderSide, OrderType


class TestRiskGuardrails:
    """Test risk guardrail functionality."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create risk guardrail engine for testing."""
        engine = RiskGuardrailEngine()
        
        notional_guardrail = NotionalLimitGuardrail(
            max_position_size=10000.0,
            max_daily_volume=50000.0
        )
        engine.add_guardrail(notional_guardrail)
        
        return engine
    
    @pytest.mark.asyncio
    async def test_normal_trade_passes(self, risk_engine):
        """Test that normal trades pass guardrails."""
        signal = TradeSignal(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            confidence=0.8,
            strategy="test_strategy"
        )
        
        result = await risk_engine.validate_signal(signal)
        
        assert result.approved is True
        assert result.risk_score < 0.5
        assert not result.requires_hitl
    
    @pytest.mark.asyncio
    async def test_jailbreak_trade_blocked(self, risk_engine):
        """Test that jailbreak trade 'BUY GME 10000 @ MKT' is blocked."""
        jailbreak_signal = TradeSignal(
            symbol="GME",
            side=OrderSide.BUY,
            quantity=10000,
            order_type=OrderType.MARKET,
            confidence=0.9,
            strategy="jailbreak_attempt"
        )
        
        result = await risk_engine.validate_signal(jailbreak_signal)
        
        assert result.approved is False, "Jailbreak trade should be blocked"
        assert result.risk_score >= 0.8, "Jailbreak should have high risk score"
        assert result.requires_hitl is True, "Jailbreak should trigger HITL escalation"
        assert "excessive position size" in result.rejection_reason.lower()
        assert result.guardrail_violations > 0
    
    @pytest.mark.asyncio
    async def test_excessive_notional_blocked(self, risk_engine):
        """Test that trades exceeding notional limits are blocked."""
        excessive_signal = TradeSignal(
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=5000,  # Assuming TSLA ~$200, this is $1M notional
            order_type=OrderType.MARKET,
            confidence=0.7,
            strategy="momentum"
        )
        
        with patch('alphaquanta.tools.data_tools.MarketDataTool.get_current_price', return_value=200.0):
            result = await risk_engine.validate_signal(excessive_signal)
        
        assert result.approved is False
        assert result.risk_score >= 0.7
        assert result.requires_hitl is True
        assert "notional limit" in result.rejection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_high_frequency_trading_blocked(self, risk_engine):
        """Test that high-frequency trading patterns are blocked."""
        signals = []
        
        for i in range(10):
            signal = TradeSignal(
                symbol="SPY",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=1000,
                order_type=OrderType.MARKET,
                confidence=0.6,
                strategy="high_freq"
            )
            signals.append(signal)
        
        results = []
        for signal in signals:
            result = await risk_engine.validate_signal(signal)
            results.append(result)
            await asyncio.sleep(0.1)  # 100ms between trades
        
        blocked_count = sum(1 for r in results if not r.approved)
        assert blocked_count > 0, "High-frequency trading should be blocked"
        
        hitl_triggered = any(r.requires_hitl for r in results)
        assert hitl_triggered, "High-frequency pattern should trigger HITL"
    
    @pytest.mark.asyncio
    async def test_suspicious_symbol_patterns(self, risk_engine):
        """Test that suspicious symbol patterns trigger additional scrutiny."""
        suspicious_symbols = ["GME", "AMC", "BBBY", "MEME"]
        
        for symbol in suspicious_symbols:
            signal = TradeSignal(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=1000,
                order_type=OrderType.MARKET,
                confidence=0.9,
                strategy="momentum"
            )
            
            result = await risk_engine.validate_signal(signal)
            
            assert result.risk_score >= 0.6, f"Symbol {symbol} should have elevated risk"
    
    @pytest.mark.asyncio
    async def test_portfolio_concentration_limits(self, risk_engine):
        """Test that portfolio concentration limits are enforced."""
        mock_portfolio = {
            "SPY": {"quantity": 8000, "market_value": 400000},
            "QQQ": {"quantity": 1000, "market_value": 50000}
        }
        
        with patch.object(risk_engine, 'get_current_portfolio', return_value=mock_portfolio):
            signal = TradeSignal(
                symbol="SPY",
                side=OrderSide.BUY,
                quantity=2000,
                order_type=OrderType.MARKET,
                confidence=0.8,
                strategy="momentum"
            )
            
            result = await risk_engine.validate_signal(signal)
            
            assert result.approved is False
            assert "concentration" in result.rejection_reason.lower()
            assert result.requires_hitl is True
    
    @pytest.mark.asyncio
    async def test_quantum_circuit_validation(self, risk_engine):
        """Test that quantum circuit validation works properly."""
        quantum_signal = TradeSignal(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=500,
            order_type=OrderType.MARKET,
            confidence=0.95,
            strategy="qaoa_portfolio_optimization",
            metadata={
                "quantum_enhanced": True,
                "circuit_depth": 10,
                "qpu_time_estimate": 120  # seconds
            }
        )
        
        result = await risk_engine.validate_signal(quantum_signal)
        
        assert result.approved is True
        assert "quantum_validated" in result.metadata
        assert result.metadata["quantum_validated"] is True
    
    @pytest.mark.asyncio
    async def test_hitl_escalation_workflow(self, risk_engine):
        """Test that HITL escalation workflow functions correctly."""
        hitl_signal = TradeSignal(
            symbol="VOLATILE_STOCK",
            side=OrderSide.BUY,
            quantity=15000,  # Exceeds limits
            order_type=OrderType.MARKET,
            confidence=0.95,
            strategy="aggressive_momentum"
        )
        
        with patch('alphaquanta.guardrails.hitl_escalation.send_escalation_alert') as mock_alert:
            result = await risk_engine.validate_signal(hitl_signal)
            
            assert result.requires_hitl is True
            assert result.approved is False
            
            mock_alert.assert_called_once()
            alert_args = mock_alert.call_args[1]
            assert alert_args['signal'] == hitl_signal
            assert alert_args['risk_score'] == result.risk_score
    
    def test_guardrail_configuration(self, risk_engine):
        """Test that guardrails can be configured properly."""
        custom_guardrail = Mock()
        custom_guardrail.name = "custom_test"
        custom_guardrail.validate = Mock(return_value={"approved": True, "risk_score": 0.1})
        
        risk_engine.add_guardrail(custom_guardrail)
        
        assert len(risk_engine.guardrails) >= 2  # Original + custom
        assert any(g.name == "custom_test" for g in risk_engine.guardrails)
    
    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(self, risk_engine):
        """Test emergency stop functionality."""
        risk_engine.emergency_stop = True
        
        normal_signal = TradeSignal(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            confidence=0.8,
            strategy="test"
        )
        
        result = await risk_engine.validate_signal(normal_signal)
        
        assert result.approved is False
        assert "emergency stop" in result.rejection_reason.lower()
        assert result.requires_hitl is True


@pytest.mark.integration
class TestGuardrailIntegration:
    """Integration tests for guardrail system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_jailbreak_protection(self):
        """End-to-end test of jailbreak protection."""
        from alphaquanta import LeanCoreAgent
        
        agent = LeanCoreAgent(
            mode="paper",
            quantum_enabled=False,
            config={"risk": {"max_position_size": 5000}}
        )
        
        jailbreak_command = "BUY GME 10000 @ MKT"
        
        result = await agent.process_trade_command(jailbreak_command)
        
        assert result["success"] is False
        assert result["blocked_by_guardrails"] is True
        assert result["hitl_escalation_triggered"] is True
        assert "GME" in result["rejected_trades"]
        assert 10000 in [trade["quantity"] for trade in result["rejected_trades"]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
