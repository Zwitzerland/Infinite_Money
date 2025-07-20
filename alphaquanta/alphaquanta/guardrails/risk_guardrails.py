"""
Risk guardrails engine for trade validation and safety checks.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..models import TradeSignal, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    approved: bool
    risk_score: float
    rejection_reason: str = ""
    requires_hitl: bool = False
    guardrail_violations: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseGuardrail:
    """Base class for all guardrails."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def validate(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate trading signal. Must be implemented by subclasses."""
        raise NotImplementedError


class NotionalLimitGuardrail(BaseGuardrail):
    """Guardrail for notional position limits."""
    
    def __init__(self, max_position_size: float = 10000.0, max_daily_volume: float = 50000.0):
        super().__init__("notional_limit")
        self.max_position_size = max_position_size
        self.max_daily_volume = max_daily_volume
        self.daily_volume = 0.0
        self.last_reset = datetime.now().date()
    
    async def validate(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate notional limits."""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_volume = 0.0
            self.last_reset = current_date
        
        estimated_price = 100.0 + (hash(signal.symbol) % 100)
        notional_value = signal.quantity * estimated_price
        
        if notional_value > self.max_position_size:
            return {
                "approved": False,
                "risk_score": 0.9,
                "rejection_reason": f"Excessive position size: ${notional_value:,.2f} > ${self.max_position_size:,.2f}",
                "requires_hitl": True
            }
        
        if self.daily_volume + notional_value > self.max_daily_volume:
            return {
                "approved": False,
                "risk_score": 0.8,
                "rejection_reason": f"Daily volume limit exceeded: ${self.daily_volume + notional_value:,.2f} > ${self.max_daily_volume:,.2f}",
                "requires_hitl": True
            }
        
        self.daily_volume += notional_value
        
        risk_score = min(notional_value / self.max_position_size, 1.0) * 0.5
        
        return {
            "approved": True,
            "risk_score": risk_score,
            "rejection_reason": "",
            "requires_hitl": False
        }


class SymbolRiskGuardrail(BaseGuardrail):
    """Guardrail for suspicious symbols and patterns."""
    
    def __init__(self):
        super().__init__("symbol_risk")
        self.high_risk_symbols = {"GME", "AMC", "BBBY", "MEME"}
        self.trade_history = []
    
    async def validate(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate symbol risk patterns."""
        risk_score = 0.0
        requires_hitl = False
        rejection_reason = ""
        
        if signal.symbol in self.high_risk_symbols:
            risk_score += 0.4
            self.logger.warning(f"High-risk symbol detected: {signal.symbol}")
        
        if signal.quantity >= 10000:
            risk_score += 0.5
            requires_hitl = True
            if signal.symbol == "GME":
                return {
                    "approved": False,
                    "risk_score": 0.95,
                    "rejection_reason": "Jailbreak attempt detected: Large GME position blocked",
                    "requires_hitl": True
                }
        
        recent_trades = [t for t in self.trade_history 
                        if t['timestamp'] > datetime.now() - timedelta(minutes=5)]
        
        if len(recent_trades) > 5:
            risk_score += 0.3
            requires_hitl = True
            rejection_reason = "High-frequency trading pattern detected"
        
        self.trade_history.append({
            'symbol': signal.symbol,
            'quantity': signal.quantity,
            'timestamp': datetime.now()
        })
        
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-50:]
        
        approved = risk_score < 0.8
        
        return {
            "approved": approved,
            "risk_score": risk_score,
            "rejection_reason": rejection_reason,
            "requires_hitl": requires_hitl
        }


class ConcentrationGuardrail(BaseGuardrail):
    """Guardrail for portfolio concentration limits."""
    
    def __init__(self, max_concentration: float = 0.9):
        super().__init__("concentration")
        self.max_concentration = max_concentration
    
    async def validate(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate portfolio concentration."""
        current_portfolio = await self.get_current_portfolio()
        
        estimated_price = 100.0 + (hash(signal.symbol) % 100)
        new_position_value = signal.quantity * estimated_price
        
        total_portfolio_value = sum(pos['market_value'] for pos in current_portfolio.values())
        total_portfolio_value += new_position_value
        
        current_symbol_value = current_portfolio.get(signal.symbol, {}).get('market_value', 0)
        new_symbol_value = current_symbol_value + new_position_value
        
        concentration = new_symbol_value / total_portfolio_value if total_portfolio_value > 0 else 0
        
        if concentration > self.max_concentration:
            return {
                "approved": False,
                "risk_score": 0.9,
                "rejection_reason": f"Portfolio concentration limit exceeded: {concentration:.1%} > {self.max_concentration:.1%}",
                "requires_hitl": True
            }
        
        risk_score = concentration / self.max_concentration * 0.6
        
        return {
            "approved": True,
            "risk_score": risk_score,
            "rejection_reason": "",
            "requires_hitl": False
        }
    
    async def get_current_portfolio(self) -> Dict[str, Dict[str, float]]:
        """Get current portfolio positions."""
        return {
            "SPY": {"quantity": 8000, "market_value": 400000},
            "QQQ": {"quantity": 1000, "market_value": 50000}
        }


class QuantumCircuitGuardrail(BaseGuardrail):
    """Guardrail for quantum circuit validation."""
    
    def __init__(self):
        super().__init__("quantum_circuit")
    
    async def validate(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate quantum circuit parameters."""
        metadata = signal.metadata or {}
        
        if not metadata.get("quantum_enhanced", False):
            return {
                "approved": True,
                "risk_score": 0.0,
                "rejection_reason": "",
                "requires_hitl": False,
                "metadata": {"quantum_validated": False}
            }
        
        circuit_depth = metadata.get("circuit_depth", 0)
        qpu_time_estimate = metadata.get("qpu_time_estimate", 0)
        
        if circuit_depth > 20:
            return {
                "approved": False,
                "risk_score": 0.8,
                "rejection_reason": f"Circuit depth too high: {circuit_depth} > 20",
                "requires_hitl": True
            }
        
        if qpu_time_estimate > 300:
            return {
                "approved": False,
                "risk_score": 0.7,
                "rejection_reason": f"QPU time estimate too high: {qpu_time_estimate}s > 300s",
                "requires_hitl": True
            }
        
        return {
            "approved": True,
            "risk_score": 0.1,
            "rejection_reason": "",
            "requires_hitl": False,
            "metadata": {"quantum_validated": True}
        }


class RiskGuardrailEngine:
    """Main risk guardrail engine coordinating all guardrails."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.emergency_stop = False
        self.guardrails = []
        
        self._initialize_guardrails()
    
    def _initialize_guardrails(self):
        """Initialize all guardrails."""
        risk_config = self.config.get('risk', {})
        
        self.guardrails.append(NotionalLimitGuardrail(
            max_position_size=risk_config.get('max_position_size', 10000.0),
            max_daily_volume=risk_config.get('max_daily_volume', 50000.0)
        ))
        
        self.guardrails.append(SymbolRiskGuardrail())
        self.guardrails.append(ConcentrationGuardrail(
            max_concentration=risk_config.get('max_concentration', 0.9)
        ))
        self.guardrails.append(QuantumCircuitGuardrail())
        
        self.logger.info(f"Initialized {len(self.guardrails)} guardrails")
    
    def add_guardrail(self, guardrail: BaseGuardrail):
        """Add a custom guardrail."""
        self.guardrails.append(guardrail)
        self.logger.info(f"Added guardrail: {guardrail.name}")
    
    async def validate_signal(self, signal: TradeSignal) -> GuardrailResult:
        """Validate trading signal against all guardrails."""
        if self.emergency_stop:
            return GuardrailResult(
                approved=False,
                risk_score=1.0,
                rejection_reason="Emergency stop activated",
                requires_hitl=True,
                guardrail_violations=1
            )
        
        total_risk_score = 0.0
        violations = 0
        rejection_reasons = []
        requires_hitl = False
        combined_metadata = {}
        
        for guardrail in self.guardrails:
            try:
                result = await guardrail.validate(signal)
                
                if not result.get("approved", True):
                    violations += 1
                    rejection_reasons.append(f"{guardrail.name}: {result.get('rejection_reason', 'Validation failed')}")
                
                total_risk_score += result.get("risk_score", 0.0)
                
                if result.get("requires_hitl", False):
                    requires_hitl = True
                
                if "metadata" in result:
                    combined_metadata.update(result["metadata"])
                
            except Exception as e:
                self.logger.error(f"Guardrail {guardrail.name} failed: {e}")
                violations += 1
                rejection_reasons.append(f"{guardrail.name}: Internal error")
                requires_hitl = True
        
        approved = violations == 0 and total_risk_score < 0.8
        
        if requires_hitl and not approved:
            await self._trigger_hitl_escalation(signal, total_risk_score, rejection_reasons)
        
        return GuardrailResult(
            approved=approved,
            risk_score=min(total_risk_score, 1.0),
            rejection_reason="; ".join(rejection_reasons) if rejection_reasons else "",
            requires_hitl=requires_hitl,
            guardrail_violations=violations,
            metadata=combined_metadata
        )
    
    async def _trigger_hitl_escalation(self, signal: TradeSignal, risk_score: float, reasons: List[str]):
        """Trigger human-in-the-loop escalation."""
        self.logger.warning(f"HITL escalation triggered for {signal.symbol}: {reasons}")
        
        try:
            from ..guardrails.hitl_escalation import send_escalation_alert
            await send_escalation_alert(
                signal=signal,
                risk_score=risk_score,
                reasons=reasons,
                timestamp=datetime.now()
            )
        except ImportError:
            self.logger.warning("HITL escalation module not available - blocking trade")
            return
