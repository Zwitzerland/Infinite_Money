"""
LeanCoreAgent - Single omni-agent for quantum-hybrid trading signals and execution.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import json

from ..models import TradingMode, TradeSet, TradeSignal, BacktestResult, OrderSide, OrderType
from ..tools.data_tools import QuantConnectDataTool, MarketDataTool
from ..tools.action_tools import IBOrderRouter, LeanBacktestRunner, PositionSizer
from ..guardrails.risk_guardrails import RiskGuardrailEngine
from ..telemetry.acu_tracker import ACUTracker
from ..telemetry.qpu_tracker import QPUTracker

logger = logging.getLogger(__name__)


class LeanCoreAgent:
    """Core trading agent with quantum-enhanced alpha discovery."""
    
    def __init__(self, mode: str, quantum_enabled: bool, config: Dict[str, Any],
                 acu_tracker: ACUTracker, qpu_tracker: Optional[QPUTracker] = None):
        self.mode = TradingMode(mode)
        self.quantum_enabled = quantum_enabled
        self.config = config
        self.acu_tracker = acu_tracker
        self.qpu_tracker = qpu_tracker
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.data_tool = QuantConnectDataTool(config.get('quantconnect', {}))
        self.market_data = MarketDataTool()
        
        if mode == 'live':
            self.executor = IBOrderRouter(config.get('interactive_brokers', {}))
        else:
            self.executor = LeanBacktestRunner(config.get('lean', {}))
            
        self.position_sizer = PositionSizer(config.get('position_sizing', {}))
        self.risk_engine = RiskGuardrailEngine(config.get('risk', {}))
        
        self.qaoa_optimizer = None
        self.diffusion_forecaster = None
        self.quantum_var_calculator = None
        if quantum_enabled and qpu_tracker:
            try:
                from ..quantum.qaoa_optimizer import QAOABasketOptimizer
                from ..quantum.diffusion_forecaster import DiffusionTSForecaster
                from ..quantum.quantum_var import QuantumVaRCalculator
                self.qaoa_optimizer = QAOABasketOptimizer(config.get('quantum', {}), qpu_tracker)
                self.diffusion_forecaster = DiffusionTSForecaster(config.get('quantum', {}), qpu_tracker)
                self.quantum_var_calculator = QuantumVaRCalculator(config.get('quantum', {}))
                self.logger.info("Quantum modules initialized successfully")
            except ImportError as e:
                self.logger.warning(f"Quantum modules not available: {e}")
                self.quantum_enabled = False
    
    async def process_trade_command(self, command: str) -> Dict[str, Any]:
        """Process a trading command and return execution result."""
        self.acu_tracker.start_operation("process_trade_command")
        
        try:
            signal = self._parse_trade_command(command)
            if not signal:
                return {
                    "success": False,
                    "error": "Failed to parse trade command",
                    "blocked_by_guardrails": False,
                    "hitl_escalation_triggered": False,
                    "rejected_trades": []
                }
            
            validation_result = await self.risk_engine.validate_signal(signal)
            
            if not validation_result.approved:
                return {
                    "success": False,
                    "error": validation_result.rejection_reason,
                    "blocked_by_guardrails": True,
                    "hitl_escalation_triggered": validation_result.requires_hitl,
                    "rejected_trades": [{"symbol": signal.symbol, "quantity": signal.quantity}]
                }
            
            execution_result = await self.executor.execute_signal(signal)
            
            return {
                "success": execution_result.get("success", False),
                "trade_id": execution_result.get("trade_id"),
                "blocked_by_guardrails": False,
                "hitl_escalation_triggered": False,
                "rejected_trades": []
            }
            
        except Exception as e:
            self.logger.error(f"Error processing trade command: {e}")
            return {
                "success": False,
                "error": str(e),
                "blocked_by_guardrails": False,
                "hitl_escalation_triggered": True,
                "rejected_trades": []
            }
        finally:
            self.acu_tracker.end_operation("process_trade_command")
    
    def _parse_trade_command(self, command: str) -> Optional[TradeSignal]:
        """Parse trading command into TradeSignal."""
        try:
            parts = command.upper().split()
            if len(parts) < 4:
                return None
            
            side_str = parts[0]
            symbol = parts[1]
            quantity = int(parts[2])
            order_type_str = parts[4] if len(parts) > 4 else "MARKET"
            
            side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
            order_type = OrderType.MARKET if order_type_str == "MKT" else OrderType.LIMIT
            
            return TradeSignal(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                confidence=0.8,
                strategy="manual_command"
            )
        except (ValueError, IndexError):
            return None
    
    async def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """Generate trading signals for given symbols."""
        self.acu_tracker.start_operation("generate_signals")
        
        try:
            signals = []
            
            for symbol in symbols:
                market_data = await self.market_data.get_current_data(symbol)
                
                classical_signal = await self._generate_classical_signal(symbol, market_data)
                
                if self.quantum_enabled and self.qaoa_optimizer:
                    quantum_signal = await self._enhance_with_quantum(classical_signal, market_data)
                    signals.append(quantum_signal)
                else:
                    signals.append(classical_signal)
            
            return signals
            
        finally:
            self.acu_tracker.end_operation("generate_signals")
    
    async def _generate_classical_signal(self, symbol: str, market_data: Dict) -> TradeSignal:
        """Generate classical trading signal using mean reversion + momentum."""
        price = market_data.get('price', 0)
        sma_20 = market_data.get('sma_20', price)
        rsi = market_data.get('rsi', 50)
        
        if price > sma_20 * 1.02 and rsi < 70:
            side = OrderSide.BUY
            confidence = 0.7
        elif price < sma_20 * 0.98 and rsi > 30:
            side = OrderSide.SELL
            confidence = 0.7
        else:
            side = OrderSide.BUY
            confidence = 0.5
        
        quantity = await self.position_sizer.calculate_position_size(symbol, price, confidence)
        
        return TradeSignal(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            confidence=confidence,
            strategy="classical_mean_reversion_momentum"
        )
    
    async def _enhance_with_quantum(self, signal: TradeSignal, market_data: Dict) -> TradeSignal:
        """Enhance signal with quantum optimization and risk assessment."""
        if not self.qaoa_optimizer:
            return signal
        
        try:
            quantum_weights = await self.qaoa_optimizer.optimize_basket(
                [signal.symbol], 
                correlation_matrix=None,
                expected_returns=None,
                budget=10000.0
            )
            
            volatility_forecast = None
            if self.diffusion_forecaster:
                historical_data = market_data.get('historical_data', [])
                if historical_data:
                    volatility_forecast = await self.diffusion_forecaster.forecast_volatility(signal.symbol, historical_data)
            
            var_result = None
            if self.quantum_var_calculator and len(market_data.get('historical_data', [])) > 20:
                portfolio_weights = [1.0]  # Single asset portfolio
                historical_data = market_data.get('historical_data', [])
                prices = [float(data.get('close', data.get('price', 100))) for data in historical_data]
                returns = [0.001] * 20 if len(prices) < 2 else [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
                
                import numpy as np
                var_result = await self.quantum_var_calculator.calculate_var(
                    np.array(portfolio_weights), 
                    np.array(returns).reshape(-1, 1), 
                    confidence_level=0.95,
                    qpu_tracker=self.qpu_tracker
                )
            
            quantum_confidence = quantum_weights.get(signal.symbol, signal.confidence)
            
            if volatility_forecast:
                vol_regime = volatility_forecast.get('volatility_regime', 'normal')
                vol_confidence = volatility_forecast.get('volatility_confidence', 0.5)
                
                if vol_regime == 'high':
                    quantum_confidence *= 0.8  # Reduce confidence in high volatility
                elif vol_regime == 'low':
                    quantum_confidence *= 1.1  # Increase confidence in low volatility
                
                quantum_confidence = min(quantum_confidence * vol_confidence, 0.95)
            
            if var_result:
                var_threshold = 0.03  # 3% daily VaR threshold
                if var_result.get('quantum_var', 0) > var_threshold:
                    risk_penalty = min(0.3, (var_result['quantum_var'] - var_threshold) * 10)
                    quantum_confidence = max(0.1, quantum_confidence - risk_penalty)
            
            enhanced_signal = signal.model_copy()
            enhanced_signal.confidence = min(quantum_confidence * 1.2, 0.95)
            enhanced_signal.strategy = "quantum_hybrid_" + signal.strategy
            enhanced_signal.metadata = {
                "quantum_enhanced": True,
                "original_confidence": signal.confidence,
                "quantum_weights": quantum_weights,
                "volatility_forecast": volatility_forecast,
                "var_result": var_result,
                "quantum_modules_used": ["qaoa", "diffusion", "var"] if var_result else ["qaoa", "diffusion"]
            }
            
            self.logger.info(f"Enhanced signal with quantum modules: confidence {signal.confidence:.2f} → {enhanced_signal.confidence:.2f}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.warning(f"Quantum enhancement failed: {e}")
            return signal
    
    async def run_backtest(self, symbol: str, start_date: str, end_date: str) -> BacktestResult:
        """Run backtest for given symbol and date range."""
        self.acu_tracker.start_operation("run_backtest")
        
        try:
            historical_data = await self.data_tool.get_historical_data(symbol, start_date, end_date)
            
            initial_capital = 100000.0
            portfolio_value = initial_capital
            trades = []
            returns = []
            
            for i, data_point in enumerate(historical_data):
                if i < 20:
                    continue
                
                signals = await self.generate_signals([symbol])
                if not signals:
                    continue
                
                signal = signals[0]
                validation_result = await self.risk_engine.validate_signal(signal)
                
                if validation_result.approved:
                    trade_result = await self.executor.execute_signal(signal)
                    if trade_result.get("success"):
                        trades.append(trade_result)
                        
                        price_change = (data_point['price'] - data_point['prev_price']) / data_point['prev_price']
                        if signal.side == OrderSide.BUY:
                            trade_return = price_change
                        else:
                            trade_return = -price_change
                        
                        returns.append(trade_return)
                        portfolio_value *= (1 + trade_return)
            
            total_return = (portfolio_value - initial_capital) / initial_capital
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            return BacktestResult(
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
                initial_capital=initial_capital,
                final_value=portfolio_value,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=len([r for r in returns if r > 0]) / len(returns) if returns else 0,
                total_trades=len(trades),
                avg_trade_return=sum(returns) / len(returns) if returns else 0
            )
            
        finally:
            self.acu_tracker.end_operation("run_backtest")
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        excess_returns = returns_array - 0.02/252
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0
        
        import numpy as np
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))
