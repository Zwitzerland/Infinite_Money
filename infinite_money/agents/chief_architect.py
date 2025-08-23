"""
Chief Architect Agent for Infinite_Money trading system.

This agent is responsible for:
- Orchestrating all other agents
- Evolving and optimizing trading strategies
- Managing the overall system performance
- Making high-level decisions about strategy deployment
- Continuous learning and adaptation
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .base_agent import BaseAgent, Task
from ..utils.logger import get_logger, log_strategy_generation
from ..utils.config import Config
from ..quantum.strategy_generator import QuantumStrategyGenerator
from ..backtest.engine import BacktestEngine
from ..utils.metrics import PerformanceMetrics


@dataclass
class Strategy:
    """Trading strategy definition."""
    strategy_id: str
    name: str
    description: str
    strategy_type: str  # momentum, mean_reversion, quantum_enhanced, etc.
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"  # active, inactive, deprecated
    generation: int = 0
    parent_strategies: List[str] = field(default_factory=list)
    children_strategies: List[str] = field(default_factory=list)
    quantum_circuit: Optional[str] = None
    complexity_score: float = 0.0


@dataclass
class SystemState:
    """Overall system state."""
    total_portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    active_strategies: int = 0
    total_trades: int = 0
    system_health: float = 1.0
    last_rebalance: Optional[datetime] = None
    market_regime: str = "unknown"  # bull, bear, sideways, volatile


class ChiefArchitectAgent(BaseAgent):
    """
    Chief Architect Agent - The brain of the autonomous trading system.
    
    Responsibilities:
    1. Strategy Evolution: Continuously generate and optimize trading strategies
    2. Agent Orchestration: Coordinate all other agents
    3. Performance Monitoring: Track system-wide performance metrics
    4. Risk Management: High-level risk oversight
    5. Market Regime Detection: Identify and adapt to market conditions
    6. Quantum Integration: Leverage quantum computing for strategy optimization
    """
    
    def __init__(self, config: Config, agent_config: Dict[str, Any] = None):
        """Initialize Chief Architect Agent."""
        super().__init__("ChiefArchitect", config, agent_config)
        
        # Strategy management
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategies: List[str] = []
        self.strategy_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # System state
        self.system_state = SystemState()
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.backtest_engine = BacktestEngine(config)
        
        # Quantum integration
        self.quantum_generator = QuantumStrategyGenerator(config)
        
        # Evolution parameters
        self.evolution_rate = self.agent_config.get("strategy_evolution_rate", 0.1)
        self.max_strategies = self.agent_config.get("max_strategies_per_generation", 50)
        self.generation_size = 10
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        
        # Market regime detection
        self.market_regime_history: List[Dict[str, Any]] = []
        self.regime_detection_window = 30  # days
        
        # Strategy evaluation metrics
        self.evaluation_metrics = [
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", 
            "max_drawdown", "win_rate", "profit_factor"
        ]
        
        self.logger.info("Chief Architect Agent initialized")
    
    def _initialize_agent(self):
        """Initialize agent-specific components."""
        # Setup event handlers
        self.add_event_handler("strategy_performance_update", self._handle_strategy_performance)
        self.add_event_handler("market_data_update", self._handle_market_data_update)
        self.add_event_handler("risk_alert", self._handle_risk_alert)
        self.add_event_handler("agent_status_update", self._handle_agent_status)
        
        # Initialize default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default trading strategies."""
        default_strategies = [
            {
                "name": "Momentum_Strategy_v1",
                "description": "Basic momentum following strategy",
                "strategy_type": "momentum",
                "parameters": {
                    "lookback_period": 20,
                    "momentum_threshold": 0.02,
                    "position_size": 0.1
                }
            },
            {
                "name": "Mean_Reversion_v1", 
                "description": "Basic mean reversion strategy",
                "strategy_type": "mean_reversion",
                "parameters": {
                    "lookback_period": 50,
                    "std_dev_threshold": 2.0,
                    "position_size": 0.1
                }
            },
            {
                "name": "Quantum_Enhanced_v1",
                "description": "Quantum-enhanced portfolio optimization",
                "strategy_type": "quantum_enhanced",
                "parameters": {
                    "quantum_circuit": "qaoa_portfolio",
                    "qubits": 8,
                    "optimization_iterations": 100
                }
            }
        ]
        
        for strategy_def in default_strategies:
            strategy = Strategy(
                strategy_id=f"strategy_{len(self.strategies) + 1}",
                **strategy_def
            )
            self.strategies[strategy.strategy_id] = strategy
            self.active_strategies.append(strategy.strategy_id)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute Chief Architect tasks."""
        task_type = task.task_type
        data = task.data
        
        try:
            if task_type == "evolve_strategies":
                return await self._evolve_strategies(data)
            elif task_type == "evaluate_strategies":
                return await self._evaluate_strategies(data)
            elif task_type == "detect_market_regime":
                return await self._detect_market_regime(data)
            elif task_type == "optimize_portfolio":
                return await self._optimize_portfolio(data)
            elif task_type == "generate_quantum_strategy":
                return await self._generate_quantum_strategy(data)
            elif task_type == "system_health_check":
                return await self._system_health_check(data)
            elif task_type == "deploy_strategy":
                return await self._deploy_strategy(data)
            elif task_type == "retire_strategy":
                return await self._retire_strategy(data)
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _evolve_strategies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve trading strategies using genetic algorithm principles."""
        self.logger.info("Starting strategy evolution")
        
        # Get current strategy performance
        strategy_performance = await self._get_strategy_performance()
        
        # Select best strategies for evolution
        best_strategies = self._select_best_strategies(strategy_performance, top_k=5)
        
        # Generate new strategies through mutation and crossover
        new_strategies = []
        
        # Mutation
        for strategy_id in best_strategies:
            mutated_strategy = await self._mutate_strategy(strategy_id)
            if mutated_strategy:
                new_strategies.append(mutated_strategy)
        
        # Crossover
        if len(best_strategies) >= 2:
            for i in range(len(best_strategies) - 1):
                for j in range(i + 1, len(best_strategies)):
                    if np.random.random() < self.crossover_rate:
                        crossover_strategy = await self._crossover_strategies(
                            best_strategies[i], best_strategies[j]
                        )
                        if crossover_strategy:
                            new_strategies.append(crossover_strategy)
        
        # Generate quantum-enhanced strategies
        quantum_strategies = await self._generate_quantum_strategies(best_strategies)
        new_strategies.extend(quantum_strategies)
        
        # Evaluate new strategies
        evaluated_strategies = []
        for strategy in new_strategies:
            performance = await self._evaluate_strategy(strategy)
            strategy.performance_metrics = performance
            evaluated_strategies.append(strategy)
        
        # Select strategies to deploy
        deployed_strategies = self._select_strategies_for_deployment(evaluated_strategies)
        
        # Update strategy registry
        for strategy in deployed_strategies:
            self.strategies[strategy.strategy_id] = strategy
            self.active_strategies.append(strategy.strategy_id)
        
        # Retire underperforming strategies
        await self._retire_underperforming_strategies()
        
        self.logger.info(f"Strategy evolution completed. Generated {len(new_strategies)} new strategies, deployed {len(deployed_strategies)}")
        
        return {
            "status": "success",
            "strategies_generated": len(new_strategies),
            "strategies_deployed": len(deployed_strategies),
            "active_strategies": len(self.active_strategies)
        }
    
    async def _mutate_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Mutate an existing strategy to create a new variant."""
        if strategy_id not in self.strategies:
            return None
        
        parent_strategy = self.strategies[strategy_id]
        
        # Create mutated parameters
        mutated_params = parent_strategy.parameters.copy()
        
        for param, value in mutated_params.items():
            if np.random.random() < self.mutation_rate:
                if isinstance(value, (int, float)):
                    # Add random noise
                    noise = np.random.normal(0, 0.1 * abs(value))
                    mutated_params[param] = value + noise
                elif isinstance(value, str) and "quantum" in value.lower():
                    # Mutate quantum circuit
                    mutated_params[param] = f"{value}_mutated_{np.random.randint(1000)}"
        
        # Create new strategy
        new_strategy = Strategy(
            strategy_id=f"strategy_{len(self.strategies) + 1}",
            name=f"{parent_strategy.name}_mutated",
            description=f"Mutated version of {parent_strategy.name}",
            strategy_type=parent_strategy.strategy_type,
            parameters=mutated_params,
            generation=parent_strategy.generation + 1,
            parent_strategies=[strategy_id],
            complexity_score=parent_strategy.complexity_score * 1.1
        )
        
        return new_strategy
    
    async def _crossover_strategies(self, strategy_id_1: str, strategy_id_2: str) -> Optional[Strategy]:
        """Create a new strategy by combining two existing strategies."""
        if strategy_id_1 not in self.strategies or strategy_id_2 not in self.strategies:
            return None
        
        strategy_1 = self.strategies[strategy_id_1]
        strategy_2 = self.strategies[strategy_id_2]
        
        # Combine parameters
        combined_params = {}
        params_1 = strategy_1.parameters
        params_2 = strategy_2.parameters
        
        all_params = set(params_1.keys()) | set(params_2.keys())
        
        for param in all_params:
            if param in params_1 and param in params_2:
                # Average the values
                combined_params[param] = (params_1[param] + params_2[param]) / 2
            elif param in params_1:
                combined_params[param] = params_1[param]
            else:
                combined_params[param] = params_2[param]
        
        # Create new strategy
        new_strategy = Strategy(
            strategy_id=f"strategy_{len(self.strategies) + 1}",
            name=f"{strategy_1.name}_x_{strategy_2.name}",
            description=f"Crossover of {strategy_1.name} and {strategy_2.name}",
            strategy_type=strategy_1.strategy_type,  # Use type from first strategy
            parameters=combined_params,
            generation=max(strategy_1.generation, strategy_2.generation) + 1,
            parent_strategies=[strategy_id_1, strategy_id_2],
            complexity_score=(strategy_1.complexity_score + strategy_2.complexity_score) / 2
        )
        
        return new_strategy
    
    async def _generate_quantum_strategies(self, base_strategies: List[str]) -> List[Strategy]:
        """Generate quantum-enhanced versions of existing strategies."""
        quantum_strategies = []
        
        for strategy_id in base_strategies:
            if strategy_id in self.strategies:
                base_strategy = self.strategies[strategy_id]
                
                # Generate quantum-enhanced version
                quantum_strategy = await self.quantum_generator.enhance_strategy(base_strategy)
                
                if quantum_strategy:
                    quantum_strategies.append(quantum_strategy)
        
        return quantum_strategies
    
    async def _evaluate_strategy(self, strategy: Strategy) -> Dict[str, float]:
        """Evaluate a strategy using backtesting."""
        try:
            # Run backtest
            backtest_result = await self.backtest_engine.run_strategy_backtest(strategy)
            
            # Calculate performance metrics
            returns = pd.Series(backtest_result.get("returns", []))
            
            if len(returns) == 0:
                return {"sharpe_ratio": 0.0, "max_drawdown": 0.0}
            
            metrics = {
                "sharpe_ratio": self.performance_metrics.calculate_sharpe_ratio(returns),
                "sortino_ratio": self.performance_metrics.calculate_sortino_ratio(returns),
                "max_drawdown": self.performance_metrics.calculate_max_drawdown(returns.cumsum()),
                "win_rate": self._calculate_win_rate(backtest_result.get("trades", [])),
                "profit_factor": self._calculate_profit_factor(backtest_result.get("trades", [])),
                "total_return": returns.sum(),
                "volatility": returns.std() * np.sqrt(252)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy {strategy.strategy_id}: {str(e)}")
            return {"sharpe_ratio": 0.0, "max_drawdown": 0.0}
    
    def _select_best_strategies(self, performance: Dict[str, Dict[str, float]], top_k: int = 5) -> List[str]:
        """Select the best performing strategies."""
        # Calculate composite score
        strategy_scores = []
        
        for strategy_id, metrics in performance.items():
            if strategy_id in self.strategies:
                score = self._calculate_strategy_score(metrics)
                strategy_scores.append((strategy_id, score))
        
        # Sort by score and return top k
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        return [strategy_id for strategy_id, _ in strategy_scores[:top_k]]
    
    def _calculate_strategy_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score for a strategy."""
        weights = {
            "sharpe_ratio": 0.3,
            "sortino_ratio": 0.2,
            "calmar_ratio": 0.2,
            "win_rate": 0.15,
            "profit_factor": 0.15
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0.0)
            score += value * weight
        
        return score
    
    def _select_strategies_for_deployment(self, strategies: List[Strategy]) -> List[Strategy]:
        """Select which strategies to deploy based on performance and diversity."""
        # Filter strategies that meet minimum performance criteria
        qualified_strategies = [
            s for s in strategies 
            if s.performance_metrics.get("sharpe_ratio", 0) > 0.5
            and s.performance_metrics.get("max_drawdown", 1) < 0.2
        ]
        
        # Sort by performance
        qualified_strategies.sort(
            key=lambda s: self._calculate_strategy_score(s.performance_metrics),
            reverse=True
        )
        
        # Select top strategies while maintaining diversity
        deployed_strategies = []
        strategy_types = set()
        
        for strategy in qualified_strategies:
            if len(deployed_strategies) >= self.generation_size:
                break
            
            # Ensure diversity in strategy types
            if strategy.strategy_type not in strategy_types or len(strategy_types) >= 3:
                deployed_strategies.append(strategy)
                strategy_types.add(strategy.strategy_type)
        
        return deployed_strategies
    
    async def _retire_underperforming_strategies(self):
        """Retire strategies that are consistently underperforming."""
        performance_threshold = 0.3  # Minimum Sharpe ratio
        
        strategies_to_retire = []
        
        for strategy_id in self.active_strategies:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                
                # Check if strategy has been active long enough
                if (datetime.utcnow() - strategy.created_at).days < 30:
                    continue
                
                # Check performance
                sharpe = strategy.performance_metrics.get("sharpe_ratio", 0)
                if sharpe < performance_threshold:
                    strategies_to_retire.append(strategy_id)
        
        # Retire strategies
        for strategy_id in strategies_to_retire:
            if strategy_id in self.strategies:
                self.strategies[strategy_id].status = "deprecated"
                self.active_strategies.remove(strategy_id)
                self.logger.info(f"Retired underperforming strategy: {strategy_id}")
    
    async def _detect_market_regime(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime using various indicators."""
        market_data = data.get("market_data", {})
        
        if not market_data:
            return {"status": "error", "message": "No market data provided"}
        
        try:
            # Calculate market regime indicators
            volatility = self._calculate_market_volatility(market_data)
            trend_strength = self._calculate_trend_strength(market_data)
            momentum = self._calculate_market_momentum(market_data)
            
            # Determine market regime
            regime = self._classify_market_regime(volatility, trend_strength, momentum)
            
            # Update system state
            self.system_state.market_regime = regime
            
            # Store regime history
            regime_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "regime": regime,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "momentum": momentum
            }
            self.market_regime_history.append(regime_data)
            
            # Keep only recent history
            if len(self.market_regime_history) > self.regime_detection_window:
                self.market_regime_history = self.market_regime_history[-self.regime_detection_window:]
            
            return {
                "status": "success",
                "market_regime": regime,
                "indicators": {
                    "volatility": volatility,
                    "trend_strength": trend_strength,
                    "momentum": momentum
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility."""
        # Implementation would use actual market data
        return np.random.uniform(0.1, 0.3)  # Placeholder
    
    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate trend strength."""
        # Implementation would use actual market data
        return np.random.uniform(-1.0, 1.0)  # Placeholder
    
    def _calculate_market_momentum(self, market_data: Dict[str, Any]) -> float:
        """Calculate market momentum."""
        # Implementation would use actual market data
        return np.random.uniform(-1.0, 1.0)  # Placeholder
    
    def _classify_market_regime(self, volatility: float, trend_strength: float, momentum: float) -> str:
        """Classify market regime based on indicators."""
        if volatility > 0.25:
            return "volatile"
        elif trend_strength > 0.5:
            return "bull"
        elif trend_strength < -0.5:
            return "bear"
        else:
            return "sideways"
    
    async def _get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all active strategies."""
        performance = {}
        
        for strategy_id in self.active_strategies:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                performance[strategy_id] = strategy.performance_metrics
        
        return performance
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor from trades."""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) > 0)
        gross_loss = abs(sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent-specific metrics."""
        return {
            "active_strategies": len(self.active_strategies),
            "total_strategies": len(self.strategies),
            "market_regime": self.system_state.market_regime,
            "system_health": self.system_state.system_health,
            "evolution_generation": max([s.generation for s in self.strategies.values()], default=0)
        }
    
    # Event handlers
    async def _handle_strategy_performance(self, message: Dict[str, Any]):
        """Handle strategy performance updates."""
        strategy_id = message.get("strategy_id")
        performance = message.get("performance", {})
        
        if strategy_id in self.strategies:
            self.strategies[strategy_id].performance_metrics = performance
            self.strategies[strategy_id].last_updated = datetime.utcnow()
    
    async def _handle_market_data_update(self, message: Dict[str, Any]):
        """Handle market data updates."""
        # Trigger market regime detection
        task_id = self.add_task("detect_market_regime", {"market_data": message.get("data", {})})
        self.logger.info(f"Added market regime detection task: {task_id}")
    
    async def _handle_risk_alert(self, message: Dict[str, Any]):
        """Handle risk alerts."""
        alert_type = message.get("alert_type")
        details = message.get("details", {})
        
        self.logger.warning(f"Risk alert received: {alert_type}", **details)
        
        # Take appropriate action based on alert type
        if alert_type == "high_drawdown":
            await self._handle_high_drawdown_alert(details)
        elif alert_type == "concentration_risk":
            await self._handle_concentration_risk_alert(details)
    
    async def _handle_agent_status(self, message: Dict[str, Any]):
        """Handle agent status updates."""
        agent_name = message.get("agent_name")
        status = message.get("status")
        
        self.logger.info(f"Agent status update: {agent_name} -> {status}")
    
    async def _handle_high_drawdown_alert(self, details: Dict[str, Any]):
        """Handle high drawdown alert."""
        # Reduce position sizes or stop trading
        self.logger.warning("Handling high drawdown alert - reducing risk exposure")
    
    async def _handle_concentration_risk_alert(self, details: Dict[str, Any]):
        """Handle concentration risk alert."""
        # Rebalance portfolio to reduce concentration
        self.logger.warning("Handling concentration risk alert - rebalancing portfolio")