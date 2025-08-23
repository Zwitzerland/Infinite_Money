"""
Main entry point for Infinite_Money autonomous trading system.
"""

import asyncio
import argparse
import signal
import sys
from typing import Dict, Any, Optional
from pathlib import Path

from .utils.config import Config
from .utils.logger import setup_logger, get_logger
from .agents import (
    ChiefArchitectAgent,
    DataEngineerAgent,
    AlphaResearcherAgent,
    PortfolioManagerAgent,
    ExecutionTraderAgent,
    ComplianceOfficerAgent,
    AgentOrchestrator
)


class AutonomousTradingSystem:
    """
    Main autonomous trading system that orchestrates all agents.
    
    This is the primary interface for running the Infinite_Money system
    in autonomous mode, with full quantum computing integration and
    continuous strategy evolution.
    """
    
    def __init__(self, config_path: Optional[str] = None, initial_capital: float = 1000000.0):
        """Initialize the autonomous trading system."""
        self.config_path = config_path or "configs/main_config.yaml"
        self.initial_capital = initial_capital
        
        # Load configuration
        self.config = Config(self.config_path)
        self.config.apply_env_overrides()
        self.config.validate_config()
        
        # Setup logging
        self.logger_system = setup_logger(self.config.system.dict())
        self.logger = get_logger("AutonomousTradingSystem")
        
        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self.agent_orchestrator: Optional[AgentOrchestrator] = None
        
        # System state
        self.is_running = False
        self.start_time = None
        
        self.logger.info("Autonomous Trading System initialized")
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing Autonomous Trading System...")
            
            # Initialize agents based on configuration
            await self._initialize_agents()
            
            # Initialize agent orchestrator
            self.agent_orchestrator = AgentOrchestrator(self.agents, self.config)
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            return False
    
    async def _initialize_agents(self):
        """Initialize all enabled agents."""
        agent_configs = self.config.agents
        
        # Chief Architect (always enabled)
        if self.config.is_agent_enabled("chief_architect"):
            self.agents["chief_architect"] = ChiefArchitectAgent(
                self.config, 
                agent_configs.get("chief_architect", {})
            )
        
        # Data Engineer
        if self.config.is_agent_enabled("data_engineer"):
            self.agents["data_engineer"] = DataEngineerAgent(
                self.config,
                agent_configs.get("data_engineer", {})
            )
        
        # Alpha Researcher
        if self.config.is_agent_enabled("alpha_researcher"):
            self.agents["alpha_researcher"] = AlphaResearcherAgent(
                self.config,
                agent_configs.get("alpha_researcher", {})
            )
        
        # Portfolio Manager
        if self.config.is_agent_enabled("portfolio_manager"):
            self.agents["portfolio_manager"] = PortfolioManagerAgent(
                self.config,
                agent_configs.get("portfolio_manager", {})
            )
        
        # Execution Trader
        if self.config.is_agent_enabled("execution_trader"):
            self.agents["execution_trader"] = ExecutionTraderAgent(
                self.config,
                agent_configs.get("execution_trader", {})
            )
        
        # Compliance Officer
        if self.config.is_agent_enabled("compliance_officer"):
            self.agents["compliance_officer"] = ComplianceOfficerAgent(
                self.config,
                agent_configs.get("compliance_officer", {})
            )
        
        self.logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self) -> bool:
        """Start the autonomous trading system."""
        try:
            self.logger.info("Starting Autonomous Trading System...")
            
            # Start all agents
            start_tasks = []
            for agent_name, agent in self.agents.items():
                self.logger.info(f"Starting agent: {agent_name}")
                start_tasks.append(agent.start())
            
            # Wait for all agents to start
            start_results = await asyncio.gather(*start_tasks, return_exceptions=True)
            
            # Check for startup failures
            failed_agents = []
            for i, (agent_name, result) in enumerate(zip(self.agents.keys(), start_results)):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to start {agent_name}: {str(result)}")
                    failed_agents.append(agent_name)
                elif not result:
                    self.logger.error(f"Agent {agent_name} failed to start")
                    failed_agents.append(agent_name)
            
            if failed_agents:
                self.logger.error(f"Failed to start agents: {failed_agents}")
                return False
            
            # Start agent orchestrator
            if self.agent_orchestrator:
                await self.agent_orchestrator.start()
            
            self.is_running = True
            self.start_time = asyncio.get_event_loop().time()
            
            self.logger.info("Autonomous Trading System started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {str(e)}")
            return False
    
    async def run_autonomous_session(self, duration_hours: int = 24, 
                                   risk_limit: float = 0.02, 
                                   target_sharpe: float = 1.5) -> bool:
        """Run an autonomous trading session."""
        try:
            self.logger.info(f"Starting autonomous session for {duration_hours} hours")
            self.logger.info(f"Risk limit: {risk_limit}, Target Sharpe: {target_sharpe}")
            
            # Set session parameters
            session_config = {
                "duration_hours": duration_hours,
                "risk_limit": risk_limit,
                "target_sharpe": target_sharpe,
                "initial_capital": self.initial_capital
            }
            
            # Start the main trading loop
            await self._run_trading_loop(session_config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in autonomous session: {str(e)}")
            return False
    
    async def _run_trading_loop(self, session_config: Dict[str, Any]):
        """Main trading loop for autonomous operation."""
        duration_seconds = session_config["duration_hours"] * 3600
        start_time = asyncio.get_event_loop().time()
        
        while self.is_running:
            try:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                # Check if session duration exceeded
                if elapsed >= duration_seconds:
                    self.logger.info("Session duration reached, stopping trading loop")
                    break
                
                # Execute trading cycle
                await self._execute_trading_cycle(session_config)
                
                # Wait for next cycle
                await asyncio.sleep(1)  # 1 second cycle
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _execute_trading_cycle(self, session_config: Dict[str, Any]):
        """Execute a single trading cycle."""
        try:
            # 1. Market data collection
            await self._collect_market_data()
            
            # 2. Strategy execution
            await self._execute_strategies()
            
            # 3. Portfolio management
            await self._manage_portfolio()
            
            # 4. Risk monitoring
            await self._monitor_risk(session_config["risk_limit"])
            
            # 5. Performance evaluation
            await self._evaluate_performance(session_config["target_sharpe"])
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")
    
    async def _collect_market_data(self):
        """Collect market data from all sources."""
        if "data_engineer" in self.agents:
            # Trigger data collection
            task_id = self.agents["data_engineer"].add_task(
                "collect_market_data",
                {"sources": ["yahoo", "alpha_vantage", "polygon"]}
            )
            self.logger.debug(f"Added market data collection task: {task_id}")
    
    async def _execute_strategies(self):
        """Execute active trading strategies."""
        if "chief_architect" in self.agents:
            # Trigger strategy execution
            task_id = self.agents["chief_architect"].add_task(
                "execute_strategies",
                {"active_strategies": True}
            )
            self.logger.debug(f"Added strategy execution task: {task_id}")
    
    async def _manage_portfolio(self):
        """Manage portfolio allocation and rebalancing."""
        if "portfolio_manager" in self.agents:
            # Trigger portfolio management
            task_id = self.agents["portfolio_manager"].add_task(
                "manage_portfolio",
                {"rebalance": True}
            )
            self.logger.debug(f"Added portfolio management task: {task_id}")
    
    async def _monitor_risk(self, risk_limit: float):
        """Monitor and manage risk."""
        if "compliance_officer" in self.agents:
            # Trigger risk monitoring
            task_id = self.agents["compliance_officer"].add_task(
                "monitor_risk",
                {"risk_limit": risk_limit}
            )
            self.logger.debug(f"Added risk monitoring task: {task_id}")
    
    async def _evaluate_performance(self, target_sharpe: float):
        """Evaluate system performance."""
        if "chief_architect" in self.agents:
            # Trigger performance evaluation
            task_id = self.agents["chief_architect"].add_task(
                "evaluate_performance",
                {"target_sharpe": target_sharpe}
            )
            self.logger.debug(f"Added performance evaluation task: {task_id}")
    
    async def shutdown(self) -> bool:
        """Shutdown the system gracefully."""
        try:
            self.logger.info("Initiating system shutdown...")
            
            self.is_running = False
            
            # Stop agent orchestrator
            if self.agent_orchestrator:
                await self.agent_orchestrator.stop()
            
            # Stop all agents
            stop_tasks = []
            for agent_name, agent in self.agents.items():
                self.logger.info(f"Stopping agent: {agent_name}")
                stop_tasks.append(agent.stop())
            
            # Wait for all agents to stop
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            # Calculate runtime
            runtime = 0
            if self.start_time:
                runtime = asyncio.get_event_loop().time() - self.start_time
            
            self.logger.info(f"System shutdown completed. Runtime: {runtime:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        agent_statuses = {}
        for agent_name, agent in self.agents.items():
            agent_statuses[agent_name] = {
                "status": agent.state.status,
                "tasks_completed": agent.state.tasks_completed,
                "tasks_failed": agent.state.tasks_failed,
                "is_healthy": agent.is_healthy()
            }
        
        return {
            "is_running": self.is_running,
            "start_time": self.start_time,
            "agents": agent_statuses,
            "config": {
                "mode": self.config.system.mode,
                "quantum_enabled": self.config.system.quantum_enabled,
                "initial_capital": self.initial_capital
            }
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Infinite_Money Autonomous Trading System")
    parser.add_argument("--config", type=str, default="configs/main_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="autonomous",
                       choices=["autonomous", "supervised", "backtest", "research"],
                       help="System operation mode")
    parser.add_argument("--capital", type=float, default=1000000.0,
                       help="Initial capital")
    parser.add_argument("--duration", type=int, default=24,
                       help="Session duration in hours")
    parser.add_argument("--risk-limit", type=float, default=0.02,
                       help="Risk limit (fraction of capital)")
    parser.add_argument("--target-sharpe", type=float, default=1.5,
                       help="Target Sharpe ratio")
    
    args = parser.parse_args()
    
    # Create and initialize system
    system = AutonomousTradingSystem(args.config, args.capital)
    
    # Override mode if specified
    if args.mode != "autonomous":
        system.config.system.mode = args.mode
    
    # Initialize system
    if not await system.initialize():
        print("Failed to initialize system")
        sys.exit(1)
    
    # Start system
    if not await system.start():
        print("Failed to start system")
        sys.exit(1)
    
    try:
        # Run autonomous session
        if args.mode == "autonomous":
            success = await system.run_autonomous_session(
                duration_hours=args.duration,
                risk_limit=args.risk_limit,
                target_sharpe=args.target_sharpe
            )
        else:
            # For other modes, just keep running
            while system.is_running:
                await asyncio.sleep(1)
            success = True
        
        if not success:
            print("Session failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        # Shutdown system
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())