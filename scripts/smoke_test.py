#!/usr/bin/env python3
"""
Comprehensive smoke test for Infinite_Money autonomous trading system.

This script validates:
1. Configuration loading and validation
2. Agent initialization and basic functionality
3. Quantum computing integration
4. Strategy generation and evaluation
5. Performance metrics calculation
6. System orchestration
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinite_money.utils.config import Config
from infinite_money.utils.logger import setup_logger, get_logger
from infinite_money.utils.metrics import PerformanceMetrics
from infinite_money.agents import (
    ChiefArchitectAgent,
    DataEngineerAgent,
    AlphaResearcherAgent,
    PortfolioManagerAgent,
    ExecutionTraderAgent,
    ComplianceOfficerAgent
)
from infinite_money.quantum.strategy_generator import QuantumStrategyGenerator
from infinite_money.backtest.engine import BacktestEngine


class SmokeTest:
    """Comprehensive smoke test for Infinite_Money system."""
    
    def __init__(self):
        """Initialize smoke test."""
        self.logger = get_logger("SmokeTest")
        self.test_results = {}
        self.start_time = time.time()
        
    async def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        self.logger.info("🚀 Starting Infinite_Money Smoke Test Suite")
        print("=" * 80)
        print("🧪 INFINITE_MONEY SMOKE TEST SUITE")
        print("=" * 80)
        
        tests = [
            ("Configuration", self.test_configuration),
            ("Logging", self.test_logging),
            ("Performance Metrics", self.test_performance_metrics),
            ("Quantum Integration", self.test_quantum_integration),
            ("Agent Initialization", self.test_agent_initialization),
            ("Strategy Generation", self.test_strategy_generation),
            ("Backtest Engine", self.test_backtest_engine),
            ("System Integration", self.test_system_integration),
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n📋 Running Test: {test_name}")
            print("-" * 50)
            
            try:
                result = await test_func()
                self.test_results[test_name] = result
                
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    all_passed = False
                    
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {str(e)}")
                self.test_results[test_name] = False
                all_passed = False
        
        # Print summary
        self._print_summary()
        
        return all_passed
    
    async def test_configuration(self) -> bool:
        """Test configuration loading and validation."""
        try:
            print("Testing configuration loading...")
            
            # Test config loading
            config = Config("infinite_money/configs/main_config.yaml")
            print(f"  ✓ Configuration loaded from: {config.config_path}")
            
            # Test validation
            config.validate_config()
            print(f"  ✓ Configuration validation passed")
            
            # Test environment overrides
            config.apply_env_overrides()
            print(f"  ✓ Environment overrides applied")
            
            # Test agent config
            agent_config = config.get_agent_config("chief_architect")
            print(f"  ✓ Agent configuration retrieved: {len(agent_config)} settings")
            
            # Test quantum config
            quantum_config = config.get_quantum_config()
            print(f"  ✓ Quantum configuration: {quantum_config.backend} backend")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Configuration test failed: {str(e)}")
            return False
    
    async def test_logging(self) -> bool:
        """Test logging system."""
        try:
            print("Testing logging system...")
            
            # Setup logger
            config = Config("infinite_money/configs/main_config.yaml")
            logger_system = setup_logger(config.system.dict())
            print(f"  ✓ Logger system initialized")
            
            # Test structured logging
            logger = get_logger("SmokeTest")
            logger.info("Test log message", test_data={"key": "value"})
            print(f"  ✓ Structured logging functional")
            
            # Test different log levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            print(f"  ✓ All log levels functional")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Logging test failed: {str(e)}")
            return False
    
    async def test_performance_metrics(self) -> bool:
        """Test performance metrics calculation."""
        try:
            print("Testing performance metrics...")
            
            metrics = PerformanceMetrics()
            
            # Generate sample returns
            import numpy as np
            import pandas as pd
            
            np.random.seed(42)
            returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
            
            # Test Sharpe ratio
            sharpe = metrics.calculate_sharpe_ratio(returns)
            print(f"  ✓ Sharpe ratio calculated: {sharpe:.3f}")
            
            # Test Sortino ratio
            sortino = metrics.calculate_sortino_ratio(returns)
            print(f"  ✓ Sortino ratio calculated: {sortino:.3f}")
            
            # Test maximum drawdown
            cumulative = (1 + returns).cumprod()
            max_dd = metrics.calculate_max_drawdown(cumulative)
            print(f"  ✓ Maximum drawdown calculated: {max_dd:.3f}")
            
            # Test VaR
            var_95 = metrics.calculate_var(returns, alpha=0.05)
            print(f"  ✓ VaR (95%) calculated: {var_95:.3f}")
            
            # Test CVaR
            cvar_95 = metrics.calculate_cvar(returns, alpha=0.05)
            print(f"  ✓ CVaR (95%) calculated: {cvar_95:.3f}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Performance metrics test failed: {str(e)}")
            return False
    
    async def test_quantum_integration(self) -> bool:
        """Test quantum computing integration."""
        try:
            print("Testing quantum integration...")
            
            config = Config("infinite_money/configs/main_config.yaml")
            
            # Test quantum strategy generator
            quantum_generator = QuantumStrategyGenerator(config)
            print(f"  ✓ Quantum strategy generator initialized")
            
            # Test quantum circuit creation
            circuit = await quantum_generator.create_quantum_circuit("qaoa", 4)
            print(f"  ✓ Quantum circuit created: {circuit.name if hasattr(circuit, 'name') else 'QAOA'}")
            
            # Test quantum optimization
            result = await quantum_generator.optimize_portfolio([0.25, 0.25, 0.25, 0.25])
            print(f"  ✓ Quantum portfolio optimization completed")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Quantum integration test failed: {str(e)}")
            return False
    
    async def test_agent_initialization(self) -> bool:
        """Test agent initialization."""
        try:
            print("Testing agent initialization...")
            
            config = Config("infinite_money/configs/main_config.yaml")
            
            # Test Chief Architect Agent
            chief_agent = ChiefArchitectAgent(config, {})
            print(f"  ✓ Chief Architect Agent initialized")
            
            # Test Data Engineer Agent
            data_agent = DataEngineerAgent(config, {})
            print(f"  ✓ Data Engineer Agent initialized")
            
            # Test Alpha Researcher Agent
            alpha_agent = AlphaResearcherAgent(config, {})
            print(f"  ✓ Alpha Researcher Agent initialized")
            
            # Test Portfolio Manager Agent
            portfolio_agent = PortfolioManagerAgent(config, {})
            print(f"  ✓ Portfolio Manager Agent initialized")
            
            # Test Execution Trader Agent
            execution_agent = ExecutionTraderAgent(config, {})
            print(f"  ✓ Execution Trader Agent initialized")
            
            # Test Compliance Officer Agent
            compliance_agent = ComplianceOfficerAgent(config, {})
            print(f"  ✓ Compliance Officer Agent initialized")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Agent initialization test failed: {str(e)}")
            return False
    
    async def test_strategy_generation(self) -> bool:
        """Test strategy generation and evolution."""
        try:
            print("Testing strategy generation...")
            
            config = Config("infinite_money/configs/main_config.yaml")
            chief_agent = ChiefArchitectAgent(config, {})
            
            # Test strategy evolution
            evolution_result = await chief_agent._evolve_strategies({})
            print(f"  ✓ Strategy evolution completed: {evolution_result.get('strategies_generated', 0)} strategies")
            
            # Test strategy evaluation
            if chief_agent.strategies:
                strategy_id = list(chief_agent.strategies.keys())[0]
                strategy = chief_agent.strategies[strategy_id]
                performance = await chief_agent._evaluate_strategy(strategy)
                print(f"  ✓ Strategy evaluation completed: Sharpe = {performance.get('sharpe_ratio', 0):.3f}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Strategy generation test failed: {str(e)}")
            return False
    
    async def test_backtest_engine(self) -> bool:
        """Test backtest engine."""
        try:
            print("Testing backtest engine...")
            
            config = Config("infinite_money/configs/main_config.yaml")
            backtest_engine = BacktestEngine(config)
            print(f"  ✓ Backtest engine initialized")
            
            # Test backtest configuration
            from infinite_money.backtest.config import BacktestConfig
            
            backtest_config = BacktestConfig(
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_capital=100000.0,
                symbols=["SPY", "QQQ"],
                strategy_name="smoke_test"
            )
            print(f"  ✓ Backtest configuration created")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Backtest engine test failed: {str(e)}")
            return False
    
    async def test_system_integration(self) -> bool:
        """Test system integration."""
        try:
            print("Testing system integration...")
            
            # Test full system initialization
            from infinite_money.main import AutonomousTradingSystem
            
            system = AutonomousTradingSystem("infinite_money/configs/main_config.yaml", 1000000.0)
            print(f"  ✓ Autonomous trading system initialized")
            
            # Test system status
            status = system.get_system_status()
            print(f"  ✓ System status retrieved: {status['config']['mode']} mode")
            
            return True
            
        except Exception as e:
            print(f"  ✗ System integration test failed: {str(e)}")
            return False
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 SMOKE TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if not result:
                    print(f"  - {test_name}")
        
        runtime = time.time() - self.start_time
        print(f"\n⏱️  Total Runtime: {runtime:.2f} seconds")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix issues.")


async def main():
    """Main smoke test entry point."""
    smoke_test = SmokeTest()
    
    try:
        success = await smoke_test.run_all_tests()
        
        if success:
            print("\n✅ Smoke test completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Smoke test failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Smoke test crashed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())