"""
Infinite Money Engine - Main Strategy Class
A comprehensive algorithmic trading system with multiple alpha models and risk management.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from QuantConnect import *
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data import *
from QuantConnect.Data.Market import TradeBar
from QuantConnect.Indicators import *
from QuantConnect.Securities import *

# Import our custom modules
from alpha_models.stat_arb import StatisticalArbitrage
from alpha_models.factors import FactorModel
from alpha_models.regime import RegimeDetector
from alpha_models.ensemble import EnsembleModel
from portfolio.optimizer import PortfolioOptimizer
from portfolio.sizing import KellyCriterion
from risk.limits import RiskLimits
from risk.stress import StressTester
from exec.routes import ExecutionRouter
from exec.costs import CostModel
from data.insider import InsiderDataProcessor
from data.utils import DataUtils
from ai.sentiment_client import SentimentClient
from ai.rl_allocator import RLAllocator


class InfiniteMoneyEngine(QCAlgorithm):
    """
    Main strategy class implementing the Infinite Money Engine.
    
    Features:
    - Multi-model alpha generation
    - Kelly criterion position sizing
    - Dynamic regime detection
    - Comprehensive risk management
    - Alternative data integration
    - AI-powered sentiment analysis
    - Reinforcement learning allocation
    """
    
    def Initialize(self):
        """Initialize the algorithm."""
        # Set basic parameters
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1000000)
        self.SetBenchmark("SPY")
        
        # Set resolution and warmup
        self.SetWarmup(252)  # 1 year of data
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted
        
        # Initialize universe
        self.symbols = self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        
        # Initialize alpha models
        self.stat_arb = StatisticalArbitrage()
        self.factor_model = FactorModel()
        self.regime_detector = RegimeDetector()
        self.ensemble = EnsembleModel()
        
        # Initialize portfolio components
        self.optimizer = PortfolioOptimizer()
        self.kelly = KellyCriterion()
        
        # Initialize risk management
        self.risk_limits = RiskLimits()
        self.stress_tester = StressTester()
        
        # Initialize execution
        self.exec_router = ExecutionRouter()
        self.cost_model = CostModel()
        
        # Initialize data processors
        self.insider_processor = InsiderDataProcessor()
        self.data_utils = DataUtils()
        
        # Initialize AI components
        self.sentiment_client = SentimentClient()
        self.rl_allocator = RLAllocator()
        
        # State variables
        self.positions = {}
        self.signals = {}
        self.regime = "normal"
        self.last_rebalance = None
        self.rebalance_frequency = timedelta(days=5)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.RebalancePortfolio)
        
        # Schedule regime detection
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.BeforeMarketClose("SPY", 60), 
                        self.UpdateRegime)
    
    def CoarseSelectionFunction(self, coarse):
        """First pass universe selection."""
        # Filter for liquid stocks
        filtered = [x for x in coarse if x.HasFundamentalData and 
                   x.Price > 10 and x.Volume > 1000000]
        
        # Sort by market cap and return top 500
        sorted_by_market_cap = sorted(filtered, key=lambda x: x.MarketCap, reverse=True)
        return [x.Symbol for x in sorted_by_market_cap[:500]]
    
    def FineSelectionFunction(self, fine):
        """Second pass universe selection."""
        # Filter for quality metrics
        quality_stocks = []
        
        for stock in fine:
            # Basic quality filters
            if (stock.FinancialStatements.BalanceSheet.TotalEquity.Value > 0 and
                stock.FinancialStatements.IncomeStatement.TotalRevenue.Value > 0 and
                stock.ValuationRatios.PERatio > 0 and stock.ValuationRatios.PERatio < 50):
                
                quality_stocks.append(stock.Symbol)
        
        return quality_stocks[:100]  # Return top 100 stocks
    
    def OnData(self, data):
        """Main data processing function."""
        if self.IsWarmingUp:
            return
        
        # Update regime detection
        self.regime = self.regime_detector.detect_regime(data)
        
        # Generate alpha signals
        self.signals = self.generate_signals(data)
        
        # Update sentiment data
        self.update_sentiment_data()
        
        # Update insider data
        self.update_insider_data()
    
    def generate_signals(self, data) -> Dict[str, float]:
        """Generate alpha signals from all models."""
        signals = {}
        
        # Statistical arbitrage signals
        stat_arb_signals = self.stat_arb.generate_signals(data)
        signals.update(stat_arb_signals)
        
        # Factor model signals
        factor_signals = self.factor_model.generate_signals(data)
        signals.update(factor_signals)
        
        # Ensemble signals
        ensemble_signals = self.ensemble.generate_signals(data, self.regime)
        signals.update(ensemble_signals)
        
        return signals
    
    def update_sentiment_data(self):
        """Update sentiment analysis data."""
        try:
            # Get sentiment for current symbols
            symbols = list(self.signals.keys())
            sentiment_data = self.sentiment_client.get_sentiment(symbols)
            
            # Incorporate sentiment into signals
            for symbol, sentiment in sentiment_data.items():
                if symbol in self.signals:
                    self.signals[symbol] *= (1 + sentiment * 0.1)  # 10% sentiment weight
        except Exception as e:
            self.logger.warning(f"Failed to update sentiment data: {e}")
    
    def update_insider_data(self):
        """Update insider trading data."""
        try:
            # Get insider data for current symbols
            symbols = list(self.signals.keys())
            insider_data = self.insider_processor.get_insider_signals(symbols)
            
            # Incorporate insider signals
            for symbol, insider_signal in insider_data.items():
                if symbol in self.signals:
                    self.signals[symbol] *= (1 + insider_signal * 0.05)  # 5% insider weight
        except Exception as e:
            self.logger.warning(f"Failed to update insider data: {e}")
    
    def RebalancePortfolio(self):
        """Main portfolio rebalancing function."""
        if self.last_rebalance and (self.Time - self.last_rebalance) < self.rebalance_frequency:
            return
        
        try:
            # Get current portfolio state
            current_positions = self.get_current_positions()
            
            # Optimize portfolio weights
            target_weights = self.optimizer.optimize(
                self.signals, 
                current_positions, 
                self.regime
            )
            
            # Apply Kelly criterion sizing
            kelly_weights = self.kelly.calculate_sizes(
                target_weights, 
                self.signals, 
                self.regime
            )
            
            # Apply risk limits
            final_weights = self.risk_limits.apply_limits(kelly_weights)
            
            # Execute trades
            self.execute_trades(final_weights)
            
            # Update state
            self.last_rebalance = self.Time
            self.positions = final_weights
            
        except Exception as e:
            self.logger.error(f"Rebalancing failed: {e}")
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current portfolio positions."""
        positions = {}
        for symbol in self.signals.keys():
            if self.Portfolio[symbol].Invested:
                positions[symbol] = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
            else:
                positions[symbol] = 0.0
        return positions
    
    def execute_trades(self, target_weights: Dict[str, float]):
        """Execute trades to reach target weights."""
        current_weights = self.get_current_positions()
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            
            if abs(target_weight - current_weight) > 0.01:  # 1% threshold
                # Calculate trade size
                trade_value = (target_weight - current_weight) * self.Portfolio.TotalPortfolioValue
                
                # Apply execution costs
                trade_value = self.cost_model.apply_costs(trade_value, symbol)
                
                # Execute trade
                if trade_value > 0:
                    self.MarketOrder(symbol, int(trade_value / self.Securities[symbol].Price))
                elif trade_value < 0:
                    self.MarketOrder(symbol, int(trade_value / self.Securities[symbol].Price))
    
    def UpdateRegime(self):
        """Update market regime detection."""
        try:
            # Get market data for regime detection
            market_data = self.get_market_data()
            
            # Update regime detector
            self.regime = self.regime_detector.detect_regime(market_data)
            
            # Log regime change
            self.logger.info(f"Market regime: {self.regime}")
            
        except Exception as e:
            self.logger.error(f"Regime update failed: {e}")
    
    def get_market_data(self) -> pd.DataFrame:
        """Get market data for regime detection."""
        # Get SPY data for market regime detection
        spy_data = self.History("SPY", 252, Resolution.Daily)
        return spy_data
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events."""
        if orderEvent.Status == OrderStatus.Filled:
            self.logger.info(f"Order filled: {orderEvent.Symbol} - {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")
    
    def OnEndOfAlgorithm(self):
        """Clean up at end of algorithm."""
        # Save final results
        self.logger.info("Algorithm completed")
        
        # Generate performance report
        self.generate_performance_report()
    
    def generate_performance_report(self):
        """Generate performance report."""
        # Calculate key metrics
        total_return = (self.Portfolio.TotalPortfolioValue - self.Portfolio.TotalPortfolioValue) / self.Portfolio.TotalPortfolioValue
        sharpe_ratio = self.Portfolio.TotalUnrealizedProfit / self.Portfolio.TotalUnrealizedProfit if self.Portfolio.TotalUnrealizedProfit != 0 else 0
        
        self.logger.info(f"Total Return: {total_return:.2%}")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
