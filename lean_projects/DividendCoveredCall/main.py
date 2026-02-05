# ruff: noqa: F403, F405
from AlgorithmImports import *


class DividendCoveredCall(QCAlgorithm):
    def Initialize(self) -> None:
        symbol = self.get_parameter("symbol", "SPY")
        self.lookback = int(self.get_parameter("lookback", 20))
        self.delta_target = float(self.get_parameter("delta", 0.30))
        self.dte_min = int(self.get_parameter("dte_min", 7))
        self.dte_max = int(self.get_parameter("dte_max", 30))
        self.max_drawdown = float(self.get_parameter("max_drawdown", 0.25))

        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000)
        self.SetBenchmark(symbol)

        self.equity = self.AddEquity(symbol, Resolution.Daily)
        self.option = self.AddOption(symbol)
        self.option.SetFilter(self._option_filter)
        self.option.PriceModel = OptionPriceModels.CrankNicolsonFD()

        self.sma = self.SMA(symbol, self.lookback, Resolution.Daily)
        self.SetWarmUp(self.lookback, Resolution.Daily)

        self.last_chain = None
        self.high_water = self.Portfolio.TotalPortfolioValue
        self.trading_enabled = True

        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen(symbol, 30),
            self.Rebalance,
        )

    def _option_filter(self, universe: OptionFilterUniverse) -> OptionFilterUniverse:
        return universe.IncludeWeeklys().Strikes(-5, 5).Expiration(
            self.dte_min, self.dte_max
        )

    def OnData(self, slice: Slice) -> None:
        chain = slice.OptionChains.get(self.option.Symbol)
        if chain:
            self.last_chain = chain

    def Rebalance(self) -> None:
        if not self.trading_enabled or self.IsWarmingUp:
            return

        value = self.Portfolio.TotalPortfolioValue
        self.high_water = max(self.high_water, value)
        drawdown = 1 - value / self.high_water if self.high_water else 0
        if drawdown > self.max_drawdown:
            self.Liquidate()
            self.trading_enabled = False
            return

        if self.last_chain is None or not self.sma.IsReady:
            return

        if self.Securities[self.equity.Symbol].Price < self.sma.Current.Value:
            return

        if self.Portfolio[self.equity.Symbol].Quantity < 100:
            self.MarketOrder(self.equity.Symbol, 100)

        for holding in self.Portfolio.Values:
            if (
                holding.Invested
                and holding.Symbol.SecurityType == SecurityType.Option
                and holding.Quantity < 0
            ):
                return

        calls = [
            contract
            for contract in self.last_chain
            if contract.Right == OptionRight.Call
        ]
        calls = [
            contract
            for contract in calls
            if contract.Greeks is not None and contract.Greeks.Delta is not None
        ]
        if not calls:
            return

        calls.sort(
            key=lambda contract: (
                abs(contract.Greeks.Delta - self.delta_target),
                contract.Expiry,
            )
        )
        contract = calls[0]
        self.Sell(contract.Symbol, 1)
