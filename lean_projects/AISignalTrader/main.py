# ruff: noqa: F403, F405
from AlgorithmImports import *


class AISignalData(PythonData):
    file_path: str = ""

    def GetSource(self, config, date, isLive):
        return SubscriptionDataSource(self.file_path, SubscriptionTransportMedium.LocalFile)

    def Reader(self, config, line, date, isLive):
        if not line.strip() or line.startswith("timestamp"):
            return None
        parts = line.split(",")
        if len(parts) < 3:
            return None
        data = AISignalData()
        ts = datetime.fromisoformat(parts[0].strip())
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        data.Symbol = config.Symbol
        data.Time = ts
        data.Value = float(parts[2])
        return data


class AISignalTrader(QCAlgorithm):
    def Initialize(self):
        self.symbol = self.get_parameter("symbol", "SPY")
        self.signal_threshold = float(self.get_parameter("signal_threshold", 0.0))
        self.max_drawdown = float(self.get_parameter("max_drawdown", 0.25))
        self.drawdown_cooldown_days = int(self.get_parameter("drawdown_cooldown_days", 20))
        self.signal_mode = self.get_parameter("signal_mode", "directional")
        self.max_exposure = float(self.get_parameter("max_exposure", 1.0))
        self.base_exposure = float(self.get_parameter("base_exposure", 0.0))

        start_date = self.get_parameter("start_date", "2016-01-01")
        end_date = self.get_parameter("end_date", "2021-03-31")
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()

        self.SetStartDate(start.year, start.month, start.day)
        self.SetEndDate(end.year, end.month, end.day)
        self.SetCash(100000)
        self.SetBenchmark(self.symbol)

        self.equity = self.AddEquity(self.symbol, Resolution.Daily).Symbol

        AISignalData.file_path = os.path.join(
            Globals.DataFolder,
            "custom",
            "ai_signals.csv",
        )
        self.signal_symbol = self.AddData(AISignalData, "AI_SIGNAL", Resolution.Daily).Symbol

        self.high_water = self.Portfolio.TotalPortfolioValue
        self.stop_until = None

    def OnData(self, data):
        if self.signal_symbol not in data:
            return

        if self.stop_until is not None and self.Time < self.stop_until:
            return

        value = self.Portfolio.TotalPortfolioValue
        self.high_water = max(self.high_water, value)
        drawdown = 1 - value / self.high_water if self.high_water else 0
        if drawdown > self.max_drawdown:
            self.Liquidate()
            self.high_water = self.Portfolio.TotalPortfolioValue
            self.stop_until = self.Time + timedelta(days=self.drawdown_cooldown_days)
            return

        signal = float(data[self.signal_symbol].Value)

        if abs(signal) < self.signal_threshold:
            signal_value = 0.0
        elif self.signal_mode == "directional":
            signal_value = self.max_exposure if signal > 0 else -self.max_exposure
        else:
            signal_value = max(-self.max_exposure, min(self.max_exposure, signal))

        if self.signal_mode == "overlay":
            target = max(-self.max_exposure, min(self.max_exposure, self.base_exposure + signal_value))
        else:
            target = signal_value

        if target == 0:
            self.Liquidate(self.equity)
        else:
            self.SetHoldings(self.equity, float(target))
