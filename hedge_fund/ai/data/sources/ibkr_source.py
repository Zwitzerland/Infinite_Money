"""IBKR historical data source (requires IB Gateway/TWS)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from ib_insync import IB, Stock

from ..market import MarketBar


@dataclass(frozen=True)
class IBKRMarketDataSource:
    """Fetch historical data from IBKR via ib_insync."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7
    duration: str = "2 Y"
    bar_size: str = "1 day"
    use_rth: bool = True

    def fetch(
        self,
        symbols: Sequence[str],
        start: str | None = None,
        end: str | None = None,
    ) -> list[MarketBar]:
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id)
        bars: list[MarketBar] = []
        try:
            for symbol in symbols:
                contract = Stock(symbol, "SMART", "USD")
                ib.qualifyContracts(contract)
                data = ib.reqHistoricalData(
                    contract,
                    endDateTime="" if end is None else end,
                    durationStr=self.duration,
                    barSizeSetting=self.bar_size,
                    whatToShow="TRADES",
                    useRTH=self.use_rth,
                    formatDate=1,
                )
                for bar in data:
                    timestamp = (
                        bar.date
                        if isinstance(bar.date, datetime)
                        else datetime.fromisoformat(str(bar.date))
                    )
                    bars.append(
                        MarketBar(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(bar.open),
                            high=float(bar.high),
                            low=float(bar.low),
                            close=float(bar.close),
                            volume=float(bar.volume),
                        )
                    )
        finally:
            ib.disconnect()
        return bars
