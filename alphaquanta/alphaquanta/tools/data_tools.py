"""
Data tools for QuantConnect API integration and market data feeds.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


class QuantConnectDataTool:
    """QuantConnect API integration for historical and live data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_token = config.get('api_token')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Get historical price data for backtesting."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            data = []
            prev_close = None
            
            for date, row in hist.iterrows():
                data_point = {
                    'date': date.isoformat(),
                    'symbol': symbol,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'price': float(row['Close']),
                    'prev_price': prev_close if prev_close else float(row['Open'])
                }
                
                if len(data) >= 20:
                    sma_20 = np.mean([d['close'] for d in data[-20:]])
                    data_point['sma_20'] = sma_20
                
                data.append(data_point)
                prev_close = float(row['Close'])
            
            self.logger.info(f"Retrieved {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def get_live_data(self, symbol: str) -> Optional[Dict]:
        """Get live market data."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'bid': float(info.get('bid', latest['Close'])),
                'ask': float(info.get('ask', latest['Close'])),
                'volume': int(latest['Volume']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching live data for {symbol}: {e}")
            return None


class MarketDataTool:
    """Market data tool for technical indicators and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_current_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data with technical indicators."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return {'symbol': symbol, 'price': 0, 'sma_20': 0, 'rsi': 50}
            
            latest = hist.iloc[-1]
            closes = hist['Close'].values
            
            price = float(latest['Close'])
            sma_20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else price
            rsi = self._calculate_rsi(closes)
            
            return {
                'symbol': symbol,
                'price': price,
                'sma_20': sma_20,
                'rsi': rsi,
                'volume': int(latest['Volume']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current data for {symbol}: {e}")
            return {'symbol': symbol, 'price': 0, 'sma_20': 0, 'rsi': 50}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        data = await self.get_current_data(symbol)
        return data.get('price', 0.0)
