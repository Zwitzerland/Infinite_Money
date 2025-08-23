"""
Data Engineer Agent for Infinite_Money trading system.

Responsible for:
- Collecting market data from multiple sources
- Data cleaning and preprocessing
- Real-time data streaming
- Data storage and retrieval
- Feature engineering
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yfinance as yf

from .base_agent import BaseAgent, Task
from ..utils.logger import get_logger
from ..utils.config import Config


class DataEngineerAgent(BaseAgent):
    """
    Data Engineer Agent - Handles all data operations for the trading system.
    
    Responsibilities:
    1. Market Data Collection: Fetch data from multiple sources
    2. Data Processing: Clean, validate, and transform data
    3. Real-time Streaming: Handle live market data feeds
    4. Feature Engineering: Create technical indicators and features
    5. Data Storage: Manage data persistence and retrieval
    """
    
    def __init__(self, config: Config, agent_config: Dict[str, Any] = None):
        """Initialize Data Engineer Agent."""
        super().__init__("DataEngineer", config, agent_config)
        
        # Data sources configuration
        self.data_sources = agent_config.get("data_sources", {})
        self.data_retention_days = agent_config.get("data_retention_days", 365)
        self.real_time_enabled = agent_config.get("real_time_enabled", True)
        
        # Data storage
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        
        # Real-time data streams
        self.active_streams: Dict[str, Any] = {}
        
        # Feature engineering
        self.technical_indicators = [
            "sma", "ema", "rsi", "macd", "bollinger_bands", 
            "stochastic", "williams_r", "cci", "adx"
        ]
        
        self.logger.info("Data Engineer Agent initialized")
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute Data Engineer tasks."""
        task_type = task.task_type
        data = task.data
        
        try:
            if task_type == "load_historical_data":
                return await self._load_historical_data(data)
            elif task_type == "collect_market_data":
                return await self._collect_market_data(data)
            elif task_type == "process_data":
                return await self._process_data(data)
            elif task_type == "generate_features":
                return await self._generate_features(data)
            elif task_type == "start_real_time_stream":
                return await self._start_real_time_stream(data)
            elif task_type == "stop_real_time_stream":
                return await self._stop_real_time_stream(data)
            elif task_type == "clean_data":
                return await self._clean_data(data)
            elif task_type == "validate_data":
                return await self._validate_data(data)
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _load_historical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load historical market data."""
        symbols = data.get("symbols", [])
        start_date = data.get("start_date", "2020-01-01")
        end_date = data.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        interval = data.get("interval", "1d")
        
        self.logger.info(f"Loading historical data for {symbols} from {start_date} to {end_date}")
        
        try:
            loaded_data = {}
            
            for symbol in symbols:
                # Check cache first
                cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
                if cache_key in self.market_data_cache:
                    loaded_data[symbol] = self.market_data_cache[cache_key]
                    self.logger.info(f"Loaded {symbol} from cache")
                    continue
                
                # Fetch from data source
                df = await self._fetch_data_from_source(symbol, start_date, end_date, interval)
                
                if df is not None and not df.empty:
                    # Clean and validate data
                    df = await self._clean_dataframe(df)
                    df = await self._validate_dataframe(df)
                    
                    # Store in cache
                    self.market_data_cache[cache_key] = df
                    loaded_data[symbol] = df
                    
                    self.logger.info(f"Loaded {len(df)} records for {symbol}")
                else:
                    self.logger.warning(f"No data found for {symbol}")
            
            return {
                "status": "success",
                "data": loaded_data,
                "symbols_loaded": len(loaded_data),
                "total_records": sum(len(df) for df in loaded_data.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _fetch_data_from_source(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from the configured data source."""
        try:
            # Use yfinance as default source
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                return None
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    async def _collect_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect current market data."""
        symbols = data.get("symbols", [])
        sources = data.get("sources", ["yahoo"])
        
        self.logger.info(f"Collecting market data for {symbols} from {sources}")
        
        try:
            collected_data = {}
            
            for symbol in symbols:
                for source in sources:
                    df = await self._fetch_current_data(symbol, source)
                    if df is not None:
                        collected_data[f"{symbol}_{source}"] = df
            
            return {
                "status": "success",
                "data": collected_data,
                "symbols_collected": len(collected_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _fetch_current_data(self, symbol: str, source: str) -> Optional[pd.DataFrame]:
        """Fetch current market data from specified source."""
        try:
            if source == "yahoo":
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1d", interval="1m")
                
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    df['symbol'] = symbol
                    df['source'] = source
                    return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching current data for {symbol} from {source}: {str(e)}")
            return None
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and transform market data."""
        input_data = data.get("data", {})
        operations = data.get("operations", [])
        
        self.logger.info(f"Processing data with {len(operations)} operations")
        
        try:
            processed_data = {}
            
            for symbol, df in input_data.items():
                processed_df = df.copy()
                
                for operation in operations:
                    if operation == "normalize":
                        processed_df = await self._normalize_data(processed_df)
                    elif operation == "resample":
                        freq = data.get("resample_freq", "1D")
                        processed_df = await self._resample_data(processed_df, freq)
                    elif operation == "fill_missing":
                        method = data.get("fill_method", "ffill")
                        processed_df = await self._fill_missing_data(processed_df, method)
                
                processed_data[symbol] = processed_df
            
            return {
                "status": "success",
                "data": processed_data,
                "operations_applied": operations
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _generate_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical indicators and features."""
        input_data = data.get("data", {})
        indicators = data.get("indicators", self.technical_indicators)
        
        self.logger.info(f"Generating features with {len(indicators)} indicators")
        
        try:
            feature_data = {}
            
            for symbol, df in input_data.items():
                features_df = df.copy()
                
                for indicator in indicators:
                    if indicator == "sma":
                        features_df = await self._add_sma(features_df, data.get("sma_periods", [20, 50]))
                    elif indicator == "ema":
                        features_df = await self._add_ema(features_df, data.get("ema_periods", [12, 26]))
                    elif indicator == "rsi":
                        features_df = await self._add_rsi(features_df, data.get("rsi_period", 14))
                    elif indicator == "macd":
                        features_df = await self._add_macd(features_df)
                    elif indicator == "bollinger_bands":
                        features_df = await self._add_bollinger_bands(features_df, data.get("bb_period", 20))
                    elif indicator == "stochastic":
                        features_df = await self._add_stochastic(features_df, data.get("stoch_period", 14))
                
                feature_data[symbol] = features_df
                
                # Cache features
                cache_key = f"features_{symbol}_{datetime.now().strftime('%Y%m%d')}"
                self.feature_cache[cache_key] = features_df
            
            return {
                "status": "success",
                "data": feature_data,
                "indicators_generated": indicators
            }
            
        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _start_real_time_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Start real-time data streaming."""
        symbols = data.get("symbols", [])
        
        if not self.real_time_enabled:
            return {"status": "error", "message": "Real-time streaming not enabled"}
        
        self.logger.info(f"Starting real-time streams for {symbols}")
        
        try:
            for symbol in symbols:
                if symbol not in self.active_streams:
                    # Start stream (placeholder for actual implementation)
                    self.active_streams[symbol] = {
                        "status": "active",
                        "start_time": datetime.now(),
                        "data_points": 0
                    }
            
            return {
                "status": "success",
                "active_streams": len(self.active_streams),
                "symbols": list(self.active_streams.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error starting real-time streams: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _stop_real_time_stream(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Stop real-time data streaming."""
        symbols = data.get("symbols", list(self.active_streams.keys()))
        
        self.logger.info(f"Stopping real-time streams for {symbols}")
        
        try:
            stopped_count = 0
            
            for symbol in symbols:
                if symbol in self.active_streams:
                    del self.active_streams[symbol]
                    stopped_count += 1
            
            return {
                "status": "success",
                "streams_stopped": stopped_count,
                "remaining_streams": len(self.active_streams)
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping real-time streams: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean market data."""
        input_data = data.get("data", {})
        
        self.logger.info("Cleaning market data")
        
        try:
            cleaned_data = {}
            
            for symbol, df in input_data.items():
                cleaned_df = await self._clean_dataframe(df)
                cleaned_data[symbol] = cleaned_df
            
            return {
                "status": "success",
                "data": cleaned_data,
                "symbols_cleaned": len(cleaned_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market data quality."""
        input_data = data.get("data", {})
        
        self.logger.info("Validating market data")
        
        try:
            validation_results = {}
            
            for symbol, df in input_data.items():
                validation_result = await self._validate_dataframe(df)
                validation_results[symbol] = validation_result
            
            return {
                "status": "success",
                "validation_results": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    # Helper methods for data processing
    async def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a dataframe."""
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        # Remove outliers (simple approach)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    async def _validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe quality."""
        validation = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "date_range": {
                "start": str(df.index.min()) if not df.empty else None,
                "end": str(df.index.max()) if not df.empty else None
            }
        }
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation["missing_columns"] = missing_columns
        
        # Check data quality
        if not df.empty:
            validation["price_range"] = {
                "min": float(df['close'].min()),
                "max": float(df['close'].max()),
                "mean": float(df['close'].mean())
            }
        
        return validation
    
    async def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize price data."""
        if df.empty or 'close' not in df.columns:
            return df
        
        # Min-max normalization
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val)
        
        return df
    
    async def _resample_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample data to different frequency."""
        if df.empty:
            return df
        
        try:
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
        except Exception as e:
            self.logger.error(f"Error resampling data: {str(e)}")
            return df
    
    async def _fill_missing_data(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Fill missing data using specified method."""
        if df.empty:
            return df
        
        if method == "ffill":
            return df.fillna(method='ffill')
        elif method == "bfill":
            return df.fillna(method='bfill')
        elif method == "interpolate":
            return df.interpolate()
        else:
            return df.fillna(method='ffill')
    
    # Technical indicators
    async def _add_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add Simple Moving Average."""
        if df.empty or 'close' not in df.columns:
            return df
        
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        return df
    
    async def _add_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add Exponential Moving Average."""
        if df.empty or 'close' not in df.columns:
            return df
        
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        return df
    
    async def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        if df.empty or 'close' not in df.columns:
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    async def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        if df.empty or 'close' not in df.columns:
            return df
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    async def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands."""
        if df.empty or 'close' not in df.columns:
            return df
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std * 2)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * 2)
        
        return df
    
    async def _add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            return df
        
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent-specific metrics."""
        return {
            "cache_size": len(self.market_data_cache),
            "feature_cache_size": len(self.feature_cache),
            "active_streams": len(self.active_streams),
            "data_sources": list(self.data_sources.keys()) if self.data_sources else []
        }