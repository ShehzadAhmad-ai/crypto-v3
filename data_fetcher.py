# data_fetcher.py - Optimized with Caching and Error Handling
import ccxt
import pandas as pd
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from config import Config
import logging
from functools import lru_cache
from threading import Lock

class DataFetcher:
    """Optimized data fetcher with caching and connection pooling"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to reuse exchange connections"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, exchange_id: Optional[str] = None):
        """Initialize exchange connection with retry logic"""
        if hasattr(self, 'initialized'):
            return
            
        self.exchange_id = (exchange_id or Config.EXCHANGE).lower()
        self.cache = {}
        self.cache_timestamps = {}
        self.retry_count = 3
        self.retry_delay = 1
        
        # Exchange configuration
        options = {
            'enableRateLimit': Config.ENABLE_RATE_LIMIT,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot'
            }
        }
        
        # Add API credentials if available
        if Config.BINANCE_API_KEY and Config.BINANCE_API_SECRET:
            options['apiKey'] = Config.BINANCE_API_KEY
            options['secret'] = Config.BINANCE_API_SECRET

        # Configure for futures if needed
        if 'binance' in self.exchange_id:
            if Config.BINANCE_API_TYPE and Config.BINANCE_API_TYPE.lower() in ('future', 'futures', 'usdm'):
                options['options']['defaultType'] = 'future'
            else:
                options['options']['defaultType'] = 'spot'

        # Initialize exchange with retry
        self.exchange = self._init_exchange_with_retry(options)
        
        # Enable sandbox for testnet
        if Config.BINANCE_USE_TESTNET and 'binance' in self.exchange_id:
            try:
                self.exchange.set_sandbox_mode(True)
                logging.info("Sandbox mode enabled")
            except Exception as e:
                logging.warning(f"Failed to set sandbox mode: {e}")
        
        self.initialized = True
        logging.info(f"DataFetcher initialized for {self.exchange_id}")
    
    def _init_exchange_with_retry(self, options: Dict) -> ccxt.Exchange:
        """Initialize exchange with retry logic"""
        for attempt in range(self.retry_count):
            try:
                Exchange = getattr(ccxt, self.exchange_id)
                exchange = Exchange(options)
                # Test connection
                exchange.load_markets()
                logging.info(f"Connected to {self.exchange_id}")
                return exchange
            except Exception as e:
                logging.warning(f"Exchange init attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logging.error(f"Failed to initialize exchange after {self.retry_count} attempts")
                    raise
    
    def _get_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """Generate cache key for request"""
        return f"{symbol}:{timeframe}:{limit}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if not Config.ENABLE_CACHING:
            return False
        if key not in self.cache_timestamps:
            return False
        age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
        return age < Config.CACHE_TTL_SECONDS
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 600) -> pd.DataFrame:
        """
        Fetch OHLCV data with caching and retry logic
        Optimized with vectorized operations
        """
        cache_key = self._get_cache_key(symbol, timeframe, limit)
        
        # Return cached data if valid
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        # Fetch with retry
        for attempt in range(self.retry_count):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    logging.warning(f"No data for {symbol} {timeframe}")
                    return pd.DataFrame()
                
                # Vectorized DataFrame creation
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Optimize data types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Convert to float in one operation
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                # Cache the result
                if Config.ENABLE_CACHING:
                    self.cache[cache_key] = df.copy()
                    self.cache_timestamps[cache_key] = datetime.now()
                
                return df
                
            except ccxt.RateLimitExceeded as e:
                wait_time = self.exchange.rateLimit / 1000 if hasattr(self.exchange, 'rateLimit') else 1
                logging.warning(f"Rate limit hit, waiting {wait_time}s")
                time.sleep(wait_time)
                
            except Exception as e:
                logging.warning(f"OHLCV fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logging.error(f"Failed to fetch OHLCV for {symbol} after {self.retry_count} attempts")
        
        return pd.DataFrame()
    
    def fetch_order_book(self, symbol: str, limit: int = 130) -> Optional[Dict[str, Any]]:
        """
        Fetch order book data with caching
        """
        cache_key = f"ob:{symbol}:{limit}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        for attempt in range(self.retry_count):
            try:
                ob = self.exchange.fetch_order_book(symbol, limit=limit)
                
                if not ob or not ob.get('bids') or not ob.get('asks'):
                    return None
                
                # Vectorized calculations
                bids = ob['bids']
                asks = ob['asks']
                
                best_bid = bids[0][0] if bids else None
                best_ask = asks[0][0] if asks else None
                
                # Calculate spread
                spread = None
                if best_bid and best_ask:
                    spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
                
                # Calculate bid/ask volumes
                bid_volume = sum(b[1] for b in bids[:10])
                ask_volume = sum(a[1] for a in asks[:10])
                
                # Detect large orders
                large_bids = [
                    {'price': b[0], 'size': b[1], 'value': b[0] * b[1]}
                    for b in bids[:10] if (b[0] * b[1]) >= Config.LARGE_ORDER_USDT
                ]
                large_asks = [
                    {'price': a[0], 'size': a[1], 'value': a[0] * a[1]}
                    for a in asks[:10] if (a[0] * a[1]) >= Config.LARGE_ORDER_USDT
                ]
                
                # Calculate order book imbalance
                total_bid_value = sum(b[0] * b[1] for b in bids[:10])
                total_ask_value = sum(a[0] * a[1] for a in asks[:10])
                total = total_bid_value + total_ask_value
                imbalance = (total_bid_value - total_ask_value) / total if total > 0 else 0
                
                result = {
                    'raw': ob,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'spread': spread,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'bid_ask_imbalance': imbalance,
                    'large_bids': large_bids,
                    'large_asks': large_asks,
                    'large_bids_count': len(large_bids),
                    'large_asks_count': len(large_asks),
                    'depth_score': min(1.0, (bid_volume + ask_volume) / 1000000)  # Normalize to 1M
                }
                
                # Cache the result
                if Config.ENABLE_CACHING:
                    self.cache[cache_key] = result.copy()
                    self.cache_timestamps[cache_key] = datetime.now()
                
                return result
                
            except Exception as e:
                logging.warning(f"Order book fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        
        return None
    
    def fetch_trades(self, symbol: str, limit: int = 210) -> List[Dict]:
        """
        Fetch recent trades with caching
        """
        cache_key = f"trades:{symbol}:{limit}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            
            # Process trades
            processed = []
            for t in trades:
                processed.append({
                    'timestamp': t['timestamp'],
                    'datetime': t['datetime'],
                    'price': t['price'],
                    'amount': t['amount'],
                    'cost': t['cost'],
                    'side': t['side']
                })
            
            # Cache the result
            if Config.ENABLE_CACHING:
                self.cache[cache_key] = processed.copy()
                self.cache_timestamps[cache_key] = datetime.now()
            
            return processed
            
        except Exception as e:
            logging.warning(f"Failed to fetch trades: {e}")
            return []
    
    def fetch_multiple_ohlcv(self, symbols: List[str], timeframe: str = '5m', limit: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple symbols in parallel (if supported)
        """
        results = {}
        
        # Sequential fetch for now (ccxt doesn't support parallel yet)
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, limit)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logging.warning(f"Failed to fetch {symbol}: {e}")
        
        return results
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logging.info("Cache cleared")
    
    def get_exchange_status(self) -> Dict[str, Any]:
        """Get exchange connection status"""
        try:
            self.exchange.fetch_status()
            return {
                'connected': True,
                'exchange': self.exchange_id,
                'rate_limit': self.exchange.rateLimit if hasattr(self.exchange, 'rateLimit') else None,
                'cache_size': len(self.cache)
            }
        except Exception as e:
            return {
                'connected': False,
                'exchange': self.exchange_id,
                'error': str(e)
            }