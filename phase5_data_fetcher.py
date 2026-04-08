# phase5_data_fetcher.py
import ccxt
import pandas as pd
import time
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from config import Config
from threading import Lock

import logging
logging.getLogger('ccxt').setLevel(logging.ERROR)

class Phase5DataFetcher:
    """
    Dedicated data fetcher for Phase 5 (Light Confirmations)
    Uses separate API keys to not affect main system
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        
        # Use Phase 5 specific API keys from config
        self.binance_api_key = Config.PHASE5_BINANCE_API_KEY
        self.binance_api_secret = Config.PHASE5_BINANCE_API_SECRET
        self.use_futures = Config.PHASE5_USE_FUTURES
        
        self.cache = {}
        self.cache_timestamps = {}
        self.retry_count = 3
        self.retry_delay = 1
        
        # Initialize spot connection
        self.spot_exchange = self._init_exchange('spot')
        
        # Initialize futures connection if enabled
        self.futures_exchange = None
        if self.use_futures:
            self.futures_exchange = self._init_exchange('future')
        
        self.initialized = True
        logging.info("Phase5DataFetcher initialized")
        logging.info(f"  Futures enabled: {self.use_futures}")
    
    def _init_exchange(self, market_type: str) -> ccxt.Exchange:
        """Initialize exchange with specific market type"""
        options = {
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': market_type
            }
        }
        
        # Add API credentials if available
        if self.binance_api_key and self.binance_api_secret:
            options['apiKey'] = self.binance_api_key
            options['secret'] = self.binance_api_secret
        
        for attempt in range(self.retry_count):
            try:
                exchange = ccxt.binance(options)
                exchange.load_markets()
                logging.info(f"Connected to Binance {market_type}")
                return exchange
            except Exception as e:
                logging.warning(f"Exchange init attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logging.error(f"Failed to initialize Binance {market_type}")
                    return None
    
    # ==================== SPOT MARKET METHODS ====================
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data for cross-asset correlation"""
        cache_key = f"ohlcv:{symbol}:{timeframe}:{limit}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        if not self.spot_exchange:
            return pd.DataFrame()
        
        for attempt in range(self.retry_count):
            try:
                ohlcv = self.spot_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    return pd.DataFrame()
                
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                self.cache[cache_key] = df.copy()
                self.cache_timestamps[cache_key] = datetime.now()
                
                return df
                
            except Exception as e:
                logging.warning(f"OHLCV fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        
        return pd.DataFrame()
    
    def fetch_multiple_ohlcv(self, symbols: List[str], timeframe: str = '1h', limit: int = 200) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols (for correlation)"""
        results = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                results[symbol] = df
        return results
    
    # ==================== FUTURES MARKET METHODS ====================
    
    def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current funding rate from Binance Futures
        Returns dict with funding rate and next funding time
        """
        if not self.futures_exchange:
            return {'current_rate': 0.0, 'next_funding_time': None}
        
        cache_key = f"funding:{symbol}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        try:
            # Convert symbol from BTC/USDT to BTCUSDT
            futures_symbol = symbol.replace('/', '')
            
            funding = self.futures_exchange.fetch_funding_rate(futures_symbol)
            
            result = {
                'current_rate': float(funding['fundingRate']),
                'next_funding_time': funding.get('nextFundingTime'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logging.warning(f"Failed to fetch funding rate for {symbol}: {e}")
            return {'current_rate': 0.0, 'next_funding_time': None}
        

    def fetch_liquidations(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Fetch recent liquidation events from Binance Futures
        Uses Binance API endpoint: /fapi/v1/forceOrders
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            limit: Number of liquidation events to fetch
        
        Returns:
            List of liquidation events with side, price, quantity, value
        """
        if not self.futures_exchange:
            logging.warning("Futures exchange not initialized for liquidations")
            return []
        
        cache_key = f"liquidations:{symbol}:{limit}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        try:
            futures_symbol = symbol.replace('/', '')
            
            # Binance Futures API endpoint for force orders (liquidations)
            # Using ccxt unified method if available
            if hasattr(self.futures_exchange, 'fetch_liquidations'):
                # Some ccxt versions support this
                liq_data = self.futures_exchange.fetch_liquidations(futures_symbol, limit=limit)
            else:
                # Manual API call
                import requests
                base_url = "https://fapi.binance.com"
                endpoint = "/fapi/v1/forceOrders"
                
                params = {
                    'symbol': futures_symbol,
                    'limit': limit
                }
                
                headers = {}
                if self.binance_api_key:
                    headers['X-MBX-APIKEY'] = self.binance_api_key
                
                response = requests.get(
                    base_url + endpoint,
                    params=params,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    liq_data = response.json()
                else:
                    logging.warning(f"Failed to fetch liquidations: {response.status_code}")
                    return []
            
            # Process liquidation data
            result = []
            for liq in liq_data:
                side = liq.get('side', 'UNKNOWN')
                price = float(liq.get('price', 0))
                quantity = float(liq.get('origQty', 0))
                value = price * quantity
                
                if side == 'SELL':
                    liq_type = 'LONG_LIQUIDATION'
                elif side == 'BUY':
                    liq_type = 'SHORT_LIQUIDATION'
                else:
                    liq_type = 'UNKNOWN'
                
                result.append({
                    'time': liq.get('time', datetime.now().timestamp()),
                    'symbol': liq.get('symbol', futures_symbol),
                    'side': side,
                    'liquidation_type': liq_type,
                    'price': price,
                    'quantity': quantity,
                    'value': value,
                    'is_long_liquidation': side == 'SELL',
                    'is_short_liquidation': side == 'BUY'
                })
            
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            logging.info(f"Fetched {len(result)} liquidations for {symbol}")
            return result
            
        except Exception as e:
            logging.error(f"Error fetching liquidations for {symbol}: {e}")
            return []
    
    def get_liquidation_summary(self, symbol: str, minutes: int = 60) -> Dict[str, Any]:
        """Get summary of liquidations in the last X minutes"""
        liquidations = self.fetch_liquidations(symbol, limit=500)
        
        if not liquidations:
            return {
                'total_long_liq': 0,
                'total_short_liq': 0,
                'long_liq_value': 0.0,
                'short_liq_value': 0.0,
                'total_liq_value': 0.0,
                'largest_liq': None
            }
        
        cutoff_time = (datetime.now().timestamp() * 1000) - (minutes * 60 * 1000)
        recent_liq = [l for l in liquidations if l.get('time', 0) > cutoff_time]
        
        long_liq = [l for l in recent_liq if l.get('is_long_liquidation', False)]
        short_liq = [l for l in recent_liq if l.get('is_short_liquidation', False)]
        
        long_value = sum(l.get('value', 0) for l in long_liq)
        short_value = sum(l.get('value', 0) for l in short_liq)
        largest = max(recent_liq, key=lambda x: x.get('value', 0)) if recent_liq else None
        
        return {
            'total_long_liq': len(long_liq),
            'total_short_liq': len(short_liq),
            'long_liq_value': round(long_value, 2),
            'short_liq_value': round(short_value, 2),
            'total_liq_value': round(long_value + short_value, 2),
            'largest_liq': largest,
            'timestamp': datetime.now()
        }

    # ==================== 3-TIER OPEN INTEREST ====================
    
    def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch open interest using 3-TIER approach:
        TIER 1: Binance Direct API
        TIER 2: CCXT
        TIER 3: Smart Proxy
        """
        if not self.futures_exchange:
            return self._tier3_proxy_open_interest(symbol, reason="no_futures")
        
        cache_key = f"oi:{symbol}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        # TIER 1: Binance Direct API
        result = self._tier1_binance_direct_open_interest(symbol)
        if result and result.get('source') == 'binance_direct':
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            logging.info(f"✅ TIER 1: Real open interest for {symbol}")
            return result
        
        # TIER 2: CCXT
        result = self._tier2_ccxt_open_interest(symbol)
        if result and result.get('source') == 'ccxt':
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            logging.info(f"✅ TIER 2: CCXT open interest for {symbol}")
            return result
        
        # TIER 3: Smart Proxy
        result = self._tier3_proxy_open_interest(symbol, reason="all_tiers_failed")
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()
        logging.info(f"✅ TIER 3: Proxy open interest for {symbol}")
        return result

    def _tier1_binance_direct_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """TIER 1: Binance Direct API for Open Interest"""
        try:
            futures_symbol = symbol.replace('/', '')
            base_url = "https://fapi.binance.com"
            endpoint = "/fapi/v1/openInterest"
            
            params = {'symbol': futures_symbol}
            headers = {}
            if self.binance_api_key:
                headers['X-MBX-APIKEY'] = self.binance_api_key
            
            response = requests.get(base_url + endpoint, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'open_interest': float(data.get('openInterest', 0)),
                    'value': float(data.get('openInterest', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'binance_direct',
                    'tier': 1
                }
        except Exception as e:
            logging.debug(f"TIER 1 open interest failed: {e}")
        return None

    def _tier2_ccxt_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """TIER 2: CCXT for Open Interest"""
        try:
            futures_symbol = symbol.replace('/', '')
            oi = self.futures_exchange.fetch_open_interest(futures_symbol)
            
            if oi:
                oi_amount = 0.0
                oi_value = 0.0
                
                if 'openInterestAmount' in oi and oi['openInterestAmount'] is not None:
                    try:
                        oi_amount = float(oi['openInterestAmount'])
                    except (TypeError, ValueError):
                        oi_amount = 0.0
                
                if 'openInterestValue' in oi and oi['openInterestValue'] is not None:
                    try:
                        oi_value = float(oi['openInterestValue'])
                    except (TypeError, ValueError):
                        oi_value = 0.0
                
                return {
                    'open_interest': oi_amount,
                    'value': oi_value,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'ccxt',
                    'tier': 2
                }
        except Exception as e:
            logging.debug(f"TIER 2 open interest failed: {e}")
        return None

    def _tier3_proxy_open_interest(self, symbol: str, reason: str = "fallback") -> Dict[str, Any]:
        """TIER 3: Smart Proxy for Open Interest using volume"""
        try:
            df = self.fetch_ohlcv(symbol, timeframe='1h', limit=24)
            
            if not df.empty and len(df) >= 24:
                volume_24h = df['volume'].sum()
                avg_price = df['close'].mean()
                estimated_oi = volume_24h * 0.2
                estimated_value = estimated_oi * avg_price
                
                return {
                    'open_interest': round(estimated_oi, 2),
                    'value': round(estimated_value, 2),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'volume_proxy',
                    'tier': 3,
                    'proxy_factors': {
                        'volume_24h': round(volume_24h, 2),
                        'avg_price': round(avg_price, 4)
                    }
                }
        except Exception as e:
            logging.debug(f"Proxy open interest failed: {e}")
        
        return {
            'open_interest': 0.0,
            'value': 0.0,
            'timestamp': datetime.now().isoformat(),
            'source': 'default_fallback',
            'tier': 3,
            'reason': reason
        }

    # ==================== 3-TIER LONG/SHORT RATIO ====================
    
    def fetch_long_short_ratio(self, symbol: str) -> Dict[str, float]:
        """
        Fetch long/short ratio using 3-TIER approach:
        TIER 1: Binance Direct API
        TIER 2: CCXT
        TIER 3: Smart Proxy
        """
        if not self.futures_exchange:
            return self._tier3_proxy_long_short(symbol, reason="no_futures")
        
        cache_key = f"lsratio:{symbol}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        # TIER 1: Binance Direct API
        result = self._tier1_binance_direct_long_short(symbol)
        if result and result.get('source') == 'binance_direct':
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            logging.info(f"✅ TIER 1: Real long/short ratio for {symbol}")
            return result
        
        # TIER 2: CCXT
        result = self._tier2_ccxt_long_short(symbol)
        if result and result.get('source') == 'ccxt':
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            logging.info(f"✅ TIER 2: CCXT long/short ratio for {symbol}")
            return result
        
        # TIER 3: Smart Proxy
        result = self._tier3_proxy_long_short(symbol, reason="all_tiers_failed")
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()
        logging.info(f"✅ TIER 3: Proxy long/short ratio for {symbol}")
        return result

    def _tier1_binance_direct_long_short(self, symbol: str) -> Optional[Dict[str, float]]:
        """TIER 1: Binance Direct API for Long/Short Ratio"""
        try:
            futures_symbol = symbol.replace('/', '')
            base_url = "https://fapi.binance.com"
            endpoint = "/futures/data/globalLongShortAccountRatio"
            
            params = {
                'symbol': futures_symbol,
                'period': '5m',
                'limit': 1
            }
            
            headers = {}
            if self.binance_api_key:
                headers['X-MBX-APIKEY'] = self.binance_api_key
            
            response = requests.get(base_url + endpoint, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    latest = data[0]
                    return {
                        'ratio': float(latest.get('longShortRatio', 1.0)),
                        'long_account': float(latest.get('longAccount', 0.5)),
                        'short_account': float(latest.get('shortAccount', 0.5)),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'binance_direct',
                        'tier': 1
                    }
        except Exception as e:
            logging.debug(f"TIER 1 long/short failed: {e}")
        return None

    def _tier2_ccxt_long_short(self, symbol: str) -> Optional[Dict[str, float]]:
        """TIER 2: CCXT for Long/Short Ratio"""
        try:
            futures_symbol = symbol.replace('/', '')
            
            if hasattr(self.futures_exchange, 'fetch_long_short_ratio'):
                ratio_data = self.futures_exchange.fetch_long_short_ratio(futures_symbol)
                if ratio_data:
                    return {
                        'ratio': float(ratio_data.get('longShortRatio', 1.0)),
                        'long_account': float(ratio_data.get('longAccount', 0.5)),
                        'short_account': float(ratio_data.get('shortAccount', 0.5)),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'ccxt',
                        'tier': 2
                    }
        except Exception as e:
            logging.debug(f"TIER 2 long/short failed: {e}")
        return None

    def _tier3_proxy_long_short(self, symbol: str, reason: str = "fallback") -> Dict[str, float]:
        """TIER 3: Smart Proxy for Long/Short Ratio using price action"""
        try:
            df = self.fetch_ohlcv(symbol, timeframe='5m', limit=20)
            
            if not df.empty and len(df) >= 10:
                recent_avg = df['close'].iloc[-5:].mean()
                previous_avg = df['close'].iloc[-10:-5].mean()
                price_trend = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                
                recent_vol = df['volume'].iloc[-5:].mean()
                previous_vol = df['volume'].iloc[-10:-5].mean()
                vol_trend = recent_vol / previous_vol if previous_vol > 0 else 1
                
                last_5 = df.iloc[-5:]
                bullish_candles = sum(1 for _, row in last_5.iterrows() if row['close'] > row['open'])
                bearish_candles = 5 - bullish_candles
                candle_ratio = bullish_candles / bearish_candles if bearish_candles > 0 else 2.0
                
                base_ratio = 1.0
                if price_trend > 0.01:
                    base_ratio += 0.2
                elif price_trend < -0.01:
                    base_ratio -= 0.2
                
                base_ratio *= (candle_ratio ** 0.3)
                
                if vol_trend > 1.2 and price_trend > 0:
                    base_ratio += 0.1
                elif vol_trend > 1.2 and price_trend < 0:
                    base_ratio -= 0.1
                
                ratio = max(0.3, min(3.0, base_ratio))
                long_pct = ratio / (1 + ratio)
                short_pct = 1 / (1 + ratio)
                
                return {
                    'ratio': round(ratio, 3),
                    'long_account': round(long_pct, 3),
                    'short_account': round(short_pct, 3),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'smart_proxy',
                    'tier': 3,
                    'proxy_factors': {
                        'price_trend': round(price_trend, 4),
                        'vol_trend': round(vol_trend, 2),
                        'candle_ratio': round(candle_ratio, 2)
                    }
                }
        except Exception as e:
            logging.debug(f"Smart proxy calculation failed: {e}")
        
        return {
            'ratio': 1.0,
            'long_account': 0.5,
            'short_account': 0.5,
            'timestamp': datetime.now().isoformat(),
            'source': 'default_fallback',
            'tier': 3,
            'reason': reason
        }
    
    def fetch_funding_rate_history(self, symbol: str, limit: int = 8) -> List[float]:
        """Fetch recent funding rate history (last 8 hours = 1 day for 8h funding)"""
        if not self.futures_exchange:
            return []
        
        cache_key = f"funding_history:{symbol}:{limit}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        try:
            futures_symbol = symbol.replace('/', '')
            
            if hasattr(self.futures_exchange, 'fetch_funding_rate_history'):
                history = self.futures_exchange.fetch_funding_rate_history(
                    futures_symbol, limit=limit
                )
                rates = [float(h['fundingRate']) for h in history]
                
                self.cache[cache_key] = rates
                self.cache_timestamps[cache_key] = datetime.now()
                return rates
            
            return []
            
        except Exception as e:
            logging.warning(f"Failed to fetch funding history for {symbol}: {e}")
            return []
        
    # ==================== MARKET DATA (3-TIER) ====================
    
    def fetch_btc_dominance(self) -> Dict[str, Any]:
        """
        Fetch BTC dominance using 3-TIER approach:
        TIER 1: Binance (BTC/USDT volume vs total)
        TIER 2: CCXT
        TIER 3: CoinGecko API (fallback)
        
        Returns:
            Dict with dominance percentage and source
        """
        cache_key = "btc_dominance"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        # TIER 1: Binance Direct (using BTC volume vs top coins)
        result = self._tier1_binance_dominance()
        if result:
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            logging.info("TIER 1: BTC dominance from Binance")
            return result
        
        # TIER 2: CCXT
        result = self._tier2_ccxt_dominance()
        if result:
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            logging.info("TIER 2: BTC dominance from CCXT")
            return result
        
        # TIER 3: CoinGecko API
        result = self._tier3_coingecko_dominance()
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()
        logging.info("TIER 3: BTC dominance from CoinGecko")
        return result
    
    def _tier1_binance_dominance(self) -> Optional[Dict[str, Any]]:
        """TIER 1: Estimate BTC dominance from Binance volume"""
        try:
            # Fetch BTC/USDT price
            btc_ticker = self.spot_exchange.fetch_ticker('BTC/USDT')
            btc_price = btc_ticker.get('last', 0)
            
            # Estimate BTC market cap (approx 19.5M BTC in circulation)
            btc_supply = 19500000
            btc_mcap = btc_price * btc_supply
            
            # Fetch top 10 coins to estimate total market cap
            tickers = self.spot_exchange.fetch_tickers()
            total_mcap = 0
            
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT'):
                    price = ticker.get('last', 0)
                    volume = ticker.get('quoteVolume', 0)
                    # Rough estimate: market cap ~ price * (volume / price) * factor
                    # Simplified: use price * (volume / price) * 100 = volume * 100
                    if price > 0:
                        est_mcap = volume * 100
                        total_mcap += est_mcap
            
            if total_mcap > 0:
                dominance = (btc_mcap / total_mcap) * 100
                return {
                    'dominance': round(dominance, 2),
                    'btc_price': btc_price,
                    'estimated_total_mcap': round(total_mcap, 2),
                    'source': 'binance_proxy',
                    'tier': 1
                }
        except Exception as e:
            logging.debug(f"TIER 1 dominance failed: {e}")
        return None
    
    def _tier2_ccxt_dominance(self) -> Optional[Dict[str, Any]]:
        """TIER 2: Estimate BTC dominance from CCXT"""
        try:
            # Use same method as tier 1 but with spot exchange
            if not self.spot_exchange:
                return None
            
            btc_ticker = self.spot_exchange.fetch_ticker('BTC/USDT')
            btc_price = btc_ticker.get('last', 0)
            btc_supply = 19500000
            btc_mcap = btc_price * btc_supply
            
            # Fetch all tickers
            tickers = self.spot_exchange.fetch_tickers()
            total_mcap = 0
            
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT'):
                    price = ticker.get('last', 0)
                    volume = ticker.get('quoteVolume', 0)
                    if price > 0:
                        est_mcap = volume * 100
                        total_mcap += est_mcap
            
            if total_mcap > 0:
                dominance = (btc_mcap / total_mcap) * 100
                return {
                    'dominance': round(dominance, 2),
                    'btc_price': btc_price,
                    'estimated_total_mcap': round(total_mcap, 2),
                    'source': 'ccxt_proxy',
                    'tier': 2
                }
        except Exception as e:
            logging.debug(f"TIER 2 dominance failed: {e}")
        return None
    
    def _tier3_coingecko_dominance(self) -> Dict[str, Any]:
        """TIER 3: Fetch BTC dominance from CoinGecko API"""
        try:
            import requests
            response = requests.get(
                'https://api.coingecko.com/api/v3/global',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                dominance = data.get('data', {}).get('market_cap_percentage', {}).get('btc', 0)
                total_mcap = data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
                btc_price = data.get('data', {}).get('market_cap_change_percentage_24h_usd', 0)
                
                return {
                    'dominance': round(dominance, 2),
                    'btc_price': btc_price,
                    'total_market_cap': total_mcap,
                    'source': 'coingecko',
                    'tier': 3
                }
        except Exception as e:
            logging.debug(f"TIER 3 dominance failed: {e}")
        
        return {
            'dominance': 50.0,
            'btc_price': 0,
            'source': 'default_fallback',
            'tier': 3,
            'reason': 'all_tiers_failed'
        }
    
    def fetch_total_market_cap(self) -> Dict[str, Any]:
        """
        Fetch total cryptocurrency market cap
        """
        cache_key = "total_market_cap"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        try:
            response = requests.get(
                'https://api.coingecko.com/api/v3/global',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                total_mcap = data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
                
                result = {
                    'total_market_cap': total_mcap,
                    'source': 'coingecko',
                    'timestamp': datetime.now().isoformat()
                }
                
                self.cache[cache_key] = result
                self.cache_timestamps[cache_key] = datetime.now()
                return result
        except Exception as e:
            logging.debug(f"Failed to fetch total market cap: {e}")
        
        return {
            'total_market_cap': 0,
            'source': 'fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def fetch_fear_greed_index(self) -> Dict[str, Any]:
        """
        Fetch Fear & Greed Index from alternative.me
        """
        cache_key = "fear_greed"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        try:
            response = requests.get(
                'https://api.alternative.me/fng/?limit=1',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and len(data['data']) > 0:
                    result = {
                        'value': int(data['data'][0]['value']),
                        'classification': data['data'][0]['value_classification'],
                        'timestamp': data['data'][0]['timestamp'],
                        'source': 'alternative.me'
                    }
                    
                    self.cache[cache_key] = result
                    self.cache_timestamps[cache_key] = datetime.now()
                    return result
        except Exception as e:
            logging.debug(f"Failed to fetch Fear & Greed: {e}")
        
        return {
            'value': 50,
            'classification': 'Neutral',
            'source': 'default_fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def fetch_top_coins(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch top coins by market cap (for sector analysis)
        """
        cache_key = f"top_coins_{limit}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].copy()
        
        try:
            # Try CoinGecko first
            response = requests.get(
                f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={limit}&page=1&sparkline=false',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                result = []
                for coin in data:
                    result.append({
                        'symbol': coin.get('symbol', '').upper() + '/USDT',
                        'name': coin.get('name', ''),
                        'price': coin.get('current_price', 0),
                        'market_cap': coin.get('market_cap', 0),
                        'volume_24h': coin.get('total_volume', 0),
                        'change_24h': coin.get('price_change_percentage_24h', 0)
                    })
                
                self.cache[cache_key] = result
                self.cache_timestamps[cache_key] = datetime.now()
                return result
        except Exception as e:
            logging.debug(f"Failed to fetch top coins from CoinGecko: {e}")
        
        # Fallback: use Binance tickers
        if self.spot_exchange:
            try:
                tickers = self.spot_exchange.fetch_tickers()
                coin_list = []
                for symbol, ticker in tickers.items():
                    if symbol.endswith('/USDT'):
                        volume = ticker.get('quoteVolume', 0)
                        if volume > 0:
                            coin_list.append({
                                'symbol': symbol,
                                'price': ticker.get('last', 0),
                                'volume_24h': volume,
                                'change_24h': ticker.get('percentage', 0)
                            })
                
                # Sort by volume
                coin_list.sort(key=lambda x: x['volume_24h'], reverse=True)
                self.cache[cache_key] = coin_list[:limit]
                self.cache_timestamps[cache_key] = datetime.now()
                return coin_list[:limit]
                
            except Exception as e:
                logging.debug(f"Failed to fetch top coins from Binance: {e}")
        
        return []
    
    # ==================== GENERAL METHODS ====================
    
    def _get_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        return f"{symbol}:{timeframe}:{limit}"
    
    def _is_cache_valid(self, key: str) -> bool:
        if not Config.ENABLE_CACHING:
            return False
        if key not in self.cache_timestamps:
            return False
        age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
        return age < Config.PHASE5_CACHE_TTL
    
    def clear_cache(self):
        self.cache.clear()
        self.cache_timestamps.clear()
        logging.info("Phase5DataFetcher cache cleared")