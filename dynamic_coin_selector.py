# dynamic_coin_selector.py - Complete Coin Selection & Ranking System
"""
Dynamic Coin Selector - Scans ALL Binance coins and selects the best ones for analysis
Features:
- Multi-timeframe analysis (5m, 1h, 4h)
- Advanced ranking with normalization
- Quick market scanner
- Configurable via config.py
- Always includes TARGET_COINS from config
- Cache system to prevent rate limits
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

from config import Config
from logger import log
from data_fetcher import DataFetcher
from technical_analyzer import TechnicalAnalyzer
from market_regime import MarketRegimeDetector
from enhanced_patterns import EnhancedPatternDetector

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CoinCandidate:
    """Represents a potential coin for trading"""
    symbol: str
    price: float
    volume_24h: float
    volume_rank: int
    volatility: float
    trend_score: float
    momentum_score: float
    volume_score: float
    pattern_score: float
    final_score: float
    reason: str
    timestamp: datetime
    sector: str = "Other"

@dataclass
class RankedCoin:
    """Ranked coin with normalized scores"""
    symbol: str
    trend_rank: float
    momentum_rank: float
    volume_rank: float
    volatility_rank: float
    pattern_rank: float
    final_rank: int
    score: float
    sector: str

@dataclass
class MarketScanResult:
    """Quick market scan results"""
    bullish: List[str]
    bearish: List[str]
    neutral: List[str]
    total: int
    timestamp: datetime

# ============================================================================
# MAIN COIN SELECTOR CLASS
# ============================================================================

class DynamicCoinSelector:
    """
    Complete coin selection and ranking system
    - Scans all Binance coins
    - Filters by volume, price, spread
    - Multi-timeframe analysis
    - Advanced ranking
    - Quick market scanning
    - Configurable via config.py
    """
    
    def __init__(self):
        # Load settings from config
        self.min_volume_usdt = getattr(Config, 'COIN_MIN_VOLUME_USDT', 100_000)
        self.max_price = getattr(Config, 'COIN_MAX_PRICE', 100000)
        self.min_price = getattr(Config, 'COIN_MIN_PRICE', 0.000001)
        self.max_spread = getattr(Config, 'COIN_MAX_SPREAD', 0.01)
        self.max_results = getattr(Config, 'COIN_MAX_RESULTS', 200)
        self.cache_minutes = getattr(Config, 'COIN_CACHE_MINUTES', 5)
        self.scan_hours = getattr(Config, 'COIN_SCAN_HOURS', 1)
        self.always_include_targets = getattr(Config, 'COIN_ALWAYS_INCLUDE_TARGETS', True)
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.tech_analyzer = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.pattern_detector = EnhancedPatternDetector()
        
        # Cache
        self.coin_cache = {}
        self.rank_cache = {}
        self.scan_cache = {}
        
        # Sector mapping
        self.sector_map = self._init_sector_map()
        
        log.info(f"DynamicCoinSelector initialized:")
        log.info(f"  Min Volume: ${self.min_volume_usdt:,.0f}")
        log.info(f"  Price Range: ${self.min_price} - ${self.max_price}")
        log.info(f"  Max Results: {self.max_results}")
        log.info(f"  Cache Duration: {self.cache_minutes} minutes")
        log.info(f"  Always include config targets: {self.always_include_targets}")
    
    def _init_sector_map(self) -> Dict[str, str]:
        """Initialize sector mapping for coins"""
        return {
            # L1s
            'BTC': 'L1', 'ETH': 'L1', 'SOL': 'L1', 'AVAX': 'L1', 'ADA': 'L1',
            'NEAR': 'L1', 'FTM': 'L1', 'ALGO': 'L1', 'EGLD': 'L1', 'FLOW': 'L1',
            'ICP': 'L1', 'APT': 'L1', 'SUI': 'L1', 'SEI': 'L1', 'TIA': 'L1',
            
            # L2s
            'MATIC': 'L2', 'ARB': 'L2', 'OP': 'L2', 'METIS': 'L2', 'BOBA': 'L2',
            'IMX': 'L2', 'LRC': 'L2', 'SKL': 'L2',
            
            # DeFi
            'UNI': 'DeFi', 'AAVE': 'DeFi', 'MKR': 'DeFi', 'COMP': 'DeFi', 'CRV': 'DeFi',
            'SNX': 'DeFi', 'SUSHI': 'DeFi', 'YFI': 'DeFi', 'BAL': 'DeFi', '1INCH': 'DeFi',
            'CAKE': 'DeFi', 'DYDX': 'DeFi', 'RUNE': 'DeFi', 'LDO': 'DeFi', 'FXS': 'DeFi',
            
            # Meme
            'DOGE': 'Meme', 'SHIB': 'Meme', 'PEPE': 'Meme', 'FLOKI': 'Meme', 'BONK': 'Meme',
            'WIF': 'Meme', 'MEME': 'Meme',
            
            # AI
            'FET': 'AI', 'AGIX': 'AI', 'OCEAN': 'AI', 'NMR': 'AI', 'GRT': 'AI',
            'RNDR': 'AI', 'AKT': 'AI',
            
            # Gaming
            'AXS': 'Gaming', 'SAND': 'Gaming', 'MANA': 'Gaming', 'GALA': 'Gaming',
            'ENJ': 'Gaming', 'ILV': 'Gaming', 'YGG': 'Gaming',
            
            # Oracle
            'LINK': 'Oracle', 'BAND': 'Oracle', 'TRB': 'Oracle', 'API3': 'Oracle',
            
            # Privacy
            'ZEC': 'Privacy', 'XMR': 'Privacy', 'DASH': 'Privacy', 'ROSE': 'Privacy',
            
            # Storage
            'FIL': 'Storage', 'AR': 'Storage', 'BLZ': 'Storage',
            
            # Exchange
            'BNB': 'Exchange', 'CRO': 'Exchange', 'OKB': 'Exchange', 'KCS': 'Exchange',
            'GT': 'Exchange', 'LEO': 'Exchange',
        }
    
    # ============================================================================
    # PUBLIC METHODS
    # ============================================================================
    
    def get_top_symbols(self, include_config_targets: bool = True) -> List[str]:
        """
        Get list of top coin symbols for main analysis
        Always includes Config.TARGET_COINS by default
        """
        coins = self.scan_all_coins()
        
        # Get top dynamic coins
        top_dynamic = [c.symbol for c in coins[:self.max_results]]
        
        if include_config_targets and self.always_include_targets:
            # Combine with config targets, remove duplicates
            config_targets = list(Config.TARGET_COINS)
            all_symbols = list(set(config_targets + top_dynamic))
            log.info(f"Combined {len(config_targets)} config targets + {len(top_dynamic)} dynamic coins = {len(all_symbols)} total")
            return all_symbols
        
        return top_dynamic
    
    def scan_all_coins(self, force_refresh: bool = False) -> List[CoinCandidate]:
        """
        Scan ALL coins on Binance and return top candidates
        """
        # Check cache first
        if not force_refresh and 'coins' in self.coin_cache:
            cache_time = self.coin_cache.get('timestamp')
            if cache_time and datetime.now() - cache_time < timedelta(minutes=self.cache_minutes):
                log.info(f"Using cached coin list ({len(self.coin_cache['coins'])} coins)")
                return self.coin_cache['coins']
        
        log.info("=" * 70)
        log.info(" SCANNING ALL BINANCE COINS")
        log.info("=" * 70)
        
        try:
            # Step 1: Fetch all tickers
            tickers = self._fetch_all_tickers()
            log.info(f"Fetched {len(tickers)} total coins")
            
            # Step 2: Apply basic filters
            filtered = self._apply_basic_filters(tickers)
            log.info(f"After basic filters: {len(filtered)} coins")
            
            # Step 3: Multi-timeframe analysis
            candidates = self._analyze_candidates_multi_tf(filtered)
            log.info(f"After technical analysis: {len(candidates)} coins")
            
            # Step 4: Rank and select top coins
            top_coins = self._rank_and_select(candidates)
            log.info(f"Selected top {len(top_coins)} coins for analysis")
            
            # Cache results
            self.coin_cache = {
                'coins': top_coins,
                'timestamp': datetime.now()
            }
            
            # Print summary
            self._print_summary(top_coins)
            
            return top_coins
            
        except Exception as e:
            log.error(f"Error scanning coins: {e}")
            return []
    
    def rank_coins(self, coins_data: Dict[str, pd.DataFrame]) -> List[RankedCoin]:
        """
        Advanced coin ranking with normalization
        """
        rankings = []
        
        if not coins_data:
            return rankings
        
        # First pass: calculate raw scores
        raw_scores = []
        for symbol, df in coins_data.items():
            if df is None or df.empty or len(df) < 50:
                continue
            
            # Get sector
            base = symbol.replace('/USDT', '')
            sector = self.sector_map.get(base, 'Other')
            
            # Calculate individual scores
            trend_score = self._score_trend(df)
            momentum_score = self._score_momentum(df)
            volume_score = self._score_volume(df)
            volatility_score = self._score_volatility(df)
            pattern_score = self._score_patterns(df)
            
            # Combined score (before normalization)
            raw_score = (
                trend_score * 0.3 +
                momentum_score * 0.25 +
                volume_score * 0.2 +
                volatility_score * 0.15 +
                pattern_score * 0.1
            )
            
            raw_scores.append({
                'symbol': symbol,
                'sector': sector,
                'trend': trend_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'volatility': volatility_score,
                'pattern': pattern_score,
                'raw_score': raw_score
            })
        
        if not raw_scores:
            return rankings
        
        # Second pass: normalize scores (0-1 range)
        max_score = max(r['raw_score'] for r in raw_scores)
        min_score = min(r['raw_score'] for r in raw_scores)
        score_range = max_score - min_score
        
        for r in raw_scores:
            if score_range > 0:
                normalized_score = (r['raw_score'] - min_score) / score_range
            else:
                normalized_score = 0.5
            
            rankings.append(RankedCoin(
                symbol=r['symbol'],
                trend_rank=r['trend'],
                momentum_rank=r['momentum'],
                volume_rank=r['volume'],
                volatility_rank=r['volatility'],
                pattern_rank=r['pattern'],
                final_rank=0,  # Will set after sorting
                score=normalized_score,
                sector=r['sector']
            ))
        
        # Sort by score
        rankings.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(rankings):
            r.final_rank = i + 1
        
        # Cache rankings
        self.rank_cache = {
            'rankings': rankings,
            'timestamp': datetime.now()
        }
        
        return rankings
    
    def quick_scan(self, symbols: List[str] = None) -> MarketScanResult:
        """
        Quick market scan for sentiment overview
        """
        # Check cache
        if 'scan' in self.scan_cache:
            cache_time = self.scan_cache.get('timestamp')
            if cache_time and datetime.now() - cache_time < timedelta(minutes=5):
                return self.scan_cache['scan']
        
        if symbols is None:
            # Use top coins if none provided
            coins = self.scan_all_coins()
            symbols = [c.symbol for c in coins[:30]]
        
        results = {
            'bullish': [],
            'bearish': [],
            'neutral': [],
            'total': 0
        }
        
        for symbol in symbols:
            try:
                # Fetch 15m data for quick scan
                df = self.data_fetcher.fetch_ohlcv(symbol, '15m', limit=50)
                if df is None or df.empty or len(df) < 30:
                    continue
                
                # Quick analysis with multiple indicators
                signal = self._quick_analysis_multi_tf(df)
                
                if signal == 'BULL':
                    results['bullish'].append(symbol)
                elif signal == 'BEAR':
                    results['bearish'].append(symbol)
                else:
                    results['neutral'].append(symbol)
                
                results['total'] += 1
                
            except Exception:
                continue
        
        scan_result = MarketScanResult(
            bullish=results['bullish'],
            bearish=results['bearish'],
            neutral=results['neutral'],
            total=results['total'],
            timestamp=datetime.now()
        )
        
        # Cache
        self.scan_cache = {
            'scan': scan_result,
            'timestamp': datetime.now()
        }
        
        return scan_result
    
    def should_scan_new_coins(self) -> bool:
        """Determine if we should scan for new coins"""
        if 'last_scan' not in self.coin_cache:
            return True
        
        last_scan = self.coin_cache.get('timestamp')
        if not last_scan:
            return True
        
        elapsed = datetime.now() - last_scan
        return elapsed > timedelta(hours=self.scan_hours)
    
    # ============================================================================
    # PRIVATE METHODS - Data Fetching
    # ============================================================================
    
    def _fetch_all_tickers(self) -> List[Dict]:
        """Fetch all USDT pairs from Binance"""
        try:
            tickers = self.data_fetcher.exchange.fetch_tickers()
            
            results = []
            for symbol, ticker in tickers.items():
                # Only USDT pairs
                if not symbol.endswith('/USDT'):
                    continue
                
                # Skip stablecoins
                if any(stable in symbol for stable in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDP']):
                    continue
                
                # Extract data
                last = ticker.get('last')
                if last is None or last <= 0:
                    continue
                
                quote_volume = ticker.get('quoteVolume') or ticker.get('quoteVolume', 0)
                if quote_volume is None:
                    quote_volume = 0
                
                high = ticker.get('high', last)
                low = ticker.get('low', last)
                
                # Calculate 24h volatility
                if high and low and high > 0:
                    volatility = (high - low) / last
                else:
                    volatility = 0
                
                # Estimate spread
                ask = ticker.get('ask', last)
                bid = ticker.get('bid', last)
                if ask and bid and ask > 0:
                    spread = (ask - bid) / ((ask + bid) / 2)
                else:
                    spread = 0.001
                
                results.append({
                    'symbol': symbol,
                    'price': last,
                    'volume_24h': quote_volume,
                    'high_24h': high,
                    'low_24h': low,
                    'change_24h': ticker.get('percentage', 0),
                    'volatility': volatility,
                    'spread': spread
                })
            
            # Sort by volume
            results.sort(key=lambda x: x['volume_24h'], reverse=True)
            return results
            
        except Exception as e:
            log.error(f"Error fetching tickers: {e}")
            return []
    
    def _apply_basic_filters(self, tickers: List[Dict]) -> List[Dict]:
        """Apply basic filters to remove low-quality coins"""
        filtered = []
        
        for ticker in tickers:
            # Volume filter
            if ticker['volume_24h'] < self.min_volume_usdt:
                continue
            
            # Price filters
            if ticker['price'] > self.max_price:
                continue
            if ticker['price'] < self.min_price:
                continue
            
            # Spread filter
            if ticker['spread'] > self.max_spread:
                continue
            
            filtered.append(ticker)
        
        return filtered
    
    # ============================================================================
    # PRIVATE METHODS - Analysis
    # ============================================================================
    
    def _analyze_candidates_multi_tf(self, candidates: List[Dict]) -> List[CoinCandidate]:
        """
        Multi-timeframe analysis of candidates
        """
        analyzed = []
        total = len(candidates)
        
        for i, candidate in enumerate(candidates):
            symbol = candidate['symbol']
    
    # Check if we should log all coins or just every 10th
            log_all = getattr(Config, 'COIN_LOG_ALL', False)
    
            if log_all or (i + 1) % 10 == 0 or i == 0:
                log.info(f"Analyzing {i+1}/{total}: {symbol}")
            
            try:
                # Fetch multiple timeframes
                df_5m = self.data_fetcher.fetch_ohlcv(symbol, '5m', limit=100)
                df_1h = self.data_fetcher.fetch_ohlcv(symbol, '1h', limit=100)
                df_4h = self.data_fetcher.fetch_ohlcv(symbol, '4h', limit=100)
                
                if df_1h is None or df_1h.empty or len(df_1h) < 50:
                    continue
                
                # Calculate indicators for each timeframe
                if df_5m is not None and not df_5m.empty:
                    df_5m = self.tech_analyzer.calculate_all_indicators(df_5m, symbol)
                if df_1h is not None and not df_1h.empty:
                    df_1h = self.tech_analyzer.calculate_all_indicators(df_1h, symbol)
                if df_4h is not None and not df_4h.empty:
                    df_4h = self.tech_analyzer.calculate_all_indicators(df_4h, symbol)
                
                # Calculate scores for each timeframe
                trend_5m = self._calculate_trend_score(df_5m) if df_5m is not None else 0.5
                trend_1h = self._calculate_trend_score(df_1h)
                trend_4h = self._calculate_trend_score(df_4h) if df_4h is not None else 0.5
                
                momentum_5m = self._calculate_momentum_score(df_5m) if df_5m is not None else 0.5
                momentum_1h = self._calculate_momentum_score(df_1h)
                momentum_4h = self._calculate_momentum_score(df_4h) if df_4h is not None else 0.5
                
                volume_1h = self._calculate_volume_score(df_1h, candidate)
                
                # Get market regime context
                regime = self.regime_detector.detect_regime(df_1h)
                
                # Combined scores with weights
                trend_score = (trend_5m * 0.3 + trend_1h * 0.4 + trend_4h * 0.3)
                momentum_score = (momentum_5m * 0.3 + momentum_1h * 0.4 + momentum_4h * 0.3)
                
                # Adjust for market regime
                if regime['type'] == 'TREND':
                    trend_score *= 1.1
                elif regime['type'] == 'RANGE':
                    trend_score *= 0.9
                
                # Pattern score
                pattern_score = self._score_patterns(df_1h)
                
                # Final score
                final_score = (
                    trend_score * 0.4 +
                    momentum_score * 0.3 +
                    volume_1h * 0.2 +
                    pattern_score * 0.1
                )
                
                # Determine reason
                reason = self._get_selection_reason(trend_score, momentum_score, volume_1h)
                
                analyzed.append(CoinCandidate(
                    symbol=symbol,
                    price=candidate['price'],
                    volume_24h=candidate['volume_24h'],
                    volume_rank=i + 1,
                    volatility=candidate['volatility'],
                    trend_score=trend_score,
                    momentum_score=momentum_score,
                    volume_score=volume_1h,
                    pattern_score=pattern_score,
                    final_score=final_score,
                    reason=reason,
                    timestamp=datetime.now()
                ))
                
            except Exception as e:
                log.debug(f"Error analyzing {symbol}: {e}")
                continue
        
        return analyzed
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend strength score (0-1)"""
        try:
            score = 0.5
            
            if df is None or df.empty:
                return 0.5
            
            # EMA alignment
            if 'ema_8' in df and 'ema_21' in df and 'ema_50' in df:
                ema8 = df['ema_8'].iloc[-1]
                ema21 = df['ema_21'].iloc[-1]
                ema50 = df['ema_50'].iloc[-1]
                price = df['close'].iloc[-1]
                
                if price > ema8 > ema21 > ema50:
                    score += 0.3
                elif price < ema8 < ema21 < ema50:
                    score += 0.2
                elif price > ema8 and price > ema21:
                    score += 0.1
            
            # ADX trend strength
            if 'adx' in df:
                adx = df['adx'].iloc[-1]
                if adx > 30:
                    score += 0.2
                elif adx > 25:
                    score += 0.1
            
            return min(1.0, max(0.1, score))
            
        except Exception:
            return 0.5
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (0-1)"""
        try:
            score = 0.5
            
            if df is None or df.empty:
                return 0.5
            
            # RSI
            if 'rsi' in df:
                rsi = df['rsi'].iloc[-1]
                if 40 < rsi < 60:
                    score += 0.1
                elif rsi > 70 or rsi < 30:
                    score += 0.2
            
            # MACD
            if 'macd_hist' in df:
                macd_hist = df['macd_hist'].iloc[-1]
                if macd_hist > 0:
                    score += 0.1
                if abs(macd_hist) > df['macd_hist'].std():
                    score += 0.1
            
            # ROC
            if 'roc_5' in df:
                roc = df['roc_5'].iloc[-1]
                if abs(roc) > 0.02:
                    score += 0.1
            
            return min(1.0, max(0.1, score))
            
        except Exception:
            return 0.5
    
    def _calculate_volume_score(self, df: pd.DataFrame, ticker: Dict) -> float:
        """Calculate volume quality score (0-1)"""
        try:
            score = 0.5
            
            if df is None or df.empty:
                return 0.5
            
            # Volume ratio
            if 'volume_ratio' in df:
                vol_ratio = df['volume_ratio'].iloc[-1]
                if vol_ratio > 2.0:
                    score += 0.3
                elif vol_ratio > 1.5:
                    score += 0.2
                elif vol_ratio > 1.2:
                    score += 0.1
            
            # Volume rank
            volume_rank_score = max(0, 1 - (ticker.get('volume_rank', 50) / 100))
            score += volume_rank_score * 0.2
            
            # Volume consistency
            if len(df) > 20 and 'volume' in df:
                vol_std = df['volume'].iloc[-20:].std()
                vol_mean = df['volume'].iloc[-20:].mean()
                if vol_mean > 0:
                    vol_cv = vol_std / vol_mean
                    if vol_cv < 0.5:
                        score += 0.1
            
            return min(1.0, max(0.1, score))
            
        except Exception:
            return 0.5
    
    def _score_volatility(self, df: pd.DataFrame) -> float:
        """Score volatility (0-1)"""
        try:
            if df is None or df.empty:
                return 0.5
            
            if 'atr' in df.columns:
                atr_pct = df['atr'].iloc[-1] / df['close'].iloc[-1]
                # Prefer moderate volatility (0.01-0.03)
                if 0.01 < atr_pct < 0.03:
                    return 0.8
                elif atr_pct < 0.01:
                    return 0.4
                elif atr_pct > 0.05:
                    return 0.3
                else:
                    return 0.6
            return 0.5
            
        except Exception:
            return 0.5
    
    def _score_patterns(self, df: pd.DataFrame) -> float:
        """Score recent patterns using pattern detector"""
        try:
            if df is None or df.empty or len(df) < 30:
                return 0.5
            
            patterns = self.pattern_detector.detect_all_patterns(df)
            
            if patterns:
                # Boost score based on recent strong patterns
                recent_patterns = [p for p in patterns if p.age_bars < 10]
                if recent_patterns:
                    avg_reliability = np.mean([p.reliability for p in recent_patterns])
                    return 0.5 + (avg_reliability * 0.5)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _score_trend(self, df: pd.DataFrame) -> float:
        """Score trend for ranking (0-1)"""
        return self._calculate_trend_score(df)
    
    def _score_momentum(self, df: pd.DataFrame) -> float:
        """Score momentum for ranking (0-1)"""
        return self._calculate_momentum_score(df)
    
    def _score_volume(self, df: pd.DataFrame) -> float:
        """Score volume for ranking (0-1)"""
        if df is None or df.empty:
            return 0.5
        
        if 'volume_ratio' in df.columns:
            vol_ratio = df['volume_ratio'].iloc[-1]
            if vol_ratio > 1.5:
                return 0.8
            elif vol_ratio > 1.2:
                return 0.7
            elif vol_ratio > 0.8:
                return 0.6
        return 0.5
    
    def _quick_analysis_multi_tf(self, df: pd.DataFrame) -> str:
        """
        Quick analysis using multiple indicators
        Returns: 'BULL', 'BEAR', 'NEUTRAL'
        """
        try:
            if df is None or df.empty or len(df) < 30:
                return 'NEUTRAL'
            
            # Calculate indicators
            ema8 = df['close'].ewm(span=8).mean().iloc[-1]
            ema21 = df['close'].ewm(span=21).mean().iloc[-1]
            price = df['close'].iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]
            
            # Volume
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            # Score system
            score = 0
            
            # Trend contribution
            if price > ema8 > ema21:
                score += 2
            elif price < ema8 < ema21:
                score -= 2
            
            # RSI contribution
            if rsi > 60:
                score += 1
            elif rsi < 40:
                score -= 1
            
            # MACD contribution
            if macd_hist > 0:
                score += 1
            elif macd_hist < 0:
                score -= 1
            
            # Volume confirmation
            if volume_ratio > 1.5:
                score = int(score * 1.2) if score > 0 else int(score * 0.8)
            
            if score >= 2:
                return 'BULL'
            elif score <= -2:
                return 'BEAR'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'NEUTRAL'
    
    def _get_selection_reason(self, trend: float, momentum: float, volume: float) -> str:
        """Generate human-readable reason for selection"""
        reasons = []
        
        if trend > 0.7:
            reasons.append("strong trend")
        elif trend > 0.6:
            reasons.append("good trend")
        
        if momentum > 0.7:
            reasons.append("strong momentum")
        elif momentum > 0.6:
            reasons.append("good momentum")
        
        if volume > 0.7:
            reasons.append("high volume")
        elif volume > 0.6:
            reasons.append("good volume")
        
        if not reasons:
            return "average setup"
        
        return ", ".join(reasons)
    
    def _rank_and_select(self, candidates: List[CoinCandidate]) -> List[CoinCandidate]:
        """Rank candidates by final score and select top N"""
        if not candidates:
            return []
        
        # Sort by final score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Take top N
        top_coins = candidates[:self.max_results]
        
        return top_coins
    
    def _print_summary(self, coins: List[CoinCandidate]):
        """Print summary of selected coins"""
        print("\n" + "="*80)
        print(f" TOP {len(coins)} COINS SELECTED FOR ANALYSIS")
        print("="*80)
        print(f"{'#':<3} {'Symbol':<12} {'Price':<12} {'Volume':<12} {'Score':<6} {'Reason'}")
        print("-"*80)
        
        for i, coin in enumerate(coins, 1):
            volume_str = f"${coin.volume_24h/1e6:.1f}M"
            print(f"{i:<3} {coin.symbol:<12} ${coin.price:<11.6f} {volume_str:<12} {coin.final_score:.2f}  {coin.reason}")
        
        print("="*80)