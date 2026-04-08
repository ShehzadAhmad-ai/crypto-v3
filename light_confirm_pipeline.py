# light_confirm_pipeline.py - ENHANCED WITH COMPONENT SCORING
"""
Light Confirmations Pipeline (Phase 5) - Enhanced Version
Runs low-weight confirmations on signals that passed Phase 4
Uses enhanced scoring methods from all components
"""
from typing import Dict, Optional, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

from config import Config
from logger import log
from signal_model import Signal, SignalStatus
from phase5_data_fetcher import Phase5DataFetcher
from cross_asset_engine import CrossAssetEngine
from funding_oi_engine import FundingOIAnalyzer
from sentiment import Sentiment
from correlation_analyzer import CorrelationAnalyzer
from volume_analyzer import VolumeAnalyzer


class LightConfirmPipeline:
    """
    Light Confirmations Pipeline (Phase 5) - Enhanced
    - Runs low-weight confirmations using enhanced scoring methods
    - Uses 3-tier data fetching from Phase5DataFetcher
    - Applies to signals that passed Phase 4 (Smart Money)
    - Weights configurable in config.py
    """
    
    def __init__(self):
        # Initialize dedicated data fetcher for Phase 5
        self.data_fetcher = Phase5DataFetcher()
        
        # Initialize all modules (already enhanced)
        self.cross_asset = CrossAssetEngine()
        self.funding_oi = FundingOIAnalyzer()
        self.sentiment = Sentiment()
        self.correlation = CorrelationAnalyzer()
        self.volume = VolumeAnalyzer()
        
        # Load weights from config
        self.weights = Config.LIGHT_CONFIRM_WEIGHTS
        self.min_score = Config.LIGHT_CONFIRM_MIN_SCORE
        
        log.info("=" * 60)
        log.info("Light Confirm Pipeline initialized (Enhanced)")
        log.info("=" * 60)
        log.info(f"  Cross Asset: {Config.ENABLE_CROSS_ASSET}")
        log.info(f"  Funding/OI: {Config.ENABLE_FUNDING_OI}")
        log.info(f"  Sentiment: {Config.ENABLE_SENTIMENT}")
        log.info(f"  Correlation: {Config.ENABLE_CORRELATION}")
        log.info(f"  Volume: {getattr(Config, 'ENABLE_VOLUME_ANALYSIS', True)}")
        log.info(f"  Min Score: {self.min_score}")
        log.info("=" * 60)
    
    def confirm(self, signal: Signal, df: pd.DataFrame) -> Optional[Signal]:
        """
        Run light confirmations on a signal that passed Phase 4
        
        Args:
            signal: Signal that passed Phase 4 (Smart Money)
            df: OHLCV DataFrame for the symbol
        
        Returns:
            Updated signal or None if filtered out
        """
        # Only run on signals that passed Smart Money
        if signal.status != SignalStatus.SMART_MONEY_PASSED:
            return None
        
        try:
            # Store component results
            component_scores = {}
            component_summaries = {}
            component_reasons = []
            
            current_price = float(df['close'].iloc[-1])
            
            # ===== 1. CROSS-ASSET ANALYSIS =====
            if Config.ENABLE_CROSS_ASSET:
                cross_asset_result = self._analyze_cross_asset_enhanced(signal.symbol, df)
                component_scores['cross_asset'] = cross_asset_result.get('score', 0.5)
                component_summaries['cross_asset'] = cross_asset_result
                component_reasons.extend(cross_asset_result.get('reasons', []))
                log.debug(f"[LC] Cross Asset score: {component_scores['cross_asset']:.2f}")
            
            # ===== 2. FUNDING & OPEN INTEREST ANALYSIS =====
            if Config.ENABLE_FUNDING_OI:
                funding_oi_result = self._analyze_funding_oi_enhanced(signal.symbol, current_price)
                component_scores['funding_oi'] = funding_oi_result.get('score', 0.5)
                component_summaries['funding_oi'] = funding_oi_result
                component_reasons.extend(funding_oi_result.get('reasons', []))
                log.debug(f"[LC] Funding/OI score: {component_scores['funding_oi']:.2f}")
            
            # ===== 3. SENTIMENT ANALYSIS =====
            if Config.ENABLE_SENTIMENT:
                sentiment_result = self._analyze_sentiment_enhanced(signal.symbol, df)
                component_scores['sentiment'] = sentiment_result.get('score', 0.5)
                component_summaries['sentiment'] = sentiment_result
                component_reasons.extend(sentiment_result.get('reasons', []))
                log.debug(f"[LC] Sentiment score: {component_scores['sentiment']:.2f}")
            
            # ===== 4. CORRELATION ANALYSIS =====
            if Config.ENABLE_CORRELATION:
                correlation_result = self._analyze_correlation_enhanced(signal.symbol, df)
                component_scores['correlation'] = correlation_result.get('score', 0.5)
                component_summaries['correlation'] = correlation_result
                component_reasons.extend(correlation_result.get('reasons', []))
                log.debug(f"[LC] Correlation score: {component_scores['correlation']:.2f}")
            
            # ===== 5. VOLUME ANALYSIS =====
            if Config.ENABLE_VOLUME_ANALYSIS:
                volume_result = self._analyze_volume_enhanced(df)
                component_scores['volume'] = volume_result.get('score', 0.5)
                component_summaries['volume'] = volume_result
                component_reasons.extend(volume_result.get('reasons', []))
                log.debug(f"[LC] Volume score: {component_scores['volume']:.2f}")
            
            # ===== 6. CALCULATE WEIGHTED SCORE =====
            light_confirm_score = self._calculate_weighted_score(component_scores)
            
            # ===== 7. APPLY FILTER =====
            if light_confirm_score < self.min_score:
                log.debug(f"{signal.symbol}: Light confirm score {light_confirm_score:.2f} < {self.min_score}")
                signal.warning_flags.append(f"Low light confirm: {light_confirm_score:.2f}")
            
            # ===== 8. UPDATE SIGNAL =====
            signal.cross_asset_score = component_scores.get('cross_asset', 0.0)
            signal.funding_oi_score = component_scores.get('funding_oi', 0.0)
            signal.sentiment_score = component_scores.get('sentiment', 0.0)
            
            # Store metadata
            if not hasattr(signal, 'metadata'):
                signal.metadata = {}
            
            signal.metadata['light_confirm'] = {
                'score': light_confirm_score,
                'components': component_scores,
                'summaries': component_summaries,
                'reasons': component_reasons[:5]
            }
            
            # Add confirmation reasons
            signal.confirmation_reasons.extend(component_reasons[:3])
            signal.status = SignalStatus.LIGHT_CONFIRM_PASSED
            
            # Update final score (combine with existing)
            signal.final_score = (
                signal.technical_score * 0.4 +
                signal.mtf_score * 0.3 +
                signal.smart_money_score * 0.2 +
                light_confirm_score * 0.1
            )
            
            # Terminal output
            self._print_light_confirm_output(signal, component_scores, component_summaries)
            
            return signal
            
        except Exception as e:
            log.error(f"Error in light confirm pipeline for {signal.symbol}: {e}", exc_info=True)
            return signal  # Don't reject on error, just return original
    
    def _analyze_cross_asset_enhanced(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced cross-asset analysis using scoring methods"""
        try:
            # Fetch data for major assets
            major_symbols = ['BTC/USDT', 'ETH/USDT']
            price_data = {}
            
            for sym in major_symbols + [symbol]:
                temp_df = self.data_fetcher.fetch_ohlcv(sym, timeframe='1h', limit=100)
                if not temp_df.empty:
                    price_data[sym] = temp_df['close']
            
            if len(price_data) < 2:
                return {'score': 0.5, 'reasons': ['Insufficient data']}
            
            # Get BTC data for correlation
            btc_prices = price_data.get('BTC/USDT', pd.Series())
            symbol_prices = price_data.get(symbol, pd.Series())
            
            if btc_prices.empty or symbol_prices.empty:
                return {'score': 0.5, 'reasons': ['No price data']}
            
            # Get BTC dominance
            btc_dominance = self.cross_asset.analyze_btc_dominance(btc_prices)
            
            # Get sector rotation
            sector_rotation = self.cross_asset.analyze_sector_rotation(price_data)
            
            # Get correlation with BTC
            btc_correlation = self.cross_asset.analyze_correlation_with_btc(
                symbol, btc_prices, symbol_prices
            )
            
            # Get cross-asset summary
            summary = self.cross_asset.get_cross_asset_summary(
                symbol, btc_correlation, btc_dominance, sector_rotation
            )
            
            return {
                'score': summary.get('score', 0.5),
                'btc_correlation': summary.get('btc_correlation', 0),
                'btc_dominance': summary.get('btc_dominance_trend', 'NEUTRAL'),
                'altcoin_season': summary.get('altcoin_season', False),
                'strongest_sector': summary.get('strongest_sector', 'UNKNOWN'),
                'reasons': summary.get('reasons', [])
            }
            
        except Exception as e:
            log.debug(f"Cross-asset analysis error: {e}")
            return {'score': 0.5, 'reasons': ['Error in analysis']}
    
    def _analyze_funding_oi_enhanced(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Enhanced funding and OI analysis using scoring methods"""
        try:
            # Fetch funding data
            funding_rate = self.data_fetcher.fetch_funding_rate(symbol)
            funding_history = self.data_fetcher.fetch_funding_rate_history(symbol, limit=8)
            
            funding_data = {
                'current_rate': funding_rate.get('current_rate', 0),
                'history_8h': funding_history,
                'history_24h': funding_history,
                'price_change_24h': 0
            }
            
            # Fetch OI data
            oi = self.data_fetcher.fetch_open_interest(symbol)
            ls_ratio = self.data_fetcher.fetch_long_short_ratio(symbol)
            
            oi_data = {
                'current_oi': oi.get('open_interest', 0),
                'oi_1h_ago': oi.get('open_interest', 0),
                'oi_24h_ago': 0,
                'volume_24h': 0,
                'concentration': 0.5,
                'long_short_ratio': ls_ratio.get('ratio', 1.0)
            }
            
            # Get summary using enhanced methods
            summary = self.funding_oi.get_funding_oi_summary(funding_data, oi_data)
            
            return {
                'score': summary.get('score', 0.5),
                'funding_score': summary.get('funding_score', 0.5),
                'oi_score': summary.get('oi_score', 0.5),
                'funding_rate': summary.get('funding_rate', 0),
                'funding_bias': summary.get('funding_bias', 'NEUTRAL'),
                'oi_trend': summary.get('oi_trend', 'FLAT'),
                'oi_bias': summary.get('oi_bias', 'NEUTRAL'),
                'reasons': summary.get('reasons', [])
            }
            
        except Exception as e:
            log.debug(f"Funding/OI analysis error: {e}")
            return {'score': 0.5, 'reasons': ['Error in analysis']}
    
    def _analyze_sentiment_enhanced(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced sentiment analysis using scoring methods"""
        try:
            # Update technical data
            if df is not None and not df.empty:
                rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df else 50
                price = float(df['close'].iloc[-1])
                ema_fast = float(df['ema_8'].iloc[-1]) if 'ema_8' in df else price
                ema_slow = float(df['ema_50'].iloc[-1]) if 'ema_50' in df else price
                
                if price > ema_fast > ema_slow:
                    trend = 'BULL'
                elif price < ema_fast < ema_slow:
                    trend = 'BEAR'
                else:
                    trend = 'NEUTRAL'
                
                self.sentiment.update_technical_data(
                    symbol=symbol,
                    rsi=rsi,
                    price=price,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    trend=trend
                )
            
            # Update volume data
            if df is not None and not df.empty:
                volume_ratio = float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df else 1.0
                
                if len(df) >= 10:
                    recent_vol = df['volume'].iloc[-5:].mean()
                    prev_vol = df['volume'].iloc[-10:-5].mean()
                    if recent_vol > prev_vol * 1.2:
                        volume_trend = 'INCREASING'
                    elif recent_vol < prev_vol * 0.8:
                        volume_trend = 'DECREASING'
                    else:
                        volume_trend = 'STABLE'
                else:
                    volume_trend = 'STABLE'
                
                self.sentiment.update_volume_data(
                    symbol=symbol,
                    volume_ratio=volume_ratio,
                    volume_trend=volume_trend
                )
            
            # Get sentiment summary
            summary = self.sentiment.get_sentiment_summary(symbol)
            
            return {
                'score': summary.get('score', 0.5),
                'signal': summary.get('signal', 'NEUTRAL'),
                'fear_greed': summary.get('fear_greed', {}).get('value', 50),
                'fear_greed_class': summary.get('fear_greed', {}).get('classification', 'Neutral'),
                'funding_bias': summary.get('funding', {}).get('bias', 'NEUTRAL'),
                'technical_trend': summary.get('technical', {}).get('trend', 'NEUTRAL'),
                'volume_trend': summary.get('volume', {}).get('trend', 'NEUTRAL'),
                'reasons': summary.get('reasons', [])
            }
            
        except Exception as e:
            log.debug(f"Sentiment analysis error: {e}")
            return {'score': 0.5, 'reasons': ['Error in analysis']}
    
    def _analyze_correlation_enhanced(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced correlation analysis using scoring methods"""
        try:
            # Fetch BTC data
            btc_df = self.data_fetcher.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
            
            if btc_df.empty:
                return {'score': 0.5, 'reasons': ['No BTC data']}
            
            # Get correlation summary
            summary = self.correlation.get_correlation_summary(
                symbol, btc_df['close'], df['close']
            )
            
            return {
                'score': summary.get('score', 0.5),
                'correlation': summary.get('correlation', 0),
                'correlation_strength': summary.get('correlation_strength', 'NEUTRAL'),
                'beta': summary.get('beta', 1.0),
                'altcoin_season': summary.get('btc_dominance', {}).get('altcoin_season', False),
                'sector': summary.get('sector', {}),
                'reasons': summary.get('reasons', [])
            }
            
        except Exception as e:
            log.debug(f"Correlation analysis error: {e}")
            return {'score': 0.5, 'reasons': ['Error in analysis']}
    
    def _analyze_volume_enhanced(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced volume analysis using scoring methods"""
        try:
            # Get volume summary
            summary = self.volume.get_volume_summary(df)
            
            return {
                'score': summary.get('combined_score', 0.5),
                'volume_ratio': summary.get('volume_ratio', 1.0),
                'volume_trend': summary.get('volume_trend', 'NEUTRAL'),
                'divergence': summary.get('divergence'),
                'reasons': summary.get('reasons', [])
            }
            
        except Exception as e:
            log.debug(f"Volume analysis error: {e}")
            return {'score': 0.5, 'reasons': ['Error in analysis']}
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted light confirmation score"""
        if not component_scores:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            weight = self.weights.get(component, 0.15)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return min(0.99, max(0.01, total_score / total_weight))
        return 0.5
    
    def _print_light_confirm_output(self, signal: Signal, scores: Dict, summaries: Dict):
        """Print formatted light confirm output"""
        log.info("\n" + "=" * 70)
        log.info(f"[LIGHT CONFIRM] {signal.symbol} {signal.timeframe}")
        log.info("=" * 70)
        log.info(f"Score: {signal.final_score:.2f}")
        log.info(f"Status: {'PASSED' if signal.status == SignalStatus.LIGHT_CONFIRM_PASSED else 'WARNING'}")
        log.info("-" * 50)
        
        # Component scores
        log.info("Component Scores:")
        for component, score in scores.items():
            log.info(f"  {component:15s}: {score:.2f}")
        
        log.info("-" * 50)
        log.info("Key Signals:")
        
        # Cross Asset
        if 'cross_asset' in summaries:
            ca = summaries['cross_asset']
            log.info(f"  Cross Asset: BTC Corr {ca.get('btc_correlation', 0):.2f} | Alt Season: {ca.get('altcoin_season', False)}")
        
        # Funding/OI
        if 'funding_oi' in summaries:
            fo = summaries['funding_oi']
            log.info(f"  Funding/OI: {fo.get('funding_bias', 'NEUTRAL')} funding | {fo.get('oi_trend', 'FLAT')} OI")
        
        # Sentiment
        if 'sentiment' in summaries:
            se = summaries['sentiment']
            log.info(f"  Sentiment: {se.get('signal', 'NEUTRAL')} | F&G: {se.get('fear_greed', 50)}")
        
        # Correlation
        if 'correlation' in summaries:
            co = summaries['correlation']
            log.info(f"  Correlation: {co.get('correlation_strength', 'NEUTRAL')} | Beta: {co.get('beta', 1.0):.2f}")
        
        # Volume
        if 'volume' in summaries:
            vo = summaries['volume']
            log.info(f"  Volume: {vo.get('volume_trend', 'NEUTRAL')} | Ratio: {vo.get('volume_ratio', 1.0):.1f}x")
        
        log.info("=" * 70)