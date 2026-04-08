# htf_analyzer.py - Multi-Timeframe Analysis Expert
"""
Higher Timeframe (HTF) Analysis Expert
- Analyzes multiple higher timeframes (10min, 15min, 30min, 1h, 4h, 1d)
- Determines HTF trend direction and strength
- Calculates alignment with main timeframe (5min)
- Provides confidence boost/penalty based on alignment
- All thresholds from ta_config.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

# Import configuration and core classes
from .ta_config import *
from .ta_core import HTFResult, HTFAnalysisResult

# Try to import logger, fallback to print if not available
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class HTFAnalyzer:
    """
    Multi-Timeframe Analysis Expert
    Analyzes higher timeframes and determines alignment with main timeframe
    """
    
    def __init__(self):
        """Initialize the HTF analyzer with thresholds from config"""
        self.active_timeframes = ACTIVE_HTFS.copy()
        self.timeframe_weights = HTF_WEIGHTS.copy()
        self.alignment_boost = HTF_ALIGNMENT_BOOST
        self.conflict_penalty = HTF_CONFLICT_PENALTY
        self.strong_alignment_boost = HTF_STRONG_ALIGNMENT_BOOST
        self.strong_conflict_penalty = HTF_STRONG_CONFLICT_PENALTY
        
        log.info(f"HTFAnalyzer initialized with timeframes: {self.active_timeframes}")
    
    # ============================================================================
    # MAIN PUBLIC METHODS
    # ============================================================================
    
    def analyze_htfs(self, main_df: pd.DataFrame, htf_data: Dict[str, pd.DataFrame], 
                     main_timeframe: str = "5min", symbol: str = "") -> HTFAnalysisResult:
        """
        Analyze all higher timeframes and calculate alignment
        
        Args:
            main_df: Main timeframe DataFrame (5min)
            htf_data: Dictionary of timeframe -> DataFrame
            main_timeframe: Main timeframe string (e.g., "5min")
            symbol: Symbol name for logging
        
        Returns:
            HTFAnalysisResult with complete multi-timeframe analysis
        """
        if main_df is None or main_df.empty:
            log.warning("Main DataFrame is empty, cannot analyze HTFs")
            return self._default_htf_result(main_timeframe)
        
        try:
            # Get main timeframe trend direction
            main_trend = self._get_timeframe_trend(main_df)
            
            # Analyze each active HTF
            htf_results: Dict[str, HTFResult] = {}
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            weighted_alignment_sum = 0.0
            total_weight = 0.0
            
            for tf in self.active_timeframes:
                if tf in htf_data and htf_data[tf] is not None and not htf_data[tf].empty:
                    # Analyze this HTF
                    htf_result = self._analyze_single_htf(htf_data[tf], tf, main_trend)
                    htf_results[tf] = htf_result
                    
                    # Count directions
                    if htf_result.direction == "BULLISH":
                        bullish_count += 1
                    elif htf_result.direction == "BEARISH":
                        bearish_count += 1
                    else:
                        neutral_count += 1
                    
                    # Calculate weighted alignment
                    weight = self.timeframe_weights.get(tf, 0.1)
                    weighted_alignment_sum += htf_result.alignment_score * weight
                    total_weight += weight
            
            # Calculate overall alignment
            overall_alignment = weighted_alignment_sum / total_weight if total_weight > 0 else 0.5
            
            # Determine if aligned with main timeframe
            is_aligned = overall_alignment > 0.6
            is_strongly_aligned = overall_alignment > 0.8
            is_conflicted = overall_alignment < 0.4
            
            # Calculate boost/penalty
            alignment_boost = 0.0
            conflict_penalty = 0.0
            
            if is_strongly_aligned:
                alignment_boost = self.strong_alignment_boost
            elif is_aligned:
                alignment_boost = self.alignment_boost
            elif is_conflicted:
                conflict_penalty = self.strong_conflict_penalty if overall_alignment < 0.3 else self.conflict_penalty
            
            # Create result
            result = HTFAnalysisResult(
                main_timeframe=main_timeframe,
                analyzed_timeframes=[tf for tf in self.active_timeframes if tf in htf_results],
                htf_results=htf_results,
                overall_alignment=round(overall_alignment, 3),
                weighted_alignment=round(weighted_alignment_sum / total_weight if total_weight > 0 else 0.5, 3),
                bullish_htf_count=bullish_count,
                bearish_htf_count=bearish_count,
                neutral_htf_count=neutral_count,
                is_aligned=is_aligned,
                alignment_boost=round(alignment_boost, 3),
                conflict_penalty=round(conflict_penalty, 3),
            )
            
            log.debug(f"HTF Analysis: {bullish_count}B / {bearish_count}B / {neutral_count}N | "
                     f"Alignment: {overall_alignment:.2f} | Boost: {alignment_boost:.2f}")
            
            return result
            
        except Exception as e:
            log.error(f"Error in HTF analysis: {e}")
            return self._default_htf_result(main_timeframe)
    
    def get_alignment_adjustment(self, htf_result: HTFAnalysisResult) -> Tuple[float, float]:
        """
        Get confidence adjustment based on HTF alignment
        
        Returns:
            Tuple of (boost, penalty) - apply boost to confidence, penalty to confidence
        """
        return htf_result.alignment_boost, htf_result.conflict_penalty
    
    def is_htf_bullish(self, htf_result: HTFAnalysisResult, threshold: float = 0.6) -> bool:
        """
        Check if HTFs are overall bullish
        
        Args:
            htf_result: HTFAnalysisResult from analyze_htfs
            threshold: Alignment threshold (0.6 = 60% bullish)
        
        Returns:
            True if HTFs are overall bullish
        """
        if not htf_result:
            return False
        
        bullish_ratio = htf_result.bullish_htf_count / len(htf_result.analyzed_timeframes) if htf_result.analyzed_timeframes else 0
        return bullish_ratio >= threshold
    
    def is_htf_bearish(self, htf_result: HTFAnalysisResult, threshold: float = 0.6) -> bool:
        """
        Check if HTFs are overall bearish
        
        Args:
            htf_result: HTFAnalysisResult from analyze_htfs
            threshold: Alignment threshold (0.6 = 60% bearish)
        
        Returns:
            True if HTFs are overall bearish
        """
        if not htf_result:
            return False
        
        bearish_ratio = htf_result.bearish_htf_count / len(htf_result.analyzed_timeframes) if htf_result.analyzed_timeframes else 0
        return bearish_ratio >= threshold
    
    def get_strongest_htf_direction(self, htf_result: HTFAnalysisResult) -> Tuple[str, float]:
        """
        Get the strongest HTF direction with confidence
        
        Returns:
            Tuple of (direction, confidence) where direction is BULLISH/BEARISH/NEUTRAL
        """
        if not htf_result or not htf_result.htf_results:
            return "NEUTRAL", 0.0
        
        # Weighted by timeframe importance
        bullish_weight = 0.0
        bearish_weight = 0.0
        total_weight = 0.0
        
        for tf, result in htf_result.htf_results.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            if result.direction == "BULLISH":
                bullish_weight += weight * result.strength
            elif result.direction == "BEARISH":
                bearish_weight += weight * result.strength
            total_weight += weight
        
        if total_weight > 0:
            bullish_score = bullish_weight / total_weight
            bearish_score = bearish_weight / total_weight
            
            if bullish_score > bearish_score and bullish_score > 0.5:
                return "BULLISH", bullish_score
            elif bearish_score > bullish_score and bearish_score > 0.5:
                return "BEARISH", bearish_score
        
        return "NEUTRAL", 0.0
    
    # ============================================================================
    # PRIVATE ANALYSIS METHODS
    # ============================================================================
    
    def _analyze_single_htf(self, df: pd.DataFrame, timeframe: str, 
                            main_trend: str) -> HTFResult:
        """
        Analyze a single higher timeframe
        
        Args:
            df: HTF DataFrame
            timeframe: Timeframe string (e.g., "1h")
            main_trend: Main timeframe trend direction
        
        Returns:
            HTFResult for this timeframe
        """
        if df is None or df.empty or len(df) < 50:
            return HTFResult(
                timeframe=timeframe,
                direction="NEUTRAL",
                strength=0.0,
                confidence=0.0,
                alignment_score=0.5,
            )
        
        try:
            # Calculate HTF indicators
            rsi = self._calculate_rsi(df)
            adx = self._calculate_adx(df)
            ema_trend = self._calculate_ema_trend(df)
            
            # Determine HTF direction
            direction, strength, confidence = self._determine_htf_direction(df, rsi, adx, ema_trend)
            
            # Calculate alignment with main timeframe
            alignment_score = self._calculate_alignment(direction, main_trend)
            
            return HTFResult(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                confidence=confidence,
                alignment_score=alignment_score,
                rsi=rsi,
                adx=adx,
                ema_trend=ema_trend,
            )
            
        except Exception as e:
            log.debug(f"Error analyzing HTF {timeframe}: {e}")
            return HTFResult(
                timeframe=timeframe,
                direction="NEUTRAL",
                strength=0.0,
                confidence=0.0,
                alignment_score=0.5,
            )
    
    def _get_timeframe_trend(self, df: pd.DataFrame) -> str:
        """
        Determine trend direction for a timeframe
        
        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if df is None or df.empty or len(df) < 50:
            return "NEUTRAL"
        
        try:
            # Use multiple factors for trend detection
            scores = []
            
            # Factor 1: Price vs EMAs
            if all(col in df for col in ['ema_8', 'ema_21', 'ema_50']):
                price = df['close'].iloc[-1]
                e8 = df['ema_8'].iloc[-1]
                e21 = df['ema_21'].iloc[-1]
                e50 = df['ema_50'].iloc[-1]
                
                if price > e8 > e21 > e50:
                    scores.append(1.0)
                elif price > e8 and price > e21:
                    scores.append(0.5)
                elif price < e8 < e21 < e50:
                    scores.append(-1.0)
                elif price < e8 and price < e21:
                    scores.append(-0.5)
                else:
                    scores.append(0.0)
            
            # Factor 2: Price slope
            slope = self._calculate_slope(df)
            if slope > 0.005:
                scores.append(0.5)
            elif slope > 0.002:
                scores.append(0.25)
            elif slope < -0.005:
                scores.append(-0.5)
            elif slope < -0.002:
                scores.append(-0.25)
            else:
                scores.append(0.0)
            
            # Factor 3: RSI
            if 'rsi' in df:
                rsi = df['rsi'].iloc[-1]
                if rsi > 60:
                    scores.append(0.5)
                elif rsi > 55:
                    scores.append(0.25)
                elif rsi < 40:
                    scores.append(-0.5)
                elif rsi < 45:
                    scores.append(-0.25)
                else:
                    scores.append(0.0)
            
            # Calculate average score
            avg_score = np.mean(scores) if scores else 0.0
            
            if avg_score > 0.3:
                return "BULLISH"
            elif avg_score < -0.3:
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            log.debug(f"Error getting timeframe trend: {e}")
            return "NEUTRAL"
    
    def _determine_htf_direction(self, df: pd.DataFrame, rsi: float, 
                                   adx: float, ema_trend: str) -> Tuple[str, float, float]:
        """
        Determine HTF direction, strength, and confidence
        
        Returns:
            Tuple of (direction, strength, confidence)
        """
        # Collect signals
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Signal 1: EMA trend
        if ema_trend == "BULLISH":
            bullish_signals += 1
        elif ema_trend == "BEARISH":
            bearish_signals += 1
        total_signals += 1
        
        # Signal 2: RSI
        if rsi:
            if rsi > 60:
                bullish_signals += 1
            elif rsi < 40:
                bearish_signals += 1
            total_signals += 1
        
        # Signal 3: Price position
        price = df['close'].iloc[-1]
        price_20_ago = df['close'].iloc[-20] if len(df) >= 20 else price
        price_change = (price - price_20_ago) / price_20_ago if price_20_ago > 0 else 0
        
        if price_change > 0.02:
            bullish_signals += 1
        elif price_change < -0.02:
            bearish_signals += 1
        total_signals += 1
        
        # Signal 4: Higher highs / lower lows
        if len(df) >= 10:
            recent_highs = df['high'].iloc[-5:].max()
            previous_highs = df['high'].iloc[-10:-5].max()
            recent_lows = df['low'].iloc[-5:].min()
            previous_lows = df['low'].iloc[-10:-5].min()
            
            if recent_highs > previous_highs and recent_lows > previous_lows:
                bullish_signals += 1
            elif recent_highs < previous_highs and recent_lows < previous_lows:
                bearish_signals += 1
            total_signals += 1
        
        # Calculate direction and strength
        if total_signals > 0:
            bullish_ratio = bullish_signals / total_signals
            bearish_ratio = bearish_signals / total_signals
            
            if bullish_ratio > bearish_ratio and bullish_ratio > 0.5:
                direction = "BULLISH"
                strength = bullish_ratio
            elif bearish_ratio > bullish_ratio and bearish_ratio > 0.5:
                direction = "BEARISH"
                strength = bearish_ratio
            else:
                direction = "NEUTRAL"
                strength = 0.5
        else:
            direction = "NEUTRAL"
            strength = 0.5
        
        # Calculate confidence based on ADX
        confidence = 0.5
        if adx:
            if adx > ADX_STRONG_TREND:
                confidence = 0.8
            elif adx > ADX_TRENDING:
                confidence = 0.65
            elif adx < ADX_WEAK_TREND:
                confidence = 0.4
        
        return direction, strength, confidence
    
    def _calculate_alignment(self, htf_direction: str, main_trend: str) -> float:
        """
        Calculate alignment score between HTF and main timeframe
        
        Returns:
            Score from 0 to 1 (1 = perfectly aligned, 0 = perfectly opposed)
        """
        if htf_direction == "NEUTRAL" or main_trend == "NEUTRAL":
            return 0.5
        
        if htf_direction == main_trend:
            return 1.0
        else:
            return 0.0
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate RSI for HTF"""
        try:
            if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]):
                return float(df['rsi'].iloc[-1])
            
            # Calculate RSI
            if len(df) >= RSI_PERIOD + 1:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
            return None
        except Exception:
            return None
    
    def _calculate_adx(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate ADX for HTF"""
        try:
            if 'adx' in df and not pd.isna(df['adx'].iloc[-1]):
                return float(df['adx'].iloc[-1])
            
            # Simplified ADX calculation
            if len(df) >= ADX_PERIOD + 1:
                high = df['high']
                low = df['low']
                close = df['close']
                
                tr1 = high - low
                tr2 = (high - close.shift()).abs()
                tr3 = (low - close.shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                atr = tr.rolling(window=ADX_PERIOD).mean()
                
                up_move = high - high.shift()
                down_move = low.shift() - low
                
                plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
                minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
                
                plus_di = 100 * plus_dm.rolling(window=ADX_PERIOD).mean() / atr
                minus_di = 100 * minus_dm.rolling(window=ADX_PERIOD).mean() / atr
                
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
                adx = dx.rolling(window=ADX_PERIOD).mean()
                
                return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None
            
            return None
        except Exception:
            return None
    
    def _calculate_ema_trend(self, df: pd.DataFrame) -> str:
        """Calculate EMA trend direction"""
        try:
            if all(col in df for col in ['ema_8', 'ema_21', 'ema_50']):
                e8 = df['ema_8'].iloc[-1]
                e21 = df['ema_21'].iloc[-1]
                e50 = df['ema_50'].iloc[-1]
                
                if e8 > e21 > e50:
                    return "BULLISH"
                elif e8 < e21 < e50:
                    return "BEARISH"
            
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"
    
    def _calculate_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate price slope"""
        try:
            if len(df) >= period:
                return (df['close'].iloc[-1] - df['close'].iloc[-period]) / (df['close'].iloc[-period] + 1e-10)
            return 0.0
        except Exception:
            return 0.0
    
    def _default_htf_result(self, main_timeframe: str) -> HTFAnalysisResult:
        """Default HTF result when analysis fails"""
        return HTFAnalysisResult(
            main_timeframe=main_timeframe,
            analyzed_timeframes=[],
            htf_results={},
            overall_alignment=0.5,
            weighted_alignment=0.5,
            bullish_htf_count=0,
            bearish_htf_count=0,
            neutral_htf_count=0,
            is_aligned=False,
            alignment_boost=0.0,
            conflict_penalty=0.0,
        )


# ============================================================================
# SIMPLE WRAPPER FUNCTIONS
# ============================================================================

def analyze_timeframes(main_df: pd.DataFrame, htf_data: Dict[str, pd.DataFrame], 
                       main_timeframe: str = "5min") -> HTFAnalysisResult:
    """
    Simple wrapper function for HTF analysis
    """
    analyzer = HTFAnalyzer()
    return analyzer.analyze_htfs(main_df, htf_data, main_timeframe)


def get_htf_alignment_boost(htf_result: HTFAnalysisResult) -> float:
    """
    Get the confidence boost from HTF alignment
    """
    if htf_result:
        return htf_result.alignment_boost
    return 0.0


def get_htf_conflict_penalty(htf_result: HTFAnalysisResult) -> float:
    """
    Get the confidence penalty from HTF conflict
    """
    if htf_result:
        return htf_result.conflict_penalty
    return 0.0