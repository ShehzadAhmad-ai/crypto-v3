"""
context_engine.py
Layer 3: Context Engine for Price Action Expert V3.5

Provides market context using pure price action:
- Trend stage detection (early/mid/late/exhaustion/consolidation)
- Session detection and weighting
- Multi-timeframe trend alignment
- Key level proximity detection
- Market structure bias (from candles only)

Can optionally integrate with:
- market_regime.py (for regime context)
- market_structure.py (for swing points)
- price_location.py (for support/resistance)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Import configuration
from .price_action_config import (
    SESSION_WEIGHTS,
    SESSION_HOURS,
    TREND_STAGE_THRESHOLDS,
    TREND_STAGE_RISK_MULTIPLIER,
    KEY_LEVEL_TOLERANCE_ATR
)


@dataclass
class MarketContext:
    """
    Complete market context for pattern enhancement
    """
    # Trend stage
    trend_stage: str                     # 'early', 'mid', 'late', 'exhaustion', 'consolidation'
    trend_stage_confidence: float        # 0-1
    risk_multiplier: float               # Based on trend stage
    
    # Session
    session: str                         # 'asia', 'london', 'new_york', etc.
    session_weight: float                # 0-1 based on session
    
    # MTF alignment
    mtf_alignment_score: float           # 0-1
    mtf_trends: Dict[str, str]           # {'15m': 'bullish', '1h': 'neutral', '4h': 'bullish'}
    
    # Key levels
    at_key_level: bool
    key_level_type: str                  # 'support', 'resistance'
    key_level_price: Optional[float]
    key_level_strength: float            # 0-1
    
    # Market structure
    structure_bias: str                  # 'bullish', 'bearish', 'neutral'
    structure_strength: float            # 0-1
    
    # Volatility context
    volatility_state: str                # 'low', 'normal', 'high', 'extreme'
    atr_pct: float                       # ATR as percentage of price
    
    # Human-readable summary
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            'trend_stage': self.trend_stage,
            'trend_stage_confidence': self.trend_stage_confidence,
            'risk_multiplier': self.risk_multiplier,
            'session': self.session,
            'session_weight': self.session_weight,
            'mtf_alignment_score': self.mtf_alignment_score,
            'mtf_trends': self.mtf_trends,
            'at_key_level': self.at_key_level,
            'key_level_type': self.key_level_type,
            'key_level_price': self.key_level_price,
            'key_level_strength': self.key_level_strength,
            'structure_bias': self.structure_bias,
            'structure_strength': self.structure_strength,
            'volatility_state': self.volatility_state,
            'atr_pct': self.atr_pct,
            'summary': self.summary
        }


class ContextEngine:
    """
    Advanced context engine using pure price action
    
    Features:
    - Trend stage detection from candle sequence
    - Session detection and weighting
    - Multi-timeframe trend analysis
    - Key level detection from price action
    - Market structure from swing points
    - Optional integration with existing modules
    """
    
    def __init__(self):
        """Initialize the context engine"""
        self.last_trend_stage = None
        self.trend_stage_history = []
    
    # =========================================================
    # TREND STAGE DETECTION (PURE PRICE ACTION)
    # =========================================================
    
    def detect_trend_stage(self, df: pd.DataFrame, 
                           candles: Optional[List] = None) -> Tuple[str, float]:
        """
        Detect trend stage using only candles
        
        Stages:
        - early: Trend just starting, strong momentum
        - mid: Established trend, healthy pullbacks
        - late: Trend mature, increasing wicks, smaller bodies
        - exhaustion: Trend exhausted, reversal patterns appearing
        - consolidation: No clear trend, ranging
        
        Returns:
            (stage, confidence)
        """
        if df is None or df.empty or len(df) < 30:
            return 'consolidation', 0.5
        
        try:
            # Use last 20-30 candles for analysis
            recent = df.iloc[-30:].copy()
            n = len(recent)
            
            # Calculate candle metrics
            bodies = abs(recent['close'] - recent['open'])
            upper_wicks = recent['high'] - np.maximum(recent['close'], recent['open'])
            lower_wicks = np.minimum(recent['close'], recent['open']) - recent['low']
            ranges = recent['high'] - recent['low']
            
            # Direction counts
            bullish = (recent['close'] > recent['open']).sum()
            bearish = (recent['close'] < recent['open']).sum()
            total = bullish + bearish
            bullish_ratio = bullish / total if total > 0 else 0.5
            
            # Wick trends (last 10 vs previous 10)
            last_10_upper = upper_wicks.iloc[-10:].mean()
            prev_10_upper = upper_wicks.iloc[-20:-10].mean() if n >= 20 else last_10_upper
            last_10_lower = lower_wicks.iloc[-10:].mean()
            prev_10_lower = lower_wicks.iloc[-20:-10].mean() if n >= 20 else last_10_lower
            
            wick_increasing = (last_10_upper > prev_10_upper * 1.1 or 
                              last_10_lower > prev_10_lower * 1.1)
            wick_decreasing = (last_10_upper < prev_10_upper * 0.9 and 
                              last_10_lower < prev_10_lower * 0.9)
            
            # Body trend (acceleration/deceleration)
            bodies_last_5 = bodies.iloc[-5:].mean()
            bodies_prev_5 = bodies.iloc[-10:-5].mean() if n >= 10 else bodies_last_5
            body_increasing = bodies_last_5 > bodies_prev_5 * 1.15
            body_decreasing = bodies_last_5 < bodies_prev_5 * 0.85
            
            # Range trend (expanding/contracting)
            ranges_last_5 = ranges.iloc[-5:].mean()
            ranges_prev_5 = ranges.iloc[-10:-5].mean() if n >= 10 else ranges_last_5
            range_expanding = ranges_last_5 > ranges_prev_5 * 1.15
            range_contracting = ranges_last_5 < ranges_prev_5 * 0.85
            
            # Recent price movement
            price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            
            # Determine stage
            stage = 'consolidation'
            confidence = 0.5
            

            # Early trend detection
            bearish_ratio = 1 - bullish_ratio  # Calculate bearish ratio

            if bullish_ratio >= TREND_STAGE_THRESHOLDS['early']['up_days_min']:
                # Bullish early trend
                if wick_decreasing and body_increasing:
                    stage = 'early'
                    confidence = 0.75
                elif body_increasing:
                    stage = 'early'
                    confidence = 0.65


            elif bearish_ratio >= TREND_STAGE_THRESHOLDS['early']['up_days_min']:
                # Bearish early trend
                if wick_decreasing and body_increasing:
                    stage = 'early'
                    confidence = 0.75
                elif body_increasing:
                    stage = 'early'
                    confidence = 0.65

                    
            # Mid trend
            elif 0.55 <= bullish_ratio <= 0.75:
                if not wick_increasing and not body_decreasing:
                    stage = 'mid'
                    confidence = 0.70
                    
            elif 0.25 <= bullish_ratio <= 0.45:
                if not wick_increasing and not body_decreasing:
                    stage = 'mid'
                    confidence = 0.70
                    
                # Late trend

            elif bullish_ratio >= 0.5 and wick_increasing:
                stage = 'late'
                confidence = 0.60
                
            elif bearish_ratio >= 0.5 and wick_increasing:
                stage = 'late'
                confidence = 0.60
                
            # Exhaustion
            if wick_increasing and range_expanding and body_decreasing:
                stage = 'exhaustion'
                confidence = 0.65
                
            # Consolidation
            if range_contracting and abs(price_change) < 0.02:
                stage = 'consolidation'
                confidence = 0.70
                
            # Adjust confidence based on trend clarity
            if abs(bullish_ratio - 0.5) > 0.3:
                confidence += 0.10
                
            # Store history
            self.trend_stage_history.append({'stage': stage, 'confidence': confidence})
            if len(self.trend_stage_history) > 10:
                self.trend_stage_history.pop(0)
                
            return stage, min(0.95, max(0.3, confidence))
            
        except Exception as e:
            return 'consolidation', 0.5
    
    def get_risk_multiplier(self, trend_stage: str) -> float:
        """Get risk multiplier based on trend stage"""
        return TREND_STAGE_RISK_MULTIPLIER.get(trend_stage, 1.0)
    
    # =========================================================
    # SESSION DETECTION
    # =========================================================
    
    def detect_session(self, timestamp: datetime) -> Tuple[str, float]:
        """
        Detect trading session based on UTC time
        
        Returns:
            (session_name, weight)
        """
        hour = timestamp.hour
        
        for session, (start, end) in SESSION_HOURS.items():
            if start <= hour < end:
                return session, SESSION_WEIGHTS.get(session, 0.8)
                
        return 'asia', SESSION_WEIGHTS.get('asia', 0.6)
    
    # =========================================================
    # MULTI-TIMEFRAME TREND DETECTION (PURE CANDLES)
    # =========================================================
    
    def detect_mtf_trend(self, htf_df: pd.DataFrame, period: int = 5) -> str:
        """
        Detect trend on higher timeframe using pure candles
        
        Args:
            htf_df: Higher timeframe DataFrame
            period: Number of candles to analyze
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if htf_df is None or htf_df.empty or len(htf_df) < period:
            return 'neutral'
        
        recent = htf_df.iloc[-period:]
        
        # Count bullish vs bearish
        bullish = (recent['close'] > recent['open']).sum()
        bearish = (recent['close'] < recent['open']).sum()
        
        if bullish > bearish + 1:
            # Check for higher highs
            highs_increasing = recent['high'].iloc[-1] > recent['high'].iloc[-period]
            if highs_increasing:
                return 'bullish'
            return 'neutral'
        elif bearish > bullish + 1:
            # Check for lower lows
            lows_decreasing = recent['low'].iloc[-1] < recent['low'].iloc[-period]
            if lows_decreasing:
                return 'bearish'
            return 'neutral'
        
        return 'neutral'
    
    def calculate_mtf_alignment(self, trends: Dict[str, str]) -> float:
        """
        Calculate MTF alignment score based on trend agreement
        
        Args:
            trends: Dictionary of timeframe -> trend direction
        
        Returns:
            Alignment score 0-1
        """
        if not trends:
            return 0.5
        
        # Count bullish vs bearish
        bullish_count = sum(1 for t in trends.values() if t == 'bullish')
        bearish_count = sum(1 for t in trends.values() if t == 'bearish')
        neutral_count = len(trends) - bullish_count - bearish_count
        
        total = len(trends)
        
        if bullish_count == total:
            return 1.0
        elif bearish_count == total:
            return 1.0
        elif bullish_count > bearish_count and neutral_count == 0:
            return 0.8
        elif bearish_count > bullish_count and neutral_count == 0:
            return 0.8
        elif bullish_count > bearish_count:
            return 0.6
        elif bearish_count > bullish_count:
            return 0.6
        elif neutral_count >= 2:
            return 0.4
        
        return 0.5
    
    # =========================================================
    # KEY LEVEL DETECTION (FROM PRICE ACTION)
    # =========================================================
    
    def detect_key_levels(self, df: pd.DataFrame, current_price: float, 
                          atr: float) -> Tuple[Optional[float], Optional[float], float]:
        """
        Detect nearest support and resistance from price action
        
        Returns:
            (nearest_support, nearest_resistance, level_strength)
        """
        if df is None or df.empty or len(df) < 30:
            return None, None, 0.0
        
        recent = df.iloc[-50:]
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Find swing highs and lows (local extrema)
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(recent) - 5):
            if highs[i] == max(highs[i-5:i+6]):
                swing_highs.append(highs[i])
            if lows[i] == min(lows[i-5:i+6]):
                swing_lows.append(lows[i])
        
        # Cluster nearby levels (within 0.5% tolerance)
        def cluster_levels(levels, tolerance_pct=0.005):
            if not levels:
                return []
            levels = sorted(levels)
            clustered = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] <= tolerance_pct:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            clustered.append(np.mean(current_cluster))
            return clustered
        
        support_levels = cluster_levels([l for l in swing_lows if l < current_price])
        resistance_levels = cluster_levels([h for h in swing_highs if h > current_price])
        
        # Find nearest
        nearest_support = max(support_levels) if support_levels else None
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        
        # Calculate level strength based on touches
        def level_strength(level, all_levels):
            if level is None:
                return 0.0
            # Count how many times price touched this level
            tolerance = level * 0.005
            touches = sum(1 for l in all_levels if abs(l - level) <= tolerance)
            return min(1.0, touches / 5)
        
        strength = 0.0
        if nearest_support:
            strength = max(strength, level_strength(nearest_support, swing_lows))
        if nearest_resistance:
            strength = max(strength, level_strength(nearest_resistance, swing_highs))
        
        return nearest_support, nearest_resistance, strength
    
    def is_at_key_level(self, current_price: float, level: Optional[float], 
                        atr: float) -> bool:
        """Check if price is at a key level"""
        if level is None:
            return False
        tolerance = atr * KEY_LEVEL_TOLERANCE_ATR
        return abs(current_price - level) <= tolerance
    
    # =========================================================
    # MARKET STRUCTURE DETECTION (PURE PRICE ACTION)
    # =========================================================
    
    def detect_structure_bias(self, df: pd.DataFrame, 
                              candles: Optional[List] = None) -> Tuple[str, float]:
        """
        Detect market structure bias from swing points
        
        Returns:
            (bias, strength) where bias is 'bullish', 'bearish', or 'neutral'
        """
        if df is None or df.empty or len(df) < 30:
            return 'neutral', 0.5
        
        # Find swing highs and lows
        highs = df['high'].values
        lows = df['low'].values
        n = len(highs)
        
        swing_highs = []
        swing_lows = []
        
        for i in range(10, n - 10):
            if highs[i] == max(highs[i-10:i+11]):
                swing_highs.append({'index': i, 'price': highs[i]})
            if lows[i] == min(lows[i-10:i+11]):
                swing_lows.append({'index': i, 'price': lows[i]})
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'neutral', 0.5
        
        # Get last 3 swings of each type
        last_highs = swing_highs[-3:]
        last_lows = swing_lows[-3:]
        
        # Check for higher highs and higher lows
        hh = len(last_highs) >= 2 and last_highs[-1]['price'] > last_highs[-2]['price']
        hl = len(last_lows) >= 2 and last_lows[-1]['price'] > last_lows[-2]['price']
        
        # Check for lower highs and lower lows
        lh = len(last_highs) >= 2 and last_highs[-1]['price'] < last_highs[-2]['price']
        ll = len(last_lows) >= 2 and last_lows[-1]['price'] < last_lows[-2]['price']
        
        if hh and hl:
            bias = 'bullish'
            strength = 0.7 + (0.1 if len(last_highs) >= 3 else 0)
        elif lh and ll:
            bias = 'bearish'
            strength = 0.7 + (0.1 if len(last_lows) >= 3 else 0)
        else:
            bias = 'neutral'
            strength = 0.5
        
        return bias, min(0.95, strength)
    
    # =========================================================
    # VOLATILITY CONTEXT (FROM CANDLES)
    # =========================================================
    
    def detect_volatility_state(self, atr_pct: float) -> str:
        """
        Detect volatility state from ATR percentage
        
        Args:
            atr_pct: ATR as percentage of price
        
        Returns:
            'low', 'normal', 'high', or 'extreme'
        """
        if atr_pct < 0.005:
            return 'low'
        elif atr_pct < 0.015:
            return 'normal'
        elif atr_pct < 0.03:
            return 'high'
        else:
            return 'extreme'
    
    # =========================================================
    # COMPLETE CONTEXT ANALYSIS
    # =========================================================
    
    def analyze(self, df: pd.DataFrame, 
                htf_data: Optional[Dict[str, pd.DataFrame]] = None,
                regime_data: Optional[Dict] = None,
                structure_data: Optional[Dict] = None,
                sr_data: Optional[Dict] = None) -> MarketContext:
        """
        Complete context analysis with optional integration
        
        Args:
            df: Primary timeframe DataFrame
            htf_data: Optional dict of higher timeframe DataFrames (e.g., {'15m': df15, '1h': df1h})
            regime_data: Optional regime data from market_regime.py
            structure_data: Optional structure data from market_structure.py
            sr_data: Optional support/resistance data from price_location.py
        
        Returns:
            MarketContext object with all context data
        """
        if df is None or df.empty:
            return self._default_context()
        
        # Basic metrics
        current_price = float(df['close'].iloc[-1])
        current_time = df.index[-1] if hasattr(df.index[-1], 'hour') else datetime.now()
        atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
        atr_pct = atr / current_price
        
        # Get candles for analysis
        from .candle_analyzer import CandleAnalyzer
        candle_analyzer = CandleAnalyzer()
        candles = candle_analyzer.analyze_all_candles(df)
        
        # ===== TREND STAGE =====
        trend_stage, trend_confidence = self.detect_trend_stage(df, candles)
        risk_multiplier = self.get_risk_multiplier(trend_stage)
        
        # ===== SESSION =====
        session, session_weight = self.detect_session(current_time)
        
        # ===== MTF ALIGNMENT =====
        mtf_trends = {}
        
        if htf_data:
            for tf, tf_df in htf_data.items():
                if tf_df is not None and not tf_df.empty:
                    mtf_trends[tf] = self.detect_mtf_trend(tf_df)
        else:
            # Use primary timeframe as fallback
            mtf_trends['primary'] = self.detect_mtf_trend(df)
        
        mtf_alignment = self.calculate_mtf_alignment(mtf_trends)
        
        # ===== KEY LEVELS =====
        # Try to use sr_data first, then fallback to pure price action
        nearest_support = None
        nearest_resistance = None
        level_strength = 0.0
        
        if sr_data:
            nearest_support = sr_data.get('closest_support')
            nearest_resistance = sr_data.get('closest_resistance')
            level_strength = sr_data.get('sr_score', 0.5)
        else:
            nearest_support, nearest_resistance, level_strength = self.detect_key_levels(
                df, current_price, atr
            )
        
        at_key_level = False
        key_level_type = ''
        key_level_price = None
        
        if self.is_at_key_level(current_price, nearest_support, atr):
            at_key_level = True
            key_level_type = 'support'
            key_level_price = nearest_support
        elif self.is_at_key_level(current_price, nearest_resistance, atr):
            at_key_level = True
            key_level_type = 'resistance'
            key_level_price = nearest_resistance
        
        # ===== MARKET STRUCTURE =====
        if structure_data:
            structure_bias = structure_data.get('market_structure', 'neutral')
            structure_strength = structure_data.get('bias_confidence', 0.5)
        else:
            structure_bias, structure_strength = self.detect_structure_bias(df, candles)
        
        # ===== VOLATILITY =====
        volatility_state = self.detect_volatility_state(atr_pct)
        
        # ===== INTEGRATE REGIME DATA (if available) =====
        if regime_data:
            # Override trend stage with regime if available
            regime_bias = regime_data.get('bias', 'NEUTRAL')
            if regime_bias == 'BULLISH' and trend_stage == 'consolidation':
                trend_stage = 'early'
                trend_confidence = regime_data.get('bias_score', 0.5)
            elif regime_bias == 'BEARISH' and trend_stage == 'consolidation':
                trend_stage = 'early'
                trend_confidence = regime_data.get('bias_score', 0.5)
            
            # Boost MTF alignment if regime confirms
            regime_score = regime_data.get('bias_score', 0)
            if abs(regime_score) > 0.3:
                mtf_alignment = min(1.0, mtf_alignment + 0.1)
        
        # ===== BUILD SUMMARY =====
        summary_parts = []
        summary_parts.append(f"Trend: {trend_stage.upper()} ({trend_confidence:.0%})")
        summary_parts.append(f"Session: {session.upper()} ({session_weight:.0%} weight)")
        summary_parts.append(f"MTF Alignment: {mtf_alignment:.0%}")
        summary_parts.append(f"Structure: {structure_bias.upper()} ({structure_strength:.0%})")
        
        if at_key_level:
            summary_parts.append(f"At {key_level_type} level @ {key_level_price:.4f}")
        
        summary = " | ".join(summary_parts)
        
        return MarketContext(
            trend_stage=trend_stage,
            trend_stage_confidence=trend_confidence,
            risk_multiplier=risk_multiplier,
            session=session,
            session_weight=session_weight,
            mtf_alignment_score=mtf_alignment,
            mtf_trends=mtf_trends,
            at_key_level=at_key_level,
            key_level_type=key_level_type,
            key_level_price=key_level_price,
            key_level_strength=level_strength,
            structure_bias=structure_bias,
            structure_strength=structure_strength,
            volatility_state=volatility_state,
            atr_pct=round(atr_pct, 4),
            summary=summary
        )
    
    def _default_context(self) -> MarketContext:
        """Return default context when analysis fails"""
        return MarketContext(
            trend_stage='consolidation',
            trend_stage_confidence=0.5,
            risk_multiplier=1.0,
            session='asia',
            session_weight=0.6,
            mtf_alignment_score=0.5,
            mtf_trends={},
            at_key_level=False,
            key_level_type='',
            key_level_price=None,
            key_level_strength=0.0,
            structure_bias='neutral',
            structure_strength=0.5,
            volatility_state='normal',
            atr_pct=0.02,
            summary='Default context (insufficient data)'
        )


# ==================== CONVENIENCE FUNCTIONS ====================

def get_market_context(df: pd.DataFrame,
                       htf_data: Optional[Dict[str, pd.DataFrame]] = None,
                       regime_data: Optional[Dict] = None,
                       structure_data: Optional[Dict] = None,
                       sr_data: Optional[Dict] = None) -> MarketContext:
    """
    Convenience function to get market context
    
    Args:
        df: Primary timeframe DataFrame
        htf_data: Optional higher timeframe DataFrames
        regime_data: Optional regime data
        structure_data: Optional structure data
        sr_data: Optional support/resistance data
    
    Returns:
        MarketContext object
    """
    engine = ContextEngine()
    return engine.analyze(df, htf_data, regime_data, structure_data, sr_data)


def get_trend_stage(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Convenience function to get trend stage only
    """
    engine = ContextEngine()
    return engine.detect_trend_stage(df)


def get_mtf_alignment(df_primary: pd.DataFrame,
                      df_15m: Optional[pd.DataFrame] = None,
                      df_1h: Optional[pd.DataFrame] = None,
                      df_4h: Optional[pd.DataFrame] = None) -> float:
    """
    Convenience function to get MTF alignment score
    """
    engine = ContextEngine()
    htf_data = {}
    if df_15m is not None:
        htf_data['15m'] = df_15m
    if df_1h is not None:
        htf_data['1h'] = df_1h
    if df_4h is not None:
        htf_data['4h'] = df_4h
    
    trends = {}
    for tf, tf_df in htf_data.items():
        if tf_df is not None and not tf_df.empty:
            trends[tf] = engine.detect_mtf_trend(tf_df)
    
    return engine.calculate_mtf_alignment(trends)


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    
    # Generate realistic price movement
    np.random.seed(42)
    base_price = 50000
    prices = [base_price]
    
    # Create a trend sequence
    for i in range(199):
        if i < 50:
            change = np.random.randn() * 30 + 5  # Uptrend
        elif i < 100:
            change = np.random.randn() * 20 + 2  # Consolidation
        elif i < 150:
            change = np.random.randn() * 30 - 5  # Downtrend
        else:
            change = np.random.randn() * 50     # Exhaustion
        prices.append(prices[-1] + change)
    
    data = []
    for i, close in enumerate(prices):
        open_p = close - np.random.randn() * 20
        high = max(open_p, close) + abs(np.random.randn() * 30)
        low = min(open_p, close) - abs(np.random.randn() * 30)
        volume = abs(np.random.randn() * 10000) + 5000
        
        data.append({
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Calculate ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Analyze context
    engine = ContextEngine()
    context = engine.analyze(df)
    
    print("=" * 60)
    print("MARKET CONTEXT ANALYSIS")
    print("=" * 60)
    print(f"Trend Stage: {context.trend_stage} (conf: {context.trend_stage_confidence:.0%})")
    print(f"Risk Multiplier: {context.risk_multiplier}")
    print(f"Session: {context.session} (weight: {context.session_weight:.0%})")
    print(f"MTF Alignment: {context.mtf_alignment_score:.0%}")
    print(f"At Key Level: {context.at_key_level} ({context.key_level_type})")
    print(f"Structure Bias: {context.structure_bias} (strength: {context.structure_strength:.0%})")
    print(f"Volatility: {context.volatility_state} (ATR: {context.atr_pct:.2%})")
    print("-" * 60)
    print(f"Summary: {context.summary}")
    print("=" * 60)