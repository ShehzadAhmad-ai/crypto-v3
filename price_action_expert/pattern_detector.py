"""
pattern_detector.py
Layer 2: Pattern Detector for Price Action Expert V3.5

Detects ALL candlestick patterns with:
- Pure price action logic (no indicators)
- Confidence scoring based on pattern quality
- Regime preference for each pattern
- Next candle confirmation requirements
- Vectorized detection where possible

Pattern Categories:
- Single Candle Patterns (8 patterns)
- Double Candle Patterns (10 patterns)
- Triple Candle Patterns (8 patterns)
- Multi-Candle Patterns (6 patterns)
- Trap Patterns (4 patterns)
- Momentum Patterns (5 patterns)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import configuration
from .price_action_config import (
    PATTERN_BASE_CONFIDENCE,
    PATTERNS_NEEDING_CONFIRMATION,
    PATTERN_CATEGORIES,
    ENGULFING_MIN_RATIO,
    PINBAR_WICK_BODY_RATIO,
    PINBAR_OPPOSITE_WICK_RATIO,
    MORNING_STAR_SECOND_BODY_MAX,
    THREE_CANDLES_MIN_BODY_ATR,
    THREE_CANDLES_WICK_MAX,
    MARUBOZU_WICK_MAX_PERCENT,
    LONG_CANDLE_MULTIPLIER,
    CONSECUTIVE_MIN_COUNT,
    CONSECUTIVE_BODY_MIN_ATR,
    MOMENTUM_SHIFT_WINDOW,
    KICKER_GAP_PERCENT,
    BLOWOFF_MOVE_PERCENT,
    BLOWOFF_VOLUME_RATIO,
    V_PATTERN_MOVE_PERCENT,
    STOP_HUNT_WICK_ATR_MULTIPLIER,
    PATTERN_NAME_MAPPING
)

# Import candle analyzer
from .candle_analyzer import CandleData, CandleAnalyzer


class PatternDirection(Enum):
    """Direction of the pattern"""
    BULLISH = "BULL"
    BEARISH = "BEAR"
    NEUTRAL = "NEUTRAL"


class PatternType(Enum):
    """Type/category of the pattern"""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    MULTI = "multi"
    TRAP = "trap"
    MOMENTUM = "momentum"


@dataclass
class DetectedPattern:
    """
    Complete pattern detection result
    """
    # Core identification
    name: str
    type: PatternType
    direction: PatternDirection
    confidence: float                    # 0-1 final confidence
    raw_confidence: float                # Before confirmation boosts
    strength: float                      # 0-1 pattern strength
    
    # Detection details
    candle_index: int                    # Index of the last candle in pattern
    candles_involved: int
    pattern_quality: str                 # 'A', 'B', 'C'
    
    # Context flags
    needs_confirmation: bool
    confirmation_received: bool
    volume_confirmed: bool
    trend_aligned: bool
    regime_aligned: bool
    key_level_confirmed: bool
    
    # Pattern-specific data
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    
    # Human-readable
    description: str = ""
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            'name': self.name,
            'type': self.type.value,
            'direction': self.direction.value,
            'confidence': round(self.confidence, 3),
            'strength': round(self.strength, 3),
            'candles_involved': self.candles_involved,
            'pattern_quality': self.pattern_quality,
            'needs_confirmation': self.needs_confirmation,
            'confirmation_received': self.confirmation_received,
            'volume_confirmed': self.volume_confirmed,
            'pattern_data': self.pattern_data,
            'description': self.description,
            'reasons': self.reasons[:5]
        }


class PatternDetector:
    """
    Advanced pattern detector for 30+ candlestick patterns
    
    Features:
    - Pure price action detection
    - Confidence scoring based on pattern quality
    - Regime preference awareness
    - Next candle confirmation
    - Vectorized detection where possible
    """
    
    def __init__(self):
        """Initialize the pattern detector"""
        self.candle_analyzer = CandleAnalyzer()
        self.volume_lookback = 20
        self.pattern_cache = {}
        
        # Store latest candles for pattern detection
        self.candles: List[CandleData] = []
        self.atr: float = 0.0
        self.avg_body: float = 0.0
        
    def _calculate_context_metrics(self, df: pd.DataFrame, idx: int) -> None:
        """
        Calculate context metrics needed for pattern detection
        """
        try:
            # Get ATR (use last 14 candles)
            if 'atr' in df.columns:
                self.atr = float(df['atr'].iloc[idx]) if idx < len(df) else float(df['atr'].iloc[-1])
            else:
                # Calculate simple ATR
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                self.atr = tr.rolling(14).mean().iloc[idx] if idx < len(df) else tr.rolling(14).mean().iloc[-1]
            
            # Calculate average body size
            bodies = abs(df['close'] - df['open'])
            self.avg_body = bodies.rolling(20).mean().iloc[idx] if idx < len(df) else bodies.rolling(20).mean().iloc[-1]
            
        except Exception:
            self.atr = float(df['close'].iloc[-1]) * 0.02
            self.avg_body = self.atr * 0.5
    
    # =========================================================
    # SINGLE CANDLE PATTERNS
    # =========================================================
    
    def detect_hammer(self, candle: CandleData, atr: float) -> Optional[DetectedPattern]:
        """
        Bullish reversal pattern with long lower wick, small body at top
        Best in downtrends at support levels
        
        Condition: lower wick > body × 2 AND upper wick < body × 0.3
        """
        if not candle.is_bearish:
            return None
        
        # Core condition: long lower wick
        if candle.lower_wick < candle.body_abs * PINBAR_WICK_BODY_RATIO:
            return None
        
        # Upper wick must be small (no rejection above)
        if candle.upper_wick > candle.body_abs * 0.3:
            return None
        
        # Calculate confidence
        confidence = PATTERN_BASE_CONFIDENCE['hammer']
        reasons = ['Hammer pattern detected']
        pattern_data = {}
        
        # Boost for lower wick size
        wick_ratio = candle.lower_wick / candle.body_abs
        if wick_ratio >= 3.0:
            confidence += 0.08
            reasons.append(f'Very long lower wick ({wick_ratio:.1f}x body)')
            pattern_data['wick_ratio'] = wick_ratio
        elif wick_ratio >= 2.5:
            confidence += 0.05
            reasons.append(f'Long lower wick ({wick_ratio:.1f}x body)')
        
        # Boost for close position (close near high)
        if candle.close_position >= 0.7:
            confidence += 0.05
            reasons.append('Close near high of candle')
        
        # Boost for volume confirmation
        if candle.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume confirmed ({candle.volume_ratio:.1f}x)')
            pattern_data['volume_ratio'] = candle.volume_ratio
        
        # Pattern quality
        pattern_quality = 'A' if confidence >= 0.75 else ('B' if confidence >= 0.65 else 'C')
        
        # Strength
        strength = min(0.95, confidence * 1.1)
        
        return DetectedPattern(
            name='hammer',
            type=PatternType.SINGLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle.index,
            candles_involved=1,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=candle.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Hammer: Bullish reversal with long lower wick, small body at top',
            reasons=reasons
        )
    
    def detect_shooting_star(self, candle: CandleData, atr: float) -> Optional[DetectedPattern]:
        """
        Bearish reversal pattern with long upper wick, small body at bottom
        Best in uptrends at resistance levels
        
        Condition: upper wick > body × 2 AND lower wick < body × 0.3
        """
        if not candle.is_bullish:
            return None
        
        # Core condition: long upper wick
        if candle.upper_wick < candle.body_abs * PINBAR_WICK_BODY_RATIO:
            return None
        
        # Lower wick must be small (no support below)
        if candle.lower_wick > candle.body_abs * 0.3:
            return None
        
        # Calculate confidence
        confidence = PATTERN_BASE_CONFIDENCE['shooting_star']
        reasons = ['Shooting Star pattern detected']
        pattern_data = {}
        
        # Boost for upper wick size
        wick_ratio = candle.upper_wick / candle.body_abs
        if wick_ratio >= 3.0:
            confidence += 0.08
            reasons.append(f'Very long upper wick ({wick_ratio:.1f}x body)')
            pattern_data['wick_ratio'] = wick_ratio
        elif wick_ratio >= 2.5:
            confidence += 0.05
            reasons.append(f'Long upper wick ({wick_ratio:.1f}x body)')
        
        # Boost for close position (close near low)
        if candle.close_position <= 0.3:
            confidence += 0.05
            reasons.append('Close near low of candle')
        
        # Boost for volume confirmation
        if candle.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume confirmed ({candle.volume_ratio:.1f}x)')
        
        pattern_quality = 'A' if confidence >= 0.75 else ('B' if confidence >= 0.65 else 'C')
        strength = min(0.95, confidence * 1.1)
        
        return DetectedPattern(
            name='shooting_star',
            type=PatternType.SINGLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle.index,
            candles_involved=1,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=candle.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Shooting Star: Bearish reversal with long upper wick, small body at bottom',
            reasons=reasons
        )
    
    def detect_marubozu(self, candle: CandleData, atr: float) -> Optional[DetectedPattern]:
        """
        Strong momentum pattern with no wicks (or very small wicks)
        Direction determined by candle color
        
        Condition: upper wick < body × 0.1 AND lower wick < body × 0.1
        """
        wick_max = candle.body_abs * MARUBOZU_WICK_MAX_PERCENT
        
        if candle.upper_wick > wick_max or candle.lower_wick > wick_max:
            return None
        
        direction = PatternDirection.BULLISH if candle.is_bullish else PatternDirection.BEARISH
        name = 'marubozu_bullish' if candle.is_bullish else 'marubozu_bearish'
        base_confidence = PATTERN_BASE_CONFIDENCE.get(name, 0.80)
        
        reasons = ['Marubozu pattern detected']
        pattern_data = {'direction': 'bullish' if candle.is_bullish else 'bearish'}
        
        confidence = base_confidence
        
        # Boost for perfect marubozu (no wicks)
        if candle.upper_wick == 0 and candle.lower_wick == 0:
            confidence += 0.10
            reasons.append('Perfect marubozu - no wicks')
            pattern_data['perfect'] = True
        elif candle.upper_wick < candle.body_abs * 0.05 and candle.lower_wick < candle.body_abs * 0.05:
            confidence += 0.05
            reasons.append('Very small wicks')
        
        # Boost for volume confirmation
        if candle.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume confirmed ({candle.volume_ratio:.1f}x)')
        
        # Boost for large body
        if candle.body_atr >= 1.2:
            confidence += 0.05
            reasons.append(f'Large body ({candle.body_atr:.1f}x ATR)')
        
        pattern_quality = 'A' if confidence >= 0.85 else ('B' if confidence >= 0.75 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name=name,
            type=PatternType.SINGLE,
            direction=direction,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle.index,
            candles_involved=1,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=f'Marubozu: Strong {direction.value.lower()} momentum with no wicks',
            reasons=reasons
        )
    
    def detect_doji(self, candle: CandleData, atr: float) -> Optional[DetectedPattern]:
        """
        Indecision pattern with very small body
        Various types: Dragonfly, Gravestone, Long-Legged, Standard
        
        Condition: body < range × 0.1
        """
        if not candle.is_doji:
            return None
        
        # Determine Doji type based on wicks
        body_ratio = candle.body_ratio
        total_range = candle.range
        
        if candle.lower_wick > total_range * 0.6:
            name = 'dragonfly_doji'
            direction = PatternDirection.BULLISH
            description = 'Dragonfly Doji: Long lower wick, small body at top - bullish reversal potential'
        elif candle.upper_wick > total_range * 0.6:
            name = 'gravestone_doji'
            direction = PatternDirection.BEARISH
            description = 'Gravestone Doji: Long upper wick, small body at bottom - bearish reversal potential'
        elif candle.lower_wick > candle.body_abs * 2 and candle.upper_wick > candle.body_abs * 2:
            name = 'long_legged_doji'
            direction = PatternDirection.NEUTRAL
            description = 'Long-Legged Doji: Extreme indecision, both wicks long'
        else:
            name = 'doji'
            direction = PatternDirection.NEUTRAL
            description = 'Doji: Indecision, market equilibrium'
        
        base_confidence = PATTERN_BASE_CONFIDENCE.get(name, 0.40)
        
        reasons = [f'{name.replace("_", " ").title()} detected']
        pattern_data = {'doji_type': name}
        
        confidence = base_confidence
        
        # Boost for very small body
        if body_ratio <= 0.05:
            confidence += 0.05
            reasons.append('Very small body - extreme indecision')
        
        # Boost for volume
        if candle.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append(f'Volume confirmed ({candle.volume_ratio:.1f}x)')
        
        pattern_quality = 'B' if confidence >= 0.55 else 'C'
        strength = confidence * 0.9
        
        return DetectedPattern(
            name=name,
            type=PatternType.SINGLE,
            direction=direction,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle.index,
            candles_involved=1,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=candle.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=description,
            reasons=reasons
        )
    
    def detect_spinning_top(self, candle: CandleData, atr: float) -> Optional[DetectedPattern]:
        """
        Indecision pattern with small body and wicks on both ends
        Indicates market uncertainty
        
        Condition: body < range × 0.3 AND body > range × 0.1
        """
        if candle.is_doji:
            return None
        
        body_ratio = candle.body_ratio
        
        if body_ratio >= 0.3 or body_ratio <= 0.1:
            return None
        
        direction = PatternDirection.NEUTRAL
        confidence = PATTERN_BASE_CONFIDENCE['spinning_top']
        
        reasons = ['Spinning Top detected']
        pattern_data = {'body_ratio': body_ratio}
        
        # Boost for balanced wicks
        if 0.4 <= candle.wick_top_ratio <= 0.6 and 0.4 <= candle.wick_bottom_ratio <= 0.6:
            confidence += 0.05
            reasons.append('Balanced wicks - extreme indecision')
        
        pattern_quality = 'C' if confidence >= 0.55 else 'D'
        strength = confidence * 0.85
        
        return DetectedPattern(
            name='spinning_top',
            type=PatternType.SINGLE,
            direction=direction,
            confidence=min(0.70, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle.index,
            candles_involved=1,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=candle.volume_ratio >= 1.2,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Spinning Top: Indecision with small body and wicks on both ends',
            reasons=reasons
        )
    
    def detect_long_candle(self, candle: CandleData, avg_body: float, atr: float) -> Optional[DetectedPattern]:
        """
        Strong momentum candle with body significantly larger than average
        
        Condition: body > avg_body × LONG_CANDLE_MULTIPLIER
        """
        if candle.body_abs < avg_body * LONG_CANDLE_MULTIPLIER:
            return None
        
        direction = PatternDirection.BULLISH if candle.is_bullish else PatternDirection.BEARISH
        name = 'long_candle_bullish' if candle.is_bullish else 'long_candle_bearish'
        
        body_multiple = candle.body_abs / avg_body
        confidence = PATTERN_BASE_CONFIDENCE.get(name, 0.70)
        
        reasons = [f'Long {direction.value.lower()} candle detected ({body_multiple:.1f}x average)']
        pattern_data = {'body_multiple': body_multiple}
        
        # Boost for very large body
        if body_multiple >= 2.5:
            confidence += 0.10
            reasons.append(f'Very large body ({body_multiple:.1f}x average)')
        elif body_multiple >= 2.0:
            confidence += 0.05
            reasons.append(f'Large body ({body_multiple:.1f}x average)')
        
        # Boost for volume confirmation
        if candle.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume confirmed ({candle.volume_ratio:.1f}x)')
        
        # Boost for close position
        if direction == PatternDirection.BULLISH and candle.close_position >= 0.8:
            confidence += 0.05
            reasons.append('Close near high')
        elif direction == PatternDirection.BEARISH and candle.close_position <= 0.2:
            confidence += 0.05
            reasons.append('Close near low')
        
        pattern_quality = 'A' if confidence >= 0.80 else ('B' if confidence >= 0.70 else 'C')
        strength = min(0.95, confidence * 1.1)
        
        return DetectedPattern(
            name=name,
            type=PatternType.SINGLE,
            direction=direction,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle.index,
            candles_involved=1,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=f'Long {direction.value.lower()} candle: Strong momentum with body {body_multiple:.1f}x average',
            reasons=reasons
        )
    
    # =========================================================
    # DOUBLE CANDLE PATTERNS
    # =========================================================
    
    def detect_bullish_engulfing(self, candle1: CandleData, candle2: CandleData, 
                                   atr: float, avg_body: float) -> Optional[DetectedPattern]:
        """
        Bullish reversal pattern: bearish candle fully engulfed by larger bullish candle
        Best in downtrends at support levels
        
        Condition: Previous bearish, current bullish, close > previous open, open < previous close
        """
        if not candle1.is_bearish or not candle2.is_bullish:
            return None
        
        # Core condition: current candle engulfs previous
        if not (candle2.close > candle1.open and candle2.open < candle1.close):
            return None
        
        # Optional: body size ratio for strength
        body_ratio = candle2.body_abs / (candle1.body_abs + 1e-8)
        
        confidence = PATTERN_BASE_CONFIDENCE['bullish_engulfing']
        reasons = ['Bullish Engulfing pattern detected']
        pattern_data = {'body_ratio': body_ratio}
        
        # Boost for strong engulfing
        if body_ratio >= ENGULFING_MIN_RATIO * 1.5:
            confidence += 0.08
            reasons.append(f'Strong engulfing ({body_ratio:.1f}x previous body)')
        elif body_ratio >= ENGULFING_MIN_RATIO:
            confidence += 0.05
            reasons.append(f'Engulfing ratio: {body_ratio:.1f}x')
        
        # Boost for volume confirmation
        if candle2.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {candle2.volume_ratio:.1f}x')
        
        # Boost for close above previous open
        if candle2.close > candle1.open:
            confidence += 0.05
            reasons.append('Close above previous open')
        
        pattern_quality = 'A' if confidence >= 0.85 else ('B' if confidence >= 0.75 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='bullish_engulfing',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Bullish Engulfing: Buyers overwhelm sellers, strong reversal signal',
            reasons=reasons
        )
    
    def detect_bearish_engulfing(self, candle1: CandleData, candle2: CandleData,
                                   atr: float, avg_body: float) -> Optional[DetectedPattern]:
        """
        Bearish reversal pattern: bullish candle fully engulfed by larger bearish candle
        Best in uptrends at resistance levels
        
        Condition: Previous bullish, current bearish, close < previous open, open > previous close
        """
        if not candle1.is_bullish or not candle2.is_bearish:
            return None
        
        if not (candle2.close < candle1.open and candle2.open > candle1.close):
            return None
        
        body_ratio = candle2.body_abs / (candle1.body_abs + 1e-8)
        
        confidence = PATTERN_BASE_CONFIDENCE['bearish_engulfing']
        reasons = ['Bearish Engulfing pattern detected']
        pattern_data = {'body_ratio': body_ratio}
        
        if body_ratio >= ENGULFING_MIN_RATIO * 1.5:
            confidence += 0.08
            reasons.append(f'Strong engulfing ({body_ratio:.1f}x previous body)')
        elif body_ratio >= ENGULFING_MIN_RATIO:
            confidence += 0.05
            reasons.append(f'Engulfing ratio: {body_ratio:.1f}x')
        
        if candle2.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {candle2.volume_ratio:.1f}x')
        
        if candle2.close < candle1.open:
            confidence += 0.05
            reasons.append('Close below previous open')
        
        pattern_quality = 'A' if confidence >= 0.85 else ('B' if confidence >= 0.75 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='bearish_engulfing',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Bearish Engulfing: Sellers overwhelm buyers, strong reversal signal',
            reasons=reasons
        )
    
    def detect_inside_bar(self, candle1: CandleData, candle2: CandleData,
                           atr: float) -> Optional[DetectedPattern]:
        """
        Compression pattern: second candle completely inside previous candle
        Indicates consolidation, breakout imminent
        
        Condition: high2 < high1 AND low2 > low1
        """
        if not (candle2.high < candle1.high and candle2.low > candle1.low):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['inside_bar']
        reasons = ['Inside Bar detected']
        pattern_data = {
            'range_ratio': (candle2.range / candle1.range) if candle1.range > 0 else 1
        }
        
        # Boost for tight inside bar
        range_ratio = candle2.range / (candle1.range + 1e-8)
        if range_ratio <= 0.5:
            confidence += 0.08
            reasons.append(f'Very tight inside bar ({range_ratio:.0%} of parent range)')
        elif range_ratio <= 0.7:
            confidence += 0.05
            reasons.append(f'Tight inside bar ({range_ratio:.0%} of parent range)')
        
        # Boost for volume contraction
        if candle2.volume_ratio <= 0.7:
            confidence += 0.05
            reasons.append('Volume contraction - compression confirmed')
        
        pattern_quality = 'B' if confidence >= 0.65 else 'C'
        strength = confidence * 0.9
        
        return DetectedPattern(
            name='inside_bar',
            type=PatternType.DOUBLE,
            direction=PatternDirection.NEUTRAL,
            confidence=min(0.80, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=False,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Inside Bar: Consolidation, watch for breakout direction',
            reasons=reasons
        )
    
    def detect_outside_bar(self, candle1: CandleData, candle2: CandleData,
                            atr: float) -> Optional[DetectedPattern]:
        """
        Expansion pattern: second candle completely engulfs previous candle
        Indicates volatility expansion, potential reversal or continuation
        
        Condition: high2 > high1 AND low2 < low1
        """
        if not (candle2.high > candle1.high and candle2.low < candle1.low):
            return None
        
        direction = PatternDirection.BULLISH if candle2.is_bullish else PatternDirection.BEARISH
        confidence = PATTERN_BASE_CONFIDENCE['outside_bar']
        reasons = ['Outside Bar detected']
        pattern_data = {
            'expansion_ratio': ((candle2.high - candle2.low) / (candle1.high - candle1.low)) if (candle1.high - candle1.low) > 0 else 1
        }
        
        # Boost for strong expansion
        expansion = (candle2.range) / (candle1.range + 1e-8)
        if expansion >= 1.5:
            confidence += 0.08
            reasons.append(f'Strong expansion ({expansion:.1f}x previous range)')
        
        # Boost for volume confirmation
        if candle2.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {candle2.volume_ratio:.1f}x')
        
        pattern_quality = 'B' if confidence >= 0.70 else 'C'
        strength = min(0.90, confidence * 1.05)
        
        return DetectedPattern(
            name='outside_bar',
            type=PatternType.DOUBLE,
            direction=direction,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=f'Outside Bar: {direction.value} expansion with volatility increase',
            reasons=reasons
        )
    
    def detect_tweezer_bottom(self, candle1: CandleData, candle2: CandleData,
                               atr: float) -> Optional[DetectedPattern]:
        """
        Bullish reversal: two candles with equal lows, first bearish, second bullish
        Indicates support holding
        
        Condition: lows equal (within tolerance), first bearish, second bullish
        """
        tolerance = atr * 0.05
        
        if not (abs(candle1.low - candle2.low) <= tolerance):
            return None
        
        if not (candle1.is_bearish and candle2.is_bullish):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['tweezer_bottom']
        reasons = ['Tweezer Bottom detected']
        pattern_data = {'low_level': candle2.low}
        
        # Boost for perfect equal lows
        if candle1.low == candle2.low:
            confidence += 0.05
            reasons.append('Perfect equal lows')
        
        # Boost for long lower wick on second candle
        if candle2.lower_wick_atr >= 0.5:
            confidence += 0.05
            reasons.append('Long lower wick on second candle - rejection')
        
        # Boost for volume
        if candle2.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append(f'Volume confirmed ({candle2.volume_ratio:.1f}x)')
        
        pattern_quality = 'B' if confidence >= 0.75 else 'C'
        strength = min(0.90, confidence * 1.1)
        
        return DetectedPattern(
            name='tweezer_bottom',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=candle2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Tweezer Bottom: Support holding, bullish reversal potential',
            reasons=reasons
        )
    
    def detect_tweezer_top(self, candle1: CandleData, candle2: CandleData,
                            atr: float) -> Optional[DetectedPattern]:
        """
        Bearish reversal: two candles with equal highs, first bullish, second bearish
        Indicates resistance holding
        """
        tolerance = atr * 0.05
        
        if not (abs(candle1.high - candle2.high) <= tolerance):
            return None
        
        if not (candle1.is_bullish and candle2.is_bearish):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['tweezer_top']
        reasons = ['Tweezer Top detected']
        pattern_data = {'high_level': candle2.high}
        
        if candle1.high == candle2.high:
            confidence += 0.05
            reasons.append('Perfect equal highs')
        
        if candle2.upper_wick_atr >= 0.5:
            confidence += 0.05
            reasons.append('Long upper wick on second candle - rejection')
        
        if candle2.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append(f'Volume confirmed ({candle2.volume_ratio:.1f}x)')
        
        pattern_quality = 'B' if confidence >= 0.75 else 'C'
        strength = min(0.90, confidence * 1.1)
        
        return DetectedPattern(
            name='tweezer_top',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=candle2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Tweezer Top: Resistance holding, bearish reversal potential',
            reasons=reasons
        )
    
    def detect_harami_bullish(self, candle1: CandleData, candle2: CandleData,
                               atr: float) -> Optional[DetectedPattern]:
        """
        Bullish reversal: small bullish candle inside previous large bearish candle
        Indicates selling pressure weakening
        """
        if not candle1.is_bearish:
            return None
        
        if not (candle2.is_bullish and candle2.high < candle1.high and candle2.low > candle1.low):
            return None
        
        body_ratio = candle2.body_abs / (candle1.body_abs + 1e-8)
        
        confidence = PATTERN_BASE_CONFIDENCE['harami_bullish']
        reasons = ['Bullish Harami detected']
        pattern_data = {'body_ratio': body_ratio}
        
        if body_ratio <= 0.3:
            confidence += 0.08
            reasons.append('Very small harami candle')
        
        if candle2.volume_ratio <= 0.7:
            confidence += 0.05
            reasons.append('Volume contraction - selling exhaustion')
        
        pattern_quality = 'B' if confidence >= 0.70 else 'C'
        strength = confidence * 0.9
        
        return DetectedPattern(
            name='harami_bullish',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.80, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=False,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Bullish Harami: Selling weakening, potential reversal',
            reasons=reasons
        )
    
    def detect_harami_bearish(self, candle1: CandleData, candle2: CandleData,
                               atr: float) -> Optional[DetectedPattern]:
        """
        Bearish reversal: small bearish candle inside previous large bullish candle
        Indicates buying pressure weakening
        """
        if not candle1.is_bullish:
            return None
        
        if not (candle2.is_bearish and candle2.high < candle1.high and candle2.low > candle1.low):
            return None
        
        body_ratio = candle2.body_abs / (candle1.body_abs + 1e-8)
        
        confidence = PATTERN_BASE_CONFIDENCE['harami_bearish']
        reasons = ['Bearish Harami detected']
        pattern_data = {'body_ratio': body_ratio}
        
        if body_ratio <= 0.3:
            confidence += 0.08
            reasons.append('Very small harami candle')
        
        if candle2.volume_ratio <= 0.7:
            confidence += 0.05
            reasons.append('Volume contraction - buying exhaustion')
        
        pattern_quality = 'B' if confidence >= 0.70 else 'C'
        strength = confidence * 0.9
        
        return DetectedPattern(
            name='harami_bearish',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.80, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=False,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Bearish Harami: Buying weakening, potential reversal',
            reasons=reasons
        )
    
    def detect_taker_bullish(self, candle1: CandleData, candle2: CandleData,
                              atr: float) -> Optional[DetectedPattern]:
        """
        Strong bullish momentum: candle closes above previous high
        Indicates buying pressure breaking resistance
        """
        if not (candle2.close > candle1.high):
            return None
        
        if not candle2.is_bullish:
            return None
        
        break_distance = (candle2.close - candle1.high) / (candle1.high + 1e-8)
        
        confidence = PATTERN_BASE_CONFIDENCE['taker_bullish']
        reasons = ['Bullish Taker detected - price broke above previous high']
        pattern_data = {'break_distance_pct': break_distance * 100}
        
        if break_distance >= 0.005:
            confidence += 0.08
            reasons.append(f'Strong break: {break_distance:.2%} above previous high')
        
        if candle2.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {candle2.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.80 else ('B' if confidence >= 0.70 else 'C')
        strength = min(0.95, confidence * 1.1)
        
        return DetectedPattern(
            name='taker_bullish',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Bullish Taker: Strong buying breaks resistance',
            reasons=reasons
        )
    
    def detect_taker_bearish(self, candle1: CandleData, candle2: CandleData,
                              atr: float) -> Optional[DetectedPattern]:
        """
        Strong bearish momentum: candle closes below previous low
        Indicates selling pressure breaking support
        """
        if not (candle2.close < candle1.low):
            return None
        
        if not candle2.is_bearish:
            return None
        
        break_distance = (candle1.low - candle2.close) / (candle1.low + 1e-8)
        
        confidence = PATTERN_BASE_CONFIDENCE['taker_bearish']
        reasons = ['Bearish Taker detected - price broke below previous low']
        pattern_data = {'break_distance_pct': break_distance * 100}
        
        if break_distance >= 0.005:
            confidence += 0.08
            reasons.append(f'Strong break: {break_distance:.2%} below previous low')
        
        if candle2.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {candle2.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.80 else ('B' if confidence >= 0.70 else 'C')
        strength = min(0.95, confidence * 1.1)
        
        return DetectedPattern(
            name='taker_bearish',
            type=PatternType.DOUBLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Bearish Taker: Strong selling breaks support',
            reasons=reasons
        )
    
    # =========================================================
    # TRIPLE CANDLE PATTERNS
    # =========================================================
    
    def detect_morning_star(self, candle1: CandleData, candle2: CandleData,
                             candle3: CandleData, atr: float, avg_body: float) -> Optional[DetectedPattern]:
        """
        Bullish reversal: bearish → indecision → bullish
        Classic three-candle reversal pattern
        """
        if not candle1.is_bearish:
            return None
        
        # Second candle must be small (indecision)
        if candle2.body_abs > avg_body * MORNING_STAR_SECOND_BODY_MAX:
            return None
        
        if not candle3.is_bullish:
            return None
        
        # Third candle should close into first candle's body
        closes_into_body = candle3.close > (candle1.open + candle1.close) / 2
        closes_above_open = candle3.close > candle1.open
        
        confidence = PATTERN_BASE_CONFIDENCE['morning_star']
        reasons = ['Morning Star pattern detected']
        pattern_data = {
            'closes_into_body': closes_into_body,
            'closes_above_open': closes_above_open
        }
        
        # Gap down detection
        if candle2.low > candle1.high:
            confidence += 0.08
            reasons.append('Gap down to second candle')
        
        # Boost for third candle strength
        if closes_above_open:
            confidence += 0.08
            reasons.append('Third candle closes above first candle open')
        elif closes_into_body:
            confidence += 0.05
            reasons.append('Third candle closes into first candle body')
        
        if candle3.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {candle3.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.85 else ('B' if confidence >= 0.75 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='morning_star',
            type=PatternType.TRIPLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Morning Star: Bullish reversal after downtrend',
            reasons=reasons
        )
    
    def detect_evening_star(self, candle1: CandleData, candle2: CandleData,
                             candle3: CandleData, atr: float, avg_body: float) -> Optional[DetectedPattern]:
        """
        Bearish reversal: bullish → indecision → bearish
        Classic three-candle reversal pattern
        """
        if not candle1.is_bullish:
            return None
        
        if candle2.body_abs > avg_body * MORNING_STAR_SECOND_BODY_MAX:
            return None
        
        if not candle3.is_bearish:
            return None
        
        closes_into_body = candle3.close < (candle1.open + candle1.close) / 2
        closes_below_open = candle3.close < candle1.open
        
        confidence = PATTERN_BASE_CONFIDENCE['evening_star']
        reasons = ['Evening Star pattern detected']
        pattern_data = {
            'closes_into_body': closes_into_body,
            'closes_below_open': closes_below_open
        }
        
        if candle2.high < candle1.low:
            confidence += 0.08
            reasons.append('Gap up to second candle')
        
        if closes_below_open:
            confidence += 0.08
            reasons.append('Third candle closes below first candle open')
        elif closes_into_body:
            confidence += 0.05
            reasons.append('Third candle closes into first candle body')
        
        if candle3.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {candle3.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.85 else ('B' if confidence >= 0.75 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='evening_star',
            type=PatternType.TRIPLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candle3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=candle3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Evening Star: Bearish reversal after uptrend',
            reasons=reasons
        )
    
    def detect_three_white_soldiers(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                     atr: float) -> Optional[DetectedPattern]:
        """
        Strong bullish continuation: three consecutive large bullish candles
        Each with higher closes and small wicks
        """
        if not (c1.is_bullish and c2.is_bullish and c3.is_bullish):
            return None
        
        # All candles must be strong
        if not (c1.body_atr >= THREE_CANDLES_MIN_BODY_ATR and
                c2.body_atr >= THREE_CANDLES_MIN_BODY_ATR and
                c3.body_atr >= THREE_CANDLES_MIN_BODY_ATR):
            return None
        
        # Wicks must be small
        if not (c1.upper_wick < c1.body_abs * THREE_CANDLES_WICK_MAX and
                c2.upper_wick < c2.body_abs * THREE_CANDLES_WICK_MAX and
                c3.upper_wick < c3.body_abs * THREE_CANDLES_WICK_MAX):
            return None
        
        # Higher closes
        if not (c2.close > c1.close and c3.close > c2.close):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['three_white_soldiers']
        reasons = ['Three White Soldiers detected']
        pattern_data = {
            'c1_body': c1.body_atr,
            'c2_body': c2.body_atr,
            'c3_body': c3.body_atr
        }
        
        # Boost for increasing body sizes (acceleration)
        if c2.body_abs > c1.body_abs and c3.body_abs > c2.body_abs:
            confidence += 0.08
            reasons.append('Accelerating momentum')
        
        # Boost for volume
        if c3.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append('Volume confirmation')
        
        pattern_quality = 'A' if confidence >= 0.90 else ('B' if confidence >= 0.80 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='three_white_soldiers',
            type=PatternType.TRIPLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Three White Soldiers: Strong bullish continuation',
            reasons=reasons
        )
    
    def detect_three_black_crows(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                   atr: float) -> Optional[DetectedPattern]:
        """
        Strong bearish continuation: three consecutive large bearish candles
        Each with lower closes and small wicks
        """
        if not (c1.is_bearish and c2.is_bearish and c3.is_bearish):
            return None
        
        if not (c1.body_atr >= THREE_CANDLES_MIN_BODY_ATR and
                c2.body_atr >= THREE_CANDLES_MIN_BODY_ATR and
                c3.body_atr >= THREE_CANDLES_MIN_BODY_ATR):
            return None
        
        if not (c1.lower_wick < c1.body_abs * THREE_CANDLES_WICK_MAX and
                c2.lower_wick < c2.body_abs * THREE_CANDLES_WICK_MAX and
                c3.lower_wick < c3.body_abs * THREE_CANDLES_WICK_MAX):
            return None
        
        if not (c2.close < c1.close and c3.close < c2.close):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['three_black_crows']
        reasons = ['Three Black Crows detected']
        pattern_data = {
            'c1_body': c1.body_atr,
            'c2_body': c2.body_atr,
            'c3_body': c3.body_atr
        }
        
        if c2.body_abs > c1.body_abs and c3.body_abs > c2.body_abs:
            confidence += 0.08
            reasons.append('Accelerating downward momentum')
        
        if c3.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append('Volume confirmation')
        
        pattern_quality = 'A' if confidence >= 0.90 else ('B' if confidence >= 0.80 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='three_black_crows',
            type=PatternType.TRIPLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Three Black Crows: Strong bearish continuation',
            reasons=reasons
        )
    
    def detect_three_inside_up(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                atr: float) -> Optional[DetectedPattern]:
        """
        Bullish reversal: bearish → inside bar → breakout above
        """
        if not c1.is_bearish:
            return None
        
        if not (c2.is_bearish and c2.high < c1.high and c2.low > c1.low):
            return None
        
        if not (c3.is_bullish and c3.close > c1.high):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['three_inside_up']
        reasons = ['Three Inside Up detected']
        pattern_data = {}
        
        if c3.volume_ratio >= 1.5:
            confidence += 0.08
            reasons.append(f'Volume spike on breakout: {c3.volume_ratio:.1f}x')
        
        if (c3.close - c1.high) / c1.high >= 0.005:
            confidence += 0.05
            reasons.append('Strong breakout above resistance')
        
        pattern_quality = 'B' if confidence >= 0.75 else 'C'
        strength = min(0.90, confidence * 1.1)
        
        return DetectedPattern(
            name='three_inside_up',
            type=PatternType.TRIPLE,
            direction=PatternDirection.BULLISH,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Three Inside Up: Bullish breakout from compression',
            reasons=reasons
        )
    
    def detect_three_inside_down(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                  atr: float) -> Optional[DetectedPattern]:
        """
        Bearish reversal: bullish → inside bar → breakdown below
        """
        if not c1.is_bullish:
            return None
        
        if not (c2.is_bullish and c2.high < c1.high and c2.low > c1.low):
            return None
        
        if not (c3.is_bearish and c3.close < c1.low):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['three_inside_down']
        reasons = ['Three Inside Down detected']
        pattern_data = {}
        
        if c3.volume_ratio >= 1.5:
            confidence += 0.08
            reasons.append(f'Volume spike on breakdown: {c3.volume_ratio:.1f}x')
        
        if (c1.low - c3.close) / c1.low >= 0.005:
            confidence += 0.05
            reasons.append('Strong breakdown below support')
        
        pattern_quality = 'B' if confidence >= 0.75 else 'C'
        strength = min(0.90, confidence * 1.1)
        
        return DetectedPattern(
            name='three_inside_down',
            type=PatternType.TRIPLE,
            direction=PatternDirection.BEARISH,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Three Inside Down: Bearish breakdown from compression',
            reasons=reasons
        )
    
    # =========================================================
    # MULTI-CANDLE PATTERNS
    # =========================================================
    
    def detect_rising_three_methods(self, candles: List[CandleData], idx: int,
                                      atr: float) -> Optional[DetectedPattern]:
        """
        Bullish continuation: large bullish → 3 small bearish → large bullish
        """
        if idx < 4:
            return None
        
        c1 = candles[idx-4]  # First bullish
        c2 = candles[idx-3]  # Small bearish
        c3 = candles[idx-2]  # Small bearish
        c4 = candles[idx-1]  # Small bearish
        c5 = candles[idx]    # Final bullish
        
        if not c1.is_bullish:
            return None
        
        # Three small bearish candles
        if not (c2.is_bearish and c3.is_bearish and c4.is_bearish):
            return None
        
        # All small bodies
        avg_small_body = (c2.body_abs + c3.body_abs + c4.body_abs) / 3
        if avg_small_body > c1.body_abs * 0.5:
            return None
        
        # Final bullish breaks above
        if not (c5.is_bullish and c5.close > c1.high):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['rising_three_methods']
        reasons = ['Rising Three Methods detected']
        pattern_data = {}
        
        if c5.volume_ratio >= 1.5:
            confidence += 0.08
            reasons.append(f'Volume spike on breakout: {c5.volume_ratio:.1f}x')
        
        pattern_quality = 'B' if confidence >= 0.75 else 'C'
        strength = min(0.90, confidence * 1.1)
        
        return DetectedPattern(
            name='rising_three_methods',
            type=PatternType.MULTI,
            direction=PatternDirection.BULLISH,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c5.index,
            candles_involved=5,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c5.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Rising Three Methods: Bullish continuation after consolidation',
            reasons=reasons
        )
    
    def detect_falling_three_methods(self, candles: List[CandleData], idx: int,
                                       atr: float) -> Optional[DetectedPattern]:
        """
        Bearish continuation: large bearish → 3 small bullish → large bearish
        """
        if idx < 4:
            return None
        
        c1 = candles[idx-4]  # First bearish
        c2 = candles[idx-3]  # Small bullish
        c3 = candles[idx-2]  # Small bullish
        c4 = candles[idx-1]  # Small bullish
        c5 = candles[idx]    # Final bearish
        
        if not c1.is_bearish:
            return None
        
        if not (c2.is_bullish and c3.is_bullish and c4.is_bullish):
            return None
        
        avg_small_body = (c2.body_abs + c3.body_abs + c4.body_abs) / 3
        if avg_small_body > c1.body_abs * 0.5:
            return None
        
        if not (c5.is_bearish and c5.close < c1.low):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['falling_three_methods']
        reasons = ['Falling Three Methods detected']
        pattern_data = {}
        
        if c5.volume_ratio >= 1.5:
            confidence += 0.08
            reasons.append(f'Volume spike on breakdown: {c5.volume_ratio:.1f}x')
        
        pattern_quality = 'B' if confidence >= 0.75 else 'C'
        strength = min(0.90, confidence * 1.1)
        
        return DetectedPattern(
            name='falling_three_methods',
            type=PatternType.MULTI,
            direction=PatternDirection.BEARISH,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c5.index,
            candles_involved=5,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c5.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Falling Three Methods: Bearish continuation after consolidation',
            reasons=reasons
        )
    
        # =========================================================
    # MULTI-CANDLE PATTERNS (CONTINUED)
    # =========================================================
    
    def detect_three_line_strike(self, candles: List[CandleData], idx: int,
                                   atr: float) -> Optional[DetectedPattern]:
        """
        Bullish reversal: 4 bearish candles → large bullish candle that reverses all
        """
        if idx < 4:
            return None
        
        c1 = candles[idx-4]  # Bearish
        c2 = candles[idx-3]  # Bearish
        c3 = candles[idx-2]  # Bearish
        c4 = candles[idx-1]  # Bearish
        c5 = candles[idx]    # Bullish reversal
        
        if not (c1.is_bearish and c2.is_bearish and c3.is_bearish and c4.is_bearish):
            return None
        
        if not c5.is_bullish:
            return None
        
        # Final candle must close above all previous highs
        if not (c5.close > c1.high and c5.close > c2.high and 
                c5.close > c3.high and c5.close > c4.high):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['three_line_strike']
        reasons = ['Three Line Strike detected']
        pattern_data = {'reversal_strength': (c5.close - max(c1.high, c2.high, c3.high, c4.high)) / c5.close}
        
        if c5.volume_ratio >= 1.8:
            confidence += 0.10
            reasons.append(f'Massive volume spike: {c5.volume_ratio:.1f}x')
        elif c5.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append(f'Volume confirmation: {c5.volume_ratio:.1f}x')
        
        if c5.body_atr >= 1.2:
            confidence += 0.08
            reasons.append('Large reversal candle')
        
        pattern_quality = 'A' if confidence >= 0.85 else ('B' if confidence >= 0.75 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='three_line_strike',
            type=PatternType.MULTI,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c5.index,
            candles_involved=5,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c5.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Three Line Strike: Bullish reversal after 4 bearish candles',
            reasons=reasons
        )
    
    def detect_identical_three_crows(self, candles: List[CandleData], idx: int,
                                       atr: float) -> Optional[DetectedPattern]:
        """
        Bearish continuation: 3 bearish candles with similar closes
        """
        if idx < 2:
            return None
        
        c1 = candles[idx-2]
        c2 = candles[idx-1]
        c3 = candles[idx]
        
        if not (c1.is_bearish and c2.is_bearish and c3.is_bearish):
            return None
        
        # Similar close prices (within 0.5%)
        close_tolerance = c1.close * 0.005
        if not (abs(c1.close - c2.close) <= close_tolerance and
                abs(c2.close - c3.close) <= close_tolerance):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE.get('identical_three_crows', 0.75)
        reasons = ['Identical Three Crows detected']
        pattern_data = {
            'close1': c1.close,
            'close2': c2.close,
            'close3': c3.close
        }
        
        if c3.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {c3.volume_ratio:.1f}x')
        
        pattern_quality = 'B' if confidence >= 0.75 else 'C'
        strength = min(0.90, confidence * 1.05)
        
        return DetectedPattern(
            name='identical_three_crows',
            type=PatternType.MULTI,
            direction=PatternDirection.BEARISH,
            confidence=min(0.85, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description='Identical Three Crows: Strong bearish pressure with consistent closes',
            reasons=reasons
        )
    
    # =========================================================
    # TRAP PATTERNS
    # =========================================================
    
    def detect_bull_trap(self, candles: List[CandleData], idx: int, 
                          resistance_level: Optional[float] = None,
                          atr: float = 0) -> Optional[DetectedPattern]:
        """
        Bull Trap: Price breaks above resistance but closes below
        Indicates fake breakout, smart money selling
        """
        if idx < 1:
            return None
        
        c1 = candles[idx-1]  # Breakout candle
        c2 = candles[idx]    # Reversal candle
        
        # Check for break above resistance
        if resistance_level is not None:
            if not (c1.high > resistance_level):
                return None
        else:
            # Use previous swing high if no resistance level provided
            # Simplified: check if candle broke above recent high
            if idx >= 10:
                recent_high = max(c.high for c in candles[idx-10:idx-1])
                if not (c1.high > recent_high * 1.001):
                    return None
        
        # Reversal: close below resistance/breakout level
        breakout_level = resistance_level if resistance_level else c1.high
        if not (c2.close < breakout_level):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['bull_trap']
        reasons = ['Bull Trap detected']
        pattern_data = {
            'breakout_high': c1.high,
            'close_below': c2.close
        }
        
        # Severity factors
        trap_score = 0.5
        
        # Wick length (long wick on breakout = stronger trap)
        if c1.upper_wick_atr >= 0.5:
            trap_score += 0.15
            reasons.append('Long upper wick on breakout - rejection')
        
        # Reversal speed (immediate reversal = stronger)
        if c2.is_bearish and c2.close < c1.open:
            trap_score += 0.15
            reasons.append('Immediate bearish reversal')
        
        # Volume spike on breakout
        if c1.volume_ratio >= 1.5:
            trap_score += 0.10
            reasons.append(f'Volume spike on breakout: {c1.volume_ratio:.1f}x')
        
        # Volume on reversal
        if c2.volume_ratio >= 1.3:
            trap_score += 0.10
            reasons.append(f'Volume on reversal: {c2.volume_ratio:.1f}x')
        
        pattern_data['trap_score'] = min(1.0, trap_score)
        confidence += trap_score * 0.2
        
        # Determine severity
        if trap_score >= 0.8:
            severity = 'extreme'
        elif trap_score >= 0.6:
            severity = 'strong'
        elif trap_score >= 0.4:
            severity = 'medium'
        else:
            severity = 'minor'
        
        pattern_quality = 'A' if trap_score >= 0.7 else ('B' if trap_score >= 0.5 else 'C')
        strength = min(0.95, trap_score * 1.2)
        
        return DetectedPattern(
            name='bull_trap',
            type=PatternType.TRAP,
            direction=PatternDirection.BEARISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=resistance_level is not None,
            pattern_data={'severity': severity, 'trap_score': trap_score},
            description=f'Bull Trap: {severity.upper()} severity - fake breakout to upside',
            reasons=reasons
        )
    
    def detect_bear_trap(self, candles: List[CandleData], idx: int,
                          support_level: Optional[float] = None,
                          atr: float = 0) -> Optional[DetectedPattern]:
        """
        Bear Trap: Price breaks below support but closes above
        Indicates fake breakdown, smart money buying
        """
        if idx < 1:
            return None
        
        c1 = candles[idx-1]  # Breakdown candle
        c2 = candles[idx]    # Reversal candle
        
        if support_level is not None:
            if not (c1.low < support_level):
                return None
        else:
            if idx >= 10:
                recent_low = min(c.low for c in candles[idx-10:idx-1])
                if not (c1.low < recent_low * 0.999):
                    return None
        
        breakdown_level = support_level if support_level else c1.low
        if not (c2.close > breakdown_level):
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['bear_trap']
        reasons = ['Bear Trap detected']
        pattern_data = {
            'breakdown_low': c1.low,
            'close_above': c2.close
        }
        
        trap_score = 0.5
        
        if c1.lower_wick_atr >= 0.5:
            trap_score += 0.15
            reasons.append('Long lower wick on breakdown - rejection')
        
        if c2.is_bullish and c2.close > c1.open:
            trap_score += 0.15
            reasons.append('Immediate bullish reversal')
        
        if c1.volume_ratio >= 1.5:
            trap_score += 0.10
            reasons.append(f'Volume spike on breakdown: {c1.volume_ratio:.1f}x')
        
        if c2.volume_ratio >= 1.3:
            trap_score += 0.10
            reasons.append(f'Volume on reversal: {c2.volume_ratio:.1f}x')
        
        pattern_data['trap_score'] = min(1.0, trap_score)
        confidence += trap_score * 0.2
        
        if trap_score >= 0.8:
            severity = 'extreme'
        elif trap_score >= 0.6:
            severity = 'strong'
        elif trap_score >= 0.4:
            severity = 'medium'
        else:
            severity = 'minor'
        
        pattern_quality = 'A' if trap_score >= 0.7 else ('B' if trap_score >= 0.5 else 'C')
        strength = min(0.95, trap_score * 1.2)
        
        return DetectedPattern(
            name='bear_trap',
            type=PatternType.TRAP,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c2.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=support_level is not None,
            pattern_data={'severity': severity, 'trap_score': trap_score},
            description=f'Bear Trap: {severity.upper()} severity - fake breakdown to downside',
            reasons=reasons
        )
    
    def detect_stop_hunt(self, candles: List[CandleData], idx: int,
                          atr: float) -> Optional[DetectedPattern]:
        """
        Stop Hunt: Long wick through key level with immediate reversal
        Smart money taking liquidity then reversing
        """
        if idx < 1:
            return None
        
        c1 = candles[idx-1]  # Stop hunt candle
        c2 = candles[idx]    # Reversal candle
        
        # Check for extreme wick
        has_extreme_wick = (c1.upper_wick_atr >= STOP_HUNT_WICK_ATR_MULTIPLIER or
                            c1.lower_wick_atr >= STOP_HUNT_WICK_ATR_MULTIPLIER)
        
        if not has_extreme_wick:
            return None
        
        # Determine direction
        if c1.lower_wick_atr >= STOP_HUNT_WICK_ATR_MULTIPLIER:
            # Downside stop hunt = bullish reversal
            if not (c2.is_bullish and c2.close > c1.open):
                return None
            direction = PatternDirection.BULLISH
            name = 'stop_hunt_bullish'
            description = 'Stop Hunt: Long lower wick swept liquidity, now reversing up'
            
            pattern_data = {'wick_type': 'lower', 'wick_size_atr': c1.lower_wick_atr}
            reasons = ['Downside stop hunt detected - liquidity sweep']
            
        elif c1.upper_wick_atr >= STOP_HUNT_WICK_ATR_MULTIPLIER:
            # Upside stop hunt = bearish reversal
            if not (c2.is_bearish and c2.close < c1.open):
                return None
            direction = PatternDirection.BEARISH
            name = 'stop_hunt_bearish'
            description = 'Stop Hunt: Long upper wick swept liquidity, now reversing down'
            
            pattern_data = {'wick_type': 'upper', 'wick_size_atr': c1.upper_wick_atr}
            reasons = ['Upside stop hunt detected - liquidity sweep']
        else:
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE.get(name, 0.85)
        
        # Boost for volume confirmation
        if c1.volume_ratio >= 1.8:
            confidence += 0.10
            reasons.append(f'Massive volume spike: {c1.volume_ratio:.1f}x')
        elif c1.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append(f'Volume spike: {c1.volume_ratio:.1f}x')
        
        # Boost for immediate reversal strength
        if direction == PatternDirection.BULLISH and c2.is_bullish:
            if c2.body_atr >= 0.8:
                confidence += 0.08
                reasons.append('Strong bullish reversal candle')
        elif direction == PatternDirection.BEARISH and c2.is_bearish:
            if c2.body_atr >= 0.8:
                confidence += 0.08
                reasons.append('Strong bearish reversal candle')
        
        pattern_quality = 'A' if confidence >= 0.85 else ('B' if confidence >= 0.75 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name=name,
            type=PatternType.TRAP,
            direction=direction,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c2.index,
            candles_involved=2,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c1.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=description,
            reasons=reasons
        )
    
    # =========================================================
    # MOMENTUM PATTERNS
    # =========================================================
    
    def detect_consecutive_bullish(self, candles: List[CandleData], idx: int,
                                     atr: float) -> Optional[DetectedPattern]:
        """
        Consecutive Bullish: 2+ bullish candles with strong bodies
        """
        if idx < CONSECUTIVE_MIN_COUNT - 1:
            return None
        
        count = 0
        for i in range(CONSECUTIVE_MIN_COUNT):
            if candles[idx - i].is_bullish and candles[idx - i].body_atr >= CONSECUTIVE_BODY_MIN_ATR:
                count += 1
            else:
                break
        
        if count < CONSECUTIVE_MIN_COUNT:
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['consecutive_bullish']
        reasons = [f'{count} consecutive bullish candles detected']
        pattern_data = {'count': count, 'body_sizes': [candles[idx - i].body_atr for i in range(count)]}
        
        # Boost for more consecutive candles
        if count >= 3:
            confidence += 0.08
            reasons.append('3+ consecutive candles - strong momentum')
        
        # Boost for increasing body sizes (acceleration)
        bodies = [candles[idx - i].body_abs for i in range(count)]
        if len(bodies) >= 2 and all(bodies[i] > bodies[i+1] for i in range(len(bodies)-1)):
            confidence += 0.05
            reasons.append('Increasing body sizes - accelerating')
        
        # Boost for volume
        latest = candles[idx]
        if latest.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append(f'Volume confirmation: {latest.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.80 else ('B' if confidence >= 0.70 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='consecutive_bullish',
            type=PatternType.MOMENTUM,
            direction=PatternDirection.BULLISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candles[idx].index,
            candles_involved=count,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=latest.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=f'{count} consecutive bullish candles - strong momentum',
            reasons=reasons
        )
    
    def detect_consecutive_bearish(self, candles: List[CandleData], idx: int,
                                     atr: float) -> Optional[DetectedPattern]:
        """
        Consecutive Bearish: 2+ bearish candles with strong bodies
        """
        if idx < CONSECUTIVE_MIN_COUNT - 1:
            return None
        
        count = 0
        for i in range(CONSECUTIVE_MIN_COUNT):
            if candles[idx - i].is_bearish and candles[idx - i].body_atr >= CONSECUTIVE_BODY_MIN_ATR:
                count += 1
            else:
                break
        
        if count < CONSECUTIVE_MIN_COUNT:
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['consecutive_bearish']
        reasons = [f'{count} consecutive bearish candles detected']
        pattern_data = {'count': count, 'body_sizes': [candles[idx - i].body_atr for i in range(count)]}
        
        if count >= 3:
            confidence += 0.08
            reasons.append('3+ consecutive candles - strong momentum')
        
        
        bodies = [candles[idx - i].body_abs for i in range(count)]
        if len(bodies) >= 2 and all(bodies[i] > bodies[i+1] for i in range(len(bodies)-1)):
            confidence += 0.05
            reasons.append('Increasing body sizes - accelerating')
        
        latest = candles[idx]
        if latest.volume_ratio >= 1.3:
            confidence += 0.05
            reasons.append(f'Volume confirmation: {latest.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.80 else ('B' if confidence >= 0.70 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='consecutive_bearish',
            type=PatternType.MOMENTUM,
            direction=PatternDirection.BEARISH,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=candles[idx].index,
            candles_involved=count,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=latest.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=f'{count} consecutive bearish candles - strong momentum',
            reasons=reasons
        )
    
    def detect_momentum_shift(self, candles: List[CandleData], idx: int,
                               atr: float) -> Optional[DetectedPattern]:
        """
        Momentum Shift: Decreasing body sizes → strong opposite direction
        """
        if idx < MOMENTUM_SHIFT_WINDOW:
            return None
        
        # Check previous candles for decreasing momentum
        prev_bodies = []
        for i in range(1, MOMENTUM_SHIFT_WINDOW):
            prev_bodies.append(candles[idx - i].body_abs)
        
        # Check if bodies are decreasing
        is_decreasing = all(prev_bodies[i] <= prev_bodies[i-1] * 1.1 for i in range(1, len(prev_bodies)))
        
        if not is_decreasing:
            return None
        
        current = candles[idx]
        previous = candles[idx-1]
        
        # Detect shift direction
        if previous.is_bearish and current.is_bullish and current.body_abs > previous.body_abs * 1.5:
            direction = PatternDirection.BULLISH
            name = 'momentum_shift_bullish'
            description = 'Momentum Shift: Bearish momentum exhausted, bullish emerging'
            reasons = ['Bearish momentum decreasing, now bullish engulfing']
            
        elif previous.is_bullish and current.is_bearish and current.body_abs > previous.body_abs * 1.5:
            direction = PatternDirection.BEARISH
            name = 'momentum_shift_bearish'
            description = 'Momentum Shift: Bullish momentum exhausted, bearish emerging'
            reasons = ['Bullish momentum decreasing, now bearish engulfing']
        else:
            return None
        
        confidence = PATTERN_BASE_CONFIDENCE['momentum_shift']
        pattern_data = {
            'prev_body_avg': np.mean(prev_bodies),
            'current_body': current.body_abs,
            'shift_ratio': current.body_abs / (np.mean(prev_bodies) + 1e-8)
        }
        
        reasons.append(f'Shift ratio: {pattern_data["shift_ratio"]:.1f}x')
        
        if current.volume_ratio >= 1.5:
            confidence += 0.08
            reasons.append(f'Volume spike on shift: {current.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.80 else ('B' if confidence >= 0.70 else 'C')
        strength = min(0.95, confidence * 1.1)
        
        return DetectedPattern(
            name=name,
            type=PatternType.MOMENTUM,
            direction=direction,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=current.index,
            candles_involved=MOMENTUM_SHIFT_WINDOW,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=current.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=description,
            reasons=reasons
        )
    
    def detect_acceleration(self, candles: List[CandleData], idx: int,
                             atr: float) -> Optional[DetectedPattern]:
        """
        Acceleration: 3+ candles with increasing body sizes
        """
        if idx < 2:
            return None
        
        c1 = candles[idx-2]
        c2 = candles[idx-1]
        c3 = candles[idx]
        
        if not (c1.is_bullish and c2.is_bullish and c3.is_bullish):
            if not (c1.is_bearish and c2.is_bearish and c3.is_bearish):
                return None
        
        # Check for increasing bodies
        if not (c2.body_abs > c1.body_abs * 1.1 and c3.body_abs > c2.body_abs * 1.1):
            return None
        
        direction = PatternDirection.BULLISH if c3.is_bullish else PatternDirection.BEARISH
        
        confidence = PATTERN_BASE_CONFIDENCE['acceleration']
        reasons = [f'Accelerating {direction.value.lower()} momentum detected']
        pattern_data = {
            'body_ratios': [
                c2.body_abs / c1.body_abs,
                c3.body_abs / c2.body_abs
            ]
        }
        
        # Boost for strong acceleration
        if pattern_data['body_ratios'][1] >= 1.5:
            confidence += 0.08
            reasons.append(f'Strong acceleration: {pattern_data["body_ratios"][1]:.1f}x increase')
        
        if c3.volume_ratio >= 1.5:
            confidence += 0.05
            reasons.append(f'Volume spike: {c3.volume_ratio:.1f}x')
        
        pattern_quality = 'A' if confidence >= 0.80 else ('B' if confidence >= 0.70 else 'C')
        strength = min(0.95, confidence * 1.05)
        
        return DetectedPattern(
            name='acceleration',
            type=PatternType.MOMENTUM,
            direction=direction,
            confidence=min(0.95, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=False,
            confirmation_received=True,
            volume_confirmed=c3.volume_ratio >= 1.3,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=f'Acceleration: {direction.value} momentum increasing',
            reasons=reasons
        )
    
    def detect_deceleration(self, candles: List[CandleData], idx: int,
                             atr: float) -> Optional[DetectedPattern]:
        """
        Deceleration: 3+ candles with decreasing body sizes
        Indicates momentum exhaustion
        """
        if idx < 2:
            return None
        
        c1 = candles[idx-2]
        c2 = candles[idx-1]
        c3 = candles[idx]
        
        # Check for decreasing bodies
        if not (c2.body_abs < c1.body_abs * 0.9 and c3.body_abs < c2.body_abs * 0.9):
            return None
        
        direction = PatternDirection.BULLISH if c3.is_bullish else PatternDirection.BEARISH
        
        confidence = PATTERN_BASE_CONFIDENCE['deceleration']
        reasons = [f'Decelerating {direction.value.lower()} momentum detected']
        pattern_data = {
            'body_ratios': [
                c2.body_abs / c1.body_abs,
                c3.body_abs / c2.body_abs
            ]
        }
        
        reasons.append('Momentum exhaustion - potential reversal')
        
        pattern_quality = 'C' if confidence >= 0.65 else 'D'
        strength = confidence * 0.85
        
        return DetectedPattern(
            name='deceleration',
            type=PatternType.MOMENTUM,
            direction=direction,
            confidence=min(0.70, confidence),
            raw_confidence=confidence,
            strength=strength,
            candle_index=c3.index,
            candles_involved=3,
            pattern_quality=pattern_quality,
            needs_confirmation=True,
            confirmation_received=False,
            volume_confirmed=False,
            trend_aligned=False,
            regime_aligned=False,
            key_level_confirmed=False,
            pattern_data=pattern_data,
            description=f'Deceleration: {direction.value} momentum weakening',
            reasons=reasons
        )
    
    # =========================================================
    # COMPLETE DETECT_ALL_PATTERNS METHOD (UPDATED)
    # =========================================================
    
    def detect_all_patterns(self, df: pd.DataFrame, 
                           regime: Optional[Dict] = None,
                           sr_data: Optional[Dict] = None,
                           volume_summary: Optional[Dict] = None) -> List[DetectedPattern]:
        """
        Detect ALL patterns in the DataFrame (30+ patterns)
        
        Args:
            df: OHLCV DataFrame
            regime: Optional market regime data
            sr_data: Optional support/resistance data
            volume_summary: Optional volume analysis data
        
        Returns:
            List of detected patterns (sorted by confidence)
        """
        if df is None or df.empty or len(df) < 20:
            return []
        
        patterns = []
        
        # Analyze all candles
        candles = self.candle_analyzer.analyze_all_candles(df)
        
        if not candles:
            return []
        
        # Calculate context metrics
        self._calculate_context_metrics(df, len(df) - 1)
        
        # Extract key levels from sr_data if available
        support_level = None
        resistance_level = None
        
        if sr_data:
            supports = sr_data.get('supports', [])
            resistances = sr_data.get('resistances', [])
            if supports:
                support_level = max(supports)  # Closest support below
            if resistances:
                resistance_level = min(resistances)  # Closest resistance above
        
        # Detect patterns on recent candles (last 50 for performance)
        recent_candles = candles[-50:] if len(candles) >= 50 else candles
        n = len(recent_candles)
        
        for i in range(n):
            candle = recent_candles[i]
            
            # ===== SINGLE CANDLE PATTERNS =====
            hammer = self.detect_hammer(candle, self.atr)
            if hammer:
                patterns.append(hammer)
            
            shooting_star = self.detect_shooting_star(candle, self.atr)
            if shooting_star:
                patterns.append(shooting_star)
            
            marubozu = self.detect_marubozu(candle, self.atr)
            if marubozu:
                patterns.append(marubozu)
            
            doji = self.detect_doji(candle, self.atr)
            if doji:
                patterns.append(doji)
            
            spinning_top = self.detect_spinning_top(candle, self.atr)
            if spinning_top:
                patterns.append(spinning_top)
            
            long_candle = self.detect_long_candle(candle, self.avg_body, self.atr)
            if long_candle:
                patterns.append(long_candle)
            
            # ===== DOUBLE CANDLE PATTERNS =====
            if i >= 1:
                c1 = recent_candles[i-1]
                c2 = candle
                
                bullish_engulfing = self.detect_bullish_engulfing(c1, c2, self.atr, self.avg_body)
                if bullish_engulfing:
                    patterns.append(bullish_engulfing)
                
                bearish_engulfing = self.detect_bearish_engulfing(c1, c2, self.atr, self.avg_body)
                if bearish_engulfing:
                    patterns.append(bearish_engulfing)
                
                inside_bar = self.detect_inside_bar(c1, c2, self.atr)
                if inside_bar:
                    patterns.append(inside_bar)
                
                outside_bar = self.detect_outside_bar(c1, c2, self.atr)
                if outside_bar:
                    patterns.append(outside_bar)
                
                tweezer_bottom = self.detect_tweezer_bottom(c1, c2, self.atr)
                if tweezer_bottom:
                    patterns.append(tweezer_bottom)
                
                tweezer_top = self.detect_tweezer_top(c1, c2, self.atr)
                if tweezer_top:
                    patterns.append(tweezer_top)
                
                harami_bullish = self.detect_harami_bullish(c1, c2, self.atr)
                if harami_bullish:
                    patterns.append(harami_bullish)
                
                harami_bearish = self.detect_harami_bearish(c1, c2, self.atr)
                if harami_bearish:
                    patterns.append(harami_bearish)
                
                taker_bullish = self.detect_taker_bullish(c1, c2, self.atr)
                if taker_bullish:
                    patterns.append(taker_bullish)
                
                taker_bearish = self.detect_taker_bearish(c1, c2, self.atr)
                if taker_bearish:
                    patterns.append(taker_bearish)
            
            # ===== TRIPLE CANDLE PATTERNS =====
            if i >= 2:
                c1 = recent_candles[i-2]
                c2 = recent_candles[i-1]
                c3 = candle
                
                morning_star = self.detect_morning_star(c1, c2, c3, self.atr, self.avg_body)
                if morning_star:
                    patterns.append(morning_star)
                
                evening_star = self.detect_evening_star(c1, c2, c3, self.atr, self.avg_body)
                if evening_star:
                    patterns.append(evening_star)
                
                three_white = self.detect_three_white_soldiers(c1, c2, c3, self.atr)
                if three_white:
                    patterns.append(three_white)
                
                three_black = self.detect_three_black_crows(c1, c2, c3, self.atr)
                if three_black:
                    patterns.append(three_black)
                
                three_inside_up = self.detect_three_inside_up(c1, c2, c3, self.atr)
                if three_inside_up:
                    patterns.append(three_inside_up)
                
                three_inside_down = self.detect_three_inside_down(c1, c2, c3, self.atr)
                if three_inside_down:
                    patterns.append(three_inside_down)
            
            # ===== MULTI-CANDLE PATTERNS =====
            if i >= 4:
                # Rising/Falling Three Methods
                rising = self.detect_rising_three_methods(recent_candles, i, self.atr)
                if rising:
                    patterns.append(rising)
                
                falling = self.detect_falling_three_methods(recent_candles, i, self.atr)
                if falling:
                    patterns.append(falling)
                
                # Three Line Strike
                three_line = self.detect_three_line_strike(recent_candles, i, self.atr)
                if three_line:
                    patterns.append(three_line)
            
            if i >= 2:
                identical = self.detect_identical_three_crows(recent_candles, i, self.atr)
                if identical:
                    patterns.append(identical)
            
            # ===== TRAP PATTERNS =====
            if i >= 1:
                bull_trap = self.detect_bull_trap(recent_candles, i, resistance_level, self.atr)
                if bull_trap:
                    patterns.append(bull_trap)
                
                bear_trap = self.detect_bear_trap(recent_candles, i, support_level, self.atr)
                if bear_trap:
                    patterns.append(bear_trap)
                
                stop_hunt = self.detect_stop_hunt(recent_candles, i, self.atr)
                if stop_hunt:
                    patterns.append(stop_hunt)
            
            # ===== MOMENTUM PATTERNS =====
            if i >= 1:
                consecutive_bull = self.detect_consecutive_bullish(recent_candles, i, self.atr)
                if consecutive_bull:
                    patterns.append(consecutive_bull)
                
                consecutive_bear = self.detect_consecutive_bearish(recent_candles, i, self.atr)
                if consecutive_bear:
                    patterns.append(consecutive_bear)
            
            if i >= MOMENTUM_SHIFT_WINDOW:
                momentum_shift = self.detect_momentum_shift(recent_candles, i, self.atr)
                if momentum_shift:
                    patterns.append(momentum_shift)
            
            if i >= 2:
                acceleration = self.detect_acceleration(recent_candles, i, self.atr)
                if acceleration:
                    patterns.append(acceleration)
                
                deceleration = self.detect_deceleration(recent_candles, i, self.atr)
                if deceleration:
                    patterns.append(deceleration)
        
        # Remove duplicates (keep highest confidence for same pattern at same index)
        unique_patterns = {}
        for p in patterns:
            key = f"{p.name}_{p.candle_index}"
            if key not in unique_patterns or p.confidence > unique_patterns[key].confidence:
                unique_patterns[key] = p
        
        patterns = list(unique_patterns.values())
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns

    def detect_patterns_at_index(self, df: pd.DataFrame, idx: int,
                                 regime: Optional[Dict] = None,
                                 sr_data: Optional[Dict] = None) -> List[DetectedPattern]:
        """
        Detect patterns at a specific index
        
        Args:
            df: OHLCV DataFrame
            idx: Index to analyze
            regime: Optional market regime data
            sr_data: Optional support/resistance data
        
        Returns:
            List of patterns detected at that index
        """
        if idx < 2 or idx >= len(df):
            return []
        
        # Get specific slice
        slice_df = df.iloc[max(0, idx-30):idx+1].copy()
        
        return self.detect_all_patterns(slice_df, regime, sr_data)
    
    def get_best_pattern(self, df: pd.DataFrame,
                        regime: Optional[Dict] = None,
                        sr_data: Optional[Dict] = None) -> Optional[DetectedPattern]:
        """
        Get the highest confidence pattern
        
        Args:
            df: OHLCV DataFrame
            regime: Optional market regime data
            sr_data: Optional support/resistance data
        
        Returns:
            Best pattern or None
        """
        patterns = self.detect_all_patterns(df, regime, sr_data)
        
        if not patterns:
            return None
        
        return patterns[0]


# ==================== CONVENIENCE FUNCTIONS ====================

def detect_patterns(df: pd.DataFrame, 
                   regime: Optional[Dict] = None,
                   sr_data: Optional[Dict] = None) -> List[DetectedPattern]:
    """
    Convenience function to detect all patterns
    
    Args:
        df: OHLCV DataFrame
        regime: Optional market regime data
        sr_data: Optional support/resistance data
    
    Returns:
        List of detected patterns
    """
    detector = PatternDetector()
    return detector.detect_all_patterns(df, regime, sr_data)


def get_best_pattern(df: pd.DataFrame,
                    regime: Optional[Dict] = None,
                    sr_data: Optional[Dict] = None) -> Optional[DetectedPattern]:
    """
    Convenience function to get best pattern
    
    Args:
        df: OHLCV DataFrame
        regime: Optional market regime data
        sr_data: Optional support/resistance data
    
    Returns:
        Best pattern or None
    """
    detector = PatternDetector()
    return detector.get_best_pattern(df, regime, sr_data)


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    # Generate candles with patterns
    data = []
    
    # Hammer pattern (candle 80)
    for i in range(100):
        if i == 80:  # Hammer
            open_p = 50000
            close_p = 50020
            high_p = 50030
            low_p = 49800  # Long lower wick
            volume = 15000
        elif i == 85:  # Shooting Star
            open_p = 50000
            close_p = 49980
            high_p = 50200  # Long upper wick
            low_p = 49970
            volume = 14000
        elif i == 90:  # Bullish Marubozu
            open_p = 50000
            close_p = 50200
            high_p = 50200
            low_p = 50000
            volume = 20000
        else:
            open_p = 50000 + np.random.randn() * 20
            close_p = open_p + np.random.randn() * 30
            high_p = max(open_p, close_p) + abs(np.random.randn() * 20)
            low_p = min(open_p, close_p) - abs(np.random.randn() * 20)
            volume = 10000 + abs(np.random.randn() * 5000)
        
        data.append({'open': open_p, 'high': high_p, 'low': low_p, 
                    'close': close_p, 'volume': volume})
    
    df = pd.DataFrame(data, index=dates)
    
    # Detect patterns
    detector = PatternDetector()
    patterns = detector.detect_all_patterns(df)
    
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns:
        print(f"  {p.name}: {p.direction.value} | Confidence: {p.confidence:.2f} | Quality: {p.pattern_quality}")
        print(f"    Reasons: {p.reasons[:2]}")
        print()