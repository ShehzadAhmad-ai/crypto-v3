"""
sequence_analyzer.py
Layer 5: Sequence Analyzer for Price Action Expert V3.5

Analyzes candle sequences to detect:
- Micro-patterns (3+ outside bars, inside bar combos, body expansion)
- Reversal sequences (bearish → indecision → bullish)
- Continuation sequences (strong → strong → strong)
- Compression sequences (inside bars → breakout)
- Story building for human-readable market interpretation

This is what makes the engine "read the story" like an expert trader
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import configuration
from .price_action_config import (
    SEQUENCE_WINDOW,
    MICRO_PATTERN_WINDOW,
    OUTSIDE_BAR_CONSECUTIVE,
    INSIDE_BAR_COMBO_MIN,
    BODY_EXPANSION_RATIO,
    VOLUME_DRYUP_RATIO,
    MICRO_PATTERN_BOOSTS
)

# Import candle analyzer
from .candle_analyzer import CandleData, CandleAnalyzer


class SequenceType(Enum):
    """Types of sequences detected"""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    COMPRESSION_BREAKOUT = "compression_breakout"
    MOMENTUM = "momentum"
    EXHAUSTION = "exhaustion"
    LIQUIDITY_RUN = "liquidity_run"
    INDECISION = "indecision"


class CandleClassification(Enum):
    """Classification for individual candles in sequence"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    WEAK_BULLISH = "weak_bullish"
    NEUTRAL = "neutral"
    WEAK_BEARISH = "weak_bearish"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"
    DOJI = "doji"
    REJECTION_UP = "rejection_up"
    REJECTION_DOWN = "rejection_down"


@dataclass
class MicroPattern:
    """Detected micro-pattern within a sequence"""
    type: str
    strength: float           # 0-1
    candles_involved: int
    description: str


@dataclass
class CandleStory:
    """
    Complete story of the candle sequence
    """
    # Core analysis
    sequence_type: SequenceType
    sequence_confidence: float      # 0-1
    momentum_score: float           # 0-1 (higher = stronger momentum)
    
    # Classifications
    candle_classifications: List[str]
    micro_patterns: List[MicroPattern]
    
    # Story elements
    story_summary: str
    full_story: str
    key_moments: List[str]
    risk_warnings: List[str]
    
    # Raw data
    sequence_length: int
    last_candle_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            'sequence_type': self.sequence_type.value,
            'sequence_confidence': round(self.sequence_confidence, 3),
            'momentum_score': round(self.momentum_score, 3),
            'candle_classifications': self.candle_classifications[-10:],
            'micro_patterns': [
                {'type': mp.type, 'strength': mp.strength, 'candles': mp.candles_involved}
                for mp in self.micro_patterns[:5]
            ],
            'story_summary': self.story_summary,
            'full_story': self.full_story,
            'key_moments': self.key_moments[:5],
            'risk_warnings': self.risk_warnings[:3],
            'sequence_length': self.sequence_length
        }


class SequenceAnalyzer:
    """
    Advanced sequence analyzer that reads the market story
    
    Features:
    - Candle classification (strong/weak/indecision)
    - Micro-pattern detection
    - Reversal sequence detection
    - Continuation sequence detection
    - Compression breakout detection
    - Story building (human-readable)
    """
    
    def __init__(self):
        """Initialize the sequence analyzer"""
        self.candle_analyzer = CandleAnalyzer()
        self.sequence_history: List[CandleStory] = []
    
    # =========================================================
    # CANDLE CLASSIFICATION
    # =========================================================
    
    def classify_candle(self, candle: CandleData, atr: float) -> CandleClassification:
        """
        Classify a single candle into categories
        
        Categories:
        - Strong Bullish: Large body, small wicks, close high
        - Bullish: Body > 0.5 ATR, bullish
        - Weak Bullish: Small body, bullish
        - Strong Bearish: Large body, small wicks, close low
        - Bearish: Body > 0.5 ATR, bearish
        - Weak Bearish: Small body, bearish
        - Doji: Body < 0.1 range
        - Rejection Up: Long upper wick
        - Rejection Down: Long lower wick
        """
        # Doji classification
        if candle.is_doji:
            return CandleClassification.DOJI
        
        # Rejection classifications
        if candle.is_rejection_top and candle.upper_wick > candle.body_abs * 2:
            return CandleClassification.REJECTION_UP
        
        if candle.is_rejection_bottom and candle.lower_wick > candle.body_abs * 2:
            return CandleClassification.REJECTION_DOWN
        
        # Strong classification based on body size
        if candle.body_atr >= 0.8:
            if candle.is_bullish:
                return CandleClassification.STRONG_BULLISH
            else:
                return CandleClassification.STRONG_BEARISH
        
        # Moderate classification
        if candle.body_atr >= 0.4:
            if candle.is_bullish:
                return CandleClassification.BULLISH
            else:
                return CandleClassification.BEARISH
        
        # Weak classification
        if candle.is_bullish:
            return CandleClassification.WEAK_BULLISH
        elif candle.is_bearish:
            return CandleClassification.WEAK_BEARISH
        else:
            return CandleClassification.NEUTRAL
    
    def get_classification_score(self, classification: CandleClassification) -> float:
        """Get numeric score for a classification (-1 to 1)"""
        scores = {
            CandleClassification.STRONG_BULLISH: 0.9,
            CandleClassification.BULLISH: 0.6,
            CandleClassification.WEAK_BULLISH: 0.3,
            CandleClassification.NEUTRAL: 0.0,
            CandleClassification.WEAK_BEARISH: -0.3,
            CandleClassification.BEARISH: -0.6,
            CandleClassification.STRONG_BEARISH: -0.9,
            CandleClassification.DOJI: 0.0,
            CandleClassification.REJECTION_UP: -0.5,
            CandleClassification.REJECTION_DOWN: 0.5,
        }
        return scores.get(classification, 0.0)
    
    # =========================================================
    # MICRO-PATTERN DETECTION
    # =========================================================
    
    def detect_micro_patterns(self, candles: List[CandleData], 
                               classifications: List[CandleClassification],
                               idx: int) -> List[MicroPattern]:
        """
        Detect micro-patterns within the candle sequence
        
        Micro-patterns detected:
        - Consecutive outside bars
        - Inside bar combo
        - Body expansion
        - Body contraction
        - Reversal pattern
        - Liquidity run
        """
        if idx < 3:
            return []
        
        patterns = []
        
        # ===== 1. Consecutive Outside Bars =====
        outside_count = 0
        for i in range(max(0, idx - OUTSIDE_BAR_CONSECUTIVE + 1), idx + 1):
            if i >= 1 and i < len(candles):
                if (candles[i].high > candles[i-1].high and 
                    candles[i].low < candles[i-1].low):
                    outside_count += 1
                else:
                    outside_count = 0
            else:
                outside_count = 0
        
        if outside_count >= OUTSIDE_BAR_CONSECUTIVE:
            boost = MICRO_PATTERN_BOOSTS.get('3_consecutive_outside_bars', 0.15)
            patterns.append(MicroPattern(
                type='consecutive_outside_bars',
                strength=boost,
                candles_involved=outside_count,
                description=f'{outside_count} consecutive outside bars - high momentum'
            ))
        
        # ===== 2. Inside Bar Combo =====
        inside_count = 0
        for i in range(max(0, idx - INSIDE_BAR_COMBO_MIN + 1), idx + 1):
            if i >= 1 and i < len(candles):
                if (candles[i].high < candles[i-1].high and 
                    candles[i].low > candles[i-1].low):
                    inside_count += 1
                else:
                    inside_count = 0
        
        if inside_count >= INSIDE_BAR_COMBO_MIN:
            boost = MICRO_PATTERN_BOOSTS.get('inside_bar_combo_breakout', 0.20)
            patterns.append(MicroPattern(
                type='inside_bar_combo',
                strength=boost,
                candles_involved=inside_count,
                description=f'{inside_count} consecutive inside bars - compression'
            ))
        
        # ===== 3. Body Expansion =====
        if idx >= 2:
            body_prev = candles[idx-2].body_abs
            body_mid = candles[idx-1].body_abs
            body_curr = candles[idx].body_abs
            
            if body_mid > body_prev * BODY_EXPANSION_RATIO and body_curr > body_mid * BODY_EXPANSION_RATIO:
                boost = MICRO_PATTERN_BOOSTS.get('body_expansion', 0.15)
                patterns.append(MicroPattern(
                    type='body_expansion',
                    strength=boost,
                    candles_involved=3,
                    description='Expanding bodies - momentum acceleration'
                ))
            elif body_mid > body_prev * BODY_EXPANSION_RATIO:
                boost = MICRO_PATTERN_BOOSTS.get('body_expansion', 0.15) * 0.6
                patterns.append(MicroPattern(
                    type='body_expansion',
                    strength=boost,
                    candles_involved=2,
                    description='Body expansion - momentum building'
                ))
        
        # ===== 4. Body Contraction =====
        if idx >= 2:
            body_prev = candles[idx-2].body_abs
            body_mid = candles[idx-1].body_abs
            body_curr = candles[idx].body_abs
            
            if body_mid < body_prev * (1 / BODY_EXPANSION_RATIO) and body_curr < body_mid * (1 / BODY_EXPANSION_RATIO):
                boost = MICRO_PATTERN_BOOSTS.get('body_contraction', -0.10)
                patterns.append(MicroPattern(
                    type='body_contraction',
                    strength=boost,
                    candles_involved=3,
                    description='Contracting bodies - momentum deceleration'
                ))
        
        # ===== 5. Reversal Pattern =====
        if idx >= 2 and len(classifications) >= 3:
            c1 = classifications[idx-2]
            c2 = classifications[idx-1]
            c3 = classifications[idx]
            
            # Bearish to bullish reversal
            if (c1 in [CandleClassification.BEARISH, CandleClassification.STRONG_BEARISH] and
                c2 in [CandleClassification.DOJI, CandleClassification.NEUTRAL] and
                c3 in [CandleClassification.BULLISH, CandleClassification.STRONG_BULLISH]):
                boost = MICRO_PATTERN_BOOSTS.get('reversal_sequence', 0.20)
                patterns.append(MicroPattern(
                    type='reversal_pattern',
                    strength=boost,
                    candles_involved=3,
                    description='Bearish → indecision → bullish reversal'
                ))
            
            # Bullish to bearish reversal
            if (c1 in [CandleClassification.BULLISH, CandleClassification.STRONG_BULLISH] and
                c2 in [CandleClassification.DOJI, CandleClassification.NEUTRAL] and
                c3 in [CandleClassification.BEARISH, CandleClassification.STRONG_BEARISH]):
                boost = MICRO_PATTERN_BOOSTS.get('reversal_sequence', 0.20)
                patterns.append(MicroPattern(
                    type='reversal_pattern',
                    strength=boost,
                    candles_involved=3,
                    description='Bullish → indecision → bearish reversal'
                ))
        
        # ===== 6. Liquidity Run =====
        if idx >= 2:
            # Check for long wick followed by strong reversal
            if (classifications[idx-1] in [CandleClassification.REJECTION_UP, CandleClassification.REJECTION_DOWN] and
                classifications[idx] in [CandleClassification.STRONG_BULLISH, CandleClassification.STRONG_BEARISH]):
                boost = MICRO_PATTERN_BOOSTS.get('liquidity_run', 0.25)
                patterns.append(MicroPattern(
                    type='liquidity_run',
                    strength=boost,
                    candles_involved=2,
                    description='Rejection wick followed by strong reversal - liquidity sweep'
                ))
        
        return patterns
    
    # =========================================================
    # SEQUENCE TYPE DETECTION
    # =========================================================
    
    def detect_sequence_type(self, classifications: List[CandleClassification],
                              micro_patterns: List[MicroPattern],
                              scores: List[float]) -> Tuple[SequenceType, float]:
        """
        Detect the overall sequence type based on classifications and micro-patterns
        """
        if len(scores) < 3:
            return SequenceType.INDECISION, 0.3
        
        # Calculate trend score (average of last 5 candles)
        trend_score = np.mean(scores[-5:]) if len(scores) >= 5 else np.mean(scores)
        
        # Check for reversal
        if len(classifications) >= 3:
            last_three = classifications[-3:]
            if (last_three[0] in [CandleClassification.BEARISH, CandleClassification.STRONG_BEARISH] and
                last_three[1] in [CandleClassification.DOJI, CandleClassification.NEUTRAL] and
                last_three[2] in [CandleClassification.BULLISH, CandleClassification.STRONG_BULLISH]):
                return SequenceType.REVERSAL, 0.8
            if (last_three[0] in [CandleClassification.BULLISH, CandleClassification.STRONG_BULLISH] and
                last_three[1] in [CandleClassification.DOJI, CandleClassification.NEUTRAL] and
                last_three[2] in [CandleClassification.BEARISH, CandleClassification.STRONG_BEARISH]):
                return SequenceType.REVERSAL, 0.8
        
        # Check for compression breakout
        for mp in micro_patterns:
            if mp.type == 'inside_bar_combo':
                # Check if breakout occurred
                if abs(trend_score) > 0.5:
                    return SequenceType.COMPRESSION_BREAKOUT, 0.75 + mp.strength
        
        # Check for momentum (strong consistent direction)
        if abs(trend_score) > 0.6:
            # Check if all recent candles are in same direction
            recent_dirs = [1 if s > 0 else -1 if s < 0 else 0 for s in scores[-5:]]
            if all(d == recent_dirs[0] for d in recent_dirs if d != 0):
                if trend_score > 0:
                    return SequenceType.MOMENTUM, 0.85
                else:
                    return SequenceType.MOMENTUM, 0.85
        
        # Check for continuation
        if abs(trend_score) > 0.3:
            return SequenceType.CONTINUATION, 0.6
        
        # Check for exhaustion
        for mp in micro_patterns:
            if mp.type == 'body_contraction':
                return SequenceType.EXHAUSTION, 0.65
        
        # Check for liquidity run
        for mp in micro_patterns:
            if mp.type == 'liquidity_run':
                return SequenceType.LIQUIDITY_RUN, 0.8 + mp.strength
        
        return SequenceType.INDECISION, 0.4
    
    # =========================================================
    # MOMENTUM SCORE CALCULATION
    # =========================================================
    
    def calculate_momentum_score(self, classifications: List[CandleClassification],
                                  scores: List[float]) -> float:
        """
        Calculate overall momentum score (0-1)
        Higher score = stronger directional momentum
        """
        if len(scores) < 3:
            return 0.5
        
        # Weighted average of recent scores (more weight to recent)
        weights = np.exp(np.linspace(0, 1, len(scores[-5:])))
        weights = weights / weights.sum()
        weighted_score = np.average(scores[-5:], weights=weights)
        
        # Convert from -1 to 1 to 0 to 1
        momentum = (weighted_score + 1) / 2
        
        # Adjust for consistency
        recent_dirs = [1 if s > 0 else -1 if s < 0 else 0 for s in scores[-5:]]
        consistent = all(d == recent_dirs[0] for d in recent_dirs if d != 0)
        if consistent:
            momentum += 0.1
        
        return min(0.95, momentum)
    
    # =========================================================
    # STORY BUILDING
    # =========================================================
    
    def build_story(self, candles: List[CandleData],
                    classifications: List[CandleClassification],
                    micro_patterns: List[MicroPattern],
                    sequence_type: SequenceType,
                    momentum_score: float) -> CandleStory:
        """
        Build a human-readable story from the sequence
        """
        # ===== Story Summary =====
        summary_parts = []
        
        if sequence_type == SequenceType.REVERSAL:
            if len(classifications) >= 3:
                first = classifications[-3]
                last = classifications[-1]
                if first in [CandleClassification.BEARISH, CandleClassification.STRONG_BEARISH]:
                    summary_parts.append("Selling pressure exhausted")
                else:
                    summary_parts.append("Buying pressure exhausted")
                summary_parts.append("indecision followed by")
                if last in [CandleClassification.BULLISH, CandleClassification.STRONG_BULLISH]:
                    summary_parts.append("buyers stepping in")
                else:
                    summary_parts.append("sellers stepping in")
        
        elif sequence_type == SequenceType.COMPRESSION_BREAKOUT:
            summary_parts.append("Compression after")
            # Find direction
            if len(scores := [self.get_classification_score(c) for c in classifications[-5:]]) > 0:
                direction = "bullish" if np.mean(scores) > 0 else "bearish"
                summary_parts.append(f"{direction} breakout")
            else:
                summary_parts.append("breakout")
        
        elif sequence_type == SequenceType.MOMENTUM:
            direction = "bullish" if momentum_score > 0.6 else "bearish"
            summary_parts.append(f"Strong {direction} momentum")
            summary_parts.append("with accelerating price action")
        
        elif sequence_type == SequenceType.LIQUIDITY_RUN:
            summary_parts.append("Liquidity sweep detected")
            summary_parts.append("smart money taking stops")
            summary_parts.append("reversal expected")
        
        elif sequence_type == SequenceType.EXHAUSTION:
            summary_parts.append("Momentum exhaustion")
            summary_parts.append("decreasing body sizes")
            summary_parts.append("potential reversal")
        
        else:
            summary_parts.append("Market indecision")
            summary_parts.append("waiting for direction")
        
        story_summary = " → ".join(summary_parts[:3])
        
        # ===== Full Story =====
        full_parts = []
        
        # Add classification sequence
        class_names = [c.value.replace('_', ' ') for c in classifications[-7:]]
        full_parts.append(f"Candle sequence: {' → '.join(class_names)}")
        
        # Add micro-patterns
        for mp in micro_patterns[:3]:
            full_parts.append(mp.description)
        
        # Add momentum assessment
        if momentum_score > 0.7:
            full_parts.append(f"Strong momentum (score: {momentum_score:.0%})")
        elif momentum_score > 0.5:
            full_parts.append(f"Moderate momentum (score: {momentum_score:.0%})")
        else:
            full_parts.append(f"Weak momentum (score: {momentum_score:.0%})")
        
        # Add sequence interpretation
        if sequence_type == SequenceType.REVERSAL:
            full_parts.append("This is a classic reversal pattern - consider fading the previous trend")
        elif sequence_type == SequenceType.COMPRESSION_BREAKOUT:
            full_parts.append("Compression resolved - expect expansion")
        elif sequence_type == SequenceType.MOMENTUM:
            full_parts.append("Strong directional momentum - consider continuation")
        elif sequence_type == SequenceType.LIQUIDITY_RUN:
            full_parts.append("Liquidity was taken - expect smart money reversal")
        elif sequence_type == SequenceType.EXHAUSTION:
            full_parts.append("Momentum is dying - avoid chasing")
        
        full_story = ". ".join(full_parts)
        
        # ===== Key Moments =====
        key_moments = []
        for mp in micro_patterns[:3]:
            key_moments.append(mp.description)
        
        # Add significant candles
        for i, c in enumerate(classifications[-3:]):
            if c in [CandleClassification.STRONG_BULLISH, CandleClassification.STRONG_BEARISH]:
                key_moments.append(f"Strong {c.value} candle at position {len(classifications)-3+i+1}")
            elif c in [CandleClassification.REJECTION_UP, CandleClassification.REJECTION_DOWN]:
                key_moments.append(f"Rejection {c.value} detected")
        
        # ===== Risk Warnings =====
        risk_warnings = []
        
        # Check for indecision
        doji_count = sum(1 for c in classifications[-5:] if c == CandleClassification.DOJI)
        if doji_count >= 2:
            risk_warnings.append("Multiple doji candles - high indecision")
        
        # Check for contraction
        for mp in micro_patterns:
            if mp.type == 'body_contraction':
                risk_warnings.append("Contracting bodies - momentum dying")
        
        # Check for volatility
        recent_bodies = [c.body_abs for c in candles[-5:]]
        if len(recent_bodies) >= 2:
            body_volatility = np.std(recent_bodies) / np.mean(recent_bodies)
            if body_volatility > 0.5:
                risk_warnings.append("High body size volatility - erratic movement")
        
        return CandleStory(
            sequence_type=sequence_type,
            sequence_confidence=0.7,
            momentum_score=momentum_score,
            candle_classifications=[c.value for c in classifications],
            micro_patterns=micro_patterns,
            story_summary=story_summary,
            full_story=full_story,
            key_moments=key_moments,
            risk_warnings=risk_warnings,
            sequence_length=len(candles),
            last_candle_index=candles[-1].index if candles else 0
        )
    
    # =========================================================
    # COMPLETE SEQUENCE ANALYSIS
    # =========================================================
    
    def analyze_sequence(self, df: pd.DataFrame, 
                         lookback: int = SEQUENCE_WINDOW) -> CandleStory:
        """
        Analyze the recent candle sequence and build story
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of candles to analyze
        
        Returns:
            CandleStory with full analysis
        """
        if df is None or df.empty or len(df) < lookback:
            return self._default_story()
        
        # Analyze candles
        candles = self.candle_analyzer.analyze_all_candles(df)
        
        if len(candles) < lookback:
            return self._default_story()
        
        # Get recent candles
        recent_candles = candles[-lookback:]
        
        # Get ATR
        atr = float(df['atr'].iloc[-1]) if 'atr' in df else float(df['close'].iloc[-1]) * 0.02
        
        # Classify each candle
        classifications = []
        scores = []
        
        for candle in recent_candles:
            cls = self.classify_candle(candle, atr)
            classifications.append(cls)
            scores.append(self.get_classification_score(cls))
        
        # Detect micro-patterns
        all_micro_patterns = []
        for i in range(len(recent_candles)):
            patterns = self.detect_micro_patterns(recent_candles, classifications, i)
            all_micro_patterns.extend(patterns)
        
        # Remove duplicates (keep highest strength for same type)
        unique_patterns = {}
        for p in all_micro_patterns:
            if p.type not in unique_patterns or p.strength > unique_patterns[p.type].strength:
                unique_patterns[p.type] = p
        
        micro_patterns = list(unique_patterns.values())
        
        # Detect sequence type
        seq_type, seq_confidence = self.detect_sequence_type(classifications, micro_patterns, scores)
        
        # Calculate momentum score
        momentum_score = self.calculate_momentum_score(classifications, scores)
        
        # Build story
        story = self.build_story(
            recent_candles, classifications, micro_patterns, seq_type, momentum_score
        )
        
        # Add sequence confidence
        story.sequence_confidence = seq_confidence
        
        # Store history
        self.sequence_history.append(story)
        if len(self.sequence_history) > 10:
            self.sequence_history.pop(0)
        
        return story
    
    def _default_story(self) -> CandleStory:
        """Return default story when analysis fails"""
        return CandleStory(
            sequence_type=SequenceType.INDECISION,
            sequence_confidence=0.3,
            momentum_score=0.5,
            candle_classifications=[],
            micro_patterns=[],
            story_summary="Insufficient data for sequence analysis",
            full_story="Not enough candles to analyze sequence.",
            key_moments=[],
            risk_warnings=["Insufficient data"],
            sequence_length=0,
            last_candle_index=0
        )


# ==================== CONVENIENCE FUNCTIONS ====================

def analyze_sequence(df: pd.DataFrame, lookback: int = 10) -> CandleStory:
    """
    Convenience function to analyze candle sequence

    Args:
        df: OHLCV DataFrame
        lookback: Number of candles to analyze
    
    Returns:
        CandleStory with full analysis
    """
    analyzer = SequenceAnalyzer()
    return analyzer.analyze_sequence(df, lookback)


def get_sequence_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to get sequence summary
    
    Args:
        df: OHLCV DataFrame
    
    Returns:
        Summary dictionary
    """
    story = analyze_sequence(df)
    return story.to_dict()


def get_momentum_score(df: pd.DataFrame) -> float:
    """
    Convenience function to get momentum score (0-1)
        
    Args:
        df: OHLCV DataFrame
    
    Returns:
    Momentum score (higher = stronger momentum)
    """
    story = analyze_sequence(df)
    return story.momentum_score


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Create test data with a reversal sequence
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    # Create a reversal sequence: bearish → indecision → bullish
    prices = []
    
    # Bearish candles
    for i in range(30):
        prices.append(50000 - i * 20)
    
    # Indecision candles
    for i in range(5):
        prices.append(prices[-1] + np.random.randn() * 10)
    
    # Bullish candles
    for i in range(30):
        prices.append(prices[-1] + i * 15)
    
    # Add some random noise
    remaining = 100 - len(prices)
    for i in range(remaining):
        prices.append(prices[-1] + np.random.randn() * 20)
    
    data = []
    for i, close in enumerate(prices):
        open_p = close - np.random.randn() * 15
        high = max(open_p, close) + abs(np.random.randn() * 25)
        low = min(open_p, close) - abs(np.random.randn() * 25)
        volume = abs(np.random.randn() * 10000) + 5000
        
        # Create doji for indecision
        if 35 <= i <= 39:
            open_p = close - 2
            high = close + 15
            low = close - 15
            volume = 8000
        
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
    
    # Analyze sequence
    analyzer = SequenceAnalyzer()
    story = analyzer.analyze_sequence(df)
    
    print("=" * 70)
    print("SEQUENCE ANALYSIS - MARKET STORY")
    print("=" * 70)
    print(f"\nSequence Type: {story.sequence_type.value.upper()}")
    print(f"Confidence: {story.sequence_confidence:.1%}")
    print(f"Momentum Score: {story.momentum_score:.1%}")
    print(f"\nStory Summary:")
    print(f"  {story.story_summary}")
    print(f"\nFull Story:")
    print(f"  {story.full_story}")
    print(f"\nKey Moments:")
    for m in story.key_moments[:5]:
        print(f"  • {m}")
    print(f"\nRisk Warnings:")
    for w in story.risk_warnings[:3]:
        print(f"  ⚠ {w}")
    print(f"\nRecent Candle Classifications:")
    for i, cls in enumerate(story.candle_classifications[-10:]):
        print(f"  {i+1}: {cls}")
    print("=" * 70)