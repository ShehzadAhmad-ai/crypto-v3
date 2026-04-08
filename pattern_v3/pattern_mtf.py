"""
pattern_mtf.py - Multi-Timeframe Confluence Analyzer for Pattern V4

Analyzes patterns across multiple timeframes to provide:
- Confluence score (how many timeframes agree)
- Boost factor for confidence (1.0 to 1.5)
- Same pattern detection on higher timeframes
- Trend alignment across timeframes

Supports: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .pattern_config import CONFIG
from .pattern_detection import PatternDetectionEngineV4


# ============================================================================
# TIMEFRAME ORDERING
# ============================================================================

TIMEFRAME_ORDER = {
    '1m': 1,
    '5m': 2,
    '15m': 3,
    '30m': 4,
    '1h': 5,
    '2h': 6,
    '4h': 7,
    '1d': 8,
    '1w': 9,
}

TIMEFRAME_WEIGHTS = {
    '1m': 0.05,
    '5m': 0.10,
    '15m': 0.15,
    '30m': 0.20,
    '1h': 0.25,
    '2h': 0.28,
    '4h': 0.30,
    '1d': 0.15,
    '1w': 0.10,
}


# ============================================================================
# MTF CONFLUENCE RESULT
# ============================================================================

class MTFConfluenceResult:
    """
    Multi-timeframe confluence analysis result.
    """
    
    def __init__(self):
        self.weighted_score: float = 0.0
        self.boost_factor: float = 1.0
        self.timeframe_scores: Dict[str, float] = {}
        self.timeframe_patterns: Dict[str, str] = {}
        self.same_pattern_count: int = 0
        self.aligned_trend_count: int = 0
        self.conflicting_count: int = 0
        self.same_pattern_timeframes: List[str] = []
        self.aligned_trend_timeframes: List[str] = []
        self.conflicting_timeframes: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'weighted_score': round(self.weighted_score, 3),
            'boost_factor': round(self.boost_factor, 3),
            'timeframe_scores': self.timeframe_scores,
            'timeframe_patterns': self.timeframe_patterns,
            'same_pattern_count': self.same_pattern_count,
            'aligned_trend_count': self.aligned_trend_count,
            'same_pattern_timeframes': self.same_pattern_timeframes,
            'aligned_trend_timeframes': self.aligned_trend_timeframes,
        }


# ============================================================================
# MTF ANALYZER
# ============================================================================

class MultiTimeframeAnalyzerV4:
    """
    Analyzes patterns across multiple timeframes.
    """
    
    def __init__(self):
        self.detection_engine = PatternDetectionEngineV4()
        self.mtf_config = CONFIG.mtf_config if hasattr(CONFIG, 'mtf_config') else {
            'enabled': True,
            'timeframe_weights': TIMEFRAME_WEIGHTS,
            'same_pattern_boost': 1.15,
            'aligned_trend_boost': 1.10,
            'conflicting_penalty': 0.85,
            'max_boost': 1.30,
            'min_boost': 0.70,
        }
        self.tf_weights = self.mtf_config.get('timeframe_weights', TIMEFRAME_WEIGHTS)
    
    def analyze_confluence(self, pattern: Dict, htf_data: Dict[str, pd.DataFrame],
                           primary_tf: str) -> MTFConfluenceResult:
        """
        Analyze confluence across all available timeframes.
        
        Args:
            pattern: Detected pattern on primary timeframe
            htf_data: Dictionary of {timeframe: DataFrame} for higher timeframes
            primary_tf: Primary timeframe (e.g., '1h')
        
        Returns:
            MTFConfluenceResult with scores and boost factor
        """
        result = MTFConfluenceResult()
        
        if not self.mtf_config.get('enabled', True):
            result.boost_factor = 1.0
            return result
        
        pattern_name = pattern.get('pattern_name', 'Unknown')
        pattern_direction = pattern.get('direction', 'NEUTRAL')
        
        # Analyze each higher timeframe
        for tf, df_tf in htf_data.items():
            if not self._is_higher_tf(tf, primary_tf):
                continue
            
            # Detect patterns on this timeframe
            tf_patterns, tf_swings = self.detection_engine.detect_all_patterns(df_tf)
            
            # Score for this timeframe
            tf_score = self._score_timeframe_confluence(
                pattern_name, pattern_direction, tf_patterns, tf_swings, df_tf
            )
            
            result.timeframe_scores[tf] = tf_score
            
            # Track patterns found
            for tf_pattern in tf_patterns:
                if tf_pattern.get('pattern_name') == pattern_name:
                    result.same_pattern_count += 1
                    result.same_pattern_timeframes.append(tf)
                    result.timeframe_patterns[tf] = tf_pattern.get('pattern_name')
            
            # Track trend alignment
            tf_trend = self._get_timeframe_trend(tf_swings, df_tf)
            if self._is_trend_aligned(pattern_direction, tf_trend):
                result.aligned_trend_count += 1
                result.aligned_trend_timeframes.append(tf)
            elif self._is_trend_conflicting(pattern_direction, tf_trend):
                result.conflicting_count += 1
                result.conflicting_timeframes.append(tf)
        
        # Calculate weighted score
        result.weighted_score = self._calculate_weighted_score(result.timeframe_scores)
        
        # Calculate boost factor
        result.boost_factor = self._calculate_boost_factor(result)
        
        return result
    
    def _is_higher_tf(self, tf: str, primary_tf: str) -> bool:
        """Check if timeframe is higher than primary"""
        tf_order = TIMEFRAME_ORDER.get(tf, 0)
        primary_order = TIMEFRAME_ORDER.get(primary_tf, 0)
        return tf_order > primary_order
    
    def _score_timeframe_confluence(self, pattern_name: str, pattern_direction: str,
                                     tf_patterns: List[Dict], tf_swings: List,
                                     df: pd.DataFrame) -> float:
        """
        Score how well the higher timeframe aligns with the pattern.
        Returns score from 0 to 1.
        """
        score = 0.5  # Neutral base
        
        # Check for same pattern
        for tf_pattern in tf_patterns:
            if tf_pattern.get('pattern_name') == pattern_name:
                score += 0.3
                break
        
        # Check for same direction patterns
        same_direction_patterns = [p for p in tf_patterns if p.get('direction') == pattern_direction]
        if same_direction_patterns:
            score += min(0.2, len(same_direction_patterns) * 0.05)
        
        # Check trend alignment
        tf_trend = self._get_timeframe_trend(tf_swings, df)
        if self._is_trend_aligned(pattern_direction, tf_trend):
            score += 0.2
        elif self._is_trend_conflicting(pattern_direction, tf_trend):
            score -= 0.15
        
        # Check for key levels
        if self._is_at_key_level(df, pattern_direction):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _get_timeframe_trend(self, swings: List, df: pd.DataFrame) -> str:
        """Determine trend on a timeframe"""
        if len(swings) < 4:
            # Fallback to EMA
            if len(df) >= 50:
                ema20 = df['close'].ewm(span=20).mean().iloc[-1]
                ema50 = df['close'].ewm(span=50).mean().iloc[-1]
                if ema20 > ema50:
                    return 'BULLISH'
                elif ema20 < ema50:
                    return 'BEARISH'
            return 'NEUTRAL'
        
        # Analyze recent swings
        recent = swings[-4:]
        bullish_swings = sum(1 for s in recent if s.type in ['HH', 'HL'])
        bearish_swings = sum(1 for s in recent if s.type in ['LH', 'LL'])
        
        if bullish_swings > bearish_swings:
            return 'BULLISH'
        elif bearish_swings > bullish_swings:
            return 'BEARISH'
        return 'NEUTRAL'
    
    def _is_trend_aligned(self, pattern_direction: str, tf_trend: str) -> bool:
        """Check if pattern direction aligns with trend"""
        if pattern_direction == 'BUY' and tf_trend == 'BULLISH':
            return True
        if pattern_direction == 'SELL' and tf_trend == 'BEARISH':
            return True
        return False
    
    def _is_trend_conflicting(self, pattern_direction: str, tf_trend: str) -> bool:
        """Check if pattern direction conflicts with trend"""
        if pattern_direction == 'BUY' and tf_trend == 'BEARISH':
            return True
        if pattern_direction == 'SELL' and tf_trend == 'BULLISH':
            return True
        return False
    
    def _is_at_key_level(self, df: pd.DataFrame, direction: str) -> bool:
        """Check if price is at a key level (support/resistance)"""
        if len(df) < 50:
            return False
        
        current_price = df['close'].iloc[-1]
        
        # Find recent highs and lows
        recent_highs = df['high'].iloc[-50:].nlargest(5).values
        recent_lows = df['low'].iloc[-50:].nsmallest(5).values
        
        if direction == 'BUY':
            # Check if near support
            for low in recent_lows:
                if abs(current_price - low) / current_price < 0.01:
                    return True
        else:
            # Check if near resistance
            for high in recent_highs:
                if abs(current_price - high) / current_price < 0.01:
                    return True
        
        return False
    
    def _calculate_weighted_score(self, timeframe_scores: Dict[str, float]) -> float:
        """Calculate weighted average of timeframe scores"""
        if not timeframe_scores:
            return 0.5
        
        total_weight = 0
        weighted_sum = 0
        
        for tf, score in timeframe_scores.items():
            weight = self.tf_weights.get(tf, 0.10)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def _calculate_boost_factor(self, result: MTFConfluenceResult) -> float:
        """Calculate boost factor based on confluence"""
        boost = 1.0
        
        # Same pattern boost
        if result.same_pattern_count >= 2:
            boost *= 1.20
        elif result.same_pattern_count >= 1:
            boost *= self.mtf_config.get('same_pattern_boost', 1.15)
        
        # Aligned trend boost
        if result.aligned_trend_count >= 2:
            boost *= 1.15
        elif result.aligned_trend_count >= 1:
            boost *= self.mtf_config.get('aligned_trend_boost', 1.10)
        
        # Conflicting penalty
        if result.conflicting_count >= 2:
            boost *= 0.75
        elif result.conflicting_count >= 1:
            boost *= self.mtf_config.get('conflicting_penalty', 0.85)
        
        # Apply limits
        max_boost = self.mtf_config.get('max_boost', 1.30)
        min_boost = self.mtf_config.get('min_boost', 0.70)
        
        return min(max_boost, max(min_boost, boost))
    
    def get_best_timeframe_for_pattern(self, pattern_name: str, 
                                        htf_data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Find which timeframe shows the pattern most clearly.
        """
        best_tf = None
        best_score = 0.0
        
        for tf, df_tf in htf_data.items():
            tf_patterns, _ = self.detection_engine.detect_all_patterns(df_tf)
            
            for tf_pattern in tf_patterns:
                if tf_pattern.get('pattern_name') == pattern_name:
                    similarity = tf_pattern.get('similarity', 0)
                    if similarity > best_score:
                        best_score = similarity
                        best_tf = tf
        
        return best_tf
    
    def get_mtf_summary(self, pattern: Dict, htf_data: Dict[str, pd.DataFrame],
                        primary_tf: str) -> Dict[str, Any]:
        """
        Get comprehensive MTF summary for logging.
        """
        result = self.analyze_confluence(pattern, htf_data, primary_tf)
        
        return {
            'primary_timeframe': primary_tf,
            'timeframes_analyzed': list(htf_data.keys()),
            'weighted_confluence': result.weighted_score,
            'boost_factor': result.boost_factor,
            'same_pattern_on': result.same_pattern_timeframes,
            'aligned_trend_on': result.aligned_trend_timeframes,
            'conflicting_on': result.conflicting_timeframes,
            'timeframe_scores': result.timeframe_scores,
        }


# ============================================================================
# SIMPLE MTF HELPER FUNCTIONS
# ============================================================================

def get_higher_timeframes(current_tf: str) -> List[str]:
    """
    Get list of higher timeframes relative to current.
    """
    all_tfs = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w']
    current_index = TIMEFRAME_ORDER.get(current_tf, 0)
    
    return [tf for tf in all_tfs if TIMEFRAME_ORDER.get(tf, 0) > current_index]


def get_lower_timeframes(current_tf: str) -> List[str]:
    """
    Get list of lower timeframes relative to current.
    """
    all_tfs = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w']
    current_index = TIMEFRAME_ORDER.get(current_tf, 0)
    
    return [tf for tf in all_tfs if TIMEFRAME_ORDER.get(tf, 0) < current_index]


def aggregate_to_higher_tf(df: pd.DataFrame, current_tf: str, target_tf: str) -> Optional[pd.DataFrame]:
    """
    Aggregate data from current timeframe to higher timeframe.
    """
    if not _is_higher_tf_simple(target_tf, current_tf):
        return None
    
    # Resample logic would go here
    # This is a placeholder - actual implementation depends on data format
    return None


def _is_higher_tf_simple(tf1: str, tf2: str) -> bool:
    """Simple check if tf1 is higher than tf2"""
    return TIMEFRAME_ORDER.get(tf1, 0) > TIMEFRAME_ORDER.get(tf2, 0)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'TIMEFRAME_ORDER',
    'TIMEFRAME_WEIGHTS',
    'MTFConfluenceResult',
    'MultiTimeframeAnalyzerV4',
    'get_higher_timeframes',
    'get_lower_timeframes',
    'aggregate_to_higher_tf',
]