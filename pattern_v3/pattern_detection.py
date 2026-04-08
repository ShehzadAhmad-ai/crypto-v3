"""
pattern_detection.py - Advanced Pattern Detection Engine for Pattern V4

Detects ALL 23 pattern types with:
- Continuous similarity scoring (0-1) - NO binary pass/fail
- Geometry-based scoring (not percentage thresholds)
- Time decay for swing importance
- Component score breakdown for logging

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Import core classes
from .pattern_core import SwingPoint, PatternV4, PatternDirection, PatternType, PatternSimilarity
from .pattern_config import CONFIG, get_pattern_weights

# Try scipy for advanced peak detection (fallback provided if not available)
try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    def argrelextrema(data, comparator, order=1):
        """Fallback: simple peak/trough detection"""
        peaks = []
        for i in range(order, len(data) - order):
            left_ok = all(comparator(data[i], data[i-j]) for j in range(1, order+1))
            right_ok = all(comparator(data[i], data[i+j]) for j in range(1, order+1))
            if left_ok and right_ok:
                peaks.append(i)
        return np.array(peaks)


# ============================================================================
# SIMILARITY FUNCTIONS (Continuous Scoring - No Thresholds)
# ============================================================================

def price_similarity(price1: float, price2: float, scale: float = None) -> float:
    """
    Calculate price similarity (0-1) based on relative difference.
    scale = 0.05 means 5% difference = 0 similarity
    """
    if scale is None:
        scale = CONFIG.similarity_scales['price_scale']
    
    if price1 == 0 or price2 == 0:
        return 0.5
    
    diff_pct = abs(price1 - price2) / max(price1, price2)
    similarity = max(0.0, 1.0 - (diff_pct / scale))
    return min(1.0, similarity)


def time_similarity(bars1: int, bars2: int, scale: int = None) -> float:
    """
    Calculate time symmetry similarity (0-1).
    scale = 10 means 10 bar difference = 0 similarity
    """
    if scale is None:
        scale = CONFIG.similarity_scales['time_scale']
    
    diff = abs(bars1 - bars2)
    similarity = max(0.0, 1.0 - (diff / scale))
    return min(1.0, similarity)


def volume_similarity(volume1: float, volume2: float, scale: float = None) -> float:
    """
    Calculate volume pattern similarity (0-1).
    scale = 1.5 means 50% difference = 0.67 similarity
    """
    if scale is None:
        scale = CONFIG.similarity_scales['volume_scale']
    
    if volume1 == 0 or volume2 == 0:
        return 0.5
    
    ratio = min(volume1, volume2) / max(volume1, volume2)
    similarity = 1.0 - ((1.0 - ratio) / scale)
    return max(0.0, min(1.0, similarity))


def proportion_similarity(actual: float, ideal: float, tolerance: float = 0.3) -> float:
    """
    Calculate how close actual proportion is to ideal.
    tolerance = 0.3 means 30% deviation = 0 similarity
    """
    if ideal == 0:
        return 0.5
    
    deviation = abs(actual - ideal) / ideal
    similarity = max(0.0, 1.0 - (deviation / tolerance))
    return min(1.0, similarity)


def slope_quality(slope: float, max_slope: float = 0.02) -> float:
    """
    Calculate slope quality (0-1). Lower slope = higher quality.
    max_slope = 0.02 means 2% slope = 0 quality
    """
    abs_slope = abs(slope)
    quality = max(0.0, 1.0 - (abs_slope / max_slope))
    return min(1.0, quality)


def fibonacci_ratio_score(actual_ratio: float, ideal_ratio: float, tolerance: float = None) -> float:
    """
    Score how close actual Fibonacci ratio is to ideal.
    """
    if tolerance is None:
        tolerance = CONFIG.similarity_scales['fib_tolerance']
    
    if ideal_ratio == 0:
        return 0.5
    
    deviation = abs(actual_ratio - ideal_ratio) / ideal_ratio
    if deviation <= tolerance:
        return 1.0 - (deviation / tolerance) * 0.5  # 0.5 to 1.0
    else:
        return max(0.0, 0.5 - (deviation - tolerance) / tolerance)


def time_weight(bar_index: int, current_index: int, decay_rate: float = None) -> float:
    """
    Exponential time decay - newer swings have higher weight.
    """
    if not CONFIG.time_decay['enabled']:
        return 1.0
    
    if decay_rate is None:
        decay_rate = CONFIG.time_decay['decay_rate']
    
    age = current_index - bar_index
    weight = decay_rate ** age
    return max(CONFIG.time_decay['min_weight'], weight)


def structure_clarity_score(swing_types: List[str]) -> float:
    """
    Calculate structure clarity based on swing alternation.
    Perfect alternation = 1.0, random = 0.5, same type repeated = lower.
    """
    if len(swing_types) < 3:
        return 0.5
    
    alternations = 0
    for i in range(1, len(swing_types)):
        if swing_types[i] != swing_types[i-1]:
            alternations += 1
    
    alternation_rate = alternations / (len(swing_types) - 1)
    
    # Perfect alternation = 1.0, random = 0.5
    return min(1.0, alternation_rate * 1.2)


def volume_pattern_score(volumes: List[float]) -> float:
    """
    Score volume pattern: should decrease then increase (contraction then expansion).
    Returns 0-1 score.
    """
    if len(volumes) < 3:
        return 0.5
    
    first = volumes[0]
    last = volumes[-1]
    middle = volumes[len(volumes) // 2]
    
    # Contraction (middle < first)
    if first > 0:
        contraction = min(1.0, max(0.0, 1.0 - (middle / first)))
    else:
        contraction = 0.5
    
    # Expansion (last > middle)
    if middle > 0:
        expansion = min(1.0, max(0.0, (last / middle) - 0.5) * 2)
    else:
        expansion = 0.5
    
    return (contraction + expansion) / 2


# ============================================================================
# SWING DETECTION (Enhanced with Time Decay)
# ============================================================================

class AdaptiveSwingDetectorV4:
    """
    Swing detection that adapts to market volatility.
    Uses ATR to determine window size dynamically.
    Applies time decay to swings (newer swings have higher weight).
    """
    
    def __init__(self):
        self.config = CONFIG.swing_config
        self.method = self.config.get('method', 'adaptive')
        self.fixed_window = self.config.get('fixed_window', 3)
        self.atr_multiplier = self.config.get('adaptive_atr_multiplier', 0.5)
        self.min_window = self.config.get('min_window', 3)
        self.max_window = self.config.get('max_window', 10)
        self.use_volume = self.config.get('use_volume_confirmation', True)
        self.volume_threshold = self.config.get('volume_threshold', 1.2)
        self.min_strength = self.config.get('min_swing_strength', 0.3)
    
    def detect_swings(self, df: pd.DataFrame, current_index: int = None) -> List[SwingPoint]:
        """
        Detect swing highs and lows with time decay weighting.
        Returns list of SwingPoint objects with time_weight applied.
        """
        if df is None or df.empty or len(df) < self.min_window * 2:
            return []
        
        if current_index is None:
            current_index = len(df) - 1
        
        # Calculate ATR if not present
        if 'atr' not in df:
            self._add_atr(df)
        
        # Apply smoothing to reduce noise
        smoothed_high = df['high'].rolling(window=3, center=True).mean().fillna(df['high'])
        smoothed_low = df['low'].rolling(window=3, center=True).mean().fillna(df['low'])
        
        swings = []
        min_swing_distance_pct = 0.002  # 0.2% minimum swing
        
        for i in range(self.min_window, len(df) - self.min_window):
            window = self._get_window_size(df, i)
            left_start = max(0, i - window)
            right_end = min(len(df), i + window + 1)
            
            if i - left_start < window or right_end - i - 1 < window:
                continue
            
            high = smoothed_high.values
            low = smoothed_low.values
            
            is_high = high[i] == max(high[left_start:right_end])
            is_low = low[i] == min(low[left_start:right_end])
            
            if not (is_high or is_low):
                continue
            
            # Determine swing type (HH, HL, LH, LL)
            swing_type = self._determine_swing_type(swings, i, df, is_high)
            
            if swing_type is None:
                continue
            
            # Check distance from last swing
            if swings and swings[-1].type == swing_type:
                price_diff_pct = abs(df[['high', 'low']].iloc[i].max() - swings[-1].price) / swings[-1].price
                if price_diff_pct < min_swing_distance_pct:
                    continue
            
            # Calculate strength
            if is_high:
                price = float(df['high'].iloc[i])
                strength = self._calculate_swing_strength(df, i, price, is_high=True)
            else:
                price = float(df['low'].iloc[i])
                strength = self._calculate_swing_strength(df, i, price, is_high=False)
            
            if strength < self.min_strength:
                continue
            
            # Calculate volume ratio
            volume_ratio = self._get_volume_ratio(df, i)
            
            # Calculate time weight (newer swings have higher weight)
            time_w = time_weight(i, current_index)
            
            swings.append(SwingPoint(
                index=i,
                price=price,
                type=swing_type,
                timestamp=df.index[i],
                strength=strength,
                volume_ratio=volume_ratio,
                time_weight=time_w
            ))
        
        # Post-process to ensure reasonable swing count
        if len(swings) > 30:
            swings = sorted(swings, key=lambda x: x.strength * x.time_weight, reverse=True)[:30]
            swings.sort(key=lambda x: x.index)
        
        return swings
    
    def _determine_swing_type(self, swings: List[SwingPoint], idx: int, 
                               df: pd.DataFrame, is_high: bool) -> Optional[str]:
        """Determine swing type: HH, HL, LH, LL"""
        if not swings:
            return 'HH' if is_high else 'LL'
        
        last_swing = swings[-1]
        current_price = df['high'].iloc[idx] if is_high else df['low'].iloc[idx]
        last_price = last_swing.price
        
        if is_high:
            if current_price > last_price:
                return 'HH'  # Higher High
            else:
                return 'LH'  # Lower High
        else:
            if current_price < last_price:
                return 'LL'  # Lower Low
            else:
                return 'HL'  # Higher Low
    
    def _calculate_swing_strength(self, df: pd.DataFrame, idx: int, 
                                   price: float, is_high: bool) -> float:
        """Calculate swing strength based on volume and price action"""
        volume = df['volume'].values[idx]
        vol_mean = df['volume'].values[max(0, idx-20):idx].mean() if idx >= 20 else volume
        vol_strength = min(1.0, volume / max(vol_mean, 1))
        
        # Check if swing stands out from neighbors
        if idx > 5 and idx < len(df) - 5:
            if is_high:
                neighbors = df['high'].values[max(0, idx-5):idx+5]
            else:
                neighbors = df['low'].values[max(0, idx-5):idx+5]
            prominence = abs(price - np.mean(neighbors)) / max(price, 0.001)
            prominence_strength = min(1.0, prominence / 0.02)
        else:
            prominence_strength = 0.5
        
        return 0.4 * vol_strength + 0.6 * prominence_strength
    
    def _get_volume_ratio(self, df: pd.DataFrame, idx: int) -> float:
        """Get volume ratio at index"""
        volume = df['volume'].values[idx]
        vol_mean = df['volume'].values[max(0, idx-20):idx].mean() if idx >= 20 else volume
        return volume / max(vol_mean, 1)
    
    def _get_window_size(self, df: pd.DataFrame, index: int) -> int:
        """Get adaptive window size based on volatility"""
        if self.method == 'fixed':
            return self.fixed_window
        
        atr = df['atr'].values[index] if index < len(df) and 'atr' in df else df['close'].values[index] * 0.02
        current_price = df['close'].values[index]
        
        if current_price > 0:
            atr_pct = atr / current_price
            window = int(self.atr_multiplier / max(0.005, atr_pct))
            window = max(self.min_window, min(self.max_window, window))
            return window
        
        return self.fixed_window
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14):
        """Add ATR column if missing"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(df)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = np.zeros(len(df))
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(df)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        df['atr'] = atr


# ============================================================================
# BASE PATTERN DETECTOR CLASS
# ============================================================================

class PatternDetectorBaseV4:
    """Base class for all pattern detectors with similarity scoring"""
    
    def __init__(self, name: str, pattern_type: PatternType):
        self.name = name
        self.pattern_type = pattern_type
        self.weights = get_pattern_weights(name)
    
    def calculate_similarity(self, pattern_data: Dict, df: pd.DataFrame, 
                              swings: List[SwingPoint], current_index: int) -> PatternSimilarity:
        """Override in child classes - returns PatternSimilarity object"""
        raise NotImplementedError
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint], 
               current_index: int) -> Optional[Dict]:
        """
        Detect pattern with similarity scoring.
        Returns dict with similarity score and components, NOT binary pass/fail.
        """
        raise NotImplementedError
    
    def _apply_time_decay_to_swings(self, swings: List[SwingPoint], 
                                     current_index: int) -> List[SwingPoint]:
        """Apply time decay to swings list"""
        for swing in swings:
            swing.time_weight = time_weight(swing.index, current_index)
        return swings
    
    def _create_pattern_result(self, pattern_name: str, direction: str,
                                similarity: PatternSimilarity,
                                swing_points: List[SwingPoint],
                                neckline: float = None,
                                pattern_height: float = None,
                                start_idx: int = None,
                                end_idx: int = None) -> Dict:
        """Create standardized pattern result dictionary"""
        
        total_similarity = similarity.weighted_score(self.weights)
        similarity.total = total_similarity
        
        result = {
            'pattern_name': pattern_name,
            'direction': direction,
            'similarity': total_similarity,
            'components': similarity.to_dict(),
            'swing_points': [s.to_dict() for s in swing_points],
            'start_idx': start_idx or swing_points[0].index,
            'end_idx': end_idx or swing_points[-1].index,
            'neckline': neckline,
            'pattern_height': pattern_height,
        }
        
        return result


# ============================================================================
# HELPER FUNCTIONS FOR PATTERN DETECTION
# ============================================================================

def find_local_extrema(data: np.ndarray, order: int = 5, mode: str = 'max') -> np.ndarray:
    """Find local maxima or minima in array"""
    if mode == 'max':
        return argrelextrema(data, np.greater, order=order)[0]
    else:
        return argrelextrema(data, np.less, order=order)[0]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR if not present"""
    if 'atr' in df.columns:
        return df['atr']
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr


def get_support_resistance_levels(df: pd.DataFrame, swings: List[SwingPoint]) -> Tuple[List[float], List[float]]:
    """Extract support and resistance levels from swings"""
    supports = [s.price for s in swings if s.type in ['LL', 'HL']]
    resistances = [s.price for s in swings if s.type in ['HH', 'LH']]
    return supports, resistances


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Similarity functions
    'price_similarity',
    'time_similarity',
    'volume_similarity',
    'proportion_similarity',
    'slope_quality',
    'fibonacci_ratio_score',
    'time_weight',
    'structure_clarity_score',
    'volume_pattern_score',
    
    # Swing detection
    'AdaptiveSwingDetectorV4',
    
    # Base classes
    'PatternDetectorBaseV4',
    
    # Helpers
    'find_local_extrema',
    'calculate_atr',
    'get_support_resistance_levels',
]
# ============================================================================
# PART 2: STRUCTURE PATTERN DETECTORS
# ============================================================================
# These patterns detect classic reversal formations
# All use continuous similarity scoring (0-1) - NO hard-coded thresholds
# ============================================================================


# ----------------------------------------------------------------------------
# HEAD & SHOULDERS / INVERSE HEAD & SHOULDERS DETECTOR
# ----------------------------------------------------------------------------

class HeadShouldersDetectorV4(PatternDetectorBaseV4):
    """
    Detects Head & Shoulders (bearish) and Inverse Head & Shoulders (bullish)
    with GEOMETRY-BASED SIMILARITY SCORING.
    
    Components scored:
    - shoulder_symmetry: Left/right shoulder height match
    - head_prominence: Head clearly higher/lower than shoulders
    - neckline_quality: Clean, not too sloped
    - structure_clarity: Clean swing alternation
    - volume_pattern: Volume decreasing into right shoulder
    - fib_ratio: Fibonacci ratio (0.618 ideal)
    - time_symmetry: Left/right time symmetry
    """
    
    def __init__(self):
        super().__init__("Head_Shoulders", PatternType.STRUCTURE)
        self.detect_inverse = True
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint], 
               current_index: int) -> Optional[Dict]:
        """Detect both regular and inverse H&S patterns"""
        
        if len(swings) < 5:
            return None
        
        # Apply time decay to swings
        swings = self._apply_time_decay_to_swings(swings, current_index)
        
        # Look for pattern: LH, HH, LH (bearish) or HL, LL, HL (bullish)
        for i in range(len(swings) - 4):
            left = swings[i]
            head = swings[i+1]
            right = swings[i+2]
            neck1 = swings[i+3]
            neck2 = swings[i+4]
            
            # Check for Bearish H&S (high swings)
            if left.type in ['HH', 'LH'] and head.type in ['HH', 'LH'] and right.type in ['HH', 'LH']:
                if head.price > left.price and head.price > right.price:
                    result = self._calculate_bearish_hs(
                        left, head, right, neck1, neck2, df, current_index
                    )
                    if result and result['similarity'] > 0.3:  # Minimum similarity to report
                        return result
            
            # Check for Bullish H&S (low swings) - Inverse Head & Shoulders
            if left.type in ['LL', 'HL'] and head.type in ['LL', 'HL'] and right.type in ['LL', 'HL']:
                if head.price < left.price and head.price < right.price:
                    result = self._calculate_bullish_hs(
                        left, head, right, neck1, neck2, df, current_index
                    )
                    if result and result['similarity'] > 0.3:
                        return result
        
        return None
    
    def _calculate_bearish_hs(self, left: SwingPoint, head: SwingPoint, right: SwingPoint,
                               neck1: SwingPoint, neck2: SwingPoint, df: pd.DataFrame,
                               current_index: int) -> Optional[Dict]:
        """Calculate similarity for bearish Head & Shoulders"""
        
        # 1. Shoulder symmetry (how close in height)
        shoulder_symmetry = price_similarity(left.price, right.price)
        shoulder_symmetry = shoulder_symmetry * ((left.time_weight + right.time_weight) / 2)
        
        # 2. Head prominence (how much higher than shoulders)
        avg_shoulder = (left.price + right.price) / 2
        head_prominence_raw = (head.price - avg_shoulder) / avg_shoulder
        head_prominence = min(1.0, head_prominence_raw / 0.05)  # 5% prominence = full score
        
        # 3. Neckline quality (slope)
        neckline = (neck1.price + neck2.price) / 2
        neck_slope = abs(neck2.price - neck1.price) / neck1.price
        neckline_quality = slope_quality(neck_slope, 0.02)
        
        # 4. Structure clarity (swing alternation)
        swing_types = [s.type for s in [left, head, right, neck1, neck2]]
        structure_clarity = structure_clarity_score(swing_types)
        
        # 5. Volume pattern (right shoulder volume lower)
        vol_left = left.volume_ratio
        vol_right = right.volume_ratio
        volume_pattern = 1.0 - min(1.0, vol_right / max(vol_left, 0.01))
        
        # 6. Fibonacci ratio (left shoulder to head vs head to right shoulder)
        left_to_head = head.price - left.price
        head_to_right = head.price - right.price
        if left_to_head > 0:
            fib_ratio = head_to_right / left_to_head
            fib_ratio_score = fibonacci_ratio_score(fib_ratio, 0.618, 0.05)
        else:
            fib_ratio_score = 0.5
        
        # 7. Time symmetry (left vs right formation time)
        left_bars = head.index - left.index
        right_bars = right.index - head.index
        time_symmetry = time_similarity(left_bars, right_bars, 10)
        
        # Create similarity object
        similarity = PatternSimilarity(
            shoulder_symmetry=shoulder_symmetry,
            head_prominence=head_prominence,
            neckline_quality=neckline_quality,
            structure_clarity=structure_clarity,
            volume_pattern=volume_pattern,
            fib_ratio=fib_ratio_score,
            time_symmetry=time_symmetry
        )
        
        # Calculate pattern height
        pattern_height = head.price - neckline
        
        return self._create_pattern_result(
            pattern_name='Head_Shoulders',
            direction='SELL',
            similarity=similarity,
            swing_points=[left, head, right, neck1, neck2],
            neckline=neckline,
            pattern_height=pattern_height,
            start_idx=left.index,
            end_idx=right.index
        )
    
    def _calculate_bullish_hs(self, left: SwingPoint, head: SwingPoint, right: SwingPoint,
                               neck1: SwingPoint, neck2: SwingPoint, df: pd.DataFrame,
                               current_index: int) -> Optional[Dict]:
        """Calculate similarity for bullish Inverse Head & Shoulders"""
        
        # 1. Shoulder symmetry (how close in height)
        shoulder_symmetry = price_similarity(left.price, right.price)
        shoulder_symmetry = shoulder_symmetry * ((left.time_weight + right.time_weight) / 2)
        
        # 2. Head prominence (how much lower than shoulders)
        avg_shoulder = (left.price + right.price) / 2
        head_prominence_raw = (avg_shoulder - head.price) / avg_shoulder
        head_prominence = min(1.0, head_prominence_raw / 0.05)
        
        # 3. Neckline quality (slope)
        neckline = (neck1.price + neck2.price) / 2
        neck_slope = abs(neck2.price - neck1.price) / neck1.price
        neckline_quality = slope_quality(neck_slope, 0.02)
        
        # 4. Structure clarity
        swing_types = [s.type for s in [left, head, right, neck1, neck2]]
        structure_clarity = structure_clarity_score(swing_types)
        
        # 5. Volume pattern (right shoulder volume lower)
        vol_left = left.volume_ratio
        vol_right = right.volume_ratio
        volume_pattern = 1.0 - min(1.0, vol_right / max(vol_left, 0.01))
        
        # 6. Fibonacci ratio
        left_to_head = left.price - head.price
        head_to_right = right.price - head.price
        if left_to_head > 0:
            fib_ratio = head_to_right / left_to_head
            fib_ratio_score = fibonacci_ratio_score(fib_ratio, 0.618, 0.05)
        else:
            fib_ratio_score = 0.5
        
        # 7. Time symmetry
        left_bars = head.index - left.index
        right_bars = right.index - head.index
        time_symmetry = time_similarity(left_bars, right_bars, 10)
        
        similarity = PatternSimilarity(
            shoulder_symmetry=shoulder_symmetry,
            head_prominence=head_prominence,
            neckline_quality=neckline_quality,
            structure_clarity=structure_clarity,
            volume_pattern=volume_pattern,
            fib_ratio=fib_ratio_score,
            time_symmetry=time_symmetry
        )
        
        pattern_height = neckline - head.price
        
        return self._create_pattern_result(
            pattern_name='Inverse_Head_Shoulders',
            direction='BUY',
            similarity=similarity,
            swing_points=[left, head, right, neck1, neck2],
            neckline=neckline,
            pattern_height=pattern_height,
            start_idx=left.index,
            end_idx=right.index
        )


# ----------------------------------------------------------------------------
# DOUBLE TOP / DOUBLE BOTTOM DETECTOR
# ----------------------------------------------------------------------------

class DoubleTopBottomDetectorV4(PatternDetectorBaseV4):
    """
    Detects Double Top (bearish) and Double Bottom (bullish)
    with GEOMETRY-BASED SIMILARITY SCORING.
    
    Components scored:
    - price_similarity: Tops/bottoms at similar level
    - valley_strength: Pullback depth significance
    - volume_pattern: Lower volume on second top/bottom
    - time_symmetry: Equal time between tops/bottoms
    - breakout_volume: Volume on neckline break
    """
    
    def __init__(self):
        super().__init__("Double_Top", PatternType.STRUCTURE)
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Double Top and Double Bottom patterns"""
        
        if len(swings) < 3:
            return None
        
        swings = self._apply_time_decay_to_swings(swings, current_index)
        
        # Separate highs and lows
        highs = [s for s in swings if s.type in ['HH', 'LH']]
        lows = [s for s in swings if s.type in ['LL', 'HL']]
        
        # Detect Double Top (bearish)
        if len(highs) >= 2:
            result = self._detect_double_top(highs, df, current_index)
            if result and result['similarity'] > 0.3:
                return result
        
        # Detect Double Bottom (bullish)
        if len(lows) >= 2:
            result = self._detect_double_bottom(lows, df, current_index)
            if result and result['similarity'] > 0.3:
                return result
        
        return None
    
    def _detect_double_top(self, highs: List[SwingPoint], df: pd.DataFrame,
                            current_index: int) -> Optional[Dict]:
        """Detect Double Top pattern"""
        
        # Check last two highs
        top1 = highs[-2]
        top2 = highs[-1]
        
        # 1. Price similarity between tops
        price_sim = price_similarity(top1.price, top2.price)
        price_sim = price_sim * ((top1.time_weight + top2.time_weight) / 2)
        
        # 2. Valley strength (how significant is the pullback)
        valley_idx = range(top1.index, top2.index)
        valley_price = min(df['low'].iloc[valley_idx])
        avg_top = (top1.price + top2.price) / 2
        valley_depth = (avg_top - valley_price) / avg_top
        valley_strength = min(1.0, valley_depth / 0.03)  # 3% depth = full strength
        
        # 3. Volume pattern (lower volume on second top)
        vol_pattern = volume_similarity(top2.volume_ratio, top1.volume_ratio, 0.5)
        vol_pattern = 1.0 - vol_pattern  # Lower volume is better
        
        # 4. Time symmetry
        time_sym = time_similarity(
            top2.index - top1.index,
            top2.index - top1.index,  # Ideal is equal spacing
            10
        )
        
        # 5. Breakout volume (check after pattern)
        neckline = valley_price
        if top2.index + 3 < len(df):
            breakout_vol = df['volume'].iloc[top2.index:top2.index+3].mean()
            avg_vol = df['volume'].iloc[max(0, top2.index-20):top2.index].mean()
            breakout_volume = min(1.0, breakout_vol / max(avg_vol, 1) / 1.5)
        else:
            breakout_volume = 0.5
        
        similarity = PatternSimilarity(
            price_similarity=price_sim,
            valley_strength=valley_strength,
            volume_pattern=vol_pattern,
            time_symmetry=time_sym,
            breakout_volume=breakout_volume
        )
        
        pattern_height = avg_top - neckline
        
        return self._create_pattern_result(
            pattern_name='Double_Top',
            direction='SELL',
            similarity=similarity,
            swing_points=[top1, top2],
            neckline=neckline,
            pattern_height=pattern_height,
            start_idx=top1.index,
            end_idx=top2.index
        )
    
    def _detect_double_bottom(self, lows: List[SwingPoint], df: pd.DataFrame,
                               current_index: int) -> Optional[Dict]:
        """Detect Double Bottom pattern"""
        
        bottom1 = lows[-2]
        bottom2 = lows[-1]
        
        # 1. Price similarity between bottoms
        price_sim = price_similarity(bottom1.price, bottom2.price)
        price_sim = price_sim * ((bottom1.time_weight + bottom2.time_weight) / 2)
        
        # 2. Peak strength (how significant is the rally)
        peak_idx = range(bottom1.index, bottom2.index)
        peak_price = max(df['high'].iloc[peak_idx])
        avg_bottom = (bottom1.price + bottom2.price) / 2
        peak_height = (peak_price - avg_bottom) / avg_bottom
        peak_strength = min(1.0, peak_height / 0.03)
        
        # 3. Volume pattern (lower volume on second bottom)
        vol_pattern = volume_similarity(bottom2.volume_ratio, bottom1.volume_ratio, 0.5)
        vol_pattern = 1.0 - vol_pattern
        
        # 4. Time symmetry
        time_sym = time_similarity(
            bottom2.index - bottom1.index,
            bottom2.index - bottom1.index,
            10
        )
        
        # 5. Breakout volume
        resistance = peak_price
        if bottom2.index + 3 < len(df):
            breakout_vol = df['volume'].iloc[bottom2.index:bottom2.index+3].mean()
            avg_vol = df['volume'].iloc[max(0, bottom2.index-20):bottom2.index].mean()
            breakout_volume = min(1.0, breakout_vol / max(avg_vol, 1) / 1.5)
        else:
            breakout_volume = 0.5
        
        similarity = PatternSimilarity(
            price_similarity=price_sim,
            valley_strength=peak_strength,
            volume_pattern=vol_pattern,
            time_symmetry=time_sym,
            breakout_volume=breakout_volume
        )
        
        pattern_height = resistance - avg_bottom
        
        return self._create_pattern_result(
            pattern_name='Double_Bottom',
            direction='BUY',
            similarity=similarity,
            swing_points=[bottom1, bottom2],
            neckline=resistance,
            pattern_height=pattern_height,
            start_idx=bottom1.index,
            end_idx=bottom2.index
        )


# ----------------------------------------------------------------------------
# TRIPLE TOP / TRIPLE BOTTOM DETECTOR
# ----------------------------------------------------------------------------

class TripleTopBottomDetectorV4(PatternDetectorBaseV4):
    """
    Detects Triple Top (bearish) and Triple Bottom (bullish)
    with GEOMETRY-BASED SIMILARITY SCORING.
    
    Components scored:
    - price_similarity: All tops/bottoms at similar levels
    - valley_strength: Pullback depth significance
    - volume_pattern: Decreasing volume on each subsequent top/bottom
    - time_symmetry: Even spacing between tops/bottoms
    - structure_clarity: Clean swing alternation
    """
    
    def __init__(self):
        super().__init__("Triple_Top", PatternType.STRUCTURE)
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Triple Top and Triple Bottom patterns"""
        
        if len(swings) < 5:
            return None
        
        swings = self._apply_time_decay_to_swings(swings, current_index)
        
        highs = [s for s in swings if s.type in ['HH', 'LH']]
        lows = [s for s in swings if s.type in ['LL', 'HL']]
        
        # Detect Triple Top (need at least 3 highs)
        if len(highs) >= 3:
            result = self._detect_triple_top(highs, df, current_index)
            if result and result['similarity'] > 0.3:
                return result
        
        # Detect Triple Bottom (need at least 3 lows)
        if len(lows) >= 3:
            result = self._detect_triple_bottom(lows, df, current_index)
            if result and result['similarity'] > 0.3:
                return result
        
        return None
    
    def _detect_triple_top(self, highs: List[SwingPoint], df: pd.DataFrame,
                            current_index: int) -> Optional[Dict]:
        """Detect Triple Top pattern"""
        
        top1 = highs[-3]
        top2 = highs[-2]
        top3 = highs[-1]
        
        # 1. Price similarity across all three tops
        avg_price = (top1.price + top2.price + top3.price) / 3
        price_deviations = [
            abs(top1.price - avg_price) / avg_price,
            abs(top2.price - avg_price) / avg_price,
            abs(top3.price - avg_price) / avg_price
        ]
        price_sim = 1.0 - min(1.0, sum(price_deviations) / 0.05)
        
        # Apply time decay weighting
        time_weighted_sim = price_sim * ((top1.time_weight + top2.time_weight + top3.time_weight) / 3)
        
        # 2. Valley strength (average pullback depth)
        valley1 = min(df['low'].iloc[top1.index:top2.index])
        valley2 = min(df['low'].iloc[top2.index:top3.index])
        avg_top = (top1.price + top2.price + top3.price) / 3
        avg_valley = (valley1 + valley2) / 2
        valley_depth = (avg_top - avg_valley) / avg_top
        valley_strength = min(1.0, valley_depth / 0.03)
        
        # 3. Volume pattern (volume should decrease on each top)
        vol_ratio1 = top2.volume_ratio / max(top1.volume_ratio, 0.01)
        vol_ratio2 = top3.volume_ratio / max(top2.volume_ratio, 0.01)
        vol_pattern = (1.0 - min(1.0, vol_ratio1) + 1.0 - min(1.0, vol_ratio2)) / 2
        
        # 4. Time symmetry (even spacing)
        bars12 = top2.index - top1.index
        bars23 = top3.index - top2.index
        time_sym = time_similarity(bars12, bars23, 10)
        
        # 5. Structure clarity
        swing_types = [s.type for s in [top1, top2, top3]]
        structure_clarity = structure_clarity_score(swing_types)
        
        similarity = PatternSimilarity(
            price_similarity=time_weighted_sim,
            valley_strength=valley_strength,
            volume_pattern=vol_pattern,
            time_symmetry=time_sym,
            structure_clarity=structure_clarity
        )
        
        neckline = avg_valley
        pattern_height = avg_top - neckline
        
        return self._create_pattern_result(
            pattern_name='Triple_Top',
            direction='SELL',
            similarity=similarity,
            swing_points=[top1, top2, top3],
            neckline=neckline,
            pattern_height=pattern_height,
            start_idx=top1.index,
            end_idx=top3.index
        )
    
    def _detect_triple_bottom(self, lows: List[SwingPoint], df: pd.DataFrame,
                               current_index: int) -> Optional[Dict]:
        """Detect Triple Bottom pattern"""
        
        bottom1 = lows[-3]
        bottom2 = lows[-2]
        bottom3 = lows[-1]
        
        # 1. Price similarity
        avg_price = (bottom1.price + bottom2.price + bottom3.price) / 3
        price_deviations = [
            abs(bottom1.price - avg_price) / avg_price,
            abs(bottom2.price - avg_price) / avg_price,
            abs(bottom3.price - avg_price) / avg_price
        ]
        price_sim = 1.0 - min(1.0, sum(price_deviations) / 0.05)
        time_weighted_sim = price_sim * ((bottom1.time_weight + bottom2.time_weight + bottom3.time_weight) / 3)
        
        # 2. Peak strength
        peak1 = max(df['high'].iloc[bottom1.index:bottom2.index])
        peak2 = max(df['high'].iloc[bottom2.index:bottom3.index])
        avg_bottom = (bottom1.price + bottom2.price + bottom3.price) / 3
        avg_peak = (peak1 + peak2) / 2
        peak_height = (avg_peak - avg_bottom) / avg_bottom
        peak_strength = min(1.0, peak_height / 0.03)
        
        # 3. Volume pattern
        vol_ratio1 = bottom2.volume_ratio / max(bottom1.volume_ratio, 0.01)
        vol_ratio2 = bottom3.volume_ratio / max(bottom2.volume_ratio, 0.01)
        vol_pattern = (1.0 - min(1.0, vol_ratio1) + 1.0 - min(1.0, vol_ratio2)) / 2
        
        # 4. Time symmetry
        bars12 = bottom2.index - bottom1.index
        bars23 = bottom3.index - bottom2.index
        time_sym = time_similarity(bars12, bars23, 10)
        
        # 5. Structure clarity
        swing_types = [s.type for s in [bottom1, bottom2, bottom3]]
        structure_clarity = structure_clarity_score(swing_types)
        
        similarity = PatternSimilarity(
            price_similarity=time_weighted_sim,
            valley_strength=peak_strength,
            volume_pattern=vol_pattern,
            time_symmetry=time_sym,
            structure_clarity=structure_clarity
        )
        
        resistance = avg_peak
        pattern_height = resistance - avg_bottom
        
        return self._create_pattern_result(
            pattern_name='Triple_Bottom',
            direction='BUY',
            similarity=similarity,
            swing_points=[bottom1, bottom2, bottom3],
            neckline=resistance,
            pattern_height=pattern_height,
            start_idx=bottom1.index,
            end_idx=bottom3.index
        )


# ----------------------------------------------------------------------------
# EXPORTS FOR PART 2
# ----------------------------------------------------------------------------

__all__ = [
    'HeadShouldersDetectorV4',
    'DoubleTopBottomDetectorV4',
    'TripleTopBottomDetectorV4',
]# ============================================================================
# PART 3: CONTINUATION PATTERN DETECTORS
# ============================================================================
# These patterns detect trend continuation formations
# All use continuous similarity scoring (0-1) - NO hard-coded thresholds
# ============================================================================


# ----------------------------------------------------------------------------
# FLAG & PENNANT DETECTOR
# ----------------------------------------------------------------------------

class FlagPennantDetectorV4(PatternDetectorBaseV4):
    """
    Detects Flag and Pennant patterns (trend continuation)
    
    Components scored:
    - pole_strength: Sharp move magnitude (stronger = better)
    - tightness: Consolidation range tightness (tighter = better)
    - volume_decrease: Volume contraction during flag/pennant
    - breakout_volume: Volume expansion on breakout
    - convergence: For pennants - lines converging
    """
    
    def __init__(self):
        super().__init__("Flag", PatternType.STRUCTURE)
        self.pole_min_bars = 5
        self.flag_max_bars = 30
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Flag and Pennant patterns"""
        
        if len(df) < 30:
            return None
        
        # Look for sharp move (flagpole) followed by consolidation
        for i in range(self.pole_min_bars, len(df) - 15):
            # Check flagpole (sharp move)
            start_price = df['close'].iloc[i - self.pole_min_bars]
            end_price = df['close'].iloc[i]
            move_pct = abs(end_price - start_price) / start_price
            
            # Pole strength (higher move = stronger)
            pole_strength = min(1.0, move_pct / 0.05)  # 5% move = full strength
            
            if pole_strength < 0.4:
                continue
            
            direction = 'BULLISH' if end_price > start_price else 'BEARISH'
            
            # Check consolidation (flag/pennant)
            consolidation = df.iloc[i:i+15]
            if len(consolidation) < 5:
                continue
            
            flag_high = consolidation['high'].max()
            flag_low = consolidation['low'].min()
            range_pct = (flag_high - flag_low) / start_price
            
            # Flag tightness (tighter = better)
            tightness = max(0.0, 1.0 - (range_pct / 0.03))  # 3% range = 0
            
            # Check if it's a pennant (converging lines)
            x = np.arange(len(consolidation))
            highs = consolidation['high'].values
            lows = consolidation['low'].values
            
            high_slope, high_intercept = np.polyfit(x, highs, 1)
            low_slope, low_intercept = np.polyfit(x, lows, 1)
            
            is_pennant = (high_slope < 0 and low_slope > 0)  # Lines converging
            is_flag = abs(high_slope - low_slope) < 0.0001  # Parallel lines
            
            if not (is_flag or is_pennant):
                continue
            
            # Volume pattern (volume should decrease during consolidation)
            vol_start = consolidation['volume'].iloc[:3].mean()
            vol_end = consolidation['volume'].iloc[-3:].mean()
            volume_decrease = max(0.0, 1.0 - (vol_end / max(vol_start, 0.01)))
            
            # Breakout volume (check after consolidation)
            end_idx = i + len(consolidation)
            if end_idx + 3 < len(df):
                breakout_vol = df['volume'].iloc[end_idx:end_idx+3].mean()
                avg_vol = df['volume'].iloc[max(0, end_idx-20):end_idx].mean()
                breakout_volume = min(1.0, (breakout_vol / max(avg_vol, 1) - 1.0) / 1.0)
                breakout_volume = max(0.0, breakout_volume)
            else:
                breakout_volume = 0.5
            
            # Convergence score for pennant
            convergence = 0.5
            if is_pennant:
                start_width = high_intercept - low_intercept
                end_width = (high_slope * len(x) + high_intercept) - (low_slope * len(x) + low_intercept)
                if start_width > 0:
                    convergence_ratio = end_width / start_width
                    convergence = min(1.0, 1.0 - convergence_ratio)
            
            pattern_name = "Pennant" if is_pennant else "Flag"
            
            similarity = PatternSimilarity(
                pole_strength=pole_strength,
                tightness=tightness,
                volume_pattern=volume_decrease,
                breakout_volume=breakout_volume,
                convergence=convergence
            )
            
            # Calculate target (flagpole projection)
            flagpole_height = abs(end_price - start_price)
            if direction == 'BULLISH':
                target = flag_high + flagpole_height
            else:
                target = flag_low - flagpole_height
            
            result = self._create_pattern_result(
                pattern_name=pattern_name,
                direction='BUY' if direction == 'BULLISH' else 'SELL',
                similarity=similarity,
                swing_points=[],
                neckline=flag_high if direction == 'BULLISH' else flag_low,
                pattern_height=flagpole_height,
                start_idx=i - self.pole_min_bars,
                end_idx=end_idx
            )
            
            # Add extra fields
            result['flagpole_start'] = start_price
            result['flagpole_end'] = end_price
            result['flag_high'] = flag_high
            result['flag_low'] = flag_low
            result['target'] = target
            
            if similarity.total > 0.4:
                return result
        
        return None


# ----------------------------------------------------------------------------
# WEDGE DETECTORS (Rising & Falling)
# ----------------------------------------------------------------------------

class RisingWedgeDetectorV4(PatternDetectorBaseV4):
    """
    Detects Rising Wedge pattern (bearish reversal)
    
    Components scored:
    - convergence: Lines converging
    - slope_quality: Clean slopes
    - volume_pattern: Volume decreasing as wedge progresses
    - breakout_position: Breakout at 2/3-3/4 point
    """
    
    def __init__(self):
        super().__init__("Rising_Wedge", PatternType.STRUCTURE)
        self.min_bars = 15
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Rising Wedge pattern"""
        
        if len(df) < self.min_bars:
            return None
        
        recent = df.iloc[-self.min_bars:]
        x = np.arange(len(recent))
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Fit trendlines
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Rising Wedge: high slope negative (down), low slope positive (up)
        if high_slope < 0 and low_slope > 0:
            # 1. Convergence (how much they narrow)
            start_width = high_intercept - low_intercept
            end_width = (high_slope * len(x) + high_intercept) - (low_slope * len(x) + low_intercept)
            
            if start_width <= 0:
                return None
            
            convergence_ratio = end_width / start_width
            convergence = min(1.0, 1.0 - convergence_ratio)
            
            # 2. Slope quality (clean slopes)
            high_slope_quality = slope_quality(abs(high_slope), 0.01)
            low_slope_quality = slope_quality(abs(low_slope), 0.01)
            slope_quality_score = (high_slope_quality + low_slope_quality) / 2
            
            # 3. Volume pattern (volume should decrease)
            vol_start = recent['volume'].iloc[:5].mean()
            vol_end = recent['volume'].iloc[-5:].mean()
            volume_pattern = max(0.0, 1.0 - (vol_end / max(vol_start, 0.01)))
            
            # 4. Breakout position (should be at 2/3-3/4)
            current_bar = len(recent) - 1
            breakout_position = current_bar / len(recent)
            position_score = 1.0 - min(1.0, abs(breakout_position - 0.7) / 0.3)
            
            similarity = PatternSimilarity(
                convergence=convergence,
                slope_quality=slope_quality_score,
                volume_pattern=volume_pattern,
                breakout_volume=position_score
            )
            
            lower_trendline = low_slope * len(x) + low_intercept
            
            return self._create_pattern_result(
                pattern_name='Rising_Wedge',
                direction='SELL',
                similarity=similarity,
                swing_points=[],
                neckline=lower_trendline,
                pattern_height=high_intercept - low_intercept,
                start_idx=len(df) - self.min_bars,
                end_idx=len(df) - 1
            )
        
        return None


class FallingWedgeDetectorV4(PatternDetectorBaseV4):
    """
    Detects Falling Wedge pattern (bullish reversal)
    
    Components scored:
    - convergence: Lines converging
    - slope_quality: Clean slopes
    - volume_pattern: Volume decreasing as wedge progresses
    - breakout_position: Breakout at 2/3-3/4 point
    """
    
    def __init__(self):
        super().__init__("Falling_Wedge", PatternType.STRUCTURE)
        self.min_bars = 15
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Falling Wedge pattern"""
        
        if len(df) < self.min_bars:
            return None
        
        recent = df.iloc[-self.min_bars:]
        x = np.arange(len(recent))
        highs = recent['high'].values
        lows = recent['low'].values
        
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Falling Wedge: high slope positive (up), low slope negative (down)
        if high_slope > 0 and low_slope < 0:
            start_width = high_intercept - low_intercept
            end_width = (high_slope * len(x) + high_intercept) - (low_slope * len(x) + low_intercept)
            
            if start_width <= 0:
                return None
            
            convergence_ratio = end_width / start_width
            convergence = min(1.0, 1.0 - convergence_ratio)
            
            high_slope_quality = slope_quality(abs(high_slope), 0.01)
            low_slope_quality = slope_quality(abs(low_slope), 0.01)
            slope_quality_score = (high_slope_quality + low_slope_quality) / 2
            
            vol_start = recent['volume'].iloc[:5].mean()
            vol_end = recent['volume'].iloc[-5:].mean()
            volume_pattern = max(0.0, 1.0 - (vol_end / max(vol_start, 0.01)))
            
            current_bar = len(recent) - 1
            breakout_position = current_bar / len(recent)
            position_score = 1.0 - min(1.0, abs(breakout_position - 0.7) / 0.3)
            
            similarity = PatternSimilarity(
                convergence=convergence,
                slope_quality=slope_quality_score,
                volume_pattern=volume_pattern,
                breakout_volume=position_score
            )
            
            upper_trendline = high_slope * len(x) + high_intercept
            
            return self._create_pattern_result(
                pattern_name='Falling_Wedge',
                direction='BUY',
                similarity=similarity,
                swing_points=[],
                neckline=upper_trendline,
                pattern_height=high_intercept - low_intercept,
                start_idx=len(df) - self.min_bars,
                end_idx=len(df) - 1
            )
        
        return None


# ----------------------------------------------------------------------------
# TRIANGLE DETECTORS (Ascending, Descending, Symmetrical)
# ----------------------------------------------------------------------------

class AscendingTriangleDetectorV4(PatternDetectorBaseV4):
    """
    Detects Ascending Triangle pattern (bullish)
    
    Components scored:
    - flat_resistance: Horizontal top line
    - rising_support: Rising bottom line
    - volume_contraction: Volume decreases during formation
    - breakout_volume: Volume spike on breakout
    """
    
    def __init__(self):
        super().__init__("Ascending_Triangle", PatternType.STRUCTURE)
        self.min_bars = 15
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Ascending Triangle pattern"""
        
        if len(df) < self.min_bars:
            return None
        
        recent = df.iloc[-self.min_bars:]
        x = np.arange(len(recent))
        highs = recent['high'].values
        lows = recent['low'].values
        
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Ascending: flat highs (|slope| < 0.01), rising lows (slope > 0.01)
        if abs(high_slope) < 0.01 and low_slope > 0.01:
            # 1. Flat resistance quality
            flat_resistance = 1.0 - min(1.0, abs(high_slope) / 0.01)
            
            # 2. Rising support quality
            rising_support = min(1.0, low_slope / 0.02)
            
            # 3. Volume contraction
            vol_start = recent['volume'].iloc[:5].mean()
            vol_end = recent['volume'].iloc[-5:].mean()
            volume_contraction = max(0.0, 1.0 - (vol_end / max(vol_start, 0.01)))
            
            # 4. Breakout volume
            if len(recent) > 0:
                breakout_vol = recent['volume'].iloc[-1]
                avg_vol = recent['volume'].iloc[:-5].mean() if len(recent) > 5 else vol_start
                breakout_volume = min(1.0, (breakout_vol / max(avg_vol, 1) - 1.0) / 1.0)
                breakout_volume = max(0.0, breakout_volume)
            else:
                breakout_volume = 0.5
            
            similarity = PatternSimilarity(
                flat_resistance=flat_resistance,
                rising_support=rising_support,
                volume_pattern=volume_contraction,
                breakout_volume=breakout_volume
            )
            
            resistance_level = high_intercept
            
            return self._create_pattern_result(
                pattern_name='Ascending_Triangle',
                direction='BUY',
                similarity=similarity,
                swing_points=[],
                neckline=resistance_level,
                pattern_height=high_intercept - low_intercept,
                start_idx=len(df) - self.min_bars,
                end_idx=len(df) - 1
            )
        
        return None


class DescendingTriangleDetectorV4(PatternDetectorBaseV4):
    """
    Detects Descending Triangle pattern (bearish)
    
    Components scored:
    - flat_support: Horizontal bottom line
    - falling_resistance: Falling top line
    - volume_contraction: Volume decreases during formation
    - breakout_volume: Volume spike on breakout
    """
    
    def __init__(self):
        super().__init__("Descending_Triangle", PatternType.STRUCTURE)
        self.min_bars = 15
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Descending Triangle pattern"""
        
        if len(df) < self.min_bars:
            return None
        
        recent = df.iloc[-self.min_bars:]
        x = np.arange(len(recent))
        highs = recent['high'].values
        lows = recent['low'].values
        
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Descending: falling highs (slope < -0.01), flat lows (|slope| < 0.01)
        if high_slope < -0.01 and abs(low_slope) < 0.01:
            flat_support = 1.0 - min(1.0, abs(low_slope) / 0.01)
            falling_resistance = min(1.0, abs(high_slope) / 0.02)
            
            vol_start = recent['volume'].iloc[:5].mean()
            vol_end = recent['volume'].iloc[-5:].mean()
            volume_contraction = max(0.0, 1.0 - (vol_end / max(vol_start, 0.01)))
            
            if len(recent) > 0:
                breakout_vol = recent['volume'].iloc[-1]
                avg_vol = recent['volume'].iloc[:-5].mean() if len(recent) > 5 else vol_start
                breakout_volume = min(1.0, (breakout_vol / max(avg_vol, 1) - 1.0) / 1.0)
                breakout_volume = max(0.0, breakout_volume)
            else:
                breakout_volume = 0.5
            
            similarity = PatternSimilarity(
                flat_support=flat_support,
                falling_resistance=falling_resistance,
                volume_pattern=volume_contraction,
                breakout_volume=breakout_volume
            )
            
            support_level = low_intercept
            
            return self._create_pattern_result(
                pattern_name='Descending_Triangle',
                direction='SELL',
                similarity=similarity,
                swing_points=[],
                neckline=support_level,
                pattern_height=high_intercept - low_intercept,
                start_idx=len(df) - self.min_bars,
                end_idx=len(df) - 1
            )
        
        return None


class SymmetricalTriangleDetectorV4(PatternDetectorBaseV4):
    """
    Detects Symmetrical Triangle pattern (neutral - direction determined by momentum)
    
    Components scored:
    - convergence: Lines converging
    - symmetry: Equal slope magnitudes
    - volume_contraction: Volume decreases during formation
    - breakout_position: Breakout at 2/3-3/4 point
    """
    
    def __init__(self):
        super().__init__("Symmetrical_Triangle", PatternType.STRUCTURE)
        self.min_bars = 15
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Symmetrical Triangle pattern"""
        
        if len(df) < self.min_bars:
            return None
        
        recent = df.iloc[-self.min_bars:]
        x = np.arange(len(recent))
        highs = recent['high'].values
        lows = recent['low'].values
        
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Symmetrical: converging (high slope negative, low slope positive)
        if high_slope < 0 and low_slope > 0:
            start_width = high_intercept - low_intercept
            end_width = (high_slope * len(x) + high_intercept) - (low_slope * len(x) + low_intercept)
            
            if start_width <= 0:
                return None
            
            convergence_ratio = end_width / start_width
            convergence = min(1.0, 1.0 - convergence_ratio)
            
            # Symmetry (slopes should be roughly equal magnitude)
            slope_magnitude_diff = abs(abs(high_slope) - abs(low_slope))
            symmetry = max(0.0, 1.0 - (slope_magnitude_diff / 0.01))
            
            vol_start = recent['volume'].iloc[:5].mean()
            vol_end = recent['volume'].iloc[-5:].mean()
            volume_contraction = max(0.0, 1.0 - (vol_end / max(vol_start, 0.01)))
            
            current_bar = len(recent) - 1
            breakout_position = current_bar / len(recent)
            position_score = 1.0 - min(1.0, abs(breakout_position - 0.7) / 0.3)
            
            similarity = PatternSimilarity(
                convergence=convergence,
                symmetry=symmetry,
                volume_pattern=volume_contraction,
                breakout_volume=position_score
            )
            
            # Determine breakout direction based on momentum
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
            direction = 'BUY' if momentum > 0.005 else 'SELL' if momentum < -0.005 else 'NEUTRAL'
            
            if direction == 'NEUTRAL':
                return None
            
            return self._create_pattern_result(
                pattern_name='Symmetrical_Triangle',
                direction=direction,
                similarity=similarity,
                swing_points=[],
                neckline=high_intercept if direction == 'BUY' else low_intercept,
                pattern_height=start_width,
                start_idx=len(df) - self.min_bars,
                end_idx=len(df) - 1
            )
        
        return None


# ----------------------------------------------------------------------------
# CUP & HANDLE DETECTOR
# ----------------------------------------------------------------------------

class CupHandleDetectorV4(PatternDetectorBaseV4):
    """
    Detects Cup & Handle pattern (bullish continuation)
    
    Components scored:
    - cup_depth: 5-20% depth ideal
    - cup_symmetry: Left/right rim symmetry
    - handle_depth: Handle 20-40% of cup
    - volume_pattern: Volume decreases in handle
    """
    
    def __init__(self):
        super().__init__("Cup_Handle", PatternType.STRUCTURE)
        self.cup_lookback = 40
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Cup & Handle pattern"""
        
        if len(df) < self.cup_lookback:
            return None
        
        recent = df.iloc[-self.cup_lookback:]
        
        # Find the lowest point in the cup
        cup_bottom_idx = recent['low'].idxmin()
        cup_bottom_pos = recent.index.get_loc(cup_bottom_idx)
        
        if cup_bottom_pos < 5 or cup_bottom_pos > self.cup_lookback - 10:
            return None
        
        # Get left and right rims
        left_rim = recent['high'].iloc[:cup_bottom_pos].max()
        right_rim = recent['high'].iloc[cup_bottom_pos:].max()
        
        # 1. Cup symmetry (rims should be similar)
        cup_symmetry = price_similarity(left_rim, right_rim)
        
        # 2. Cup depth (5-20% is ideal)
        cup_bottom = recent['low'].iloc[cup_bottom_pos]
        avg_rim = (left_rim + right_rim) / 2
        cup_depth_pct = (avg_rim - cup_bottom) / avg_rim
        cup_depth = 1.0 - min(1.0, abs(cup_depth_pct - 0.10) / 0.15)  # 10% ideal
        
        # Look for handle (small pullback after cup)
        handle_section = recent.iloc[cup_bottom_pos + 5:]
        if len(handle_section) < 5:
            return None
        
        handle_high = handle_section['high'].max()
        handle_low = handle_section['low'].min()
        handle_depth_pct = (handle_high - handle_low) / handle_high
        handle_depth = 1.0 - min(1.0, abs(handle_depth_pct - 0.30) / 0.20)  # 30% ideal
        
        # Volume pattern (volume should decrease in handle)
        vol_cup = recent['volume'].iloc[:cup_bottom_pos].mean()
        vol_handle = handle_section['volume'].mean()
        volume_pattern = max(0.0, 1.0 - (vol_handle / max(vol_cup, 0.01)))
        
        similarity = PatternSimilarity(
            cup_symmetry=cup_symmetry,
            cup_depth=cup_depth,
            handle_depth=handle_depth,
            volume_pattern=volume_pattern
        )
        
        return self._create_pattern_result(
            pattern_name='Cup_Handle',
            direction='BUY',
            similarity=similarity,
            swing_points=[],
            neckline=handle_high,
            pattern_height=avg_rim - cup_bottom,
            start_idx=recent.index[0],
            end_idx=recent.index[-1]
        )


# ----------------------------------------------------------------------------
# ADAM & EVE DETECTOR
# ----------------------------------------------------------------------------

class AdamEveDetectorV4(PatternDetectorBaseV4):
    """
    Detects Adam & Eve Double Bottom pattern
    
    Components scored:
    - price_similarity: Bottoms at similar level
    - shape_quality: Adam sharp, Eve rounded
    - peak_strength: Peak between bottoms
    """
    
    def __init__(self):
        super().__init__("Adam_Eve", PatternType.STRUCTURE)
        self.min_distance = 10
        self.max_distance = 40
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Adam & Eve pattern"""
        
        if len(df) < self.max_distance:
            return None
        
        # Find potential bottoms
        lows = df['low'].values
        bottom_indices = find_local_extrema(lows, order=5, mode='min')
        
        if len(bottom_indices) < 2:
            return None
        
        for i in range(len(bottom_indices) - 1):
            idx1 = bottom_indices[i]
            idx2 = bottom_indices[i+1]
            
            if idx2 - idx1 < self.min_distance or idx2 - idx1 > self.max_distance:
                continue
            
            bottom1 = lows[idx1]
            bottom2 = lows[idx2]
            
            # Price similarity
            price_sim = price_similarity(bottom1, bottom2)
            
            # Shape quality: is first bottom sharp (Adam) and second rounded (Eve)?
            adam_sharp = self._is_sharp_bottom(df, idx1)
            eve_rounded = self._is_rounded_bottom(df, idx2)
            
            if not (adam_sharp and eve_rounded):
                # Try opposite order
                adam_sharp = self._is_sharp_bottom(df, idx2)
                eve_rounded = self._is_rounded_bottom(df, idx1)
            
            if adam_sharp and eve_rounded:
                shape_quality = 0.8 + (0.2 if adam_sharp and eve_rounded else 0)
                
                # Peak between bottoms
                peak = max(df['high'].iloc[idx1:idx2])
                avg_bottom = (bottom1 + bottom2) / 2
                peak_strength = min(1.0, (peak - avg_bottom) / avg_bottom / 0.03)
                
                similarity = PatternSimilarity(
                    price_similarity=price_sim,
                    shape_quality=shape_quality,
                    peak_strength=peak_strength
                )
                
                return self._create_pattern_result(
                    pattern_name='Adam_Eve',
                    direction='BUY',
                    similarity=similarity,
                    swing_points=[],
                    neckline=peak,
                    pattern_height=peak - avg_bottom,
                    start_idx=idx1,
                    end_idx=idx2
                )
        
        return None
    
    def _is_sharp_bottom(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if bottom is sharp (V-shaped)"""
        if idx < 3 or idx > len(df) - 4:
            return False
        
        left_min = df['low'].iloc[idx-3:idx].min()
        right_min = df['low'].iloc[idx+1:idx+4].min()
        bottom = df['low'].iloc[idx]
        
        return (bottom <= left_min * 0.995) and (bottom <= right_min * 0.995)
    
    def _is_rounded_bottom(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if bottom is rounded (U-shaped)"""
        if idx < 7 or idx > len(df) - 8:
            return False
        
        left_slope = (df['low'].iloc[idx-7] - df['low'].iloc[idx]) / 7
        right_slope = (df['low'].iloc[idx+7] - df['low'].iloc[idx]) / 7
        
        return (abs(left_slope) < 0.001 and abs(right_slope) < 0.001)


# ----------------------------------------------------------------------------
# QUASIMODO DETECTOR
# ----------------------------------------------------------------------------

class QuasimodoDetectorV4(PatternDetectorBaseV4):
    """
    Detects Quasimodo pattern (Over and Under)
    
    Components scored:
    - structure_quality: Higher high → lower low pattern
    - reversal_strength: Clear reversal
    - volume_confirmation: Volume supports reversal
    """
    
    def __init__(self):
        super().__init__("Quasimodo", PatternType.STRUCTURE)
        self.min_swings = 4
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Quasimodo pattern"""
        
        if len(swings) < self.min_swings:
            return None
        
        swings = self._apply_time_decay_to_swings(swings, current_index)
        
        for i in range(len(swings) - 3):
            p1 = swings[i]
            p2 = swings[i+1]
            p3 = swings[i+2]
            p4 = swings[i+3]
            
            # Bullish Quasimodo: Higher high → Lower low → Reversal
            if (p1.type in ['HH', 'LH'] and p2.type in ['LL', 'HL'] and
                p3.type in ['HH', 'LH'] and p4.type in ['LL', 'HL']):
                
                if p3.price > p1.price and p4.price < p2.price:
                    structure_quality = 0.7
                    reversal_strength = min(1.0, (p3.price - p4.price) / p4.price / 0.03)
                    
                    # Check current price above structure
                    current_price = df['close'].iloc[-1]
                    if current_price > p3.price:
                        similarity = PatternSimilarity(
                            structure_clarity=structure_quality,
                            reversal_strength=reversal_strength
                        )
                        
                        return self._create_pattern_result(
                            pattern_name='Quasimodo_Bullish',
                            direction='BUY',
                            similarity=similarity,
                            swing_points=[p1, p2, p3, p4],
                            neckline=p3.price,
                            pattern_height=p3.price - p4.price,
                            start_idx=p1.index,
                            end_idx=p4.index
                        )
            
            # Bearish Quasimodo: Lower low → Higher high → Reversal
            elif (p1.type in ['LL', 'HL'] and p2.type in ['HH', 'LH'] and
                  p3.type in ['LL', 'HL'] and p4.type in ['HH', 'LH']):
                
                if p3.price < p1.price and p4.price > p2.price:
                    structure_quality = 0.7
                    reversal_strength = min(1.0, (p4.price - p3.price) / p3.price / 0.03)
                    
                    current_price = df['close'].iloc[-1]
                    if current_price < p4.price:
                        similarity = PatternSimilarity(
                            structure_clarity=structure_quality,
                            reversal_strength=reversal_strength
                        )
                        
                        return self._create_pattern_result(
                            pattern_name='Quasimodo_Bearish',
                            direction='SELL',
                            similarity=similarity,
                            swing_points=[p1, p2, p3, p4],
                            neckline=p4.price,
                            pattern_height=p4.price - p3.price,
                            start_idx=p1.index,
                            end_idx=p4.index
                        )
        
        return None


# ----------------------------------------------------------------------------
# EXPORTS FOR PART 3
# ----------------------------------------------------------------------------

__all__ = [
    'FlagPennantDetectorV4',
    'RisingWedgeDetectorV4',
    'FallingWedgeDetectorV4',
    'AscendingTriangleDetectorV4',
    'DescendingTriangleDetectorV4',
    'SymmetricalTriangleDetectorV4',
    'CupHandleDetectorV4',
    'AdamEveDetectorV4',
    'QuasimodoDetectorV4',
]# ============================================================================
# PART 4: HARMONIC, VOLUME & WAVE PATTERN DETECTORS
# ============================================================================
# These patterns detect harmonic formations, VCP, Wolfe Wave, and Divergence
# All use continuous similarity scoring (0-1) - NO hard-coded thresholds
# ============================================================================


# ----------------------------------------------------------------------------
# HARMONIC PATTERN BASE CLASS
# ----------------------------------------------------------------------------

class HarmonicPatternBaseV4(PatternDetectorBaseV4):
    """
    Base class for harmonic patterns (XABCD structure)
    Provides Fibonacci ratio calculation and validation
    """
    
    def __init__(self, name: str):
        super().__init__(name, PatternType.HARMONIC)
        self.fib_tolerance = CONFIG.similarity_scales['fib_tolerance']
    
    def _find_xabcd(self, swings: List[SwingPoint]) -> List[Dict]:
        """Find XABCD pattern from swings"""
        patterns = []
        
        if len(swings) < 5:
            return patterns
        
        for i in range(len(swings) - 4):
            x = swings[i]
            a = swings[i+1]
            b = swings[i+2]
            c = swings[i+3]
            d = swings[i+4]
            
            # Determine pattern direction based on swing types
            is_bullish = (x.type in ['LL', 'HL'] and a.type in ['HH', 'LH'] and 
                          b.type in ['LL', 'HL'] and c.type in ['HH', 'LH'] and 
                          d.type in ['LL', 'HL'])
            
            is_bearish = (x.type in ['HH', 'LH'] and a.type in ['LL', 'HL'] and 
                          b.type in ['HH', 'LH'] and c.type in ['LL', 'HL'] and 
                          d.type in ['HH', 'LH'])
            
            if not (is_bullish or is_bearish):
                continue
            
            # Calculate Fibonacci ratios
            x_price = x.price
            a_price = a.price
            b_price = b.price
            c_price = c.price
            d_price = d.price
            
            xa_move = abs(a_price - x_price)
            ab_move = abs(b_price - a_price)
            bc_move = abs(c_price - b_price)
            
            if xa_move > 0:
                ab_retrace = ab_move / xa_move
            else:
                ab_retrace = 0
            
            if ab_move > 0:
                bc_retrace = bc_move / ab_move
            else:
                bc_retrace = 0
            
            if xa_move > 0:
                xd_extension = abs(d_price - x_price) / xa_move
            else:
                xd_extension = 0
            
            patterns.append({
                'x': x, 'a': a, 'b': b, 'c': c, 'd': d,
                'ab_retrace': ab_retrace,
                'bc_retrace': bc_retrace,
                'xd_extension': xd_extension,
                'is_bullish': is_bullish,
                'start_idx': x.index,
                'end_idx': d.index
            })
        
        return patterns
    
    def _calculate_fib_accuracy(self, actual: float, ideal: float) -> float:
        """Calculate Fibonacci ratio accuracy score"""
        if ideal == 0:
            return 0.5
        deviation = abs(actual - ideal) / ideal
        if deviation <= self.fib_tolerance:
            return 1.0 - (deviation / self.fib_tolerance) * 0.5
        else:
            return max(0.0, 0.5 - (deviation - self.fib_tolerance) / self.fib_tolerance)


# ----------------------------------------------------------------------------
# GARTLEY DETECTOR
# ----------------------------------------------------------------------------

class GartleyDetectorV4(HarmonicPatternBaseV4):
    """
    Gartley Pattern (harmonic)
    - AB retracement: 0.618 of XA
    - BC retracement: 0.382-0.886 of AB
    - XD extension: 0.786 of XA
    """
    
    def __init__(self):
        super().__init__("Gartley")
        self.ideal_ab = 0.618
        self.ideal_xd = 0.786
        self.bc_min = 0.382
        self.bc_max = 0.886
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Gartley pattern"""
        
        patterns = self._find_xabcd(swings)
        
        for p in patterns:
            # Calculate component scores
            ab_accuracy = self._calculate_fib_accuracy(p['ab_retrace'], self.ideal_ab)
            xd_accuracy = self._calculate_fib_accuracy(p['xd_extension'], self.ideal_xd)
            
            bc_in_range = self.bc_min - self.fib_tolerance <= p['bc_retrace'] <= self.bc_max + self.fib_tolerance
            bc_score = 1.0 if bc_in_range else max(0.0, 1.0 - abs(p['bc_retrace'] - 0.618) / 0.3)
            
            # Structure clarity
            swing_types = [s.type for s in [p['x'], p['a'], p['b'], p['c'], p['d']]]
            structure_clarity = structure_clarity_score(swing_types)
            
            # Overall fib accuracy
            fib_accuracy = (ab_accuracy + xd_accuracy + bc_score) / 3
            
            similarity = PatternSimilarity(
                ab_retrace=ab_accuracy,
                bc_retrace=bc_score,
                xd_extension=xd_accuracy,
                fib_accuracy=fib_accuracy,
                structure_clarity=structure_clarity
            )
            
            direction = 'BUY' if p['is_bullish'] else 'SELL'
            
            return self._create_pattern_result(
                pattern_name='Gartley',
                direction=direction,
                similarity=similarity,
                swing_points=[p['x'], p['a'], p['b'], p['c'], p['d']],
                neckline=p['d'].price,
                pattern_height=abs(p['d'].price - p['x'].price),
                start_idx=p['start_idx'],
                end_idx=p['end_idx']
            )
        
        return None


# ----------------------------------------------------------------------------
# BAT DETECTOR
# ----------------------------------------------------------------------------

class BatDetectorV4(HarmonicPatternBaseV4):
    """
    Bat Pattern (harmonic)
    - AB retracement: 0.382-0.5 of XA
    - BC retracement: 0.382-0.886 of AB
    - XD extension: 0.886 of XA
    """
    
    def __init__(self):
        super().__init__("Bat")
        self.ab_min = 0.382
        self.ab_max = 0.5
        self.ideal_xd = 0.886
        self.bc_min = 0.382
        self.bc_max = 0.886
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Bat pattern"""
        
        patterns = self._find_xabcd(swings)
        
        for p in patterns:
            # AB in range
            ab_in_range = self.ab_min - self.fib_tolerance <= p['ab_retrace'] <= self.ab_max + self.fib_tolerance
            ab_score = 1.0 if ab_in_range else max(0.0, 1.0 - abs(p['ab_retrace'] - 0.44) / 0.2)
            
            xd_accuracy = self._calculate_fib_accuracy(p['xd_extension'], self.ideal_xd)
            
            bc_in_range = self.bc_min - self.fib_tolerance <= p['bc_retrace'] <= self.bc_max + self.fib_tolerance
            bc_score = 1.0 if bc_in_range else max(0.0, 1.0 - abs(p['bc_retrace'] - 0.618) / 0.3)
            
            swing_types = [s.type for s in [p['x'], p['a'], p['b'], p['c'], p['d']]]
            structure_clarity = structure_clarity_score(swing_types)
            
            fib_accuracy = (ab_score + xd_accuracy + bc_score) / 3
            
            similarity = PatternSimilarity(
                ab_retrace=ab_score,
                bc_retrace=bc_score,
                xd_extension=xd_accuracy,
                fib_accuracy=fib_accuracy,
                structure_clarity=structure_clarity
            )
            
            direction = 'BUY' if p['is_bullish'] else 'SELL'
            
            return self._create_pattern_result(
                pattern_name='Bat',
                direction=direction,
                similarity=similarity,
                swing_points=[p['x'], p['a'], p['b'], p['c'], p['d']],
                neckline=p['d'].price,
                pattern_height=abs(p['d'].price - p['x'].price),
                start_idx=p['start_idx'],
                end_idx=p['end_idx']
            )
        
        return None


# ----------------------------------------------------------------------------
# CRAB DETECTOR
# ----------------------------------------------------------------------------

class CrabDetectorV4(HarmonicPatternBaseV4):
    """
    Crab Pattern (harmonic - high probability)
    - AB retracement: 0.382-0.618 of XA
    - BC retracement: 0.382-0.886 of AB
    - XD extension: 1.618 of XA
    """
    
    def __init__(self):
        super().__init__("Crab")
        self.ab_min = 0.382
        self.ab_max = 0.618
        self.ideal_xd = 1.618
        self.bc_min = 0.382
        self.bc_max = 0.886
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Crab pattern"""
        
        patterns = self._find_xabcd(swings)
        
        for p in patterns:
            ab_in_range = self.ab_min - self.fib_tolerance <= p['ab_retrace'] <= self.ab_max + self.fib_tolerance
            ab_score = 1.0 if ab_in_range else max(0.0, 1.0 - abs(p['ab_retrace'] - 0.5) / 0.25)
            
            xd_accuracy = self._calculate_fib_accuracy(p['xd_extension'], self.ideal_xd)
            
            bc_in_range = self.bc_min - self.fib_tolerance <= p['bc_retrace'] <= self.bc_max + self.fib_tolerance
            bc_score = 1.0 if bc_in_range else max(0.0, 1.0 - abs(p['bc_retrace'] - 0.618) / 0.3)
            
            swing_types = [s.type for s in [p['x'], p['a'], p['b'], p['c'], p['d']]]
            structure_clarity = structure_clarity_score(swing_types)
            
            fib_accuracy = (ab_score + xd_accuracy + bc_score) / 3
            
            similarity = PatternSimilarity(
                ab_retrace=ab_score,
                bc_retrace=bc_score,
                xd_extension=xd_accuracy,
                fib_accuracy=fib_accuracy,
                structure_clarity=structure_clarity
            )
            
            direction = 'BUY' if p['is_bullish'] else 'SELL'
            
            return self._create_pattern_result(
                pattern_name='Crab',
                direction=direction,
                similarity=similarity,
                swing_points=[p['x'], p['a'], p['b'], p['c'], p['d']],
                neckline=p['d'].price,
                pattern_height=abs(p['d'].price - p['x'].price),
                start_idx=p['start_idx'],
                end_idx=p['end_idx']
            )
        
        return None


# ----------------------------------------------------------------------------
# BUTTERFLY DETECTOR
# ----------------------------------------------------------------------------

class ButterflyDetectorV4(HarmonicPatternBaseV4):
    """
    Butterfly Pattern (harmonic)
    - AB retracement: 0.786 of XA
    - BC retracement: 0.382-0.886 of AB
    - XD extension: 1.27 of XA
    """
    
    def __init__(self):
        super().__init__("Butterfly")
        self.ideal_ab = 0.786
        self.ideal_xd = 1.27
        self.bc_min = 0.382
        self.bc_max = 0.886
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Butterfly pattern"""
        
        patterns = self._find_xabcd(swings)
        
        for p in patterns:
            ab_accuracy = self._calculate_fib_accuracy(p['ab_retrace'], self.ideal_ab)
            xd_accuracy = self._calculate_fib_accuracy(p['xd_extension'], self.ideal_xd)
            
            bc_in_range = self.bc_min - self.fib_tolerance <= p['bc_retrace'] <= self.bc_max + self.fib_tolerance
            bc_score = 1.0 if bc_in_range else max(0.0, 1.0 - abs(p['bc_retrace'] - 0.618) / 0.3)
            
            swing_types = [s.type for s in [p['x'], p['a'], p['b'], p['c'], p['d']]]
            structure_clarity = structure_clarity_score(swing_types)
            
            fib_accuracy = (ab_accuracy + xd_accuracy + bc_score) / 3
            
            similarity = PatternSimilarity(
                ab_retrace=ab_accuracy,
                bc_retrace=bc_score,
                xd_extension=xd_accuracy,
                fib_accuracy=fib_accuracy,
                structure_clarity=structure_clarity
            )
            
            direction = 'BUY' if p['is_bullish'] else 'SELL'
            
            return self._create_pattern_result(
                pattern_name='Butterfly',
                direction=direction,
                similarity=similarity,
                swing_points=[p['x'], p['a'], p['b'], p['c'], p['d']],
                neckline=p['d'].price,
                pattern_height=abs(p['d'].price - p['x'].price),
                start_idx=p['start_idx'],
                end_idx=p['end_idx']
            )
        
        return None


# ----------------------------------------------------------------------------
# THREE DRIVES DETECTOR
# ----------------------------------------------------------------------------

class ThreeDrivesDetectorV4(HarmonicPatternBaseV4):
    """
    Three Drives Pattern (harmonic reversal)
    - Three successive drives with weakening momentum
    - Each drive should be smaller than previous
    - RSI divergence confirmation
    """
    
    def __init__(self):
        super().__init__("Three_Drives")
        self.drive_ratio_min = 0.5
        self.drive_ratio_max = 0.9
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Three Drives pattern"""
        
        if len(swings) < 6:
            return None
        
        for i in range(len(swings) - 5):
            points = swings[i:i+6]
            
            # Check alternating pattern
            types = [p.type for p in points]
            
            is_bullish = (types[0] in ['LL', 'HL'] and types[1] in ['HH', 'LH'] and
                          types[2] in ['LL', 'HL'] and types[3] in ['HH', 'LH'] and
                          types[4] in ['LL', 'HL'] and types[5] in ['HH', 'LH'])
            
            is_bearish = (types[0] in ['HH', 'LH'] and types[1] in ['LL', 'HL'] and
                          types[2] in ['HH', 'LH'] and types[3] in ['LL', 'HL'] and
                          types[4] in ['HH', 'LH'] and types[5] in ['LL', 'HL'])
            
            if not (is_bullish or is_bearish):
                continue
            
            if is_bullish:
                drive1 = points[1].price - points[0].price
                drive2 = points[3].price - points[2].price
                drive3 = points[5].price - points[4].price
                
                if drive2 >= drive1 or drive3 >= drive2:
                    continue
                
                ratio1 = drive2 / drive1 if drive1 > 0 else 0
                ratio2 = drive3 / drive2 if drive2 > 0 else 0
            else:
                drive1 = points[0].price - points[1].price
                drive2 = points[2].price - points[3].price
                drive3 = points[4].price - points[5].price
                
                if drive2 >= drive1 or drive3 >= drive2:
                    continue
                
                ratio1 = drive2 / drive1 if drive1 > 0 else 0
                ratio2 = drive3 / drive2 if drive2 > 0 else 0
            
            # Drive ratio scores
            ratio1_score = 1.0 - min(1.0, abs(ratio1 - 0.7) / 0.3)
            ratio2_score = 1.0 - min(1.0, abs(ratio2 - 0.7) / 0.3)
            drive_ratio = (ratio1_score + ratio2_score) / 2
            
            # Alternation quality
            alternation = structure_clarity_score(types)
            
            # RSI divergence check
            rsi_divergence = self._check_rsi_divergence(points, df, is_bullish)
            
            similarity = PatternSimilarity(
                drive_ratio=drive_ratio,
                alternation=alternation,
                rsi_divergence=rsi_divergence,
                structure_clarity=alternation
            )
            
            direction = 'BUY' if is_bullish else 'SELL'
            
            return self._create_pattern_result(
                pattern_name='Three_Drives',
                direction=direction,
                similarity=similarity,
                swing_points=points,
                neckline=points[-1].price,
                pattern_height=drive3,
                start_idx=points[0].index,
                end_idx=points[-1].index
            )
        
        return None
    
    def _check_rsi_divergence(self, points: List[SwingPoint], df: pd.DataFrame, 
                               is_bullish: bool) -> float:
        """Check RSI divergence strength"""
        if 'rsi' not in df.columns:
            return 0.5
        
        try:
            rsi_values = [df['rsi'].iloc[p.index] for p in points]
            
            if is_bullish:
                # Price higher lows, RSI lower highs = bullish divergence
                price_higher_lows = points[2].price > points[0].price and points[4].price > points[2].price
                rsi_lower_highs = rsi_values[1] < rsi_values[3] and rsi_values[3] < rsi_values[5]
                
                if price_higher_lows and rsi_lower_highs:
                    return 0.9
                elif price_higher_lows or rsi_lower_highs:
                    return 0.6
                return 0.4
            else:
                # Price lower highs, RSI higher lows = bearish divergence
                price_lower_highs = points[2].price < points[0].price and points[4].price < points[2].price
                rsi_higher_lows = rsi_values[1] > rsi_values[3] and rsi_values[3] > rsi_values[5]
                
                if price_lower_highs and rsi_higher_lows:
                    return 0.9
                elif price_lower_highs or rsi_higher_lows:
                    return 0.6
                return 0.4
        except:
            return 0.5


# ----------------------------------------------------------------------------
# VCP DETECTOR (Volatility Contraction Pattern)
# ----------------------------------------------------------------------------

class VCPDetectorV4(PatternDetectorBaseV4):
    """
    VCP (Volatility Contraction Pattern) - Mark Minervini
    - Multiple pullbacks with decreasing volatility
    - Volume contraction during pullbacks
    - Breakout after final contraction
    """
    
    def __init__(self):
        super().__init__("VCP", PatternType.VOLUME)
        self.min_contractions = 2
        self.max_contractions = 5
        self.volatility_threshold = 0.7
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect VCP pattern"""
        
        if len(df) < 50:
            return None
        
        # Determine trend direction
        trend = self._determine_trend(df)
        if trend == 'NEUTRAL':
            return None
        
        if trend == 'BULLISH':
            return self._find_bullish_vcp(df, swings)
        else:
            return self._find_bearish_vcp(df, swings)
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        if len(df) < 50:
            return 'NEUTRAL'
        
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price > ema50 * 1.02:
            return 'BULLISH'
        elif current_price < ema50 * 0.98:
            return 'BEARISH'
        return 'NEUTRAL'
    
    def _find_bullish_vcp(self, df: pd.DataFrame, swings: List[SwingPoint]) -> Optional[Dict]:
        """Find bullish VCP (higher lows with decreasing volatility)"""
        
        lows = [s for s in swings if s.type in ['LL', 'HL']]
        if len(lows) < self.min_contractions + 1:
            return None
        
        for i in range(len(lows) - self.min_contractions):
            volatilities = []
            contraction_ratios = []
            
            for j in range(self.min_contractions + 1):
                if j > 0 and lows[i+j].price <= lows[i+j-1].price:
                    break
                
                # Calculate volatility from swing
                swing_idx = lows[i+j].index
                if swing_idx > 0 and swing_idx < len(df):
                    if 'atr' in df.columns:
                        vol = float(df['atr'].iloc[swing_idx]) / float(df['close'].iloc[swing_idx])
                    else:
                        recent_range = df['high'].iloc[max(0, swing_idx-10):swing_idx].max() - df['low'].iloc[max(0, swing_idx-10):swing_idx].min()
                        vol = recent_range / float(df['close'].iloc[swing_idx]) if df['close'].iloc[swing_idx] > 0 else 0.02
                    volatilities.append(min(0.5, vol))
            
            if len(volatilities) >= 2:
                for k in range(1, len(volatilities)):
                    contraction_ratios.append(volatilities[k] / volatilities[k-1])
                
                avg_contraction = np.mean(contraction_ratios)
                contraction_count_score = min(1.0, len(contraction_ratios) / 4)
                volatility_reduction = max(0.0, 1.0 - avg_contraction)
                
                similarity = PatternSimilarity(
                    contraction_count=contraction_count_score,
                    volatility_reduction=volatility_reduction,
                    volume_pattern=0.5  # Would need volume data
                )
                
                if avg_contraction < self.volatility_threshold:
                    return self._create_pattern_result(
                        pattern_name='VCP',
                        direction='BUY',
                        similarity=similarity,
                        swing_points=lows[i:i+self.min_contractions+1],
                        neckline=max(df['high'].iloc[lows[i].index:lows[i+self.min_contractions].index]),
                        pattern_height=0,
                        start_idx=lows[i].index,
                        end_idx=lows[i+self.min_contractions].index
                    )
        
        return None
    
    def _find_bearish_vcp(self, df: pd.DataFrame, swings: List[SwingPoint]) -> Optional[Dict]:
        """Find bearish VCP (lower highs with decreasing volatility)"""
        
        highs = [s for s in swings if s.type in ['HH', 'LH']]
        if len(highs) < self.min_contractions + 1:
            return None
        
        for i in range(len(highs) - self.min_contractions):
            volatilities = []
            contraction_ratios = []
            
            for j in range(self.min_contractions + 1):
                if j > 0 and highs[i+j].price >= highs[i+j-1].price:
                    break
                
                swing_idx = highs[i+j].index
                if swing_idx > 0 and swing_idx < len(df):
                    if 'atr' in df.columns:
                        vol = float(df['atr'].iloc[swing_idx]) / float(df['close'].iloc[swing_idx])
                    else:
                        recent_range = df['high'].iloc[max(0, swing_idx-10):swing_idx].max() - df['low'].iloc[max(0, swing_idx-10):swing_idx].min()
                        vol = recent_range / float(df['close'].iloc[swing_idx]) if df['close'].iloc[swing_idx] > 0 else 0.02
                    volatilities.append(min(0.5, vol))
            
            if len(volatilities) >= 2:
                for k in range(1, len(volatilities)):
                    contraction_ratios.append(volatilities[k] / volatilities[k-1])
                
                avg_contraction = np.mean(contraction_ratios)
                contraction_count_score = min(1.0, len(contraction_ratios) / 4)
                volatility_reduction = max(0.0, 1.0 - avg_contraction)
                
                similarity = PatternSimilarity(
                    contraction_count=contraction_count_score,
                    volatility_reduction=volatility_reduction,
                    volume_pattern=0.5
                )
                
                if avg_contraction < self.volatility_threshold:
                    return self._create_pattern_result(
                        pattern_name='VCP',
                        direction='SELL',
                        similarity=similarity,
                        swing_points=highs[i:i+self.min_contractions+1],
                        neckline=min(df['low'].iloc[highs[i].index:highs[i+self.min_contractions].index]),
                        pattern_height=0,
                        start_idx=highs[i].index,
                        end_idx=highs[i+self.min_contractions].index
                    )
        
        return None


# ----------------------------------------------------------------------------
# WOLFE WAVE DETECTOR
# ----------------------------------------------------------------------------

class WolfeWaveDetectorV4(PatternDetectorBaseV4):
    """
    Wolfe Wave Pattern (5-wave predictive structure)
    - Points: 1-2-3-4-5, with point 5 being entry
    - Target along line from point 1 to 4
    """
    
    def __init__(self):
        super().__init__("Wolfe_Wave", PatternType.WAVE)
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect Wolfe Wave pattern"""
        
        if len(swings) < 5:
            return None
        
        for i in range(len(swings) - 4):
            points = swings[i:i+5]
            types = [p.type for p in points]
            
            # Bullish: LOW, HIGH, LOW, HIGH, LOW
            is_bullish = (types[0] in ['LL', 'HL'] and types[1] in ['HH', 'LH'] and
                          types[2] in ['LL', 'HL'] and types[3] in ['HH', 'LH'] and
                          types[4] in ['LL', 'HL'])
            
            # Bearish: HIGH, LOW, HIGH, LOW, HIGH
            is_bearish = (types[0] in ['HH', 'LH'] and types[1] in ['LL', 'HL'] and
                          types[2] in ['HH', 'LH'] and types[3] in ['LL', 'HL'] and
                          types[4] in ['HH', 'LH'])
            
            if not (is_bullish or is_bearish):
                continue
            
            # Wave alternation quality
            wave_alternation = structure_clarity_score(types)
            
            # Wave symmetry (time and price relationships)
            if is_bullish:
                wave1 = points[1].price - points[0].price
                wave2 = points[2].price - points[1].price
                wave3 = points[3].price - points[2].price
                wave4 = points[4].price - points[3].price
            else:
                wave1 = points[0].price - points[1].price
                wave2 = points[1].price - points[2].price
                wave3 = points[2].price - points[3].price
                wave4 = points[3].price - points[4].price
            
            # Check wave relationships (waves should alternate)
            wave_symmetry = 0.5
            if wave1 > 0 and wave3 > 0:
                wave_ratio = min(wave1, wave3) / max(wave1, wave3)
                wave_symmetry = (wave_symmetry + wave_ratio) / 2
            
            # Calculate target (line from point 1 to 4)
            p1 = points[0]
            p4 = points[3]
            slope = (p4.price - p1.price) / (p4.index - p1.index) if p4.index != p1.index else 0
            target_price = p4.price + slope * (current_index - p4.index)
            
            similarity = PatternSimilarity(
                wave_alternation=wave_alternation,
                wave_symmetry=wave_symmetry,
                target_alignment=0.5
            )
            
            direction = 'BUY' if is_bullish else 'SELL'
            
            result = self._create_pattern_result(
                pattern_name='Wolfe_Wave',
                direction=direction,
                similarity=similarity,
                swing_points=points,
                neckline=points[-1].price,
                pattern_height=abs(target_price - points[-1].price),
                start_idx=points[0].index,
                end_idx=points[-1].index
            )
            result['target'] = target_price
            
            if similarity.total > 0.4:
                return result
        
        return None


# ----------------------------------------------------------------------------
# DIVERGENCE DETECTOR
# ----------------------------------------------------------------------------

class DivergenceDetectorV4(PatternDetectorBaseV4):
    """
    Detects RSI divergence patterns (bullish and bearish)
    
    Components scored:
    - price_swing_magnitude: Size of price swing
    - divergence_strength: RSI divergence magnitude
    - volume_confirmation: Volume supports reversal
    """
    
    def __init__(self):
        super().__init__("Divergence", PatternType.DIVERGENCE)
    
    def detect(self, df: pd.DataFrame, swings: List[SwingPoint],
               current_index: int) -> Optional[Dict]:
        """Detect RSI divergence patterns"""
        
        if 'rsi' not in df.columns or len(swings) < 2:
            return None
        
        # Find swing points in price and RSI
        price_highs = [s for s in swings if s.type in ['HH', 'LH']]
        price_lows = [s for s in swings if s.type in ['LL', 'HL']]
        
        # Bullish Divergence (price lower low, RSI higher low)
        for i in range(len(price_lows) - 1):
            low1 = price_lows[i]
            low2 = price_lows[i+1]
            
            if low2.price < low1.price:  # Lower low in price
                rsi1 = df['rsi'].iloc[low1.index]
                rsi2 = df['rsi'].iloc[low2.index]
                
                if rsi2 > rsi1:  # Higher low in RSI
                    price_swing = (low1.price - low2.price) / low1.price
                    price_swing_magnitude = min(1.0, price_swing / 0.03)
                    
                    rsi_divergence = (rsi2 - rsi1) / 100
                    divergence_strength = min(1.0, rsi_divergence / 0.05)
                    
                    similarity = PatternSimilarity(
                        price_swing_magnitude=price_swing_magnitude,
                        divergence_strength=divergence_strength,
                        volume_pattern=0.5
                    )
                    
                    return self._create_pattern_result(
                        pattern_name='Bullish_Divergence',
                        direction='BUY',
                        similarity=similarity,
                        swing_points=[low1, low2],
                        neckline=low2.price,
                        pattern_height=abs(low1.price - low2.price),
                        start_idx=low1.index,
                        end_idx=low2.index
                    )
        
        # Bearish Divergence (price higher high, RSI lower high)
        for i in range(len(price_highs) - 1):
            high1 = price_highs[i]
            high2 = price_highs[i+1]
            
            if high2.price > high1.price:  # Higher high in price
                rsi1 = df['rsi'].iloc[high1.index]
                rsi2 = df['rsi'].iloc[high2.index]
                
                if rsi2 < rsi1:  # Lower high in RSI
                    price_swing = (high2.price - high1.price) / high1.price
                    price_swing_magnitude = min(1.0, price_swing / 0.03)
                    
                    rsi_divergence = (rsi1 - rsi2) / 100
                    divergence_strength = min(1.0, rsi_divergence / 0.05)
                    
                    similarity = PatternSimilarity(
                        price_swing_magnitude=price_swing_magnitude,
                        divergence_strength=divergence_strength,
                        volume_pattern=0.5
                    )
                    
                    return self._create_pattern_result(
                        pattern_name='Bearish_Divergence',
                        direction='SELL',
                        similarity=similarity,
                        swing_points=[high1, high2],
                        neckline=high2.price,
                        pattern_height=abs(high2.price - high1.price),
                        start_idx=high1.index,
                        end_idx=high2.index
                    )
        
        return None


# ----------------------------------------------------------------------------
# MAIN PATTERN DETECTION ENGINE
# ----------------------------------------------------------------------------

class PatternDetectionEngineV4:
    """
    Main pattern detection engine that coordinates all detectors.
    Detects ALL 23 pattern types with continuous similarity scoring.
    """
    
    def __init__(self):
        self.swing_detector = AdaptiveSwingDetectorV4()
        
        # Initialize ALL 23 pattern detectors
        self.detectors = [
            # Structure Patterns (12)
            HeadShouldersDetectorV4(),
            DoubleTopBottomDetectorV4(),
            TripleTopBottomDetectorV4(),
            CupHandleDetectorV4(),
            AdamEveDetectorV4(),
            QuasimodoDetectorV4(),
            
            # Continuation Patterns (8)
            FlagPennantDetectorV4(),
            RisingWedgeDetectorV4(),
            FallingWedgeDetectorV4(),
            AscendingTriangleDetectorV4(),
            DescendingTriangleDetectorV4(),
            SymmetricalTriangleDetectorV4(),
            
            # Harmonic Patterns (5)
            GartleyDetectorV4(),
            BatDetectorV4(),
            CrabDetectorV4(),
            ButterflyDetectorV4(),
            ThreeDrivesDetectorV4(),
            
            # Volume & Wave Patterns (2)
            VCPDetectorV4(),
            WolfeWaveDetectorV4(),
            
            # Divergence (2 directions - handled in one detector)
            DivergenceDetectorV4(),
        ]
        
        self.max_patterns = CONFIG.performance_config.get('max_patterns_per_symbol', 20)
    
    def detect_all_patterns(self, df: pd.DataFrame, symbol: str = "",
                            regime_data: Dict = None) -> Tuple[List[Dict], List[SwingPoint]]:
        """
        Detect all patterns with similarity scoring.
        Returns list of pattern results and swings.
        """
        patterns = []
        
        if df is None or df.empty or len(df) < 30:
            return patterns, []
        
        # Add indicators if missing
        if 'rsi' not in df.columns:
            self._add_rsi(df)
        
        if 'atr' not in df.columns:
            self._add_atr(df)
        
        # Detect swings
        current_index = len(df) - 1
        swings = self.swing_detector.detect_swings(df, current_index)
        
        if len(swings) < 5:
            return patterns, swings
        
        # Run each detector
        for detector in self.detectors:
            try:
                result = detector.detect(df, swings, current_index)
                if result and result.get('similarity', 0) > 0.3:  # Minimum similarity to report
                    result['symbol'] = symbol
                    patterns.append(result)
            except Exception as e:
                continue
        
        # Sort by similarity (highest first) and limit
        patterns.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return patterns[:self.max_patterns], swings
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14):
        """Add RSI indicator"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14):
        """Add ATR indicator"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(df)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = np.zeros(len(df))
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(df)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        df['atr'] = atr


# ----------------------------------------------------------------------------
# EXPORTS FOR PART 4
# ----------------------------------------------------------------------------

__all__ = [
    'HarmonicPatternBaseV4',
    'GartleyDetectorV4',
    'BatDetectorV4',
    'CrabDetectorV4',
    'ButterflyDetectorV4',
    'ThreeDrivesDetectorV4',
    'VCPDetectorV4',
    'WolfeWaveDetectorV4',
    'DivergenceDetectorV4',
    'PatternDetectionEngineV4',
]