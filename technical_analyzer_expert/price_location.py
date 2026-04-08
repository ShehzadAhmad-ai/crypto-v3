# price_location.py - Enhanced with Premium/Discount Zones
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def detect_support_resistance(df: pd.DataFrame, window: int = 20, sensitivity: float = 0.004) -> Dict[str, List[float]]:
    """
    Detect candidate S/R by local extrema over sliding window.
    Enhanced with vectorized operations.
    """
    res = {
        'supports': [], 
        'resistances': [], 
        'range_high': None, 
        'range_low': None, 
        'equilibrium': None,
        'premium_discount_zones': {}
    }
    
    try:
        if df is None or df.empty or len(df) < window * 2:
            return res
        
        current = df['close'].iloc[-1]
        
        # Vectorized swing detection
        highs = df['high'].values
        lows = df['low'].values
        
        supports = []
        resistances = []
        
        for i in range(window, len(highs) - window):
            # Local minima (support)
            if lows[i] == np.min(lows[i-window:i+window+1]):
                if lows[i] < current * (1 - sensitivity):
                    if not any(abs(lows[i] - s) < lows[i] * sensitivity for s in supports):
                        supports.append(float(lows[i]))
            
            # Local maxima (resistance)
            if highs[i] == np.max(highs[i-window:i+window+1]):
                if highs[i] > current * (1 + sensitivity):
                    if not any(abs(highs[i] - r) < highs[i] * sensitivity for r in resistances):
                        resistances.append(float(highs[i]))
        
        # Sort and take recent ones
        res['supports'] = sorted(supports)[-5:]
        res['resistances'] = sorted(resistances)[:5]
        
        # Range boundaries
        res['range_high'] = float(df['high'].iloc[-window:].max())
        res['range_low'] = float(df['low'].iloc[-window:].min())
        res['equilibrium'] = float((res['range_high'] + res['range_low']) / 2.0)
        
        # Calculate premium/discount zones
        if res['range_high'] and res['range_low']:
            total_range = res['range_high'] - res['range_low']
            if total_range > 0:
                position = (current - res['range_low']) / total_range
                res['premium_discount_zones'] = {
                    'current_zone': _get_zone_name(position),
                    'position_in_range': position,
                    'deep_discount': res['range_low'] + total_range * 0.3,
                    'discount': res['range_low'] + total_range * 0.45,
                    'equilibrium': res['equilibrium'],
                    'premium': res['range_low'] + total_range * 0.55,
                    'deep_premium': res['range_low'] + total_range * 0.7
                }
        
        return res
        
    except Exception as e:
        return res

def _get_zone_name(position: float) -> str:
    """Get zone name based on position in range"""
    if position < 0.3:
        return 'DEEP_DISCOUNT'
    elif position < 0.45:
        return 'DISCOUNT'
    elif position <= 0.55:
        return 'EQUILIBRIUM'
    elif position <= 0.7:
        return 'PREMIUM'
    else:
        return 'DEEP_PREMIUM'

def vwap_position(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns position of price relative to VWAP: premium/discount and distance.
    Calculates VWAP if missing.
    """
    out = {
        'vwap': None, 
        'dist': 0.0, 
        'position': 'NEUTRAL',
        'vwap_bands': {},
        'vwap_trend': 'NEUTRAL'
    }
    
    try:
        # If VWAP column doesn't exist, calculate it
        if 'vwap' not in df or df['vwap'].isnull().all():
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        v = float(df['vwap'].iloc[-1])
        price = float(df['close'].iloc[-1])
        dist = (price - v) / v if v != 0 else 0.0
        
        out['vwap'] = v
        out['dist'] = float(dist)
        
        # Position classification
        if dist > 0.02:
            out['position'] = 'STRONG_PREMIUM'
        elif dist > 0.01:
            out['position'] = 'PREMIUM'
        elif dist > 0.005:
            out['position'] = 'SLIGHT_PREMIUM'
        elif dist < -0.02:
            out['position'] = 'STRONG_DISCOUNT'
        elif dist < -0.01:
            out['position'] = 'DISCOUNT'
        elif dist < -0.005:
            out['position'] = 'SLIGHT_DISCOUNT'
        else:
            out['position'] = 'NEUTRAL'
        
        # VWAP bands (standard deviations) - calculate if missing
        if len(df) > 20:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap_std = (typical_price * df['volume']).rolling(20).std() / df['volume'].rolling(20).mean()
            out['vwap_bands'] = {
                'upper_1': v + vwap_std.iloc[-1] if not pd.isna(vwap_std.iloc[-1]) else None,
                'lower_1': v - vwap_std.iloc[-1] if not pd.isna(vwap_std.iloc[-1]) else None,
                'upper_2': v + 2 * vwap_std.iloc[-1] if not pd.isna(vwap_std.iloc[-1]) else None,
                'lower_2': v - 2 * vwap_std.iloc[-1] if not pd.isna(vwap_std.iloc[-1]) else None
            }
        
        # VWAP trend - calculate if missing
        if len(df) > 10:
            vwap_slope = (df['vwap'].iloc[-1] - df['vwap'].iloc[-10]) / df['vwap'].iloc[-10] if df['vwap'].iloc[-10] != 0 else 0
            if vwap_slope > 0.01:
                out['vwap_trend'] = 'UP'
            elif vwap_slope < -0.01:
                out['vwap_trend'] = 'DOWN'
            else:
                out['vwap_trend'] = 'SIDEWAYS'
        
        return out
        
    except Exception as e:
        return out

def get_sr_score(df: pd.DataFrame) -> float:
    """
    Get aggregate support/resistance score (0-1)
    Higher score = price in discount zone (bullish), Lower score = price in premium zone (bearish)
    Calculates from OHLCV if needed.
    """
    try:
        if df is None or df.empty:
            return 0.5
        
        # Calculate S/R and VWAP if needed
        sr = detect_support_resistance(df)
        vwap = vwap_position(df)
        
        score = 0.5
        current_price = float(df['close'].iloc[-1])
        
        # Factor 1: Premium/Discount zone (calculated from OHLCV)
        zones = sr.get('premium_discount_zones', {})
        current_zone = zones.get('current_zone', 'EQUILIBRIUM')
        
        if current_zone == 'DEEP_DISCOUNT':
            score += 0.25
        elif current_zone == 'DISCOUNT':
            score += 0.15
        elif current_zone == 'SLIGHT_DISCOUNT':
            score += 0.08
        elif current_zone == 'PREMIUM':
            score -= 0.15
        elif current_zone == 'DEEP_PREMIUM':
            score -= 0.25
        
        # Factor 2: VWAP position (calculated if missing)
        vwap_pos = vwap.get('position', 'NEUTRAL')
        if vwap_pos == 'STRONG_DISCOUNT':
            score += 0.2
        elif vwap_pos == 'DISCOUNT':
            score += 0.1
        elif vwap_pos == 'STRONG_PREMIUM':
            score -= 0.2
        elif vwap_pos == 'PREMIUM':
            score -= 0.1
        
        # Factor 3: Distance to nearest support/resistance (calculated from data)
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])
        
        if supports:
            nearest_support = max(supports)
            support_distance = (current_price - nearest_support) / current_price
            if support_distance < 0.01:
                score += 0.15
            elif support_distance < 0.02:
                score += 0.08
        
        if resistances:
            nearest_resistance = min(resistances)
            resistance_distance = (nearest_resistance - current_price) / current_price
            if resistance_distance < 0.01:
                score -= 0.15
            elif resistance_distance < 0.02:
                score -= 0.08
        
        return max(0.0, min(1.0, score))
        
    except Exception:
        return 0.5


def get_sr_bias(df: pd.DataFrame) -> float:
    """
    Get net support/resistance bias (-1 to 1)
    Positive = bullish (price in discount), Negative = bearish (price in premium)
    Calculates from OHLCV if needed.
    """
    try:
        if df is None or df.empty:
            return 0.0
        
        # Calculate S/R and VWAP if needed
        sr = detect_support_resistance(df)
        vwap = vwap_position(df)
        
        bias = 0.0
        
        # Premium/Discount contribution (from OHLCV)
        zones = sr.get('premium_discount_zones', {})
        current_zone = zones.get('current_zone', 'EQUILIBRIUM')
        
        zone_map = {
            'DEEP_DISCOUNT': 0.5,
            'DISCOUNT': 0.3,
            'SLIGHT_DISCOUNT': 0.15,
            'EQUILIBRIUM': 0,
            'SLIGHT_PREMIUM': -0.15,
            'PREMIUM': -0.3,
            'DEEP_PREMIUM': -0.5
        }
        bias += zone_map.get(current_zone, 0)
        
        # VWAP contribution (calculated if missing)
        vwap_pos = vwap.get('position', 'NEUTRAL')
        vwap_map = {
            'STRONG_DISCOUNT': 0.4,
            'DISCOUNT': 0.25,
            'SLIGHT_DISCOUNT': 0.1,
            'NEUTRAL': 0,
            'SLIGHT_PREMIUM': -0.1,
            'PREMIUM': -0.25,
            'STRONG_PREMIUM': -0.4
        }
        bias += vwap_map.get(vwap_pos, 0)
        
        return max(-1.0, min(1.0, bias))
        
    except Exception:
        return 0.0


def get_closest_support(df: pd.DataFrame) -> Optional[float]:
    """
    Get the closest support level below current price
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Support price or None
    """
    sr = detect_support_resistance(df)
    current_price = float(df['close'].iloc[-1])
    supports = sr.get('supports', [])
    
    if not supports:
        return None
    
    # Find supports below current price
    below = [s for s in supports if s < current_price]
    if not below:
        return None
    
    return max(below)  # Closest support below


def get_closest_resistance(df: pd.DataFrame) -> Optional[float]:
    """
    Get the closest resistance level above current price
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Resistance price or None
    """
    sr = detect_support_resistance(df)
    current_price = float(df['close'].iloc[-1])
    resistances = sr.get('resistances', [])
    
    if not resistances:
        return None
    
    # Find resistances above current price
    above = [r for r in resistances if r > current_price]
    if not above:
        return None
    
    return min(above)  # Closest resistance above


def has_support_nearby(df: pd.DataFrame, threshold_pct: float = 0.01) -> bool:
    """
    Check if price is near a support level
    
    Args:
        df: DataFrame with OHLCV data
        threshold_pct: Percentage threshold (1% default)
    
    Returns:
        True if near support
    """
    support = get_closest_support(df)
    if support is None:
        return False
    
    current_price = float(df['close'].iloc[-1])
    distance = (current_price - support) / current_price
    
    return distance < threshold_pct


def has_resistance_nearby(df: pd.DataFrame, threshold_pct: float = 0.01) -> bool:
    """
    Check if price is near a resistance level
    
    Args:
        df: DataFrame with OHLCV data
        threshold_pct: Percentage threshold (1% default)
    
    Returns:
        True if near resistance
    """
    resistance = get_closest_resistance(df)
    if resistance is None:
        return False
    
    current_price = float(df['close'].iloc[-1])
    distance = (resistance - current_price) / current_price
    
    return distance < threshold_pct


def get_sr_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive support/resistance summary for scoring layer
    Calculates all data from OHLCV if needed.
    """
    try:
        if df is None or df.empty:
            return {
                'supports': [], 'resistances': [], 'range_high': None, 'range_low': None,
                'equilibrium': None, 'premium_discount_zones': {}, 'vwap': None,
                'vwap_position': 'NEUTRAL', 'vwap_dist': 0, 'vwap_trend': 'NEUTRAL',
                'closest_support': None, 'closest_resistance': None,
                'support_distance_pct': None, 'resistance_distance_pct': None,
                'is_near_support': False, 'is_near_resistance': False,
                'sr_score': 0.5, 'sr_bias': 0.0
            }
        
        # Calculate all required data from OHLCV
        sr = detect_support_resistance(df)
        vwap = vwap_position(df)
        
        current_price = float(df['close'].iloc[-1])
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])
        zones = sr.get('premium_discount_zones', {})
        
        closest_support = get_closest_support(df)
        closest_resistance = get_closest_resistance(df)
        
        support_distance = (current_price - closest_support) / current_price if closest_support else None
        resistance_distance = (closest_resistance - current_price) / current_price if closest_resistance else None
        
        return {
            'supports': supports[-5:],
            'resistances': resistances[:5],
            'range_high': sr.get('range_high'),
            'range_low': sr.get('range_low'),
            'equilibrium': sr.get('equilibrium'),
            'premium_discount_zones': zones,
            'vwap': vwap.get('vwap'),
            'vwap_position': vwap.get('position'),
            'vwap_dist': vwap.get('dist'),
            'vwap_trend': vwap.get('vwap_trend'),
            'closest_support': closest_support,
            'closest_resistance': closest_resistance,
            'support_distance_pct': support_distance,
            'resistance_distance_pct': resistance_distance,
            'is_near_support': has_support_nearby(df),
            'is_near_resistance': has_resistance_nearby(df),
            'sr_score': get_sr_score(df),
            'sr_bias': get_sr_bias(df)
        }
        
    except Exception:
        return {
            'supports': [], 'resistances': [], 'range_high': None, 'range_low': None,
            'equilibrium': None, 'premium_discount_zones': {}, 'vwap': None,
            'vwap_position': 'NEUTRAL', 'vwap_dist': 0, 'vwap_trend': 'NEUTRAL',
            'closest_support': None, 'closest_resistance': None,
            'support_distance_pct': None, 'resistance_distance_pct': None,
            'is_near_support': False, 'is_near_resistance': False,
            'sr_score': 0.5, 'sr_bias': 0.0
        }