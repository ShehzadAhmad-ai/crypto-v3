# market_structure.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
EPSILON = 1e-10  # Small value to prevent division by zero
class OrderBlockStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    EXTREME = "EXTREME"

class MitigationState(Enum):
    UNMITIGATED = "UNMITIGATED"
    PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
    FULLY_MITIGATED = "FULLY_MITIGATED"
    INVALIDATED = "INVALIDATED"

class BreakerType(Enum):
    BULLISH_BREAKER = "BULLISH_BREAKER"
    BEARISH_BREAKER = "BEARISH_BREAKER"
    NONE = "NONE"

@dataclass
class OrderBlock:
    price: float
    high: float
    low: float
    timestamp: pd.Timestamp
    type: str  # 'BULLISH' or 'BEARISH'
    volume: float
    strength_score: float  # 0-1
    strength_label: OrderBlockStrength
    mitigation_state: MitigationState
    mitigation_price: Optional[float] = None
    touched_count: int = 0
    age_bars: int = 0
    is_breaker: bool = False
    breaker_type: BreakerType = BreakerType.NONE

@dataclass
class Swing:
    index: int
    price: float
    type: str  # 'HH'|'LL'|'HL'|'LH'
    timestamp: pd.Timestamp
    volume: float = 0.0
    strength: float = 1.0

def detect_swings(df: pd.DataFrame, window: int = 3) -> Dict[str, Any]:
    """
    Enhanced swing detection with Smart Money Concepts.
    Now includes:
    - Order Block Strength Scoring
    - Mitigation State Tracking
    - Breaker Block Detection
    - Institutional Bias
    """
    out = {'swings': [], 'last_swings': [], 'structure_counts': {}, 
           'order_blocks': [], 'mitigation_tracker': [], 'breakers': [],
           'institutional_bias': {}, 'liquidity_sweep_severity': []}
    
    try:
        if df is None or df.empty or len(df) < (window*2 + 3):
            return out
        
        n = len(df)
        swings: List[Swing] = []
        
        # Detect swings with volume
        for i in range(window, n - window):
            left_h = df['high'].iloc[i-window:i].max()
            right_h = df['high'].iloc[i+1:i+1+window].max()
            left_l = df['low'].iloc[i-window:i].min()
            right_l = df['low'].iloc[i+1:i+1+window].min()
            
            volume = float(df['volume'].iloc[i])
            vol_mean = df['volume'].iloc[max(0, i-20):i].mean() if i >= 20 else df['volume'].mean()
            strength = min(1.0, volume / max(vol_mean, 1))
            
            if df['high'].iat[i] > left_h and df['high'].iat[i] > right_h:
                swings.append(Swing(i, float(df['high'].iat[i]), 'HH', df.index[i], volume, strength))
            if df['low'].iat[i] < left_l and df['low'].iat[i] < right_l:
                swings.append(Swing(i, float(df['low'].iat[i]), 'LL', df.index[i], volume, strength))
        
        swings = sorted(swings, key=lambda s: s.index)
        out['swings'] = swings
        out['last_swings'] = [{'type': s.type, 'price': s.price, 'timestamp': s.timestamp.isoformat(), 
                               'strength': s.strength} for s in swings[-6:]]
        
        cnt = {'HH': sum(1 for s in swings if s.type == 'HH'), 
               'LL': sum(1 for s in swings if s.type == 'LL')}
        out['structure_counts'] = cnt
        
        # Enhanced BOS & CHOCH detection
        bos = []
        choch = []
        for i in range(1, len(swings)):
            prev = swings[i-1]
            curr = swings[i]
            
            # Bullish BOS with strength
            if prev.type == 'LL' and curr.type == 'HH' and curr.price > prev.price * 1.002:
                strength = (curr.price - prev.price) / prev.price * 10  # Normalize
                bos.append({
                    'type': 'BULLISH_BOS',
                    'price': curr.price,
                    'timestamp': curr.timestamp.isoformat(),
                    'ref': prev.price,
                    'strength': min(1.0, strength),
                    'volume_confirmation': curr.strength > 1.2
                })
            
            # Bearish BOS with strength
            if prev.type == 'HH' and curr.type == 'LL' and curr.price < prev.price * 0.998:
                strength = (prev.price - curr.price) / prev.price * 10
                bos.append({
                    'type': 'BEARISH_BOS',
                    'price': curr.price,
                    'timestamp': curr.timestamp.isoformat(),
                    'ref': prev.price,
                    'strength': min(1.0, strength),
                    'volume_confirmation': curr.strength > 1.2
                })
            
            # Enhanced CHOCH detection
            if i >= 2:
                s2 = swings[i-2]
                if s2.type == 'HH' and prev.type == 'LL' and curr.type == 'HH' and curr.price < s2.price * 0.995:
                    strength = (s2.price - curr.price) / s2.price * 10
                    choch.append({
                        'type': 'BULLISH_CHOCH',
                        'direction': 'BULLISH',
                        'timestamp': curr.timestamp.isoformat(),
                        'level': curr.price,
                        'strength': min(1.0, strength)
                    })
                if s2.type == 'LL' and prev.type == 'HH' and curr.type == 'LL' and curr.price > s2.price * 1.005:
                    strength = (curr.price - s2.price) / s2.price * 10
                    choch.append({
                        'type': 'BEARISH_CHOCH',
                        'direction': 'BEARISH',
                        'timestamp': curr.timestamp.isoformat(),
                        'level': curr.price,
                        'strength': min(1.0, strength)
                    })
        
        out['bos'] = bos
        out['choch'] = choch
        
        # ==================== ORDER BLOCK STRENGTH SCORING ====================
        order_blocks = _detect_order_blocks_with_strength(df, swings)
        out['order_blocks'] = order_blocks
        
        # ==================== MITIGATION STATE TRACKER ====================
        mitigation_tracker = _track_mitigation_states(df, order_blocks)
        out['mitigation_tracker'] = mitigation_tracker
        
        # ==================== BREAKER BLOCK DETECTION ====================
        breakers = _detect_breaker_blocks(df, swings, order_blocks)
        out['breakers'] = breakers
        
        # ==================== INSTITUTIONAL BIAS ENGINE ====================
        institutional_bias = _calculate_institutional_bias(swings, bos, choch, order_blocks)
        out['institutional_bias'] = institutional_bias
        
        # ==================== LIQUIDITY SWEEP SEVERITY ====================
        liquidity_sweep_severity = _calculate_liquidity_sweep_severity(df, swings)
        out['liquidity_sweep_severity'] = liquidity_sweep_severity
        
        # Internal/External mapping
        hh = cnt['HH']
        ll = cnt['LL']
        if hh > ll:
            out['market_structure'] = 'EXTERNAL_BULLISH'
        elif ll > hh:
            out['market_structure'] = 'EXTERNAL_BEARISH'
        else:
            out['market_structure'] = 'INTERNAL_CONSOLIDATION'
        
        return out
    except Exception as e:
        out['error'] = str(e)
        return out


def _detect_order_blocks_with_strength(df: pd.DataFrame, swings: List[Swing]) -> List[Dict[str, Any]]:
    """Order Block Strength Scoring - Score OBs based on volume, position, and rejection"""
    order_blocks = []
    
    try:
        if len(swings) < 3:
            return order_blocks
        
        for i in range(1, len(swings) - 1):
            prev = swings[i-1]
            curr = swings[i]
            next_swing = swings[i+1]
            
            # Bullish Order Block: LL -> HH (reversal)
            if prev.type == 'LL' and curr.type == 'HH':
                # Find the candle that formed the low
                for j in range(max(0, prev.index - 5), min(len(df), prev.index + 5)):
                    candle = df.iloc[j]
                    if abs(candle['low'] - prev.price) / prev.price < 0.002:
                        # Calculate strength score (0-1)
                        volume_score = min(1.0, candle['volume'] / df['volume'].iloc[max(0, j-20):j].mean())
                        rejection_score = min(1.0, (candle['high'] - candle['low']) / (candle['close'] - candle['low'] + 0.001))
                        position_score = 0.8 if candle['close'] > candle['open'] else 0.5
                        
                        strength_score = (volume_score * 0.4 + rejection_score * 0.4 + position_score * 0.2)
                        
                        # Determine strength label
                        if strength_score >= 0.8:
                            strength_label = OrderBlockStrength.EXTREME
                        elif strength_score >= 0.6:
                            strength_label = OrderBlockStrength.STRONG
                        elif strength_score >= 0.4:
                            strength_label = OrderBlockStrength.MODERATE
                        else:
                            strength_label = OrderBlockStrength.WEAK
                        
                        # Check if touched/mitigated
                        touched_count = 0
                        for k in range(j + 1, len(df)):
                            if df['low'].iloc[k] <= candle['low'] * 1.001:
                                touched_count += 1
                        
                        order_blocks.append({
                            'type': 'BULLISH_OB',
                            'price': float(candle['low']),
                            'high': float(candle['high']),
                            'low': float(candle['low']),
                            'timestamp': df.index[j].isoformat(),
                            'volume': float(candle['volume']),
                            'strength_score': round(strength_score, 3),
                            'strength_label': strength_label.value,
                            'mitigation_state': MitigationState.UNMITIGATED.value,
                            'touched_count': touched_count,
                            'age_bars': len(df) - j,
                            'is_breaker': False
                        })
                        break
            
            # Bearish Order Block: HH -> LL (reversal)
            if prev.type == 'HH' and curr.type == 'LL':
                # Find the candle that formed the high
                for j in range(max(0, prev.index - 5), min(len(df), prev.index + 5)):
                    candle = df.iloc[j]
                    if abs(candle['high'] - prev.price) / prev.price < 0.002:
                        volume_score = min(1.0, candle['volume'] / df['volume'].iloc[max(0, j-20):j].mean())
                        rejection_score = min(1.0, (candle['high'] - candle['low']) / (candle['high'] - candle['close'] + 0.001))
                        position_score = 0.8 if candle['close'] < candle['open'] else 0.5
                        
                        strength_score = (volume_score * 0.4 + rejection_score * 0.4 + position_score * 0.2)
                        
                        if strength_score >= 0.8:
                            strength_label = OrderBlockStrength.EXTREME
                        elif strength_score >= 0.6:
                            strength_label = OrderBlockStrength.STRONG
                        elif strength_score >= 0.4:
                            strength_label = OrderBlockStrength.MODERATE
                        else:
                            strength_label = OrderBlockStrength.WEAK
                        
                        touched_count = 0
                        for k in range(j + 1, len(df)):
                            if df['high'].iloc[k] >= candle['high'] * 0.999:
                                touched_count += 1
                        
                        order_blocks.append({
                            'type': 'BEARISH_OB',
                            'price': float(candle['high']),
                            'high': float(candle['high']),
                            'low': float(candle['low']),
                            'timestamp': df.index[j].isoformat(),
                            'volume': float(candle['volume']),
                            'strength_score': round(strength_score, 3),
                            'strength_label': strength_label.value,
                            'mitigation_state': MitigationState.UNMITIGATED.value,
                            'touched_count': touched_count,
                            'age_bars': len(df) - j,
                            'is_breaker': False
                        })
                        break
        
        return order_blocks
    except Exception:
        return order_blocks


def _track_mitigation_states(df: pd.DataFrame, order_blocks: List[Dict]) -> List[Dict]:
    """Track mitigation states of order blocks"""
    mitigation_tracker = []
    
    try:
        current_price = float(df['close'].iloc[-1])
        
        for ob in order_blocks:
            state = MitigationState.UNMITIGATED.value
            mitigation_price = None
            
            if ob['type'] == 'BULLISH_OB':
                # Bullish OB is mitigated when price returns to it
                if current_price <= ob['price'] * 1.005:
                    # Check how much of the OB is mitigated
                    if current_price <= ob['low']:
                        state = MitigationState.FULLY_MITIGATED.value
                        mitigation_price = current_price
                    elif current_price <= ob['high']:
                        state = MitigationState.PARTIALLY_MITIGATED.value
                        mitigation_price = current_price
            
            elif ob['type'] == 'BEARISH_OB':
                # Bearish OB is mitigated when price returns to it
                if current_price >= ob['price'] * 0.995:
                    if current_price >= ob['high']:
                        state = MitigationState.FULLY_MITIGATED.value
                        mitigation_price = current_price
                    elif current_price >= ob['low']:
                        state = MitigationState.PARTIALLY_MITIGATED.value
                        mitigation_price = current_price
            
            # Check if invalidated (price moved too far without touching)
            if ob['age_bars'] > 50 and ob['touched_count'] == 0:
                state = MitigationState.INVALIDATED.value
            
            mitigation_tracker.append({
                'order_block': ob,
                'current_state': state,
                'mitigation_price': mitigation_price,
                'distance_from_current': abs(current_price - ob['price']) / current_price,
                'touched_since_formation': ob['touched_count'] > 0
            })
        
        return mitigation_tracker
    except Exception:
        return mitigation_tracker


def _detect_breaker_blocks(df: pd.DataFrame, swings: List[Swing], order_blocks: List[Dict]) -> List[Dict]:
    """Breaker Block Detection - OBs that become support/resistance after mitigation"""
    breakers = []
    
    try:
        if len(swings) < 4:
            return breakers
        
        current_price = float(df['close'].iloc[-1])
        
        for i in range(2, len(swings) - 1):
            s3 = swings[i-2]
            s2 = swings[i-1]
            s1 = swings[i]
            s0 = swings[i+1]
            
            # Bullish Breaker: HH -> LL -> HH (lower high) -> breaks below
            if (s3.type == 'HH' and s2.type == 'LL' and s1.type == 'HH' and 
                s1.price < s3.price and s0.type == 'LL' and s0.price < s2.price):
                
                # Find the OB that became the breaker
                for ob in order_blocks:
                    if ob['type'] == 'BEARISH_OB' and abs(ob['price'] - s2.price) / s2.price < 0.01:
                        breakers.append({
                            'type': 'BULLISH_BREAKER',
                            'price': ob['price'],
                            'formation_time': ob['timestamp'],
                            'break_time': s0.timestamp.isoformat(),
                            'strength': ob['strength_score'],
                            'distance_from_current': abs(current_price - ob['price']) / current_price
                        })
                        break
            
            # Bearish Breaker: LL -> HH -> LL (higher low) -> breaks above
            if (s3.type == 'LL' and s2.type == 'HH' and s1.type == 'LL' and 
                s1.price > s3.price and s0.type == 'HH' and s0.price > s2.price):
                
                for ob in order_blocks:
                    if ob['type'] == 'BULLISH_OB' and abs(ob['price'] - s2.price) / s2.price < 0.01:
                        breakers.append({
                            'type': 'BEARISH_BREAKER',
                            'price': ob['price'],
                            'formation_time': ob['timestamp'],
                            'break_time': s0.timestamp.isoformat(),
                            'strength': ob['strength_score'],
                            'distance_from_current': abs(current_price - ob['price']) / current_price
                        })
                        break
        
        return breakers
    except Exception:
        return breakers


def _calculate_institutional_bias(swings: List[Swing], bos: List[Dict], choch: List[Dict], 
                                  order_blocks: List[Dict]) -> Dict[str, Any]:
    """Institutional Bias Engine - Determine where smart money is positioned"""
    
    try:
        if not swings:
            return {'bias': 'NEUTRAL', 'confidence': 0.0}
        
        # Count swing types with strength weighting
        hh_strength = sum(s.strength for s in swings if s.type == 'HH')
        ll_strength = sum(s.strength for s in swings if s.type == 'LL')
        
        # BOS direction with strength
        bos_bullish = sum(b.get('strength', 0) for b in bos if b['type'] == 'BULLISH_BOS')
        bos_bearish = sum(b.get('strength', 0) for b in bos if b['type'] == 'BEARISH_BOS')
        
        # CHOCH direction
        choch_bullish = len([c for c in choch if c['type'] == 'BULLISH_CHOCH'])
        choch_bearish = len([c for c in choch if c['type'] == 'BEARISH_CHOCH'])
        
        # Order block strength
        ob_bullish = sum(ob.get('strength_score', 0) for ob in order_blocks if ob['type'] == 'BULLISH_OB')
        ob_bearish = sum(ob.get('strength_score', 0) for ob in order_blocks if ob['type'] == 'BEARISH_OB')
        
        # Calculate weighted bias score (-1 to 1)
        total_score = 0
        total_weight = 0
        
        # Swings weight
        if hh_strength + ll_strength > 0:
            swing_score = (hh_strength - ll_strength) / (hh_strength + ll_strength)
            total_score += swing_score * 0.3
            total_weight += 0.3
        
        # BOS weight
        if bos_bullish + bos_bearish > 0:
            bos_score = (bos_bullish - bos_bearish) / (bos_bullish + bos_bearish + EPSILON)
            total_score += bos_score * 0.4
            total_weight += 0.4
        
        # CHOCH weight
        if choch_bullish + choch_bearish > 0:
            choch_score = (choch_bullish - choch_bearish) / (choch_bullish + choch_bearish + EPSILON)
            total_score += choch_score * 0.15
            total_weight += 0.15
        
        # Order block weight
        if ob_bullish + ob_bearish > 0:
            ob_score = (ob_bullish - ob_bearish) / (ob_bullish + ob_bearish + EPSILON)
            total_score += ob_score * 0.15
            total_weight += 0.15
        
        final_score = total_score / max(total_weight, 0.001)
        confidence = min(1.0, abs(final_score) * 1.5)
        
        if final_score > 0.2:
            bias = 'BULLISH'
        elif final_score < -0.2:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'
        
        return {
            'bias': bias,
            'score': round(final_score, 3),
            'confidence': round(confidence, 3),
            'components': {
                'swing_bias': round(hh_strength - ll_strength, 3),
                'bos_bias': round(bos_bullish - bos_bearish, 3),
                'choch_bias': choch_bullish - choch_bearish,
                'ob_bias': round(ob_bullish - ob_bearish, 3)
            }
        }
    except Exception:
        return {'bias': 'NEUTRAL', 'confidence': 0.0}


def _calculate_liquidity_sweep_severity(df: pd.DataFrame, swings: List[Swing]) -> List[Dict]:
    """Liquidity Sweep Severity - Measure how aggressively liquidity was taken"""
    
    severity = []
    
    try:
        if len(swings) < 2:
            return severity
        
        current_price = float(df['close'].iloc[-1])
        atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
        
        for i in range(1, len(swings)):
            prev = swings[i-1]
            curr = swings[i]
            
            # Check for sweep of previous high
            if prev.type == 'HH' and curr.type == 'HH' and curr.price > prev.price:
                sweep_distance = (curr.price - prev.price) / atr
                volume_at_sweep = curr.volume / df['volume'].iloc[max(0, curr.index-20):curr.index].mean()
                
                severity_score = min(1.0, sweep_distance * volume_at_sweep / 3)
                
                severity.append({
                    'type': 'HIGH_SWEEP',
                    'swept_level': prev.price,
                    'sweep_price': curr.price,
                    'sweep_distance_atr': round(sweep_distance, 2),
                    'volume_ratio': round(volume_at_sweep, 2),
                    'severity': round(severity_score, 3),
                    'timestamp': curr.timestamp.isoformat()
                })
            
            # Check for sweep of previous low
            if prev.type == 'LL' and curr.type == 'LL' and curr.price < prev.price:
                sweep_distance = (prev.price - curr.price) / atr
                volume_at_sweep = curr.volume / df['volume'].iloc[max(0, curr.index-20):curr.index].mean()
                
                severity_score = min(1.0, sweep_distance * volume_at_sweep / 3)
                
                severity.append({
                    'type': 'LOW_SWEEP',
                    'swept_level': prev.price,
                    'sweep_price': curr.price,
                    'sweep_distance_atr': round(sweep_distance, 2),
                    'volume_ratio': round(volume_at_sweep, 2),
                    'severity': round(severity_score, 3),
                    'timestamp': curr.timestamp.isoformat()
                })
        
        return severity
    except Exception:
        return severity
    


# ==================== ADD THESE FUNCTIONS AT THE END OF THE FILE ====================

def get_structure_score(df: pd.DataFrame) -> float:
    """
    Get aggregate structure score (0-1) for scoring layer
    Higher score = stronger bullish structure, Lower score = stronger bearish structure
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Score between 0 and 1
    """
    structure = detect_swings(df)
    
    # Get institutional bias
    bias = structure.get('institutional_bias', {})
    bias_score = bias.get('score', 0)  # -1 to 1
    bias_confidence = bias.get('confidence', 0.5)
    
    # Convert bias_score (-1 to 1) to score (0 to 1)
    # -1 = bearish → score 0, +1 = bullish → score 1
    raw_score = (bias_score + 1) / 2
    
    # Adjust by confidence
    final_score = 0.5 + ((raw_score - 0.5) * bias_confidence)
    
    # Additional factors
    # Check if there are recent BOS events
    bos = structure.get('bos', [])
    if bos:
        latest_bos = bos[-1]
        if latest_bos.get('type') == 'BULLISH_BOS':
            final_score += 0.05 * latest_bos.get('strength', 0.5)
        elif latest_bos.get('type') == 'BEARISH_BOS':
            final_score -= 0.05 * latest_bos.get('strength', 0.5)
    
    # Check order block strength
    order_blocks = structure.get('order_blocks', [])
    if order_blocks:
        bullish_obs = [ob for ob in order_blocks if ob['type'] == 'BULLISH_OB']
        bearish_obs = [ob for ob in order_blocks if ob['type'] == 'BEARISH_OB']
        
        bullish_strength = sum(ob.get('strength_score', 0) for ob in bullish_obs)
        bearish_strength = sum(ob.get('strength_score', 0) for ob in bearish_obs)
        total = bullish_strength + bearish_strength
        
        if total > 0:
            ob_bias = (bullish_strength - bearish_strength) / total
            final_score += ob_bias * 0.1
    
    # Clamp to 0-1
    return max(0.0, min(1.0, final_score))


def get_structure_bias(df: pd.DataFrame) -> float:
    """
    Get net structure bias (-1 to 1)
    Positive = bullish bias, Negative = bearish bias
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Bias score between -1 and 1
    """
    structure = detect_swings(df)
    bias = structure.get('institutional_bias', {})
    return bias.get('score', 0.0)


def get_structure_confidence(df: pd.DataFrame) -> float:
    """
    Get confidence in the current structure analysis (0-1)
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Confidence score between 0 and 1
    """
    structure = detect_swings(df)
    
    confidence = 0.5
    
    # Factor 1: Institutional bias confidence
    bias = structure.get('institutional_bias', {})
    bias_confidence = bias.get('confidence', 0.5)
    confidence += bias_confidence * 0.3
    
    # Factor 2: Number of swings (more swings = more data)
    swings = structure.get('swings', [])
    swing_count = len(swings)
    if swing_count >= 10:
        confidence += 0.15
    elif swing_count >= 5:
        confidence += 0.1
    elif swing_count >= 3:
        confidence += 0.05
    
    # Factor 3: Recent BOS events
    bos = structure.get('bos', [])
    if bos:
        latest_bos = bos[-1]
        confidence += 0.1 * latest_bos.get('strength', 0.5)
    
    # Factor 4: Order block strength
    order_blocks = structure.get('order_blocks', [])
    if order_blocks:
        strong_blocks = [ob for ob in order_blocks if ob.get('strength_label') in ['STRONG', 'EXTREME']]
        if strong_blocks:
            confidence += 0.1 * min(1.0, len(strong_blocks) / 3)
    
    return max(0.0, min(1.0, confidence))


def get_structure_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive structure summary for scoring layer
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with structure summary
    """
    structure = detect_swings(df)
    bias = structure.get('institutional_bias', {})
    
    # Get recent BOS and CHOCH
    bos = structure.get('bos', [])
    choch = structure.get('choch', [])
    
    # Get recent order blocks
    order_blocks = structure.get('order_blocks', [])
    recent_obs = order_blocks[-3:] if len(order_blocks) > 3 else order_blocks
    
    # Calculate trend direction
    bias_score = bias.get('score', 0)
    if bias_score > 0.2:
        trend = 'BULLISH'
    elif bias_score < -0.2:
        trend = 'BEARISH'
    else:
        trend = 'NEUTRAL'
    
    # Check for recent structure breaks
    recent_bos = bos[-1] if bos else None
    recent_choch = choch[-1] if choch else None
    
    # Check for unmitigated order blocks
    unmitigated_bullish = [ob for ob in recent_obs if ob['type'] == 'BULLISH_OB' and ob.get('mitigation_state') == 'UNMITIGATED']
    unmitigated_bearish = [ob for ob in recent_obs if ob['type'] == 'BEARISH_OB' and ob.get('mitigation_state') == 'UNMITIGATED']
    
    return {
        'trend': trend,
        'bias_score': bias.get('score', 0),
        'bias_confidence': bias.get('confidence', 0),
        'swing_count': len(structure.get('swings', [])),
        'bos_count': len(bos),
        'bos_last_direction': recent_bos.get('type', 'NONE') if recent_bos else 'NONE',
        'bos_last_strength': recent_bos.get('strength', 0) if recent_bos else 0,
        'choch_count': len(choch),
        'choch_last_direction': recent_choch.get('direction', 'NONE') if recent_choch else 'NONE',
        'order_blocks_count': len(order_blocks),
        'unmitigated_bullish_obs': len(unmitigated_bullish),
        'unmitigated_bearish_obs': len(unmitigated_bearish),
        'has_recent_bos': len(bos) > 0,
        'has_recent_choch': len(choch) > 0,
        'has_strong_bullish_ob': any(ob.get('strength_label') in ['STRONG', 'EXTREME'] and ob['type'] == 'BULLISH_OB' for ob in recent_obs),
        'has_strong_bearish_ob': any(ob.get('strength_label') in ['STRONG', 'EXTREME'] and ob['type'] == 'BEARISH_OB' for ob in recent_obs),
        'structure_score': get_structure_score(df),
        'structure_bias': get_structure_bias(df),
        'structure_confidence': get_structure_confidence(df)
    }


def has_strong_structure(df: pd.DataFrame, min_confidence: float = 0.7) -> Dict[str, Any]:
    """
    Check if there is strong structure bias
    
    Args:
        df: DataFrame with OHLCV data
        min_confidence: Minimum confidence for strong structure
    
    Returns:
        Dictionary with strong structure information
    """
    summary = get_structure_summary(df)
    
    is_strong_bullish = (
        summary['trend'] == 'BULLISH' and
        summary['bias_confidence'] >= min_confidence and
        (summary['has_recent_bos'] or summary['has_strong_bullish_ob'])
    )
    
    is_strong_bearish = (
        summary['trend'] == 'BEARISH' and
        summary['bias_confidence'] >= min_confidence and
        (summary['has_recent_bos'] or summary['has_strong_bearish_ob'])
    )
    
    return {
        'has_strong_bullish': is_strong_bullish,
        'has_strong_bearish': is_strong_bearish,
        'confidence': summary['bias_confidence'],
        'bias_score': summary['bias_score'],
        'trend': summary['trend'],
        'reasons': []
    }



def get_structure_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get counts of structure events
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with structure counts
    """
    structure = detect_swings(df)
    
    bos = structure.get('bos', [])
    choch = structure.get('choch', [])
    order_blocks = structure.get('order_blocks', [])
    
    return {
        'bos_bullish': sum(1 for b in bos if b['type'] == 'BULLISH_BOS'),
        'bos_bearish': sum(1 for b in bos if b['type'] == 'BEARISH_BOS'),
        'choch_bullish': sum(1 for c in choch if c['type'] == 'BULLISH_CHOCH'),
        'choch_bearish': sum(1 for c in choch if c['type'] == 'BEARISH_CHOCH'),
        'order_blocks_bullish': sum(1 for ob in order_blocks if ob['type'] == 'BULLISH_OB'),
        'order_blocks_bearish': sum(1 for ob in order_blocks if ob['type'] == 'BEARISH_OB'),
        'total_bos': len(bos),
        'total_choch': len(choch),
        'total_order_blocks': len(order_blocks)
    }