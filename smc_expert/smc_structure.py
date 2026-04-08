"""
SMC Expert V3 - Market Structure Analysis (COMPLETE REWRITE)
Swing detection, BOS/CHOCH/MSS, and AMD phase detection
FIXED: Proper object access, no .get() on Swing objects, ATR fallbacks
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .smc_core import (
    Swing, SwingType, Direction, AMDPhase, Candle,
    SMCContext, calculate_atr, normalize, safe_get_swing_price, safe_get_swing_type
)
from .smc_config import CONFIG


class SwingDetector:
    """Detects HH, HL, LH, LL swing points - FIXED: No .get() on objects"""
    
    def __init__(self):
        self.swings: List[Swing] = []
        self.window = CONFIG.SWING_WINDOW
        self.min_distance_atr = CONFIG.MIN_SWING_DISTANCE_ATR
    
    def detect_swings(self, df: pd.DataFrame) -> List[Swing]:
        """
        Detect swing points using local extrema
        Returns list of Swing objects
        """
        self.swings = []
        atr = calculate_atr(df)
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50  # Fallback
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        timestamps = df.index
        
        for i in range(self.window, len(df) - self.window):
            # Check for swing high (HH or LH)
            is_swing_high = self._is_local_high(highs, i)
            if is_swing_high:
                swing_type = self._determine_swing_type(highs, lows, i, 'high')
                if swing_type:
                    strength = self._calculate_swing_strength(df, i, atr, 'high')
                    timestamp = timestamps[i] if isinstance(timestamps[i], datetime) else datetime.now()
                    swing = Swing(
                        index=i,
                        price=highs[i],
                        type=swing_type,
                        strength=strength,
                        timestamp=timestamp,
                        confirmed=True
                    )
                    self.swings.append(swing)
            
            # Check for swing low (HL or LL)
            is_swing_low = self._is_local_low(lows, i)
            if is_swing_low:
                swing_type = self._determine_swing_type(highs, lows, i, 'low')
                if swing_type:
                    strength = self._calculate_swing_strength(df, i, atr, 'low')
                    timestamp = timestamps[i] if isinstance(timestamps[i], datetime) else datetime.now()
                    swing = Swing(
                        index=i,
                        price=lows[i],
                        type=swing_type,
                        strength=strength,
                        timestamp=timestamp,
                        confirmed=True
                    )
                    self.swings.append(swing)
        
        # Filter swings by minimum distance
        self.swings = self._filter_by_distance(self.swings, atr)
        
        return self.swings
    
    def _is_local_high(self, highs: np.ndarray, i: int) -> bool:
        """Check if index i is a local high"""
        left = max(0, i - self.window)
        right = min(len(highs) - 1, i + self.window)
        
        for j in range(left, right + 1):
            if j == i:
                continue
            if highs[j] >= highs[i]:
                return False
        return True
    
    def _is_local_low(self, lows: np.ndarray, i: int) -> bool:
        """Check if index i is a local low"""
        left = max(0, i - self.window)
        right = min(len(lows) - 1, i + self.window)
        
        for j in range(left, right + 1):
            if j == i:
                continue
            if lows[j] <= lows[i]:
                return False
        return True
    
    def _determine_swing_type(self, highs: np.ndarray, lows: np.ndarray, 
                               i: int, swing_type: str) -> Optional[SwingType]:
        """Determine if swing is HH/HL or LH/LL based on previous swings"""
        if not self.swings:
            return SwingType.HH if swing_type == 'high' else SwingType.HL
        
        last_swing = self.swings[-1]
        last_price = last_swing.price
        
        if swing_type == 'high':
            if highs[i] > last_price:
                return SwingType.HH
            elif highs[i] < last_price:
                return SwingType.LH
        else:  # low
            if lows[i] > last_price:
                return SwingType.HL
            elif lows[i] < last_price:
                return SwingType.LL
        
        return None
    
    def _calculate_swing_strength(self, df: pd.DataFrame, i: int, 
                                   atr: float, swing_type: str) -> float:
        """Calculate swing strength based on momentum and volume"""
        if i < 5 or atr <= 0:
            return 0.5
        
        # Momentum: price change over swing formation
        if swing_type == 'high':
            price_change = df['high'].iloc[i] - df['high'].iloc[i - 5]
        else:
            price_change = df['low'].iloc[i] - df['low'].iloc[i - 5]
        
        momentum = abs(price_change) / atr
        momentum_score = min(1.0, momentum / 2.0)
        
        # Volume score
        avg_volume = df['volume'].iloc[max(0, i-20):i].mean()
        volume_ratio = df['volume'].iloc[i] / avg_volume if avg_volume > 0 else 1.0
        volume_score = min(1.0, volume_ratio / 1.5)
        
        # Candle strength
        if 'body_ratio' in df.columns:
            candle_strength = df['body_ratio'].iloc[i]
        else:
            candle_strength = 0.5
        
        strength = (momentum_score * 0.4 + volume_score * 0.3 + candle_strength * 0.3)
        
        return min(1.0, strength)
    
    def _filter_by_distance(self, swings: List[Swing], atr: float) -> List[Swing]:
        """Remove swings that are too close to each other"""
        if len(swings) < 2 or atr <= 0:
            return swings
        
        filtered = [swings[0]]
        min_distance = self.min_distance_atr * atr
        
        for swing in swings[1:]:
            if abs(swing.price - filtered[-1].price) >= min_distance:
                filtered.append(swing)
            elif swing.strength > filtered[-1].strength:
                filtered[-1] = swing
        
        return filtered


class BOSCHOCHDetector:
    """Detects Break of Structure (BOS) and Change of Character (CHOCH/MSS)"""
    
    def __init__(self):
        self.bos_points: List[Dict] = []
        self.choch_points: List[Dict] = []
        self.mss_points: List[Dict] = []
    
    def detect_bos(self, swings: List[Swing], df: pd.DataFrame) -> List[Dict]:
        """
        Break of Structure - price breaks above HH or below LL
        """
        self.bos_points = []
        atr = calculate_atr(df)
        
        if len(swings) < 3 or atr <= 0:
            return self.bos_points
        
        for i in range(2, len(swings)):
            current = swings[i]
            earlier = swings[i - 2]
            
            # Bullish BOS: break above previous HH
            if current.type == SwingType.HH and current.price > earlier.price:
                move_size = current.price - earlier.price
                strength = self._calculate_bos_strength(df, current.index, move_size, atr)
                
                self.bos_points.append({
                    'type': 'BULLISH',
                    'price': current.price,
                    'break_level': earlier.price,
                    'index': current.index,
                    'timestamp': current.timestamp,
                    'strength': strength,
                    'direction': Direction.BUY
                })
            
            # Bearish BOS: break below previous LL
            elif current.type == SwingType.LL and current.price < earlier.price:
                move_size = earlier.price - current.price
                strength = self._calculate_bos_strength(df, current.index, move_size, atr)
                
                self.bos_points.append({
                    'type': 'BEARISH',
                    'price': current.price,
                    'break_level': earlier.price,
                    'index': current.index,
                    'timestamp': current.timestamp,
                    'strength': strength,
                    'direction': Direction.SELL
                })
        
        return self.bos_points
    
    def detect_choch(self, swings: List[Swing], df: pd.DataFrame) -> List[Dict]:
        """
        Change of Character - price breaks above LH or below HL
        This is the same as Market Structure Shift (MSS)
        """
        self.choch_points = []
        atr = calculate_atr(df)
        
        if len(swings) < 4 or atr <= 0:
            return self.choch_points
        
        for i in range(3, len(swings)):
            current = swings[i]
            prev_high = None
            prev_low = None
            
            # Find previous LH and HL
            for j in range(i - 1, max(0, i - 10), -1):
                if swings[j].type == SwingType.LH and prev_high is None:
                    prev_high = swings[j]
                if swings[j].type == SwingType.HL and prev_low is None:
                    prev_low = swings[j]
                if prev_high and prev_low:
                    break
            
            # Bullish CHOCH: break above previous LH
            if prev_high and current.type == SwingType.HH and current.price > prev_high.price:
                strength = self._calculate_choch_strength(df, current, prev_high, atr)
                
                self.choch_points.append({
                    'type': 'BULLISH',
                    'price': current.price,
                    'break_level': prev_high.price,
                    'index': current.index,
                    'timestamp': current.timestamp,
                    'strength': strength,
                    'direction': Direction.BUY
                })
            
            # Bearish CHOCH: break below previous HL
            elif prev_low and current.type == SwingType.LL and current.price < prev_low.price:
                strength = self._calculate_choch_strength(df, current, prev_low, atr)
                
                self.choch_points.append({
                    'type': 'BEARISH',
                    'price': current.price,
                    'break_level': prev_low.price,
                    'index': current.index,
                    'timestamp': current.timestamp,
                    'strength': strength,
                    'direction': Direction.SELL
                })
        
        # MSS is the same as CHOCH
        self.mss_points = self.choch_points.copy()
        
        return self.choch_points
    
    def _calculate_bos_strength(self, df: pd.DataFrame, index: int, 
                                 move_size: float, atr: float) -> float:
        """Calculate BOS strength"""
        if index >= len(df) or atr <= 0:
            return 0.5
        
        move_score = min(1.0, move_size / (CONFIG.BOS_MIN_MOVE_ATR * atr))
        
        # Volume at breakout
        avg_volume = df['volume'].iloc[max(0, index-20):index].mean()
        volume_ratio = df['volume'].iloc[index] / avg_volume if avg_volume > 0 else 1.0
        volume_score = min(1.0, volume_ratio / 1.5)
        
        strength = (move_score * 0.5 + volume_score * 0.5)
        
        return min(1.0, strength)
    
    def _calculate_choch_strength(self, df: pd.DataFrame, current: Swing,
                                   break_level: Swing, atr: float) -> float:
        """Calculate CHOCH/MSS strength"""
        if atr <= 0:
            return 0.5
        
        move_size = abs(current.price - break_level.price)
        move_score = min(1.0, move_size / (CONFIG.MSS_MIN_MOVE_ATR * atr))
        
        # Volume at shift
        avg_volume = df['volume'].iloc[max(0, current.index-20):current.index].mean()
        volume_ratio = df['volume'].iloc[current.index] / avg_volume if avg_volume > 0 else 1.0
        volume_score = min(1.0, volume_ratio / 1.5)
        
        strength = (move_score * 0.5 + volume_score * 0.5)
        
        return min(1.0, strength)


class AMDPhaseDetector:
    """Detects Accumulation, Manipulation, Distribution phases"""
    
    def __init__(self):
        self.phase = AMDPhase.UNKNOWN
        self.confidence = 0.0
    
    def detect_phase(self, df: pd.DataFrame, swings: List[Swing],
                      bos_points: List[Dict], sweeps: List[Dict]) -> Tuple[AMDPhase, float]:
        """
        Detect current AMD phase based on price action
        """
        atr = calculate_atr(df)
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50
        
        # Detect Accumulation
        accumulation_score = self._detect_accumulation(df, atr)
        
        # Detect Manipulation
        manipulation_score = self._detect_manipulation(df, sweeps, atr)
        
        # Detect Distribution
        distribution_score = self._detect_distribution(df, bos_points, atr)
        
        # Determine phase based on highest score
        scores = {
            AMDPhase.ACCUMULATION: accumulation_score,
            AMDPhase.MANIPULATION: manipulation_score,
            AMDPhase.DISTRIBUTION: distribution_score
        }
        
        self.phase = max(scores, key=scores.get)
        self.confidence = scores[self.phase]
        
        return self.phase, self.confidence
    
    def _detect_accumulation(self, df: pd.DataFrame, atr: float) -> float:
        """Detect accumulation phase (range-bound with decreasing volume)"""
        lookback = min(CONFIG.ACCUMULATION_MIN_BARS, len(df))
        recent_df = df.tail(lookback)
        
        # Range width
        price_range = recent_df['high'].max() - recent_df['low'].min()
        range_score = 1.0 if price_range < CONFIG.ACCUMULATION_RANGE_ATR * atr else max(0.0, 1.0 - (price_range / (atr * 3)))
        
        # Volume trend (decreasing)
        volumes = recent_df['volume'].values
        if len(volumes) > 5 and volumes[0] > 0:
            volume_trend = (volumes[-1] - volumes[0]) / volumes[0]
            volume_score = max(0.0, 1.0 - abs(volume_trend))
        else:
            volume_score = 0.5
        
        # Consolidation pattern (small candles)
        if 'body_ratio' in recent_df.columns:
            body_ratios = recent_df['body_ratio'].values
            consolidation_score = 1.0 - min(1.0, np.mean(body_ratios))
        else:
            consolidation_score = 0.5
        
        score = (range_score * 0.4 + volume_score * 0.3 + consolidation_score * 0.3)
        
        return score
    
    def _detect_manipulation(self, df: pd.DataFrame, sweeps: List[Dict], atr: float) -> float:
        """Detect manipulation phase (liquidity sweeps)"""
        if not sweeps:
            return 0.3
        
        # Recent sweeps (last 10 bars)
        recent_sweeps = [s for s in sweeps if s.get('index', 0) > len(df) - 15]
        
        if not recent_sweeps:
            return 0.3
        
        sweep_count_score = min(1.0, len(recent_sweeps) / 3)
        sweep_strength = np.mean([s.get('strength', 0.5) for s in recent_sweeps])
        
        score = (sweep_count_score * 0.5 + sweep_strength * 0.5)
        
        return score
    
    def _detect_distribution(self, df: pd.DataFrame, bos_points: List[Dict], atr: float) -> float:
        """Detect distribution phase (trend with expanding volume)"""
        if not bos_points:
            return 0.3
        
        lookback = 20
        recent_df = df.tail(lookback)
        
        # Trend strength
        price_change = recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]
        trend_score = min(1.0, abs(price_change) / (CONFIG.DISTRIBUTION_TREND_ATR * atr)) if atr > 0 else 0.5
        
        # Volume trend (increasing)
        volumes = recent_df['volume'].values
        if len(volumes) > 5 and volumes[0] > 0:
            volume_trend = (volumes[-1] - volumes[0]) / volumes[0]
            volume_score = min(1.0, max(0.0, volume_trend))
        else:
            volume_score = 0.5
        
        # Recent BOS strength
        recent_bos = [b for b in bos_points if b.get('index', 0) > len(df) - lookback]
        bos_strength = np.mean([b.get('strength', 0.5) for b in recent_bos]) if recent_bos else 0.5
        
        score = (trend_score * 0.4 + volume_score * 0.3 + bos_strength * 0.3)
        
        return score


class MarketStructureAnalyzer:
    """Main market structure analyzer that orchestrates all components"""
    
    def __init__(self):
        self.swing_detector = SwingDetector()
        self.bos_choch_detector = BOSCHOCHDetector()
        self.amd_detector = AMDPhaseDetector()
        
        self.swings: List[Swing] = []
        self.bos_points: List[Dict] = []
        self.choch_points: List[Dict] = []
        self.mss_points: List[Dict] = []
        self.amd_phase: AMDPhase = AMDPhase.UNKNOWN
        self.amd_confidence: float = 0.0
    
    def analyze(self, df: pd.DataFrame, sweeps: List[Dict] = None) -> Dict:
        """
        Complete market structure analysis
        """
        if sweeps is None:
            sweeps = []
        
        # Detect swings
        self.swings = self.swing_detector.detect_swings(df)
        
        # Detect BOS and CHOCH/MSS
        self.bos_points = self.bos_choch_detector.detect_bos(self.swings, df)
        self.choch_points = self.bos_choch_detector.detect_choch(self.swings, df)
        self.mss_points = self.bos_choch_detector.mss_points
        
        # Detect AMD phase
        self.amd_phase, self.amd_confidence = self.amd_detector.detect_phase(
            df, self.swings, self.bos_points, sweeps
        )
        
        # Get current structure direction
        current_structure = self._get_current_structure()
        
        return {
            'swings': self.swings,
            'bos_points': self.bos_points,
            'choch_points': self.choch_points,
            'mss_points': self.mss_points,
            'amd_phase': self.amd_phase,
            'amd_confidence': self.amd_confidence,
            'current_structure': current_structure['direction'],
            'structure_strength': current_structure['strength'],
            'trend_regime': current_structure['regime']
        }
    
    def _get_current_structure(self) -> Dict:
        """Determine current market structure direction and strength"""
        if len(self.swings) < 4:
            return {'direction': 'NEUTRAL', 'strength': 0.5, 'regime': 'NEUTRAL'}
        
        # Look at last 4 swings
        last_4 = self.swings[-4:]
        
        hh_count = sum(1 for s in last_4 if s.type == SwingType.HH)
        hl_count = sum(1 for s in last_4 if s.type == SwingType.HL)
        lh_count = sum(1 for s in last_4 if s.type == SwingType.LH)
        ll_count = sum(1 for s in last_4 if s.type == SwingType.LL)
        
        if hh_count >= 2 and hl_count >= 2:
            direction = 'BULLISH'
            strength = min(1.0, (hh_count + hl_count) / 6)
            regime = 'TRENDING' if strength > 0.7 else 'RANGING'
        elif lh_count >= 2 and ll_count >= 2:
            direction = 'BEARISH'
            strength = min(1.0, (lh_count + ll_count) / 6)
            regime = 'TRENDING' if strength > 0.7 else 'RANGING'
        else:
            direction = 'NEUTRAL'
            strength = 0.5
            regime = 'RANGING'
        
        return {'direction': direction, 'strength': strength, 'regime': regime}
    
    def update_context(self, context: SMCContext) -> SMCContext:
        """Update SMCContext with structure data"""
        context.swings = self.swings
        context.bos_points = self.bos_points
        context.choch_points = self.choch_points
        context.amd_phase = self.amd_phase
        context.amd_confidence = self.amd_confidence
        
        structure = self._get_current_structure()
        context.current_structure = structure['direction']
        context.structure_strength = structure['strength']
        
        return context