# enhanced_patterns.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import talib
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class PatternType(Enum):
    HARMONIC = "harmonic"
    STRUCTURE = "structure"
    VOLUME = "volume"
    TREND = "trend"

@dataclass
class Pattern:
    """Represents a detected pattern with all metadata"""
    name: str
    direction: str  # 'BUY', 'SELL', or 'NEUTRAL'
    reliability: float  # 0-1 score
    timeframe: str
    volume_confirmation: bool
    confirmation_candles: int
    pattern_type: str
    formation_strength: float  # 0-1 score
    age_bars: int  # How many bars since pattern completed
    context_score: float  # 0-1 score for market context

class EnhancedPatternDetector:
    """
    Advanced pattern detection combining:
    - Harmonic patterns (Gartley, Bat, Crab, etc.)
    - Advanced structure patterns
    - Volume analysis
    """
    
    def __init__(self):
        # Fibonacci ratios for harmonic patterns
        self.fib_levels = {
            'retracement': [0.382, 0.5, 0.618, 0.786],
            'extension': [1.13, 1.27, 1.414, 1.618, 2.0, 2.24, 2.618, 3.14, 3.618]
        }
        
        # Pattern thresholds
        self.pattern_thresholds = {
            'gartley': 0.05,
            'bat': 0.05,
            'crab': 0.06,
            'butterfly': 0.06,
            'shark': 0.07,
            'cypher': 0.06
        }
        
        # Volume thresholds
        self.volume_ma_period = 20
        self.volume_spike_threshold = 1.5
        
    def detect_all_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Detects ALL pattern types and returns comprehensive list.
        """
        patterns = []
        if df is None or df.empty or len(df) < 20:
            return patterns

        # Add technical indicators if not present
        df = self._add_technical_indicators(df)
        
        # Detect all pattern types
        patterns.extend(self._detect_harmonic_patterns(df))
        patterns.extend(self._detect_structure_patterns(df))
        patterns.extend(self._detect_three_drives_pattern(df))
        patterns.extend(self._detect_adam_eve_pattern(df))
        patterns.extend(self._detect_quasimodo_pattern(df))
        
        # Filter duplicates and sort by reliability
        patterns = self._deduplicate_patterns(patterns)
        patterns.sort(key=lambda x: (x.reliability * x.context_score), reverse=True)
        
        return patterns
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add necessary technical indicators for pattern detection"""
        try:
            # Make a copy to avoid modifying original
            df = df.copy()
        # Ensure data types are float64 for TA-Lib
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                   df[col] = df[col].astype(np.float64)
            
            # RSI
            if 'rsi' not in df.columns:
                df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
            
            # Moving averages
            if 'sma_20' not in df.columns:
                df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
            if 'sma_50' not in df.columns:
                df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)
            if 'ema_12' not in df.columns:
                df['ema_12'] = talib.EMA(df['close'].values, timeperiod=12)
            if 'ema_26' not in df.columns:
                df['ema_26'] = talib.EMA(df['close'].values, timeperiod=26)
            
            # MACD
            if 'macd' not in df.columns:
                macd, signal, hist = talib.MACD(df['close'].values)
                df['macd'] = macd
                df['macd_signal'] = signal
                df['macd_hist'] = hist
            
            # Bollinger Bands
            if 'bb_upper' not in df.columns:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values)
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle
                df['bb_lower'] = bb_lower
            
            # ATR for volatility
            if 'atr' not in df.columns:
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # Volume indicators
            if 'volume_sma' not in df.columns and 'volume' in df.columns:
                df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=self.volume_ma_period)
            
            return df
            
        except Exception as e:
            print(f"Error adding indicators: {e}")
            return df
    
    def _find_swing_points(self, df: pd.DataFrame, window: int = 5) -> List[int]:
        """Find swing highs and lows using local extrema"""
        swings = []
        try:
            if len(df) < window * 2:
                return swings
            
            for i in range(window, len(df) - window):
                # Swing high
                if df['high'].iloc[i] == max(df['high'].iloc[i-window:i+window+1]):
                    swings.append(i)
                # Swing low
                elif df['low'].iloc[i] == min(df['low'].iloc[i-window:i+window+1]):
                    swings.append(i)
            
            return sorted(swings)
            
        except Exception as e:
            print(f"Error finding swing points: {e}")
            return swings
    
    def _detect_three_drives_pattern(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Three Drives Pattern - Harmonic pattern with 3 drives
        Each drive should be weaker (divergence)
        """
        patterns = []
        try:
            if len(df) < 30:
                return patterns
            
            # Find swing points
            swings = self._find_swing_points(df, window=5)
            if len(swings) < 6:
                return patterns
            
            # Need 6 points: X, A, B, C, D, E (3 drives)
            for i in range(len(swings) - 5):
                x_idx = swings[i]
                a_idx = swings[i+1]
                b_idx = swings[i+2]
                c_idx = swings[i+3]
                d_idx = swings[i+4]
                e_idx = swings[i+5]
                
                # Get prices
                x_price = float(df['low'].iloc[x_idx]) if x_idx < len(df) else 0
                a_price = float(df['high'].iloc[a_idx]) if a_idx < len(df) else 0
                b_price = float(df['low'].iloc[b_idx]) if b_idx < len(df) else 0
                c_price = float(df['high'].iloc[c_idx]) if c_idx < len(df) else 0
                d_price = float(df['low'].iloc[d_idx]) if d_idx < len(df) else 0
                e_price = float(df['close'].iloc[-1])  # Current price
                
                # Check pattern structure (alternating highs/lows)
                if a_price > x_price and b_price < a_price and c_price > b_price and d_price < c_price:
                    # Calculate Fibonacci ratios (ideally 1.272, 1.618)
                    drive1 = (a_price - x_price) / (b_price - x_price) if b_price != x_price else 0
                    drive2 = (c_price - b_price) / (d_price - b_price) if d_price != b_price else 0
                    
                    # Check for divergence (weakening momentum)
                    rsi_values = df['rsi'].values if 'rsi' in df else None
                    if rsi_values is not None and len(rsi_values) > e_idx:
                        rsi_x = rsi_values[x_idx]
                        rsi_a = rsi_values[a_idx]
                        rsi_b = rsi_values[b_idx]
                        rsi_c = rsi_values[c_idx]
                        rsi_d = rsi_values[d_idx]
                        
                        # Bearish divergence: price higher, RSI lower
                        if c_price > a_price and rsi_c < rsi_a:
                            patterns.append(Pattern(
                                name='Three Drives (Bearish)',
                                direction='SELL',
                                reliability=0.75,
                                timeframe=self._infer_timeframe(df),
                                volume_confirmation=False,
                                confirmation_candles=2,
                                pattern_type='HARMONIC',
                                formation_strength=0.8,
                                age_bars=len(df) - e_idx,
                                context_score=0.75
                            ))
                        
                        # Bullish divergence: price lower, RSI higher
                        elif d_price < b_price and rsi_d > rsi_b:
                            patterns.append(Pattern(
                                name='Three Drives (Bullish)',
                                direction='BUY',
                                reliability=0.75,
                                timeframe=self._infer_timeframe(df),
                                volume_confirmation=False,
                                confirmation_candles=2,
                                pattern_type='HARMONIC',
                                formation_strength=0.8,
                                age_bars=len(df) - e_idx,
                                context_score=0.75
                            ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting three drives: {e}")
            return patterns

    def _detect_adam_eve_pattern(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Adam & Eve Pattern - Double bottom with one sharp (Adam) and one rounded (Eve)
        """
        patterns = []
        try:
            if len(df) < 30:
                return patterns
            
            # Find bottoms
            from scipy.signal import argrelextrema
            lows = df['low'].values
            bottom_indices = argrelextrema(lows, np.less, order=5)[0]
            
            if len(bottom_indices) < 2:
                return patterns
            
            for i in range(len(bottom_indices) - 1):
                idx1 = bottom_indices[i]
                idx2 = bottom_indices[i+1]
                
                # Check if bottoms are within reasonable distance (10-30 bars)
                if idx2 - idx1 < 10 or idx2 - idx1 > 40:
                    continue
                
                # Get price levels
                bottom1 = lows[idx1]
                bottom2 = lows[idx2]
                
                # Check if bottoms are at similar level (±2%)
                if abs(bottom2 - bottom1) / bottom1 > 0.02:
                    continue
                
                # Analyze shape of each bottom
                # Adam: sharp V shape (few bars)
                # Eve: rounded U shape (more bars)
                
                # Adam detection (first bottom should be sharp)
                left_bars1 = 3
                right_bars1 = 3
                if idx1 > left_bars1 and idx1 < len(df) - right_bars1:
                    left_range1 = df['low'].iloc[idx1-left_bars1:idx1].min()
                    right_range1 = df['low'].iloc[idx1+1:idx1+right_bars1+1].min()
                    adam_sharp = (bottom1 <= left_range1 * 0.995) and (bottom1 <= right_range1 * 0.995)
                else:
                    adam_sharp = False
                
                # Eve detection (second bottom should be rounded)
                left_bars2 = 7
                right_bars2 = 7
                if idx2 > left_bars2 and idx2 < len(df) - right_bars2:
                    # Check for gradual U shape
                    left_slope = (df['low'].iloc[idx2-left_bars2] - bottom2) / left_bars2
                    right_slope = (df['low'].iloc[idx2+right_bars2] - bottom2) / right_bars2
                    eve_rounded = (abs(left_slope) < 0.001 and abs(right_slope) < 0.001)
                else:
                    eve_rounded = False
                
                # Also check opposite (Eve then Adam)
                adam_first = adam_sharp and eve_rounded
                eve_first = False
                
                # Check if it could be Eve then Adam
                if not adam_first and idx2 > left_bars2 and idx2 < len(df) - right_bars2:
                    # Check if first bottom is rounded
                    left_range2 = df['low'].iloc[idx2-left_bars2:idx2].min()
                    right_range2 = df['low'].iloc[idx2+1:idx2+right_bars2+1].min()
                    eve_rounded_first = (bottom2 <= left_range2 * 0.998) and (bottom2 <= right_range2 * 0.998)
                    
                    # Check if second bottom is sharp (would need another bottom)
                    if idx2 + 5 < len(lows):
                        next_bottom_idx = argrelextrema(lows[idx2+1:], np.less, order=3)[0]
                        if len(next_bottom_idx) > 0:
                            next_idx = idx2 + 1 + next_bottom_idx[0]
                            if next_idx - idx2 > 5 and next_idx - idx2 < 20:
                                next_bottom = lows[next_idx]
                                if abs(next_bottom - bottom2) / bottom2 < 0.02:
                                    # Check if sharp
                                    left_bars3 = 3
                                    if next_idx > left_bars3:
                                        left_range3 = df['low'].iloc[next_idx-left_bars3:next_idx].min()
                                        right_range3 = df['low'].iloc[next_idx+1:next_idx+left_bars3+1].min()
                                        if next_bottom <= left_range3 * 0.995 and next_bottom <= right_range3 * 0.995:
                                            eve_first = True
                
                if adam_first or eve_first:
                    patterns.append(Pattern(
                        name='Adam & Eve Double Bottom',
                        direction='BUY',
                        reliability=0.78,
                        timeframe=self._infer_timeframe(df),
                        volume_confirmation=True,
                        confirmation_candles=3,
                        pattern_type='STRUCTURE',
                        formation_strength=0.8,
                        age_bars=len(df) - idx2,
                        context_score=0.75
                    ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting Adam & Eve: {e}")
            return patterns

    def _detect_quasimodo_pattern(self, df: pd.DataFrame) -> List[Pattern]:
        """
        Quasimodo Pattern (Over and Under)
        Structure: Higher high → Lower low → Reversal
        """
        patterns = []
        try:
            if len(df) < 30:
                return patterns
            
            # Find swing points
            swings = self._find_swing_points(df, window=3)
            if len(swings) < 4:
                return patterns
            
            for i in range(len(swings) - 3):
                idx1 = swings[i]
                idx2 = swings[i+1]
                idx3 = swings[i+2]
                idx4 = swings[i+3]
                
                # Get price values
                price1 = float(df['high'].iloc[idx1]) if idx1 < len(df) else 0
                price2 = float(df['low'].iloc[idx2]) if idx2 < len(df) else 0
                price3 = float(df['high'].iloc[idx3]) if idx3 < len(df) else 0
                price4 = float(df['low'].iloc[idx4]) if idx4 < len(df) else 0
                
                # Bullish Quasimodo: Higher high, lower low, then reversal
                if price3 > price1 and price4 < price2:
                    # Check if current price is above the structure
                    current_price = float(df['close'].iloc[-1])
                    if current_price > price3:
                        patterns.append(Pattern(
                            name='Quasimodo (Bullish)',
                            direction='BUY',
                            reliability=0.72,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=True,
                            confirmation_candles=3,
                            pattern_type='STRUCTURE',
                            formation_strength=0.75,
                            age_bars=len(df) - idx4,
                            context_score=0.7
                        ))
                
                # Bearish Quasimodo: Lower low, higher high, then reversal
                elif price3 < price1 and price4 > price2:
                    current_price = float(df['close'].iloc[-1])
                    if current_price < price4:
                        patterns.append(Pattern(
                            name='Quasimodo (Bearish)',
                            direction='SELL',
                            reliability=0.72,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=True,
                            confirmation_candles=3,
                            pattern_type='STRUCTURE',
                            formation_strength=0.75,
                            age_bars=len(df) - idx4,
                            context_score=0.7
                        ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting Quasimodo: {e}")
            return patterns
    
    def _detect_harmonic_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect harmonic patterns like Gartley, Bat, Crab, etc."""
        patterns = []
        try:
            if len(df) < 50:  # Need enough data
                return patterns
            
            # Find swing points for harmonic patterns
            swings = self._find_swing_points(df, window=5)
            if len(swings) < 5:
                return patterns
            
            # Check for XABCD patterns
            for i in range(len(swings) - 4):
                x_idx = swings[i]
                a_idx = swings[i+1]
                b_idx = swings[i+2]
                c_idx = swings[i+3]
                d_idx = swings[i+4]
                
                # Get prices
                x_price = df['low'].iloc[x_idx] if df['low'].iloc[x_idx] < df['low'].iloc[a_idx] else df['high'].iloc[x_idx]
                a_price = df['high'].iloc[a_idx] if df['high'].iloc[a_idx] > df['high'].iloc[x_idx] else df['low'].iloc[a_idx]
                b_price = df['low'].iloc[b_idx] if df['low'].iloc[b_idx] < df['low'].iloc[a_idx] else df['high'].iloc[b_idx]
                c_price = df['high'].iloc[c_idx] if df['high'].iloc[c_idx] > df['high'].iloc[b_idx] else df['low'].iloc[c_idx]
                d_price = df['close'].iloc[-1]  # Current price as potential completion
                
                # Calculate retracements
                ab_retrace = abs((a_price - b_price) / (a_price - x_price)) if abs(a_price - x_price) > 0 else 0
                bc_retrace = abs((b_price - c_price) / (a_price - b_price)) if abs(a_price - b_price) > 0 else 0
                
                # Check for Gartley pattern
                if 0.618 - self.pattern_thresholds['gartley'] <= ab_retrace <= 0.618 + self.pattern_thresholds['gartley']:
                    if 0.382 - self.pattern_thresholds['gartley'] <= bc_retrace <= 0.886 + self.pattern_thresholds['gartley']:
                        direction = 'BUY' if d_price < c_price else 'SELL'
                        patterns.append(Pattern(
                            name='Gartley Pattern',
                            direction=direction,
                            reliability=0.75,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=False,
                            confirmation_candles=1,
                            pattern_type='HARMONIC',
                            formation_strength=0.7,
                            age_bars=len(df) - d_idx,
                            context_score=0.7
                        ))
                
                # Check for Bat pattern
                if 0.382 - self.pattern_thresholds['bat'] <= ab_retrace <= 0.5 + self.pattern_thresholds['bat']:
                    if 0.382 - self.pattern_thresholds['bat'] <= bc_retrace <= 0.886 + self.pattern_thresholds['bat']:
                        direction = 'BUY' if d_price < c_price else 'SELL'
                        patterns.append(Pattern(
                            name='Bat Pattern',
                            direction=direction,
                            reliability=0.8,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=False,
                            confirmation_candles=1,
                            pattern_type='HARMONIC',
                            formation_strength=0.75,
                            age_bars=len(df) - d_idx,
                            context_score=0.75
                        ))
                
                # Check for Crab pattern
                if 0.382 - self.pattern_thresholds['crab'] <= ab_retrace <= 0.618 + self.pattern_thresholds['crab']:
                    if 0.382 - self.pattern_thresholds['crab'] <= bc_retrace <= 0.886 + self.pattern_thresholds['crab']:
                        direction = 'BUY' if d_price < c_price else 'SELL'
                        patterns.append(Pattern(
                            name='Crab Pattern',
                            direction=direction,
                            reliability=0.85,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=False,
                            confirmation_candles=1,
                            pattern_type='HARMONIC',
                            formation_strength=0.8,
                            age_bars=len(df) - d_idx,
                            context_score=0.8
                        ))
                
                # Check for Butterfly pattern
                if 0.786 - self.pattern_thresholds['butterfly'] <= ab_retrace <= 0.786 + self.pattern_thresholds['butterfly']:
                    if 0.382 - self.pattern_thresholds['butterfly'] <= bc_retrace <= 0.886 + self.pattern_thresholds['butterfly']:
                        direction = 'BUY' if d_price < c_price else 'SELL'
                        patterns.append(Pattern(
                            name='Butterfly Pattern',
                            direction=direction,
                            reliability=0.82,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=False,
                            confirmation_candles=1,
                            pattern_type='HARMONIC',
                            formation_strength=0.78,
                            age_bars=len(df) - d_idx,
                            context_score=0.78
                        ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting harmonic patterns: {e}")
            return patterns
    
    def _detect_structure_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect advanced structure patterns"""
        patterns = []
        try:
            if len(df) < 30:
                return patterns
            
            # Find swing points
            swings = self._find_swing_points(df, window=5)
            if len(swings) < 4:
                return patterns
            
            # Detect Head & Shoulders
            patterns.extend(self._detect_head_shoulders(df, swings))
            
            # Detect Double/Triple Tops/Bottoms
            patterns.extend(self._detect_multiple_tops_bottoms(df, swings))
            
            # Detect Wedges
            patterns.extend(self._detect_wedges(df))
            
            # Detect Triangles
            patterns.extend(self._detect_triangles(df))
            
            # Detect Flags & Pennants
            patterns.extend(self._detect_flags_pennants(df))
            
            # Detect Cup & Handle
            patterns.extend(self._detect_cup_handle(df))
            
            # Detect divergence using RSI
            patterns.extend(self._detect_divergence(df, swings))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting structure patterns: {e}")
            return patterns
    
    def _detect_head_shoulders(self, df: pd.DataFrame, swings: List[int]) -> List[Pattern]:
        """Detect Head & Shoulders pattern"""
        patterns = []
        try:
            if len(swings) < 5:
                return patterns
            
            for i in range(len(swings) - 4):
                left_shoulder_idx = swings[i]
                head_idx = swings[i+1]
                right_shoulder_idx = swings[i+2]
                neckline_idx1 = swings[i+3]
                neckline_idx2 = swings[i+4]
                
                # Get highs for potential head & shoulders
                left_high = df['high'].iloc[left_shoulder_idx]
                head_high = df['high'].iloc[head_idx]
                right_high = df['high'].iloc[right_shoulder_idx]
                
                # Check if head is highest
                if head_high > left_high and head_high > right_high:
                    # Check if shoulders are roughly equal
                    shoulder_diff = abs(left_high - right_high) / max(left_high, right_high)
                    
                    if shoulder_diff < 0.03:  # Shoulders within 3%
                        # Check neckline (downward sloping)
                        neckline_price1 = df['low'].iloc[neckline_idx1]
                        neckline_price2 = df['low'].iloc[neckline_idx2]
                        
                        if neckline_price2 < neckline_price1:  # Descending neckline
                            current_price = df['close'].iloc[-1]
                            
                            # If price breaks below neckline
                            if current_price < neckline_price2:
                                patterns.append(Pattern(
                                    name='Head & Shoulders',
                                    direction='SELL',
                                    reliability=0.75,
                                    timeframe=self._infer_timeframe(df),
                                    volume_confirmation=True,
                                    confirmation_candles=2,
                                    pattern_type='STRUCTURE',
                                    formation_strength=0.8,
                                    age_bars=len(df) - right_shoulder_idx,
                                    context_score=0.75
                                ))
            
            # Inverse Head & Shoulders (for bottoms)
            for i in range(len(swings) - 4):
                left_shoulder_idx = swings[i]
                head_idx = swings[i+1]
                right_shoulder_idx = swings[i+2]
                neckline_idx1 = swings[i+3]
                neckline_idx2 = swings[i+4]
                
                # Get lows for inverse head & shoulders
                left_low = df['low'].iloc[left_shoulder_idx]
                head_low = df['low'].iloc[head_idx]
                right_low = df['low'].iloc[right_shoulder_idx]
                
                # Check if head is lowest
                if head_low < left_low and head_low < right_low:
                    # Check if shoulders are roughly equal
                    shoulder_diff = abs(left_low - right_low) / max(left_low, right_low)
                    
                    if shoulder_diff < 0.03:  # Shoulders within 3%
                        # Check neckline (upward sloping)
                        neckline_price1 = df['high'].iloc[neckline_idx1]
                        neckline_price2 = df['high'].iloc[neckline_idx2]
                        
                        if neckline_price2 > neckline_price1:  # Ascending neckline
                            current_price = df['close'].iloc[-1]
                            
                            # If price breaks above neckline
                            if current_price > neckline_price2:
                                patterns.append(Pattern(
                                    name='Inverse Head & Shoulders',
                                    direction='BUY',
                                    reliability=0.75,
                                    timeframe=self._infer_timeframe(df),
                                    volume_confirmation=True,
                                    confirmation_candles=2,
                                    pattern_type='STRUCTURE',
                                    formation_strength=0.8,
                                    age_bars=len(df) - right_shoulder_idx,
                                    context_score=0.75
                                ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting head & shoulders: {e}")
            return patterns
    
    def _detect_multiple_tops_bottoms(self, df: pd.DataFrame, swings: List[int]) -> List[Pattern]:
        """Detect Double/Triple Top/Bottom patterns"""
        patterns = []
        try:
            if len(swings) < 4:
                return patterns
            
            # Get high and low points
            high_indices = [i for i in swings if df['high'].iloc[i] > df['low'].iloc[i]]
            low_indices = [i for i in swings if df['low'].iloc[i] < df['high'].iloc[i]]
            
            # Double Top
            if len(high_indices) >= 2:
                idx1 = high_indices[-2]
                idx2 = high_indices[-1]
                
                high1 = df['high'].iloc[idx1]
                high2 = df['high'].iloc[idx2]
                
                # Check if highs are similar
                high_diff = abs(high1 - high2) / max(high1, high2)
                
                if high_diff < 0.02:  # Within 2%
                    # Check if there's a valley between them
                    between_lows = df['low'].iloc[idx1:idx2].min()
                    valley_depth = (high1 + high2) / 2 - between_lows
                    
                    if valley_depth > 0:
                        current_price = df['close'].iloc[-1]
                        
                        # If price breaks below valley
                        if current_price < between_lows:
                            patterns.append(Pattern(
                                name='Double Top',
                                direction='SELL',
                                reliability=0.7,
                                timeframe=self._infer_timeframe(df),
                                volume_confirmation=True,
                                confirmation_candles=2,
                                pattern_type='STRUCTURE',
                                formation_strength=0.75,
                                age_bars=len(df) - idx2,
                                context_score=0.7
                            ))
            
            # Double Bottom
            if len(low_indices) >= 2:
                idx1 = low_indices[-2]
                idx2 = low_indices[-1]
                
                low1 = df['low'].iloc[idx1]
                low2 = df['low'].iloc[idx2]
                
                # Check if lows are similar
                low_diff = abs(low1 - low2) / max(low1, low2)
                
                if low_diff < 0.02:  # Within 2%
                    # Check if there's a peak between them
                    between_highs = df['high'].iloc[idx1:idx2].max()
                    peak_height = between_highs - (low1 + low2) / 2
                    
                    if peak_height > 0:
                        current_price = df['close'].iloc[-1]
                        
                        # If price breaks above peak
                        if current_price > between_highs:
                            patterns.append(Pattern(
                                name='Double Bottom',
                                direction='BUY',
                                reliability=0.7,
                                timeframe=self._infer_timeframe(df),
                                volume_confirmation=True,
                                confirmation_candles=2,
                                pattern_type='STRUCTURE',
                                formation_strength=0.75,
                                age_bars=len(df) - idx2,
                                context_score=0.7
                            ))
            
            # Triple Top
            if len(high_indices) >= 3:
                idx1 = high_indices[-3]
                idx2 = high_indices[-2]
                idx3 = high_indices[-1]
                
                high1 = df['high'].iloc[idx1]
                high2 = df['high'].iloc[idx2]
                high3 = df['high'].iloc[idx3]
                
                # Check if all highs are similar
                high_avg = (high1 + high2 + high3) / 3
                high_diffs = [abs(h - high_avg) / high_avg for h in [high1, high2, high3]]
                
                if all(diff < 0.02 for diff in high_diffs):  # All within 2%
                    # Find the lowest point between them
                    between_lows = df['low'].iloc[idx1:idx3].min()
                    
                    current_price = df['close'].iloc[-1]
                    
                    # If price breaks below support
                    if current_price < between_lows:
                        patterns.append(Pattern(
                            name='Triple Top',
                            direction='SELL',
                            reliability=0.75,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=True,
                            confirmation_candles=3,
                            pattern_type='STRUCTURE',
                            formation_strength=0.8,
                            age_bars=len(df) - idx3,
                            context_score=0.75
                        ))
            
            # Triple Bottom
            if len(low_indices) >= 3:
                idx1 = low_indices[-3]
                idx2 = low_indices[-2]
                idx3 = low_indices[-1]
                
                low1 = df['low'].iloc[idx1]
                low2 = df['low'].iloc[idx2]
                low3 = df['low'].iloc[idx3]
                
                # Check if all lows are similar
                low_avg = (low1 + low2 + low3) / 3
                low_diffs = [abs(l - low_avg) / low_avg for l in [low1, low2, low3]]
                
                if all(diff < 0.02 for diff in low_diffs):  # All within 2%
                    # Find the highest point between them
                    between_highs = df['high'].iloc[idx1:idx3].max()
                    
                    current_price = df['close'].iloc[-1]
                    
                    # If price breaks above resistance
                    if current_price > between_highs:
                        patterns.append(Pattern(
                            name='Triple Bottom',
                            direction='BUY',
                            reliability=0.75,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=True,
                            confirmation_candles=3,
                            pattern_type='STRUCTURE',
                            formation_strength=0.8,
                            age_bars=len(df) - idx3,
                            context_score=0.75
                        ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting multiple tops/bottoms: {e}")
            return patterns
    
    def _detect_wedges(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect Rising & Falling Wedges"""
        patterns = []
        try:
            if len(df) < 20:
                return patterns
            
            # Get last 20 bars
            recent_high = df['high'].iloc[-20:].values
            recent_low = df['low'].iloc[-20:].values
            x = np.arange(len(recent_high))
            
            # Fit trendlines
            high_slope, high_intercept = np.polyfit(x, recent_high, 1)
            low_slope, low_intercept = np.polyfit(x, recent_low, 1)
            
            # Calculate convergence
            high_end = high_slope * len(x) + high_intercept
            low_end = low_slope * len(x) + low_intercept
            
            high_start = high_intercept
            low_start = low_intercept
            
            # Rising Wedge (downward sloping highs, upward sloping lows - converging)
            if high_slope < 0 and low_slope > 0:
                # Check if they're converging
                start_width = high_start - low_start
                end_width = high_end - low_end
                
                if end_width < start_width * 0.7:  # Converging
                    patterns.append(Pattern(
                        name='Rising Wedge',
                        direction='SELL',
                        reliability=0.68,
                        timeframe=self._infer_timeframe(df),
                        volume_confirmation=True,
                        confirmation_candles=2,
                        pattern_type='STRUCTURE',
                        formation_strength=0.7,
                        age_bars=0,
                        context_score=0.7
                    ))
            
            # Falling Wedge (upward sloping highs, downward sloping lows - converging)
            elif high_slope > 0 and low_slope < 0:
                start_width = high_start - low_start
                end_width = high_end - low_end
                
                if end_width < start_width * 0.7:  # Converging
                    patterns.append(Pattern(
                        name='Falling Wedge',
                        direction='BUY',
                        reliability=0.68,
                        timeframe=self._infer_timeframe(df),
                        volume_confirmation=True,
                        confirmation_candles=2,
                        pattern_type='STRUCTURE',
                        formation_strength=0.7,
                        age_bars=0,
                        context_score=0.7
                    ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting wedges: {e}")
            return patterns
    
    def _detect_triangles(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect Symmetrical, Ascending, Descending Triangles"""
        patterns = []
        try:
            if len(df) < 20:
                return patterns
            
            # Get last 20 bars
            recent_high = df['high'].iloc[-20:].values
            recent_low = df['low'].iloc[-20:].values
            x = np.arange(len(recent_high))
            
            # Fit trendlines
            high_slope, high_intercept = np.polyfit(x, recent_high, 1)
            low_slope, low_intercept = np.polyfit(x, recent_low, 1)
            
            # Calculate convergence
            high_end = high_slope * len(x) + high_intercept
            low_end = low_slope * len(x) + low_intercept
            
            high_start = high_intercept
            low_start = low_intercept
            
            start_width = high_start - low_start
            end_width = high_end - low_end
            
            # Symmetrical Triangle (converging, slopes opposite direction)
            if abs(high_slope) < 0.01 and abs(low_slope) < 0.01 and end_width < start_width * 0.7:
                patterns.append(Pattern(
                    name='Symmetrical Triangle',
                    direction='NEUTRAL',
                    reliability=0.65,
                    timeframe=self._infer_timeframe(df),
                    volume_confirmation=True,
                    confirmation_candles=2,
                    pattern_type='STRUCTURE',
                    formation_strength=0.7,
                    age_bars=0,
                    context_score=0.65
                ))
            
            # Ascending Triangle (flat highs, rising lows)
            elif abs(high_slope) < 0.01 and low_slope > 0.01:
                patterns.append(Pattern(
                    name='Ascending Triangle',
                    direction='BUY',
                    reliability=0.7,
                    timeframe=self._infer_timeframe(df),
                    volume_confirmation=True,
                    confirmation_candles=2,
                    pattern_type='STRUCTURE',
                    formation_strength=0.75,
                    age_bars=0,
                    context_score=0.7
                ))
            
            # Descending Triangle (falling highs, flat lows)
            elif high_slope < -0.01 and abs(low_slope) < 0.01:
                patterns.append(Pattern(
                    name='Descending Triangle',
                    direction='SELL',
                    reliability=0.7,
                    timeframe=self._infer_timeframe(df),
                    volume_confirmation=True,
                    confirmation_candles=2,
                    pattern_type='STRUCTURE',
                    formation_strength=0.75,
                    age_bars=0,
                    context_score=0.7
                ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting triangles: {e}")
            return patterns
    
    def _detect_flags_pennants(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect Flag & Pennant patterns"""
        patterns = []
        try:
            if len(df) < 20:
                return patterns
            
            # Look for sharp move followed by consolidation
            recent = df.iloc[-15:]
            
            # Calculate price change over last 5 bars (potential flagpole)
            start_price = recent['close'].iloc[0]
            end_price = recent['close'].iloc[4] if len(recent) > 4 else recent['close'].iloc[-1]
            
            move_pct = abs(end_price - start_price) / start_price
            
            # If sharp move (>3%)
            if move_pct > 0.03:
                # Check for consolidation in last 10 bars
                consolidation = recent.iloc[5:]
                
                if len(consolidation) > 3:
                    high = consolidation['high'].max()
                    low = consolidation['low'].min()
                    consolidation_range = (high - low) / start_price
                    
                    # Tight consolidation (<2% range)
                    if consolidation_range < 0.02:
                        direction = 'BUY' if end_price > start_price else 'SELL'
                        
                        patterns.append(Pattern(
                            name='Flag',
                            direction=direction,
                            reliability=0.72,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=True,
                            confirmation_candles=2,
                            pattern_type='STRUCTURE',
                            formation_strength=0.75,
                            age_bars=0,
                            context_score=0.72
                        ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting flags/pennants: {e}")
            return patterns
    
    def _detect_cup_handle(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect Cup & Handle pattern"""
        patterns = []
        try:
            if len(df) < 30:
                return patterns
            
            # Look for U-shaped recovery
            recent = df.iloc[-30:]
            
            # Find lowest point in last 30 bars
            min_idx = recent['low'].idxmin()
            min_pos = recent.index.get_loc(min_idx)
            
            if min_pos > 5 and min_pos < 25:  # Cup bottom not at edges
                # Left side (before bottom)
                left_high = recent['high'].iloc[:min_pos].max()
                
                # Right side (after bottom)
                right_high = recent['high'].iloc[min_pos:].max()
                
                # Cup rims should be similar
                rim_diff = abs(left_high - right_high) / max(left_high, right_high)
                
                if rim_diff < 0.05:  # Rims within 5%
                    # Cup depth
                    cup_depth = (left_high + right_high) / 2 - recent['low'].iloc[min_pos]
                    
                    # Check for handle (small pullback on right side)
                    if len(recent) > min_pos + 5:
                        handle_high = recent['high'].iloc[min_pos+5:].max()
                        handle_low = recent['low'].iloc[min_pos+5:].min()
                        handle_depth = handle_high - handle_low
                        
                        # Handle should be smaller than cup
                        if handle_depth < cup_depth * 0.4:
                            current_price = recent['close'].iloc[-1]
                            
                            # If price near right rim
                            if current_price > right_high * 0.98:
                                patterns.append(Pattern(
                                    name='Cup & Handle',
                                    direction='BUY',
                                    reliability=0.78,
                                    timeframe=self._infer_timeframe(df),
                                    volume_confirmation=True,
                                    confirmation_candles=3,
                                    pattern_type='STRUCTURE',
                                    formation_strength=0.8,
                                    age_bars=0,
                                    context_score=0.78
                                ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting cup & handle: {e}")
            return patterns
    
    def _detect_divergence(self, df: pd.DataFrame, swings: List[int]) -> List[Pattern]:
        """Detect RSI divergence patterns"""
        patterns = []
        try:
            if 'rsi' not in df.columns or len(swings) < 2:
                return patterns
            
            # Bullish divergence (price lower low, RSI higher low)
            for i in range(len(swings) - 1):
                idx1 = swings[i]
                idx2 = swings[i+1]
                
                # Check if both are lows
                if df['low'].iloc[idx2] < df['low'].iloc[idx1]:  # Lower low
                    if df['rsi'].iloc[idx2] > df['rsi'].iloc[idx1]:  # Higher RSI
                        patterns.append(Pattern(
                            name='Bullish Divergence',
                            direction='BUY',
                            reliability=0.8,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=True,
                            confirmation_candles=2,
                            pattern_type='STRUCTURE',
                            formation_strength=0.8,
                            age_bars=len(df) - idx2,
                            context_score=0.8
                        ))
                
                # Bearish divergence (price higher high, RSI lower high)
                if df['high'].iloc[idx2] > df['high'].iloc[idx1]:  # Higher high
                    if df['rsi'].iloc[idx2] < df['rsi'].iloc[idx1]:  # Lower RSI
                        patterns.append(Pattern(
                            name='Bearish Divergence',
                            direction='SELL',
                            reliability=0.8,
                            timeframe=self._infer_timeframe(df),
                            volume_confirmation=True,
                            confirmation_candles=2,
                            pattern_type='STRUCTURE',
                            formation_strength=0.8,
                            age_bars=len(df) - idx2,
                            context_score=0.8
                        ))
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting divergence: {e}")
            return patterns
    
    def _infer_timeframe(self, df: pd.DataFrame) -> str:
        """Infer timeframe from dataframe index or length"""
        try:
            # Try to infer from datetime index if available
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                time_diff = df.index[-1] - df.index[-2]
                seconds = time_diff.total_seconds()
                
                if seconds < 60:
                    return f"{int(seconds)}s"
                elif seconds < 3600:
                    return f"{int(seconds/60)}m"
                elif seconds < 86400:
                    return f"{int(seconds/3600)}h"
                else:
                    return f"{int(seconds/86400)}d"
            
            # Default based on length
            if len(df) > 1000:
                return "1h"
            elif len(df) > 500:
                return "15m"
            elif len(df) > 200:
                return "5m"
            else:
                return "1m"
                
        except:
            return "unknown"
    
    def _deduplicate_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Remove duplicate patterns keeping the highest reliability"""
        unique_patterns = {}
        
        for pattern in patterns:
            key = f"{pattern.name}_{pattern.direction}"
            if key not in unique_patterns or \
               (pattern.reliability * pattern.context_score) > \
               (unique_patterns[key].reliability * unique_patterns[key].context_score):
                unique_patterns[key] = pattern
        
        return list(unique_patterns.values())
    
    def get_pattern_summary(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """Get summary statistics of detected patterns"""
        if not patterns:
            return {
                'total_patterns': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'neutral_signals': 0,
                'best_pattern': None,
                'pattern_types': {},
                'directions': {}
            }
        
        buy_count = sum(1 for p in patterns if p.direction == 'BUY')
        sell_count = sum(1 for p in patterns if p.direction == 'SELL')
        neutral_count = sum(1 for p in patterns if p.direction == 'NEUTRAL')
        
        # Find best pattern
        best_pattern = max(patterns, key=lambda x: x.reliability * x.context_score)
        
        # Count by type
        pattern_types = {}
        for p in patterns:
            pattern_types[p.pattern_type] = pattern_types.get(p.pattern_type, 0) + 1
        
        # Count by direction
        directions = {
            'BUY': buy_count,
            'SELL': sell_count,
            'NEUTRAL': neutral_count
        }
        
        return {
            'total_patterns': len(patterns),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'neutral_signals': neutral_count,
            'best_pattern': best_pattern,
            'pattern_types': pattern_types,
            'directions': directions
        }

    def get_pattern_score(self, df: pd.DataFrame, direction: str = 'BUY') -> float:
        """
        Get aggregate pattern score for a specific direction (0-1)
        
        Args:
            df: DataFrame with OHLCV data
            direction: 'BUY' or 'SELL'
        
        Returns:
            Score between 0 and 1
        """
        patterns = self.detect_all_patterns(df)
        
        if not patterns:
            return 0.5
        
        if direction == 'BUY':
            bullish = [p for p in patterns if p.direction == 'BUY']
            if bullish:
                # Weighted by reliability and context score
                weighted_reliability = sum(p.reliability * p.context_score for p in bullish)
                avg_reliability = weighted_reliability / len(bullish)
                # Bonus for multiple patterns
                count_bonus = min(0.2, len(bullish) * 0.05)
                return min(1.0, avg_reliability * 1.2 + count_bonus)
            return 0.3
        else:
            bearish = [p for p in patterns if p.direction == 'SELL']
            if bearish:
                weighted_reliability = sum(p.reliability * p.context_score for p in bearish)
                avg_reliability = weighted_reliability / len(bearish)
                count_bonus = min(0.2, len(bearish) * 0.05)
                return min(1.0, avg_reliability * 1.2 + count_bonus)
            return 0.3

    def has_strong_pattern(self, df: pd.DataFrame, min_reliability: float = 0.75) -> Dict[str, Any]:
        """
        Check if there are strong patterns (reliability >= min_reliability)
        
        Args:
            df: DataFrame with OHLCV data
            min_reliability: Minimum reliability threshold for strong pattern
        
        Returns:
            Dictionary with strong pattern information
        """
        patterns = self.detect_all_patterns(df)
        
        strong_buy = [p for p in patterns if p.direction == 'BUY' and p.reliability >= min_reliability]
        strong_sell = [p for p in patterns if p.direction == 'SELL' and p.reliability >= min_reliability]
        
        return {
            'has_strong_buy': len(strong_buy) > 0,
            'has_strong_sell': len(strong_sell) > 0,
            'strong_buy_count': len(strong_buy),
            'strong_sell_count': len(strong_sell),
            'strong_buy_patterns': [p.name for p in strong_buy],
            'strong_sell_patterns': [p.name for p in strong_sell],
            'best_buy': max(strong_buy, key=lambda x: x.reliability).name if strong_buy else None,
            'best_sell': max(strong_sell, key=lambda x: x.reliability).name if strong_sell else None,
            'best_buy_reliability': max([p.reliability for p in strong_buy]) if strong_buy else 0,
            'best_sell_reliability': max([p.reliability for p in strong_sell]) if strong_sell else 0
        }

    def get_net_bias(self, df: pd.DataFrame, regime: Optional[Dict] = None) -> float:
        """
        Get net pattern bias (-1 to 1)
        Positive = bullish bias, Negative = bearish bias
        
        Args:
            df: DataFrame with OHLCV data
            regime: Optional regime dictionary for adjustment
        
        Returns:
            Net bias score between -1 and 1
        """
        patterns = self.detect_all_patterns(df)
        
        if not patterns:
            return 0.0
        
        # Calculate weighted directional sum
        buy_score = 0.0
        sell_score = 0.0
        
        for p in patterns:
            weighted_score = p.reliability * p.context_score
            if p.direction == 'BUY':
                buy_score += weighted_score
            elif p.direction == 'SELL':
                sell_score += weighted_score
        
        total = buy_score + sell_score
        if total == 0:
            return 0.0
        
        net_bias = (buy_score - sell_score) / total
        
        # Apply regime adjustment if provided
        if regime:
            regime_name = regime.get('regime', 'UNKNOWN')
            if 'BULL' in regime_name:
                # In bull market, bullish patterns carry more weight
                net_bias = min(1.0, net_bias * 1.2)
            elif 'BEAR' in regime_name:
                # In bear market, bearish patterns carry more weight
                net_bias = max(-1.0, net_bias * 1.2)
        
        return round(net_bias, 3)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(200) * 0.5)
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(200) * 0.5)
    
    # Initialize detector
    detector = EnhancedPatternDetector()
    
    # Detect all patterns
    patterns = detector.detect_all_patterns(df)
    
    # Print summary
    summary = detector.get_pattern_summary(patterns)
    print(f"Total patterns detected: {summary['total_patterns']}")
    print(f"Buy signals: {summary['buy_signals']}")
    print(f"Sell signals: {summary['sell_signals']}")
    print(f"Pattern types: {summary['pattern_types']}")
    
    # Print detailed patterns
    for pattern in patterns:
        print(f"\nPattern: {pattern.name}")
        print(f"  Direction: {pattern.direction}")
        print(f"  Reliability: {pattern.reliability:.2f}")
        print(f"  Type: {pattern.pattern_type}")
        print(f"  Context Score: {pattern.context_score:.2f}")
    
    # Test new methods
    print("\n" + "="*50)
    print("Testing new methods:")
    print("="*50)
    
    buy_score = detector.get_pattern_score(df, 'BUY')
    sell_score = detector.get_pattern_score(df, 'SELL')
    print(f"BUY Pattern Score: {buy_score:.3f}")
    print(f"SELL Pattern Score: {sell_score:.3f}")
    
    strong_patterns = detector.has_strong_pattern(df, min_reliability=0.75)
    print(f"Strong Patterns: {strong_patterns}")
    
    net_bias = detector.get_net_bias(df)
    print(f"Net Pattern Bias: {net_bias:.3f}")