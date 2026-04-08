# wolfe_waves.py
"""
Wolfe Wave Pattern Detection
5-wave structure with predictive targeting
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from logger import log

@dataclass
class WolfeWave:
    """Wolfe Wave pattern data"""
    wave_points: List[Tuple[int, float]]  # [(index, price), ...]
    entry_price: float
    target_price: float
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    timestamp: pd.Timestamp

class WolfeWaveDetector:
    """
    Detects Wolfe Wave patterns (5-wave predictive structure)
    Points: 1-2-3-4-5, with point 5 being entry
    Target is along line from point 1 to 4
    """
    
    def __init__(self):
        self.min_wave_height = 0.02  # 2% minimum wave movement
        self.max_wave_length = 50    # Maximum bars for complete pattern
    
    def detect_wolfe_waves(self, df: pd.DataFrame) -> List[WolfeWave]:
        """
        Detect Wolfe Wave patterns in the dataframe
        """
        waves = []
        try:
            if df is None or df.empty or len(df) < 50:
                return waves
            
            # Find swing points
            swings = self._find_swing_points(df)
            if len(swings) < 5:
                return waves
            
            # Look for 5-wave patterns
            for i in range(len(swings) - 4):
                points = swings[i:i+5]
                
                # Check if this forms a Wolfe Wave
                wave = self._validate_wolfe_wave(df, points)
                if wave:
                    waves.append(wave)
            
            return waves
            
        except Exception as e:
            log.error(f"Error detecting Wolfe Waves: {e}")
            return waves
    
    def _find_swing_points(self, df: pd.DataFrame, window: int = 3) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows"""
        swings = []
        try:
            highs = df['high'].values
            lows = df['low'].values
            
            for i in range(window, len(df) - window):
                # Swing high
                if highs[i] == max(highs[i-window:i+window+1]):
                    swings.append((i, float(highs[i]), 'HIGH'))
                
                # Swing low
                if lows[i] == min(lows[i-window:i+window+1]):
                    swings.append((i, float(lows[i]), 'LOW'))
            
            # Sort by index
            swings.sort(key=lambda x: x[0])
            
        except Exception as e:
            log.error(f"Error finding swings: {e}")
        
        return swings
    
    def _validate_wolfe_wave(self, df: pd.DataFrame, points: List[Tuple[int, float, str]]) -> Optional[WolfeWave]:
        """
        Validate if 5 points form a valid Wolfe Wave
        Rules:
        1. Alternating pattern: HIGH-LOW-HIGH-LOW-HIGH or LOW-HIGH-LOW-HIGH-LOW
        2. Wave 3 > Wave 1 (for bullish) or Wave 3 < Wave 1 (for bearish)
        3. Wave 4 > Wave 2 (for bullish) or Wave 4 < Wave 2 (for bearish)
        4. Wave 5 should be the entry point
        """
        try:
            idx1, price1, type1 = points[0]
            idx2, price2, type2 = points[1]
            idx3, price3, type3 = points[2]
            idx4, price4, type4 = points[3]
            idx5, price5, type5 = points[4]
            
            # Check alternating pattern
            types = [type1, type2, type3, type4, type5]
            expected_bullish = ['LOW', 'HIGH', 'LOW', 'HIGH', 'LOW']
            expected_bearish = ['HIGH', 'LOW', 'HIGH', 'LOW', 'HIGH']
            
            is_bullish = types == expected_bullish
            is_bearish = types == expected_bearish
            
            if not (is_bullish or is_bearish):
                return None
            
            # Check wave relationships
            if is_bullish:
                # Wave 3 should be higher than Wave 1
                if price3 <= price1:
                    return None
                # Wave 4 should be higher than Wave 2
                if price4 <= price2:
                    return None
                # Wave 5 should be lower than Wave 3
                if price5 >= price3:
                    return None
            else:  # bearish
                if price3 >= price1:
                    return None
                if price4 >= price2:
                    return None
                if price5 <= price3:
                    return None
            
            # Calculate target (line from point 1 to 4)
            # Target is where price is expected to go after point 5
            slope = (price4 - price1) / (idx4 - idx1) if idx4 != idx1 else 0
            target_price = price4 + slope * (len(df) - idx4)
            
            # Calculate confidence based on wave symmetry
            wave1_height = abs(price2 - price1)
            wave3_height = abs(price4 - price3)
            symmetry = 1.0 - min(abs(wave3_height - wave1_height) / wave1_height, 1.0)
            
            confidence = 0.6 + (symmetry * 0.3)
            
            # Current price is point 5
            current_price = price5
            
            # Determine entry direction
            direction = 'BUY' if is_bullish else 'SELL'
            
            return WolfeWave(
                wave_points=[(idx1, price1), (idx2, price2), (idx3, price3), (idx4, price4), (idx5, price5)],
                entry_price=current_price,
                target_price=float(target_price),
                direction=direction,
                confidence=min(0.95, confidence),
                timestamp=df.index[idx5] if idx5 < len(df) else df.index[-1]
            )
            
        except Exception as e:
            log.error(f"Error validating Wolfe Wave: {e}")
            return None
    
    def get_entry_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal from latest Wolfe Wave
        """
        waves = self.detect_wolfe_waves(df)
        if not waves:
            return None
        
        # Get most recent wave
        latest_wave = waves[-1]
        
        return {
            'name': 'Wolfe Wave',
            'direction': latest_wave.direction,
            'confidence': latest_wave.confidence,
            'entry_price': latest_wave.entry_price,
            'target_price': latest_wave.target_price,
            'wave_points': latest_wave.wave_points,
            'reasons': [f"Wolfe Wave pattern detected with {latest_wave.confidence:.0%} confidence"],
            'type': 'WOLFE_WAVE'
        }