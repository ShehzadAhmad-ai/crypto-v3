"""
SMC Expert V3 - Fair Value Gaps & Liquidity Voids (COMPLETE REWRITE)
FVG/IFVG detection, mitigation tracking, and liquidity voids
FIXED: IndexError, KeyError, proper candle object access
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .smc_core import (
    FVG, FVGType, Direction, MitigationState, Candle,
    calculate_atr, normalize, df_row_to_candle
)
from .smc_config import CONFIG


class FVGDetector:
    """Detects Fair Value Gaps and Inverse FVGs - COMPLETE REWRITE"""
    
    def __init__(self):
        self.fvgs: List[FVG] = []
        self.ifvgs: List[FVG] = []
        self.candles: List[Candle] = []
    
    def detect_all(self, df: pd.DataFrame) -> Dict[str, List[FVG]]:
        """Detect both FVGs and IFVGs"""
        # Convert to Candle objects FIRST (fixes KeyError)
        self.candles = self._df_to_candles(df)
        
        if len(self.candles) < 5:
            return {'fvgs': [], 'ifvgs': []}
        
        # Detect FVGs
        self.fvgs = self._detect_fvgs(self.candles)
        
        # Detect IFVGs from existing FVGs
        self.ifvgs = self._detect_ifvgs(self.candles, self.fvgs)
        
        # Filter by age
        self.fvgs = [f for f in self.fvgs if f.age_bars <= CONFIG.FVG_MAX_AGE_BARS]
        self.ifvgs = [f for f in self.ifvgs if f.age_bars <= CONFIG.FVG_MAX_AGE_BARS]
        
        # Filter by strength
        self.fvgs = [f for f in self.fvgs if f.strength >= CONFIG.FVG_MIN_STRENGTH]
        self.ifvgs = [f for f in self.ifvgs if f.strength >= CONFIG.FVG_MIN_STRENGTH]
        
        # Deduplicate
        atr = calculate_atr(df)
        self.fvgs = self._deduplicate_fvgs(self.fvgs, atr)
        self.ifvgs = self._deduplicate_fvgs(self.ifvgs, atr)
        
        return {
            'fvgs': self.fvgs,
            'ifvgs': self.ifvgs
        }
    
    def _df_to_candles(self, df: pd.DataFrame) -> List[Candle]:
        """Convert DataFrame to Candle objects safely"""
        candles = []
        for i in range(len(df)):
            row = df.iloc[i]
            try:
                candle = Candle(
                    index=i,
                    timestamp=row.name if isinstance(row.name, datetime) else datetime.now(),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']) if 'volume' in row else 0
                )
                candles.append(candle)
            except Exception:
                # Skip problematic rows
                continue
        return candles
    
    def _detect_fvgs(self, candles: List[Candle]) -> List[FVG]:
        """
        Detect Fair Value Gaps (3-candle pattern)
        Bullish FVG: Candle 1 high < Candle 3 low (gap up)
        Bearish FVG: Candle 1 low > Candle 3 high (gap down)
        FIXED: Bounds checking to prevent IndexError
        """
        fvgs = []
        
        if len(candles) < 5:
            return fvgs
        
        # Calculate ATR for gap sizing
        atr = self._calculate_atr_from_candles(candles)
        
        # SAFE: Loop with proper bounds (i+2 must exist)
        for i in range(len(candles) - 3):  # -3 ensures i+2 is valid
            candle1 = candles[i]
            candle2 = candles[i + 1]
            candle3 = candles[i + 2]
            
            # Bullish FVG: gap between candle1 high and candle3 low
            if candle1.high < candle3.low:
                gap = candle3.low - candle1.high
                gap_atr = gap / atr if atr > 0 else 0
                
                if gap_atr >= CONFIG.FVG_MIN_GAP_ATR:
                    strength = self._calculate_fvg_strength(candles, i, gap_atr, 'bullish')
                    
                    fvg = FVG(
                        type=FVGType.BULLISH,
                        direction=Direction.BUY,
                        upper=candle3.low,
                        lower=candle1.high,
                        mid=(candle1.high + candle3.low) / 2,
                        strength=strength,
                        age_bars=len(candles) - i,
                        timestamp=candle1.timestamp
                    )
                    fvgs.append(fvg)
            
            # Bearish FVG: gap between candle1 low and candle3 high
            elif candle1.low > candle3.high:
                gap = candle1.low - candle3.high
                gap_atr = gap / atr if atr > 0 else 0
                
                if gap_atr >= CONFIG.FVG_MIN_GAP_ATR:
                    strength = self._calculate_fvg_strength(candles, i, gap_atr, 'bearish')
                    
                    fvg = FVG(
                        type=FVGType.BEARISH,
                        direction=Direction.SELL,
                        upper=candle1.low,
                        lower=candle3.high,
                        mid=(candle1.low + candle3.high) / 2,
                        strength=strength,
                        age_bars=len(candles) - i,
                        timestamp=candle1.timestamp
                    )
                    fvgs.append(fvg)
        
        return fvgs
    
    def _detect_ifvgs(self, candles: List[Candle], fvgs: List[FVG]) -> List[FVG]:
        """
        Detect Inverse FVGs (IFVG) - FVGs that have been mitigated and reversed
        FIXED: Proper bounds checking and candle access
        """
        ifvgs = []
        
        if len(candles) < 10:
            return ifvgs
        
        for fvg in fvgs:
            # Skip if already mitigated
            if fvg.mitigation_state != MitigationState.UNMITIGATED:
                continue
            
            # Look for reversal after entering FVG
            start_idx = max(0, len(candles) - 30)
            
            for i in range(start_idx, len(candles) - 2):
                candle = candles[i]
                
                if fvg.type == FVGType.BULLISH:
                    # Bullish FVG: price entered gap, then reversed down
                    if candle.low <= fvg.upper and candle.low >= fvg.lower:
                        # Check next candle for reversal
                        next_candle = candles[i + 1]
                        if next_candle.is_bearish and next_candle.close < fvg.mid:
                            ifvg = FVG(
                                type=FVGType.INVERSE,
                                direction=Direction.SELL,
                                upper=fvg.upper,
                                lower=fvg.lower,
                                mid=fvg.mid,
                                strength=fvg.strength * 0.7,
                                age_bars=len(candles) - i,
                                timestamp=candle.timestamp
                            )
                            ifvgs.append(ifvg)
                            break
                
                elif fvg.type == FVGType.BEARISH:
                    # Bearish FVG: price entered gap, then reversed up
                    if candle.high >= fvg.lower and candle.high <= fvg.upper:
                        next_candle = candles[i + 1]
                        if next_candle.is_bullish and next_candle.close > fvg.mid:
                            ifvg = FVG(
                                type=FVGType.INVERSE,
                                direction=Direction.BUY,
                                upper=fvg.upper,
                                lower=fvg.lower,
                                mid=fvg.mid,
                                strength=fvg.strength * 0.7,
                                age_bars=len(candles) - i,
                                timestamp=candle.timestamp
                            )
                            ifvgs.append(ifvg)
                            break
        
        return ifvgs
    
    def _calculate_fvg_strength(self, candles: List[Candle], index: int, 
                                 gap_atr: float, fvg_type: str) -> float:
        """Calculate FVG strength based on gap size, volume, and candle strength"""
        # Gap size score (larger gap = stronger)
        gap_score = min(1.0, gap_atr / 0.5)  # 0.5 ATR gap = full score
        
        # Volume around FVG formation
        start_idx = max(0, index - 2)
        end_idx = min(len(candles) - 1, index + 3)
        volumes = [c.volume for c in candles[start_idx:end_idx]]
        avg_volume = sum(volumes) / len(volumes) if volumes else 1
        
        # Compare to historical volume
        hist_start = max(0, index - 30)
        hist_volumes = [c.volume for c in candles[hist_start:start_idx]]
        hist_avg = sum(hist_volumes) / len(hist_volumes) if hist_volumes else avg_volume
        
        volume_score = min(1.0, avg_volume / hist_avg) if hist_avg > 0 else 0.5
        
        # Candle strength (body ratios of the 3 candles)
        body_ratios = []
        for i in range(index, min(index + 3, len(candles))):
            body_ratios.append(candles[i].body_ratio)
        candle_score = sum(body_ratios) / len(body_ratios) if body_ratios else 0.5
        
        strength = (gap_score * 0.4 + volume_score * 0.3 + candle_score * 0.3)
        
        return min(1.0, strength)
    
    def _deduplicate_fvgs(self, fvgs: List[FVG], atr: float) -> List[FVG]:
        """Remove overlapping FVGs"""
        if len(fvgs) < 2:
            return fvgs
        
        deduped = []
        tolerance = atr * 0.1  # 0.1 ATR tolerance
        
        for fvg in sorted(fvgs, key=lambda x: x.strength, reverse=True):
            overlap = False
            for existing in deduped:
                if (abs(fvg.lower - existing.lower) < tolerance and 
                    abs(fvg.upper - existing.upper) < tolerance):
                    overlap = True
                    break
            
            if not overlap:
                deduped.append(fvg)
        
        return deduped
    
    def _calculate_atr_from_candles(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate ATR from candle list"""
        if len(candles) < period + 1:
            return 0.001
        
        true_ranges = []
        for i in range(1, len(candles)):
            high_low = candles[i].high - candles[i].low
            high_close = abs(candles[i].high - candles[i-1].close)
            low_close = abs(candles[i].low - candles[i-1].close)
            tr = max(high_low, high_close, low_close)
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return 0.001
        
        atr = sum(true_ranges[-period:]) / period
        return max(atr, 0.001)
    
    def check_mitigation(self, fvg: FVG, current_price: float, atr: float) -> Tuple[MitigationState, float]:
        """
        Check FVG mitigation status
        Partial: Price entered FVG zone
        Full: Price completely filled FVG
        """
        tolerance = atr * 0.05
        
        if fvg.type in [FVGType.BULLISH, FVGType.INVERSE]:
            # Bullish FVG: price should fill the gap from below
            if current_price >= fvg.upper - tolerance:
                return MitigationState.FULL, 0.3
            elif current_price > fvg.lower + tolerance:
                # Calculate mitigation percentage
                mitigated = (current_price - fvg.lower) / (fvg.upper - fvg.lower)
                if mitigated > 0.5:
                    return MitigationState.PARTIAL, 0.5
                else:
                    return MitigationState.PARTIAL, 0.7
            else:
                return MitigationState.UNMITIGATED, 1.0
        else:
            # Bearish FVG: price should fill the gap from above
            if current_price <= fvg.lower + tolerance:
                return MitigationState.FULL, 0.3
            elif current_price < fvg.upper - tolerance:
                mitigated = (fvg.upper - current_price) / (fvg.upper - fvg.lower)
                if mitigated > 0.5:
                    return MitigationState.PARTIAL, 0.5
                else:
                    return MitigationState.PARTIAL, 0.7
            else:
                return MitigationState.UNMITIGATED, 1.0
    
    def update_mitigation(self, fvg: FVG, current_price: float, atr: float):
        """Update FVG mitigation state"""
        state, score = self.check_mitigation(fvg, current_price, atr)
        fvg.mitigation_state = state
        if state != MitigationState.UNMITIGATED:
            fvg.mitigation_percent = score
        return state, score


class FVGManager:
    """Main FVG manager that orchestrates all gap-related analysis"""
    
    def __init__(self):
        self.fvg_detector = FVGDetector()
        self.fvgs: List[FVG] = []
        self.ifvgs: List[FVG] = []
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Complete FVG analysis"""
        # Detect FVGs and IFVGs
        fvg_results = self.fvg_detector.detect_all(df)
        self.fvgs = fvg_results['fvgs']
        self.ifvgs = fvg_results['ifvgs']
        
        # Update mitigation for FVGs
        current_price = df['close'].iloc[-1]
        atr = calculate_atr(df)
        
        for fvg in self.fvgs:
            self.fvg_detector.update_mitigation(fvg, current_price, atr)
        
        for ifvg in self.ifvgs:
            self.fvg_detector.update_mitigation(ifvg, current_price, atr)
        
        return {
            'fvgs': self.fvgs,
            'ifvgs': self.ifvgs,
            'active_fvgs': [f for f in self.fvgs if f.mitigation_state == MitigationState.UNMITIGATED],
            'active_ifvgs': [f for f in self.ifvgs if f.mitigation_state == MitigationState.UNMITIGATED]
        }
    
    def get_nearest_fvg(self, current_price: float, direction: Direction) -> Optional[FVG]:
        """Get nearest unmitigated FVG in the direction of trade"""
        candidates = []
        
        for fvg in self.fvgs:
            if fvg.mitigation_state != MitigationState.UNMITIGATED:
                continue
            
            if direction == Direction.BUY:
                # For BUY, look for bullish FVGs below price
                if fvg.type == FVGType.BULLISH and fvg.upper < current_price:
                    distance = current_price - fvg.upper
                    candidates.append((distance, fvg))
            else:
                # For SELL, look for bearish FVGs above price
                if fvg.type == FVGType.BEARISH and fvg.lower > current_price:
                    distance = fvg.lower - current_price
                    candidates.append((distance, fvg))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]