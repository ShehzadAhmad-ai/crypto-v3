"""
SMC Expert V3 - Order Block Analysis (COMPLETE REWRITE)
Order blocks, breaker blocks, reclaimed blocks with strength scoring
FIXED: No KeyError, proper object access, working detection logic
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .smc_core import (
    OrderBlock, OBType, Direction, MitigationState, Candle, Swing, SwingType,
    calculate_atr, normalize, safe_get_swing_price, safe_get_swing_type
)
from .smc_config import CONFIG


class OrderBlockDetector:
    """Detects bullish and bearish order blocks with strength scoring - FIXED"""
    
    def __init__(self):
        self.order_blocks: List[OrderBlock] = []
        self.breaker_blocks: List[OrderBlock] = []
        self.reclaimed_blocks: List[OrderBlock] = []
        self.candles: List[Candle] = []
    
    def detect_all(self, df: pd.DataFrame, swings: List[Swing]) -> Dict[str, List[OrderBlock]]:
        """Detect all types of order blocks"""
        # Convert DataFrame to Candle objects FIRST (fixes KeyError)
        self.candles = self._df_to_candles(df)
        
        if len(self.candles) < 10:
            return {'order_blocks': [], 'breaker_blocks': [], 'reclaimed_blocks': []}
        
        # Detect primary order blocks
        self.order_blocks = self._detect_order_blocks(self.candles, swings)
        
        # Detect breaker blocks (failed OBs that reversed)
        self.breaker_blocks = self._detect_breaker_blocks(self.candles, self.order_blocks)
        
        # Detect reclaimed blocks (broken OBs that got reclaimed)
        self.reclaimed_blocks = self._detect_reclaimed_blocks(self.candles, self.order_blocks)
        
        return {
            'order_blocks': self.order_blocks,
            'breaker_blocks': self.breaker_blocks,
            'reclaimed_blocks': self.reclaimed_blocks
        }
    
    def _df_to_candles(self, df: pd.DataFrame) -> List[Candle]:
        """Convert DataFrame to Candle objects - FIXES KeyError issues"""
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
            except Exception as e:
                # Skip problematic rows
                continue
        return candles
    
    def _detect_order_blocks(self, candles: List[Candle], swings: List[Swing]) -> List[OrderBlock]:
        """
        Detect bullish and bearish order blocks
        Bullish OB: Strong down candle followed by bullish displacement
        Bearish OB: Strong up candle followed by bearish displacement
        """
        blocks = []
        
        if len(candles) < 10:
            return blocks
        
        # Calculate ATR from candles
        atr = self._calculate_atr_from_candles(candles)
        
        for i in range(2, len(candles) - 3):
            candle = candles[i]
            
            # Check for displacement after candle i
            displacement = self._check_displacement(candles, i, atr)
            
            if displacement['strength'] < CONFIG.OB_MIN_DISPLACEMENT_ATR:
                continue
            
            # Bullish Order Block: Bearish candle followed by bullish displacement
            if candle.is_bearish and displacement['direction'] == Direction.BUY:
                block = self._create_bullish_ob(candles, i, displacement, atr)
                if block and block.strength >= CONFIG.OB_MIN_STRENGTH:
                    blocks.append(block)
            
            # Bearish Order Block: Bullish candle followed by bearish displacement
            elif candle.is_bullish and displacement['direction'] == Direction.SELL:
                block = self._create_bearish_ob(candles, i, displacement, atr)
                if block and block.strength >= CONFIG.OB_MIN_STRENGTH:
                    blocks.append(block)
        
        # Filter by age
        current_idx = len(candles)
        blocks = [b for b in blocks if b.age_bars <= CONFIG.OB_MAX_AGE_BARS]
        
        # Remove duplicates
        blocks = self._deduplicate_blocks(blocks, atr)
        
        return blocks
    
    def _check_displacement(self, candles: List[Candle], i: int, atr: float) -> Dict:
        """Check if candle i+1 shows displacement"""
        if i + 1 >= len(candles):
            return {'strength': 0, 'direction': Direction.NEUTRAL, 'move_atr': 0, 'volume_ratio': 1.0}
        
        next_candle = candles[i + 1]
        
        # Calculate move size
        if next_candle.is_bullish:
            move = next_candle.close - next_candle.open
            direction = Direction.BUY
        else:
            move = next_candle.open - next_candle.close
            direction = Direction.SELL
        
        # Move size in ATR multiples
        move_atr = move / atr if atr > 0 else 0
        
        # Volume confirmation (using candle volume vs average)
        start_idx = max(0, i - 20)
        avg_volume = sum(c.volume for c in candles[start_idx:i]) / max(1, i - start_idx)
        volume_ratio = next_candle.volume / avg_volume if avg_volume > 0 else 1.0
        
        # Body ratio (strong candles have high body ratio)
        body_ratio = next_candle.body_ratio
        
        strength = (min(1.0, move_atr / 1.5) * 0.4 + 
                    min(1.0, volume_ratio / 1.5) * 0.3 + 
                    body_ratio * 0.3)
        
        return {
            'strength': strength,
            'direction': direction,
            'move_atr': move_atr,
            'volume_ratio': volume_ratio
        }
    
    def _create_bullish_ob(self, candles: List[Candle], i: int, 
                            displacement: Dict, atr: float) -> Optional[OrderBlock]:
        """Create bullish order block from bearish candle"""
        candle = candles[i]
        
        # Bullish OB entry is the HIGH of the bearish candle
        entry = candle.high
        
        # Stop loss is the LOW of the bearish candle
        stop = candle.low
        
        # Calculate age in bars
        age_bars = len(candles) - i
        
        # Calculate volume ratio
        start_idx = max(0, i - 20)
        avg_volume = sum(c.volume for c in candles[start_idx:i]) / max(1, i - start_idx)
        volume_ratio = candle.volume / avg_volume if avg_volume > 0 else 1.0
        
        # Wick ratio (lower wick for bullish OB - shows rejection)
        lower_wick = candle.lower_wick
        wick_ratio = lower_wick / candle.range_ if candle.range_ > 0 else 0.3
        
        # Displacement strength
        displacement_strength = displacement['strength']
        
        # Overall strength
        strength = (min(1.0, volume_ratio / 1.5) * 0.3 + 
                    min(1.0, wick_ratio) * 0.3 + 
                    displacement_strength * 0.4)
        
        return OrderBlock(
            type=OBType.BULLISH,
            direction=Direction.BUY,
            price=entry,
            stop=stop,
            strength=strength,
            age_bars=age_bars,
            timestamp=candle.timestamp,
            volume_ratio=volume_ratio,
            displacement_strength=displacement_strength
        )
    
    def _create_bearish_ob(self, candles: List[Candle], i: int, 
                            displacement: Dict, atr: float) -> Optional[OrderBlock]:
        """Create bearish order block from bullish candle"""
        candle = candles[i]
        
        # Bearish OB entry is the LOW of the bullish candle
        entry = candle.low
        
        # Stop loss is the HIGH of the bullish candle
        stop = candle.high
        
        # Age
        age_bars = len(candles) - i
        
        # Volume ratio
        start_idx = max(0, i - 20)
        avg_volume = sum(c.volume for c in candles[start_idx:i]) / max(1, i - start_idx)
        volume_ratio = candle.volume / avg_volume if avg_volume > 0 else 1.0
        
        # Wick ratio (upper wick for bearish OB - shows rejection)
        upper_wick = candle.upper_wick
        wick_ratio = upper_wick / candle.range_ if candle.range_ > 0 else 0.3
        
        # Strength
        strength = (min(1.0, volume_ratio / 1.5) * 0.3 + 
                    min(1.0, wick_ratio) * 0.3 + 
                    displacement['strength'] * 0.4)
        
        return OrderBlock(
            type=OBType.BEARISH,
            direction=Direction.SELL,
            price=entry,
            stop=stop,
            strength=strength,
            age_bars=age_bars,
            timestamp=candle.timestamp,
            volume_ratio=volume_ratio,
            displacement_strength=displacement['strength']
        )
    
    def _detect_breaker_blocks(self, candles: List[Candle], 
                                order_blocks: List[OrderBlock]) -> List[OrderBlock]:
        """
        Breaker blocks: Order blocks that failed and reversed
        Bullish breaker: Price broke below bullish OB, then reversed up
        Bearish breaker: Price broke above bearish OB, then reversed down
        """
        breakers = []
        
        for ob in order_blocks:
            # Check if OB was broken
            broken = self._is_ob_broken(candles, ob)
            
            if not broken:
                continue
            
            # Check for reversal after break
            reversed_ = self._check_reversal_after_break(candles, ob, broken['break_index'])
            
            if reversed_:
                # Create breaker block (opposite direction)
                if ob.type == OBType.BULLISH:
                    breaker_type = OBType.BREAKER_BULLISH
                    direction = Direction.SELL
                    price = ob.stop
                    stop = ob.price
                else:
                    breaker_type = OBType.BREAKER_BEARISH
                    direction = Direction.BUY
                    price = ob.stop
                    stop = ob.price
                
                breaker = OrderBlock(
                    type=breaker_type,
                    direction=direction,
                    price=price,
                    stop=stop,
                    strength=ob.strength * 0.8,
                    age_bars=len(candles) - broken['break_index'],
                    timestamp=broken['break_timestamp'],
                    volume_ratio=ob.volume_ratio,
                    displacement_strength=ob.displacement_strength
                )
                
                breakers.append(breaker)
        
        return breakers
    
    def _detect_reclaimed_blocks(self, candles: List[Candle], 
                                   order_blocks: List[OrderBlock]) -> List[OrderBlock]:
        """
        Reclaimed blocks: Order blocks that were broken and then reclaimed
        Strong reversal signal
        """
        reclaimed = []
        
        for ob in order_blocks:
            # Check if OB was broken
            broken = self._is_ob_broken(candles, ob)
            
            if not broken:
                continue
            
            # Check if OB was reclaimed
            reclaimed_ = self._check_reclaim(candles, ob, broken['break_index'])
            
            if reclaimed_:
                reclaimed_ob = OrderBlock(
                    type=OBType.RECLAIMED_BULLISH if ob.type == OBType.BULLISH else OBType.RECLAIMED_BEARISH,
                    direction=ob.direction,
                    price=ob.price,
                    stop=ob.stop,
                    strength=min(1.0, ob.strength * 1.2),
                    age_bars=len(candles) - reclaimed_['reclaim_index'],
                    timestamp=reclaimed_['reclaim_timestamp'],
                    volume_ratio=ob.volume_ratio,
                    displacement_strength=ob.displacement_strength
                )
                
                reclaimed.append(reclaimed_ob)
        
        return reclaimed
    
    def _is_ob_broken(self, candles: List[Candle], ob: OrderBlock) -> Optional[Dict]:
        """Check if order block was broken"""
        start_idx = max(0, len(candles) - ob.age_bars)
        
        for i in range(start_idx, len(candles)):
            candle = candles[i]
            
            if ob.type == OBType.BULLISH and candle.low < ob.stop:
                return {
                    'broken': True,
                    'break_index': i,
                    'break_timestamp': candle.timestamp,
                    'break_price': candle.low
                }
            
            if ob.type == OBType.BEARISH and candle.high > ob.stop:
                return {
                    'broken': True,
                    'break_index': i,
                    'break_timestamp': candle.timestamp,
                    'break_price': candle.high
                }
        
        return None
    
    def _check_reversal_after_break(self, candles: List[Candle], ob: OrderBlock, 
                                      break_index: int) -> bool:
        """Check if price reversed after breaking OB"""
        # Look at last 5 candles for reversal
        start = max(break_index, len(candles) - 10)
        
        for i in range(start, len(candles)):
            candle = candles[i]
            
            if ob.type == OBType.BULLISH:
                # After breaking down, price should come back up to OB
                if candle.high > ob.price:
                    return True
            else:
                # After breaking up, price should come back down to OB
                if candle.low < ob.price:
                    return True
        
        return False
    
    def _check_reclaim(self, candles: List[Candle], ob: OrderBlock, 
                        break_index: int) -> Optional[Dict]:
        """Check if OB was reclaimed after being broken"""
        for i in range(break_index + 1, len(candles) - 2):
            candle = candles[i]
            
            if ob.type == OBType.BULLISH:
                # Price comes back above OB price and holds
                if candle.close > ob.price:
                    # Check if it holds for 2+ candles
                    if (i + 2 < len(candles) and
                        candles[i+1].close > ob.price and 
                        candles[i+2].close > ob.price):
                        return {
                            'reclaimed': True,
                            'reclaim_index': i,
                            'reclaim_timestamp': candle.timestamp
                        }
            else:
                # Price comes back below OB price and holds
                if candle.close < ob.price:
                    if (i + 2 < len(candles) and
                        candles[i+1].close < ob.price and 
                        candles[i+2].close < ob.price):
                        return {
                            'reclaimed': True,
                            'reclaim_index': i,
                            'reclaim_timestamp': candle.timestamp
                        }
        
        return None
    
    def _deduplicate_blocks(self, blocks: List[OrderBlock], atr: float) -> List[OrderBlock]:
        """Remove duplicate or overlapping order blocks"""
        if len(blocks) < 2:
            return blocks
        
        deduped = []
        tolerance = atr * CONFIG.OB_MITIGATION_TOLERANCE_ATR
        
        for block in sorted(blocks, key=lambda x: x.strength, reverse=True):
            duplicate = False
            for existing in deduped:
                if abs(block.price - existing.price) < tolerance:
                    duplicate = True
                    break
            
            if not duplicate:
                deduped.append(block)
        
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


class MitigationTracker:
    """Tracks mitigation status for order blocks and FVGs"""
    
    def __init__(self):
        self.tracked_items: Dict[str, Dict] = {}
    
    def check_ob_mitigation(self, ob: OrderBlock, current_price: float, 
                             atr: float) -> Tuple[MitigationState, float]:
        """Check order block mitigation status"""
        tolerance = atr * CONFIG.OB_MITIGATION_TOLERANCE_ATR
        
        if ob.type == OBType.BULLISH:
            # Bullish OB: price should stay above stop, entry is at high
            if current_price <= ob.stop - tolerance:
                return MitigationState.INVALIDATED, 0.0
            elif current_price >= ob.price:
                return MitigationState.FULL, 0.3
            elif current_price > ob.stop:
                return MitigationState.PARTIAL, 0.7
            else:
                return MitigationState.UNMITIGATED, 1.0
        else:
            # Bearish OB: price should stay below stop, entry is at low
            if current_price >= ob.stop + tolerance:
                return MitigationState.INVALIDATED, 0.0
            elif current_price <= ob.price:
                return MitigationState.FULL, 0.3
            elif current_price < ob.stop:
                return MitigationState.PARTIAL, 0.7
            else:
                return MitigationState.UNMITIGATED, 1.0
    
    def update_ob_mitigation(self, ob: OrderBlock, current_price: float, atr: float):
        """Update order block mitigation state"""
        state, score = self.check_ob_mitigation(ob, current_price, atr)
        ob.mitigation_state = state
        ob.mitigation_price = current_price if state != MitigationState.UNMITIGATED else None
        return state, score