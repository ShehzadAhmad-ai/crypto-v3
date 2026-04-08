"""
sl_tp_engine.py
Layer 6: SL/TP Engine for Price Action Expert V3.5

Calculates structure-based stop losses and take profits using:
- Pattern structure (engulfing low, pin bar wick)
- Market structure (swing highs/lows, order blocks)
- Support/Resistance levels
- ATR-based fallback

Priority hierarchy ensures stops are placed at logical invalidation points
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import configuration
from .price_action_config import (
    STOP_LOSS_PRIORITY,
    TAKE_PROFIT_PRIORITY,
    MIN_STOP_ATR,
    STOP_BUFFER_PERCENT,
    FALLBACK_RISK_REWARD,
    MAX_RISK_REWARD
)

# Import candle analyzer
from .candle_analyzer import CandleData, CandleAnalyzer


class StopType(Enum):
    """Types of stop loss placement"""
    PATTERN_STRUCTURE = "pattern_structure"
    MARKET_STRUCTURE = "market_structure"
    SUPPORT_RESISTANCE = "support_resistance"
    ORDER_BLOCK = "order_block"
    ATR_FALLBACK = "atr_fallback"


class TargetType(Enum):
    """Types of take profit placement"""
    HTF_LEVEL = "htf_level"
    MARKET_STRUCTURE = "market_structure"
    PATTERN_PROJECTION = "pattern_projection"
    ATR_FALLBACK = "atr_fallback"


@dataclass
class TradeSetup:
    """
    Complete trade setup with SL/TP
    """
    # Core trade parameters
    direction: str                       # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    
    # Stop details
    stop_type: StopType
    stop_reason: str
    stop_distance_pct: float
    stop_distance_atr: float
    
    # Target details
    target_type: TargetType
    target_reason: str
    target_distance_pct: float
    
    # Risk management
    risk_amount: float                   # % of account risk
    position_size_multiplier: float     # 0.5-1.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            'direction': self.direction,
            'entry': round(self.entry_price, 6),
            'stop_loss': round(self.stop_loss, 6),
            'take_profit': round(self.take_profit, 6),
            'risk_reward': round(self.risk_reward, 2),
            'stop_type': self.stop_type.value,
            'stop_reason': self.stop_reason,
            'target_type': self.target_type.value,
            'target_reason': self.target_reason,
            'stop_distance_pct': round(self.stop_distance_pct, 4),
            'risk_reward': round(self.risk_reward, 2)
        }


class SLTPEngine:
    """
    Advanced SL/TP engine with priority hierarchy
    
    Features:
    - Priority-based stop loss placement
    - Priority-based take profit placement
    - Structure-based levels from price action
    - ATR-based fallback with configurable multipliers
    - Risk/reward calculation with caps
    """
    
    def __init__(self):
        """Initialize the SL/TP engine"""
        self.candle_analyzer = CandleAnalyzer()
    
    # =========================================================
    # STOP LOSS CALCULATION (PRIORITY HIERARCHY)
    # =========================================================
    
    def get_stop_from_pattern(self, direction: str, pattern_name: str,
                               candles: List[CandleData], pattern_index: int,
                               atr: float) -> Optional[Tuple[float, str, float]]:
        """
        Get stop loss from pattern structure
        
        Priority 1: Pattern-specific structure
        
        Returns:
            (stop_price, reason, distance_atr) or None
        """
        if pattern_index < 0 or pattern_index >= len(candles):
            return None
        
        candle = candles[pattern_index]
        
        if direction == 'BUY':
            # For BUY: stop below pattern low
            stop = candle.low * (1 - STOP_BUFFER_PERCENT)
            
            # Pattern-specific adjustments
            if 'engulfing' in pattern_name.lower():
                # Use the engulfing candle low
                stop = candle.low * (1 - STOP_BUFFER_PERCENT)
                reason = f"Below bullish engulfing low at {candle.low:.4f}"
            elif 'hammer' in pattern_name.lower() or 'pin' in pattern_name.lower():
                # Use the wick low for pin bars
                stop = candle.low * (1 - STOP_BUFFER_PERCENT * 0.5)
                reason = f"Below hammer/pin bar wick at {candle.low:.4f}"
            elif 'morning' in pattern_name.lower():
                # Use the lowest of the three candles
                if pattern_index >= 2:
                    lowest = min(candles[pattern_index-2].low, candles[pattern_index-1].low, candle.low)
                    stop = lowest * (1 - STOP_BUFFER_PERCENT)
                    reason = f"Below morning star low at {lowest:.4f}"
                else:
                    stop = candle.low * (1 - STOP_BUFFER_PERCENT)
                    reason = f"Below pattern low at {candle.low:.4f}"
            else:
                reason = f"Below pattern low at {candle.low:.4f}"
            
            distance_atr = (candle.close - stop) / (atr + 1e-8)
            
        else:  # SELL
            # For SELL: stop above pattern high
            stop = candle.high * (1 + STOP_BUFFER_PERCENT)
            
            if 'engulfing' in pattern_name.lower():
                stop = candle.high * (1 + STOP_BUFFER_PERCENT)
                reason = f"Above bearish engulfing high at {candle.high:.4f}"
            elif 'shooting' in pattern_name.lower() or 'pin' in pattern_name.lower():
                stop = candle.high * (1 + STOP_BUFFER_PERCENT * 0.5)
                reason = f"Above shooting star wick at {candle.high:.4f}"
            elif 'evening' in pattern_name.lower():
                if pattern_index >= 2:
                    highest = max(candles[pattern_index-2].high, candles[pattern_index-1].high, candle.high)
                    stop = highest * (1 + STOP_BUFFER_PERCENT)
                    reason = f"Above evening star high at {highest:.4f}"
                else:
                    stop = candle.high * (1 + STOP_BUFFER_PERCENT)
                    reason = f"Above pattern high at {candle.high:.4f}"
            else:
                reason = f"Above pattern high at {candle.high:.4f}"
            
            distance_atr = (stop - candle.close) / (atr + 1e-8)
        
        return stop, reason, distance_atr
    
    def get_stop_from_market_structure(self, direction: str, df: pd.DataFrame,
                                        current_price: float, atr: float,
                                        structure_data: Optional[Dict] = None) -> Optional[Tuple[float, str, float]]:
        """
        Get stop loss from market structure (swing highs/lows)
        
        Priority 2: Recent swing points
        """
        try:
            if structure_data:
                # Use provided structure data
                swings = structure_data.get('swings', [])
                if swings:
                    if direction == 'BUY':
                        # Find nearest swing low below current price
                        swing_lows = [s['price'] for s in swings if s.get('type') == 'LL' and s['price'] < current_price]
                        if swing_lows:
                            nearest_low = max(swing_lows)
                            stop = nearest_low * (1 - STOP_BUFFER_PERCENT)
                            distance_atr = (current_price - stop) / (atr + 1e-8)
                            return stop, f"Below recent swing low at {nearest_low:.4f}", distance_atr
                    else:
                        # Find nearest swing high above current price
                        swing_highs = [s['price'] for s in swings if s.get('type') == 'HH' and s['price'] > current_price]
                        if swing_highs:
                            nearest_high = min(swing_highs)
                            stop = nearest_high * (1 + STOP_BUFFER_PERCENT)
                            distance_atr = (stop - current_price) / (atr + 1e-8)
                            return stop, f"Above recent swing high at {nearest_high:.4f}", distance_atr
            
            # Fallback: calculate swing points from price action
            if len(df) >= 20:
                recent = df.iloc[-20:]
                
                if direction == 'BUY':
                    # Find recent swing lows
                    lows = recent['low'].values
                    swing_lows = []
                    for i in range(2, len(lows) - 2):
                        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                            swing_lows.append(lows[i])
                    
                    if swing_lows:
                        nearest_low = max([l for l in swing_lows if l < current_price])
                        stop = nearest_low * (1 - STOP_BUFFER_PERCENT)
                        distance_atr = (current_price - stop) / (atr + 1e-8)
                        return stop, f"Below swing low at {nearest_low:.4f}", distance_atr
                else:
                    highs = recent['high'].values
                    swing_highs = []
                    for i in range(2, len(highs) - 2):
                        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                            swing_highs.append(highs[i])
                    
                    if swing_highs:
                        nearest_high = min([h for h in swing_highs if h > current_price])
                        stop = nearest_high * (1 + STOP_BUFFER_PERCENT)
                        distance_atr = (stop - current_price) / (atr + 1e-8)
                        return stop, f"Above swing high at {nearest_high:.4f}", distance_atr
            
            return None
            
        except Exception:
            return None
    
    def get_stop_from_support_resistance(self, direction: str, current_price: float,
                                          atr: float, sr_data: Optional[Dict] = None) -> Optional[Tuple[float, str, float]]:
        """
        Get stop loss from support/resistance levels
        
        Priority 3: S/R levels
        """
        try:
            if sr_data:
                if direction == 'BUY':
                    support = sr_data.get('closest_support')
                    if support and support < current_price:
                        stop = support * (1 - STOP_BUFFER_PERCENT)
                        distance_atr = (current_price - stop) / (atr + 1e-8)
                        return stop, f"Below support at {support:.4f}", distance_atr
                else:
                    resistance = sr_data.get('closest_resistance')
                    if resistance and resistance > current_price:
                        stop = resistance * (1 + STOP_BUFFER_PERCENT)
                        distance_atr = (stop - current_price) / (atr + 1e-8)
                        return stop, f"Above resistance at {resistance:.4f}", distance_atr
            
            return None
            
        except Exception:
            return None
    
    def get_stop_from_order_block(self, direction: str, current_price: float,
                                   atr: float, structure_data: Optional[Dict] = None) -> Optional[Tuple[float, str, float]]:
        """
        Get stop loss from order blocks
        
        Priority 4: Unmitigated order blocks
        """
        try:
            if structure_data:
                order_blocks = structure_data.get('order_blocks', [])
                for ob in order_blocks:
                    if direction == 'BUY':
                        # Bullish order block becomes support
                        if ob.get('type') == 'BULLISH_OB' and ob.get('mitigation_state') == 'UNMITIGATED':
                            ob_price = ob.get('low', ob.get('price', 0))
                            if ob_price < current_price:
                                stop = ob_price * (1 - STOP_BUFFER_PERCENT)
                                distance_atr = (current_price - stop) / (atr + 1e-8)
                                return stop, f"Below bullish order block at {ob_price:.4f}", distance_atr
                    else:
                        # Bearish order block becomes resistance
                        if ob.get('type') == 'BEARISH_OB' and ob.get('mitigation_state') == 'UNMITIGATED':
                            ob_price = ob.get('high', ob.get('price', 0))
                            if ob_price > current_price:
                                stop = ob_price * (1 + STOP_BUFFER_PERCENT)
                                distance_atr = (stop - current_price) / (atr + 1e-8)
                                return stop, f"Above bearish order block at {ob_price:.4f}", distance_atr
            
            return None
            
        except Exception:
            return None
    
    def get_stop_atr_fallback(self, direction: str, current_price: float,
                               atr: float) -> Tuple[float, str, float]:
        """
        ATR-based fallback stop loss
        
        Priority 5: Last resort
        """
        min_stop_distance = atr * MIN_STOP_ATR
        
        if direction == 'BUY':
            stop = current_price - min_stop_distance
            distance_atr = MIN_STOP_ATR
            reason = f"ATR fallback: {MIN_STOP_ATR:.1f}x ATR below entry"
        else:
            stop = current_price + min_stop_distance
            distance_atr = MIN_STOP_ATR
            reason = f"ATR fallback: {MIN_STOP_ATR:.1f}x ATR above entry"
        
        return stop, reason, distance_atr
    
    def calculate_stop_loss(self, direction: str, current_price: float,
                            atr: float, pattern_name: str = "",
                            candles: Optional[List[CandleData]] = None,
                            pattern_index: int = -1,
                            structure_data: Optional[Dict] = None,
                            sr_data: Optional[Dict] = None) -> Tuple[float, StopType, str, float]:
        """
        Calculate stop loss using priority hierarchy
        
        Priority:
        1. Pattern structure
        2. Market structure (swings)
        3. Support/Resistance
        4. Order blocks
        5. ATR fallback
        
        Returns:
            (stop_price, stop_type, reason, distance_atr)
        """
        # Priority 1: Pattern structure
        if candles and pattern_index >= 0 and pattern_name:
            result = self.get_stop_from_pattern(direction, pattern_name, candles, pattern_index, atr)
            if result:
                stop, reason, distance_atr = result
                # Check minimum distance
                if distance_atr >= MIN_STOP_ATR * 0.5:
                    return stop, StopType.PATTERN_STRUCTURE, reason, distance_atr
        
        # Priority 2: Market structure
        result = self.get_stop_from_market_structure(direction, pd.DataFrame(), current_price, atr, structure_data)
        if result:
            stop, reason, distance_atr = result
            if distance_atr >= MIN_STOP_ATR * 0.5:
                return stop, StopType.MARKET_STRUCTURE, reason, distance_atr
        
        # Priority 3: Support/Resistance
        result = self.get_stop_from_support_resistance(direction, current_price, atr, sr_data)
        if result:
            stop, reason, distance_atr = result
            if distance_atr >= MIN_STOP_ATR * 0.5:
                return stop, StopType.SUPPORT_RESISTANCE, reason, distance_atr
        
        # Priority 4: Order blocks
        result = self.get_stop_from_order_block(direction, current_price, atr, structure_data)
        if result:
            stop, reason, distance_atr = result
            if distance_atr >= MIN_STOP_ATR * 0.5:
                return stop, StopType.ORDER_BLOCK, reason, distance_atr
        
        # Priority 5: ATR fallback
        stop, reason, distance_atr = self.get_stop_atr_fallback(direction, current_price, atr)
        return stop, StopType.ATR_FALLBACK, reason, distance_atr
    
    # =========================================================
    # TAKE PROFIT CALCULATION (PRIORITY HIERARCHY)
    # =========================================================
    
    def get_target_from_htf_level(self, direction: str, current_price: float,
                                   atr: float, htf_levels: Optional[Dict] = None) -> Optional[Tuple[float, str, float]]:
        """
        Get take profit from higher timeframe levels
        
        Priority 1: HTF resistance/support
        """
        try:
            if htf_levels:
                if direction == 'BUY':
                    # Next resistance level
                    resistances = htf_levels.get('resistances', [])
                    if resistances:
                        next_resistance = min([r for r in resistances if r > current_price])
                        if next_resistance:
                            distance_atr = (next_resistance - current_price) / (atr + 1e-8)
                            return next_resistance, f"Next HTF resistance at {next_resistance:.4f}", distance_atr
                else:
                    # Next support level
                    supports = htf_levels.get('supports', [])
                    if supports:
                        next_support = max([s for s in supports if s < current_price])
                        if next_support:
                            distance_atr = (current_price - next_support) / (atr + 1e-8)
                            return next_support, f"Next HTF support at {next_support:.4f}", distance_atr
            
            return None
            
        except Exception:
            return None
    
    def get_target_from_market_structure(self, direction: str, current_price: float,
                                          atr: float, structure_data: Optional[Dict] = None) -> Optional[Tuple[float, str, float]]:
        """
        Get take profit from market structure (swing highs/lows)
        
        Priority 2: Swing points
        """
        try:
            if structure_data:
                swings = structure_data.get('swings', [])
                if swings:
                    if direction == 'BUY':
                        # Next swing high
                        swing_highs = [s['price'] for s in swings if s.get('type') == 'HH' and s['price'] > current_price]
                        if swing_highs:
                            next_high = min(swing_highs)
                            distance_atr = (next_high - current_price) / (atr + 1e-8)
                            return next_high, f"Next swing high at {next_high:.4f}", distance_atr
                    else:
                        # Next swing low
                        swing_lows = [s['price'] for s in swings if s.get('type') == 'LL' and s['price'] < current_price]
                        if swing_lows:
                            next_low = max(swing_lows)
                            distance_atr = (current_price - next_low) / (atr + 1e-8)
                            return next_low, f"Next swing low at {next_low:.4f}", distance_atr
            
            return None
            
        except Exception:
            return None
    
    def get_target_from_pattern_projection(self, direction: str, current_price: float,
                                            stop_distance_atr: float, atr: float) -> Tuple[float, str, float]:
        """
        Get take profit from pattern projection (2x risk)
        
        Priority 3: Standard risk/reward projection
        """
        risk_atr = stop_distance_atr
        target_atr = risk_atr * FALLBACK_RISK_REWARD
        
        if direction == 'BUY':
            target = current_price + (target_atr * atr)
        else:
            target = current_price - (target_atr * atr)
        
        reason = f"Pattern projection: {FALLBACK_RISK_REWARD:.1f}x risk"
        return target, reason, target_atr
    
    def get_target_atr_fallback(self, direction: str, current_price: float,
                                 atr: float, stop_distance_atr: float) -> Tuple[float, str, float]:
        """
        ATR-based fallback take profit
        
        Priority 4: Last resort
        """
        target_atr = stop_distance_atr * FALLBACK_RISK_REWARD
        target_atr = min(target_atr, MAX_RISK_REWARD)
        
        if direction == 'BUY':
            target = current_price + (target_atr * atr)
        else:
            target = current_price - (target_atr * atr)
        
        reason = f"ATR fallback: {target_atr:.1f}x ATR ({FALLBACK_RISK_REWARD:.1f}:1 risk/reward)"
        return target, reason, target_atr
    
    def calculate_take_profit(self, direction: str, current_price: float,
                              stop_distance_atr: float, atr: float,
                              htf_levels: Optional[Dict] = None,
                              structure_data: Optional[Dict] = None) -> Tuple[float, TargetType, str, float]:
        """
        Calculate take profit using priority hierarchy
        
        Priority:
        1. HTF resistance/support
        2. Market structure (swing highs/lows)
        3. Pattern projection (2x risk)
        4. ATR fallback
        
        Returns:
            (target_price, target_type, reason, distance_atr)
        """
        # Priority 1: HTF levels
        result = self.get_target_from_htf_level(direction, current_price, atr, htf_levels)
        if result:
            target, reason, distance_atr = result
            return target, TargetType.HTF_LEVEL, reason, distance_atr
        
        # Priority 2: Market structure
        result = self.get_target_from_market_structure(direction, current_price, atr, structure_data)
        if result:
            target, reason, distance_atr = result
            # Check if distance is reasonable (at least 1x risk)
            if distance_atr >= stop_distance_atr * 0.8:
                return target, TargetType.MARKET_STRUCTURE, reason, distance_atr
        
        # Priority 3: Pattern projection
        target, reason, distance_atr = self.get_target_from_pattern_projection(
            direction, current_price, stop_distance_atr, atr
        )
        return target, TargetType.PATTERN_PROJECTION, reason, distance_atr
    
    # =========================================================
    # COMPLETE TRADE SETUP
    # =========================================================
    
    def calculate_setup(self, direction: str, current_price: float,
                        atr: float, pattern_name: str = "",
                        candles: Optional[List[CandleData]] = None,
                        pattern_index: int = -1,
                        structure_data: Optional[Dict] = None,
                        sr_data: Optional[Dict] = None,
                        htf_levels: Optional[Dict] = None,
                        risk_percent: float = 1.0) -> TradeSetup:
        """
        Complete trade setup calculation
        
        Args:
            direction: 'BUY' or 'SELL'
            current_price: Current market price
            atr: Average True Range
            pattern_name: Name of the pattern (for pattern-based stops)
            candles: List of candle data
            pattern_index: Index of the pattern candle
            structure_data: Optional market structure data
            sr_data: Optional support/resistance data
            htf_levels: Optional higher timeframe levels
            risk_percent: Risk percentage of account (for position sizing)
        
        Returns:
            TradeSetup with all parameters
        """
        # Calculate stop loss
        stop, stop_type, stop_reason, stop_distance_atr = self.calculate_stop_loss(
            direction, current_price, atr, pattern_name, candles, pattern_index,
            structure_data, sr_data
        )
        
        # Calculate take profit
        target, target_type, target_reason, target_distance_atr = self.calculate_take_profit(
            direction, current_price, stop_distance_atr, atr, htf_levels, structure_data
        )
        
        # Calculate risk/reward
        risk = abs(current_price - stop)
        reward = abs(target - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        risk_reward = min(MAX_RISK_REWARD, risk_reward)
        
        # Calculate stop distances
        stop_distance_pct = risk / current_price
        target_distance_pct = reward / current_price
        
        # Determine position size multiplier based on risk/reward
        if risk_reward >= 3.0:
            position_multiplier = 1.2
        elif risk_reward >= 2.0:
            position_multiplier = 1.0
        elif risk_reward >= 1.5:
            position_multiplier = 0.8
        else:
            position_multiplier = 0.5
        
        return TradeSetup(
            direction=direction,
            entry_price=current_price,
            stop_loss=stop,
            take_profit=target,
            risk_reward=risk_reward,
            stop_type=stop_type,
            stop_reason=stop_reason,
            stop_distance_pct=stop_distance_pct,
            stop_distance_atr=stop_distance_atr,
            target_type=target_type,
            target_reason=target_reason,
            target_distance_pct=target_distance_pct,
            risk_amount=risk_percent,
            position_size_multiplier=position_multiplier
        )


# ==================== CONVENIENCE FUNCTIONS ====================

def calculate_trade_setup(df: pd.DataFrame, direction: str,
                          pattern_name: str = "",
                          pattern_index: int = -1,
                          structure_data: Optional[Dict] = None,
                          sr_data: Optional[Dict] = None,
                          htf_levels: Optional[Dict] = None) -> Optional[TradeSetup]:
    """
    Convenience function to calculate trade setup
    
    Args:
        df: OHLCV DataFrame
        direction: 'BUY' or 'SELL'
        pattern_name: Name of the pattern
        pattern_index: Index of the pattern candle
        structure_data: Optional market structure data
        sr_data: Optional support/resistance data
        htf_levels: Optional higher timeframe levels
    
    Returns:
        TradeSetup or None if calculation fails
    """
    if df is None or df.empty:
        return None
    
    current_price = float(df['close'].iloc[-1])
    atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
    
    # Get candles if pattern_index provided
    candles = None
    if pattern_index >= 0:
        from .candle_analyzer import CandleAnalyzer
        analyzer = CandleAnalyzer()
        candles = analyzer.analyze_all_candles(df)
    
    engine = SLTPEngine()
    return engine.calculate_setup(
        direction, current_price, atr, pattern_name, candles, pattern_index,
        structure_data, sr_data, htf_levels
    )


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    np.random.seed(42)
    base_price = 50000
    prices = [base_price]
    
    for i in range(99):
        change = np.random.randn() * 50
        prices.append(prices[-1] + change)
    
    data = []
    for i, close in enumerate(prices):
        open_p = close - np.random.randn() * 20
        high = max(open_p, close) + abs(np.random.randn() * 30)
        low = min(open_p, close) - abs(np.random.randn() * 30)
        volume = abs(np.random.randn() * 10000) + 5000
        
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
    
    # Test structure data
    structure_data = {
        'swings': [
            {'type': 'LL', 'price': 49800},
            {'type': 'HH', 'price': 50200},
            {'type': 'LL', 'price': 49900},
            {'type': 'HH', 'price': 50300},
        ]
    }
    
    # Test S/R data
    sr_data = {
        'closest_support': 49850,
        'closest_resistance': 50250
    }
    
    # Calculate setup
    engine = SLTPEngine()
    setup = engine.calculate_setup(
        direction='BUY',
        current_price=50000,
        atr=150,
        pattern_name='bullish_engulfing',
        structure_data=structure_data,
        sr_data=sr_data
    )
    
    print("=" * 60)
    print("TRADE SETUP CALCULATION")
    print("=" * 60)
    print(f"Direction: {setup.direction}")
    print(f"Entry: {setup.entry_price:.2f}")
    print(f"Stop Loss: {setup.stop_loss:.2f} ({setup.stop_type.value})")
    print(f"  Reason: {setup.stop_reason}")
    print(f"Take Profit: {setup.take_profit:.2f} ({setup.target_type.value})")
    print(f"  Reason: {setup.target_reason}")
    print(f"Risk/Reward: {setup.risk_reward:.2f}")
    print(f"Stop Distance: {setup.stop_distance_pct:.2%} ({setup.stop_distance_atr:.1f}x ATR)")
    print(f"Position Multiplier: {setup.position_size_multiplier:.1f}x")
    print("=" * 60)