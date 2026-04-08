# price_action.py - ULTIMATE ADVANCED CANDLESTICK PATTERN DETECTION
"""
Complete Advanced Candlestick Pattern Detection System
Includes:
- 15+ Single Candle Patterns
- 15+ Multi-Candle Patterns  
- 10+ Rare/Institutional Patterns
- Vectorized operations for speed
- Confidence scoring with volume/trend confirmation
- Pattern combinations detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PatternType(Enum):
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    MULTI = "multi"
    RARE = "rare"
    INSTITUTIONAL = "institutional"

class PatternStrength(Enum):
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 0.95

@dataclass
class CandlePattern:
    """Complete candle pattern data"""
    name: str
    type: PatternType
    direction: str  # 'BULL', 'BEAR', 'NEUTRAL'
    confidence: float
    strength: PatternStrength
    pattern_type: str
    candles_involved: int
    volume_confirmed: bool
    trend_aligned: bool
    description: str
    entry_logic: str
    stop_logic: str
    target_logic: str
    indicators_used: List[str]
    raw_data: Dict[str, Any]
    
    @property
    def reliability(self) -> float:
        """Compatibility with old code that expects reliability attribute"""
        return self.confidence

class AdvancedPriceAction:
    """
    Complete Advanced Candlestick Pattern Detection System
    Detects all major patterns with institutional-grade accuracy
    """
    
    def __init__(self):
        self.min_pattern_bars = 50
        self.volume_lookback = 20
        self.atr_multiplier = 1.5
        
    def detect_all_patterns(self, df: pd.DataFrame) -> List[CandlePattern]:
        """
        Detect ALL candlestick patterns in one pass
        Vectorized for maximum performance
        """
        patterns = []
        
        if df is None or df.empty or len(df) < 5:
            return patterns
        
        try:
            # ===== CONVERT TO NUMPY FOR VECTORIZED OPERATIONS =====
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            
            # Calculate common values once
            bodies = np.abs(closes - opens)
            upper_wick = highs - np.maximum(closes, opens)
            lower_wick = np.minimum(closes, opens) - lows
            total_range = highs - lows
            
            # Moving averages for volume
            vol_ma = pd.Series(volumes).rolling(self.volume_lookback).mean().values
            vol_ratio = volumes / (vol_ma + 1e-8)
            
            # ATR for volatility context
            if 'atr' in df.columns:
                atr = df['atr'].values
            else:
                tr = np.maximum(highs - lows, 
                               np.abs(highs - np.roll(closes, 1)),
                               np.abs(lows - np.roll(closes, 1)))
                atr = pd.Series(tr).rolling(14).mean().values
            
            # ========== SINGLE CANDLE PATTERNS ==========
            
            # HAMMER (Bullish Reversal)
            # Allow for bodies that are very small (doji-like)
            hammer_mask = (lower_wick > bodies * 3) & (upper_wick < (bodies * 0.3 + 1e-8)) & (closes > opens)
            for i in np.where(hammer_mask)[0]:
                if i >= len(df) - 5:  # Only recent candles
                    patterns.append(self._create_pattern(
                        name='Hammer',
                        type=PatternType.SINGLE,
                        direction='BULL',
                        confidence=0.7 + (0.1 if vol_ratio[i] > 1.5 else 0),
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # SHOOTING STAR (Bearish Reversal)
            # Allow for bodies that are very small (doji-like)
            shooting_mask = (upper_wick > bodies * 3) & (lower_wick < (bodies * 0.3 + 1e-8)) & (closes < opens)
            for i in np.where(shooting_mask)[0]:
                if i >= len(df) - 5:
                    patterns.append(self._create_pattern(
                        name='Shooting Star',
                        type=PatternType.SINGLE,
                        direction='BEAR',
                        confidence=0.7 + (0.1 if vol_ratio[i] > 1.5 else 0),
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            # DOJI (Indecision)
            doji_mask = bodies < (total_range * 0.1)
            for i in np.where(doji_mask)[0]:
                if i >= len(df) - 5:
                    # Classify Doji type
                    if lower_wick[i] > total_range[i] * 0.6:
                        name = 'Dragonfly Doji'
                        direction = 'BULL' if i < len(df)-1 and closes[i+1] > closes[i] else 'NEUTRAL'
                    elif upper_wick[i] > total_range[i] * 0.6:
                        name = 'Gravestone Doji'
                        direction = 'BEAR' if i < len(df)-1 and closes[i+1] < closes[i] else 'NEUTRAL'
                    elif lower_wick[i] > bodies[i]*2 and upper_wick[i] > bodies[i]*2:
                        name = 'Long-Legged Doji'
                        direction = 'NEUTRAL'
                    else:
                        name = 'Doji'
                        direction = 'NEUTRAL'
                    
                    patterns.append(self._create_pattern(
                        name=name,
                        type=PatternType.SINGLE,
                        direction=direction,
                        confidence=0.6,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # MARUBOZU (Strong Momentum)
            marubozu_mask = (upper_wick < bodies * 0.1) & (lower_wick < bodies * 0.1)
            for i in np.where(marubozu_mask)[0]:
                if i >= len(df) - 5:
                    direction = 'BULL' if closes[i] > opens[i] else 'BEAR'
                    patterns.append(self._create_pattern(
                        name='Marubozu',
                        type=PatternType.SINGLE,
                        direction=direction,
                        confidence=0.75 + (0.1 if vol_ratio[i] > 1.5 else 0),
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # SPINNING TOP (Indecision)
            spinning_mask = (bodies < total_range * 0.3) & (bodies > total_range * 0.1)
            for i in np.where(spinning_mask)[0]:
                if i >= len(df) - 5:
                    patterns.append(self._create_pattern(
                        name='Spinning Top',
                        type=PatternType.SINGLE,
                        direction='NEUTRAL',
                        confidence=0.5,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # LONG CANDLE (Strong momentum)
            avg_body = np.mean(bodies[-20:]) if len(bodies) >= 20 else np.mean(bodies)
            long_mask = bodies > avg_body * 1.7
            for i in np.where(long_mask)[0]:
                if i >= len(df) - 5:
                    direction = 'BULL' if closes[i] > opens[i] else 'BEAR'
                    patterns.append(self._create_pattern(
                        name='Long Candle',
                        type=PatternType.SINGLE,
                        direction=direction,
                        confidence=0.7 + (0.1 if vol_ratio[i] > 1.5 else 0),
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # ========== DOUBLE CANDLE PATTERNS ==========
            for i in range(1, len(df) - 1):
                if i < len(df) - 5:
                    continue
                
                # BULLISH ENGULFING
                # More flexible: close at or above previous open
                if (closes[i-1] < opens[i-1] and  # Previous bearish
                    closes[i] > opens[i] and        # Current bullish
                    closes[i] >= opens[i-1] and     # Closes at or above prev open
                    opens[i] <= closes[i-1]):       # Opens at or below prev close
                    
                    confidence = 0.8
                    if vol_ratio[i] > 1.5:
                        confidence += 0.1
                    
                    patterns.append(self._create_pattern(
                        name='Bullish Engulfing',
                        type=PatternType.DOUBLE,
                        direction='BULL',
                        confidence=confidence,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # BEARISH ENGULFING
                # More flexible: close at or below previous open
                if (closes[i-1] > opens[i-1] and  # Previous bullish
                    closes[i] < opens[i] and        # Current bearish
                    opens[i] >= closes[i-1] and     # Opens at or above prev close
                    closes[i] <= opens[i-1]):       # Closes at or below prev open
                    
                    confidence = 0.8
                    if vol_ratio[i] > 1.5:
                        confidence += 0.1
                    
                    patterns.append(self._create_pattern(
                        name='Bearish Engulfing',
                        type=PatternType.DOUBLE,
                        direction='BEAR',
                        confidence=confidence,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))

                # INSIDE BAR
                # Allow for equal highs/lows (within 0.1%)
                if (highs[i] <= highs[i-1] * 1.001 and lows[i] >= lows[i-1] * 0.999):
                    patterns.append(self._create_pattern(
                        name='Inside Bar',
                        type=PatternType.DOUBLE,
                        direction='NEUTRAL',
                        confidence=0.6,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                # OUTSIDE BAR
                if (highs[i] > highs[i-1] and lows[i] < lows[i-1]):
                    direction = 'BULL' if closes[i] > opens[i] else 'BEAR'
                    patterns.append(self._create_pattern(
                        name='Outside Bar',
                        type=PatternType.DOUBLE,
                        direction=direction,
                        confidence=0.65,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # TAKER (Bullish/Bearish)
                if i > 0 and closes[i] > highs[i-1]:
                    patterns.append(self._create_pattern(
                        name='Taker Bullish',
                        type=PatternType.DOUBLE,
                        direction='BULL',
                        confidence=0.7,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                if i > 0 and closes[i] < lows[i-1]:
                    patterns.append(self._create_pattern(
                        name='Taker Bearish',
                        type=PatternType.DOUBLE,
                        direction='BEAR',
                        confidence=0.7,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # TWEEZER TOP
                if (i > 0 and abs(highs[i] - highs[i-1]) / highs[i] < 0.001 and
                    closes[i] < opens[i] and closes[i-1] > opens[i-1]):
                    patterns.append(self._create_pattern(
                        name='Tweezer Top',
                        type=PatternType.DOUBLE,
                        direction='BEAR',
                        confidence=0.75,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # TWEEZER BOTTOM
                if (i > 0 and abs(lows[i] - lows[i-1]) / lows[i] < 0.001 and
                    closes[i] > opens[i] and closes[i-1] < opens[i-1]):
                    patterns.append(self._create_pattern(
                        name='Tweezer Bottom',
                        type=PatternType.DOUBLE,
                        direction='BULL',
                        confidence=0.75,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # HARAMI (Pregnant) - Bullish
                if (i > 0 and bodies[i] < bodies[i-1] * 0.5 and
                    opens[i] > closes[i-1] and closes[i] < opens[i-1]):
                    patterns.append(self._create_pattern(
                        name='Harami Bullish',
                        type=PatternType.DOUBLE,
                        direction='BULL',
                        confidence=0.7,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # HARAMI - Bearish
                if (i > 0 and bodies[i] < bodies[i-1] * 0.5 and
                    opens[i] < closes[i-1] and closes[i] > opens[i-1]):
                    patterns.append(self._create_pattern(
                        name='Harami Bearish',
                        type=PatternType.DOUBLE,
                        direction='BEAR',
                        confidence=0.7,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # ========== TRIPLE CANDLE PATTERNS ==========
            for i in range(2, len(df) - 1):
                if i < len(df) - 5:
                    continue
                
                # MORNING STAR (Bullish Reversal)
                # Allow for second candle to be very small
                if (closes[i-2] < opens[i-2] and  # First bearish
                    bodies[i-1] < avg_body * 0.7 and  # Second small body
                    closes[i] > opens[i] and  # Third bullish
                    closes[i] > (opens[i-2] + closes[i-2]) / 2):  # Closes above midpoint
                    
                    patterns.append(self._create_pattern(
                        name='Morning Star',
                        type=PatternType.TRIPLE,
                        direction='BULL',
                        confidence=0.8,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                # EVENING STAR (Bearish Reversal)
                # Allow for second candle to be very small
                if (closes[i-2] > opens[i-2] and  # First bullish
                    bodies[i-1] < avg_body * 0.7 and  # Second small body
                    closes[i] < opens[i] and  # Third bearish
                    closes[i] < (opens[i-2] + closes[i-2]) / 2):  # Closes below midpoint
                    
                    patterns.append(self._create_pattern(
                        name='Evening Star',
                        type=PatternType.TRIPLE,
                        direction='BEAR',
                        confidence=0.8,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # THREE WHITE SOLDIERS (Bullish Continuation)
                if (i >= 3 and
                    closes[i-2] > opens[i-2] and  # All bullish
                    closes[i-1] > opens[i-1] and
                    closes[i] > opens[i] and
                    closes[i-2] <= closes[i-1] <= closes[i] and  # Higher closes (allow equal)
                    bodies[i-2] > avg_body * 0.5 and  # Decent size (lowered from 0.7)
                    upper_wick[i-2] < bodies[i-2] * 0.5 and  # Small wicks (relaxed from 0.3)
                    upper_wick[i-1] < bodies[i-1] * 0.5 and
                    upper_wick[i] < bodies[i] * 0.5):
                    
                    patterns.append(self._create_pattern(
                        name='Three White Soldiers',
                        type=PatternType.TRIPLE,
                        direction='BULL',
                        confidence=0.85,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                    
            
                # THREE BLACK CROWS (Bearish Continuation)
                if (i >= 3 and
                    closes[i-2] < opens[i-2] and  # All bearish
                    closes[i-1] < opens[i-1] and
                    closes[i] < opens[i] and
                    closes[i-2] >= closes[i-1] >= closes[i] and  # Lower closes (allow equal)
                    bodies[i-2] > avg_body * 0.5 and  # Decent size (lowered from 0.7)
                    lower_wick[i-2] < bodies[i-2] * 0.5 and  # Small wicks (relaxed from 0.3)
                    lower_wick[i-1] < bodies[i-1] * 0.5 and
                    lower_wick[i] < bodies[i] * 0.5):
                    
                    patterns.append(self._create_pattern(
                        name='Three Black Crows',
                        type=PatternType.TRIPLE,
                        direction='BEAR',
                        confidence=0.85,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                # THREE INSIDE UP (Bullish)
                if (i >= 2 and
                    closes[i-2] < opens[i-2] and  # First bearish
                    closes[i-1] < opens[i-1] and  # Second inside
                    highs[i-1] < highs[i-2] and lows[i-1] > lows[i-2] and
                    closes[i] > opens[i] and  # Third bullish
                    closes[i] > highs[i-2]):  # Breaks out
                    
                    patterns.append(self._create_pattern(
                        name='Three Inside Up',
                        type=PatternType.TRIPLE,
                        direction='BULL',
                        confidence=0.75,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # THREE INSIDE DOWN (Bearish)
                if (i >= 2 and
                    closes[i-2] > opens[i-2] and  # First bullish
                    closes[i-1] > opens[i-1] and  # Second inside
                    highs[i-1] < highs[i-2] and lows[i-1] > lows[i-2] and
                    closes[i] < opens[i] and  # Third bearish
                    closes[i] < lows[i-2]):  # Breaks down
                    
                    patterns.append(self._create_pattern(
                        name='Three Inside Down',
                        type=PatternType.TRIPLE,
                        direction='BEAR',
                        confidence=0.75,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # ========== MULTI-CANDLE PATTERNS ==========
            for i in range(3, len(df) - 1):
                if i < len(df) - 10:
                    continue
                
                # RISING THREE METHODS (Bullish Continuation)
                if (i >= 4 and
                    closes[i-4] > opens[i-4] and  # First bullish
                    all(closes[j] < opens[j] for j in range(i-3, i)) and  # 3 small bearish
                    closes[i] > opens[i] and  # Final bullish
                    closes[i] > highs[i-4]):  # Breaks above
                    
                    patterns.append(self._create_pattern(
                        name='Rising Three Methods',
                        type=PatternType.MULTI,
                        direction='BULL',
                        confidence=0.8,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # FALLING THREE METHODS (Bearish Continuation)
                if (i >= 4 and
                    closes[i-4] < opens[i-4] and  # First bearish
                    all(closes[j] > opens[j] for j in range(i-3, i)) and  # 3 small bullish
                    closes[i] < opens[i] and  # Final bearish
                    closes[i] < lows[i-4]):  # Breaks below
                    
                    patterns.append(self._create_pattern(
                        name='Falling Three Methods',
                        type=PatternType.MULTI,
                        direction='BEAR',
                        confidence=0.8,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # THREE LINE STRIKE (Bullish)
                if (i >= 4 and
                    closes[i-4] < opens[i-4] and  # First bearish
                    closes[i-3] < opens[i-3] and  # Second bearish
                    closes[i-2] < opens[i-2] and  # Third bearish
                    closes[i-1] < opens[i-1] and  # Fourth bearish
                    closes[i] > opens[i] and  # Fifth bullish
                    closes[i] > highs[i-4]):  # Reverses all
                    
                    patterns.append(self._create_pattern(
                        name='Three Line Strike',
                        type=PatternType.MULTI,
                        direction='BULL',
                        confidence=0.8,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # IDENTICAL THREE CROWS (Bearish)
                if (i >= 3 and
                    closes[i-3] < opens[i-3] and  # All bearish
                    closes[i-2] < opens[i-2] and
                    closes[i-1] < opens[i-1] and
                    abs(closes[i-3] - closes[i-2]) < bodies[i-3] * 0.2 and  # Similar closes
                    abs(closes[i-2] - closes[i-1]) < bodies[i-2] * 0.2):
                    
                    patterns.append(self._create_pattern(
                        name='Identical Three Crows',
                        type=PatternType.MULTI,
                        direction='BEAR',
                        confidence=0.8,
                        index=i-1,
                        df=df,
                        vol_ratio=vol_ratio[i-1],
                        atr=atr[i-1]
                    ))
            
            # ========== RARE / INSTITUTIONAL PATTERNS ==========
            
            # KICKER PATTERN (Strong reversal)
            for i in range(1, len(df) - 1):
                if i < len(df) - 5:
                    continue
                
                # Bullish Kicker
                if (closes[i-1] < opens[i-1] and  # Previous bearish
                    opens[i] > closes[i-1] * 1.005 and  # Gap up (1.01 -> 1.005)
                    closes[i] > opens[i] and  # Current bullish
                    closes[i] > opens[i-1]):  # Strong close
                    
                    patterns.append(self._create_pattern(
                        name='Kicker Bullish',
                        type=PatternType.RARE,
                        direction='BULL',
                        confidence=0.9,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # Bearish Kicker
                if (closes[i-1] > opens[i-1] and  # Previous bullish
                    opens[i] < closes[i-1] * 0.995 and  # Gap down (0.99 -> 0.995)
                    closes[i] < opens[i] and  # Current bearish
                    closes[i] < opens[i-1]):  # Strong close
                    
                    patterns.append(self._create_pattern(
                        name='Kicker Bearish',
                        type=PatternType.RARE,
                        direction='BEAR',
                        confidence=0.9,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
              
            
            # ISLAND REVERSAL
            for i in range(1, len(df) - 2):
                if i < len(df) - 10:
                    continue
                
                # Check for gap before and after
                gap_before = (lows[i] > highs[i-1]) or (highs[i] < lows[i-1])
                gap_after = (lows[i+1] > highs[i]) or (highs[i+1] < lows[i])
                
                if gap_before and gap_after:
                    # Island of candles
                    island_direction = 'BULL' if closes[i] > opens[i] else 'BEAR'
                    patterns.append(self._create_pattern(
                        name='Island Reversal',
                        type=PatternType.RARE,
                        direction=island_direction,
                        confidence=0.85,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # PARABOLIC BLOW-OFF (Exhaustion)
            for i in range(len(df) - 1):
                if i < 20 or i < len(df) - 10:
                    continue
                
                # Check for extreme move with volume climax
                recent_returns = (closes[i] - closes[i-10]) / closes[i-10]
                vol_climax = vol_ratio[i] > 3.0
                
                if abs(recent_returns) > 0.15 and vol_climax:  # 15% move in 10 bars
                    direction = 'BEAR' if recent_returns > 0 else 'BULL'  # Reversal after climax
                    patterns.append(self._create_pattern(
                        name='Parabolic Blow-Off',
                        type=PatternType.RARE,
                        direction=direction,
                        confidence=0.85,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # MAT HOLD PATTERN
            for i in range(4, len(df) - 1):
                if i < len(df) - 10:
                    continue
                
                # First candle strong bullish
                if (closes[i-4] > opens[i-4] and 
                    bodies[i-4] > avg_body * 1.5 and
                    vol_ratio[i-4] > 1.5):
                    
                    # 3-4 small pullback candles
                    pullback_valid = True
                    for j in range(i-3, i):
                        if closes[j] > closes[i-4] * 0.98:  # Stay within range
                            pullback_valid = False
                            break
                    
                    # Final breakout candle
                    if (pullback_valid and closes[i] > opens[i] and
                        closes[i] > highs[i-4] and vol_ratio[i] > 1.5):
                        
                        patterns.append(self._create_pattern(
                            name='Mat Hold',
                            type=PatternType.RARE,
                            direction='BULL',
                            confidence=0.85,
                            index=i,
                            df=df,
                            vol_ratio=vol_ratio[i],
                            atr=atr[i]
                        ))
            
            # HOOK REVERSAL
            for i in range(1, len(df) - 1):
                if i < len(df) - 5:
                    continue
                
                # Higher high but lower close
                if (highs[i] > highs[i-1] and closes[i] < closes[i-1] and
                    closes[i] < opens[i]):  # Bearish hook
                    
                    patterns.append(self._create_pattern(
                        name='Hook Reversal Bearish',
                        type=PatternType.RARE,
                        direction='BEAR',
                        confidence=0.7,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # Lower low but higher close
                if (lows[i] < lows[i-1] and closes[i] > closes[i-1] and
                    closes[i] > opens[i]):  # Bullish hook
                    
                    patterns.append(self._create_pattern(
                        name='Hook Reversal Bullish',
                        type=PatternType.RARE,
                        direction='BULL',
                        confidence=0.7,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # DEAD CAT BOUNCE (After crash)
            for i in range(2, len(df) - 1):
                if i < 20:
                    continue
                
                # Sharp drop
                if closes[i-2] < closes[i-20] * 0.85:  # 15% drop
                    # Small bounce
                    if (closes[i] > closes[i-2] and closes[i] < closes[i-2] * 1.05):
                        patterns.append(self._create_pattern(
                            name='Dead Cat Bounce',
                            type=PatternType.RARE,
                            direction='BEAR',
                            confidence=0.75,
                            index=i,
                            df=df,
                            vol_ratio=vol_ratio[i],
                            atr=atr[i]
                        ))
            
            # V-BOTTOM / V-TOP
            for i in range(5, len(df) - 1):
                if i < 20:
                    continue
                
                # V-Bottom: Sharp down then sharp up
                down_move = (closes[i-5] - closes[i-3]) / closes[i-5] if closes[i-5] > 0 else 0
                up_move = (closes[i] - closes[i-3]) / closes[i-3] if closes[i-3] > 0 else 0
                
                if down_move > 0.05 and up_move > 0.05 and abs(up_move - down_move) < 0.02:
                    patterns.append(self._create_pattern(
                        name='V-Bottom',
                        type=PatternType.RARE,
                        direction='BULL',
                        confidence=0.8,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
                
                # V-Top: Sharp up then sharp down
                up_move = (closes[i-3] - closes[i-5]) / closes[i-5] if closes[i-5] > 0 else 0
                down_move = (closes[i-3] - closes[i]) / closes[i-3] if closes[i-3] > 0 else 0
                
                if up_move > 0.05 and down_move > 0.05 and abs(up_move - down_move) < 0.02:
                    patterns.append(self._create_pattern(
                        name='V-Top',
                        type=PatternType.RARE,
                        direction='BEAR',
                        confidence=0.8,
                        index=i,
                        df=df,
                        vol_ratio=vol_ratio[i],
                        atr=atr[i]
                    ))
            
            # ========== PATTERN COMBINATIONS ==========
            patterns = self._detect_pattern_combinations(patterns)
            
            # Sort by confidence
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            return patterns
            
        except Exception as e:
            print(f"Error detecting patterns: {e}")
            return []
    
    def _create_pattern(self, name: str, type: PatternType, direction: str,
                       confidence: float, index: int, df: pd.DataFrame,
                       vol_ratio: float, atr: float) -> CandlePattern:
        """Create a pattern object with all metadata"""
        
        # Determine strength
        if confidence >= 0.85:
            strength = PatternStrength.VERY_STRONG
        elif confidence >= 0.75:
            strength = PatternStrength.STRONG
        elif confidence >= 0.65:
            strength = PatternStrength.MODERATE
        elif confidence >= 0.5:
            strength = PatternStrength.WEAK
        else:
            strength = PatternStrength.VERY_WEAK
        
        # Volume confirmation
        volume_confirmed = vol_ratio > 1.2
        
        # Trend alignment (simplified - can be enhanced)
        if 'ema_8' in df.columns and 'ema_21' in df.columns:
            ema8 = df['ema_8'].iloc[-1]
            ema21 = df['ema_21'].iloc[-1]
            price = df['close'].iloc[-1]
            
            if direction == 'BULL' and price > ema8 > ema21:
                trend_aligned = True
            elif direction == 'BEAR' and price < ema8 < ema21:
                trend_aligned = True
            else:
                trend_aligned = False
        else:
            trend_aligned = False
        
        # Entry/Stop/Target logic based on pattern
        entry_logic, stop_logic, target_logic = self._get_trade_logic(name, direction)
        
        return CandlePattern(
            name=name,
            type=type,
            direction=direction,
            confidence=min(0.95, confidence),
            strength=strength,
            pattern_type=type.value,
            candles_involved=3 if type == PatternType.TRIPLE else (2 if type == PatternType.DOUBLE else 1),
            volume_confirmed=volume_confirmed,
            trend_aligned=trend_aligned,
            description=self._get_pattern_description(name),
            entry_logic=entry_logic,
            stop_logic=stop_logic,
            target_logic=target_logic,
            indicators_used=['volume', 'atr', 'price_action'],
            raw_data={
                'index': index,
                'price': float(df['close'].iloc[index]) if index < len(df) else 0,
                'timestamp': df.index[index].isoformat() if index < len(df) and hasattr(df.index[index], 'isoformat') else str(df.index[-1] if len(df) > 0 else ''),
                'vol_ratio': float(vol_ratio),
                'atr': float(atr)
            }
        )
    
    def _get_pattern_description(self, name: str) -> str:
        """Get description for pattern"""
        descriptions = {
            'Hammer': 'Bullish reversal at support, long lower wick shows rejection of lows',
            'Shooting Star': 'Bearish reversal at resistance, long upper wick shows rejection of highs',
            'Doji': 'Indecision, market in equilibrium',
            'Dragonfly Doji': 'Bullish reversal, long lower wick shows buying pressure',
            'Gravestone Doji': 'Bearish reversal, long upper wick shows selling pressure',
            'Long-Legged Doji': 'Extreme indecision, battle between buyers and sellers',
            'Marubozu': 'Strong momentum with no wicks',
            'Spinning Top': 'Indecision with small body',
            'Long Candle': 'Strong momentum candle',
            'Bullish Engulfing': 'Buyers overwhelm sellers, strong bullish reversal',
            'Bearish Engulfing': 'Sellers overwhelm buyers, strong bearish reversal',
            'Inside Bar': 'Consolidation, breakout imminent',
            'Outside Bar': 'Increased volatility, potential reversal',
            'Taker Bullish': 'Price takes out previous high, strong buying',
            'Taker Bearish': 'Price takes out previous low, strong selling',
            'Tweezer Top': 'Double top rejection, bearish reversal',
            'Tweezer Bottom': 'Double bottom support, bullish reversal',
            'Harami Bullish': 'Small body inside large bearish candle, potential reversal',
            'Harami Bearish': 'Small body inside large bullish candle, potential reversal',
            'Morning Star': 'Three-candle bullish reversal after downtrend',
            'Evening Star': 'Three-candle bearish reversal after uptrend',
            'Three White Soldiers': 'Strong bullish continuation with three large bullish candles',
            'Three Black Crows': 'Strong bearish continuation with three large bearish candles',
            'Three Inside Up': 'Bullish reversal with inside bar setup',
            'Three Inside Down': 'Bearish reversal with inside bar setup',
            'Rising Three Methods': 'Bullish continuation with pullback',
            'Falling Three Methods': 'Bearish continuation with pullback',
            'Three Line Strike': 'Bearish trap then bullish reversal',
            'Identical Three Crows': 'Strong bearish momentum',
            'Kicker Bullish': 'Gap up with strong momentum, powerful reversal',
            'Kicker Bearish': 'Gap down with strong momentum, powerful reversal',
            'Island Reversal': 'Price isolated by gaps, significant reversal',
            'Parabolic Blow-Off': 'Exhaustion move, reversal imminent',
            'Mat Hold': 'Bullish continuation with pullback',
            'Hook Reversal Bullish': 'Lower low but higher close, bullish reversal',
            'Hook Reversal Bearish': 'Higher high but lower close, bearish reversal',
            'Dead Cat Bounce': 'Temporary bounce in downtrend, selling opportunity',
            'V-Bottom': 'Sharp reversal from bottom',
            'V-Top': 'Sharp reversal from top'
        }
        return descriptions.get(name, f'{name} candlestick pattern')
    
    def _get_trade_logic(self, name: str, direction: str) -> Tuple[str, str, str]:
        """Get entry, stop, and target logic for pattern"""
        
        if direction == 'BULL':
            entry = 'Enter on close above pattern high'
            stop = 'Stop below pattern low'
            target = 'Target: 2x risk or previous resistance'
        else:
            entry = 'Enter on close below pattern low'
            stop = 'Stop above pattern high'
            target = 'Target: 2x risk or previous support'
        
        # Special cases
        if 'Engulfing' in name:
            entry = 'Enter on close of engulfing candle'
        elif 'Morning' in name or 'Evening' in name:
            entry = 'Enter on close of third candle'
        elif 'Kicker' in name:
            entry = 'Enter on open of kicker candle'
        elif 'Island' in name:
            entry = 'Enter on breakout from island'
        
        return entry, stop, target
    
    def _detect_pattern_combinations(self, patterns: List[CandlePattern]) -> List[CandlePattern]:
        """Detect when multiple patterns occur together"""
        if len(patterns) < 2:
            return patterns
        
        # Group by direction and time
        bull_patterns = [p for p in patterns if p.direction == 'BULL']
        bear_patterns = [p for p in patterns if p.direction == 'BEAR']
        
        # If multiple bullish patterns in last 5 candles, boost confidence
        recent_bull = [p for p in bull_patterns if p.raw_data.get('index', 0) > len(patterns) - 5]
        if len(recent_bull) >= 2:
            for p in recent_bull:
                p.confidence = min(0.95, p.confidence + 0.05)
                p.description += " (Confirmed by multiple bullish patterns)"
        
        recent_bear = [p for p in bear_patterns if p.raw_data.get('index', 0) > len(patterns) - 5]
        if len(recent_bear) >= 2:
            for p in recent_bear:
                p.confidence = min(0.95, p.confidence + 0.05)
                p.description += " (Confirmed by multiple bearish patterns)"
        
        return patterns
    
    # ========== NEW METHODS FOR LAYER SCORING INTEGRATION ==========
    
    def get_pattern_summary(self, df: pd.DataFrame, 
                            regime: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get pattern summary for scoring layer
        
        Args:
            df: DataFrame with OHLCV data
            regime: Optional regime dictionary from regime detection
        
        Returns:
            Dictionary with pattern statistics and scores
        """
        patterns = self.detect_all_patterns(df)
        
        if not patterns:
            return {
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'total_patterns': 0,
                'avg_bullish_confidence': 0,
                'avg_bearish_confidence': 0,
                'net_score': 0,
                'best_pattern': None,
                'best_pattern_confidence': 0,
                'pattern_list': []
            }
        
        # Count bullish and bearish
        bullish = [p for p in patterns if p.direction == 'BULL']
        bearish = [p for p in patterns if p.direction == 'BEAR']
        neutral = [p for p in patterns if p.direction == 'NEUTRAL']
        
        # Calculate average confidences
        bullish_conf = np.mean([p.confidence for p in bullish]) if bullish else 0
        bearish_conf = np.mean([p.confidence for p in bearish]) if bearish else 0
        
        # Net score (-1 to 1)
        total_bullish_score = sum([p.confidence for p in bullish])
        total_bearish_score = sum([p.confidence for p in bearish])
        total_score = total_bullish_score + total_bearish_score
        net_score = (total_bullish_score - total_bearish_score) / total_score if total_score > 0 else 0
        
        # Best pattern
        best = max(patterns, key=lambda x: x.confidence) if patterns else None
        
        # Apply regime adjustment if provided
        if regime:
            regime_name = regime.get('regime', 'UNKNOWN')
            if 'BULL' in regime_name:
                # In bull market, bullish patterns are stronger
                net_score = min(1.0, net_score * 1.2)
            elif 'BEAR' in regime_name:
                # In bear market, bearish patterns are stronger
                net_score = max(-1.0, net_score * 1.2)
        
        return {
            'bullish_count': len(bullish),
            'bearish_count': len(bearish),
            'neutral_count': len(neutral),
            'total_patterns': len(patterns),
            'avg_bullish_confidence': round(bullish_conf, 3),
            'avg_bearish_confidence': round(bearish_conf, 3),
            'net_score': round(net_score, 3),
            'best_pattern': best.name if best else None,
            'best_pattern_confidence': best.confidence if best else 0,
            'pattern_list': [
                {
                    'name': p.name,
                    'direction': p.direction,
                    'confidence': p.confidence,
                    'volume_confirmed': p.volume_confirmed,
                    'trend_aligned': p.trend_aligned,
                    'strength': p.strength.value if hasattr(p.strength, 'value') else p.strength
                }
                for p in patterns[:10]  # Top 10 patterns
            ]
        }
    
    def get_pattern_score(self, df: pd.DataFrame, direction: str = 'BULL') -> float:
        """
        Get aggregate pattern score for a specific direction (0-1)
        
        Args:
            df: DataFrame with OHLCV data
            direction: 'BULL' or 'BEAR'
        
        Returns:
            Score between 0 and 1
        """
        patterns = self.detect_all_patterns(df)
        
        if not patterns:
            return 0.5
        
        if direction == 'BULL':
            bullish = [p for p in patterns if p.direction == 'BULL']
            if bullish:
                # Weighted by confidence, with bonus for multiple patterns
                total_confidence = sum(p.confidence for p in bullish)
                count_bonus = min(0.2, len(bullish) * 0.05)  # Small bonus for multiple patterns
                return min(1.0, (total_confidence / len(bullish)) * 1.2 + count_bonus)
            return 0.3
        else:
            bearish = [p for p in patterns if p.direction == 'BEAR']
            if bearish:
                total_confidence = sum(p.confidence for p in bearish)
                count_bonus = min(0.2, len(bearish) * 0.05)
                return min(1.0, (total_confidence / len(bearish)) * 1.2 + count_bonus)
            return 0.3
    
    def has_strong_pattern(self, df: pd.DataFrame, min_confidence: float = 0.75) -> Dict[str, Any]:
        """
        Check if there are strong patterns (confidence >= min_confidence)
        
        Args:
            df: DataFrame with OHLCV data
            min_confidence: Minimum confidence threshold for strong pattern
        
        Returns:
            Dictionary with strong pattern information
        """
        patterns = self.detect_all_patterns(df)
        
        strong_bull = [p for p in patterns if p.direction == 'BULL' and p.confidence >= min_confidence]
        strong_bear = [p for p in patterns if p.direction == 'BEAR' and p.confidence >= min_confidence]
        
        return {
            'has_strong_bull': len(strong_bull) > 0,
            'has_strong_bear': len(strong_bear) > 0,
            'strong_bull_count': len(strong_bull),
            'strong_bear_count': len(strong_bear),
            'strong_bull_patterns': [p.name for p in strong_bull],
            'strong_bear_patterns': [p.name for p in strong_bear],
            'best_bull': max(strong_bull, key=lambda x: x.confidence).name if strong_bull else None,
            'best_bear': max(strong_bear, key=lambda x: x.confidence).name if strong_bear else None,
            'best_bull_confidence': max([p.confidence for p in strong_bull]) if strong_bull else 0,
            'best_bear_confidence': max([p.confidence for p in strong_bear]) if strong_bear else 0,
            'best_bull_reliability': max([p.confidence for p in strong_bull]) if strong_bull else 0,
            'best_bear_reliability': max([p.confidence for p in strong_bear]) if strong_bear else 0
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
         weighted_score = p.confidence
         if p.direction == 'BULL':
             buy_score += weighted_score
         elif p.direction == 'BEAR':
             sell_score += weighted_score
    
     total = buy_score + sell_score
     if total == 0:
         return 0.0
    
     net_bias = (buy_score - sell_score) / total
    
    # Apply regime adjustment if provided
     if regime:
         regime_name = regime.get('regime', 'UNKNOWN')
         if 'BULL' in regime_name:
             net_bias = min(1.0, net_bias * 1.2)
         elif 'BEAR' in regime_name:
             net_bias = max(-1.0, net_bias * 1.2)
    
     return round(net_bias, 3)

# ========== LEGACY FUNCTION FOR BACKWARD COMPATIBILITY ==========

def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility
    Returns list of pattern dicts
    """
    detector = AdvancedPriceAction()
    patterns = detector.detect_all_patterns(df)
    
    result = []
    for p in patterns:
        result.append({
            'name': p.name,
            'confidence': p.confidence,
            'direction': p.direction,
            'type': p.pattern_type,
            'volume_confirmed': p.volume_confirmed,
            'trend_aligned': p.trend_aligned,
            'description': p.description,
            'entry_logic': p.entry_logic,
            'stop_logic': p.stop_logic,
            'target_logic': p.target_logic
        })
    
    return result