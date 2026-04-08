"""
trap_engine.py
Layer 4: Trap Engine for Price Action Expert V3.5

Advanced trap detection using pure price action:
- Bull Trap: Break above resistance, close below
- Bear Trap: Break below support, close above
- Stop Hunt: Long wick through level, immediate reversal
- Liquidity Sweep: Wick through multiple key levels
- Trap Severity Scoring: Minor/Medium/Strong/Extreme
- Optional integration with liquidity.py and liquidation_intelligence.py

This is the EDGE layer — identifies where smart money is trapping retail traders
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import configuration
from .price_action_config import (
    TRAP_WICK_BODY_RATIO,
    TRAP_CLOSE_POSITION_HIGH,
    TRAP_CLOSE_POSITION_LOW,
    TRAP_REVERSAL_CANDLES_MAX,
    STOP_HUNT_WICK_ATR_MULTIPLIER,
    STOP_HUNT_VOLUME_RATIO,
    LIQUIDITY_SWEEP_DEPTH_ATR,
    LIQUIDITY_SWEEP_MULTIPLE_COUNT,
    TRAP_SEVERITY,
    TRAP_SEVERITY_WEIGHTS,
    LEVEL_IMPORTANCE_SCORES
)

# Import candle analyzer
from .candle_analyzer import CandleData, CandleAnalyzer


class TrapType(Enum):
    """Types of traps detected"""
    BULL_TRAP = "bull_trap"
    BEAR_TRAP = "bear_trap"
    STOP_HUNT_BULLISH = "stop_hunt_bullish"
    STOP_HUNT_BEARISH = "stop_hunt_bearish"
    LIQUIDITY_SWEEP_BULLISH = "liquidity_sweep_bullish"
    LIQUIDITY_SWEEP_BEARISH = "liquidity_sweep_bearish"


class TrapSeverity(Enum):
    """Severity level of detected trap"""
    NONE = "none"
    MINOR = "minor"
    MEDIUM = "medium"
    STRONG = "strong"
    EXTREME = "extreme"


@dataclass
class DetectedTrap:
    """
    Complete trap detection result
    """
    # Core identification
    type: TrapType
    direction: str                      # 'BULLISH' or 'BEARISH' (after trap)
    severity: TrapSeverity
    severity_score: float               # 0-1
    confidence: float                   # 0-1
    
    # Detection details
    candle_index: int
    trap_price: float
    reversal_price: float
    
    # Liquidity sweep details
    liquidity_sweep: bool
    sweep_levels: List[float] = field(default_factory=list)
    sweep_count: int = 0
    sweep_depth_atr: float = 0.0
    
    # Volume confirmation
    volume_ratio_breakout: float = 0.0
    volume_ratio_reversal: float = 0.0
    
    # Reversal speed
    reversal_candles: int = 1
    
    # Level importance
    level_importance: float = 0.0
    level_type: str = ""                # 'resistance', 'support', 'swing_high', 'swing_low'
    
    # Human-readable
    description: str = ""
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            'type': self.type.value,
            'direction': self.direction,
            'severity': self.severity.value,
            'severity_score': round(self.severity_score, 3),
            'confidence': round(self.confidence, 3),
            'trap_price': self.trap_price,
            'reversal_price': self.reversal_price,
            'liquidity_sweep': self.liquidity_sweep,
            'sweep_count': self.sweep_count,
            'sweep_depth_atr': round(self.sweep_depth_atr, 2),
            'reversal_candles': self.reversal_candles,
            'description': self.description,
            'reasons': self.reasons[:5]
        }


class TrapEngine:
    """
    Advanced trap detection engine
    
    Features:
    - Bull/Bear trap detection with key levels
    - Stop hunt detection (liquidity sweeps)
    - Multi-level liquidity sweep detection
    - Severity scoring with multiple factors
    - Optional integration with existing liquidity modules
    """
    
    def __init__(self):
        """Initialize the trap engine"""
        self.candle_analyzer = CandleAnalyzer()
        self.trap_history: List[DetectedTrap] = []
    
    # =========================================================
    # BULL TRAP DETECTION
    # =========================================================
    
    def detect_bull_trap(self, candles: List[CandleData], idx: int,
                         resistance_level: Optional[float] = None,
                         atr: float = 0,
                         recent_highs: Optional[List[float]] = None) -> Optional[DetectedTrap]:
        """
        Detect Bull Trap: Price breaks above resistance but closes below
        
        Bull Trap indicates:
        - Smart money is selling into the breakout
        - Retail traders who bought are trapped
        - Expected direction after trap: BEARISH
        
        Args:
            candles: List of candle data
            idx: Index of the reversal candle
            resistance_level: Optional pre-calculated resistance level
            atr: ATR value for normalization
            recent_highs: Optional list of recent swing highs for liquidity sweep detection
        
        Returns:
            DetectedTrap or None
        """
        if idx < 1 or len(candles) <= idx:
            return None
        
        breakout_candle = candles[idx - 1]
        reversal_candle = candles[idx]
        
        # Determine resistance level
        if resistance_level is not None:
            resistance = resistance_level
        else:
            # Use recent swing high if available
            if recent_highs and len(recent_highs) >= 2:
                resistance = max(recent_highs[-2:])
            else:
                # Fallback: use previous candle high as reference
                if idx >= 10:
                    resistance = max(c.high for c in candles[idx-10:idx-1])
                else:
                    return None
        
        # Check breakout: price broke above resistance
        if not (breakout_candle.high > resistance):
            return None
        
        # Check trap: reversal candle closes below resistance
        if not (reversal_candle.close < resistance):
            return None
        
        # Calculate trap score
        trap_score = 0.5
        reasons = []
        sweep_levels = []
        
        # ===== Factor 1: Wick length (rejection strength) =====
        wick_length = breakout_candle.upper_wick
        wick_atr = wick_length / (atr + 1e-8)
        
        if wick_atr >= 1.0:
            trap_score += 0.15
            reasons.append(f'Long upper wick on breakout ({wick_atr:.1f}x ATR)')
        elif wick_atr >= 0.5:
            trap_score += 0.08
            reasons.append(f'Upper wick on breakout ({wick_atr:.1f}x ATR)')
        
        # ===== Factor 2: Reversal speed =====
        if reversal_candle.is_bearish and reversal_candle.close < breakout_candle.open:
            trap_score += 0.15
            reasons.append('Immediate bearish reversal')
        elif reversal_candle.close < reversal_candle.open:
            trap_score += 0.08
            reasons.append('Bearish reversal candle')
        
        # ===== Factor 3: Volume confirmation =====
        if breakout_candle.volume_ratio >= STOP_HUNT_VOLUME_RATIO:
            trap_score += 0.12
            reasons.append(f'Volume spike on breakout: {breakout_candle.volume_ratio:.1f}x')
        elif breakout_candle.volume_ratio >= 1.5:
            trap_score += 0.06
            reasons.append(f'Above average volume: {breakout_candle.volume_ratio:.1f}x')
        
        if reversal_candle.volume_ratio >= 1.5:
            trap_score += 0.08
            reasons.append(f'Volume on reversal: {reversal_candle.volume_ratio:.1f}x')
        
        # ===== Factor 4: Liquidity sweep detection =====
        liquidity_sweep = False
        sweep_count = 0
        sweep_depth_atr = 0.0
        
        if recent_highs:
            for high in recent_highs[-3:]:
                if breakout_candle.high > high:
                    sweep_count += 1
                    sweep_levels.append(high)
            
            if sweep_count >= 1:
                liquidity_sweep = True
                trap_score += 0.05 * min(3, sweep_count)
                reasons.append(f'Swept {sweep_count} recent high(s)')
                
                # Calculate sweep depth
                deepest_sweep = max([h for h in recent_highs[-3:] if breakout_candle.high > h], default=0)
                if deepest_sweep > 0:
                    sweep_depth_atr = (breakout_candle.high - deepest_sweep) / (atr + 1e-8)
                    if sweep_depth_atr >= LIQUIDITY_SWEEP_DEPTH_ATR:
                        trap_score += 0.08
                        reasons.append(f'Deep sweep: {sweep_depth_atr:.1f}x ATR')
        
        # ===== Factor 5: Level importance =====
        level_importance = 0.5
        if resistance_level is not None:
            # Check if this level has been tested multiple times
            level_touches = 0
            for c in candles[max(0, idx-50):idx]:
                if abs(c.high - resistance) / resistance <= 0.002:
                    level_touches += 1
            
            if level_touches >= 3:
                level_importance = 0.9
                trap_score += 0.10
                reasons.append(f'Key resistance (touched {level_touches}x)')
            elif level_touches >= 2:
                level_importance = 0.7
                trap_score += 0.05
                reasons.append(f'Important resistance (touched {level_touches}x)')
        
        # Cap trap score
        trap_score = min(0.95, trap_score)
        
        # Determine severity
        severity, severity_label = self._get_severity(trap_score)
        
        # Calculate confidence
        confidence = 0.5 + (trap_score * 0.5)
        
        # Build description
        description = f"Bull Trap: Price broke above {resistance:.4f} but closed below. "
        description += f"Severity: {severity_label.upper()}. "
        description += "Smart money selling into breakout."
        
        return DetectedTrap(
            type=TrapType.BULL_TRAP,
            direction='BEARISH',
            severity=severity,
            severity_score=trap_score,
            confidence=min(0.95, confidence),
            candle_index=reversal_candle.index,
            trap_price=breakout_candle.high,
            reversal_price=reversal_candle.close,
            liquidity_sweep=liquidity_sweep,
            sweep_levels=sweep_levels,
            sweep_count=sweep_count,
            sweep_depth_atr=sweep_depth_atr,
            volume_ratio_breakout=breakout_candle.volume_ratio,
            volume_ratio_reversal=reversal_candle.volume_ratio,
            reversal_candles=1,
            level_importance=level_importance,
            level_type='resistance',
            description=description,
            reasons=reasons
        )
    
    # =========================================================
    # BEAR TRAP DETECTION
    # =========================================================
    
    def detect_bear_trap(self, candles: List[CandleData], idx: int,
                         support_level: Optional[float] = None,
                         atr: float = 0,
                         recent_lows: Optional[List[float]] = None) -> Optional[DetectedTrap]:
        """
        Detect Bear Trap: Price breaks below support but closes above
        
        Bear Trap indicates:
        - Smart money is buying into the breakdown
        - Retail traders who sold are trapped
        - Expected direction after trap: BULLISH
        
        Args:
            candles: List of candle data
            idx: Index of the reversal candle
            support_level: Optional pre-calculated support level
            atr: ATR value for normalization
            recent_lows: Optional list of recent swing lows for liquidity sweep detection
        
        Returns:
            DetectedTrap or None
        """
        if idx < 1 or len(candles) <= idx:
            return None
        
        breakdown_candle = candles[idx - 1]
        reversal_candle = candles[idx]
        
        # Determine support level
        if support_level is not None:
            support = support_level
        else:
            if recent_lows and len(recent_lows) >= 2:
                support = min(recent_lows[-2:])
            else:
                if idx >= 10:
                    support = min(c.low for c in candles[idx-10:idx-1])
                else:
                    return None
        
        # Check breakdown: price broke below support
        if not (breakdown_candle.low < support):
            return None
        
        # Check trap: reversal candle closes above support
        if not (reversal_candle.close > support):
            return None
        
        # Calculate trap score
        trap_score = 0.5
        reasons = []
        sweep_levels = []
        
        # ===== Factor 1: Wick length (rejection strength) =====
        wick_length = breakdown_candle.lower_wick
        wick_atr = wick_length / (atr + 1e-8)
        
        if wick_atr >= 1.0:
            trap_score += 0.15
            reasons.append(f'Long lower wick on breakdown ({wick_atr:.1f}x ATR)')
        elif wick_atr >= 0.5:
            trap_score += 0.08
            reasons.append(f'Lower wick on breakdown ({wick_atr:.1f}x ATR)')
        
        # ===== Factor 2: Reversal speed =====
        if reversal_candle.is_bullish and reversal_candle.close > breakdown_candle.open:
            trap_score += 0.15
            reasons.append('Immediate bullish reversal')
        elif reversal_candle.close > reversal_candle.open:
            trap_score += 0.08
            reasons.append('Bullish reversal candle')
        
        # ===== Factor 3: Volume confirmation =====
        if breakdown_candle.volume_ratio >= STOP_HUNT_VOLUME_RATIO:
            trap_score += 0.12
            reasons.append(f'Volume spike on breakdown: {breakdown_candle.volume_ratio:.1f}x')
        elif breakdown_candle.volume_ratio >= 1.5:
            trap_score += 0.06
            reasons.append(f'Above average volume: {breakdown_candle.volume_ratio:.1f}x')
        
        if reversal_candle.volume_ratio >= 1.5:
            trap_score += 0.08
            reasons.append(f'Volume on reversal: {reversal_candle.volume_ratio:.1f}x')
        
        # ===== Factor 4: Liquidity sweep detection =====
        liquidity_sweep = False
        sweep_count = 0
        sweep_depth_atr = 0.0
        
        if recent_lows:
            for low in recent_lows[-3:]:
                if breakdown_candle.low < low:
                    sweep_count += 1
                    sweep_levels.append(low)
            
            if sweep_count >= 1:
                liquidity_sweep = True
                trap_score += 0.05 * min(3, sweep_count)
                reasons.append(f'Swept {sweep_count} recent low(s)')
                
                deepest_sweep = min([l for l in recent_lows[-3:] if breakdown_candle.low < l], default=0)
                if deepest_sweep > 0:
                    sweep_depth_atr = (deepest_sweep - breakdown_candle.low) / (atr + 1e-8)
                    if sweep_depth_atr >= LIQUIDITY_SWEEP_DEPTH_ATR:
                        trap_score += 0.08
                        reasons.append(f'Deep sweep: {sweep_depth_atr:.1f}x ATR')
        
        # ===== Factor 5: Level importance =====
        level_importance = 0.5
        if support_level is not None:
            level_touches = 0
            for c in candles[max(0, idx-50):idx]:
                if abs(c.low - support) / support <= 0.002:
                    level_touches += 1
            
            if level_touches >= 3:
                level_importance = 0.9
                trap_score += 0.10
                reasons.append(f'Key support (touched {level_touches}x)')
            elif level_touches >= 2:
                level_importance = 0.7
                trap_score += 0.05
                reasons.append(f'Important support (touched {level_touches}x)')
        
        trap_score = min(0.95, trap_score)
        severity, severity_label = self._get_severity(trap_score)
        confidence = 0.5 + (trap_score * 0.5)
        
        description = f"Bear Trap: Price broke below {support:.4f} but closed above. "
        description += f"Severity: {severity_label.upper()}. "
        description += "Smart money buying into breakdown."
        
        return DetectedTrap(
            type=TrapType.BEAR_TRAP,
            direction='BULLISH',
            severity=severity,
            severity_score=trap_score,
            confidence=min(0.95, confidence),
            candle_index=reversal_candle.index,
            trap_price=breakdown_candle.low,
            reversal_price=reversal_candle.close,
            liquidity_sweep=liquidity_sweep,
            sweep_levels=sweep_levels,
            sweep_count=sweep_count,
            sweep_depth_atr=sweep_depth_atr,
            volume_ratio_breakout=breakdown_candle.volume_ratio,
            volume_ratio_reversal=reversal_candle.volume_ratio,
            reversal_candles=1,
            level_importance=level_importance,
            level_type='support',
            description=description,
            reasons=reasons
        )
    
    # =========================================================
    # STOP HUNT DETECTION
    # =========================================================
    
    def detect_stop_hunt(self, candles: List[CandleData], idx: int,
                         atr: float = 0,
                         key_level: Optional[float] = None,
                         level_type: str = '') -> Optional[DetectedTrap]:
        """
        Detect Stop Hunt: Long wick through key level with immediate reversal
        
        Stop hunt indicates:
        - Smart money taking liquidity (stop losses)
        - Immediate reversal after liquidity taken
        - Strong directional move after the hunt
        
        Args:
            candles: List of candle data
            idx: Index of the reversal candle
            atr: ATR value for normalization
            key_level: Optional key level (support/resistance)
            level_type: 'support' or 'resistance'
        
        Returns:
            DetectedTrap or None
        """
        if idx < 1 or len(candles) <= idx:
            return None
        
        hunt_candle = candles[idx - 1]
        reversal_candle = candles[idx]
        
        # Check for extreme wick
        has_upper_wick = hunt_candle.upper_wick_atr >= STOP_HUNT_WICK_ATR_MULTIPLIER
        has_lower_wick = hunt_candle.lower_wick_atr >= STOP_HUNT_WICK_ATR_MULTIPLIER
        
        if not (has_upper_wick or has_lower_wick):
            return None
        
        # Determine direction based on wick
        if has_lower_wick:
            # Downside stop hunt = bullish reversal
            if not reversal_candle.is_bullish:
                return None
            direction = 'BULLISH'
            trap_type = TrapType.STOP_HUNT_BULLISH
            wick_type = 'lower'
            wick_size = hunt_candle.lower_wick
            wick_atr = hunt_candle.lower_wick_atr
            
            # Check if it swept a key level
            if key_level and level_type == 'support':
                if hunt_candle.low < key_level and reversal_candle.close > key_level:
                    level_swept = True
                else:
                    level_swept = False
            else:
                level_swept = False
                
        elif has_upper_wick:
            # Upside stop hunt = bearish reversal
            if not reversal_candle.is_bearish:
                return None
            direction = 'BEARISH'
            trap_type = TrapType.STOP_HUNT_BEARISH
            wick_type = 'upper'
            wick_size = hunt_candle.upper_wick
            wick_atr = hunt_candle.upper_wick_atr
            
            if key_level and level_type == 'resistance':
                if hunt_candle.high > key_level and reversal_candle.close < key_level:
                    level_swept = True
                else:
                    level_swept = False
            else:
                level_swept = False
        else:
            return None
        
        # Calculate trap score
        trap_score = 0.6  # Base for stop hunt
        reasons = [f'Stop hunt detected: {wick_type.upper()} wick']
        
        # ===== Factor 1: Wick size =====
        if wick_atr >= 1.5:
            trap_score += 0.12
            reasons.append(f'Very long wick ({wick_atr:.1f}x ATR)')
        elif wick_atr >= 1.0:
            trap_score += 0.08
            reasons.append(f'Long wick ({wick_atr:.1f}x ATR)')
        
        # ===== Factor 2: Reversal strength =====
        if direction == 'BULLISH':
            if reversal_candle.body_atr >= 0.8:
                trap_score += 0.12
                reasons.append('Strong bullish reversal candle')
            elif reversal_candle.close_position >= 0.7:
                trap_score += 0.08
                reasons.append('Close near high')
        else:
            if reversal_candle.body_atr >= 0.8:
                trap_score += 0.12
                reasons.append('Strong bearish reversal candle')
            elif reversal_candle.close_position <= 0.3:
                trap_score += 0.08
                reasons.append('Close near low')
        
        # ===== Factor 3: Volume confirmation =====
        if hunt_candle.volume_ratio >= STOP_HUNT_VOLUME_RATIO:
            trap_score += 0.10
            reasons.append(f'Volume spike during hunt: {hunt_candle.volume_ratio:.1f}x')
        
        # ===== Factor 4: Key level sweep =====
        if level_swept:
            trap_score += 0.10
            reasons.append(f'Swept key {level_type} level')
        
        trap_score = min(0.95, trap_score)
        severity, severity_label = self._get_severity(trap_score)
        confidence = 0.5 + (trap_score * 0.5)
        
        description = f"Stop Hunt: {wick_type.upper()} wick swept liquidity. "
        description += f"Severity: {severity_label.upper()}. "
        description += f"Expected {direction.lower()} move."
        
        return DetectedTrap(
            type=trap_type,
            direction=direction,
            severity=severity,
            severity_score=trap_score,
            confidence=min(0.95, confidence),
            candle_index=reversal_candle.index,
            trap_price=hunt_candle.high if wick_type == 'upper' else hunt_candle.low,
            reversal_price=reversal_candle.close,
            liquidity_sweep=level_swept,
            sweep_levels=[key_level] if level_swept else [],
            sweep_count=1 if level_swept else 0,
            sweep_depth_atr=wick_atr,
            volume_ratio_breakout=hunt_candle.volume_ratio,
            volume_ratio_reversal=reversal_candle.volume_ratio,
            reversal_candles=1,
            level_importance=0.7 if level_swept else 0.5,
            level_type=level_type,
            description=description,
            reasons=reasons
        )
    
    # =========================================================
    # LIQUIDITY SWEEP DETECTION (MULTI-LEVEL)
    # =========================================================
    
    def detect_liquidity_sweep(self, candles: List[CandleData], idx: int,
                                swing_highs: Optional[List[float]] = None,
                                swing_lows: Optional[List[float]] = None,
                                atr: float = 0) -> Optional[DetectedTrap]:
        """
        Detect multi-level liquidity sweep
        
        A liquidity sweep occurs when price breaks through multiple key levels
        before reversing. This indicates strong smart money activity.
        
        Args:
            candles: List of candle data
            idx: Index of the sweep candle
            swing_highs: List of recent swing highs
            swing_lows: List of recent swing lows
            atr: ATR value
        
        Returns:
            DetectedTrap or None
        """
        if idx < 0 or len(candles) <= idx:
            return None
        
        sweep_candle = candles[idx]
        
        # Check for sweep of multiple highs
        swept_highs = []
        swept_lows = []
        
        if swing_highs:
            for high in swing_highs[-5:]:
                if sweep_candle.high > high:
                    swept_highs.append(high)
        
        if swing_lows:
            for low in swing_lows[-5:]:
                if sweep_candle.low < low:
                    swept_lows.append(low)
        
        total_sweeps = len(swept_highs) + len(swept_lows)
        
        if total_sweeps < LIQUIDITY_SWEEP_MULTIPLE_COUNT:
            return None
        
        # Determine direction
        if len(swept_highs) > len(swept_lows):
            direction = 'BEARISH'  # Swept highs = bearish after reversal
            trap_type = TrapType.LIQUIDITY_SWEEP_BEARISH
            level_type = 'resistance'
            sweep_levels = swept_highs
        else:
            direction = 'BULLISH'   # Swept lows = bullish after reversal
            trap_type = TrapType.LIQUIDITY_SWEEP_BULLISH
            level_type = 'support'
            sweep_levels = swept_lows
        
        # Calculate trap score
        trap_score = 0.5 + (min(1.0, total_sweeps / 5) * 0.3)
        reasons = [f'Multi-level liquidity sweep: {total_sweeps} levels']
        
        # Add each swept level to reasons
        for level in sweep_levels[:3]:
            reasons.append(f'Swept level: {level:.4f}')
        
        # Check for immediate reversal
        if idx + 1 < len(candles):
            next_candle = candles[idx + 1]
            if direction == 'BULLISH' and next_candle.is_bullish:
                trap_score += 0.10
                reasons.append('Immediate bullish reversal')
            elif direction == 'BEARISH' and next_candle.is_bearish:
                trap_score += 0.10
                reasons.append('Immediate bearish reversal')
        
        # Volume confirmation
        if sweep_candle.volume_ratio >= 1.5:
            trap_score += 0.10
            reasons.append(f'Volume spike: {sweep_candle.volume_ratio:.1f}x')
        
        trap_score = min(0.95, trap_score)
        severity, severity_label = self._get_severity(trap_score)
        confidence = 0.5 + (trap_score * 0.5)
        
        description = f"Liquidity Sweep: Swept {total_sweeps} {level_type} levels. "
        description += f"Severity: {severity_label.upper()}. "
        description += f"Expected {direction.lower()} move."
        
        return DetectedTrap(
            type=trap_type,
            direction=direction,
            severity=severity,
            severity_score=trap_score,
            confidence=min(0.95, confidence),
            candle_index=sweep_candle.index,
            trap_price=sweep_candle.high if direction == 'BEARISH' else sweep_candle.low,
            reversal_price=sweep_candle.close,
            liquidity_sweep=True,
            sweep_levels=sweep_levels,
            sweep_count=total_sweeps,
            sweep_depth_atr=max([abs(sweep_candle.high - l) for l in swept_highs] if swept_highs else [0] +
                                [abs(sweep_candle.low - l) for l in swept_lows] if swept_lows else [0]) / (atr + 1e-8),
            volume_ratio_breakout=sweep_candle.volume_ratio,
            volume_ratio_reversal=0,
            reversal_candles=0,
            level_importance=min(1.0, total_sweeps / 5),
            level_type=level_type,
            description=description,
            reasons=reasons
        )
    
    # =========================================================
    # SEVERITY SCORING
    # =========================================================
    
    def _get_severity(self, score: float) -> Tuple[TrapSeverity, str]:
        """Get severity level based on score"""
        if score >= TRAP_SEVERITY['extreme']['min_score']:
            return TrapSeverity.EXTREME, 'extreme'
        elif score >= TRAP_SEVERITY['strong']['min_score']:
            return TrapSeverity.STRONG, 'strong'
        elif score >= TRAP_SEVERITY['medium']['min_score']:
            return TrapSeverity.MEDIUM, 'medium'
        elif score >= TRAP_SEVERITY['minor']['min_score']:
            return TrapSeverity.MINOR, 'minor'
        else:
            return TrapSeverity.NONE, 'none'
    
    def get_position_multiplier(self, severity: TrapSeverity) -> float:
        """Get position size multiplier based on trap severity"""
        return TRAP_SEVERITY.get(severity.value, {}).get('position_multiplier', 1.0)
    
    # =========================================================
    # COMPLETE TRAP DETECTION
    # =========================================================
    
    def detect_all_traps(self, df: pd.DataFrame,
                         key_levels: Optional[Dict] = None,
                         structure_data: Optional[Dict] = None,
                         liquidity_data: Optional[Dict] = None) -> List[DetectedTrap]:
        """
        Detect all traps in the DataFrame
        
        Args:
            df: OHLCV DataFrame
            key_levels: Optional key levels from price_location.py
            structure_data: Optional structure data from market_structure.py
            liquidity_data: Optional liquidity data from liquidity.py
        
        Returns:
            List of detected traps (sorted by severity)
        """
        traps = []
        
        if df is None or df.empty or len(df) < 20:
            return traps
        
        # Analyze candles
        candles = self.candle_analyzer.analyze_all_candles(df)
        
        if not candles:
            return traps
        
        # Get ATR
        atr = float(df['atr'].iloc[-1]) if 'atr' in df else float(df['close'].iloc[-1]) * 0.02
        
        # Extract key levels from structure_data
        resistance_level = None
        support_level = None
        recent_highs = []
        recent_lows = []
        
        if structure_data:
            swings = structure_data.get('swings', [])
            
            # FIX: Handle both dict and Swing object formats
            for s in swings[-10:]:
                if hasattr(s, 'type'):  # It's a Swing object
                    # Check if type is string or enum
                    if hasattr(s.type, 'value'):
                        type_str = s.type.value  # Enum
                    else:
                        type_str = str(s.type)   # String
                    
                    if type_str in ['HH', 'LH']:
                        recent_highs.append(s.price)
                    elif type_str in ['HL', 'LL']:
                        recent_lows.append(s.price)
                        
                elif isinstance(s, dict):  # It's a dictionary
                    if s.get('type') in ['HH', 'LH']:
                        recent_highs.append(s.get('price', 0))
                    elif s.get('type') in ['HL', 'LL']:
                        recent_lows.append(s.get('price', 0))
        
        # Extract key levels from key_levels
        if key_levels:
            resistances = key_levels.get('resistances', [])
            supports = key_levels.get('supports', [])
            if resistances:
                resistance_level = min(resistances)  # Closest resistance above
            if supports:
                support_level = max(supports)       # Closest support below
        
        # Detect traps on recent candles
        n = len(candles)
        for i in range(max(1, n - 30), n):
            # Bull Trap
            bull_trap = self.detect_bull_trap(candles, i, resistance_level, atr, recent_highs)
            if bull_trap:
                traps.append(bull_trap)
            
            # Bear Trap
            bear_trap = self.detect_bear_trap(candles, i, support_level, atr, recent_lows)
            if bear_trap:
                traps.append(bear_trap)
            
            # Stop Hunt
            stop_hunt = self.detect_stop_hunt(candles, i, atr, 
                                              resistance_level if i > 0 else None, 'resistance')
            if stop_hunt:
                traps.append(stop_hunt)
            
            stop_hunt_down = self.detect_stop_hunt(candles, i, atr,
                                                   support_level if i > 0 else None, 'support')
            if stop_hunt_down:
                traps.append(stop_hunt_down)
            
            # Liquidity Sweep
            if len(recent_highs) >= 3 or len(recent_lows) >= 3:
                sweep = self.detect_liquidity_sweep(candles, i, recent_highs, recent_lows, atr)
                if sweep:
                    traps.append(sweep)
        
        # Sort by severity (highest first)
        traps.sort(key=lambda x: x.severity_score, reverse=True)
        
        return traps
    
    def get_best_trap(self, df: pd.DataFrame,
                      key_levels: Optional[Dict] = None,
                      structure_data: Optional[Dict] = None) -> Optional[DetectedTrap]:
        """Get the highest severity trap"""
        traps = self.detect_all_traps(df, key_levels, structure_data)
        return traps[0] if traps else None

    
    def get_best_trap(self, df: pd.DataFrame,
                      key_levels: Optional[Dict] = None,
                      structure_data: Optional[Dict] = None) -> Optional[DetectedTrap]:
        """Get the highest severity trap"""
        traps = self.detect_all_traps(df, key_levels, structure_data)
        return traps[0] if traps else None


# ==================== CONVENIENCE FUNCTIONS ====================

def detect_traps(df: pd.DataFrame,
                 key_levels: Optional[Dict] = None,
                 structure_data: Optional[Dict] = None) -> List[DetectedTrap]:
    """
    Convenience function to detect all traps
    
    Args:
        df: OHLCV DataFrame
        key_levels: Optional key levels
        structure_data: Optional structure data
    
    Returns:
        List of detected traps
    """
    engine = TrapEngine()
    return engine.detect_all_traps(df, key_levels, structure_data)


def get_best_trap(df: pd.DataFrame,
                  key_levels: Optional[Dict] = None,
                  structure_data: Optional[Dict] = None) -> Optional[DetectedTrap]:
    """
    Convenience function to get highest severity trap
    
    Args:
        df: OHLCV DataFrame
        key_levels: Optional key levels
        structure_data: Optional structure data
    
    Returns:
        Best trap or None
    """
    engine = TrapEngine()
    return engine.get_best_trap(df, key_levels, structure_data)


def get_trap_severity_score(df: pd.DataFrame,
                            key_levels: Optional[Dict] = None) -> float:
    """
    Convenience function to get overall trap severity score (0-1)
    Higher score = stronger trap signal
    """
    traps = detect_traps(df, key_levels)
    if not traps:
        return 0.0
    
    # Weighted average of top traps
    weights = [1.0, 0.5, 0.25][:len(traps)]
    total_weight = sum(weights)
    
    weighted_score = sum(t.severity_score * w for t, w in zip(traps, weights))
    return weighted_score / total_weight if total_weight > 0 else 0.0


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Create test data with trap patterns
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    np.random.seed(42)
    base_price = 50000
    prices = [base_price]
    
    # Create a bull trap pattern
    for i in range(99):
        if i == 40:  # Breakout
            prices.append(prices[-1] + 150)
        elif i == 41:  # Trap reversal
            prices.append(prices[-1] - 200)
        else:
            change = np.random.randn() * 30
            prices.append(prices[-1] + change)
    
    data = []
    for i, close in enumerate(prices):
        open_p = close - np.random.randn() * 20
        high = max(open_p, close) + abs(np.random.randn() * 30)
        low = min(open_p, close) - abs(np.random.randn() * 30)
        volume = abs(np.random.randn() * 10000) + 5000
        
        # Create volume spike for trap
        if i == 40:
            volume = 25000
        
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
    
    # Detect traps
    engine = TrapEngine()
    traps = engine.detect_all_traps(df)
    
    print("=" * 60)
    print("TRAP DETECTION RESULTS")
    print("=" * 60)
    
    for trap in traps:
        print(f"\nType: {trap.type.value.upper()}")
        print(f"Direction: {trap.direction}")
        print(f"Severity: {trap.severity.value} ({trap.severity_score:.1%})")
        print(f"Confidence: {trap.confidence:.1%}")
        print(f"Trap Price: {trap.trap_price:.2f}")
        print(f"Reversal Price: {trap.reversal_price:.2f}")
        print(f"Liquidity Sweep: {trap.liquidity_sweep} (swept {trap.sweep_count} levels)")
        print(f"Reasons:")
        for r in trap.reasons[:3]:
            print(f"  - {r}")
        print(f"Description: {trap.description}")
    
    print("=" * 60)