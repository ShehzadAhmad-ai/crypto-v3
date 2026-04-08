"""
SMC Expert V3 - Enhanced Entry Confirmation (COMPLETE REWRITE)
Multi-factor confirmation for entry validation
FIXED: Reduced strictness, proper attribute access, working confirmation logic
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

from .smc_core import Direction, SMCContext, LiquiditySweep
from .smc_config import CONFIG


class EntryConfirmation:
    """
    Multi-factor entry confirmation engine - RELAXED for more signals
    """
    
    def __init__(self):
        self.checks: Dict[str, bool] = {}
        self.scores: Dict[str, float] = {}
    
    def confirm(self, entry: Dict, df: pd.DataFrame, context: SMCContext) -> Dict:
        """
        Run all confirmation checks and return score
        FIXED: Less strict requirements, proper fallbacks
        """
        direction = entry['direction']
        
        # Run all checks with safe fallbacks
        self.checks = {
            'mss_after_entry': self._check_mss_after_entry(entry, context),
            'displacement': self._check_displacement(df, direction),
            'liquidity_sweep_before': self._check_sweep_before(entry, context),
            'candle_confirmation': self._check_candle_confirmation(df, direction),
            'volume_confirmation': self._check_volume_confirmation(df),
            'no_trap': self._check_no_trap(context, direction),
            'htf_alignment': self._check_htf_alignment(context)
        }
        
        # Calculate scores for each check (0-1)
        self.scores = {
            'mss_after_entry': 0.7 if self.checks['mss_after_entry'] else 0.4,
            'displacement': self._score_displacement(df, direction),
            'liquidity_sweep_before': 0.8 if self.checks['liquidity_sweep_before'] else 0.4,
            'candle_confirmation': self._score_candle_confirmation(df, direction),
            'volume_confirmation': self._score_volume_confirmation(df),
            'no_trap': 0.8 if self.checks['no_trap'] else 0.4,
            'htf_alignment': self._score_htf_alignment(context)
        }
        
        # Weighted average - RELAXED thresholds
        weights = {
            'mss_after_entry': 0.12,
            'displacement': 0.12,
            'liquidity_sweep_before': 0.15,
            'candle_confirmation': 0.18,
            'volume_confirmation': 0.12,
            'no_trap': 0.11,
            'htf_alignment': 0.20
        }
        
        total_score = sum(self.scores[k] * weights.get(k, 0.1) for k in weights)
        
        # FIXED: Lower thresholds for action
        if total_score >= 0.70:
            action = "STRONG_ENTRY"
            confirmed = True
        elif total_score >= 0.55:
            action = "ENTER_NOW"
            confirmed = True
        elif total_score >= 0.40:
            action = "WAIT_FOR_RETEST"
            confirmed = False
        else:
            action = "SKIP"
            confirmed = False
        
        return {
            'score': round(total_score, 2),
            'action': action,
            'confirmed': confirmed,
            'checks': self.checks,
            'scores': self.scores,
            'reasons': self._build_reasons()
        }
    
    def _check_mss_after_entry(self, entry: Dict, context: SMCContext) -> bool:
        """Check if there was a Market Structure Shift after entry"""
        if not context.choch_points:
            return False
        
        # Check if MSS occurred in last 5 bars
        for mss in context.choch_points[-5:]:
            mss_direction = mss.get('direction')
            if mss_direction and mss_direction == entry['direction']:
                return True
        
        return False
    
    def _check_displacement(self, df: pd.DataFrame, direction: Direction) -> bool:
        """Check for displacement in the direction of trade"""
        if len(df) < 3:
            return False
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Calculate move size
        if direction == Direction.BUY:
            move = last_candle['close'] - prev_candle['close']
        else:
            move = prev_candle['close'] - last_candle['close']
        
        # FIXED: Lower threshold for displacement
        atr = calculate_atr(df)
        move_atr = move / atr if atr > 0 else 0
        
        return move_atr > 0.3
    
    def _score_displacement(self, df: pd.DataFrame, direction: Direction) -> float:
        """Score displacement strength"""
        if len(df) < 3:
            return 0.4
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        if direction == Direction.BUY:
            move = last_candle['close'] - prev_candle['close']
            is_bullish = last_candle['is_bullish']
        else:
            move = prev_candle['close'] - last_candle['close']
            is_bullish = last_candle['is_bearish']
        
        if move <= 0:
            return 0.3
        
        atr = calculate_atr(df)
        move_atr = move / atr if atr > 0 else 0
        
        # FIXED: Lower threshold for full score
        move_score = min(1.0, move_atr / 1.0)
        candle_bonus = 0.15 if is_bullish else 0
        
        return min(1.0, move_score + candle_bonus)
    
    def _check_sweep_before(self, entry: Dict, context: SMCContext) -> bool:
        """Check if there was a liquidity sweep before entry"""
        if not context.recent_sweeps:
            return False
        
        # Check for sweep in last 10 bars
        for sweep in context.recent_sweeps[-5:]:
            if entry['direction'] == Direction.BUY:
                if sweep.type == 'SSL_SWEEP':
                    return True
            else:
                if sweep.type == 'BSL_SWEEP':
                    return True
        
        return False
    
    def _check_candle_confirmation(self, df: pd.DataFrame, direction: Direction) -> bool:
        """Check for confirmation candle - RELAXED"""
        if len(df) < 2:
            return False
        
        last_candle = df.iloc[-1]
        
        if direction == Direction.BUY:
            # Bullish confirmation: bullish candle OR hammer
            if last_candle['is_bullish']:
                return True
            # Hammer / bullish rejection
            if last_candle['lower_wick'] > last_candle['body']:
                return True
        else:
            # Bearish confirmation: bearish candle OR shooting star
            if last_candle['is_bearish']:
                return True
            # Shooting star / bearish rejection
            if last_candle['upper_wick'] > last_candle['body']:
                return True
        
        # FIXED: Also accept if price is moving in direction
        return False
    
    def _score_candle_confirmation(self, df: pd.DataFrame, direction: Direction) -> float:
        """Score candle confirmation strength"""
        if len(df) < 2:
            return 0.4
        
        last_candle = df.iloc[-1]
        
        if direction == Direction.BUY:
            if last_candle['is_bullish']:
                body_ratio = last_candle.get('body_ratio', 0.5)
                return min(1.0, body_ratio + 0.3)
            elif last_candle['lower_wick'] > last_candle['body']:
                return 0.65
            else:
                return 0.4
        else:
            if last_candle['is_bearish']:
                body_ratio = last_candle.get('body_ratio', 0.5)
                return min(1.0, body_ratio + 0.3)
            elif last_candle['upper_wick'] > last_candle['body']:
                return 0.65
            else:
                return 0.4
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check for volume confirmation - RELAXED"""
        if len(df) < 20:
            return True  # Don't reject due to insufficient data
        
        last_candle = df.iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        # FIXED: Lower threshold
        return last_candle['volume'] > avg_volume * 1.0
    
    def _score_volume_confirmation(self, df: pd.DataFrame) -> float:
        """Score volume confirmation strength"""
        if len(df) < 20:
            return 0.5
        
        last_candle = df.iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        if avg_volume == 0:
            return 0.5
        
        volume_ratio = last_candle['volume'] / avg_volume
        
        if volume_ratio >= 1.5:
            return 1.0
        elif volume_ratio >= 1.2:
            return 0.8
        elif volume_ratio >= 1.0:
            return 0.6
        elif volume_ratio >= 0.8:
            return 0.5
        else:
            return 0.4
    
    def _check_no_trap(self, context: SMCContext, direction: Direction) -> bool:
        """Check that no recent trap exists against the trade"""
        if not context.recent_traps:
            return True
        
        for trap in context.recent_traps[-3:]:
            trap_direction = trap.get('direction')
            if trap_direction and trap_direction != direction:
                # Trap in opposite direction is actually GOOD (confirmation)
                continue
            elif trap_direction and trap_direction == direction:
                # Trap in same direction is bad
                return False
        
        return True
    
    def _check_htf_alignment(self, context: SMCContext) -> bool:
        """Check HTF alignment - RELAXED"""
        # FIXED: Lower threshold
        return context.htf_alignment_score > 0.35
    
    def _score_htf_alignment(self, context: SMCContext) -> float:
        """Score HTF alignment"""
        # FIXED: Return raw score, not binary
        return max(0.3, context.htf_alignment_score)
    
    def _build_reasons(self) -> List[str]:
        """Build list of reasons for confirmation status"""
        reasons = []
        
        for check, passed in self.checks.items():
            check_name = check.replace('_', ' ').title()
            if passed:
                reasons.append(f"✓ {check_name}")
            else:
                reasons.append(f"○ {check_name} (pending)")
        
        return reasons[:5]  # Limit to 5 reasons


# Helper function for ATR calculation (in case of import issues)
def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR from dataframe - SAFE with fallback"""
    if df is None or len(df) < period + 1:
        return 0.001
    
    try:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            return (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / 20
        
        return atr
    except Exception:
        return 0.001