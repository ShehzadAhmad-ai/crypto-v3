"""
pattern_execution.py - Advanced Execution Engine for Pattern V4

Generates complete trade setups:
- Draft execution (based on pattern structure only)
- Final execution (adjusted with confidence, cluster, regime)
- Action determination (ENTER_NOW, WAIT_FOR_RETEST, etc.)
- Retest confirmation tracking
- Latency detection

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .pattern_core import PatternV4, PatternDirection, PatternStage, ActionType
from .pattern_config import CONFIG, get_completion_threshold, get_action


# ============================================================================
# DRAFT EXECUTION ENGINE
# ============================================================================

class DraftExecutionEngineV4:
    """
    Generates rough entry/stop/target based on pattern structure only.
    This is before scoring and clustering - used for early filtering.
    """
    
    def __init__(self):
        self.min_draft_rr = 1.0  # Minimum RR for draft (early filter)
    
    def generate_draft(self, pattern: Dict, df: pd.DataFrame) -> Dict:
        """
        Generate draft trade setup based on pattern structure.
        Returns pattern with draft_entry, draft_stop, draft_target.
        """
        
        pattern_name = pattern.get('pattern_name', 'Unknown')
        direction = pattern.get('direction', 'NEUTRAL')
        current_price = float(df['close'].iloc[-1])
        atr = self._get_atr(df)
        
        # Get pattern-specific levels
        neckline = pattern.get('neckline', current_price)
        pattern_height = pattern.get('pattern_height', atr * 2)
        
        # Calculate based on pattern type
        if 'Head_Shoulders' in pattern_name or 'Inverse_Head_Shoulders' in pattern_name:
            self._hs_draft(pattern, direction, neckline, pattern_height, current_price, atr)
        elif 'Double' in pattern_name or 'Triple' in pattern_name:
            self._double_draft(pattern, direction, neckline, pattern_height, current_price, atr)
        elif 'Flag' in pattern_name or 'Pennant' in pattern_name:
            self._flag_draft(pattern, direction, pattern, current_price, atr)
        elif 'Wedge' in pattern_name or 'Triangle' in pattern_name:
            self._triangle_draft(pattern, direction, neckline, pattern_height, current_price, atr)
        elif 'Divergence' in pattern_name:
            self._divergence_draft(pattern, direction, current_price, atr)
        else:
            self._default_draft(pattern, direction, current_price, atr)
        
        # Calculate draft risk/reward
        pattern['draft_rr'] = self._calculate_rr(
            pattern.get('draft_entry', current_price),
            pattern.get('draft_stop', current_price),
            pattern.get('draft_target', current_price),
            direction
        )
        
        return pattern
    
    def _hs_draft(self, pattern: Dict, direction: str, neckline: float,
                  pattern_height: float, current_price: float, atr: float):
        """Head & Shoulders draft setup"""
        
        if direction == 'SELL':
            pattern['draft_entry'] = neckline * 0.995
            pattern['draft_stop'] = neckline + pattern_height * 0.5 + atr * 0.5
            pattern['draft_target'] = neckline - pattern_height
            pattern['stop_reason'] = "Above neckline + pattern height/2 + ATR buffer"
        else:  # BUY (Inverse H&S)
            pattern['draft_entry'] = neckline * 1.005
            pattern['draft_stop'] = neckline - pattern_height * 0.5 - atr * 0.5
            pattern['draft_target'] = neckline + pattern_height
            pattern['stop_reason'] = "Below neckline - pattern height/2 - ATR buffer"
        
        pattern['target_reason'] = f"Pattern height projection: {pattern_height:.2f}"
        pattern['retest_level'] = neckline
    
    def _double_draft(self, pattern: Dict, direction: str, neckline: float,
                      pattern_height: float, current_price: float, atr: float):
        """Double Top/Bottom draft setup"""
        
        if direction == 'SELL':  # Double Top
            pattern['draft_entry'] = neckline * 0.995
            pattern['draft_stop'] = neckline + pattern_height * 0.5 + atr * 0.5
            pattern['draft_target'] = neckline - pattern_height
        else:  # Double Bottom
            pattern['draft_entry'] = neckline * 1.005
            pattern['draft_stop'] = neckline - pattern_height * 0.5 - atr * 0.5
            pattern['draft_target'] = neckline + pattern_height
        
        pattern['retest_level'] = neckline
        pattern['stop_reason'] = "Beyond pattern extreme + buffer"
        pattern['target_reason'] = "Pattern height projection"
    
    def _flag_draft(self, pattern: Dict, direction: str, pattern_data: Dict,
                    current_price: float, atr: float):
        """Flag/Pennant draft setup"""
        
        flag_high = pattern_data.get('flag_high', current_price * 1.02)
        flag_low = pattern_data.get('flag_low', current_price * 0.98)
        flagpole_end = pattern_data.get('flagpole_end', current_price)
        flagpole_height = pattern_data.get('pattern_height', atr * 3)
        
        if direction == 'BUY':
            pattern['draft_entry'] = flag_high * 1.005
            pattern['draft_stop'] = flag_low * 0.995
            pattern['draft_target'] = flagpole_end + flagpole_height
            pattern['retest_level'] = flag_high
        else:
            pattern['draft_entry'] = flag_low * 0.995
            pattern['draft_stop'] = flag_high * 1.005
            pattern['draft_target'] = flagpole_end - flagpole_height
            pattern['retest_level'] = flag_low
        
        pattern['stop_reason'] = "Beyond flag consolidation"
        pattern['target_reason'] = "Flagpole projection"
    
    def _triangle_draft(self, pattern: Dict, direction: str, neckline: float,
                        pattern_height: float, current_price: float, atr: float):
        """Triangle/Wedge draft setup"""
        
        if direction == 'BUY':
            pattern['draft_entry'] = neckline * 1.005
            pattern['draft_stop'] = neckline - pattern_height * 0.3 - atr * 0.5
            pattern['draft_target'] = neckline + pattern_height
        else:
            pattern['draft_entry'] = neckline * 0.995
            pattern['draft_stop'] = neckline + pattern_height * 0.3 + atr * 0.5
            pattern['draft_target'] = neckline - pattern_height
        
        pattern['retest_level'] = neckline
        pattern['stop_reason'] = "Beyond triangle + buffer"
        pattern['target_reason'] = "Triangle height projection"
    
    def _divergence_draft(self, pattern: Dict, direction: str,
                          current_price: float, atr: float):
        """Divergence draft setup"""
        
        swing_points = pattern.get('swing_points', [])
        if len(swing_points) >= 2:
            point1_price = swing_points[0].get('price', current_price) if isinstance(swing_points[0], dict) else swing_points[0].price
            point2_price = swing_points[1].get('price', current_price) if isinstance(swing_points[1], dict) else swing_points[1].price
            
            if direction == 'BUY':  # Bullish divergence
                pattern['draft_entry'] = current_price * 1.002
                pattern['draft_stop'] = point2_price * 0.99
                pattern['draft_target'] = point1_price
            else:  # Bearish divergence
                pattern['draft_entry'] = current_price * 0.998
                pattern['draft_stop'] = point2_price * 1.01
                pattern['draft_target'] = point1_price
        else:
            self._default_draft(pattern, direction, current_price, atr)
        
        pattern['retest_level'] = pattern.get('draft_entry', current_price)
        pattern['stop_reason'] = "Beyond divergence point"
        pattern['target_reason'] = "Previous swing point"
    
    def _default_draft(self, pattern: Dict, direction: str, current_price: float, atr: float):
        """Default draft for unknown patterns"""
        
        if direction == 'BUY':
            pattern['draft_entry'] = current_price * 1.002
            pattern['draft_stop'] = current_price - atr * 1.5
            pattern['draft_target'] = current_price + atr * 3.0
        else:
            pattern['draft_entry'] = current_price * 0.998
            pattern['draft_stop'] = current_price + atr * 1.5
            pattern['draft_target'] = current_price - atr * 3.0
        
        pattern['stop_reason'] = "ATR-based stop"
        pattern['target_reason'] = "2x risk target"
    
    def _get_atr(self, df: pd.DataFrame) -> float:
        """Get ATR value"""
        if 'atr' in df and not pd.isna(df['atr'].iloc[-1]):
            return float(df['atr'].iloc[-1])
        return float(df['close'].iloc[-1]) * 0.02
    
    def _calculate_rr(self, entry: float, stop: float, target: float, direction: str) -> float:
        """Calculate risk/reward ratio"""
        if entry is None or stop is None or target is None:
            return 0.0
        
        if direction == 'BUY':
            risk = entry - stop
            reward = target - entry
        else:
            risk = stop - entry
            reward = entry - target
        
        if risk <= 0:
            return 0.0
        
        return reward / risk


# ============================================================================
# FINAL EXECUTION ADJUSTMENT
# ============================================================================

class FinalExecutionAdjustmentV4:
    """
    Refines draft setup based on:
    - Final confidence score
    - Cluster strength
    - Market regime
    - Liquidity confirmation
    """
    
    def __init__(self):
        self.stop_config = CONFIG.stop_loss_config if hasattr(CONFIG, 'stop_loss_config') else {
            'atr_multiplier_min': 1.2,
            'atr_multiplier_max': 2.0,
        }
    
    def adjust(self, pattern: Dict, final_confidence: float,
               cluster_score: float, regime: str,
               liquidity_score: float) -> Dict:
        """
        Apply final adjustments to draft setup.
        """
        
        # Start with draft values
        entry = pattern.get('draft_entry')
        stop = pattern.get('draft_stop')
        target = pattern.get('draft_target')
        
        if entry is None or stop is None or target is None:
            return pattern
        
        direction = pattern.get('direction', 'NEUTRAL')
        
        # 1. Adjust entry based on confidence
        entry = self._adjust_entry(entry, pattern.get('retest_level', entry), 
                                   final_confidence, direction)
        
        # 2. Adjust stop based on confidence and liquidity
        stop = self._adjust_stop(stop, entry, final_confidence, liquidity_score, direction)
        
        # 3. Adjust target based on cluster and regime
        target = self._adjust_target(target, entry, cluster_score, regime, direction)
        
        # 4. Set final values
        pattern['entry'] = entry
        pattern['stop_loss'] = stop
        pattern['take_profit'] = target
        pattern['risk_reward'] = self._calculate_rr(entry, stop, target, direction)
        
        return pattern
    
    def _adjust_entry(self, entry: float, retest_level: float,
                      confidence: float, direction: str) -> float:
        """Adjust entry based on confidence"""
        
        if retest_level is None:
            return entry
        
        # High confidence = tighter entry (wait for better price)
        if confidence > 0.80:
            if direction == 'BUY':
                return retest_level * 1.002
            else:
                return retest_level * 0.998
        
        # Medium confidence = standard entry
        elif confidence > 0.65:
            if direction == 'BUY':
                return retest_level * 1.005
            else:
                return retest_level * 0.995
        
        return entry
    
    def _adjust_stop(self, stop: float, entry: float, confidence: float,
                     liquidity_score: float, direction: str) -> float:
        """Adjust stop based on confidence and liquidity"""
        
        # Get ATR (estimated from pattern or default)
        atr = abs(entry - stop) * 0.5 if entry and stop else entry * 0.02
        
        # Calculate volatility buffer
        vol_buffer = atr * self.stop_config.get('atr_multiplier_min', 1.2)
        
        # High confidence = tighter stop
        if confidence > 0.80:
            vol_buffer = atr * 1.2
        elif confidence < 0.65:
            vol_buffer = atr * 1.8
        
        # Strong liquidity = tighter stop
        if liquidity_score > 0.7:
            vol_buffer = vol_buffer * 0.9
        
        # Adjust stop based on structure
        if direction == 'BUY':
            new_stop = min(stop, entry - vol_buffer) if stop else entry - vol_buffer
        else:
            new_stop = max(stop, entry + vol_buffer) if stop else entry + vol_buffer
        
        return new_stop
    
    def _adjust_target(self, target: float, entry: float, cluster_score: float,
                       regime: str, direction: str) -> float:
        """Adjust target based on cluster and regime"""
        
        if target is None or entry is None:
            return target
        
        # Calculate base reward
        if direction == 'BUY':
            reward = target - entry
        else:
            reward = entry - target
        
        # Strong cluster = extended target
        if cluster_score > 0.80:
            reward = reward * 1.2
        
        # Strong trend = extended target
        if 'TREND' in regime:
            reward = reward * 1.15
        
        # Volatile market = reduced target
        if 'VOLATILE' in regime:
            reward = reward * 0.9
        
        # Apply adjustment
        if direction == 'BUY':
            return entry + reward
        else:
            return entry - reward
    
    def _calculate_rr(self, entry: float, stop: float, target: float, direction: str) -> float:
        """Calculate risk/reward ratio"""
        if entry is None or stop is None or target is None:
            return 0.0
        
        if direction == 'BUY':
            risk = entry - stop
            reward = target - entry
        else:
            risk = stop - entry
            reward = entry - target
        
        if risk <= 0:
            return 0.0
        
        return reward / risk


# ============================================================================
# RETEST CONFIRMATION TRACKER
# ============================================================================

class RetestConfirmationTrackerV4:
    """
    Tracks whether a retest has occurred and been confirmed.
    This is updated in real-time as new price data arrives.
    """
    
    def __init__(self):
        self.retest_tolerance = CONFIG.lifecycle_config.get('retest_tolerance', 0.005) if hasattr(CONFIG, 'lifecycle_config') else 0.005
    
    def update(self, pattern: Dict, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Update retest confirmation status based on latest price action.
        """
        
        retest_level = pattern.get('retest_level')
        if retest_level is None:
            return pattern
        
        direction = pattern.get('direction', 'NEUTRAL')
        
        # Get recent price action
        recent = df.iloc[max(0, current_idx - 10):current_idx + 1]
        
        # Check if price touched retest level
        touched = False
        bars_since_retest = None
        
        for i in range(len(recent)):
            if direction == 'BUY':
                if recent['low'].iloc[i] <= retest_level * (1 + self.retest_tolerance):
                    touched = True
                    bars_since_retest = current_idx - recent.index[i]
            else:
                if recent['high'].iloc[i] >= retest_level * (1 - self.retest_tolerance):
                    touched = True
                    bars_since_retest = current_idx - recent.index[i]
        
        # Check if retest is confirmed (touched and reversed)
        retest_confirmed = False
        if touched and bars_since_retest is not None and bars_since_retest <= 2:
            last_candle = recent.iloc[-1]
            if direction == 'BUY' and last_candle['close'] > last_candle['open']:
                retest_confirmed = True
            elif direction == 'SELL' and last_candle['close'] < last_candle['open']:
                retest_confirmed = True
        
        pattern['retest_confirmed'] = retest_confirmed
        pattern['bars_since_retest'] = bars_since_retest if bars_since_retest is not None else 999
        
        return pattern


# ============================================================================
# LATENCY DETECTOR
# ============================================================================

class LatencyDetectorV4:
    """
    Detects if we're late to the move.
    Prevents entering after price has already moved significantly.
    """
    
    def __init__(self):
        self.max_breakout_latency = CONFIG.decision_thresholds.get('max_latency_bars', 3) if hasattr(CONFIG, 'decision_thresholds') else 3
        self.max_retest_latency = 1
    
    def check(self, pattern: Dict, current_idx: int) -> Tuple[bool, str]:
        """
        Check if we're too late to enter.
        Returns (is_late, reason)
        """
        
        # Check breakout latency
        bars_since_breakout = pattern.get('bars_since_breakout', 0)
        if bars_since_breakout > self.max_breakout_latency:
            return True, f"Breakout was {bars_since_breakout} bars ago"
        
        # Check retest latency
        bars_since_retest = pattern.get('bars_since_retest', 999)
        if bars_since_retest <= self.max_retest_latency:
            # Not late, just retested
            return False, ""
        elif bars_since_retest < 10:
            # Slightly late but acceptable
            return False, ""
        elif bars_since_retest < 999:
            return True, f"Retest was {bars_since_retest} bars ago"
        
        return False, ""


# ============================================================================
# COMPLETION CALCULATOR
# ============================================================================

class CompletionCalculatorV4:
    """
    Calculates pattern completion percentage for early entry.
    """
    
    def __init__(self):
        self.thresholds = CONFIG.completion_thresholds if hasattr(CONFIG, 'completion_thresholds') else {}
    
    def calculate_completion(self, pattern: Dict, swings: List, current_idx: int) -> float:
        """
        Calculate how complete the pattern is (0-1).
        Allows early entry before full confirmation.
        """
        pattern_name = pattern.get('pattern_name', 'Unknown')
        start_idx = pattern.get('start_idx', 0)
        end_idx = pattern.get('end_idx', current_idx)
        
        formation_bars = end_idx - start_idx
        expected_bars = self._get_expected_bars(pattern_name)
        
        if expected_bars > 0:
            completion = min(1.0, formation_bars / expected_bars)
        else:
            completion = 0.5
        
        # Adjust based on swing completion
        swing_points = pattern.get('swing_points', [])
        if swing_points:
            expected_swings = self._get_expected_swings(pattern_name)
            actual_swings = len(swing_points)
            swing_completion = min(1.0, actual_swings / expected_swings) if expected_swings > 0 else 0.5
            completion = (completion + swing_completion) / 2
        
        return min(1.0, completion)
    
    def _get_expected_bars(self, pattern_name: str) -> int:
        """Get expected formation bars for pattern type"""
        expected = {
            'Head_Shoulders': 20,
            'Double_Top': 15,
            'Triple_Top': 25,
            'Flag': 10,
            'Triangle': 20,
            'Wedge': 18,
        }
        for key, value in expected.items():
            if key in pattern_name:
                return value
        return 15
    
    def _get_expected_swings(self, pattern_name: str) -> int:
        """Get expected swing count for pattern type"""
        expected = {
            'Head_Shoulders': 5,
            'Double_Top': 3,
            'Triple_Top': 5,
            'Flag': 2,
            'Triangle': 4,
        }
        for key, value in expected.items():
            if key in pattern_name:
                return value
        return 4


# ============================================================================
# MAIN EXECUTION ENGINE
# ============================================================================

class PatternExecutionEngineV4:
    """
    Main execution engine that orchestrates all execution components.
    """
    
    def __init__(self):
        self.draft_engine = DraftExecutionEngineV4()
        self.final_adjustment = FinalExecutionAdjustmentV4()
        self.retest_tracker = RetestConfirmationTrackerV4()
        self.latency_detector = LatencyDetectorV4()
        self.completion_calculator = CompletionCalculatorV4()
    
    def generate_setup(self, pattern: Dict, df: pd.DataFrame,
                       final_confidence: float, cluster_score: float,
                       regime: str, liquidity_score: float,
                       current_idx: int) -> Dict:
        """
        Generate complete trade setup with action.
        """
        
        # 1. Generate draft setup (if not already done)
        if pattern.get('draft_entry') is None:
            pattern = self.draft_engine.generate_draft(pattern, df)
        
        # 2. Calculate completion percentage
        swings = pattern.get('swing_points', [])
        completion_pct = self.completion_calculator.calculate_completion(pattern, swings, current_idx)
        pattern['completion_pct'] = completion_pct
        
        # 3. Apply final adjustments
        pattern = self.final_adjustment.adjust(
            pattern, final_confidence, cluster_score, regime, liquidity_score
        )
        
        # 4. Update retest confirmation
        pattern = self.retest_tracker.update(pattern, df, current_idx)
        
        # 5. Check latency
        is_late, latency_reason = self.latency_detector.check(pattern, current_idx)
        if is_late:
            pattern['invalid'] = True
            pattern['invalid_reason'] = latency_reason
            pattern['action'] = 'SKIP'
            pattern['action_detail'] = latency_reason
        
        # 6. Set stage based on completion and retest
        pattern['stage'] = self._determine_stage(pattern, completion_pct)
        
        return pattern
    
    def _determine_stage(self, pattern: Dict, completion_pct: float) -> str:
        """Determine pattern stage based on completion"""
        
        if pattern.get('invalid', False):
            return "INVALIDATED"
        
        if pattern.get('retest_confirmed', False):
            return "CONFIRMED"
        
        if completion_pct >= 0.95:
            if pattern.get('retest_level'):
                return "RETEST"
            return "BREAKOUT"
        
        if completion_pct >= 0.70:
            return "BREAKOUT"
        
        return "FORMING"
    
    def update_active_trade(self, pattern: Dict, df: pd.DataFrame,
                            current_idx: int) -> Dict:
        """
        Update active trade with new price data.
        Checks if target or stop reached.
        """
        
        current_price = float(df['close'].iloc[-1])
        direction = pattern.get('direction', 'NEUTRAL')
        entry = pattern.get('entry')
        stop = pattern.get('stop_loss')
        target = pattern.get('take_profit')
        
        if entry is None or stop is None or target is None:
            return pattern
        
        # Check if target reached
        if direction == 'BUY':
            if current_price >= target:
                pattern['target_reached'] = True
                pattern['action'] = 'TAKE_PROFIT'
                pattern['action_detail'] = f"Target reached at {target:.4f}"
            elif current_price <= stop:
                pattern['stop_reached'] = True
                pattern['action'] = 'CANCEL'
                pattern['action_detail'] = f"Stop hit at {stop:.4f}"
        else:
            if current_price <= target:
                pattern['target_reached'] = True
                pattern['action'] = 'TAKE_PROFIT'
                pattern['action_detail'] = f"Target reached at {target:.4f}"
            elif current_price >= stop:
                pattern['stop_reached'] = True
                pattern['action'] = 'CANCEL'
                pattern['action_detail'] = f"Stop hit at {stop:.4f}"
        
        return pattern


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'DraftExecutionEngineV4',
    'FinalExecutionAdjustmentV4',
    'RetestConfirmationTrackerV4',
    'LatencyDetectorV4',
    'CompletionCalculatorV4',
    'PatternExecutionEngineV4',
]