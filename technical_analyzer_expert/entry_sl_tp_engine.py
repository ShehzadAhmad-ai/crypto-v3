# entry_sl_tp_engine.py - Advanced Entry, Stop Loss & Take Profit Engine
"""
Entry, Stop Loss & Take Profit Expert
- Calculates entry price based on EMA21 zone (configurable priority)
- Calculates stop loss using ATR + recent swing high/low (indicator-based)
- Calculates take profit using ATR with confidence adjustment
- Dynamic risk/reward based on confidence and regime
- Returns trade setup with validation
- All thresholds from ta_config.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import configuration and core classes
from .ta_config import *
from .ta_core import TradeSetup, TASignal, RegimeResult

# Try to import logger
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class EntrySLTPEngine:
    """
    Advanced Entry, Stop Loss & Take Profit Engine
    Calculates trade levels using indicator-based methods
    """
    
    def __init__(self):
        """Initialize the engine with thresholds from config"""
        # Entry settings
        self.entry_priority = ENTRY_PRIORITY
        self.ema21_offset = EMA21_ENTRY_OFFSET
        
        # Stop Loss settings
        self.sl_priority = SL_PRIORITY
        self.sl_swing_offset_buy = SL_SWING_OFFSET_BUY
        self.sl_swing_offset_sell = SL_SWING_OFFSET_SELL
        self.sl_ema_slow_offset_buy = SL_EMA_SLOW_OFFSET_BUY
        self.sl_ema_slow_offset_sell = SL_EMA_SLOW_OFFSET_SELL
        
        # Take Profit settings
        self.tp_priority = TP_PRIORITY
        
        # ATR settings
        self.atr_period = ATR_PERIOD
        self.atr_stop_multiplier = ATR_STOP_MULTIPLIER
        self.atr_target_multiplier = ATR_TARGET_MULTIPLIER
        
        # Dynamic RR settings
        self.dynamic_rr = DYNAMIC_RR
        
        # Limits
        self.max_stop_percent = MAX_STOP_PERCENT
        self.min_stop_percent = MIN_STOP_PERCENT
        self.max_tp_percent = MAX_TP_PERCENT
        
        # Swing detection window
        self.swing_window = SWING_WINDOW
        
        log.info("EntrySLTPEngine initialized")
    
    # ============================================================================
    # MAIN CALCULATION METHODS
    # ============================================================================
    
    def calculate_trade_setup(self, 
                              df: pd.DataFrame,
                              direction: str,
                              confidence: float,
                              regime: Optional[RegimeResult] = None,
                              current_price: Optional[float] = None) -> TradeSetup:
        """
        Calculate complete trade setup (entry, stop loss, take profit)
        
        Args:
            df: DataFrame with OHLCV and indicator data
            direction: "BUY" or "SELL"
            confidence: Signal confidence (0-1)
            regime: Optional regime result for dynamic adjustments
            current_price: Optional current price (uses close if not provided)
        
        Returns:
            TradeSetup object with all levels
        """
        if df is None or df.empty:
            log.warning("No data for trade setup calculation")
            return TradeSetup(
                entry=0, stop_loss=0, take_profit=0, risk_reward=0,
                is_valid=False, validation_errors=["No data available"]
            )
        
        try:
            # Get current price
            if current_price is None:
                current_price = float(df['close'].iloc[-1])
            
            # Step 1: Calculate entry price
            entry, entry_method = self._calculate_entry(df, direction, current_price, regime)
            
            if entry <= 0:
                return TradeSetup(
                    entry=0, stop_loss=0, take_profit=0, risk_reward=0,
                    is_valid=False, validation_errors=["Failed to calculate entry price"]
                )
            
            # Step 2: Calculate stop loss
            stop_loss, stop_method = self._calculate_stop_loss(df, direction, entry, regime)
            
            if stop_loss <= 0:
                return TradeSetup(
                    entry=entry, stop_loss=0, take_profit=0, risk_reward=0,
                    is_valid=False, validation_errors=["Failed to calculate stop loss"],
                    entry_method=entry_method
                )
            
            # Step 3: Calculate take profit
            take_profit, tp_method = self._calculate_take_profit(df, direction, entry, stop_loss, confidence, regime)
            
            if take_profit <= 0:
                return TradeSetup(
                    entry=entry, stop_loss=stop_loss, take_profit=0, risk_reward=0,
                    is_valid=False, validation_errors=["Failed to calculate take profit"],
                    entry_method=entry_method, stop_method=stop_method
                )
            
            # Step 4: Calculate risk/reward ratio
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Step 5: Validate trade setup
            is_valid, errors = self._validate_setup(entry, stop_loss, take_profit, risk_reward, direction)
            
            # Step 6: Apply dynamic RR requirement
            min_rr = self._get_min_risk_reward(confidence, regime)
            if risk_reward < min_rr:
                is_valid = False
                errors.append(f"Risk/Reward {risk_reward:.2f} below minimum {min_rr:.2f}")
            
            # Create trade setup
            setup = TradeSetup(
                entry=round(entry, 6),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                risk_reward=round(risk_reward, 2),
                entry_method=entry_method,
                stop_method=stop_method,
                tp_method=tp_method,
                is_valid=is_valid,
                validation_errors=errors
            )
            
            log.debug(f"Trade setup: {direction} @ {entry:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}, RR: {risk_reward:.2f}")
            
            return setup
            
        except Exception as e:
            log.error(f"Error calculating trade setup: {e}")
            return TradeSetup(
                entry=0, stop_loss=0, take_profit=0, risk_reward=0,
                is_valid=False, validation_errors=[f"Calculation error: {e}"]
            )
    
    # ============================================================================
    # ENTRY CALCULATION METHODS
    # ============================================================================
    
    def _calculate_entry(self, df: pd.DataFrame, direction: str,
                         current_price: float, regime: Optional[RegimeResult]) -> Tuple[float, str]:
        """
        Calculate entry price based on priority order
        
        Returns:
            Tuple of (entry_price, method_used)
        """
        for method in self.entry_priority:
            if method == "ema21":
                entry = self._entry_ema21(df, direction)
                if entry:
                    return entry, "EMA21"
            
            elif method == "vwap":
                entry = self._entry_vwap(df, direction)
                if entry:
                    return entry, "VWAP"
            
            elif method == "support_resistance":
                entry = self._entry_support_resistance(df, direction, current_price)
                if entry:
                    return entry, "Support/Resistance"
            
            elif method == "bollinger_mid":
                entry = self._entry_bollinger_mid(df, direction)
                if entry:
                    return entry, "Bollinger Middle"
        
        # Fallback: Use current price
        log.warning("No entry method succeeded, using current price")
        return current_price, "Current Price"
    
    def _entry_ema21(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """Calculate entry using EMA21 zone"""
        try:
            if 'ema_21' not in df.columns:
                return None
            
            ema21 = float(df['ema_21'].iloc[-1])
            
            if direction == "BUY":
                entry = ema21 * (1 + self.ema21_offset)
            else:
                entry = ema21 * (1 - self.ema21_offset)
            
            return entry
            
        except Exception as e:
            log.debug(f"EMA21 entry failed: {e}")
            return None
    
    def _entry_vwap(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """Calculate entry using VWAP"""
        try:
            if 'vwap' not in df.columns or pd.isna(df['vwap'].iloc[-1]):
                return None
            
            vwap = float(df['vwap'].iloc[-1])
            
            # Use VWAP with small offset
            if direction == "BUY":
                entry = vwap * 1.001
            else:
                entry = vwap * 0.999
            
            return entry
            
        except Exception as e:
            log.debug(f"VWAP entry failed: {e}")
            return None
    
    def _entry_support_resistance(self, df: pd.DataFrame, direction: str,
                                   current_price: float) -> Optional[float]:
        """Calculate entry using nearest support/resistance"""
        try:
            # Find support and resistance levels
            supports, resistances = self._find_support_resistance(df)
            
            if direction == "BUY":
                # Find nearest support below current price
                supports_below = [s for s in supports if s < current_price]
                if supports_below:
                    entry = max(supports_below)  # Closest support
                    return entry
            else:
                # Find nearest resistance above current price
                resistances_above = [r for r in resistances if r > current_price]
                if resistances_above:
                    entry = min(resistances_above)  # Closest resistance
                    return entry
            
            return None
            
        except Exception as e:
            log.debug(f"Support/Resistance entry failed: {e}")
            return None
    
    def _entry_bollinger_mid(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """Calculate entry using Bollinger Middle band"""
        try:
            if 'bb_middle' not in df.columns or pd.isna(df['bb_middle'].iloc[-1]):
                return None
            
            bb_mid = float(df['bb_middle'].iloc[-1])
            
            if direction == "BUY":
                entry = bb_mid * 0.998
            else:
                entry = bb_mid * 1.002
            
            return entry
            
        except Exception as e:
            log.debug(f"Bollinger entry failed: {e}")
            return None
    
    # ============================================================================
    # STOP LOSS CALCULATION METHODS
    # ============================================================================
    
    def _calculate_stop_loss(self, df: pd.DataFrame, direction: str,
                              entry: float, regime: Optional[RegimeResult]) -> Tuple[float, str]:
        """
        Calculate stop loss based on priority order
        
        Returns:
            Tuple of (stop_loss, method_used)
        """
        for method in self.sl_priority:
            if method == "swing":
                stop = self._stop_swing(df, direction, entry)
                if stop:
                    return stop, "Swing High/Low"
            
            elif method == "ema_slow":
                stop = self._stop_ema_slow(df, direction)
                if stop:
                    return stop, "EMA Slow"
            
            elif method == "bollinger_opposite":
                stop = self._stop_bollinger_opposite(df, direction)
                if stop:
                    return stop, "Bollinger Opposite"
            
            elif method == "atr":
                stop = self._stop_atr(df, direction, entry, regime)
                if stop:
                    return stop, "ATR"
        
        # Fallback: ATR-based stop
        stop = self._stop_atr(df, direction, entry, regime)
        if stop:
            return stop, "ATR (Fallback)"
        
        # Final fallback: Percentage stop
        if direction == "BUY":
            stop = entry * (1 - self.max_stop_percent)
        else:
            stop = entry * (1 + self.max_stop_percent)
        
        return stop, "Percentage (Fallback)"
    
    def _stop_swing(self, df: pd.DataFrame, direction: str, entry: float) -> Optional[float]:
        """Calculate stop using recent swing high/low"""
        try:
            # Get recent lows/highs
            recent_lows = df['low'].iloc[-self.swing_window:].min()
            recent_highs = df['high'].iloc[-self.swing_window:].max()
            
            if direction == "BUY":
                stop = recent_lows * self.sl_swing_offset_buy
                # Ensure stop is below entry
                if stop >= entry:
                    stop = entry * (1 - self.min_stop_percent)
                return stop
            else:
                stop = recent_highs * self.sl_swing_offset_sell
                if stop <= entry:
                    stop = entry * (1 + self.min_stop_percent)
                return stop
                
        except Exception as e:
            log.debug(f"Swing stop failed: {e}")
            return None
    
    def _stop_ema_slow(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """Calculate stop using EMA200"""
        try:
            if 'ema_200' not in df.columns or pd.isna(df['ema_200'].iloc[-1]):
                return None
            
            ema200 = float(df['ema_200'].iloc[-1])
            
            if direction == "BUY":
                stop = ema200 * self.sl_ema_slow_offset_buy
            else:
                stop = ema200 * self.sl_ema_slow_offset_sell
            
            return stop
            
        except Exception as e:
            log.debug(f"EMA stop failed: {e}")
            return None
    
    def _stop_bollinger_opposite(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """Calculate stop using opposite Bollinger Band"""
        try:
            if direction == "BUY":
                # For BUY, stop at lower band
                if 'bb_lower' in df.columns and not pd.isna(df['bb_lower'].iloc[-1]):
                    return float(df['bb_lower'].iloc[-1])
            else:
                # For SELL, stop at upper band
                if 'bb_upper' in df.columns and not pd.isna(df['bb_upper'].iloc[-1]):
                    return float(df['bb_upper'].iloc[-1])
            
            return None
            
        except Exception as e:
            log.debug(f"Bollinger stop failed: {e}")
            return None
    
    def _stop_atr(self, df: pd.DataFrame, direction: str,
                  entry: float, regime: Optional[RegimeResult]) -> Optional[float]:
        """Calculate stop using ATR"""
        try:
            # Get ATR
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
                atr = float(df['atr'].iloc[-1])
            else:
                # Calculate approximate ATR
                atr = entry * 0.02
            
            # Adjust multiplier based on regime
            multiplier = self.atr_stop_multiplier
            if regime:
                if regime.volatility_state in ['HIGH', 'EXTREME_HIGH']:
                    multiplier *= ATR_HIGH_VOL_MULTIPLIER
                elif regime.volatility_state in ['LOW', 'EXTREME_LOW']:
                    multiplier *= ATR_LOW_VOL_MULTIPLIER
            
            if direction == "BUY":
                stop = entry - (atr * multiplier)
            else:
                stop = entry + (atr * multiplier)
            
            return stop
            
        except Exception as e:
            log.debug(f"ATR stop failed: {e}")
            return None
    
    # ============================================================================
    # TAKE PROFIT CALCULATION METHODS
    # ============================================================================
    
    def _calculate_take_profit(self, df: pd.DataFrame, direction: str,
                                entry: float, stop_loss: float,
                                confidence: float, regime: Optional[RegimeResult]) -> Tuple[float, str]:
        """
        Calculate take profit based on priority order
        
        Returns:
            Tuple of (take_profit, method_used)
        """
        for method in self.tp_priority:
            if method == "resistance_support":
                tp = self._tp_resistance_support(df, direction, entry)
                if tp:
                    return tp, "Resistance/Support"
            
            elif method == "bollinger_opposite":
                tp = self._tp_bollinger_opposite(df, direction)
                if tp:
                    return tp, "Bollinger Opposite"
            
            elif method == "atr_multiple":
                tp = self._tp_atr_multiple(df, direction, entry, stop_loss, confidence, regime)
                if tp:
                    return tp, "ATR Multiple"
        
        # Fallback: ATR-based
        tp = self._tp_atr_multiple(df, direction, entry, stop_loss, confidence, regime)
        if tp:
            return tp, "ATR Multiple (Fallback)"
        
        # Final fallback: Fixed risk/reward
        risk = abs(entry - stop_loss)
        if direction == "BUY":
            tp = entry + (risk * 2.0)
        else:
            tp = entry - (risk * 2.0)
        
        return tp, "Fixed RR (Fallback)"
    
    def _tp_resistance_support(self, df: pd.DataFrame, direction: str,
                                entry: float) -> Optional[float]:
        """Calculate TP using next resistance/support level"""
        try:
            supports, resistances = self._find_support_resistance(df)
            
            if direction == "BUY":
                # Find next resistance above entry
                resistances_above = [r for r in resistances if r > entry]
                if resistances_above:
                    return min(resistances_above)
            else:
                # Find next support below entry
                supports_below = [s for s in supports if s < entry]
                if supports_below:
                    return max(supports_below)
            
            return None
            
        except Exception as e:
            log.debug(f"Resistance/Support TP failed: {e}")
            return None
    
    def _tp_bollinger_opposite(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """Calculate TP using opposite Bollinger Band"""
        try:
            if direction == "BUY":
                if 'bb_upper' in df.columns and not pd.isna(df['bb_upper'].iloc[-1]):
                    return float(df['bb_upper'].iloc[-1])
            else:
                if 'bb_lower' in df.columns and not pd.isna(df['bb_lower'].iloc[-1]):
                    return float(df['bb_lower'].iloc[-1])
            
            return None
            
        except Exception as e:
            log.debug(f"Bollinger TP failed: {e}")
            return None
    
    def _tp_atr_multiple(self, df: pd.DataFrame, direction: str,
                          entry: float, stop_loss: float,
                          confidence: float, regime: Optional[RegimeResult]) -> Optional[float]:
        """Calculate TP using ATR multiple with confidence adjustment"""
        try:
            # Get ATR
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
                atr = float(df['atr'].iloc[-1])
            else:
                atr = entry * 0.02
            
            # Get multiplier based on confidence
            multiplier = self._get_tp_multiplier(confidence, regime)
            
            # Calculate risk
            risk = abs(entry - stop_loss)
            
            # Use max of ATR-based or risk-based target
            atr_target = atr * multiplier
            risk_target = risk * self.dynamic_rr.get("medium_confidence", 1.5)
            target = max(atr_target, risk_target)
            
            # Cap maximum target
            max_target = entry * self.max_tp_percent
            target = min(target, max_target)
            
            if direction == "BUY":
                tp = entry + target
            else:
                tp = entry - target
            
            return tp
            
        except Exception as e:
            log.debug(f"ATR TP failed: {e}")
            return None
    
    # ============================================================================
    # SUPPORT/RESISTANCE DETECTION
    # ============================================================================
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels
        
        Returns:
            Tuple of (supports, resistances)
        """
        try:
            supports = []
            resistances = []
            
            # Use Donchian if available
            if 'donchian_low' in df.columns and not pd.isna(df['donchian_low'].iloc[-1]):
                supports.append(float(df['donchian_low'].iloc[-1]))
            if 'donchian_high' in df.columns and not pd.isna(df['donchian_high'].iloc[-1]):
                resistances.append(float(df['donchian_high'].iloc[-1]))
            
            # Use previous swing points
            if 'previous_low' in df.columns and not pd.isna(df['previous_low'].iloc[-1]):
                supports.append(float(df['previous_low'].iloc[-1]))
            if 'previous_high' in df.columns and not pd.isna(df['previous_high'].iloc[-1]):
                resistances.append(float(df['previous_high'].iloc[-1]))
            
            # Use EMAs as dynamic support/resistance
            if 'ema_50' in df.columns and not pd.isna(df['ema_50'].iloc[-1]):
                supports.append(float(df['ema_50'].iloc[-1]))
                resistances.append(float(df['ema_50'].iloc[-1]))
            
            # Use VWAP
            if 'vwap' in df.columns and not pd.isna(df['vwap'].iloc[-1]):
                supports.append(float(df['vwap'].iloc[-1]))
                resistances.append(float(df['vwap'].iloc[-1]))
            
            # Remove duplicates within 0.1%
            unique_supports = []
            for s in sorted(supports):
                if not unique_supports or abs(s - unique_supports[-1]) / s > 0.001:
                    unique_supports.append(s)
            
            unique_resistances = []
            for r in sorted(resistances):
                if not unique_resistances or abs(r - unique_resistances[-1]) / r > 0.001:
                    unique_resistances.append(r)
            
            return unique_supports[-5:], unique_resistances[:5]
            
        except Exception as e:
            log.debug(f"Support/Resistance detection failed: {e}")
            return [], []
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _get_tp_multiplier(self, confidence: float, regime: Optional[RegimeResult]) -> float:
        """Get take profit multiplier based on confidence"""
        if confidence >= 0.85:
            multiplier = 2.5
        elif confidence >= 0.75:
            multiplier = 2.0
        elif confidence >= 0.65:
            multiplier = 1.5
        else:
            multiplier = 1.2
        
        # Adjust for volatility
        if regime:
            if regime.volatility_state in ['HIGH', 'EXTREME_HIGH']:
                multiplier *= 1.2
            elif regime.volatility_state in ['LOW', 'EXTREME_LOW']:
                multiplier *= 0.8
        
        return multiplier
    
    def _get_min_risk_reward(self, confidence: float, regime: Optional[RegimeResult]) -> float:
        """Get minimum risk/reward requirement based on confidence"""
        if confidence >= 0.85:
            min_rr = self.dynamic_rr["high_confidence"]
        elif confidence >= 0.75:
            min_rr = self.dynamic_rr["medium_confidence"]
        else:
            min_rr = self.dynamic_rr["low_confidence"]
        
        # Adjust for regime
        if regime:
            if regime.regime_type == "VOLATILE":
                min_rr *= 1.2  # Need higher RR in volatile markets
        
        return min_rr
    
    def _validate_setup(self, entry: float, stop_loss: float,
                        take_profit: float, risk_reward: float,
                        direction: str) -> Tuple[bool, List[str]]:
        """
        Validate trade setup
        
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for zero/invalid values
        if entry <= 0:
            errors.append("Invalid entry price")
        if stop_loss <= 0:
            errors.append("Invalid stop loss")
        if take_profit <= 0:
            errors.append("Invalid take profit")
        
        # Check stop loss direction
        if direction == "BUY" and stop_loss >= entry:
            errors.append(f"Stop loss ({stop_loss:.4f}) must be below entry ({entry:.4f})")
        if direction == "SELL" and stop_loss <= entry:
            errors.append(f"Stop loss ({stop_loss:.4f}) must be above entry ({entry:.4f})")
        
        # Check take profit direction
        if direction == "BUY" and take_profit <= entry:
            errors.append(f"Take profit ({take_profit:.4f}) must be above entry ({entry:.4f})")
        if direction == "SELL" and take_profit >= entry:
            errors.append(f"Take profit ({take_profit:.4f}) must be below entry ({entry:.4f})")
        
        # Check stop loss percentage
        stop_percent = abs(entry - stop_loss) / entry
        if stop_percent > self.max_stop_percent:
            errors.append(f"Stop loss too wide: {stop_percent:.2%} > {self.max_stop_percent:.0%}")
        if stop_percent < self.min_stop_percent:
            errors.append(f"Stop loss too tight: {stop_percent:.2%} < {self.min_stop_percent:.2%}")
        
        # Check take profit percentage
        tp_percent = abs(take_profit - entry) / entry
        if tp_percent > self.max_tp_percent:
            errors.append(f"Take profit too large: {tp_percent:.2%} > {self.max_tp_percent:.0%}")
        
        # Check risk/reward
        if risk_reward <= 0:
            errors.append("Invalid risk/reward ratio")
        
        return len(errors) == 0, errors


# ============================================================================
# SIMPLE WRAPPER FUNCTIONS
# ============================================================================

def calculate_trade_levels(df: pd.DataFrame,
                           direction: str,
                           confidence: float,
                           regime: Optional[RegimeResult] = None,
                           current_price: Optional[float] = None) -> TradeSetup:
    """
    Simple wrapper function for trade setup calculation
    """
    engine = EntrySLTPEngine()
    return engine.calculate_trade_setup(df, direction, confidence, regime, current_price)


def get_dynamic_stop_loss(df: pd.DataFrame, direction: str, entry: float) -> float:
    """
    Get dynamic stop loss based on ATR
    """
    engine = EntrySLTPEngine()
    stop, _ = engine._calculate_stop_loss(df, direction, entry, None)
    return stop


def get_dynamic_take_profit(df: pd.DataFrame, direction: str, entry: float, confidence: float) -> float:
    """
    Get dynamic take profit based on ATR and confidence
    """
    engine = EntrySLTPEngine()
    # Create a dummy stop loss for calculation
    stop, _ = engine._calculate_stop_loss(df, direction, entry, None)
    tp, _ = engine._calculate_take_profit(df, direction, entry, stop, confidence, None)
    return tp