# signal_validator.py - Phase 10: Final Signal Validation
"""
Final Signal Validation Pipeline (Phase 10) - Enhanced

Performs final safety checks before signal output:
- Probability threshold validation
- Risk/Reward verification
- Entry/Stop/Take profit logic
- Duplicate signal prevention
- Cooldown enforcement
- Signal expiry checks
- Market condition compatibility
- FAKEOUT DETECTION
- LIQUIDITY TRAP DETECTION
- STOP HUNT DETECTION

This is the FINAL gate before signal is considered executable.
All signals that pass this phase go to Phase 11 (Timing Predictor) or directly to output.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field

from config import Config
from logger import log
from unified_signal import UnifiedSignal, SignalStatus, EntryType

@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    is_valid: bool
    reason: str
    details: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class SignalValidator:
    """
    Final Signal Validator - Phase 10
    
    Validates signals from Phase 9 (Final Scoring) and applies final filters.
    This is the last gate before signal execution.
    """
    
    def __init__(self):
        self.validation_history: List[ValidationResult] = []
        self.recent_signals: Dict[str, List[datetime]] = {}
        self.max_history = 1000
        
        # ===== LOAD CONFIGURATION =====
        # Phase 9 thresholds
        self.min_probability = getattr(Config, 'FINAL_MIN_PROBABILITY', 0.65)
        self.min_grade = getattr(Config, 'VALIDATOR_MIN_GRADE', 'C')
        
        # Risk/Reward
        self.min_rr = getattr(Config, 'MIN_RISK_REWARD', 1.5)
        
        # Cooldown
        self.cooldown_minutes = getattr(Config, 'SYMBOL_COOLDOWN_MINUTES', 60)
        self.max_signal_age_minutes = getattr(Config, 'MAX_SIGNAL_AGE_MINUTES', 120)
        
        # Time filter
        self.enable_time_filter = getattr(Config, 'TIME_FILTER_ENABLED', True)
        
        # ===== ADVANCED DETECTION SETTINGS =====
        self.enable_fakeout_detection = getattr(Config, 'ENABLE_FAKEOUT_DETECTION', True)
        self.enable_liquidity_trap_detection = getattr(Config, 'ENABLE_LIQUIDITY_TRAP_DETECTION', True)
        self.enable_stop_hunt_detection = getattr(Config, 'ENABLE_STOP_HUNT_DETECTION', True)
        
        # Detection thresholds
        self.fakeout_wick_threshold = getattr(Config, 'FAKEOUT_WICK_THRESHOLD', 0.015)  # 1.5%
        self.fakeout_volume_threshold = getattr(Config, 'FAKEOUT_VOLUME_THRESHOLD', 1.5)  # 1.5x
        self.liquidity_trap_distance = getattr(Config, 'LIQUIDITY_TRAP_DISTANCE', 0.01)  # 1%
        self.stop_hunt_atr_multiple = getattr(Config, 'STOP_HUNT_ATR_MULTIPLE', 1.5)
        
        log.info("=" * 60)
        log.info("Signal Validator initialized (Phase 10)")
        log.info("=" * 60)
        log.info(f"  Min Probability: {self.min_probability:.1%}")
        log.info(f"  Min Grade: {self.min_grade}")
        log.info(f"  Min RR: {self.min_rr}")
        log.info(f"  Cooldown: {self.cooldown_minutes} min")
        log.info(f"  Max Age: {self.max_signal_age_minutes} min")
        log.info(f"  Fakeout Detection: {self.enable_fakeout_detection}")
        log.info(f"  Liquidity Trap: {self.enable_liquidity_trap_detection}")
        log.info(f"  Stop Hunt: {self.enable_stop_hunt_detection}")
        log.info("=" * 60)
    
    def validate_final_signal(self, signal: UnifiedSignal, df: Optional[pd.DataFrame] = None,
                             market_context: Optional[Dict] = None,
                             liquidity_data: Optional[Dict] = None,
                             structure_data: Optional[Dict] = None) -> ValidationResult:
        """
        Perform final validation on a signal that passed Phase 9 (Final Scoring)
        
        Args:
            signal: Signal object from Phase 9 (with final probability and grade)
            df: OHLCV DataFrame for additional checks
            market_context: Optional market context for regime validation
            liquidity_data: Optional liquidity data from Phase 6
            structure_data: Optional market structure data
        
        Returns:
            ValidationResult with is_valid flag and details
        """
        details = {'checks': []}
        warnings = []
        suggestions = []
        all_passed = True
        primary_reason = "Signal validated successfully"
        
        # Generate unique validation ID
        validation_id = f"VAL_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # ===== CHECK 1: Signal Object Integrity =====
            check1_passed, check1_msg, check1_details = self._check_signal_integrity(signal)
            details['checks'].append({
                'name': 'signal_integrity',
                'passed': check1_passed,
                'message': check1_msg,
                'details': check1_details
            })
            if not check1_passed:
                all_passed = False
                primary_reason = check1_msg
            
            # ===== CHECK 2: Probability Threshold (from Phase 9) =====
            check2_passed, check2_msg, check2_details = self._check_probability_threshold(signal)
            details['checks'].append({
                'name': 'probability_threshold',
                'passed': check2_passed,
                'message': check2_msg,
                'details': check2_details
            })
            if not check2_passed:
                all_passed = False
                primary_reason = check2_msg
            
            # ===== CHECK 3: Grade Threshold (from Phase 9) =====
            check3_passed, check3_msg, check3_details = self._check_grade_threshold(signal)
            details['checks'].append({
                'name': 'grade_threshold',
                'passed': check3_passed,
                'message': check3_msg,
                'details': check3_details
            })
            if not check3_passed:
                all_passed = False
                primary_reason = check3_msg
            
            # ===== CHECK 4: Risk/Reward Validation (from Phase 8) =====
            check4_passed, check4_msg, check4_details = self._check_risk_reward(signal)
            details['checks'].append({
                'name': 'risk_reward',
                'passed': check4_passed,
                'message': check4_msg,
                'details': check4_details
            })
            if not check4_passed:
                all_passed = False
                primary_reason = check4_msg
            elif signal.risk_reward_ratio < self.min_rr * 1.2:
                warnings.append(f"RR below optimal: {signal.risk_reward_ratio:.2f} (target > {self.min_rr*1.2:.2f})")
            
            # ===== CHECK 5: Entry Zone Logic =====
            check5_passed, check5_msg, check5_details = self._check_entry_zone(signal)
            details['checks'].append({
                'name': 'entry_zone',
                'passed': check5_passed,
                'message': check5_msg,
                'details': check5_details
            })
            if not check5_passed:
                all_passed = False
                primary_reason = check5_msg
            
            # ===== CHECK 6: Stop Loss Logic =====
            check6_passed, check6_msg, check6_details = self._check_stop_loss(signal)
            details['checks'].append({
                'name': 'stop_loss',
                'passed': check6_passed,
                'message': check6_msg,
                'details': check6_details
            })
            if not check6_passed:
                all_passed = False
                primary_reason = check6_msg
            
            # ===== CHECK 7: Take Profit Logic =====
            check7_passed, check7_msg, check7_details = self._check_take_profit(signal)
            details['checks'].append({
                'name': 'take_profit',
                'passed': check7_passed,
                'message': check7_msg,
                'details': check7_details
            })
            if not check7_passed:
                all_passed = False
                primary_reason = check7_msg
            
            # ===== CHECK 8: Duplicate Signal Prevention =====
            check8_passed, check8_msg, check8_details = self._check_duplicate_signal(signal)
            details['checks'].append({
                'name': 'duplicate',
                'passed': check8_passed,
                'message': check8_msg,
                'details': check8_details
            })
            if not check8_passed:
                all_passed = False
                primary_reason = check8_msg
            
            # ===== CHECK 9: Signal Expiry =====
            check9_passed, check9_msg, check9_details = self._check_signal_expiry(signal)
            details['checks'].append({
                'name': 'expiry',
                'passed': check9_passed,
                'message': check9_msg,
                'details': check9_details
            })
            if not check9_passed:
                all_passed = False
                primary_reason = check9_msg
            
            # ===== CHECK 10: Time Filter (Optional) =====
            if self.enable_time_filter:
                check10_passed, check10_msg, check10_details = self._check_time_filter()
                details['checks'].append({
                    'name': 'time_filter',
                    'passed': check10_passed,
                    'message': check10_msg,
                    'details': check10_details
                })
                if not check10_passed:
                    warnings.append(check10_msg)
                    suggestions.append("Consider waiting for better trading hours")
            
            # ===== CHECK 11: Market Regime Compatibility =====
            if market_context:
                check11_passed, check11_msg, check11_details = self._check_market_regime_compatibility(signal, market_context)
                details['checks'].append({
                    'name': 'market_regime',
                    'passed': check11_passed,
                    'message': check11_msg,
                    'details': check11_details
                })
                if not check11_passed:
                    warnings.append(check11_msg)
                    suggestions.append("Signal may be counter to dominant market regime")
            
            # ===== CHECK 12: Volatility Compatibility =====
            if df is not None and not df.empty:
                check12_passed, check12_msg, check12_details = self._check_volatility_compatibility(signal, df)
                details['checks'].append({
                    'name': 'volatility',
                    'passed': check12_passed,
                    'message': check12_msg,
                    'details': check12_details
                })
                if not check12_passed:
                    warnings.append(check12_msg)
            
            # ===== CHECK 13: Position Size Sanity =====
            if hasattr(signal, 'position_size') and signal.position_size > 0:
                check13_passed, check13_msg, check13_details = self._check_position_size(signal)
                details['checks'].append({
                    'name': 'position_size',
                    'passed': check13_passed,
                    'message': check13_msg,
                    'details': check13_details
                })
                if not check13_passed:
                    warnings.append(check13_msg)
            
            # ===== CHECK 14: FAKEOUT DETECTION =====
            if self.enable_fakeout_detection and df is not None and not df.empty:
                check14_passed, check14_msg, check14_details = self._check_fakeout(signal, df)
                details['checks'].append({
                    'name': 'fakeout_detection',
                    'passed': check14_passed,
                    'message': check14_msg,
                    'details': check14_details
                })
                if not check14_passed:
                    warnings.append(check14_msg)
                    suggestions.append("Wait for confirmation after potential fakeout")
            
            # ===== CHECK 15: LIQUIDITY TRAP DETECTION =====
            if self.enable_liquidity_trap_detection and liquidity_data:
                check15_passed, check15_msg, check15_details = self._check_liquidity_trap(signal, liquidity_data, df)
                details['checks'].append({
                    'name': 'liquidity_trap',
                    'passed': check15_passed,
                    'message': check15_msg,
                    'details': check15_details
                })
                if not check15_passed:
                    warnings.append(check15_msg)
                    suggestions.append("Avoid liquidity trap - wait for confirmation")
            
            # ===== CHECK 16: STOP HUNT DETECTION =====
            if self.enable_stop_hunt_detection and df is not None and not df.empty:
                check16_passed, check16_msg, check16_details = self._check_stop_hunt(signal, df)
                details['checks'].append({
                    'name': 'stop_hunt',
                    'passed': check16_passed,
                    'message': check16_msg,
                    'details': check16_details
                })
                if not check16_passed:
                    warnings.append(check16_msg)
                    suggestions.append("Stop hunt detected - wait for reversal confirmation")
            
            # ===== CHECK 17: Expert Consensus Consistency =====
            check17_passed, check17_msg, check17_details = self._check_expert_consistency(signal)
            details['checks'].append({
                'name': 'expert_consistency',
                'passed': check17_passed,
                'message': check17_msg,
                'details': check17_details
            })
            if not check17_passed:
                warnings.append(check17_msg)
            
            # ===== GENERATE SUGGESTIONS =====
            suggestions.extend(self._generate_suggestions(signal))
            
            # ===== CREATE VALIDATION RESULT =====
            result = ValidationResult(
                is_valid=all_passed,
                reason=primary_reason,
                details=details,
                warnings=warnings,
                suggestions=suggestions[:5],
                validation_id=validation_id,
                timestamp=datetime.now()
            )
            
            # Record successful validation
            if all_passed:
                self._record_signal(signal)
            
            # Log validation result
            self._log_validation_result(result, signal)
            
            # Store in history
            self.validation_history.append(result)
            if len(self.validation_history) > self.max_history:
                self.validation_history = self.validation_history[-self.max_history:]
            
            return result
            
        except Exception as e:
            log.error(f"Validation error for {signal.symbol}: {e}")
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                details={'error': str(e)},
                warnings=['Validation process failed'],
                validation_id=validation_id
            )
    
    # ==================== NEW CHECK: EXPERT CONSISTENCY ====================
    
    def _check_expert_consistency(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Check if expert signals are consistent with final direction"""
        details = {}
        
        if not hasattr(signal, 'metadata') or not signal.metadata:
            return True, "No expert metadata available", details
        
        expert_details = signal.metadata.get('expert_details', {})
        if not expert_details:
            return True, "No expert details available", details
        
        # Count agreeing experts
        agreeing = sum(1 for e in expert_details.values() if e.get('agreed', False))
        total = len(expert_details)
        agreement_ratio = agreeing / total if total > 0 else 0
        
        details['agreeing_experts'] = agreeing
        details['total_experts'] = total
        details['agreement_ratio'] = agreement_ratio
        
        if agreement_ratio < 0.6:
            return False, f"Low expert agreement: {agreement_ratio:.0%} ({agreeing}/{total})", details
        
        return True, f"Expert agreement: {agreement_ratio:.0%}", details
    
    # ==================== NEW CHECK: GRADE THRESHOLD ====================
    
    def _check_grade_threshold(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Check if signal grade meets minimum threshold"""
        grade_order = {'A+': 1, 'A': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8, 'D': 9, 'F': 10}
        
        min_grade_order = grade_order.get(self.min_grade, 10)
        current_grade_order = grade_order.get(signal.grade, 10)
        
        details = {
            'signal_grade': signal.grade,
            'min_grade': self.min_grade,
            'grade_order_current': current_grade_order,
            'grade_order_min': min_grade_order
        }
        
        if current_grade_order > min_grade_order:
            return False, f"Grade {signal.grade} below minimum {self.min_grade}", details
        
        return True, f"Grade {signal.grade} meets threshold", details
    
    # ==================== EXISTING METHODS (Keep as is) ====================
    
    def _check_signal_integrity(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Check if signal object has all required fields"""
        details = {}
        missing_fields = []
        
        # Phase 9 fields
        required_fields = ['symbol', 'timeframe', 'direction', 'probability', 'signal_grade',
                          'entry_zone_low', 'entry_zone_high', 'stop_loss', 'take_profit',
                          'risk_reward_ratio']
        
        for field in required_fields:
            if not hasattr(signal, field):
                missing_fields.append(field)
            elif getattr(signal, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}", details
        
        return True, "Signal integrity OK", details
    
    def _check_probability_threshold(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Check if final probability meets minimum threshold"""
        details = {
            'probability': signal.probability,
            'threshold': self.min_probability
        }
        
        if signal.probability < self.min_probability:
            return False, f"Probability {signal.probability:.1%} below threshold {self.min_probability:.1%}", details
        
        return True, f"Probability {signal.probability:.1%} meets threshold", details
    
    def _check_risk_reward(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Validate risk/reward ratio"""
        details = {
            'risk_reward': signal.risk_reward_ratio,
            'min_required': self.min_rr
        }
        
        if signal.risk_reward_ratio < self.min_rr:
            return False, f"Risk/Reward {signal.risk_reward_ratio:.2f} below minimum {self.min_rr:.2f}", details
        
        return True, f"Risk/Reward {signal.risk_reward_ratio:.2f} OK", details
    
    def _check_entry_zone(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Validate entry zone logic"""
        details = {
            'zone_low': signal.entry_zone_low,
            'zone_high': signal.entry_zone_high,
            'entry_type': signal.entry_type.value if signal.entry_type else 'UNKNOWN'
        }
        
        if signal.entry_zone_low is None or signal.entry_zone_high is None:
            return False, "Entry zone not defined", details
        
        if signal.entry_zone_low >= signal.entry_zone_high:
            return False, f"Invalid entry zone: low {signal.entry_zone_low:.6f} >= high {signal.entry_zone_high:.6f}", details
        
        # Check zone width (not too wide)
        zone_width_pct = (signal.entry_zone_high - signal.entry_zone_low) / signal.entry_zone_low
        if zone_width_pct > 0.02:  # More than 2% wide
            details['zone_width_pct'] = zone_width_pct
            return False, f"Entry zone too wide: {zone_width_pct:.2%}", details
        
        return True, "Entry zone valid", details
    
    def _check_stop_loss(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Validate stop loss logic based on direction"""
        details = {
            'stop_loss': signal.stop_loss,
            'entry_zone_high': signal.entry_zone_high,
            'entry_zone_low': signal.entry_zone_low,
            'direction': signal.direction
        }
        
        if signal.stop_loss is None:
            return False, "Stop loss not defined", details
        
        if signal.direction == 'BUY':
            if signal.stop_loss >= signal.entry_zone_high:
                return False, f"Stop loss {signal.stop_loss:.6f} not below entry zone {signal.entry_zone_high:.6f}", details
        else:  # SELL
            if signal.stop_loss <= signal.entry_zone_low:
                return False, f"Stop loss {signal.stop_loss:.6f} not above entry zone {signal.entry_zone_low:.6f}", details
        
        return True, "Stop loss valid", details
    
    def _check_take_profit(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Validate take profit logic based on direction"""
        details = {
            'take_profit': signal.take_profit,
            'entry_zone_high': signal.entry_zone_high,
            'entry_zone_low': signal.entry_zone_low,
            'direction': signal.direction
        }
        
        if signal.take_profit is None:
            return False, "Take profit not defined", details
        
        if signal.direction == 'BUY':
            if signal.take_profit <= signal.entry_zone_low:
                return False, f"Take profit {signal.take_profit:.6f} not above entry zone {signal.entry_zone_low:.6f}", details
        else:  # SELL
            if signal.take_profit >= signal.entry_zone_high:
                return False, f"Take profit {signal.take_profit:.6f} not below entry zone {signal.entry_zone_high:.6f}", details
        
        return True, "Take profit valid", details
    
    def _check_duplicate_signal(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Check for duplicate signals within cooldown period"""
        details = {
            'cooldown_minutes': self.cooldown_minutes
        }
        
        key = f"{signal.symbol}_{signal.direction}"
        now = datetime.now()
        
        if key in self.recent_signals:
            # Remove old signals outside cooldown
            self.recent_signals[key] = [ts for ts in self.recent_signals[key] 
                                       if (now - ts).total_seconds() / 60 < self.cooldown_minutes]
            
            if self.recent_signals[key]:
                latest = max(self.recent_signals[key])
                minutes_ago = (now - latest).total_seconds() / 60
                remaining = self.cooldown_minutes - minutes_ago
                
                details['last_signal'] = latest.isoformat()
                details['minutes_ago'] = round(minutes_ago, 1)
                details['remaining'] = round(remaining, 1)
                
                return False, f"Duplicate signal within cooldown ({remaining:.1f} min remaining)", details
        
        return True, "No duplicate signal detected", details
    
    def _check_signal_expiry(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Check if signal has expired"""
        age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
        
        details = {
            'signal_time': signal.timestamp.isoformat(),
            'age_minutes': round(age_minutes, 1),
            'max_age': self.max_signal_age_minutes
        }
        
        if age_minutes > self.max_signal_age_minutes:
            return False, f"Signal expired ({age_minutes:.1f} min old)", details
        
        return True, "Signal within expiry window", details
    
    def _check_time_filter(self) -> Tuple[bool, str, Dict]:
        """Check if current time is within optimal trading hours"""
        hour = datetime.now().hour
        
        # Optimal trading hours (UTC) - high liquidity periods
        optimal_hours = [0,1,2,3,4,5,13,14,15,16,17,18,19,20,21]  # London/NY overlap
        acceptable_hours = [6,7,8,9,10,11,12,22,23]  # Asian session, late NY
        
        details = {'current_hour': hour}
        
        if hour in optimal_hours:
            return True, "Optimal trading hour", details
        elif hour in acceptable_hours:
            return True, "Acceptable trading hour", details
        else:
            return False, f"Suboptimal trading hour ({hour}:00 UTC)", details
    
    def _check_market_regime_compatibility(self, signal: UnifiedSignal, market_context: Dict) -> Tuple[bool, str, Dict]:
        """Check if signal direction aligns with market regime"""
        regime = market_context.get('composite_regime', 'UNKNOWN')
        trend = market_context.get('trend', 'NEUTRAL')
        
        details = {
            'regime': regime,
            'trend': trend,
            'signal_direction': signal.direction
        }
        
        # Strong trend alignment
        if 'STRONG_TREND_BULL' in regime and signal.direction == 'BUY':
            return True, "Signal aligned with strong bull trend", details
        elif 'STRONG_TREND_BEAR' in regime and signal.direction == 'SELL':
            return True, "Signal aligned with strong bear trend", details
        
        # Basic trend alignment
        if (trend == 'BULL' and signal.direction == 'BUY') or (trend == 'BEAR' and signal.direction == 'SELL'):
            return True, "Signal aligned with trend", details
        
        # Counter-trend signal
        if (trend == 'BULL' and signal.direction == 'SELL') or (trend == 'BEAR' and signal.direction == 'BUY'):
            return False, "Signal counter to dominant trend", details
        
        return True, "Neutral regime", details
    
    def _check_volatility_compatibility(self, signal: UnifiedSignal, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check if volatility is appropriate for this signal"""
        if 'atr' not in df.columns:
            return True, "No volatility data", {}
        
        current_atr = float(df['atr'].iloc[-1])
        current_price = float(df['close'].iloc[-1])
        atr_pct = current_atr / current_price
        
        # Calculate distance to stop
        if signal.direction == 'BUY':
            stop_distance = abs(current_price - signal.stop_loss) / current_price
        else:
            stop_distance = abs(signal.stop_loss - current_price) / current_price
        
        details = {
            'atr_pct': round(atr_pct, 4),
            'stop_distance_pct': round(stop_distance, 4),
            'volatility_state': signal.volatility_state
        }
        
        # Check if stop is too tight relative to volatility
        if stop_distance < atr_pct * 0.5:
            return False, f"Stop too tight: {stop_distance:.2%} vs ATR {atr_pct:.2%}", details
        
        return True, "Volatility compatible", details
    
    def _check_position_size(self, signal: UnifiedSignal) -> Tuple[bool, str, Dict]:
        """Check position size sanity"""
        if not hasattr(signal, 'position_size') or not hasattr(signal, 'position_value'):
            return True, "No position size data", {}
        
        details = {
            'position_size': signal.position_size,
            'position_value': signal.position_value
        }
        
        # Check for extremely small positions (dust)
        if signal.position_value < 10:  # Less than $10
            return False, f"Position value too small: ${signal.position_value:.2f}", details
        
        return True, "Position size OK", details
    
    # ==================== ADVANCED DETECTION METHODS ====================
    
    def _check_fakeout(self, signal: UnifiedSignal, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """
        Detect fakeout patterns:
        - Price breaks above resistance but closes below (bearish fakeout)
        - Price breaks below support but closes above (bullish fakeout)
        """
        try:
            if len(df) < 5:
                return True, "Insufficient data for fakeout detection", {}
            
            current_price = float(df['close'].iloc[-1])
            current_high = float(df['high'].iloc[-1])
            current_low = float(df['low'].iloc[-1])
            current_close = float(df['close'].iloc[-1])
            
            # Calculate ATR for wick threshold
            atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
            wick_threshold = max(atr * self.fakeout_wick_threshold, current_price * 0.005)
            
            # Get recent resistance and support levels
            recent_high = df['high'].iloc[-20:-1].max() if len(df) > 20 else current_high
            recent_low = df['low'].iloc[-20:-1].min() if len(df) > 20 else current_low
            
            details = {
                'recent_high': recent_high,
                'recent_low': recent_low,
                'current_price': current_price,
                'wick_threshold': wick_threshold
            }
            
            if signal.direction == 'BUY':
                # Bullish fakeout: price breaks below support but closes above (bullish trap)
                if current_low < recent_low and current_close > recent_low:
                    wick_size = recent_low - current_low
                    if wick_size > wick_threshold:
                        details['fakeout_type'] = 'BULLISH_FAKEOUT_BELOW_SUPPORT'
                        # This is actually a bullish signal (false breakdown)
                        return True, "Bullish fakeout detected (false breakdown) - potential reversal up", details
                
                # Bearish fakeout: price breaks above resistance but closes below (bearish trap)
                if current_high > recent_high and current_close < recent_high:
                    wick_size = current_high - recent_high
                    if wick_size > wick_threshold:
                        volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df else 1
                        if volume_ratio > self.fakeout_volume_threshold:
                            details['fakeout_type'] = 'BEARISH_FAKEOUT'
                            return False, f"Bearish fakeout detected: price broke above resistance but closed below", details
            
            else:  # SELL
                # Bearish fakeout: price breaks above resistance but closes below (bearish trap)
                if current_high > recent_high and current_close < recent_high:
                    wick_size = current_high - recent_high
                    if wick_size > wick_threshold:
                        details['fakeout_type'] = 'BEARISH_FAKEOUT_ABOVE_RESISTANCE'
                        return True, "Bearish fakeout detected (false breakout) - potential reversal down", details
                
                # Bullish fakeout: price breaks below support but closes above (bullish trap)
                if current_low < recent_low and current_close > recent_low:
                    wick_size = recent_low - current_low
                    if wick_size > wick_threshold:
                        volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df else 1
                        if volume_ratio > self.fakeout_volume_threshold:
                            details['fakeout_type'] = 'BULLISH_FAKEOUT'
                            return False, f"Bullish fakeout detected: price broke below support but closed above", details
            
            return True, "No fakeout detected", details
            
        except Exception as e:
            log.debug(f"Fakeout detection error: {e}")
            return True, f"Fakeout check error: {e}", {'error': str(e)}
    
    def _check_liquidity_trap(self, signal: UnifiedSignal, liquidity_data: Dict, 
                               df: Optional[pd.DataFrame] = None) -> Tuple[bool, str, Dict]:
        """Detect liquidity traps near equal highs/lows"""
        try:
            details = {}
            
            sweeps = liquidity_data.get('sweeps', [])
            equal_highs = liquidity_data.get('equal_highs', [])
            equal_lows = liquidity_data.get('equal_lows', [])
            
            current_price = signal.entry_zone_high if signal.direction == 'BUY' else signal.entry_zone_low
            if df is not None and not df.empty:
                current_price = float(df['close'].iloc[-1])
            
            details = {
                'sweeps_count': len(sweeps),
                'equal_highs_count': len(equal_highs),
                'equal_lows_count': len(equal_lows)
            }
            
            if signal.direction == 'BUY':
                for eq_low in equal_lows[:3]:
                    distance = abs(current_price - eq_low['price']) / current_price
                    if distance < self.liquidity_trap_distance:
                        return False, f"Liquidity trap: price near equal low ({eq_low['price']:.6f})", details
            else:
                for eq_high in equal_highs[:3]:
                    distance = abs(current_price - eq_high['price']) / current_price
                    if distance < self.liquidity_trap_distance:
                        return False, f"Liquidity trap: price near equal high ({eq_high['price']:.6f})", details
            
            return True, "No liquidity trap detected", details
            
        except Exception as e:
            log.debug(f"Liquidity trap detection error: {e}")
            return True, f"Liquidity trap check error: {e}", {'error': str(e)}
    
    def _check_stop_hunt(self, signal: UnifiedSignal, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Detect stop hunts with large wicks and reversals"""
        try:
            if len(df) < 10:
                return True, "Insufficient data for stop hunt detection", {}
            
            current_price = float(df['close'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
            stop_hunt_threshold = atr * self.stop_hunt_atr_multiple
            
            recent = df.iloc[-5:]
            vol_mean = recent['volume'].mean()
            
            details = {'atr': atr, 'stop_hunt_threshold': stop_hunt_threshold}
            
            for i in range(len(recent) - 1):
                candle = recent.iloc[i]
                next_candle = recent.iloc[i + 1]
                
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                
                if signal.direction == 'BUY':
                    if lower_wick > stop_hunt_threshold:
                        if candle['volume'] > vol_mean * 1.5:
                            if next_candle['close'] > candle['open']:
                                return False, "Stop hunt detected: large lower wick with reversal up", details
                else:
                    if upper_wick > stop_hunt_threshold:
                        if candle['volume'] > vol_mean * 1.5:
                            if next_candle['close'] < candle['open']:
                                return False, "Stop hunt detected: large upper wick with reversal down", details
            
            return True, "No stop hunt detected", details
            
        except Exception as e:
            log.debug(f"Stop hunt detection error: {e}")
            return True, f"Stop hunt check error: {e}", {'error': str(e)}
    
    # ==================== UTILITY METHODS ====================
    
    def _generate_suggestions(self, signal: UnifiedSignal) -> List[str]:
        """Generate actionable suggestions based on validation results"""
        suggestions = []
        
        # Entry type suggestion
        if signal.entry_type == EntryType.LIMIT:
            suggestions.append(f"Place LIMIT order between {signal.entry_zone_low:.4f} and {signal.entry_zone_high:.4f}")
        elif signal.entry_type == EntryType.MARKET:
            suggestions.append(f"Consider MARKET entry at current price")
        
        # Position sizing suggestion
        if hasattr(signal, 'position_value') and signal.position_value > 0:
            pct = (signal.position_value / getattr(Config, 'PORTFOLIO_VALUE', 10000)) * 100
            suggestions.append(f"Risk ${signal.position_value:.0f} ({pct:.1f}% of portfolio)")
        
        # Stop loss suggestion
        suggestions.append(f"Place STOP at {signal.stop_loss:.4f}")
        
        # Take profit suggestion (primary TP)
        if hasattr(signal, 'take_profit_levels') and signal.take_profit_levels:
            tp1 = signal.take_profit_levels[0].price
            suggestions.append(f"First target at {tp1:.4f} (RR: {signal.risk_reward_ratio:.2f})")
        elif signal.take_profit:
            suggestions.append(f"Take profit at {signal.take_profit:.4f} (RR: {signal.risk_reward_ratio:.2f})")
        
        # Add from metadata
        if hasattr(signal, 'metadata'):
            story = signal.metadata.get('story_summary', '')
            if story:
                suggestions.append(f"Signal: {story[:60]}...")
        
        return suggestions[:5]
    
    def _record_signal(self, signal: UnifiedSignal):
        """Record successfully validated signal for duplicate prevention"""
        key = f"{signal.symbol}_{signal.direction}"
        if key not in self.recent_signals:
            self.recent_signals[key] = []
        self.recent_signals[key].append(datetime.now())
        
        # Clean old entries
        cutoff = datetime.now() - timedelta(minutes=self.cooldown_minutes)
        self.recent_signals[key] = [ts for ts in self.recent_signals[key] if ts > cutoff]
    
    def _log_validation_result(self, result: ValidationResult, signal: UnifiedSignal):
        """Log validation result with appropriate level"""
        if result.is_valid:
            log.info(f"\n[VALIDATION PASSED] {signal.symbol} {signal.timeframe}")
            log.info(f"  Grade: {signal.grade} | Prob: {signal.probability:.1%} | RR: {signal.risk_reward_ratio:.2f}")
            if result.warnings:
                log.info(f"  Warnings: {', '.join(result.warnings)}")
            if result.suggestions:
                log.info(f"  Suggestion: {result.suggestions[0]}")
        else:
            log.debug(f"[VALIDATION FAILED] {signal.symbol} - {result.reason}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        if not self.validation_history:
            return {'total': 0, 'pass_rate': 0, 'avg_checks': 0, 'common_warnings': []}
        
        total = len(self.validation_history)
        passed = sum(1 for v in self.validation_history if v.is_valid)
        
        # Collect common warnings
        all_warnings = []
        for v in self.validation_history[-100:]:
            all_warnings.extend(v.warnings)
        
        warning_counts = {}
        for w in all_warnings:
            warning_counts[w] = warning_counts.get(w, 0) + 1
        
        common_warnings = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_validations': total,
            'passed': passed,
            'pass_rate': round(passed / total * 100, 1) if total > 0 else 0,
            'avg_checks_passed': round(sum(len(v.details.get('checks', [])) for v in self.validation_history) / total, 1),
            'common_warnings': [f"{w} ({c}x)" for w, c in common_warnings],
            'recent_validations': [
                {
                    'symbol': v.details.get('checks', [{}])[0].get('details', {}).get('symbol', 'UNKNOWN'),
                    'result': 'PASS' if v.is_valid else 'FAIL',
                    'reason': v.reason
                }
                for v in self.validation_history[-10:]
            ]
        }
    
    def clear_history(self):
        """Clear validation history"""
        self.validation_history = []
        self.recent_signals = {}
        log.info("Validation history cleared")
