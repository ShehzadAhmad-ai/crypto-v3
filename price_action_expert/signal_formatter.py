"""
signal_formatter.py
Layer 8: Signal Formatter for Price Action Expert V3.5

Converts internal detection results into standardized output format
matching the Pattern System for seamless integration.

Output format matches:
- pattern_id: pa_YYYYMMDD_HHMMSS
- pattern_name: Standardized pattern name
- action: WAIT / ENTER_NOW / STRONG_ENTRY / FLIP_TO_* / SKIP
- Complete trade setup with entry, stop_loss, take_profit, risk_reward
- confidence, grade, position_multiplier
- decision_reason with full story
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Import configuration
from .price_action_config import (
    PATTERN_NAME_MAPPING,
    MAX_PATTERN_AGE_BARS,
    RETEST_CONFIRMATION_BARS,
    MIN_TRADE_CONFIDENCE
)


class SignalAction:
    """Standardized signal actions matching Pattern System"""
    WAIT = "WAIT"
    WAIT_FOR_RETEST = "WAIT_FOR_RETEST"
    ENTER_NOW = "ENTER_NOW"
    STRONG_ENTRY = "STRONG_ENTRY"
    FLIP_TO_BUY = "FLIP_TO_BUY"
    FLIP_TO_SELL = "FLIP_TO_SELL"
    CANCEL = "CANCEL"
    SKIP = "SKIP"
    HOLD = "HOLD"


class SignalStage:
    """Signal stages matching Pattern System"""
    FORMING = "forming"
    BREAKOUT = "breakout"
    RETEST = "retest"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class FormattedSignal:
    """
    Standardized signal output matching Pattern System format
    """
    # Core identification
    pattern_id: str
    pattern_name: str
    symbol: str
    timeframe: str
    direction: str                      # BUY, SELL, NEUTRAL, HOLD
    action: str
    action_detail: str
    
    # Trade setup
    entry: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    
    # Confidence and risk
    confidence: float
    grade: str
    position_multiplier: float
    
    # Context
    decision_reason: str
    retest_level: Optional[float]
    retest_confirmed: bool
    stage: str
    age_bars: int
    timestamp: str
    decision_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'action': self.action,
            'action_detail': self.action_detail,
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward': self.risk_reward,
            'confidence': round(self.confidence, 3),
            'grade': self.grade,
            'position_multiplier': self.position_multiplier,
            'decision_reason': self.decision_reason,
            'retest_level': self.retest_level,
            'retest_confirmed': self.retest_confirmed,
            'stage': self.stage,
            'age_bars': self.age_bars,
            'timestamp': self.timestamp,
            'decision_id': self.decision_id
        }


class SignalFormatter:
    """
    Formats internal detection results into standardized output
    
    Features:
    - Standardized pattern names
    - Action determination based on confidence and stage
    - Human-readable decision reasons
    - Retest level tracking
    - Age-based signal management
    """
    
    def __init__(self):
        """Initialize the signal formatter"""
        self.pattern_name_mapping = PATTERN_NAME_MAPPING
    
    def _get_standardized_name(self, internal_name: str) -> str:
        """Convert internal pattern name to standardized output name"""
        return self.pattern_name_mapping.get(internal_name.lower(), internal_name.replace('_', ' ').title())
    
    def _determine_action(self, confidence: float, grade: str,
                          trap_type: Optional[str] = None,
                          is_trap: bool = False,
                          stage: str = SignalStage.CONFIRMED,
                          age_bars: int = 0) -> Tuple[str, str]:
        """
        Determine action and action_detail based on confidence and context
        
        Returns:
            (action, action_detail)
        """
        # Trap override (highest priority)
        if is_trap and trap_type:
            if trap_type in ['bear_trap', 'stop_hunt_bullish']:
                return SignalAction.FLIP_TO_BUY, f"{trap_type.replace('_', ' ').title()} detected - flipping direction"
            elif trap_type in ['bull_trap', 'stop_hunt_bearish']:
                return SignalAction.FLIP_TO_SELL, f"{trap_type.replace('_', ' ').title()} detected - flipping direction"
        
        # Age-based cancellation
        if age_bars > MAX_PATTERN_AGE_BARS:
            return SignalAction.CANCEL, f"Pattern invalidated - signal was {age_bars} bars ago"
        
        # Stage-based actions
        if stage == SignalStage.FORMING:
            return SignalAction.WAIT, "Pattern forming - waiting for completion"
        
        if stage == SignalStage.BREAKOUT:
            return SignalAction.WAIT_FOR_RETEST, "Breakout confirmed - waiting for retest"
        
        if stage == SignalStage.FAILED:
            return SignalAction.CANCEL, "Pattern invalidated - setup failed"
        
        # Confidence-based actions
        if confidence >= 0.85 and grade in ['A+', 'A']:
            return SignalAction.STRONG_ENTRY, f"Strong entry - {grade} grade signal"
        elif confidence >= MIN_TRADE_CONFIDENCE:
            return SignalAction.ENTER_NOW, "Retest confirmed - enter now"
        else:
            return SignalAction.SKIP, f"Low confidence ({confidence:.0%}) - skipping"
    
    def _build_decision_reason(self, pattern_name: str, direction: str,
                                confidence: float, grade: str,
                                trap_type: Optional[str] = None,
                                trap_severity: Optional[str] = None,
                                mtf_alignment: float = 0.0,
                                at_key_level: bool = False,
                                sequence_type: str = "",
                                reasons: List[str] = None) -> str:
        """
        Build human-readable decision reason
        """
        reason_parts = []
        
        # Pattern and direction
        if direction == 'BUY':
            reason_parts.append(f"Bullish {pattern_name}")
        elif direction == 'SELL':
            reason_parts.append(f"Bearish {pattern_name}")
        else:
            reason_parts.append(pattern_name)
        
        # Trap information
        if trap_type and trap_severity:
            reason_parts.append(f"{trap_severity} {trap_type.replace('_', ' ')} detected")
        
        # Context information
        if at_key_level:
            reason_parts.append("at key level")
        
        if mtf_alignment >= 0.7:
            reason_parts.append("MTF aligned")
        
        if sequence_type and sequence_type not in ['indecision']:
            reason_parts.append(f"{sequence_type} sequence")
        
        # Confidence and grade
        reason_parts.append(f"Confidence {confidence:.0%} ({grade})")
        
        # Additional reasons from detection
        if reasons:
            # Add top 2 additional reasons
            for r in reasons[:2]:
                if r and not any(part in r for part in reason_parts):
                    reason_parts.append(r.lower())
        
        return " | ".join(reason_parts)
    
    def _determine_stage(self, pattern_age: int, has_retest: bool,
                         is_confirmed: bool) -> str:
        """
        Determine signal stage based on pattern age and confirmation status
        """
        if pattern_age <= 1:
            if is_confirmed:
                return SignalStage.CONFIRMED
            elif has_retest:
                return SignalStage.RETEST
            else:
                return SignalStage.BREAKOUT
        elif pattern_age <= 3:
            if is_confirmed:
                return SignalStage.CONFIRMED
            else:
                return SignalStage.RETEST
        elif pattern_age <= MAX_PATTERN_AGE_BARS:
            return SignalStage.FORMING
        else:
            return SignalStage.FAILED
    
    def format_signal(self,
                      symbol: str,
                      timeframe: str,
                      pattern_name: str,
                      direction: str,
                      confidence: float,
                      grade: str,
                      position_multiplier: float,
                      trade_setup: Optional[Dict] = None,
                      pattern_index: int = 0,
                      pattern_age: int = 0,
                      retest_level: Optional[float] = None,
                      retest_confirmed: bool = False,
                      trap_type: Optional[str] = None,
                      trap_severity: Optional[str] = None,
                      mtf_alignment: float = 0.0,
                      at_key_level: bool = False,
                      sequence_type: str = "",
                      additional_reasons: Optional[List[str]] = None,
                      is_trap: bool = False) -> FormattedSignal:
        """
        Format a complete signal for output
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            pattern_name: Internal pattern name
            direction: BUY, SELL, NEUTRAL, HOLD
            confidence: 0-1 confidence score
            grade: Signal grade (A+, A, B+, etc.)
            position_multiplier: Position size multiplier
            trade_setup: Optional dict with entry, stop_loss, take_profit, risk_reward
            pattern_index: Index of pattern candle
            pattern_age: Bars since pattern formed
            retest_level: Level to watch for retest
            retest_confirmed: Whether retest occurred
            trap_type: Type of trap detected
            trap_severity: Severity of trap
            mtf_alignment: MTF alignment score
            at_key_level: Whether at key level
            sequence_type: Type of sequence detected
            additional_reasons: Additional reasons for the signal
            is_trap: Whether this is a trap signal
        
        Returns:
            FormattedSignal object ready for output
        """
        # Standardize pattern name
        standardized_name = self._get_standardized_name(pattern_name)
        
        # Determine stage
        stage = self._determine_stage(pattern_age, retest_confirmed, confidence >= MIN_TRADE_CONFIDENCE)
        
        # Determine action
        action, action_detail = self._determine_action(
            confidence, grade, trap_type, is_trap, stage, pattern_age
        )
        
        # Build decision reason
        decision_reason = self._build_decision_reason(
            standardized_name, direction, confidence, grade,
            trap_type, trap_severity, mtf_alignment, at_key_level,
            sequence_type, additional_reasons
        )
        
        # Extract trade setup values
        entry = None
        stop_loss = None
        take_profit = None
        risk_reward = None
        
        if trade_setup:
            entry = trade_setup.get('entry')
            stop_loss = trade_setup.get('stop_loss')
            take_profit = trade_setup.get('take_profit')
            risk_reward = trade_setup.get('risk_reward')
        
        # Generate IDs
        timestamp = datetime.now().isoformat(timespec='seconds')
        timestamp_short = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_id = f"pa_{timestamp_short}_{uuid.uuid4().hex[:6]}"
        decision_id = f"dec_pa_{timestamp_short}"
        
        return FormattedSignal(
            pattern_id=pattern_id,
            pattern_name=standardized_name,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            action=action,
            action_detail=action_detail,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=confidence,
            grade=grade,
            position_multiplier=position_multiplier,
            decision_reason=decision_reason,
            retest_level=retest_level,
            retest_confirmed=retest_confirmed,
            stage=stage,
            age_bars=pattern_age,
            timestamp=timestamp,
            decision_id=decision_id
        )
    
    def format_skip(self, symbol: str, timeframe: str,
                    reason: str) -> FormattedSignal:
        """
        Format a SKIP signal when no trade is recommended
        """
        timestamp = datetime.now().isoformat(timespec='seconds')
        timestamp_short = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_id = f"pa_{timestamp_short}_skip"
        decision_id = f"dec_pa_{timestamp_short}_skip"
        
        return FormattedSignal(
            pattern_id=pattern_id,
            pattern_name="",
            symbol=symbol,
            timeframe=timeframe,
            direction="NEUTRAL",
            action=SignalAction.SKIP,
            action_detail=reason,
            entry=None,
            stop_loss=None,
            take_profit=None,
            risk_reward=None,
            confidence=0.0,
            grade="D",
            position_multiplier=0.0,
            decision_reason=reason,
            retest_level=None,
            retest_confirmed=False,
            stage=SignalStage.FORMING,
            age_bars=0,
            timestamp=timestamp,
            decision_id=decision_id
        )
    
    def format_wait(self, symbol: str, timeframe: str,
                    pattern_name: str, direction: str,
                    confidence: float, pattern_progress: float,
                    retest_level: Optional[float] = None) -> FormattedSignal:
        """
        Format a WAIT signal when pattern is forming
        """
        standardized_name = self._get_standardized_name(pattern_name)
        timestamp = datetime.now().isoformat(timespec='seconds')
        timestamp_short = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_id = f"pa_{timestamp_short}_wait"
        decision_id = f"dec_pa_{timestamp_short}_wait"
        
        return FormattedSignal(
            pattern_id=pattern_id,
            pattern_name=standardized_name,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            action=SignalAction.WAIT,
            action_detail=f"Pattern forming - {pattern_progress:.0%} complete",
            entry=None,
            stop_loss=None,
            take_profit=None,
            risk_reward=None,
            confidence=confidence,
            grade="C" if confidence >= 0.5 else "D",
            position_multiplier=0.5,
            decision_reason=f"{standardized_name} - Pattern forming - {pattern_progress:.0%} complete",
            retest_level=retest_level,
            retest_confirmed=False,
            stage=SignalStage.FORMING,
            age_bars=0,
            timestamp=timestamp,
            decision_id=decision_id
        )
    
    def format_wait_for_retest(self, symbol: str, timeframe: str,
                                pattern_name: str, direction: str,
                                confidence: float, retest_level: float,
                                retest_confirmed: bool = False) -> FormattedSignal:
        """
        Format a WAIT_FOR_RETEST signal when breakout occurred
        """
        standardized_name = self._get_standardized_name(pattern_name)
        timestamp = datetime.now().isoformat(timespec='seconds')
        timestamp_short = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_id = f"pa_{timestamp_short}_retest"
        decision_id = f"dec_pa_{timestamp_short}_retest"
        
        action_detail = f"Breakout confirmed - waiting for retest at {retest_level:.6f}"
        
        return FormattedSignal(
            pattern_id=pattern_id,
            pattern_name=standardized_name,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            action=SignalAction.WAIT_FOR_RETEST,
            action_detail=action_detail,
            entry=None,
            stop_loss=None,
            take_profit=None,
            risk_reward=None,
            confidence=confidence,
            grade="B-" if confidence >= 0.65 else "C+",
            position_multiplier=0.8,
            decision_reason=f"{standardized_name} - Breakout confirmed - waiting for retest at {retest_level:.6f}",
            retest_level=retest_level,
            retest_confirmed=retest_confirmed,
            stage=SignalStage.BREAKOUT,
            age_bars=1,
            timestamp=timestamp,
            decision_id=decision_id
        )
    
    def format_flip(self, symbol: str, timeframe: str,
                    pattern_name: str, direction: str,
                    confidence: float, grade: str,
                    trade_setup: Dict,
                    trap_type: str, trap_severity: str,
                    retest_level: float = None) -> FormattedSignal:
        """
        Format a FLIP signal when trap is detected
        """
        standardized_name = self._get_standardized_name(pattern_name)
        timestamp = datetime.now().isoformat(timespec='seconds')
        timestamp_short = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_id = f"pa_{timestamp_short}_flip"
        decision_id = f"dec_pa_{timestamp_short}_flip"
        
        action = SignalAction.FLIP_TO_BUY if direction == 'BUY' else SignalAction.FLIP_TO_SELL
        action_detail = f"{trap_type.replace('_', ' ').title()} detected - flipping direction"
        
        return FormattedSignal(
            pattern_id=pattern_id,
            pattern_name=standardized_name,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            action=action,
            action_detail=action_detail,
            entry=trade_setup.get('entry'),
            stop_loss=trade_setup.get('stop_loss'),
            take_profit=trade_setup.get('take_profit'),
            risk_reward=trade_setup.get('risk_reward'),
            confidence=confidence,
            grade=grade,
            position_multiplier=1.2 if trap_severity in ['strong', 'extreme'] else 1.0,
            decision_reason=f"{trap_type.replace('_', ' ').title()} detected: {standardized_name} - flipping direction",
            retest_level=retest_level,
            retest_confirmed=True,
            stage=SignalStage.CONFIRMED,
            age_bars=0,
            timestamp=timestamp,
            decision_id=decision_id
        )


# ==================== CONVENIENCE FUNCTIONS ====================

def format_signal(symbol: str, timeframe: str,
                  pattern_name: str, direction: str,
                  confidence: float, grade: str,
                  position_multiplier: float,
                  trade_setup: Optional[Dict] = None,
                  **kwargs) -> FormattedSignal:
    """
    Convenience function to format a signal
    """
    formatter = SignalFormatter()
    return formatter.format_signal(
        symbol=symbol,
        timeframe=timeframe,
        pattern_name=pattern_name,
        direction=direction,
        confidence=confidence,
        grade=grade,
        position_multiplier=position_multiplier,
        trade_setup=trade_setup,
        **kwargs
    )


def format_skip(symbol: str, timeframe: str, reason: str) -> FormattedSignal:
    """
    Convenience function to format a SKIP signal
    """
    formatter = SignalFormatter()
    return formatter.format_skip(symbol, timeframe, reason)


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    formatter = SignalFormatter()
    
    print("=" * 70)
    print("SIGNAL FORMATTER EXAMPLES")
    print("=" * 70)
    
    # Example 1: Strong BUY signal
    print("\n1. STRONG BUY SIGNAL")
    print("-" * 50)
    
    strong_setup = {
        'entry': 51200.00,
        'stop_loss': 50800.00,
        'take_profit': 52000.00,
        'risk_reward': 2.5
    }
    
    signal = formatter.format_signal(
        symbol="BTC/USDT",
        timeframe="5m",
        pattern_name="bullish_engulfing",
        direction="BUY",
        confidence=0.87,
        grade="A",
        position_multiplier=1.2,
        trade_setup=strong_setup,
        mtf_alignment=0.85,
        at_key_level=True,
        sequence_type="reversal",
        additional_reasons=["At 4h support", "High volume confirmation"]
    )
    
    import json
    print(json.dumps(signal.to_dict(), indent=2))
    
    # Example 2: Trap detection (FLIP)
    print("\n2. TRAP DETECTION - FLIP TO BUY")
    print("-" * 50)
    
    trap_setup = {
        'entry': 50800.00,
        'stop_loss': 50400.00,
        'take_profit': 51600.00,
        'risk_reward': 2.67
    }
    
    signal = formatter.format_flip(
        symbol="BTC/USDT",
        timeframe="15m",
        pattern_name="bear_trap",
        direction="BUY",
        confidence=0.83,
        grade="B+",
        trade_setup=trap_setup,
        trap_type="bear_trap",
        trap_severity="strong",
        retest_level=50600.00
    )
    
    print(json.dumps(signal.to_dict(), indent=2))
    
    # Example 3: WAIT signal
    print("\n3. WAIT SIGNAL")
    print("-" * 50)
    
    signal = formatter.format_wait(
        symbol="ETH/USDT",
        timeframe="1h",
        pattern_name="hammer",
        direction="BUY",
        confidence=0.65,
        pattern_progress=0.85,
        retest_level=1850.00
    )
    
    print(json.dumps(signal.to_dict(), indent=2))
    
    # Example 4: SKIP signal
    print("\n4. SKIP SIGNAL")
    print("-" * 50)
    
    signal = formatter.format_skip(
        symbol="XRP/USDT",
        timeframe="5m",
        reason="Market choppy - no clear pattern"
    )
    
    print(json.dumps(signal.to_dict(), indent=2))
    
    print("=" * 70)