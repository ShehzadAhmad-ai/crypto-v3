"""
adapter_pattern_v3.py - Pattern V4 to ExpertSignal Adapter

Converts Pattern V4 output to unified ExpertSignal format for the expert aggregator.
Maintains backward compatibility with existing trading system.

Version: 4.0
Author: Pattern Intelligence System
"""

from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ExpertSignal:
    """
    Unified expert signal format for all 5 experts.
    Used by expert_aggregator.py and expert_consensus.py
    """
    direction: str                      # "BUY" or "SELL"
    confidence: float                   # 0-1, calibrated confidence
    entry: float                        # Entry price
    stop_loss: float                    # Stop loss price
    take_profit: float                  # Take profit price
    grade: str                          # "A+", "A", "B+", "B", "B-", "C+", "C", "D", "F"
    reasons: List[str]                  # List of reasons for the signal
    expert_name: str                    # "pattern_v4"
    timestamp: datetime                 # Signal timestamp
    entry_zone_low: float = None        # Optional entry zone (for limit orders)
    entry_zone_high: float = None       # Optional entry zone (for limit orders)
    tp_levels: List[dict] = None        # Optional multiple take profit levels
    pattern_id: str = None              # Unique pattern identifier
    pattern_name: str = None            # Name of the pattern detected
    completion_pct: float = None        # Pattern completion percentage (0-1)
    stage: str = None                   # Pattern stage (FORMING, BREAKOUT, RETEST, CONFIRMED)


def pattern_v4_to_expert_signal(decision: Dict[str, Any], symbol: str) -> ExpertSignal:
    """
    Convert Pattern V4 output to unified ExpertSignal.
    
    Args:
        decision: Pattern V4 decision dictionary from PatternFactoryV4
        symbol: Trading pair symbol (e.g., "BTC/USDT")
    
    Returns:
        ExpertSignal object ready for expert_aggregator
    """
    
    # Extract core fields with fallbacks
    direction = decision.get('direction', 'NEUTRAL')
    confidence = decision.get('confidence', decision.get('final_confidence', 0.5))
    entry = decision.get('entry', 0)
    stop_loss = decision.get('stop_loss', 0)
    take_profit = decision.get('take_profit', 0)
    grade = decision.get('grade', 'F')
    
    # Build reasons list
    reasons = decision.get('reasons', [])
    if not reasons and decision.get('decision_reason'):
        reasons = [decision.get('decision_reason')]
    
    # Calculate entry zone (1% around entry)
    entry_zone_low = entry * 0.99 if entry else None
    entry_zone_high = entry * 1.01 if entry else None
    
    # Build take profit levels (optional, for scaling out)
    tp_levels = None
    if decision.get('tp_levels'):
        tp_levels = decision.get('tp_levels')
    else:
        # Create default single TP level
        tp_levels = [{
            'price': take_profit,
            'percentage': 100,
            'description': 'Primary Target'
        }] if take_profit else None
    
    return ExpertSignal(
        direction=direction,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        grade=grade,
        reasons=reasons,
        expert_name="pattern_v4",
        timestamp=datetime.now(),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        tp_levels=tp_levels,
        pattern_id=decision.get('pattern_id'),
        pattern_name=decision.get('pattern_name'),
        completion_pct=decision.get('completion_pct', 1.0),
        stage=decision.get('stage', 'CONFIRMED')
    )


def pattern_v4_batch_to_expert_signals(decisions: List[Dict[str, Any]], 
                                        symbol: str) -> List[ExpertSignal]:
    """
    Convert multiple Pattern V4 decisions to ExpertSignals.
    
    Args:
        decisions: List of Pattern V4 decision dictionaries
        symbol: Trading pair symbol
    
    Returns:
        List of ExpertSignal objects
    """
    signals = []
    for decision in decisions:
        try:
            signal = pattern_v4_to_expert_signal(decision, symbol)
            signals.append(signal)
        except Exception as e:
            continue
    return signals


# Legacy function name for backward compatibility
pattern_v3_to_expert_signal = pattern_v4_to_expert_signal


# ============================================================================
# HELPER FUNCTIONS FOR SIGNAL VALIDATION
# ============================================================================

def validate_expert_signal(signal: ExpertSignal) -> Tuple[bool, str]:
    """
    Validate an ExpertSignal before sending to aggregator.
    
    Returns:
        (is_valid, reason)
    """
    # Check direction
    if signal.direction not in ['BUY', 'SELL']:
        return False, f"Invalid direction: {signal.direction}"
    
    # Check confidence range
    if not (0 <= signal.confidence <= 1):
        return False, f"Confidence out of range: {signal.confidence}"
    
    # Check prices are positive
    if signal.entry <= 0:
        return False, f"Invalid entry price: {signal.entry}"
    
    if signal.stop_loss <= 0:
        return False, f"Invalid stop loss: {signal.stop_loss}"
    
    if signal.take_profit <= 0:
        return False, f"Invalid take profit: {signal.take_profit}"
    
    # Check stop loss is on correct side
    if signal.direction == 'BUY' and signal.stop_loss >= signal.entry:
        return False, f"Stop loss above entry for BUY: {signal.stop_loss} >= {signal.entry}"
    
    if signal.direction == 'SELL' and signal.stop_loss <= signal.entry:
        return False, f"Stop loss below entry for SELL: {signal.stop_loss} <= {signal.entry}"
    
    # Check take profit is on correct side
    if signal.direction == 'BUY' and signal.take_profit <= signal.entry:
        return False, f"Take profit below entry for BUY: {signal.take_profit} <= {signal.entry}"
    
    if signal.direction == 'SELL' and signal.take_profit >= signal.entry:
        return False, f"Take profit above entry for SELL: {signal.take_profit} >= {signal.entry}"
    
    return True, "Valid"


def get_signal_summary(signal: ExpertSignal) -> Dict[str, Any]:
    """
    Get a summary dictionary of the signal for logging.
    """
    return {
        'expert': signal.expert_name,
        'direction': signal.direction,
        'confidence': round(signal.confidence, 3),
        'grade': signal.grade,
        'entry': round(signal.entry, 4),
        'stop_loss': round(signal.stop_loss, 4),
        'take_profit': round(signal.take_profit, 4),
        'risk_reward': round(
            abs(signal.take_profit - signal.entry) / abs(signal.stop_loss - signal.entry)
            if signal.stop_loss != signal.entry else 0, 2
        ),
        'pattern_name': signal.pattern_name,
        'completion_pct': signal.completion_pct,
        'stage': signal.stage,
        'reasons_count': len(signal.reasons),
        'timestamp': signal.timestamp.isoformat(),
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ExpertSignal',
    'pattern_v4_to_expert_signal',
    'pattern_v3_to_expert_signal',  # Legacy alias
    'pattern_v4_batch_to_expert_signals',
    'validate_expert_signal',
    'get_signal_summary',
]