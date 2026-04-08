# adapter.py - FIXED VERSION

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ExpertSignal:
    """Unified expert signal format for aggregator"""
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    grade: str
    reasons: List[str]
    expert_name: str
    timestamp: datetime
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    tp_levels: Optional[List[dict]] = None


def price_action_to_expert_signal(signal, symbol: str) -> ExpertSignal:
    """
    Convert Price Action V3.5 FormattedSignal to unified ExpertSignal format.
    """
    # Helper to safely get numeric values
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    # Helper to safely get string
    def safe_str(value, default=""):
        if value is None:
            return default
        return str(value)
    
    # Build reasons list
    reasons = []
    
    # Add pattern name if available
    pattern_name = safe_str(getattr(signal, 'pattern_name', ''))
    if pattern_name:
        reasons.append(pattern_name)
    
    # Add decision reason
    decision_reason = safe_str(getattr(signal, 'decision_reason', ''))
    if decision_reason:
        reasons.append(decision_reason)
    
    # Add action detail
    action_detail = safe_str(getattr(signal, 'action_detail', ''))
    if action_detail and action_detail != decision_reason:
        reasons.append(action_detail)
    
    # Get entry
    entry = safe_float(getattr(signal, 'entry', None))
    if entry == 0:
        entry = safe_float(getattr(signal, 'entry_price', None))
    
    stop_loss = safe_float(getattr(signal, 'stop_loss', None))
    take_profit = safe_float(getattr(signal, 'take_profit', None))
    
    # Calculate entry zone (1% buffer around entry)
    entry_zone_low = entry * 0.99 if entry > 0 else None
    entry_zone_high = entry * 1.01 if entry > 0 else None
    
    return ExpertSignal(
        direction=safe_str(getattr(signal, 'direction', 'NEUTRAL')),
        confidence=safe_float(getattr(signal, 'confidence', 0.0)),
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        grade=safe_str(getattr(signal, 'grade', 'F')),
        reasons=reasons[:5],
        expert_name="price_action",
        timestamp=datetime.now(),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        tp_levels=None
    )


def is_tradeable_signal(signal) -> bool:
    """Quick check if a signal is tradeable"""
    action = getattr(signal, 'action', 'SKIP')
    return action in ["ENTER_NOW", "STRONG_ENTRY", "FLIP_TO_BUY", "FLIP_TO_SELL"]