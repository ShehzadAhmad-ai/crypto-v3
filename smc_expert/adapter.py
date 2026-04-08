"""
adapter.py - SMC Expert to Unified ExpertSignal Adapter
"""

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


def smc_to_expert_signal(signal: dict, symbol: str) -> ExpertSignal:
    """
    Convert SMC V3 output to unified ExpertSignal format.
    
    Args:
        signal: Dict from SMCFactory.analyze()
        symbol: Trading pair symbol
    
    Returns:
        ExpertSignal ready for aggregator
    """
    
    # Build reasons list
    reasons = [signal.get('decision_reason', 'SMC signal')]
    
    # Add context summary if available
    context = signal.get('context_summary', {})
    if context:
        reasons.append(f"Structure: {context.get('market_bias', 'neutral')}")
        reasons.append(f"Phase: {context.get('amd_phase', 'unknown')}")
    
    # Add narrative key points if available
    narrative = signal.get('narrative', {})
    if narrative.get('key_points'):
        reasons.extend(narrative['key_points'][:2])
    
    # Calculate entry zone (1% buffer around entry)
    entry = signal['entry']
    entry_zone_low = entry * 0.99
    entry_zone_high = entry * 1.01
    
    return ExpertSignal(
        direction=signal['direction'],
        confidence=signal['confidence'],
        entry=entry,
        stop_loss=signal['stop_loss'],
        take_profit=signal['take_profit'],
        grade=signal['grade'],
        reasons=reasons[:5],  # Max 5 reasons
        expert_name="smc",
        timestamp=datetime.now(),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        tp_levels=None  # Single TP, can add multiple later
    )


def is_tradeable_signal(signal: dict) -> bool:
    """
    Quick check if a signal is tradeable.
    
    Args:
        signal: Dict from SMCFactory.analyze()
    
    Returns:
        True if signal should be sent to aggregator
    """
    action = signal.get('action', 'SKIP')
    return action in ['STRONG_ENTRY', 'ENTER_NOW']