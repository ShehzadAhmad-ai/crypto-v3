"""
adapter.py - Technical Analyzer to Unified ExpertSignal Adapter
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

from .ta_core import TASignal


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


def technical_to_expert_signal(signal: TASignal) -> ExpertSignal:
    """
    Convert Technical Analyzer TASignal to unified ExpertSignal format.
    
    Args:
        signal: TASignal from TechnicalAnalyzerFactory.analyze()
    
    Returns:
        ExpertSignal ready for aggregator
    """
    
    # Build reasons list from decision_reason and metadata
    reasons = [signal.decision_reason]
    
    # Add regime context if available
    if hasattr(signal, 'regime') and signal.regime:
        reasons.append(f"Regime: {signal.regime}")
    
    # Add HTF alignment if available
    if hasattr(signal, 'htf_aligned') and signal.htf_aligned:
        reasons.append("HTF aligned")
    
    # Add divergence count if available
    if hasattr(signal, 'divergence_count') and signal.divergence_count > 0:
        reasons.append(f"{signal.divergence_count} divergence(s)")
    
    # Calculate entry zone (1% buffer around entry)
    entry = signal.entry if signal.entry else 0
    entry_zone_low = entry * 0.99 if entry else None
    entry_zone_high = entry * 1.01 if entry else None
    
    return ExpertSignal(
        direction=signal.direction,
        confidence=signal.confidence,
        entry=entry,
        stop_loss=signal.stop_loss if signal.stop_loss else 0,
        take_profit=signal.take_profit if signal.take_profit else 0,
        grade=signal.grade,
        reasons=reasons[:5],  # Max 5 reasons
        expert_name="technical_analyzer",
        timestamp=datetime.now(),
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        tp_levels=None  # Single TP, can add multiple later
    )


def is_tradeable_signal(signal: TASignal) -> bool:
    """
    Quick check if a signal is tradeable.
    
    Args:
        signal: TASignal from TechnicalAnalyzerFactory.analyze()
    
    Returns:
        True if signal should be sent to aggregator
    """
    return signal.action in ["STRONG_ENTRY", "ENTER_NOW"]