"""
expert_consensus.py - Consensus Helper Functions

Simple utility functions for checking expert agreement.
Main consensus logic is in expert_aggregator.py.

Version: 2.0
"""

from typing import List, Tuple, Dict
from expert_interface import ExpertSignal, is_tradeable_signal


def get_direction_counts(signals: List[ExpertSignal]) -> Dict[str, int]:
    """
    Get count of BUY, SELL, NEUTRAL, HOLD signals
    
    Args:
        signals: List of expert signals
    
    Returns:
        Dictionary with direction counts
    """
    return {
        'BUY': sum(1 for s in signals if s.direction == 'BUY'),
        'SELL': sum(1 for s in signals if s.direction == 'SELL'),
        'NEUTRAL': sum(1 for s in signals if s.direction == 'NEUTRAL'),
        'HOLD': sum(1 for s in signals if s.direction == 'HOLD')
    }


def get_majority_direction(signals: List[ExpertSignal]) -> Tuple[str, int]:
    """
    Get the majority direction and count
    
    Args:
        signals: List of expert signals
    
    Returns:
        Tuple of (direction, count) - direction is 'BUY', 'SELL', or 'NEUTRAL'
    """
    counts = get_direction_counts(signals)
    
    if counts['BUY'] > counts['SELL']:
        return ('BUY', counts['BUY'])
    elif counts['SELL'] > counts['BUY']:
        return ('SELL', counts['SELL'])
    else:
        return ('NEUTRAL', max(counts['BUY'], counts['SELL']))


def get_agreement_ratio(signals: List[ExpertSignal]) -> float:
    """
    Calculate agreement ratio (percentage of experts agreeing on majority direction)
    
    Args:
        signals: List of expert signals
    
    Returns:
        Agreement ratio (0-1)
    """
    tradeable = [s for s in signals if is_tradeable_signal(s)]
    
    if not tradeable:
        return 0.0
    
    direction, count = get_majority_direction(tradeable)
    return count / len(tradeable)


def get_agreeing_experts(signals: List[ExpertSignal], direction: str) -> List[str]:
    """
    Get list of experts that agree with a given direction
    
    Args:
        signals: List of expert signals
        direction: 'BUY' or 'SELL'
    
    Returns:
        List of expert names
    """
    return [s.expert_name for s in signals if s.direction == direction and is_tradeable_signal(s)]


def get_opposing_experts(signals: List[ExpertSignal], direction: str) -> List[str]:
    """
    Get list of experts that oppose a given direction
    
    Args:
        signals: List of expert signals
        direction: 'BUY' or 'SELL'
    
    Returns:
        List of expert names
    """
    opposite = 'SELL' if direction == 'BUY' else 'BUY'
    return [s.expert_name for s in signals if s.direction == opposite and is_tradeable_signal(s)]


def has_sufficient_agreement(signals: List[ExpertSignal], min_agreement: int = 3) -> bool:
    """
    Check if there is sufficient agreement among experts
    
    Args:
        signals: List of expert signals
        min_agreement: Minimum number of experts that must agree
    
    Returns:
        True if sufficient agreement
    """
    direction, count = get_majority_direction(signals)
    return count >= min_agreement


def has_minimum_signals(signals: List[ExpertSignal], min_signals: int = 3) -> bool:
    """
    Check if minimum number of experts generated signals
    
    Args:
        signals: List of expert signals
        min_signals: Minimum number of tradeable signals required
    
    Returns:
        True if sufficient signals
    """
    tradeable_count = sum(1 for s in signals if is_tradeable_signal(s))
    return tradeable_count >= min_signals


def get_consensus_summary(signals: List[ExpertSignal]) -> Dict:
    """
    Get a human-readable consensus summary
    
    Args:
        signals: List of expert signals
    
    Returns:
        Dictionary with consensus summary
    """
    counts = get_direction_counts(signals)
    direction, count = get_majority_direction(signals)
    agreement_ratio = get_agreement_ratio(signals)
    
    return {
        'total_experts': len(signals),
        'tradeable_signals': sum(1 for s in signals if is_tradeable_signal(s)),
        'direction_counts': counts,
        'majority_direction': direction,
        'majority_count': count,
        'agreement_ratio': round(agreement_ratio, 3),
        'has_consensus': has_sufficient_agreement(signals),
        'agreeing_experts': get_agreeing_experts(signals, direction) if direction != 'NEUTRAL' else [],
        'opposing_experts': get_opposing_experts(signals, direction) if direction != 'NEUTRAL' else []
    }