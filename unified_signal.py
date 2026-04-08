"""
unified_signal.py - Single Signal Class for ALL Phases

This one class works with:
- Phase 4 (Expert Aggregator)
- Phase 5 (MTF Pipeline)
- Phase 6 (Smart Money Pipeline)
- Phase 7 (Light Confirm Pipeline)
- Phase 8 (Risk Pipeline)
- Phase 9 (Final Scoring)
- Phase 10 (Signal Validator)
- Phase 11 (Timing Predictor)
- Phase 12 (Signal Exporter)
- Phase 13 (Performance Tracker)

NO CONVERSION NEEDED!
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class SignalStatus(str, Enum):
    """Signal status across all phases"""
    PENDING = "PENDING"
    MTF_PASSED = "MTF_PASSED"
    MTF_REJECTED = "MTF_REJECTED"
    SMART_MONEY_PASSED = "SMART_MONEY_PASSED"
    SMART_MONEY_REJECTED = "SMART_MONEY_REJECTED"
    LIGHT_CONFIRM_PASSED = "LIGHT_CONFIRM_PASSED"
    RISK_PASSED = "RISK_PASSED"
    FINAL = "FINAL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class TPLevel:
    """Take Profit Level with partial exit"""
    price: float
    percentage: float = 0.25
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'price': round(self.price, 6),
            'percentage': round(self.percentage, 3),
            'description': self.description
        }


@dataclass
class UnifiedSignal:
    """
    ONE SIGNAL CLASS TO RULE THEM ALL
    
    Contains EVERY field from:
    - Signal (legacy)
    - CombinedSignal (Phase 4)
    
    Works with ALL pipelines without conversion!
    """
    
    # =========================================================================
    # CORE IDENTIFICATION
    # =========================================================================
    symbol: str = ""
    direction: str = ""  # BUY, SELL, HOLD, NEUTRAL
    timeframe: str = "5m"
    signal_id: str = ""
    
    # =========================================================================
    # CONFIDENCE & QUALITY
    # =========================================================================
    confidence: float = 0.0
    weighted_confidence: float = 0.0  # MTF adjusted
    grade: str = "C"  # A+, A, B+, B, B-, C+, C, D, F
    probability: float = 0.0  # Final probability
    
    # =========================================================================
    # TRADE SETUP
    # =========================================================================
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0  # Primary TP (legacy)
    tp_levels: List[TPLevel] = field(default_factory=list)  # Multiple TPs
    
    # Entry zone (for limit orders)
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    
    # =========================================================================
    # CONSENSUS (from Phase 4)
    # =========================================================================
    consensus_reached: bool = False
    total_experts: int = 5
    agreeing_experts: int = 0
    opposing_experts: int = 0
    neutral_experts: int = 0
    agreement_ratio: float = 0.0
    expert_details: Dict[str, Dict] = field(default_factory=dict)
    # Format: {'pattern_v3': {'direction': 'BUY', 'confidence': 0.85, 'weight': 1.25, 'agreed': True}}
    
    # =========================================================================
    # PHASE 5: MTF CONFIRMATION
    # =========================================================================
    mtf_score: float = 0.0
    mtf_aligned: bool = False
    mtf_alignment_quality: str = "NEUTRAL"  # STRONG, MODERATE, WEAK, CONFLICT
    htf_alignment: float = 0.0
    confirming_timeframes: List[str] = field(default_factory=list)
    conflicting_timeframes: List[str] = field(default_factory=list)
    
    # =========================================================================
    # PHASE 6: SMART MONEY
    # =========================================================================
    smart_money_score: float = 0.0
    smart_money_bias: float = 0.0  # -1 to 1
    orderflow_bias: str = "NEUTRAL"
    smart_money_confidence: float = 0.0
    
    # =========================================================================
    # PHASE 7: LIGHT CONFIRMATIONS
    # =========================================================================
    light_confirm_score: float = 0.0
    cross_asset_score: float = 0.0
    funding_oi_score: float = 0.0
    sentiment_score: float = 0.0
    
    # =========================================================================
    # PHASE 8: RISK MANAGEMENT
    # =========================================================================
    position_size: float = 0.0  # Units/coins
    position_value: float = 0.0  # USD value
    risk_amount: float = 0.0  # USD risk
    risk_percent: float = 0.0  # % of portfolio
    
    # =========================================================================
    # PHASE 9: FINAL SCORING
    # =========================================================================
    final_score: float = 0.0
    confidence_level: float = 0.0
    edge_persistence: float = 0.0
    
    # =========================================================================
    # PHASE 10: VALIDATION
    # =========================================================================
    status: str = "PENDING"
    warning_flags: List[str] = field(default_factory=list)
    confirmation_reasons: List[str] = field(default_factory=list)
    validation_id: str = ""
    
    # =========================================================================
    # PHASE 11: TIMING
    # =========================================================================
    expected_minutes_to_entry: int = 0
    expected_candles_to_entry: int = 0
    expected_minutes_to_tp: int = 0
    
    # =========================================================================
    # STORY & REASONS (for output)
    # =========================================================================
    story: str = ""
    story_summary: str = ""
    key_points: List[str] = field(default_factory=list)
    decision_reason: str = ""
    
    # =========================================================================
    # MARKET CONTEXT
    # =========================================================================
    market_regime: str = "UNKNOWN"
    volatility_state: str = "NORMAL"
    
    # =========================================================================
    # METADATA (for any extra data)
    # =========================================================================
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # =========================================================================
    # TIMESTAMP
    # =========================================================================
    timestamp: datetime = field(default_factory=datetime.now)
    
    # =========================================================================
    # PROPERTIES (calculated fields)
    # =========================================================================
    
    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio using first TP"""
        if self.direction == 'BUY':
            risk = self.entry - self.stop_loss
            first_tp = self.tp_levels[0].price if self.tp_levels else self.take_profit
            reward = first_tp - self.entry
        elif self.direction == 'SELL':
            risk = self.stop_loss - self.entry
            first_tp = self.tp_levels[0].price if self.tp_levels else self.take_profit
            reward = self.entry - first_tp
        else:
            return 0.0
        return reward / risk if risk > 0 else 0
    
    @property
    def tp_prices(self) -> List[float]:
        """Get list of all TP prices"""
        if self.tp_levels:
            return [tp.price for tp in self.tp_levels]
        return [self.take_profit] if self.take_profit > 0 else []
    
    @property
    def is_tradeable(self) -> bool:
        """Check if signal is tradeable (BUY or SELL)"""
        return self.direction in ['BUY', 'SELL']
    
    @property
    def is_buy(self) -> bool:
        return self.direction == 'BUY'
    
    @property
    def is_sell(self) -> bool:
        return self.direction == 'SELL'
    
    @property
    def is_hold(self) -> bool:
        return self.direction in ['HOLD', 'NEUTRAL']
    
    # =========================================================================
    # METHODS
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'timeframe': self.timeframe,
            'confidence': round(self.confidence, 3),
            'grade': self.grade,
            'entry': round(self.entry, 6),
            'stop_loss': round(self.stop_loss, 6),
            'tp_levels': [tp.to_dict() for tp in self.tp_levels],
            'risk_reward': round(self.risk_reward, 2),
            'status': self.status,
            'story': self.story,
            'story_summary': self.story_summary,
            'key_points': self.key_points[:5],
            'warning_flags': self.warning_flags[:3],
            'expert_details': self.expert_details,
            'mtf_score': round(self.mtf_score, 3),
            'smart_money_score': round(self.smart_money_score, 3),
            'position_size': round(self.position_size, 4),
            'position_value': round(self.position_value, 2),
            'expected_minutes_to_entry': self.expected_minutes_to_entry,
            'timestamp': self.timestamp.isoformat()
        }
    
    def update_status(self, new_status: str, reason: str = ""):
        """Update signal status"""
        self.status = new_status
        if reason:
            self.confirmation_reasons.append(reason)
    
    def add_warning(self, warning: str):
        """Add a warning flag"""
        if warning not in self.warning_flags:
            self.warning_flags.append(warning)
    
    def add_key_point(self, point: str):
        """Add a key point to story"""
        if point not in self.key_points:
            self.key_points.append(point)
    
    def to_short_string(self) -> str:
        """Short string representation for logging"""
        return f"{self.symbol} {self.direction} [{self.grade}] {self.confidence:.0%}"


# ============================================================================
# CONVERTER FUNCTIONS (for backward compatibility)
# ============================================================================

def from_legacy_signal(legacy_signal) -> UnifiedSignal:
    """
    Convert legacy Signal object to UnifiedSignal
    
    Use this once when integrating old code
    """
    signal = UnifiedSignal(
        symbol=getattr(legacy_signal, 'symbol', ''),
        direction=getattr(legacy_signal, 'direction', 'NEUTRAL'),
        confidence=getattr(legacy_signal, 'confidence', 0.0),
        grade=getattr(legacy_signal, 'grade', 'C'),
        entry=getattr(legacy_signal, 'entry', 0.0),
        stop_loss=getattr(legacy_signal, 'stop_loss', 0.0),
        take_profit=getattr(legacy_signal, 'take_profit', 0.0),
        timeframe=getattr(legacy_signal, 'timeframe', '5m'),
        status=getattr(legacy_signal, 'status', 'PENDING'),
        metadata=getattr(legacy_signal, 'metadata', {})
    )
    
    # Copy TP levels if present
    if hasattr(legacy_signal, 'take_profit_levels') and legacy_signal.take_profit_levels:
        for tp in legacy_signal.take_profit_levels:
            if isinstance(tp, dict):
                signal.tp_levels.append(TPLevel(
                    price=tp.get('price', 0),
                    percentage=tp.get('percentage', 0.25),
                    description=tp.get('description', '')
                ))
    
    return signal


def from_combined_signal(combined_signal) -> UnifiedSignal:
    """
    Convert CombinedSignal to UnifiedSignal
    
    Use this in Phase 4 to output UnifiedSignal
    """
    signal = UnifiedSignal(
        symbol=getattr(combined_signal, 'symbol', ''),
        direction=combined_signal.direction,
        confidence=combined_signal.confidence,
        grade=combined_signal.grade,
        entry=combined_signal.entry,
        stop_loss=combined_signal.stop_loss,
        timeframe=getattr(combined_signal, 'timeframe', '5m'),
        consensus_reached=combined_signal.consensus_reached,
        total_experts=combined_signal.total_experts,
        agreeing_experts=combined_signal.agreeing_experts,
        opposing_experts=combined_signal.opposing_experts,
        neutral_experts=combined_signal.neutral_experts,
        agreement_ratio=combined_signal.agreement_ratio,
        expert_details=combined_signal.expert_details,
        weighted_confidence=getattr(combined_signal, 'weighted_confidence', 0.0),
        metadata=getattr(combined_signal, 'metadata', {})
    )
    
    # Copy TP levels
    if hasattr(combined_signal, 'tp_levels') and combined_signal.tp_levels:
        signal.tp_levels = combined_signal.tp_levels.copy()
    
    return signal


def to_legacy_signal(unified: UnifiedSignal):
    """
    Convert UnifiedSignal to legacy Signal object
    
    Use this for old code that expects Signal
    """
    # Import here to avoid circular imports
    from signal_model import Signal, SignalStatus as LegacyStatus
    
    status_map = {
        'PENDING': LegacyStatus.PENDING,
        'MTF_PASSED': LegacyStatus.MTF_PASSED,
        'FINAL': LegacyStatus.FINAL,
        'REJECTED': LegacyStatus.REJECTED
    }
    
    signal = Signal(
        symbol=unified.symbol,
        direction=unified.direction,
        entry=unified.entry,
        stop_loss=unified.stop_loss,
        take_profit=unified.take_profit or (unified.tp_levels[0].price if unified.tp_levels else 0),
        confidence=unified.confidence,
        timeframe=unified.timeframe
    )
    
    signal.grade = unified.grade
    signal.consensus_reached = unified.consensus_reached
    signal.expert_details = unified.expert_details
    signal.take_profit_levels = [tp.to_dict() for tp in unified.tp_levels]
    signal.status = status_map.get(unified.status, LegacyStatus.PENDING)
    signal.metadata = unified.metadata
    
    return signal