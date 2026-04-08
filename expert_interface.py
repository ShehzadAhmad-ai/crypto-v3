"""
expert_interface.py - Common Data Classes for All 5 Experts

All experts (Pattern, Price Action, SMC, Technical, Strategy) 
must return ExpertSignal objects. This file defines the common format.

Version: 2.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class ExpertName(Enum):
    """Names of all 5 experts"""
    PATTERN_V3 = "pattern_v3"
    PRICE_ACTION = "price_action"
    SMC = "smc"
    TECHNICAL = "technical"
    STRATEGY = "strategy"


@dataclass
class TPLevel:
    """Individual Take Profit level with partial exit percentage"""
    price: float                        # Price level for this TP
    percentage: float                   # Percentage of position to exit (0-1)
    description: str = ""               # e.g., "Conservative target", "Primary target"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': round(self.price, 6),
            'percentage': round(self.percentage, 3),
            'description': self.description
        }


@dataclass
class ExpertSignal:
    """
    Standardized output format for ALL 5 experts
    Each expert must return this exact structure
    """
    # Core identification
    expert_name: str                    # 'pattern_v3', 'price_action', 'smc', 'technical', 'strategy'
    direction: str                      # 'BUY', 'SELL', 'NEUTRAL', 'HOLD'
    confidence: float                   # 0-1, confidence in this signal
    
    # Trade setup (0 if not available)
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 0.0
    
    # Quality metrics
    grade: str = "F"                    # 'A+', 'A', 'B+', 'B', 'B-', 'C+', 'C', 'D', 'F'
    reasons: List[str] = field(default_factory=list)
    
    # NEW FIELDS for Phase 4
    is_direction_only: bool = False     # True if expert couldn't give full signal (only direction)
    direction_confidence: float = 0.0   # Confidence in direction (for direction-only signals)
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    tp_levels: List[TPLevel] = field(default_factory=list)
    
    # Weight (will be set by weight manager)
    weight: float = 1.0
    
    # Expert-specific metadata (for debugging)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_valid(self) -> bool:
        """Validate that the signal has required fields"""
        if self.expert_name not in ['pattern_v3', 'price_action', 'smc', 'technical', 'strategy']:
            return False
        if self.direction not in ['BUY', 'SELL', 'NEUTRAL', 'HOLD']:
            return False
        if not 0 <= self.confidence <= 1:
            return False
        
        # For direction-only signals, entry/SL/TP can be 0
        if self.is_direction_only:
            return True
        
        # For full signals, validate trade setup
        if self.direction in ['BUY', 'SELL']:
            if self.entry <= 0:
                return False
            if self.stop_loss <= 0:
                return False
            if self.take_profit <= 0:
                return False
            
            # Validate SL/TP are on correct side
            if self.direction == 'BUY':
                if not (self.stop_loss < self.entry < self.take_profit):
                    return False
            elif self.direction == 'SELL':
                if not (self.take_profit < self.entry < self.stop_loss):
                    return False
        
        return True
    
    def is_tradeable(self) -> bool:
        """Check if signal is tradeable (not HOLD/NEUTRAL)"""
        return self.direction in ['BUY', 'SELL']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'expert_name': self.expert_name,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'is_direction_only': self.is_direction_only,
            'grade': self.grade,
            'reasons': self.reasons[:3],
            'weight': round(self.weight, 3),
            'timestamp': self.timestamp.isoformat()
        }
        
        # Add trade setup if available
        if self.entry > 0:
            result['entry'] = round(self.entry, 6)
        if self.stop_loss > 0:
            result['stop_loss'] = round(self.stop_loss, 6)
        if self.take_profit > 0:
            result['take_profit'] = round(self.take_profit, 6)
        if self.risk_reward > 0:
            result['risk_reward'] = round(self.risk_reward, 2)
        if self.entry_zone_low:
            result['entry_zone_low'] = round(self.entry_zone_low, 6)
        if self.entry_zone_high:
            result['entry_zone_high'] = round(self.entry_zone_high, 6)
        if self.tp_levels:
            result['tp_levels'] = [tp.to_dict() for tp in self.tp_levels]
        
        return result


@dataclass
class CombinedSignal:
    """
    Final aggregated signal from all 5 experts
    This is what Phase 4 outputs to Phase 5 (MTF Confirmation)
    """
    # Core direction (must be BUY or SELL, never HOLD)
    direction: str                      # 'BUY' or 'SELL'
    
    # Trade setup
    entry: float                        # Weighted average entry
    stop_loss: float                    # Weighted average stop loss
    
    # Multiple take profit levels (3-5 levels)
    tp_levels: List[TPLevel]            # List of TPLevel objects
    
    # Confidence and quality
    confidence: float                   # Weighted average confidence (0-1)
    grade: str = "C"                    # Overall grade
    
    # Consensus stats
    total_experts: int = 5
    agreeing_experts: int = 0
    opposing_experts: int = 0
    neutral_experts: int = 0
    agreement_ratio: float = 0.0
    consensus_reached: bool = False
    
    # Expert details (for debugging/output)
    expert_details: Dict[str, Dict] = field(default_factory=dict)
    # Format: {
    #   'pattern_v3': {'direction': 'BUY', 'confidence': 0.85, 'weight': 1.25, 'agreed': True},
    #   ...
    # }
    
    # For convenience (extracted from tp_levels)
    tp_prices: List[float] = field(default_factory=list)

    weighted_confidence: float = 0.0  # MTF-adjusted confidence
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

   # Add these NEW FIELDS to your existing CombinedSignal dataclass:

    # ===== IDENTIFICATION (for later phases) =====
    symbol: str = ""
    timeframe: str = ""
    
    # ===== Phase 5: MTF Confirmation =====
    mtf_score: float = 0.0
    mtf_aligned: bool = False
    htf_alignment: float = 0.0
    mtf_alignment_quality: str = "NEUTRAL"  # STRONG, MODERATE, WEAK, CONFLICT
    
    # ===== Phase 6: Smart Money =====
    smart_money_score: float = 0.0
    smart_money_bias: float = 0.0           # -1 to 1
    smart_money_confidence: float = 0.0
    
    # ===== Phase 7: Light Confirmations =====
    light_confirm_score: float = 0.0
    cross_asset_score: float = 0.0
    funding_oi_score: float = 0.0
    sentiment_score: float = 0.0
    
    # ===== Phase 8: Risk Management =====
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    position_size: float = 0.0              # Units
    position_value: float = 0.0             # USD
    risk_amount: float = 0.0                # USD
    risk_percent: float = 0.0               # Percentage of portfolio
    
    # ===== Phase 9: Final Scoring =====
    probability: float = 0.0
    final_score: float = 0.0
    confidence_level: float = 0.0
    edge_persistence: float = 0.0
    
    # ===== Phase 10: Validation =====
    status: str = "PENDING"                 # RAW, MTF_PASSED, SMART_MONEY_PASSED, etc.
    warning_flags: List[str] = field(default_factory=list)
    confirmation_reasons: List[str] = field(default_factory=list)
    validation_id: str = ""
    
    # ===== Phase 11: Timing =====
    expected_minutes_to_entry: int = 0
    expected_candles_to_entry: int = 0
    expected_minutes_to_tp: int = 0
    
    # ===== Story (for output) =====
    story: str = ""
    story_summary: str = ""
    key_points: List[str] = field(default_factory=list)
    
    # ===== Market Context =====
    market_regime: str = "UNKNOWN"
    volatility_state: str = "NORMAL"
    
    # ===== Additional Metadata =====
    metadata: Dict[str, Any] = field(default_factory=dict)



    
    def __post_init__(self):
        """Extract TP prices from tp_levels if not set"""
        if not self.tp_prices and self.tp_levels:
            self.tp_prices = [tp.price for tp in self.tp_levels]
    
    # Add these methods after __post_init__

    def to_signal_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for signal storage (Phase 12)"""
        return {
            'signal_id': self.metadata.get('signal_id', ''),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'timestamp': self.timestamp.isoformat(),
            'entry_zone_low': self.entry_zone_low,
            'entry_zone_high': self.entry_zone_high,
            'stop_loss': self.stop_loss,
            'tp_levels': [tp.to_dict() for tp in self.tp_levels],
            'risk_reward': self.risk_reward,
            'probability': self.probability,
            'grade': self.grade,
            'status': self.status,
            'story': self.story,
            'story_summary': self.story_summary,
            'key_points': self.key_points[:5],
            'warning_flags': self.warning_flags[:3],
            'expert_details': self.expert_details,
            'mtf_score': self.mtf_score,
            'smart_money_score': self.smart_money_score,
            'position_size': self.position_size,
            'position_value': self.position_value,
            'expected_minutes_to_entry': self.expected_minutes_to_entry
        }
    
    def update_status(self, new_status: str, reason: str = ""):
        """Update signal status and add to confirmation reasons"""
        self.status = new_status
        if reason:
            self.confirmation_reasons.append(reason)
    
    def add_warning(self, warning: str):
        """Add a warning flag"""
        if warning not in self.warning_flags:
            self.warning_flags.append(warning)






    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio (using first TP as reference)"""
        if self.direction == 'BUY':
            risk = self.entry - self.stop_loss
            reward = self.tp_prices[0] - self.entry if self.tp_prices else 0
        else:
            risk = self.stop_loss - self.entry
            reward = self.entry - self.tp_prices[0] if self.tp_prices else 0
        
        return reward / risk if risk > 0 else 0
    
    def is_valid(self) -> bool:
        """Validate combined signal"""
        if self.direction not in ['BUY', 'SELL']:
            return False
        if self.entry <= 0:
            return False
        if self.stop_loss <= 0:
            return False
        if not self.tp_levels:
            return False
        if not self.consensus_reached:
            return False
        
        # Validate direction
        if self.direction == 'BUY':
            if not (self.stop_loss < self.entry < self.tp_levels[0].price):
                return False
        else:
            if not (self.tp_levels[0].price < self.entry < self.stop_loss):
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output"""
        return {
            'direction': self.direction,
            'entry': round(self.entry, 6),
            'stop_loss': round(self.stop_loss, 6),
            'tp_levels': [tp.to_dict() for tp in self.tp_levels],
            'confidence': round(self.confidence, 3),
            'grade': self.grade,
            'risk_reward': round(self.risk_reward, 2),
            'consensus': {
                'total_experts': self.total_experts,
                'agreeing_experts': self.agreeing_experts,
                'opposing_experts': self.opposing_experts,
                'neutral_experts': self.neutral_experts,
                'agreement_ratio': round(self.agreement_ratio, 3),
                'reached': self.consensus_reached
            },
            'expert_details': self.expert_details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ConsensusResult:
    """Result of consensus check among experts"""
    reached: bool                       # True if consensus reached
    direction: str                      # 'BUY' or 'SELL' if reached
    agreeing_experts: List[str]         # Names of experts that agreed
    opposing_experts: List[str]         # Names that opposed
    neutral_experts: List[str]          # Names that were NEUTRAL/HOLD
    agreement_ratio: float              # agreeing_experts / total_experts
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'reached': self.reached,
            'direction': self.direction,
            'agreeing_experts': self.agreeing_experts,
            'opposing_experts': self.opposing_experts,
            'neutral_experts': self.neutral_experts,
            'agreement_ratio': round(self.agreement_ratio, 3),
            'details': self.details
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_skip_signal(expert_name: str, reason: str) -> ExpertSignal:
    """Create a SKIP/HOLD signal for when an expert has no trade"""
    return ExpertSignal(
        expert_name=expert_name,
        direction="HOLD",
        confidence=0.0,
        grade="F",
        reasons=[reason],
        is_direction_only=False
    )


def create_direction_only_signal(expert_name: str, direction: str, 
                                  confidence: float, reason: str) -> ExpertSignal:
    """Create a direction-only signal (when expert can't give full trade setup)"""
    return ExpertSignal(
        expert_name=expert_name,
        direction=direction,
        confidence=confidence,
        direction_confidence=confidence,
        grade="D",
        reasons=[reason],
        is_direction_only=True
    )


def is_tradeable_signal(signal: ExpertSignal) -> bool:
    """Check if an expert signal is tradeable (BUY or SELL)"""
    return signal.direction in ['BUY', 'SELL']


def is_full_signal(signal: ExpertSignal) -> bool:
    """Check if signal has full trade setup (not direction-only)"""
    return not signal.is_direction_only and signal.entry > 0 and signal.stop_loss > 0