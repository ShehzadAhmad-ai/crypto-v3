# signal_model.py - UPGRADED FOR VERSION 3
"""
Signal Model for Trading System V3
- Now includes expert consensus data from all 5 experts
- Supports multiple take profit levels (TP1, TP2, TP3, TP4, TP5)
- Tracks which experts agreed and their confidence
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class SignalStatus(Enum):
    """Overall signal status"""
    RAW = "raw"
    TECHNICAL_PASSED = "technical_passed"
    MTF_PASSED = "mtf_passed"
    SMART_MONEY_PASSED = "smart_money_passed"
    LIGHT_CONFIRM_PASSED = "light_confirm_passed"
    RISK_PASSED = "risk_passed"
    TIMING_PASSED = "timing_passed"
    FINAL = "final"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


class SignalStage(Enum):
    """Pipeline stage tracking"""
    RAW = "raw"              # Phase 2 only
    CONFIRMED = "confirmed"  # Phases 3-8 passed
    FINAL = "final"          # Phase 9 passed (validated)


class EntryType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"


# ============================================================================
# NEW: TAKE PROFIT LEVEL DATA CLASS
# ============================================================================

@dataclass
class TakeProfitLevel:
    """
    Individual take profit level with partial exit percentage
    Used for multi-target exit strategy
    """
    price: float                        # Price level for this TP
    percentage: float                   # Percentage of position to exit (0-1)
    description: str = ""               # e.g., "Conservative target", "Primary target"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': round(self.price, 6),
            'percentage': round(self.percentage, 3),
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TakeProfitLevel':
        return cls(
            price=data['price'],
            percentage=data['percentage'],
            description=data.get('description', '')
        )


# ============================================================================
# NEW: EXPERT SIGNAL DATA CLASS (for storing individual expert results)
# ============================================================================

@dataclass
class ExpertSignalData:
    """
    Stores the signal from a single expert for debugging/analysis
    """
    expert_name: str                    # 'pattern_v3', 'price_action', 'smc', 'technical', 'strategy'
    direction: str                      # 'BUY', 'SELL', 'HOLD'
    confidence: float                   # 0-1
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    grade: str                          # 'A', 'B', 'C', 'D', 'F'
    weight: float = 1.0                 # Current weight of this expert
    agreed: bool = True                 # Whether this expert agreed with consensus
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'expert_name': self.expert_name,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'entry': round(self.entry, 6),
            'stop_loss': round(self.stop_loss, 6),
            'take_profit': round(self.take_profit, 6),
            'risk_reward': round(self.risk_reward, 2),
            'grade': self.grade,
            'weight': round(self.weight, 3),
            'agreed': self.agreed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertSignalData':
        return cls(
            expert_name=data['expert_name'],
            direction=data['direction'],
            confidence=data['confidence'],
            entry=data['entry'],
            stop_loss=data['stop_loss'],
            take_profit=data['take_profit'],
            risk_reward=data['risk_reward'],
            grade=data['grade'],
            weight=data.get('weight', 1.0),
            agreed=data.get('agreed', True)
        )


# ============================================================================
# MAIN SIGNAL CLASS (UPGRADED FOR VERSION 3)
# ============================================================================

@dataclass
class Signal:
    # ===== CORE IDENTIFIERS =====
    symbol: str
    timeframe: str
    direction: str  # 'BUY' or 'SELL'
    timestamp: datetime = field(default_factory=datetime.now)
    
    # ===== PIPELINE STAGE TRACKING =====
    stage: SignalStage = SignalStage.RAW
    
    # ===== VERSION 3: EXPERT CONSENSUS DATA =====
    # Did all 5 experts agree?
    expert_consensus_reached: bool = False
    consensus_direction: str = "NEUTRAL"      # Direction all experts agreed on
    consensus_confidence: float = 0.0         # Weighted average confidence
    
    # Individual expert signals (for debugging/analysis)
    expert_signals: List[ExpertSignalData] = field(default_factory=list)
    expert_details: Dict[str, Dict] = field(default_factory=dict)  # Legacy format
    
    # ===== VERSION 3: MULTIPLE TAKE PROFIT LEVELS =====
    # Primary TP (legacy)
    take_profit: Optional[float] = None
    
    # Multiple TP levels (new)
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)
    tp_prices: List[float] = field(default_factory=list)  # Just prices (legacy compatibility)
    tp_percentages: List[float] = field(default_factory=list)  # Partial exit percentages
    
    # ===== STAGE 2 - TECHNICAL (DEPRECATED but kept for compatibility) =====
    technical_score: float = 0.0
    technical_reasons: List[str] = field(default_factory=list)
    strategy_confirmed: str = ""  # Name of strategy that generated this
    
    # ===== STAGE 3 - MTF =====
    mtf_score: float = 0.0
    htf_trend: str = "NEUTRAL"
    trend_alignment: str = "NEUTRAL"
    
    # ===== STAGE 4 - SMART MONEY =====
    smart_money_score: float = 0.0
    liquidity_sweep: Optional[Dict] = None
    orderflow_bias: str = "NEUTRAL"
    
    # ===== STAGE 5 - LIGHT CONFIRMATIONS =====
    cross_asset_score: float = 0.0
    funding_oi_score: float = 0.0
    sentiment_score: float = 0.0
    light_confirm_score: float = 0.0
    
    # ===== STAGE 6 - RISK (DYNAMIC) =====
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    entry_type: EntryType = EntryType.LIMIT
    stop_loss: Optional[float] = None
    risk_reward_ratio: float = 0.0
    position_size: float = 0.0      # Position size in units
    position_value: float = 0.0      # Position value in USD
    risk_amount: float = 0.0         # Risk amount in USD
    
    # ===== STAGE 7 - TIMING PREDICTOR =====
    expected_candles_to_entry: int = 0
    expected_minutes_to_entry: int = 0
    expected_minutes_to_tp: int = 0   # minutes
    
    # ===== STAGE 8 - FINAL SCORING =====
    probability: float = 0.0
    confidence_level: float = 0.0
    final_score: float = 0.0
    signal_grade: str = "N/A"
    edge_persistence: float = 0.0
    warning_flags: List[str] = field(default_factory=list)
    confirmation_reasons: List[str] = field(default_factory=list)
    
    # ===== STAGE 9 - VALIDATION =====
    status: SignalStatus = SignalStatus.RAW
    validation_errors: List[str] = field(default_factory=list)
    
    # ===== STAGE 11 - PERFORMANCE TRACKING =====
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ===== STORY FIELDS =====
    story: str = ""                    # Human-readable story
    story_summary: str = ""            # One-line summary
    key_points: List[str] = field(default_factory=list)  # Key bullet points
    
    # ===== METADATA =====
    volatility_state: str = "NORMAL"
    market_regime: str = "UNKNOWN"
    cooldown_until: Optional[datetime] = None
    
    # ===== HELPER PROPERTIES =====
    
    @property
    def has_consensus(self) -> bool:
        """Check if expert consensus was reached"""
        return self.expert_consensus_reached
    
    @property
    def agreeing_experts(self) -> List[str]:
        """Get list of experts that agreed with consensus"""
        return [e.expert_name for e in self.expert_signals if e.agreed]
    
    @property
    def disagreeing_experts(self) -> List[str]:
        """Get list of experts that disagreed with consensus"""
        return [e.expert_name for e in self.expert_signals if not e.agreed]
    
    @property
    def first_tp(self) -> Optional[float]:
        """Get first (closest) take profit level"""
        if self.take_profit_levels:
            return self.take_profit_levels[0].price
        return self.take_profit
    
    @property
    def last_tp(self) -> Optional[float]:
        """Get last (farthest) take profit level"""
        if self.take_profit_levels:
            return self.take_profit_levels[-1].price
        return self.take_profit
    
    @property
    def num_tp_levels(self) -> int:
        """Number of take profit levels"""
        return len(self.take_profit_levels)
    
    # ===== CONVENIENCE METHODS =====
    
    def add_expert_signal(self, expert_name: str, direction: str, 
                          confidence: float, entry: float, 
                          stop_loss: float, take_profit: float,
                          risk_reward: float, grade: str,
                          weight: float = 1.0, agreed: bool = True):
        """Add a single expert signal to the list"""
        self.expert_signals.append(ExpertSignalData(
            expert_name=expert_name,
            direction=direction,
            confidence=confidence,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            grade=grade,
            weight=weight,
            agreed=agreed
        ))
    
    def add_tp_level(self, price: float, percentage: float, description: str = ""):
        """Add a take profit level"""
        self.take_profit_levels.append(TakeProfitLevel(
            price=price,
            percentage=percentage,
            description=description
        ))
        # Also update tp_prices for legacy compatibility
        self.tp_prices.append(price)
        self.tp_percentages.append(percentage)
    
    def get_expert_summary(self) -> Dict[str, Any]:
        """Get summary of expert consensus"""
        return {
            'consensus_reached': self.expert_consensus_reached,
            'consensus_direction': self.consensus_direction,
            'consensus_confidence': round(self.consensus_confidence, 3),
            'agreeing_experts': self.agreeing_experts,
            'disagreeing_experts': self.disagreeing_experts,
            'total_experts': len(self.expert_signals),
            'agreement_ratio': len(self.agreeing_experts) / len(self.expert_signals) if self.expert_signals else 0
        }
    
    def get_tp_summary(self) -> Dict[str, Any]:
        """Get summary of take profit levels"""
        if not self.take_profit_levels:
            return {'has_multiple_tp': False, 'primary_tp': self.take_profit}
        
        return {
            'has_multiple_tp': True,
            'levels': [tp.to_dict() for tp in self.take_profit_levels],
            'num_levels': len(self.take_profit_levels),
            'primary_tp': self.take_profit_levels[0].price if self.take_profit_levels else None,
            'max_tp': self.take_profit_levels[-1].price if self.take_profit_levels else None
        }
    
    # ===== SERIALIZATION METHODS =====
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        for key, value in self.__dict__.items():
            # Skip large data when saving to JSON
            if key in ['df', 'indicators_df']:
                continue
            
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list):
                # Handle list of custom objects
                if key == 'expert_signals' and value:
                    result[key] = [v.to_dict() for v in value]
                elif key == 'take_profit_levels' and value:
                    result[key] = [v.to_dict() for v in value]
                else:
                    result[key] = [str(v) if isinstance(v, Enum) else v for v in value]
            elif isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create Signal from dictionary"""
        # Handle enum fields
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = SignalStatus(data['status'])
        if 'stage' in data and isinstance(data['stage'], str):
            data['stage'] = SignalStage(data['stage'])
        if 'entry_type' in data and isinstance(data['entry_type'], str):
            data['entry_type'] = EntryType(data['entry_type'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'cooldown_until' in data and isinstance(data['cooldown_until'], str):
            data['cooldown_until'] = datetime.fromisoformat(data['cooldown_until'])
        
        # Handle expert_signals list
        if 'expert_signals' in data and data['expert_signals']:
            data['expert_signals'] = [
                ExpertSignalData.from_dict(s) for s in data['expert_signals']
            ]
        
        # Handle take_profit_levels list
        if 'take_profit_levels' in data and data['take_profit_levels']:
            data['take_profit_levels'] = [
                TakeProfitLevel.from_dict(tp) for tp in data['take_profit_levels']
            ]
        
        # ===== HANDLE OLD FIELD NAMES =====
        # Map old field name to new field name
        if 'expected_time_to_tp' in data and 'expected_minutes_to_tp' not in data:
            data['expected_minutes_to_tp'] = data['expected_time_to_tp']
            del data['expected_time_to_tp']
        elif 'expected_time_to_tp' in data and 'expected_minutes_to_tp' in data:
            del data['expected_time_to_tp']
        
        # ===== REMOVE FIELDS THAT DON'T EXIST IN DATACLASS =====
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    # ===== FORMATTED OUTPUT METHODS =====
    
    def to_expert_output(self) -> Dict[str, Any]:
        """
        Generate formatted output for the final signal
        This matches the expected output format for the system
        """
        # Build TP levels list
        if self.take_profit_levels:
            tp_output = [tp.to_dict() for tp in self.take_profit_levels]
        elif self.take_profit:
            tp_output = [{'price': self.take_profit, 'percentage': 1.0, 'description': 'Primary target'}]
        else:
            tp_output = []
        
        return {
            # Core identifiers
            'signal_id': self.metadata.get('signal_id', f"{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"),
            'module': 'combined',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'timestamp': self.timestamp.isoformat(),
            
            # Expert consensus
            'consensus': {
                'reached': self.expert_consensus_reached,
                'direction': self.consensus_direction,
                'confidence': round(self.consensus_confidence, 3),
                'agreeing_experts': self.agreeing_experts,
                'disagreeing_experts': self.disagreeing_experts,
                'agreement_ratio': len(self.agreeing_experts) / len(self.expert_signals) if self.expert_signals else 0
            },
            
            # Expert details
            'expert_details': {
                e.expert_name: {
                    'direction': e.direction,
                    'confidence': round(e.confidence, 3),
                    'weight': round(e.weight, 3),
                    'grade': e.grade,
                    'agreed': e.agreed
                }
                for e in self.expert_signals
            },
            
            # Trade setup
            'entry': self.entry_zone_high or self.entry_zone_low or 0,
            'entry_zone_low': self.entry_zone_low,
            'entry_zone_high': self.entry_zone_high,
            'entry_type': self.entry_type.value if self.entry_type else 'LIMIT',
            'stop_loss': self.stop_loss,
            'take_profit_levels': tp_output,
            'risk_reward': round(self.risk_reward_ratio, 2),
            
            # Position sizing
            'position_size': self.position_size,
            'position_value': round(self.position_value, 2),
            'risk_amount': round(self.risk_amount, 2),
            'position_multiplier': self.metadata.get('position_multiplier', 1.0),
            
            # Confidence and grade
            'confidence': round(self.probability or self.consensus_confidence, 3),
            'grade': self.signal_grade,
            
            # Action
            'action': self._determine_action(),
            'decision_reason': self._build_decision_reason(),
            
            # Timing
            'expected_minutes_to_entry': self.expected_minutes_to_entry,
            
            # Story
            'story': self.story,
            'story_summary': self.story_summary,
            'key_points': self.key_points[:5],
            'warnings': self.warning_flags[:3],
            
            # Market context
            'market_regime': self.market_regime,
            'volatility_state': self.volatility_state,
            'htf_aligned': self.trend_alignment == 'ALIGNED'
        }
    
    def _determine_action(self) -> str:
        """Determine action based on signal quality"""
        if self.signal_grade in ['A+', 'A']:
            return "STRONG_ENTRY"
        elif self.signal_grade in ['B+', 'B', 'B-']:
            return "ENTER_NOW"
        elif self.signal_grade in ['C+', 'C', 'C-']:
            return "CAUTIOUS_ENTRY"
        elif self.signal_grade == 'D':
            return "REDUCED_ENTRY"
        else:
            return "SKIP"
    
    def _build_decision_reason(self) -> str:
        """Build human-readable decision reason"""
        parts = []
        
        # Direction and consensus
        parts.append(f"{self.direction}")
        if self.expert_consensus_reached:
            parts.append(f"{len(self.agreeing_experts)}/5 experts agree")
        
        # Grade and confidence
        parts.append(f"Grade {self.signal_grade}")
        parts.append(f"{self.probability:.0%} confidence")
        
        # Risk/Reward
        if self.risk_reward_ratio > 0:
            parts.append(f"RR {self.risk_reward_ratio:.1f}")
        
        # Multiple TP levels
        if len(self.take_profit_levels) > 1:
            parts.append(f"{len(self.take_profit_levels)} TP levels")
        
        return " | ".join(parts)
    
    # ===== VALIDATION METHODS =====
    
    def is_tradeable(self) -> bool:
        """Check if signal is tradeable"""
        return (
            self.expert_consensus_reached and
            self.direction in ['BUY', 'SELL'] and
            self.stop_loss is not None and
            self.take_profit is not None and
            self.risk_reward_ratio >= 1.5 and
            self.probability >= 0.65 and
            self.signal_grade not in ['D', 'F']
        )
    
    def has_strong_consensus(self, threshold: float = 0.8) -> bool:
        """Check if consensus is strong (agreement ratio >= threshold)"""
        if not self.expert_signals:
            return False
        agreement_ratio = len(self.agreeing_experts) / len(self.expert_signals)
        return agreement_ratio >= threshold
    
    def get_best_expert(self) -> Optional[ExpertSignalData]:
        """Get the highest confidence expert signal"""
        if not self.expert_signals:
            return None
        return max(self.expert_signals, key=lambda x: x.confidence * x.weight)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_empty_signal(symbol: str, timeframe: str) -> Signal:
    """Create an empty signal (for initialization)"""
    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        direction="NEUTRAL"
    )


def create_skip_signal(symbol: str, timeframe: str, reason: str) -> Signal:
    """Create a skip signal"""
    signal = Signal(
        symbol=symbol,
        timeframe=timeframe,
        direction="NEUTRAL",
        status=SignalStatus.REJECTED,
        probability=0.0,
        signal_grade="F",
        warning_flags=[reason],
        story_summary=f"SKIP: {reason}"
    )
    return signal