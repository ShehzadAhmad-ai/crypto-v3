"""
pattern_core.py - Core Pattern Classes for V4 Pattern Intelligence System

This file contains the fundamental data structures used throughout the pattern system.
All pattern data flows through these classes with proper typing and validation.

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import uuid


# ============================================================================
# ENUMS - Type Safety
# ============================================================================

class PatternDirection(Enum):
    """Trading direction for patterns"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class PatternType(Enum):
    """Classification of pattern families"""
    HARMONIC = "HARMONIC"
    STRUCTURE = "STRUCTURE"
    VOLUME = "VOLUME"
    TREND = "TREND"
    DIVERGENCE = "DIVERGENCE"
    LIQUIDITY = "LIQUIDITY"
    WAVE = "WAVE"


class PatternStage(Enum):
    """Lifecycle stage of a pattern"""
    FORMING = "forming"
    BREAKOUT = "breakout"
    RETEST = "retest"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class ActionType(Enum):
    """Action to take based on pattern"""
    WAIT = "WAIT"
    WAIT_FOR_RETEST = "WAIT_FOR_RETEST"
    ENTER_NOW = "ENTER_NOW"
    STRONG_ENTRY = "STRONG_ENTRY"
    HOLD = "HOLD"
    TAKE_PROFIT = "TAKE_PROFIT"
    CANCEL = "CANCEL"
    FLIP_TO_BUY = "FLIP_TO_BUY"
    FLIP_TO_SELL = "FLIP_TO_SELL"
    SKIP = "SKIP"


class TrapType(Enum):
    """Types of traps that can be detected"""
    BULL_TRAP = "bull_trap"
    BEAR_TRAP = "bear_trap"
    LIQUIDITY_GRAB = "liquidity_grab"
    FAILED_BREAKOUT = "failed_breakout"
    INDUCEMENT = "inducement"
    SWEEP_FAILURE = "sweep_failure"
    NONE = "none"


class Grade(Enum):
    """Pattern quality grade"""
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    D = "D"
    F = "F"


# ============================================================================
# PATTERN SIMILARITY - Continuous Scoring (0-1)
# ============================================================================

@dataclass
class PatternSimilarity:
    """
    Geometry-based similarity score (0-1).
    Each component represents how close the pattern matches ideal geometry.
    """
    total: float = 0.0
    
    # ===== Head & Shoulders components =====
    shoulder_symmetry: float = 0.0      # Left/right shoulder height match
    head_prominence: float = 0.0        # Head clearly higher/lower
    neckline_quality: float = 0.0       # Clean, not too sloped
    structure_clarity: float = 0.0      # Clean swing alternation
    volume_pattern: float = 0.0         # Volume contraction/expansion
    fib_ratio: float = 0.0              # Fibonacci ratio accuracy
    time_symmetry: float = 0.0          # Left/right time symmetry
    
    # ===== Double/Triple Top/Bottom components =====
    price_similarity: float = 0.0       # Tops/bottoms at similar levels
    valley_strength: float = 0.0        # Pullback depth significance
    breakout_volume: float = 0.0        # Volume on breakout
    
    # ===== Flag/Pennant components =====
    pole_strength: float = 0.0          # Sharp move magnitude
    tightness: float = 0.0              # Consolidation range tightness
    convergence: float = 0.0            # Lines converging (for pennant/wedge)
    slope_quality: float = 0.0          # Clean slope
    
    # ===== Triangle components =====
    flat_resistance: float = 0.0        # Horizontal top (ascending)
    rising_support: float = 0.0         # Rising bottom (ascending)
    flat_support: float = 0.0           # Horizontal bottom (descending)
    falling_resistance: float = 0.0     # Falling top (descending)
    symmetry: float = 0.0               # Equal slope magnitudes
    
    # ===== Harmonic components =====
    ab_retrace: float = 0.0             # AB retracement of XA
    bc_retrace: float = 0.0             # BC retracement of AB
    xd_extension: float = 0.0           # XD extension of XA
    fib_accuracy: float = 0.0           # Overall Fibonacci accuracy
    drive_ratio: float = 0.0            # Each drive ratio (Three Drives)
    alternation: float = 0.0            # Clean high/low alternation
    rsi_divergence: float = 0.0         # RSI divergence strength
    
    # ===== Cup & Handle components =====
    cup_depth: float = 0.0              # 5-20% depth
    cup_symmetry: float = 0.0           # Left/right rim symmetry
    handle_depth: float = 0.0           # Handle 20-40% of cup
    
    # ===== Adam & Eve components =====
    shape_quality: float = 0.0          # Adam sharp, Eve rounded
    peak_strength: float = 0.0          # Peak between bottoms
    
    # ===== Quasimodo components =====
    reversal_strength: float = 0.0      # Clear reversal
    
    # ===== VCP components =====
    contraction_count: float = 0.0      # 2-5 contractions
    volatility_reduction: float = 0.0   # Each contraction tighter
    
    # ===== Wolfe Wave components =====
    wave_alternation: float = 0.0       # 1-2-3-4-5 pattern
    wave_symmetry: float = 0.0          # Wave relationships
    target_alignment: float = 0.0       # Point 1-4 line target
    
    # ===== Divergence components =====
    price_swing_magnitude: float = 0.0  # Price swing size
    divergence_strength: float = 0.0    # RSI divergence magnitude
    
    # Raw measurements for debugging
    raw_measurements: Dict[str, float] = field(default_factory=dict)
    
    def weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted total from components"""
        total = 0.0
        weight_sum = 0.0
        
        for comp, weight in weights.items():
            if hasattr(self, comp):
                total += getattr(self, comp) * weight
                weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total': round(self.total, 3),
            'shoulder_symmetry': round(self.shoulder_symmetry, 3),
            'head_prominence': round(self.head_prominence, 3),
            'neckline_quality': round(self.neckline_quality, 3),
            'structure_clarity': round(self.structure_clarity, 3),
            'volume_pattern': round(self.volume_pattern, 3),
            'fib_ratio': round(self.fib_ratio, 3),
            'time_symmetry': round(self.time_symmetry, 3),
            'price_similarity': round(self.price_similarity, 3),
            'valley_strength': round(self.valley_strength, 3),
            'breakout_volume': round(self.breakout_volume, 3),
            'pole_strength': round(self.pole_strength, 3),
            'tightness': round(self.tightness, 3),
            'convergence': round(self.convergence, 3),
            'slope_quality': round(self.slope_quality, 3),
            'symmetry': round(self.symmetry, 3),
            'fib_accuracy': round(self.fib_accuracy, 3),
            'cup_depth': round(self.cup_depth, 3),
            'shape_quality': round(self.shape_quality, 3),
            'contraction_count': round(self.contraction_count, 3),
            'volatility_reduction': round(self.volatility_reduction, 3),
            'divergence_strength': round(self.divergence_strength, 3),
        }


# ============================================================================
# CONTEXT SCORE - Market Context with SMC
# ============================================================================

@dataclass
class ContextScore:
    """
    Market context score with SMC concepts (0-1).
    Higher score = more favorable market conditions.
    """
    total: float = 0.0
    
    # Core components
    trend_alignment: float = 0.0        # Pattern vs market trend
    support_resistance: float = 0.0     # Reaction at key level
    volume_confirmation: float = 0.0    # Volume supports move
    volatility_condition: float = 0.0   # Optimal volatility range
    liquidity_sweep: float = 0.0        # Sweep/BOS/Order block present
    volume_profile: float = 0.0         # Volume profile strength
    
    # Dynamic weights applied (for logging)
    applied_weights: Dict[str, float] = field(default_factory=dict)
    applied_regime: str = "NEUTRAL"
    
    # SMC Flags
    at_support: bool = False
    at_resistance: bool = False
    sweep_detected: bool = False
    bos_detected: bool = False          # Break of Structure
    order_block_nearby: bool = False
    fvg_nearby: bool = False            # Fair Value Gap
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total': round(self.total, 3),
            'trend_alignment': round(self.trend_alignment, 3),
            'support_resistance': round(self.support_resistance, 3),
            'volume_confirmation': round(self.volume_confirmation, 3),
            'volatility_condition': round(self.volatility_condition, 3),
            'liquidity_sweep': round(self.liquidity_sweep, 3),
            'volume_profile': round(self.volume_profile, 3),
            'applied_regime': self.applied_regime,
            'at_support': self.at_support,
            'at_resistance': self.at_resistance,
            'sweep_detected': self.sweep_detected,
            'bos_detected': self.bos_detected,
            'order_block_nearby': self.order_block_nearby,
        }


# ============================================================================
# HTF CONFLUENCE - Multi-Timeframe Analysis
# ============================================================================

@dataclass
class HTFConfluence:
    """
    Multi-timeframe pattern confluence score.
    Higher score = more timeframes agree.
    """
    weighted_score: float = 0.0
    boost_factor: float = 1.0
    
    # Individual timeframe results
    timeframe_scores: Dict[str, float] = field(default_factory=dict)
    timeframe_patterns: Dict[str, str] = field(default_factory=dict)
    
    same_pattern_count: int = 0
    aligned_trend_count: int = 0
    conflicting_count: int = 0
    
    # Which timeframes had same pattern
    same_pattern_timeframes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'weighted_score': round(self.weighted_score, 3),
            'boost_factor': round(self.boost_factor, 3),
            'timeframe_scores': self.timeframe_scores,
            'same_pattern_count': self.same_pattern_count,
            'aligned_trend_count': self.aligned_trend_count,
            'same_pattern_timeframes': self.same_pattern_timeframes,
        }


# ============================================================================
# PATTERN EVOLUTION - Real-Time Tracking
# ============================================================================

@dataclass
class PatternEvolution:
    """
    Real-time pattern evolution tracking.
    Tracks how pattern confidence changes over time.
    """
    initial_confidence: float = 0.0
    current_confidence: float = 0.0
    confidence_trend: str = "STABLE"    # IMPROVING, DEGRADING, STABLE
    last_update: str = ""
    completion_pct: float = 0.0
    expected_completion_bars: int = 0
    
    # History of updates
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, new_confidence: float, new_completion: float, timestamp: str):
        """Update pattern state"""
        old_confidence = self.current_confidence
        self.current_confidence = new_confidence
        self.completion_pct = new_completion
        self.last_update = timestamp
        
        # Determine trend
        if new_confidence > old_confidence * 1.05:
            self.confidence_trend = "IMPROVING"
        elif new_confidence < old_confidence * 0.95:
            self.confidence_trend = "DEGRADING"
        else:
            self.confidence_trend = "STABLE"
        
        # Add to history
        self.update_history.append({
            'timestamp': timestamp,
            'confidence': new_confidence,
            'completion': new_completion,
            'trend': self.confidence_trend
        })
        
        # Keep last 20 updates
        if len(self.update_history) > 20:
            self.update_history = self.update_history[-20:]
    
    def has_degraded(self, threshold: float = 0.20) -> bool:
        """Check if confidence has degraded too much"""
        if self.initial_confidence == 0:
            return False
        degradation = (self.initial_confidence - self.current_confidence) / self.initial_confidence
        return degradation > threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'initial_confidence': round(self.initial_confidence, 3),
            'current_confidence': round(self.current_confidence, 3),
            'confidence_trend': self.confidence_trend,
            'last_update': self.last_update,
            'completion_pct': round(self.completion_pct, 3),
            'update_count': len(self.update_history),
        }


# ============================================================================
# FALSE BREAKOUT RISK
# ============================================================================

@dataclass
class FalseBreakoutRisk:
    """
    False breakout detection and risk scoring.
    Lower risk_score = lower chance of false breakout.
    """
    risk_score: float = 0.0         # 0-1, lower is better
    penalty_applied: float = 0.0
    
    features: Dict[str, bool] = field(default_factory=dict)
    feature_penalties: Dict[str, float] = field(default_factory=dict)
    
    volume_ok: bool = True
    wick_ok: bool = True
    reversal_ok: bool = True
    htf_level_ok: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'risk_score': round(self.risk_score, 3),
            'penalty_applied': round(self.penalty_applied, 3),
            'features': self.features,
            'volume_ok': self.volume_ok,
            'wick_ok': self.wick_ok,
            'reversal_ok': self.reversal_ok,
            'htf_level_ok': self.htf_level_ok,
        }


# ============================================================================
# LIQUIDITY ANALYSIS
# ============================================================================

@dataclass
class LiquidityAnalysis:
    """
    Liquidity intelligence results.
    """
    score: float = 0.5
    net_bias: float = 0.0           # -1 to 1, positive = bullish
    direction: str = "NEUTRAL"
    
    # Sweep detection
    sweeps: List[Dict[str, Any]] = field(default_factory=list)
    has_down_sweep: bool = False
    has_up_sweep: bool = False
    sweep_strength: float = 0.0
    
    # Inducements
    inducements: List[Dict[str, Any]] = field(default_factory=list)
    inducement_count: int = 0
    
    # Sweep failures
    sweep_failures: List[Dict[str, Any]] = field(default_factory=list)
    sweep_failure_count: int = 0
    
    # Stop hunts
    stop_hunt_probability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'score': round(self.score, 3),
            'net_bias': round(self.net_bias, 3),
            'direction': self.direction,
            'has_down_sweep': self.has_down_sweep,
            'has_up_sweep': self.has_up_sweep,
            'sweep_strength': round(self.sweep_strength, 3),
            'inducement_count': self.inducement_count,
            'sweep_failure_count': self.sweep_failure_count,
            'stop_hunt_probability': round(self.stop_hunt_probability, 3),
        }


# ============================================================================
# TRAP ANALYSIS
# ============================================================================

@dataclass
class TrapAnalysis:
    """
    Trap detection and conversion results.
    """
    detected: bool = False
    trap_type: TrapType = TrapType.NONE
    strength: float = 0.0
    converted: bool = False
    convert_to: Optional[PatternDirection] = None
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'detected': self.detected,
            'trap_type': self.trap_type.value,
            'strength': round(self.strength, 3),
            'converted': self.converted,
            'convert_to': self.convert_to.value if self.convert_to else None,
            'confidence': round(self.confidence, 3),
        }


# ============================================================================
# TRADE SETUP
# ============================================================================

@dataclass
class TradeSetup:
    """
    Complete trade setup with entry, stop, target.
    """
    # Draft (initial calculation)
    draft_entry: Optional[float] = None
    draft_stop: Optional[float] = None
    draft_target: Optional[float] = None
    draft_rr: float = 0.0
    
    # Final (after all adjustments)
    final_entry: Optional[float] = None
    final_stop: Optional[float] = None
    final_target: Optional[float] = None
    final_rr: float = 0.0
    
    # Execution metadata
    entry_mode: str = "retest"       # retest, breakout, anticipation
    stop_reason: str = ""
    target_reason: str = ""
    
    # Retest tracking
    retest_level: Optional[float] = None
    retest_confirmed: bool = False
    bars_since_breakout: int = 0
    bars_since_retest: int = 0
    
    # For active trades
    entry_reached: bool = False
    stop_reached: bool = False
    target_reached: bool = False
    
    def calculate_rr(self) -> float:
        """Calculate risk/reward ratio"""
        if self.final_entry is None or self.final_stop is None or self.final_target is None:
            return 0.0
        
        risk = abs(self.final_entry - self.final_stop)
        reward = abs(self.final_target - self.final_entry)
        
        return reward / risk if risk > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entry': self.final_entry,
            'stop_loss': self.final_stop,
            'take_profit': self.final_target,
            'risk_reward': round(self.final_rr, 2),
            'entry_mode': self.entry_mode,
            'retest_level': self.retest_level,
            'retest_confirmed': self.retest_confirmed,
        }


# ============================================================================
# PATTERN CLUSTERING
# ============================================================================

@dataclass
class PatternClustering:
    """
    Multi-pattern confluence data.
    """
    cluster_id: Optional[str] = None
    cluster_size: int = 1
    cluster_score: float = 0.0
    aligned_patterns: List[str] = field(default_factory=list)
    conflict_detected: bool = False
    conflict_resolution: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cluster_id': self.cluster_id,
            'cluster_size': self.cluster_size,
            'cluster_score': round(self.cluster_score, 3),
            'aligned_patterns': self.aligned_patterns[:5],
            'conflict_detected': self.conflict_detected,
        }


# ============================================================================
# PATTERN LIFECYCLE
# ============================================================================

@dataclass
class PatternLifecycle:
    """
    Pattern lifecycle tracking.
    """
    stage: PatternStage = PatternStage.FORMING
    stage_confidence: float = 0.5
    age_bars: int = 0
    formation_bars: int = 0
    completion_pct: float = 0.0
    bars_since_breakout: int = 0
    bars_since_retest: int = 0
    next_trigger: Optional[str] = None
    failure_reason: Optional[str] = None
    
    # Time decay
    decay_factor: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'stage': self.stage.value,
            'stage_confidence': round(self.stage_confidence, 3),
            'age_bars': self.age_bars,
            'formation_bars': self.formation_bars,
            'completion_pct': round(self.completion_pct, 3),
            'decay_factor': round(self.decay_factor, 3),
        }


# ============================================================================
# MAIN PATTERN CLASS - Pattern V4
# ============================================================================

@dataclass
class PatternV4:
    """
    Complete pattern with all V4 enhancements.
    This is the main pattern class used throughout the system.
    """
    
    # ===== IDENTIFICATION =====
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    direction: PatternDirection = PatternDirection.NEUTRAL
    pattern_type: PatternType = PatternType.STRUCTURE
    timeframe: str = ""
    symbol: str = ""
    
    # ===== DETECTION DATA =====
    points: List[Tuple[int, float]] = field(default_factory=list)  # (index, price)
    formation_bars: int = 0
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    # ===== SIMILARITY SCORES (NEW for V4) =====
    similarity: PatternSimilarity = field(default_factory=PatternSimilarity)
    raw_similarity: float = 0.0          # Just the pattern similarity (0-1)
    
    # ===== CONTEXT SCORES (NEW for V4) =====
    context_score_obj: ContextScore = field(default_factory=ContextScore)
    context_score: float = 0.0           # Weighted context score (0-1)
    
    # ===== HTF CONFLUENCE (NEW for V4) =====
    htf_confluence: HTFConfluence = field(default_factory=HTFConfluence)
    
    # ===== PATTERN EVOLUTION (NEW for V4) =====
    evolution: PatternEvolution = field(default_factory=PatternEvolution)
    
    # ===== FALSE BREAKOUT (NEW for V4) =====
    false_breakout_risk: FalseBreakoutRisk = field(default_factory=FalseBreakoutRisk)
    
    # ===== LIQUIDITY =====
    liquidity: LiquidityAnalysis = field(default_factory=LiquidityAnalysis)
    
    # ===== TRAP ANALYSIS =====
    trap: TrapAnalysis = field(default_factory=TrapAnalysis)
    
    # ===== LIFECYCLE =====
    lifecycle: PatternLifecycle = field(default_factory=PatternLifecycle)
    
    # ===== TRADE SETUP =====
    trade_setup: TradeSetup = field(default_factory=TradeSetup)
    
    # ===== CLUSTERING =====
    clustering: PatternClustering = field(default_factory=PatternClustering)
    
    # ===== CONFIDENCE SCORES =====
    final_confidence: float = 0.0        # Final calibrated confidence (0-1)
    grade: Grade = Grade.F
    
    # ===== REGIME & MULTIPLIERS =====
    regime_multiplier: float = 1.0
    position_multiplier: float = 1.0
    
    # ===== DECISION =====
    action: ActionType = ActionType.SKIP
    action_detail: str = ""
    decision_reason: str = ""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # ===== VALIDATION =====
    invalid: bool = False
    invalid_reason: str = ""
    
    # ===== METADATA =====
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    
    # ===== HELPER METHODS =====
    
    def get_tradeable_stage(self) -> bool:
        """Check if pattern is in a tradable stage"""
        return self.lifecycle.stage in [
            PatternStage.BREAKOUT,
            PatternStage.RETEST,
            PatternStage.CONFIRMED
        ]
    
    def should_trade(self, min_confidence: float = 0.65) -> bool:
        """Final check if pattern should be traded"""
        return (
            not self.invalid and
            self.final_confidence >= min_confidence and
            self.trade_setup.final_rr >= 1.5 and
            self.get_tradeable_stage() and
            self.action in [ActionType.ENTER_NOW, ActionType.STRONG_ENTRY]
        )
    
    def to_external_decision(self) -> Dict[str, Any]:
        """
        Convert to clean external decision output.
        This is what gets sent to technical_pipeline / expert_aggregator.
        """
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.name,
            'direction': self.direction.value,
            'action': self.action.value,
            'action_detail': self.action_detail,
            'entry': self.trade_setup.final_entry,
            'stop_loss': self.trade_setup.final_stop,
            'take_profit': self.trade_setup.final_target,
            'risk_reward': self.trade_setup.final_rr,
            'confidence': self.final_confidence,
            'grade': self.grade.value,
            'position_multiplier': self.position_multiplier,
            'decision_reason': self.decision_reason,
            'reasons': self.reasons,
            'timestamp': self.timestamp.isoformat(),
            'completion_pct': self.lifecycle.completion_pct,
            'evolution': self.evolution.to_dict(),
            'similarity_components': self.similarity.to_dict(),
            'context_components': self.context_score_obj.to_dict(),
        }
    
    def to_internal_dict(self) -> Dict[str, Any]:
        """
        Convert to complete internal dictionary for debugging/analysis.
        """
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'direction': self.direction.value,
            'type': self.pattern_type.value,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'similarity': self.similarity.to_dict(),
            'raw_similarity': round(self.raw_similarity, 3),
            'context_score': round(self.context_score, 3),
            'context_details': self.context_score_obj.to_dict(),
            'htf_confluence': self.htf_confluence.to_dict(),
            'evolution': self.evolution.to_dict(),
            'false_breakout_risk': self.false_breakout_risk.to_dict(),
            'liquidity': self.liquidity.to_dict(),
            'trap': self.trap.to_dict(),
            'lifecycle': self.lifecycle.to_dict(),
            'trade_setup': self.trade_setup.to_dict(),
            'clustering': self.clustering.to_dict(),
            'final_confidence': round(self.final_confidence, 3),
            'grade': self.grade.value,
            'action': self.action.value,
            'decision_reason': self.decision_reason,
            'reasons': self.reasons,
            'invalid': self.invalid,
            'invalid_reason': self.invalid_reason,
        }


# ============================================================================
# SWING POINT CLASS
# ============================================================================

@dataclass
class SwingPoint:
    """
    Swing point for pattern detection.
    """
    index: int                          # Bar index in DataFrame
    price: float                        # Price at swing
    type: str                           # 'HH', 'HL', 'LH', 'LL'
    timestamp: pd.Timestamp             # Time of swing
    strength: float = 0.5               # Swing strength (0-1)
    volume_ratio: float = 1.0           # Volume at swing / average volume
    time_weight: float = 1.0            # Time decay weight (newer = higher)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'index': self.index,
            'price': round(self.price, 4),
            'type': self.type,
            'strength': round(self.strength, 3),
            'volume_ratio': round(self.volume_ratio, 3),
            'time_weight': round(self.time_weight, 3),
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_pattern_v4(
    name: str,
    direction: str,
    pattern_type: str,
    points: List[Tuple[int, float]],
    timestamp: pd.Timestamp,
    symbol: str = "",
    timeframe: str = ""
) -> PatternV4:
    """
    Factory function to create a new pattern with defaults.
    """
    # Convert string direction to enum
    direction_map = {
        "BUY": PatternDirection.BUY,
        "SELL": PatternDirection.SELL,
        "NEUTRAL": PatternDirection.NEUTRAL
    }
    dir_enum = direction_map.get(direction.upper(), PatternDirection.NEUTRAL)
    
    # Convert string type to enum
    type_map = {
        "HARMONIC": PatternType.HARMONIC,
        "STRUCTURE": PatternType.STRUCTURE,
        "VOLUME": PatternType.VOLUME,
        "TREND": PatternType.TREND,
        "DIVERGENCE": PatternType.DIVERGENCE,
        "WAVE": PatternType.WAVE,
    }
    type_enum = type_map.get(pattern_type.upper(), PatternType.STRUCTURE)
    
    return PatternV4(
        name=name,
        direction=dir_enum,
        pattern_type=type_enum,
        points=points,
        formation_bars=len(points),
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'PatternDirection',
    'PatternType',
    'PatternStage',
    'ActionType',
    'TrapType',
    'Grade',
    
    # Data classes
    'PatternSimilarity',
    'ContextScore',
    'HTFConfluence',
    'PatternEvolution',
    'FalseBreakoutRisk',
    'LiquidityAnalysis',
    'TrapAnalysis',
    'TradeSetup',
    'PatternClustering',
    'PatternLifecycle',
    'PatternV4',
    'SwingPoint',
    
    # Factory functions
    'create_pattern_v4',
]