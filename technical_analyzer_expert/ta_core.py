# ta_core.py - Core Data Classes for Technical Analyzer Expert
"""
Core data structures for Technical Analyzer Expert
Defines all signal, indicator, and result objects used throughout the system
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class SignalDirection(Enum):
    """Trading signal direction"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class SignalAction(Enum):
    """Action to take based on signal"""
    STRONG_ENTRY = "STRONG_ENTRY"
    ENTER_NOW = "ENTER_NOW"
    SKIP = "SKIP"
    WAIT = "WAIT"


class IndicatorSignalType(Enum):
    """Type of indicator signal"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    OVERBOUGHT = "OVERBOUGHT"
    OVERSOLD = "OVERSOLD"
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"
    DIVERGENCE_BULLISH = "DIVERGENCE_BULLISH"
    DIVERGENCE_BEARISH = "DIVERGENCE_BEARISH"
    SQUEEZE = "SQUEEZE"
    BREAKOUT = "BREAKOUT"


class RegimeType(Enum):
    """Market regime classification"""
    STRONG_BULL_TREND = "STRONG_BULL_TREND"
    BULL_TREND = "BULL_TREND"
    WEAK_BULL_TREND = "WEAK_BULL_TREND"
    RANGING_BULL_BIAS = "RANGING_BULL_BIAS"
    RANGING_NEUTRAL = "RANGING_NEUTRAL"
    RANGING_BEAR_BIAS = "RANGING_BEAR_BIAS"
    WEAK_BEAR_TREND = "WEAK_BEAR_TREND"
    BEAR_TREND = "BEAR_TREND"
    STRONG_BEAR_TREND = "STRONG_BEAR_TREND"
    VOLATILE_EXPANSION = "VOLATILE_EXPANSION"
    VOLATILE_CONTRACTION = "VOLATILE_CONTRACTION"
    UNKNOWN = "UNKNOWN"


class Grade(Enum):
    """Signal grade from A+ to F"""
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
# INDICATOR RESULT CLASSES
# ============================================================================

@dataclass
class IndicatorValue:
    """Raw indicator value with metadata"""
    name: str
    value: float
    timestamp: Optional[datetime] = None
    is_valid: bool = True
    error_message: str = ""


@dataclass
class IndicatorSignal:
    """
    Individual indicator analysis result
    Each indicator produces one of these
    """
    name: str                           # Indicator name (e.g., "RSI", "MACD")
    value: float                        # Current indicator value
    signal: IndicatorSignalType         # Bullish/Bearish/Neutral etc.
    strength: float                     # 0 to 1 - how strong is this signal
    weight: float                       # Weight in final scoring (0 to 1)
    direction: str                      # "RISING", "FALLING", "FLAT"
    divergence: Optional[str] = None    # "BULLISH", "BEARISH", None
    reason: str = ""                    # Human-readable reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "value": round(self.value, 6) if self.value else 0,
            "signal": self.signal.value if hasattr(self.signal, 'value') else str(self.signal),
            "strength": round(self.strength, 3),
            "weight": round(self.weight, 3),
            "direction": self.direction,
            "divergence": self.divergence,
            "reason": self.reason,
        }


@dataclass
class CategoryScore:
    """Score for a category of indicators (momentum, trend, etc.)"""
    name: str                           # Category name
    bullish_score: float                # 0 to 1 - bullish strength
    bearish_score: float                # 0 to 1 - bearish strength
    net_score: float                    # -1 to 1 (bullish positive, bearish negative)
    agreement: float                    # 0 to 1 - how many indicators agree
    signals: List[IndicatorSignal] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "bullish_score": round(self.bullish_score, 3),
            "bearish_score": round(self.bearish_score, 3),
            "net_score": round(self.net_score, 3),
            "agreement": round(self.agreement, 3),
            "signals_count": len(self.signals),
        }


@dataclass
class DivergenceResult:
    """Divergence detection result"""
    indicator: str                      # Indicator name (RSI, MACD, OBV)
    type: str                           # "BULLISH" or "BEARISH"
    strength: float                     # 0 to 1 - divergence strength
    start_bar: int                      # Starting bar index
    end_bar: int                        # Ending bar index
    price_low: float                    # Price low for bullish divergence
    price_high: float                   # Price high for bearish divergence
    indicator_low: float                # Indicator low for bullish divergence
    indicator_high: float               # Indicator high for bearish divergence
    is_hidden: bool = False             # Hidden divergence flag
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "indicator": self.indicator,
            "type": self.type,
            "strength": round(self.strength, 3),
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "price_low": round(self.price_low, 6) if self.price_low else None,
            "price_high": round(self.price_high, 6) if self.price_high else None,
            "is_hidden": self.is_hidden,
        }


# ============================================================================
# REGIME RESULT CLASS
# ============================================================================

@dataclass
class RegimeResult:
    """Market regime analysis result"""
    regime: str                         # Regime name (from RegimeType)
    regime_type: str                    # "TREND", "RANGE", "VOLATILE", "UNKNOWN"
    confidence: float                   # 0 to 1 - confidence in regime detection
    
    bias: str                           # "BULLISH", "BEARISH", "NEUTRAL"
    bias_score: float                   # -1 to 1 (bullish positive, bearish negative)
    
    trend_strength: float               # 0 to 1 - how strong is the trend
    volatility_state: str               # "LOW", "NORMAL", "HIGH", "EXTREME"
    liquidity_state: str                # "LOW", "NORMAL", "HIGH"
    
    adx: float                          # Current ADX value
    atr_pct: float                      # ATR as percentage of price
    slope: float                        # Price slope over 20 bars
    
    is_squeeze: bool                    # Bollinger squeeze detected
    squeeze_intensity: float            # 0 to 1 - squeeze tightness
    
    wyckoff_phase: str                  # Accumulation, Markup, Distribution, Markdown
    wyckoff_confidence: float           # 0 to 1 - confidence in Wyckoff phase
    
    recommended_strategies: List[str] = field(default_factory=list)
    avoid_strategies: List[str] = field(default_factory=list)
    
    # Dynamic settings for indicators (from dynamic_indicators.py)
    indicator_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Human-readable summary
    summary: str = ""
    
    # Raw details for debugging
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "regime": self.regime,
            "regime_type": self.regime_type,
            "confidence": round(self.confidence, 3),
            "bias": self.bias,
            "bias_score": round(self.bias_score, 3),
            "trend_strength": round(self.trend_strength, 3),
            "volatility_state": self.volatility_state,
            "liquidity_state": self.liquidity_state,
            "adx": round(self.adx, 1),
            "atr_pct": round(self.atr_pct, 4),
            "slope": round(self.slope, 4),
            "is_squeeze": self.is_squeeze,
            "squeeze_intensity": round(self.squeeze_intensity, 3),
            "wyckoff_phase": self.wyckoff_phase,
            "wyckoff_confidence": round(self.wyckoff_confidence, 3),
            "recommended_strategies": self.recommended_strategies[:5],
            "avoid_strategies": self.avoid_strategies[:3],
            "summary": self.summary,
        }


# ============================================================================
# HTF RESULT CLASS
# ============================================================================

@dataclass
class HTFResult:
    """Higher timeframe analysis result"""
    timeframe: str                      # e.g., "1h", "4h", "1d"
    direction: str                      # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float                     # 0 to 1 - trend strength on HTF
    confidence: float                   # 0 to 1 - confidence in HTF analysis
    alignment_score: float              # 0 to 1 - how aligned with main TF
    
    # HTF indicator values
    rsi: Optional[float] = None
    adx: Optional[float] = None
    ema_trend: Optional[str] = None     # "BULLISH", "BEARISH", "NEUTRAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timeframe": self.timeframe,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "confidence": round(self.confidence, 3),
            "alignment_score": round(self.alignment_score, 3),
            "rsi": round(self.rsi, 1) if self.rsi else None,
            "adx": round(self.adx, 1) if self.adx else None,
            "ema_trend": self.ema_trend,
        }


@dataclass
class HTFAnalysisResult:
    """Complete HTF analysis combining all timeframes"""
    main_timeframe: str                 # Main trading timeframe (e.g., "5min")
    analyzed_timeframes: List[str]      # List of HTFs analyzed
    
    # Individual HTF results
    htf_results: Dict[str, HTFResult] = field(default_factory=dict)
    
    # Combined metrics
    overall_alignment: float = 0.0      # 0 to 1 - overall HTF alignment
    weighted_alignment: float = 0.0     # Weighted by HTF importance
    bullish_htf_count: int = 0
    bearish_htf_count: int = 0
    neutral_htf_count: int = 0
    
    # Alignment with main TF
    is_aligned: bool = False
    alignment_boost: float = 0.0        # Boost to apply to confidence (0 to 0.15)
    conflict_penalty: float = 0.0       # Penalty to apply to confidence (0 to 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "main_timeframe": self.main_timeframe,
            "analyzed_timeframes": self.analyzed_timeframes,
            "overall_alignment": round(self.overall_alignment, 3),
            "weighted_alignment": round(self.weighted_alignment, 3),
            "bullish_htf_count": self.bullish_htf_count,
            "bearish_htf_count": self.bearish_htf_count,
            "neutral_htf_count": self.neutral_htf_count,
            "is_aligned": self.is_aligned,
            "alignment_boost": round(self.alignment_boost, 3),
            "conflict_penalty": round(self.conflict_penalty, 3),
            "htf_results": {tf: r.to_dict() for tf, r in self.htf_results.items()},
        }


# ============================================================================
# TRADE SETUP CLASS
# ============================================================================

@dataclass
class TradeSetup:
    """Entry, stop loss, and take profit levels"""
    entry: float                        # Entry price
    stop_loss: float                    # Stop loss price
    take_profit: float                  # Take profit price
    risk_reward: float                  # Risk/reward ratio
    
    # Additional levels
    alternative_entry: Optional[float] = None    # Secondary entry zone
    trailing_stop_activation: Optional[float] = None
    take_profit_2: Optional[float] = None        # Second target
    take_profit_3: Optional[float] = None        # Third target
    
    # Metadata
    entry_method: str = ""              # How entry was calculated
    stop_method: str = ""               # How stop was calculated
    tp_method: str = ""                 # How target was calculated
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "entry": round(self.entry, 6) if self.entry else None,
            "stop_loss": round(self.stop_loss, 6) if self.stop_loss else None,
            "take_profit": round(self.take_profit, 6) if self.take_profit else None,
            "risk_reward": round(self.risk_reward, 2) if self.risk_reward else 0,
            "alternative_entry": round(self.alternative_entry, 6) if self.alternative_entry else None,
            "entry_method": self.entry_method,
            "stop_method": self.stop_method,
            "tp_method": self.tp_method,
            "is_valid": self.is_valid,
        }


# ============================================================================
# FINAL SIGNAL CLASS (Output)
# ============================================================================

@dataclass
class TASignal:
    """
    Final trading signal from Technical Analyzer Expert
    This is the main output that matches Pattern Expert format
    """
    # Core signal fields
    module: str = "technical_analyzer"
    signal_id: str = ""                 # Unique ID: ta_SYMBOL_DIRECTION_TIMESTAMP_HASH
    direction: str = "NEUTRAL"          # "BUY", "SELL", "NEUTRAL"
    confidence: float = 0.0             # 0 to 1 - overall confidence
    grade: str = "F"                    # A+, A, B+, B, B-, C+, C, D, F
    position_multiplier: float = 0.0    # Position size multiplier based on grade
    
    # Trade setup
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    
    # Action decision
    action: str = "SKIP"                # "STRONG_ENTRY", "ENTER_NOW", "SKIP", "WAIT"
    decision_reason: str = ""           # Human-readable reason for decision
    
    # Timestamp
    timestamp: str = ""                 # ISO format timestamp
    
    # Additional metadata (for debugging/analysis)
    symbol: str = ""
    regime: Optional[str] = None
    htf_aligned: bool = False
    category_scores: Dict[str, Any] = field(default_factory=dict)
    divergence_count: int = 0
    indicator_count: int = 0
    
    # Raw scores (for debugging)
    raw_bullish_score: float = 0.0
    raw_bearish_score: float = 0.0
    agreement_score: float = 0.0
    
    def __post_init__(self):
        """Auto-generate signal_id and timestamp if not provided"""
        if not self.signal_id and self.symbol and self.direction:
            self.signal_id = self.generate_signal_id()
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def generate_signal_id(self) -> str:
        """Generate unique signal ID: ta_BTCUSDT_BUY_20240117_143022_a3f2"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_str = f"{self.symbol}_{self.direction}_{timestamp_str}_{time.time_ns()}"
        short_hash = hashlib.md5(unique_str.encode()).hexdigest()[:6]
        return f"ta_{self.symbol}_{self.direction}_{timestamp_str}_{short_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output (matches Pattern Expert)"""
        result = {
            "module": self.module,
            "signal_id": self.signal_id,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "grade": self.grade,
            "position_multiplier": round(self.position_multiplier, 2),
            "action": self.action,
            "decision_reason": self.decision_reason,
            "timestamp": self.timestamp,
        }
        
        # Add trade setup only if entry exists
        if self.entry is not None:
            result["entry"] = round(self.entry, 6)
        if self.stop_loss is not None:
            result["stop_loss"] = round(self.stop_loss, 6)
        if self.take_profit is not None:
            result["take_profit"] = round(self.take_profit, 6)
        if self.risk_reward is not None:
            result["risk_reward"] = round(self.risk_reward, 2)
        
        # Add optional metadata for debugging (not in standard output)
        if self.symbol:
            result["_metadata"] = {
                "symbol": self.symbol,
                "regime": self.regime,
                "htf_aligned": self.htf_aligned,
                "raw_bullish_score": round(self.raw_bullish_score, 3),
                "raw_bearish_score": round(self.raw_bearish_score, 3),
                "agreement_score": round(self.agreement_score, 3),
            }
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def is_tradeable(self) -> bool:
        """Check if signal is tradeable (not SKIP)"""
        return self.action in ["STRONG_ENTRY", "ENTER_NOW"]
    
    def skip_reason(self) -> Optional[str]:
        """Get reason why signal was skipped, if applicable"""
        if self.action != "SKIP":
            return None
        return self.decision_reason


# ============================================================================
# ANALYSIS CONTEXT (Internal Use)
# ============================================================================

@dataclass
class AnalysisContext:
    """Context object passed through analysis pipeline"""
    symbol: str
    main_df: Any                       # DataFrame with OHLCV data
    htf_data: Dict[str, Any]           # Dict of timeframe -> DataFrame
    regime: Optional[RegimeResult] = None
    htf_analysis: Optional[HTFAnalysisResult] = None
    indicator_signals: List[IndicatorSignal] = field(default_factory=list)
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)
    divergences: List[DivergenceResult] = field(default_factory=list)
    trade_setup: Optional[TradeSetup] = None
    
    # Performance tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_skip_signal(symbol: str, reason: str, confidence: float = 0.0) -> TASignal:
    """Create a SKIP signal with reason"""
    return TASignal(
        symbol=symbol,
        direction="NEUTRAL",
        confidence=confidence,
        grade="F",
        position_multiplier=0.0,
        action="SKIP",
        decision_reason=reason,
    )


def create_neutral_signal(symbol: str, reason: str = "No clear direction") -> TASignal:
    """Create a neutral signal"""
    return TASignal(
        symbol=symbol,
        direction="NEUTRAL",
        confidence=0.0,
        grade="F",
        position_multiplier=0.0,
        action="SKIP",
        decision_reason=reason,
    )


def get_grade_from_score(score: float, grade_thresholds: Dict[str, float]) -> str:
    """Convert numerical score to grade based on thresholds"""
    for grade, threshold in sorted(grade_thresholds.items(), key=lambda x: x[1], reverse=True):
        if score >= threshold:
            return grade
    return "F"


def get_position_multiplier_from_grade(grade: str, position_multipliers: Dict[str, float]) -> float:
    """Get position multiplier based on grade"""
    return position_multipliers.get(grade, 0.0)