# price_action_expert/__init__.py
"""
Price Action Expert V3.5
Pure price action trading signals using candlestick patterns, traps, and sequence analysis

This expert uses ONLY price action - no indicators needed.
Detects 35+ candlestick patterns with confidence scoring.
"""

from .expert_price_action import ExpertPriceAction, analyze_price_action, get_best_setup
from .candle_analyzer import CandleAnalyzer, CandleData, analyze_candles, get_latest_candle, get_candle_summary
from .pattern_detector import PatternDetector, DetectedPattern, PatternDirection, PatternType, detect_patterns, get_best_pattern
from .context_engine import ContextEngine, MarketContext, get_market_context, get_trend_stage, get_mtf_alignment
from .trap_engine import TrapEngine, DetectedTrap, TrapType, TrapSeverity, detect_traps, get_best_trap, get_trap_severity_score
from .sequence_analyzer import SequenceAnalyzer, CandleStory, SequenceType, analyze_sequence, get_sequence_summary, get_momentum_score
from .sl_tp_engine import SLTPEngine, TradeSetup, StopType, TargetType, calculate_trade_setup
from .scoring_engine import ScoringEngine, ScoringResult, SignalGrade, calculate_score, get_confidence, is_tradeable
from .signal_formatter import SignalFormatter, FormattedSignal, SignalAction, SignalStage, format_signal, format_skip
from .adapter import price_action_to_expert_signal, ExpertSignal, is_tradeable_signal

__version__ = "3.5.0"

__all__ = [
    # Main expert
    "ExpertPriceAction",
    "analyze_price_action",
    "get_best_setup",
    
    # Layer 1: Candle Analyzer
    "CandleAnalyzer",
    "CandleData",
    "analyze_candles",
    "get_latest_candle",
    "get_candle_summary",
    
    # Layer 2: Pattern Detector
    "PatternDetector",
    "DetectedPattern",
    "PatternDirection",
    "PatternType",
    "detect_patterns",
    "get_best_pattern",
    
    # Layer 3: Context Engine
    "ContextEngine",
    "MarketContext",
    "get_market_context",
    "get_trend_stage",
    "get_mtf_alignment",
    
    # Layer 4: Trap Engine
    "TrapEngine",
    "DetectedTrap",
    "TrapType",
    "TrapSeverity",
    "detect_traps",
    "get_best_trap",
    "get_trap_severity_score",
    
    # Layer 5: Sequence Analyzer
    "SequenceAnalyzer",
    "CandleStory",
    "SequenceType",
    "analyze_sequence",
    "get_sequence_summary",
    "get_momentum_score",
    
    # Layer 6: SL/TP Engine
    "SLTPEngine",
    "TradeSetup",
    "StopType",
    "TargetType",
    "calculate_trade_setup",
    
    # Layer 7: Scoring Engine
    "ScoringEngine",
    "ScoringResult",
    "SignalGrade",
    "calculate_score",
    "get_confidence",
    "is_tradeable",
    
    # Layer 8: Signal Formatter
    "SignalFormatter",
    "FormattedSignal",
    "SignalAction",
    "SignalStage",
    "format_signal",
    "format_skip",
    
    # Adapter
    "price_action_to_expert_signal",
    "ExpertSignal",
    "is_tradeable_signal",
]