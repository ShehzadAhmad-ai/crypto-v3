# __init__.py - Technical Analyzer Expert Module
"""
Technical Analyzer Expert - Advanced Technical Analysis System
Complete technical analysis with 50+ indicators, regime detection,
HTF analysis, divergence detection, and signal generation.

Version: 2.0.0
Author: Technical Analyzer Expert
"""

# Import core classes
from .ta_core import (
    TASignal,
    TradeSetup,
    RegimeResult,
    CategoryScore,
    IndicatorSignal,
    IndicatorValue,
    DivergenceResult,
    HTFResult,
    HTFAnalysisResult,
    AnalysisContext,
    SignalDirection,
    SignalAction,
    IndicatorSignalType,
    RegimeType,
    Grade,
    create_skip_signal,
    create_neutral_signal,
    get_grade_from_score,
    get_position_multiplier_from_grade,
)

# Import factory and main functions
from .ta_factory import (
    TechnicalAnalyzerFactory,
    analyze_technical,
)

# Import configuration
from .ta_config import *

# Import all analyzers (for advanced usage)
from .market_regime_advanced import MarketRegimeAdvanced, detect_market_regime
from .htf_analyzer import HTFAnalyzer, analyze_timeframes, get_htf_alignment_boost
from .indicator_analyzer import IndicatorAnalyzer, analyze_all_indicators
from .divergence_detector import DivergenceDetector, detect_divergences, get_divergence_bias
from .signal_generator import SignalGenerator, generate_technical_signal, is_signal_valid
from .entry_sl_tp_engine import EntrySLTPEngine, calculate_trade_levels, get_dynamic_stop_loss
from .scoring_engine import ScoringEngine, score_technical_signal, calculate_agreement

# Import existing technical analyzer (for backward compatibility)
try:
    from .technical_analyzer import TechnicalAnalyzer
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYZER_AVAILABLE = False
    TechnicalAnalyzer = None

# Version
__version__ = "2.0.0"
__author__ = "Technical Analyzer Expert"
__description__ = "Advanced Technical Analysis System with 50+ Indicators"

# Public API
__all__ = [
    # Main entry points
    'TechnicalAnalyzerFactory',
    'analyze_technical',
    
    # Core data classes
    'TASignal',
    'TradeSetup',
    'RegimeResult',
    'CategoryScore',
    'IndicatorSignal',
    'IndicatorValue',
    'DivergenceResult',
    'HTFResult',
    'HTFAnalysisResult',
    'AnalysisContext',
    
    # Enums
    'SignalDirection',
    'SignalAction',
    'IndicatorSignalType',
    'RegimeType',
    'Grade',
    
    # Helper functions
    'create_skip_signal',
    'create_neutral_signal',
    'get_grade_from_score',
    'get_position_multiplier_from_grade',
    'is_signal_valid',
    
    # Analyzer classes (for advanced usage)
    'MarketRegimeAdvanced',
    'HTFAnalyzer',
    'IndicatorAnalyzer',
    'DivergenceDetector',
    'SignalGenerator',
    'EntrySLTPEngine',
    'ScoringEngine',
    
    # Wrapper functions
    'detect_market_regime',
    'analyze_timeframes',
    'get_htf_alignment_boost',
    'analyze_all_indicators',
    'detect_divergences',
    'get_divergence_bias',
    'generate_technical_signal',
    'calculate_trade_levels',
    'get_dynamic_stop_loss',
    'score_technical_signal',
    'calculate_agreement',
]

# Conditionally add TechnicalAnalyzer to exports
if TECHNICAL_ANALYZER_AVAILABLE:
    __all__.append('TechnicalAnalyzer')

# Module metadata
MODULE_NAME = "technical_analyzer_expert"
MODULE_VERSION = __version__


# ============================================================================
# Quick Start Guide
# ============================================================================

def quick_start():
    """
    Quick start guide for using the Technical Analyzer Expert.
    Prints usage examples and basic instructions.
    """
    guide = """
    ============================================================================
    TECHNICAL ANALYZER EXPERT - QUICK START GUIDE
    ============================================================================
    
    BASIC USAGE:
    ------------
    from technical_analyzer_expert import analyze_technical
    
    # Simple analysis
    signal = analyze_technical("BTCUSDT", df)
    
    # With HTF data
    htf_data = {
        "1h": df_1h,
        "4h": df_4h,
    }
    signal = analyze_technical("BTCUSDT", df, htf_data)
    
    # Check if tradeable
    if signal.is_tradeable():
        print(f"Enter {signal.direction} at {signal.entry}")
        print(f"Stop Loss: {signal.stop_loss}")
        print(f"Take Profit: {signal.take_profit}")
        print(f"Risk/Reward: {signal.risk_reward}")
    
    ADVANCED USAGE:
    --------------
    from technical_analyzer_expert import TechnicalAnalyzerFactory
    
    factory = TechnicalAnalyzerFactory()
    signal = factory.analyze(df, "BTCUSDT", htf_data)
    
    # Batch analysis
    signals = factory.analyze_batch({
        "BTCUSDT": df_btc,
        "ETHUSDT": df_eth,
    })
    
    OUTPUT FORMAT:
    -------------
    {
        "module": "technical_analyzer",
        "signal_id": "ta_BTCUSDT_BUY_20240117_143022_a3f2",
        "direction": "BUY",
        "confidence": 0.82,
        "grade": "A",
        "position_multiplier": 1.3,
        "entry": 51200.00,
        "stop_loss": 50800.00,
        "take_profit": 52000.00,
        "risk_reward": 2.5,
        "action": "STRONG_ENTRY",
        "decision_reason": "BUY signal - Grade A | very high confidence | bullish trend regime",
        "timestamp": "2024-01-17T14:30:22"
    }
    
    ============================================================================
    """
    print(guide)


# ============================================================================
# Module Information
# ============================================================================

def get_info():
    """
    Get module information and status.
    """
    return {
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": __description__,
        "author": __author__,
        "technical_analyzer_available": TECHNICAL_ANALYZER_AVAILABLE,
        "indicators_count": 50,  # Base count from technical_analyzer.py
        "timeframes_supported": ["5min", "15min", "1h", "4h"],
        "divergence_indicators": ["RSI", "MACD", "OBV", "Stochastic", "CCI", "MFI"],
    }


# ============================================================================
# Module Initialization
# ============================================================================

# Log module load
import logging
logger = logging.getLogger(__name__)
logger.info(f"Technical Analyzer Expert v{__version__} loaded")
if TECHNICAL_ANALYZER_AVAILABLE:
    logger.info("TechnicalAnalyzer (existing) loaded successfully")
else:
    logger.warning("TechnicalAnalyzer not found - using fallback calculations")