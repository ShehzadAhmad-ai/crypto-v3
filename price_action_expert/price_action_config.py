"""
price_action_config.py
Configuration for Price Action Expert V3.5
All thresholds and weights are tunable without changing core logic
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
# ============================================
# CANDLE ANALYZER CONFIGURATION
# ============================================

# ATR calculation
ATR_PERIOD = 14
ATR_MULTIPLIER_WEAK = 0.3          # Body < 0.3× ATR = weak candle
ATR_MULTIPLIER_STRONG = 0.8        # Body > 0.8× ATR = strong candle

# Candle strength classification (body / range)
CANDLE_STRENGTH = {
    'marubozu': {'min_ratio': 0.90, 'strength': 0.95},
    'strong': {'min_ratio': 0.70, 'strength': 0.80},
    'moderate': {'min_ratio': 0.50, 'strength': 0.65},
    'weak': {'min_ratio': 0.30, 'strength': 0.50},
    'doji': {'max_ratio': 0.10, 'strength': 0.30},
}

# Doji threshold (body < range × threshold)
DOJI_THRESHOLD = 0.10

# ============================================
# PATTERN DETECTION CONFIGURATION
# ============================================

# Base confidence for each pattern (without confirmation)
PATTERN_BASE_CONFIDENCE = {
    # Single candle patterns
    'hammer': 0.65,
    'shooting_star': 0.65,
    'marubozu_bullish': 0.80,
    'marubozu_bearish': 0.80,
    'doji': 0.40,
    'spinning_top': 0.45,
    'long_candle_bullish': 0.70,
    'long_candle_bearish': 0.70,
    
    # Double candle patterns
    'bullish_engulfing': 0.75,
    'bearish_engulfing': 0.75,
    'inside_bar': 0.50,
    'outside_bar': 0.60,
    'tweezer_bottom': 0.70,
    'tweezer_top': 0.70,
    'harami_bullish': 0.60,
    'harami_bearish': 0.60,
    'taker_bullish': 0.65,
    'taker_bearish': 0.65,
    
    # Triple candle patterns
    'morning_star': 0.80,
    'evening_star': 0.80,
    'three_white_soldiers': 0.85,
    'three_black_crows': 0.85,
    'three_inside_up': 0.75,
    'three_inside_down': 0.75,
    
    # Multi-candle patterns
    'rising_three_methods': 0.75,
    'falling_three_methods': 0.75,
    'three_line_strike': 0.80,
    
    # Rare/institutional patterns
    'kicker_bullish': 0.85,
    'kicker_bearish': 0.85,
    'island_reversal': 0.80,
    'parabolic_blowoff': 0.80,
    'mat_hold': 0.80,
    'v_bottom': 0.75,
    'v_top': 0.75,
    
    # Trap patterns
    'bull_trap': 0.80,
    'bear_trap': 0.80,
    'stop_hunt_bullish': 0.85,
    'stop_hunt_bearish': 0.85,
    
    # Momentum patterns
    'consecutive_bullish': 0.70,
    'consecutive_bearish': 0.70,
    'momentum_shift': 0.75,
    'acceleration': 0.78,
    'deceleration': 0.60,
}

# Patterns that need next candle confirmation
PATTERNS_NEEDING_CONFIRMATION = [
    'hammer', 'shooting_star', 'inside_bar', 'doji', 'spinning_top',
    'harami_bullish', 'harami_bearish', 'tweezer_bottom', 'tweezer_top'
]

# Pattern categories for scoring
PATTERN_CATEGORIES = {
    'reversal': {
        'patterns': ['hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing',
                    'morning_star', 'evening_star', 'tweezer_bottom', 'tweezer_top',
                    'harami_bullish', 'harami_bearish', 'three_inside_up', 'three_inside_down',
                    'kicker_bullish', 'kicker_bearish', 'island_reversal', 'v_bottom', 'v_top'],
        'weight': 0.25
    },
    'continuation': {
        'patterns': ['rising_three_methods', 'falling_three_methods', 'mat_hold', 'three_line_strike'],
        'weight': 0.20
    },
    'momentum': {
        'patterns': ['marubozu_bullish', 'marubozu_bearish', 'three_white_soldiers', 'three_black_crows',
                    'consecutive_bullish', 'consecutive_bearish', 'momentum_shift', 'acceleration',
                    'long_candle_bullish', 'long_candle_bearish'],
        'weight': 0.30
    },
    'compression': {
        'patterns': ['inside_bar', 'outside_bar', 'doji', 'spinning_top'],
        'weight': 0.15
    },
    'trap': {
        'patterns': ['bull_trap', 'bear_trap', 'stop_hunt_bullish', 'stop_hunt_bearish', 'parabolic_blowoff'],
        'weight': 0.10
    }
}

# Pattern-specific detection parameters
ENGULFING_MIN_RATIO = 1.2           # Current body > previous body × 1.2
PINBAR_WICK_BODY_RATIO = 2.0        # Wick > body × 2
PINBAR_OPPOSITE_WICK_RATIO = 2.0    # Long wick > opposite wick × 2
MORNING_STAR_SECOND_BODY_MAX = 0.7  # Middle candle body < avg_body × 0.7
THREE_CANDLES_MIN_BODY_ATR = 0.5    # Min body > 0.5 × ATR
THREE_CANDLES_WICK_MAX = 0.5        # Wick < body × 0.5
MARUBOZU_WICK_MAX_PERCENT = 0.1     # Wick < body × 0.1
LONG_CANDLE_MULTIPLIER = 1.7        # Body > avg_body × 1.7
CONSECUTIVE_MIN_COUNT = 2           # 2+ candles in same direction
CONSECUTIVE_BODY_MIN_ATR = 0.6      # Each body > 0.6 × ATR
MOMENTUM_SHIFT_WINDOW = 4           # 4-candle sequence for shift detection
KICKER_GAP_PERCENT = 0.005          # 0.5% gap required for kicker
BLOWOFF_MOVE_PERCENT = 0.15         # 15% move in 10 bars for blowoff
BLOWOFF_VOLUME_RATIO = 3.0          # Volume > 3× average
V_PATTERN_MOVE_PERCENT = 0.05       # 5% move in each direction for V pattern

# ============================================
# TRAP ENGINE CONFIGURATION
# ============================================

# Trap detection
TRAP_WICK_BODY_RATIO = 1.5          # Wick > body × 1.5 for trap detection
TRAP_CLOSE_POSITION_HIGH = 0.7      # Close > 70% = strong close
TRAP_CLOSE_POSITION_LOW = 0.3       # Close < 30% = weak close
TRAP_REVERSAL_CANDLES_MAX = 2       # Reversal within 2 candles

# Stop hunt detection
STOP_HUNT_WICK_ATR_MULTIPLIER = 0.3 # Wick > 0.3 × ATR = stop hunt
STOP_HUNT_VOLUME_RATIO = 1.8        # Volume > 1.8× average

# Liquidity sweep detection
LIQUIDITY_SWEEP_DEPTH_ATR = 0.2     # Sweep depth > 0.2 × ATR
LIQUIDITY_SWEEP_MULTIPLE_COUNT = 2  # 2+ sweeps = high severity

# Trap severity thresholds
TRAP_SEVERITY = {
    'minor': {'min_score': 0.40, 'position_multiplier': 0.8, 'confidence_boost': 0.05},
    'medium': {'min_score': 0.60, 'position_multiplier': 1.0, 'confidence_boost': 0.10},
    'strong': {'min_score': 0.80, 'position_multiplier': 1.2, 'confidence_boost': 0.15},
    'extreme': {'min_score': 0.90, 'position_multiplier': 1.5, 'confidence_boost': 0.20},
}

# Trap severity weights
TRAP_SEVERITY_WEIGHTS = {
    'wick_length': 0.25,
    'reversal_speed': 0.20,
    'level_importance': 0.25,
    'volume': 0.15,
    'sweep_count': 0.15,
}

# Level importance for trap scoring (from price_location.py)
LEVEL_IMPORTANCE_SCORES = {
    '15m_swing': 0.5,
    '1h_swing': 0.7,
    '4h_swing': 0.8,
    'daily_swing': 0.9,
    'weekly_swing': 1.0,
    'multi_touch': 0.85,
}

# ============================================
# SEQUENCE ANALYZER CONFIGURATION
# ============================================

SEQUENCE_WINDOW = 10                 # Analyze last 10 candles
MICRO_PATTERN_WINDOW = 5            # Window for micro-patterns

# Micro-pattern thresholds
OUTSIDE_BAR_CONSECUTIVE = 3         # 3 outside bars = high momentum
INSIDE_BAR_COMBO_MIN = 2            # 2 inside bars = compression
BODY_EXPANSION_RATIO = 1.5          # 50% increase = expansion
VOLUME_DRYUP_RATIO = 0.5            # 50% decrease = dry up

# Micro-pattern confidence boosts
MICRO_PATTERN_BOOSTS = {
    '3_consecutive_outside_bars': 0.15,
    'inside_bar_combo_breakout': 0.20,
    'body_expansion': 0.15,
    'reversal_sequence': 0.20,
    'liquidity_run': 0.25,
    'body_contraction': -0.10,
    'alternating_bull_bear': -0.05,
}

# ============================================
# CONTEXT ENGINE CONFIGURATION
# ============================================

# Session weights (crypto optimized)
SESSION_WEIGHTS = {
    'asia': 0.60,
    'asia_london_transition': 0.70,
    'london': 0.90,
    'london_ny_overlap': 1.00,
    'new_york': 0.85,
    'london_close': 0.75,
    'late_ny': 0.65,
}

# Session hours (UTC)
SESSION_HOURS = {
    'asia': (0, 8),
    'asia_london_transition': (8, 9),
    'london': (8, 16),
    'london_ny_overlap': (13, 16),
    'new_york': (13, 22),
    'london_close': (16, 17),
    'late_ny': (22, 0),
}

# Trend stage detection (from candle sequence)
TREND_STAGE_THRESHOLDS = {
    'early': {'up_days_min': 0.70, 'wick_trend': 'decreasing'},
    'mid': {'up_days_min': 0.60, 'wick_trend': 'stable'},
    'late': {'up_days_min': 0.50, 'wick_trend': 'increasing'},
    'exhaustion': {'wick_trend': 'strong_increasing', 'range_trend': 'expanding'},
}

# Risk multipliers per trend stage
TREND_STAGE_RISK_MULTIPLIER = {
    'early': 1.2,
    'mid': 1.0,
    'late': 0.7,
    'exhaustion': 0.5,
    'consolidation': 0.8,
}

# Key level tolerance (ATR multiple)
KEY_LEVEL_TOLERANCE_ATR = 0.2

# ============================================
# SL/TP ENGINE CONFIGURATION
# ============================================

# Stop loss priority order (highest priority first)
STOP_LOSS_PRIORITY = [
    'pattern_structure',      # Pattern low for BUY, pattern high for SELL
    'market_structure',       # Recent swing low/high
    'support_resistance',     # Nearest S/R level
    'order_block',            # Unmitigated order block
    'atr_fallback',           # ATR-based (0.5× ATR)
]

# Take profit priority order
TAKE_PROFIT_PRIORITY = [
    'htf_resistance_support', # Next HTF level
    'market_structure',       # Recent swing high/low
    'pattern_projection',     # 2× risk
    'atr_fallback',           # ATR-based (2× ATR)
]

# Minimum stop distance (ATR multiple)
MIN_STOP_ATR = 0.5
STOP_BUFFER_PERCENT = 0.002         # 0.2% buffer

# Fallback risk/reward
FALLBACK_RISK_REWARD = 2.0
MAX_RISK_REWARD = 5.0

# ============================================
# SCORING ENGINE CONFIGURATION
# ============================================

# Base scoring weights
BASE_SCORING_WEIGHTS = {
    'pattern_quality': 0.20,
    'pattern_alignment': 0.05,       # Multiple patterns aligned
    'regime_alignment': 0.12,
    'structure_alignment': 0.10,
    'sr_alignment': 0.08,
    'trap_severity': 0.15,
    'sequence_confidence': 0.12,
    'volume_confirmation': 0.08,
    'liquidity_confirmation': 0.10,
}

# Dynamic weight adjustments
DYNAMIC_WEIGHT_ADJUSTMENTS = {
    'high_volatility': {
        'pattern_quality': -0.05,
        'trap_severity': 0.05,
    },
    'low_volatility': {
        'pattern_quality': 0.05,
        'trap_severity': -0.05,
    },
    'early_trend': {
        'momentum': 0.05,
        'sequence_confidence': 0.05,
    },
    'late_trend': {
        'pattern_quality': -0.05,
        'trap_severity': 0.05,
    },
}

# Grade thresholds
GRADE_THRESHOLDS = {
    'A+': 0.92,
    'A': 0.85,
    'B+': 0.78,
    'B': 0.70,
    'B-': 0.65,
    'C+': 0.60,
    'C': 0.55,
    'D': 0.00,
}

# Position multipliers per grade
GRADE_POSITION_MULTIPLIER = {
    'A+': 1.5,
    'A': 1.2,
    'B+': 1.0,
    'B': 0.8,
    'B-': 0.7,
    'C+': 0.5,
    'C': 0.3,
    'D': 0.0,
}

# Minimum thresholds
MIN_TRADE_CONFIDENCE = 0.60
MIN_RISK_REWARD = 1.2

# ============================================
# OUTPUT FORMATTING
# ============================================

MAX_PATTERN_AGE_BARS = 5
RETEST_CONFIRMATION_BARS = 2

# Pattern name mapping for output (standardized)
PATTERN_NAME_MAPPING = {
    # Single candle
    'hammer': 'Hammer',
    'shooting_star': 'Shooting_Star',
    'marubozu_bullish': 'Marubozu_Bullish',
    'marubozu_bearish': 'Marubozu_Bearish',
    'doji': 'Doji',
    'spinning_top': 'Spinning_Top',
    'long_candle_bullish': 'Long_Candle_Bullish',
    'long_candle_bearish': 'Long_Candle_Bearish',
    
    # Double candle
    'bullish_engulfing': 'Bullish_Engulfing',
    'bearish_engulfing': 'Bearish_Engulfing',
    'inside_bar': 'Inside_Bar',
    'outside_bar': 'Outside_Bar',
    'tweezer_bottom': 'Tweezer_Bottom',
    'tweezer_top': 'Tweezer_Top',
    'harami_bullish': 'Harami_Bullish',
    'harami_bearish': 'Harami_Bearish',
    'taker_bullish': 'Taker_Bullish',
    'taker_bearish': 'Taker_Bearish',
    
    # Triple candle
    'morning_star': 'Morning_Star',
    'evening_star': 'Evening_Star',
    'three_white_soldiers': 'Three_White_Soldiers',
    'three_black_crows': 'Three_Black_Crows',
    'three_inside_up': 'Three_Inside_Up',
    'three_inside_down': 'Three_Inside_Down',
    
    # Multi-candle
    'rising_three_methods': 'Rising_Three_Methods',
    'falling_three_methods': 'Falling_Three_Methods',
    'three_line_strike': 'Three_Line_Strike',
    
    # Rare
    'kicker_bullish': 'Kicker_Bullish',
    'kicker_bearish': 'Kicker_Bearish',
    'island_reversal': 'Island_Reversal',
    'parabolic_blowoff': 'Parabolic_Blowoff',
    'mat_hold': 'Mat_Hold',
    'v_bottom': 'V_Bottom',
    'v_top': 'V_Top',
    
    # Trap
    'bull_trap': 'Bull_Trap',
    'bear_trap': 'Bear_Trap',
    'stop_hunt_bullish': 'Stop_Hunt_Bullish',
    'stop_hunt_bearish': 'Stop_Hunt_Bearish',
    
    # Momentum
    'consecutive_bullish': 'Consecutive_Bullish',
    'consecutive_bearish': 'Consecutive_Bearish',
    'momentum_shift': 'Momentum_Shift',
    'acceleration': 'Acceleration',
    'deceleration': 'Deceleration',
    
    # Multi-signal
    'multi_pa_cluster': 'Multi_PA_Cluster',
}

# ============================================
# PERFORMANCE & OPTIMIZATION
# ============================================

ENABLE_CACHING = True
CACHE_TTL_SECONDS = 60

# Logging level for price action module
LOG_LEVEL = 'INFO'