# config.py - Master Configuration for Trading System V3
"""
Trading System V3 - Master Configuration
All system-wide settings centralized here.

This is the SINGLE SOURCE OF TRUTH for all system settings.
Each expert has its own internal config; this file controls:
- System behavior (debug, logging, directories)
- Exchange connections
- Phase thresholds (MTF, Smart Money, Risk, etc.)
- Expert agreement requirements
- Performance tracking settings
- All thresholds for ALL 13 PHASES

Version: 3.0
Author: Trading System V3
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

# Debug mode - enables verbose logging
DEBUG = True

# Vectorized operations for performance
VECTORIZED = True

# Trading mode: 'paper', 'live', 'backtest'
TRADING_MODE = "paper"

# Portfolio value in USD
PORTFOLIO_VALUE = 10000.0

# Maximum concurrent positions
MAX_POSITIONS = 5

# Maximum trades per day
MAX_DAILY_TRADES = 10

# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = "INFO"


# ============================================================================
# EXCHANGE SETTINGS
# ============================================================================

# Exchange to use
EXCHANGE = "binance"

# Binance API credentials (for live trading)
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', "")
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET', "")

# Use testnet for testing (recommended)
BINANCE_USE_TESTNET = True

# API type: 'spot' or 'future'
BINANCE_API_TYPE = "spot"

# Rate limiting
ENABLE_RATE_LIMIT = True
CACHE_TTL_SECONDS = 60

# Large order detection (for order book analysis)
LARGE_ORDER_USDT = 50000


# ============================================================================
# TIMEFRAME SETTINGS
# ============================================================================

# Primary trading timeframe
PRIMARY_TIMEFRAME = "5m"

# Data lookback (number of candles)
LOOKBACK_PERIOD = 500

# Minimum candles required for analysis
MIN_CANDLES_REQUIRED = 200

# Maximum candles to fetch
MAX_CANDLES_TO_FETCH = 1000

# Minutes per candle (for timing calculations)
MINUTES_PER_CANDLE = 5


# ============================================================================
# COIN SELECTION (Phase 1.2)
# ============================================================================

# Always include these coins in analysis
TARGET_COINS = ["BTC/USDT", "ETH/USDT"]

# Minimum 24h volume in USD
COIN_MIN_VOLUME_USDT = 1_000_00

# Price range filters
COIN_MAX_PRICE = 10000000
COIN_MIN_PRICE = 0.000001

# Maximum spread (percentage)
COIN_MAX_SPREAD = 0.0001  # 1%

# Maximum number of coins to return
COIN_MAX_RESULTS = 2000

# Cache coin list for this many minutes
COIN_CACHE_MINUTES = 5

# Rescan for new coins every X hours
COIN_SCAN_HOURS = 1

# Always include TARGET_COINS in analysis
COIN_ALWAYS_INCLUDE_TARGETS = True

# Log all coins during scan (True) or only every 10th (False)
COIN_LOG_ALL = False




# ============================================================================
# ADD THIS SECTION (after EXPERT_INITIAL_WEIGHTS)
# ============================================================================

# Expert names (must match module names)
EXPERT_NAMES = ['pattern_v3', 'price_action', 'smc', 'technical', 'strategy']

# Expert confidence thresholds
EXPERT_MIN_CONFIDENCE = 0.60
EXPERT_MIN_GRADE = "C"

# ============================================================================
# ADD THIS SECTION (after MTF settings)
# ============================================================================

# Dynamic indicators based on market regime
ENABLE_DYNAMIC_INDICATORS = True
REGIME_INDICATOR_MULTIPLIERS = {
    'trending': 1.2,
    'ranging': 0.8,
    'volatile': 1.5,
    'quiet': 0.7
}


# ============================================================================
# ADD THIS SECTION (after Phase 10 settings)
# ============================================================================

# Signal limits
SIGNAL_EXPIRY_MINUTES = 120
MAX_SIGNALS_PER_COIN_PER_DAY = 3

# ============================================================================
# ADD THIS SECTION (after Phase 11 settings)
# ============================================================================

# Timing prediction limits
TIMING_MAX_CANDLES = 48
TIMING_MIN_CANDLES = 1
TIMING_PREDICTION_CONFIDENCE = 0.65









# ============================================================================
# PHASE 4: EXPERT AGGREGATOR SETTINGS
# ============================================================================

# Minimum number of experts that must agree on direction (1-5)
MIN_EXPERTS_TO_AGREE = 3

# Minimum number of experts that must generate a signal (not HOLD)
MIN_EXPERTS_WITH_SIGNAL = 3

# Minimum consensus confidence to consider signal (0-1)
MIN_EXPERT_CONSENSUS_CONFIDENCE = 0.65

# Expert initial weights (will be updated dynamically by performance)
EXPERT_INITIAL_WEIGHTS = {
    'pattern_v3': 1.25,
    'price_action': 1.20,
    'smc': 1.30,
    'technical': 1.15,
    'strategy': 1.10
}

# Weight limits
EXPERT_MIN_WEIGHT = 0.5
EXPERT_MAX_WEIGHT = 2.0

# Target performance score (win_rate × avg_rr)
EXPERT_TARGET_SCORE = 0.75

# Minimum trades before updating weights
EXPERT_UPDATE_AFTER_TRADES = 20

# TP level configuration
TP_PERCENTAGES = [0.25, 0.35, 0.25, 0.10, 0.05]  # Sum to 1.0
TP_DESCRIPTIONS = [
    "Conservative target (partial)",
    "Primary target (partial)",
    "Main target (partial)",
    "Extended target (partial)",
    "Aggressive target (partial)"
]
TP_CLUSTER_TOLERANCE = 0.005  # 0.5% tolerance for clustering
MAX_TP_DISTANCE_PCT = 0.15    # 15% max distance from entry


# ============================================================================
# PHASE 5: MTF CONFIRMATION SETTINGS
# ============================================================================

# Higher timeframes to analyze
MTF_HIGHER_TIMEFRAMES = ["15m", "1h", "4h"]

# Minimum MTF score to confirm signal (0-1)
MTF_MIN_SCORE = 0.70

# Maximum confidence boost from MTF alignment
MTF_MAX_BOOST = 0.15

# Maximum confidence penalty from MTF conflict
MTF_MAX_PENALTY = -0.15

# Alignment thresholds
MTF_STRONG_ALIGNMENT = 0.80   # Strong alignment = max boost
MTF_WEAK_ALIGNMENT = 0.50     # Weak alignment = no boost

# Enable pullback detection
MTF_ENABLE_PULLBACK = True

# Cache TTL for HTF results (seconds)
MTF_CACHE_TTL = 60

# Timeframe weights for MTF aggregation
MTF_TIMEFRAME_WEIGHTS = {
    "15m": 0.5,
    "1h": 1.0,
    "4h": 1.5,
    "1d": 2.0
}

# Pullback detection thresholds
MTF_PULLBACK_DEPTH_MAX = 0.05           # Max pullback depth (5%)
MTF_PULLBACK_VOLUME_THRESHOLD = 0.8     # Volume threshold for healthy pullback
MTF_PULLBACK_RSI_THRESHOLD = 10         # RSI movement threshold


# ============================================================================
# PHASE 6: SMART MONEY FILTER SETTINGS
# ============================================================================

# Minimum smart money score to confirm (0-1)
SMART_MONEY_MIN_SCORE = 0.60

# Enable/disable smart money components
ENABLE_LIQUIDITY_ANALYSIS = True
ENABLE_ORDERFLOW_ANALYSIS = True
ENABLE_MICROSTRUCTURE_ANALYSIS = True
ENABLE_MARKET_MAKER_ANALYSIS = True
SM_ENABLE_LIQUIDATION_ANALYSIS = True
SM_ENABLE_REGIME_ALIGNMENT = True

# Smart Money component weights (must sum to 1.0)
SM_WEIGHT_LIQUIDITY = 0.20
SM_WEIGHT_ORDERFLOW = 0.20
SM_WEIGHT_MARKET_STRUCTURE = 0.20
SM_WEIGHT_MICROSTRUCTURE = 0.15
SM_WEIGHT_LIQUIDATIONS = 0.10
SM_WEIGHT_MARKET_MAKER = 0.10
SM_WEIGHT_REGIME_ALIGNMENT = 0.05

# Liquidity detection thresholds
SM_SWEEP_STRENGTH_THRESHOLD = 0.6
SM_ENABLE_INDUCEMENT_DETECTION = True
SM_ENABLE_SWEEP_FAILURE_DETECTION = True

# Order flow thresholds
SM_ABSORPTION_VOLUME_THRESHOLD = 1.5
SM_EXHAUSTION_VOLUME_THRESHOLD = 2.0

# Microstructure thresholds
SM_FVG_STRENGTH_THRESHOLD = 0.6
SM_DISPLACEMENT_STRENGTH_THRESHOLD = 0.7

# Liquidation analysis thresholds
SM_LIQUIDATION_VALUE_THRESHOLD = 100000      # $100K
SM_LIQUIDATION_LOOKBACK_MINUTES = 60
SM_LIQUIDATION_CLUSTER_PERCENT = 0.01        # 1% cluster tolerance
SM_CASCADE_RISK_THRESHOLD = 0.5


# ============================================================================
# PHASE 7: LIGHT CONFIRMATIONS SETTINGS
# ============================================================================

# Enable/disable light confirmation components
ENABLE_CROSS_ASSET = True
ENABLE_FUNDING_OI = True
ENABLE_SENTIMENT = True
ENABLE_CORRELATION = True
ENABLE_VOLUME_ANALYSIS = True

# Light confirm component weights (must sum to 1.0)
LIGHT_CONFIRM_WEIGHTS = {
    'cross_asset': 0.25,
    'funding_oi': 0.25,
    'sentiment': 0.20,
    'correlation': 0.15,
    'volume': 0.15
}

# Minimum light confirm score to pass (0-1)
LIGHT_CONFIRM_MIN_SCORE = 0.55

# Fear & Greed thresholds
FEAR_GREED_EXTREME_FEAR = 25
FEAR_GREED_FEAR = 40
FEAR_GREED_NEUTRAL = 60
FEAR_GREED_GREED = 75

# Funding rate thresholds (8-hour rate)
FUNDING_EXTREME_POSITIVE = 0.01    # 1% - extreme bullish sentiment
FUNDING_HIGH_POSITIVE = 0.005      # 0.5% - high bullish sentiment
FUNDING_EXTREME_NEGATIVE = -0.01   # -1% - extreme bearish sentiment
FUNDING_HIGH_NEGATIVE = -0.005     # -0.5% - high bearish sentiment

# Open Interest thresholds
OI_SURGE_THRESHOLD = 0.20          # 20% increase = surge
OI_DECLINE_THRESHOLD = -0.15       # 15% decrease = decline


# ============================================================================
# PHASE 8: RISK MANAGEMENT SETTINGS
# ============================================================================

# ----- RISK/REWARD -----
MIN_RISK_REWARD = 1.5
TARGET_RISK_REWARD = 2.5

# ----- POSITION SIZING -----
MAX_RISK_PER_TRADE = 0.02          # 2% of portfolio
MAX_POSITION_CONCENTRATION = 0.25   # 25% max in one position

# ----- ENTRY ZONE (Multi-Method) -----
ENTRY_ZONE_ATR_MULTIPLIER = 0.5
ENTRY_ZONE_WEIGHTS = {
    'smc_order_block': 0.25,
    'smc_fvg': 0.20,
    'smc_liquidity': 0.15,
    'indicator_bollinger': 0.10,
    'indicator_vwap': 0.10,
    'structure_sr': 0.10,
    'volume_profile': 0.10
}

# ----- STOP LOSS (Priority Hierarchy) -----
# Stop loss priority order
STOP_PRIORITY = [
    'order_block',      # Priority 1: SMC Order Blocks
    'fair_value_gap',   # Priority 2: Fair Value Gaps
    'liquidity_sweep',  # Priority 3: Liquidity Sweeps
    'swing_point',      # Priority 4: Swing Points
    'support_resistance', # Priority 5: S/R Levels
    'advanced_indicator', # Priority 6: Keltner, Donchian, PSAR, ATR Trailing
    'volume_profile',   # Priority 7: Volume Profile POC
    'atr_fallback'      # Priority 8: Advanced ATR Fallback
]

# Advanced indicators for stop loss
ENABLE_ADVANCED_STOP_INDICATORS = True
STOP_INDICATORS = ['keltner', 'donchian', 'psar', 'atr_trailing']

# Stop loss ATR multipliers
ATR_STOP_MULTIPLIER = 1.5
ATR_LOW_VOL_MULTIPLIER = 1.2
ATR_HIGH_VOL_MULTIPLIER = 2.0
ATR_EXTREME_VOL_MULTIPLIER = 2.5

# Stop loss buffer (percentage)
STOP_BUFFER_PCT = 0.002             # 0.2%

# Maximum/minimum stop distance (percentage of entry)
MAX_STOP_DISTANCE_PCT = 0.05        # 5%
MIN_STOP_DISTANCE_PCT = 0.002       # 0.2%

# ----- TAKE PROFIT (Priority Hierarchy) -----
TP_PRIORITY = [
    'htf_level',        # Priority 1: Higher timeframe levels
    'liquidity_cluster', # Priority 2: Liquidity clusters
    'fibonacci',        # Priority 3: Fibonacci extensions
    'market_structure', # Priority 4: Swing points
    'indicator',        # Priority 5: Bollinger/Keltner opposite band
    'rr_based'          # Priority 6: Dynamic risk/reward
]

# ATR TP multiplier
ATR_TP_MULTIPLIER = 3.0
MAX_TP_DISTANCE_PCT = 0.15          # 15% max

# ----- KELLY CRITERION -----
ENABLE_KELLY = True
KELLY_FRACTION = 0.25               # Use 25% of full Kelly

# ----- PORTFOLIO HEAT -----
MAX_PORTFOLIO_HEAT = 80             # 0-100 scale
MAX_SECTOR_EXPOSURE = 0.5           # 50% max per sector
MAX_PORTFOLIO_EXPOSURE = 2.0        # 200% max total exposure

# ----- CORRELATION RISK -----
CORRELATION_RISK_THRESHOLD = 0.7
CORRELATION_REDUCTION = 0.5

# ----- TRADE PROTECTION -----
ENABLE_TRAILING_STOP = True
ENABLE_PARTIAL_TP = True
ENABLE_BREAKEVEN = True
ENABLE_TIME_EXIT = True

# Trailing stop settings
TRAILING_STOP_ACTIVATION = 1.5      # Activate after 1.5x risk in profit
TRAILING_STOP_ATR_MULTIPLIER = 2.0

# Breakeven settings
BREAKEVEN_TRIGGER = 1.0             # Move to breakeven after 1x risk in profit

# Time exit (max hold bars)
MAX_HOLD_BARS = 48                  # 48 bars (4 hours on 5m)

# Adverse excursion
ADVERSE_EXCURSION_LIMIT = 0.03      # 3% max adverse movement


# ============================================================================
# PHASE 9: FINAL SCORING SETTINGS
# ============================================================================

# Final scoring weights (must sum to 1.0)
FINAL_WEIGHT_CONSENSUS = 0.25        # Phase 4: Expert consensus
FINAL_WEIGHT_MTF = 0.15             # Phase 5: MTF confirmation
FINAL_WEIGHT_SMART_MONEY = 0.15     # Phase 6: Smart Money
FINAL_WEIGHT_LIGHT_CONFIRM = 0.10   # Phase 7: Light confirmations
FINAL_WEIGHT_RISK_REWARD = 0.10     # Phase 8: Risk/Reward
FINAL_WEIGHT_POSITION_SIZING = 0.05 # Phase 8: Position size
FINAL_WEIGHT_STRATEGY_AGREEMENT = 0.10 # Phase 4: Strategy agreement
FINAL_WEIGHT_TIMING = 0.10          # Phase 11: Timing prediction

# Minimum probability to generate signal (0-1)
FINAL_MIN_PROBABILITY = 0.65

# Minimum agreement ratio to consider (0-1)
FINAL_MIN_AGREEMENT = 0.55

# Grade thresholds (score to grade mapping)
FINAL_GRADE_A_PLUS = 0.92
FINAL_GRADE_A = 0.85
FINAL_GRADE_B_PLUS = 0.78
FINAL_GRADE_B = 0.72
FINAL_GRADE_B_MINUS = 0.65
FINAL_GRADE_C_PLUS = 0.60
FINAL_GRADE_C = 0.55
FINAL_GRADE_D = 0.50
FINAL_GRADE_F = 0.00

# Position multipliers by grade
POSITION_MULTIPLIER_A_PLUS = 1.5
POSITION_MULTIPLIER_A = 1.3
POSITION_MULTIPLIER_B_PLUS = 1.1
POSITION_MULTIPLIER_B = 1.0
POSITION_MULTIPLIER_B_MINUS = 0.9
POSITION_MULTIPLIER_C_PLUS = 0.75
POSITION_MULTIPLIER_C = 0.5
POSITION_MULTIPLIER_D = 0.25
POSITION_MULTIPLIER_F = 0.0

# Bayesian adjustment
FINAL_ENABLE_BAYESIAN = True
FINAL_BAYESIAN_WEIGHT = 0.30        # 30% weight on Bayesian adjustment
FINAL_MAX_HISTORY = 200             # Maximum trade history to keep


# ============================================================================
# PHASE 10: SIGNAL VALIDATOR SETTINGS
# ============================================================================

# Validation thresholds
VALIDATOR_MIN_PROBABILITY = 0.65    # Same as FINAL_MIN_PROBABILITY
VALIDATOR_MIN_GRADE = "C"           # Minimum grade to accept

# Cooldown settings
SYMBOL_COOLDOWN_MINUTES = 60         # Cooldown per symbol
MAX_SIGNAL_AGE_MINUTES = 120         # Signal expires after 2 hours

# Time filter (skip low-liquidity hours)
TIME_FILTER_ENABLED = True

# Fakeout detection
ENABLE_FAKEOUT_DETECTION = True
FAKEOUT_WICK_THRESHOLD = 0.015       # 1.5% wick for fakeout
FAKEOUT_VOLUME_THRESHOLD = 1.5       # 1.5x volume for confirmation

# Liquidity trap detection
ENABLE_LIQUIDITY_TRAP_DETECTION = True
LIQUIDITY_TRAP_DISTANCE = 0.01       # 1% distance to liquidity

# Stop hunt detection
ENABLE_STOP_HUNT_DETECTION = True
STOP_HUNT_ATR_MULTIPLE = 1.5         # Wick > 1.5x ATR = stop hunt


# ============================================================================
# PHASE 11: TIMING PREDICTOR SETTINGS
# ============================================================================

# Model weights for timing ensemble
TIMING_MODEL_WEIGHTS = {
    'atr': 0.30,
    'velocity': 0.30,
    'regime': 0.20,
    'volume': 0.20
}

# Timing prediction lookbacks
TIMING_ATR_LOOKBACK = 20
TIMING_VELOCITY_LOOKBACK = 10
TIMING_VOLUME_LOOKBACK = 20

# Confidence thresholds for timing
TIMING_HIGH_CONFIDENCE = 0.70
TIMING_MODERATE_CONFIDENCE = 0.50


# ============================================================================
# PHASE 12: SIGNAL OUTPUT SETTINGS
# ============================================================================

# Summary formats to generate
ENABLE_TXT_SUMMARY = True
ENABLE_CSV_SUMMARY = True
ENABLE_MD_SUMMARY = True

# Maximum items in summaries
MAX_KEY_POINTS = 8
MAX_WARNINGS = 5
MAX_STORY_LENGTH = 500


# ============================================================================
# PHASE 13: PERFORMANCE TRACKING SETTINGS
# ============================================================================

# Enable performance tracking
ENABLE_PERFORMANCE_TRACKING = True

# Trade file retention (days)
PERFORMANCE_RETENTION_DAYS = 90

# Max check hours for trade outcome (7 days)
MAX_CHECK_HOURS = 168

# Outcome classification
OUTCOME_WIN = "WIN"
OUTCOME_PARTIAL_WIN = "PARTIAL_WIN"
OUTCOME_LOSS = "LOSS"
OUTCOME_NEVER_ENTERED_TP_HIT = "NEVER_ENTERED_TP_HIT"
OUTCOME_NEVER_ENTERED_SL_HIT = "NEVER_ENTERED_SL_HIT"
OUTCOME_NEVER_ENTERED_NO_HIT = "NEVER_ENTERED_NO_HIT"
OUTCOME_EXPIRED = "EXPIRED"
OUTCOME_PENDING = "PENDING"

# Performance report settings
ENABLE_DAILY_REPORTS = True
ENABLE_WEEKLY_REPORTS = True
ENABLE_MONTHLY_REPORTS = True


# ============================================================================
# DATA FETCHER SETTINGS
# ============================================================================

# Enable caching
ENABLE_CACHING = True

# Phase 5 data fetcher settings (for light confirmations)
PHASE5_BINANCE_API_KEY = os.environ.get('PHASE5_BINANCE_API_KEY', BINANCE_API_KEY)
PHASE5_BINANCE_API_SECRET = os.environ.get('PHASE5_BINANCE_API_SECRET', BINANCE_API_SECRET)
PHASE5_USE_FUTURES = True
PHASE5_CACHE_TTL = 60


# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Base directories (relative to project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_DIR = os.path.join(BASE_DIR, 'signals')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PERFORMANCE_DIR = os.path.join(SIGNALS_DIR, 'performance')

# Subdirectories
SIGNALS_RAW_DIR = os.path.join(SIGNALS_DIR, 'raw')
SIGNALS_CONFIRMED_DIR = os.path.join(SIGNALS_DIR, 'confirmed')
SIGNALS_FINAL_DIR = os.path.join(SIGNALS_DIR, 'final')
SIGNALS_SUMMARY_DIR = os.path.join(SIGNALS_DIR, 'summary')
PERFORMANCE_DAILY_DIR = os.path.join(PERFORMANCE_DIR, 'daily')
PERFORMANCE_MONTHLY_DIR = os.path.join(PERFORMANCE_DIR, 'monthly')
PERFORMANCE_TRADES_DIR = os.path.join(PERFORMANCE_DIR, 'trades')
PERFORMANCE_EXPERTS_DIR = os.path.join(PERFORMANCE_DIR, 'experts')

# Create directories if they don't exist
for dir_path in [SIGNALS_RAW_DIR, SIGNALS_CONFIRMED_DIR, SIGNALS_FINAL_DIR,
                 SIGNALS_SUMMARY_DIR, PERFORMANCE_DAILY_DIR, PERFORMANCE_MONTHLY_DIR,
                 PERFORMANCE_TRADES_DIR, PERFORMANCE_EXPERTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of key configuration values"""
    return {
        'system': {
            'debug': DEBUG,
            'trading_mode': TRADING_MODE,
            'portfolio_value': PORTFOLIO_VALUE,
            'max_positions': MAX_POSITIONS
        },
        'timeframe': {
            'primary': PRIMARY_TIMEFRAME,
            'higher_timeframes': MTF_HIGHER_TIMEFRAMES
        },
        'expert_agreement': {
            'min_experts_to_agree': MIN_EXPERTS_TO_AGREE,
            'min_experts_with_signal': MIN_EXPERTS_WITH_SIGNAL,
            'min_consensus_confidence': MIN_EXPERT_CONSENSUS_CONFIDENCE
        },
        'risk': {
            'min_risk_reward': MIN_RISK_REWARD,
            'max_risk_per_trade': f"{MAX_RISK_PER_TRADE:.1%}",
            'max_position_concentration': f"{MAX_POSITION_CONCENTRATION:.1%}"
        },
        'scoring': {
            'min_probability': FINAL_MIN_PROBABILITY,
            'min_agreement': FINAL_MIN_AGREEMENT
        },
        'cooldown': {
            'symbol_cooldown_minutes': SYMBOL_COOLDOWN_MINUTES,
            'max_signal_age_minutes': MAX_SIGNAL_AGE_MINUTES
        }
    }


def validate_config() -> List[str]:
    """Validate configuration and return any issues"""
    issues = []
    
    # Check weights sum to 1.0
    light_confirm_sum = sum(LIGHT_CONFIRM_WEIGHTS.values())
    if abs(light_confirm_sum - 1.0) > 0.01:
        issues.append(f"Light confirm weights sum to {light_confirm_sum:.2f}, should be 1.0")
    
    final_weights = [
        FINAL_WEIGHT_CONSENSUS, FINAL_WEIGHT_MTF, FINAL_WEIGHT_SMART_MONEY,
        FINAL_WEIGHT_LIGHT_CONFIRM, FINAL_WEIGHT_RISK_REWARD, FINAL_WEIGHT_POSITION_SIZING,
        FINAL_WEIGHT_STRATEGY_AGREEMENT, FINAL_WEIGHT_TIMING
    ]
    final_sum = sum(final_weights)
    if abs(final_sum - 1.0) > 0.01:
        issues.append(f"Final scoring weights sum to {final_sum:.2f}, should be 1.0")
    
    entry_zone_sum = sum(ENTRY_ZONE_WEIGHTS.values())
    if abs(entry_zone_sum - 1.0) > 0.01:
        issues.append(f"Entry zone weights sum to {entry_zone_sum:.2f}, should be 1.0")
    
    sm_weights = [
        SM_WEIGHT_LIQUIDITY, SM_WEIGHT_ORDERFLOW, SM_WEIGHT_MARKET_STRUCTURE,
        SM_WEIGHT_MICROSTRUCTURE, SM_WEIGHT_LIQUIDATIONS, SM_WEIGHT_MARKET_MAKER,
        SM_WEIGHT_REGIME_ALIGNMENT
    ]
    sm_sum = sum(sm_weights)
    if abs(sm_sum - 1.0) > 0.01:
        issues.append(f"Smart Money weights sum to {sm_sum:.2f}, should be 1.0")
    
    timing_sum = sum(TIMING_MODEL_WEIGHTS.values())
    if abs(timing_sum - 1.0) > 0.01:
        issues.append(f"Timing model weights sum to {timing_sum:.2f}, should be 1.0")
    
    # Check min_experts_to_agree range
    if not 1 <= MIN_EXPERTS_TO_AGREE <= 5:
        issues.append(f"MIN_EXPERTS_TO_AGREE must be between 1 and 5, got {MIN_EXPERTS_TO_AGREE}")
    
    # Check thresholds
    if MIN_RISK_REWARD < 1.0:
        issues.append(f"MIN_RISK_REWARD should be >= 1.0, got {MIN_RISK_REWARD}")
    
    if FINAL_MIN_PROBABILITY < 0.5:
        issues.append(f"FINAL_MIN_PROBABILITY should be >= 0.5, got {FINAL_MIN_PROBABILITY}")
    
    # Check TP percentages sum to 1.0
    tp_sum = sum(TP_PERCENTAGES)
    if abs(tp_sum - 1.0) > 0.01:
        issues.append(f"TP_PERCENTAGES sum to {tp_sum:.2f}, should be 1.0")
    
    return issues


# ============================================================================
# CONFIG CLASS (for backward compatibility with existing code)
# ============================================================================

class Config:
    """Configuration class that exposes all settings as attributes"""
    
    # System settings
    DEBUG = DEBUG
    VECTORIZED = VECTORIZED
    TRADING_MODE = TRADING_MODE
    PORTFOLIO_VALUE = PORTFOLIO_VALUE
    MAX_POSITIONS = MAX_POSITIONS
    MAX_DAILY_TRADES = MAX_DAILY_TRADES
    LOG_LEVEL = LOG_LEVEL
    
    # Exchange settings
    EXCHANGE = EXCHANGE
    BINANCE_API_KEY = BINANCE_API_KEY
    BINANCE_API_SECRET = BINANCE_API_SECRET
    BINANCE_USE_TESTNET = BINANCE_USE_TESTNET
    BINANCE_API_TYPE = BINANCE_API_TYPE
    ENABLE_RATE_LIMIT = ENABLE_RATE_LIMIT
    CACHE_TTL_SECONDS = CACHE_TTL_SECONDS
    LARGE_ORDER_USDT = LARGE_ORDER_USDT
    
    # Timeframe settings
    PRIMARY_TIMEFRAME = PRIMARY_TIMEFRAME
    LOOKBACK_PERIOD = LOOKBACK_PERIOD
    MIN_CANDLES_REQUIRED = MIN_CANDLES_REQUIRED
    MAX_CANDLES_TO_FETCH = MAX_CANDLES_TO_FETCH
    MINUTES_PER_CANDLE = MINUTES_PER_CANDLE
    
    # Coin selection
    TARGET_COINS = TARGET_COINS
    COIN_MIN_VOLUME_USDT = COIN_MIN_VOLUME_USDT
    COIN_MAX_PRICE = COIN_MAX_PRICE
    COIN_MIN_PRICE = COIN_MIN_PRICE
    COIN_MAX_SPREAD = COIN_MAX_SPREAD
    COIN_MAX_RESULTS = COIN_MAX_RESULTS
    COIN_CACHE_MINUTES = COIN_CACHE_MINUTES
    COIN_SCAN_HOURS = COIN_SCAN_HOURS
    COIN_ALWAYS_INCLUDE_TARGETS = COIN_ALWAYS_INCLUDE_TARGETS
    COIN_LOG_ALL = COIN_LOG_ALL
    
    # Phase 4: Expert Aggregator
    MIN_EXPERTS_TO_AGREE = MIN_EXPERTS_TO_AGREE
    MIN_EXPERTS_WITH_SIGNAL = MIN_EXPERTS_WITH_SIGNAL
    MIN_EXPERT_CONSENSUS_CONFIDENCE = MIN_EXPERT_CONSENSUS_CONFIDENCE
    EXPERT_INITIAL_WEIGHTS = EXPERT_INITIAL_WEIGHTS
    EXPERT_MIN_WEIGHT = EXPERT_MIN_WEIGHT
    EXPERT_MAX_WEIGHT = EXPERT_MAX_WEIGHT
    EXPERT_TARGET_SCORE = EXPERT_TARGET_SCORE
    EXPERT_UPDATE_AFTER_TRADES = EXPERT_UPDATE_AFTER_TRADES
    TP_PERCENTAGES = TP_PERCENTAGES
    TP_DESCRIPTIONS = TP_DESCRIPTIONS
    TP_CLUSTER_TOLERANCE = TP_CLUSTER_TOLERANCE
    MAX_TP_DISTANCE_PCT = MAX_TP_DISTANCE_PCT
    
    # Phase 5: MTF
    MTF_HIGHER_TIMEFRAMES = MTF_HIGHER_TIMEFRAMES
    MTF_MIN_SCORE = MTF_MIN_SCORE
    MTF_MAX_BOOST = MTF_MAX_BOOST
    MTF_MAX_PENALTY = MTF_MAX_PENALTY
    MTF_STRONG_ALIGNMENT = MTF_STRONG_ALIGNMENT
    MTF_WEAK_ALIGNMENT = MTF_WEAK_ALIGNMENT
    MTF_ENABLE_PULLBACK = MTF_ENABLE_PULLBACK
    MTF_CACHE_TTL = MTF_CACHE_TTL
    MTF_TIMEFRAME_WEIGHTS = MTF_TIMEFRAME_WEIGHTS
    MTF_PULLBACK_DEPTH_MAX = MTF_PULLBACK_DEPTH_MAX
    MTF_PULLBACK_VOLUME_THRESHOLD = MTF_PULLBACK_VOLUME_THRESHOLD
    MTF_PULLBACK_RSI_THRESHOLD = MTF_PULLBACK_RSI_THRESHOLD
    
    # Phase 6: Smart Money
    SMART_MONEY_MIN_SCORE = SMART_MONEY_MIN_SCORE
    ENABLE_LIQUIDITY_ANALYSIS = ENABLE_LIQUIDITY_ANALYSIS
    ENABLE_ORDERFLOW_ANALYSIS = ENABLE_ORDERFLOW_ANALYSIS
    ENABLE_MICROSTRUCTURE_ANALYSIS = ENABLE_MICROSTRUCTURE_ANALYSIS
    ENABLE_MARKET_MAKER_ANALYSIS = ENABLE_MARKET_MAKER_ANALYSIS
    SM_ENABLE_LIQUIDATION_ANALYSIS = SM_ENABLE_LIQUIDATION_ANALYSIS
    SM_ENABLE_REGIME_ALIGNMENT = SM_ENABLE_REGIME_ALIGNMENT
    SM_WEIGHT_LIQUIDITY = SM_WEIGHT_LIQUIDITY
    SM_WEIGHT_ORDERFLOW = SM_WEIGHT_ORDERFLOW
    SM_WEIGHT_MARKET_STRUCTURE = SM_WEIGHT_MARKET_STRUCTURE
    SM_WEIGHT_MICROSTRUCTURE = SM_WEIGHT_MICROSTRUCTURE
    SM_WEIGHT_LIQUIDATIONS = SM_WEIGHT_LIQUIDATIONS
    SM_WEIGHT_MARKET_MAKER = SM_WEIGHT_MARKET_MAKER
    SM_WEIGHT_REGIME_ALIGNMENT = SM_WEIGHT_REGIME_ALIGNMENT
    SM_SWEEP_STRENGTH_THRESHOLD = SM_SWEEP_STRENGTH_THRESHOLD
    SM_ENABLE_INDUCEMENT_DETECTION = SM_ENABLE_INDUCEMENT_DETECTION
    SM_ENABLE_SWEEP_FAILURE_DETECTION = SM_ENABLE_SWEEP_FAILURE_DETECTION
    SM_ABSORPTION_VOLUME_THRESHOLD = SM_ABSORPTION_VOLUME_THRESHOLD
    SM_EXHAUSTION_VOLUME_THRESHOLD = SM_EXHAUSTION_VOLUME_THRESHOLD
    SM_FVG_STRENGTH_THRESHOLD = SM_FVG_STRENGTH_THRESHOLD
    SM_DISPLACEMENT_STRENGTH_THRESHOLD = SM_DISPLACEMENT_STRENGTH_THRESHOLD
    SM_LIQUIDATION_VALUE_THRESHOLD = SM_LIQUIDATION_VALUE_THRESHOLD
    SM_LIQUIDATION_LOOKBACK_MINUTES = SM_LIQUIDATION_LOOKBACK_MINUTES
    SM_LIQUIDATION_CLUSTER_PERCENT = SM_LIQUIDATION_CLUSTER_PERCENT
    SM_CASCADE_RISK_THRESHOLD = SM_CASCADE_RISK_THRESHOLD
    
    # Phase 7: Light Confirmations
    ENABLE_CROSS_ASSET = ENABLE_CROSS_ASSET
    ENABLE_FUNDING_OI = ENABLE_FUNDING_OI
    ENABLE_SENTIMENT = ENABLE_SENTIMENT
    ENABLE_CORRELATION = ENABLE_CORRELATION
    ENABLE_VOLUME_ANALYSIS = ENABLE_VOLUME_ANALYSIS
    LIGHT_CONFIRM_WEIGHTS = LIGHT_CONFIRM_WEIGHTS
    LIGHT_CONFIRM_MIN_SCORE = LIGHT_CONFIRM_MIN_SCORE
    FEAR_GREED_EXTREME_FEAR = FEAR_GREED_EXTREME_FEAR
    FEAR_GREED_FEAR = FEAR_GREED_FEAR
    FEAR_GREED_NEUTRAL = FEAR_GREED_NEUTRAL
    FEAR_GREED_GREED = FEAR_GREED_GREED
    FUNDING_EXTREME_POSITIVE = FUNDING_EXTREME_POSITIVE
    FUNDING_HIGH_POSITIVE = FUNDING_HIGH_POSITIVE
    FUNDING_EXTREME_NEGATIVE = FUNDING_EXTREME_NEGATIVE
    FUNDING_HIGH_NEGATIVE = FUNDING_HIGH_NEGATIVE
    OI_SURGE_THRESHOLD = OI_SURGE_THRESHOLD
    OI_DECLINE_THRESHOLD = OI_DECLINE_THRESHOLD
    
    # Phase 8: Risk Management
    MIN_RISK_REWARD = MIN_RISK_REWARD
    TARGET_RISK_REWARD = TARGET_RISK_REWARD
    MAX_RISK_PER_TRADE = MAX_RISK_PER_TRADE
    MAX_POSITION_CONCENTRATION = MAX_POSITION_CONCENTRATION
    ENTRY_ZONE_ATR_MULTIPLIER = ENTRY_ZONE_ATR_MULTIPLIER
    ENTRY_ZONE_WEIGHTS = ENTRY_ZONE_WEIGHTS
    STOP_PRIORITY = STOP_PRIORITY
    ENABLE_ADVANCED_STOP_INDICATORS = ENABLE_ADVANCED_STOP_INDICATORS
    STOP_INDICATORS = STOP_INDICATORS
    ATR_STOP_MULTIPLIER = ATR_STOP_MULTIPLIER
    ATR_LOW_VOL_MULTIPLIER = ATR_LOW_VOL_MULTIPLIER
    ATR_HIGH_VOL_MULTIPLIER = ATR_HIGH_VOL_MULTIPLIER
    ATR_EXTREME_VOL_MULTIPLIER = ATR_EXTREME_VOL_MULTIPLIER
    STOP_BUFFER_PCT = STOP_BUFFER_PCT
    MAX_STOP_DISTANCE_PCT = MAX_STOP_DISTANCE_PCT
    MIN_STOP_DISTANCE_PCT = MIN_STOP_DISTANCE_PCT
    TP_PRIORITY = TP_PRIORITY
    ATR_TP_MULTIPLIER = ATR_TP_MULTIPLIER
    MAX_TP_DISTANCE_PCT = MAX_TP_DISTANCE_PCT
    ENABLE_KELLY = ENABLE_KELLY
    KELLY_FRACTION = KELLY_FRACTION
    MAX_PORTFOLIO_HEAT = MAX_PORTFOLIO_HEAT
    MAX_SECTOR_EXPOSURE = MAX_SECTOR_EXPOSURE
    MAX_PORTFOLIO_EXPOSURE = MAX_PORTFOLIO_EXPOSURE
    CORRELATION_RISK_THRESHOLD = CORRELATION_RISK_THRESHOLD
    CORRELATION_REDUCTION = CORRELATION_REDUCTION
    ENABLE_TRAILING_STOP = ENABLE_TRAILING_STOP
    ENABLE_PARTIAL_TP = ENABLE_PARTIAL_TP
    ENABLE_BREAKEVEN = ENABLE_BREAKEVEN
    ENABLE_TIME_EXIT = ENABLE_TIME_EXIT
    TRAILING_STOP_ACTIVATION = TRAILING_STOP_ACTIVATION
    TRAILING_STOP_ATR_MULTIPLIER = TRAILING_STOP_ATR_MULTIPLIER
    BREAKEVEN_TRIGGER = BREAKEVEN_TRIGGER
    MAX_HOLD_BARS = MAX_HOLD_BARS
    ADVERSE_EXCURSION_LIMIT = ADVERSE_EXCURSION_LIMIT
    
    # Phase 9: Final Scoring
    FINAL_WEIGHT_CONSENSUS = FINAL_WEIGHT_CONSENSUS
    FINAL_WEIGHT_MTF = FINAL_WEIGHT_MTF
    FINAL_WEIGHT_SMART_MONEY = FINAL_WEIGHT_SMART_MONEY
    FINAL_WEIGHT_LIGHT_CONFIRM = FINAL_WEIGHT_LIGHT_CONFIRM
    FINAL_WEIGHT_RISK_REWARD = FINAL_WEIGHT_RISK_REWARD
    FINAL_WEIGHT_POSITION_SIZING = FINAL_WEIGHT_POSITION_SIZING
    FINAL_WEIGHT_STRATEGY_AGREEMENT = FINAL_WEIGHT_STRATEGY_AGREEMENT
    FINAL_WEIGHT_TIMING = FINAL_WEIGHT_TIMING
    FINAL_MIN_PROBABILITY = FINAL_MIN_PROBABILITY
    FINAL_MIN_AGREEMENT = FINAL_MIN_AGREEMENT
    FINAL_GRADE_A_PLUS = FINAL_GRADE_A_PLUS
    FINAL_GRADE_A = FINAL_GRADE_A
    FINAL_GRADE_B_PLUS = FINAL_GRADE_B_PLUS
    FINAL_GRADE_B = FINAL_GRADE_B
    FINAL_GRADE_B_MINUS = FINAL_GRADE_B_MINUS
    FINAL_GRADE_C_PLUS = FINAL_GRADE_C_PLUS
    FINAL_GRADE_C = FINAL_GRADE_C
    FINAL_GRADE_D = FINAL_GRADE_D
    FINAL_GRADE_F = FINAL_GRADE_F
    POSITION_MULTIPLIER_A_PLUS = POSITION_MULTIPLIER_A_PLUS
    POSITION_MULTIPLIER_A = POSITION_MULTIPLIER_A
    POSITION_MULTIPLIER_B_PLUS = POSITION_MULTIPLIER_B_PLUS
    POSITION_MULTIPLIER_B = POSITION_MULTIPLIER_B
    POSITION_MULTIPLIER_B_MINUS = POSITION_MULTIPLIER_B_MINUS
    POSITION_MULTIPLIER_C_PLUS = POSITION_MULTIPLIER_C_PLUS
    POSITION_MULTIPLIER_C = POSITION_MULTIPLIER_C
    POSITION_MULTIPLIER_D = POSITION_MULTIPLIER_D
    POSITION_MULTIPLIER_F = POSITION_MULTIPLIER_F
    FINAL_ENABLE_BAYESIAN = FINAL_ENABLE_BAYESIAN
    FINAL_BAYESIAN_WEIGHT = FINAL_BAYESIAN_WEIGHT
    FINAL_MAX_HISTORY = FINAL_MAX_HISTORY
    
    # Phase 10: Signal Validator
    VALIDATOR_MIN_PROBABILITY = VALIDATOR_MIN_PROBABILITY
    VALIDATOR_MIN_GRADE = VALIDATOR_MIN_GRADE
    SYMBOL_COOLDOWN_MINUTES = SYMBOL_COOLDOWN_MINUTES
    MAX_SIGNAL_AGE_MINUTES = MAX_SIGNAL_AGE_MINUTES
    TIME_FILTER_ENABLED = TIME_FILTER_ENABLED
    ENABLE_FAKEOUT_DETECTION = ENABLE_FAKEOUT_DETECTION
    FAKEOUT_WICK_THRESHOLD = FAKEOUT_WICK_THRESHOLD
    FAKEOUT_VOLUME_THRESHOLD = FAKEOUT_VOLUME_THRESHOLD
    ENABLE_LIQUIDITY_TRAP_DETECTION = ENABLE_LIQUIDITY_TRAP_DETECTION
    LIQUIDITY_TRAP_DISTANCE = LIQUIDITY_TRAP_DISTANCE
    ENABLE_STOP_HUNT_DETECTION = ENABLE_STOP_HUNT_DETECTION
    STOP_HUNT_ATR_MULTIPLE = STOP_HUNT_ATR_MULTIPLE
    
    # Phase 11: Timing Predictor
    TIMING_MODEL_WEIGHTS = TIMING_MODEL_WEIGHTS
    TIMING_ATR_LOOKBACK = TIMING_ATR_LOOKBACK
    TIMING_VELOCITY_LOOKBACK = TIMING_VELOCITY_LOOKBACK
    TIMING_VOLUME_LOOKBACK = TIMING_VOLUME_LOOKBACK
    TIMING_HIGH_CONFIDENCE = TIMING_HIGH_CONFIDENCE
    TIMING_MODERATE_CONFIDENCE = TIMING_MODERATE_CONFIDENCE
    
    # Phase 12: Signal Output
    ENABLE_TXT_SUMMARY = ENABLE_TXT_SUMMARY
    ENABLE_CSV_SUMMARY = ENABLE_CSV_SUMMARY
    ENABLE_MD_SUMMARY = ENABLE_MD_SUMMARY
    MAX_KEY_POINTS = MAX_KEY_POINTS
    MAX_WARNINGS = MAX_WARNINGS
    MAX_STORY_LENGTH = MAX_STORY_LENGTH
    
    # Phase 13: Performance Tracking
    ENABLE_PERFORMANCE_TRACKING = ENABLE_PERFORMANCE_TRACKING
    PERFORMANCE_RETENTION_DAYS = PERFORMANCE_RETENTION_DAYS
    MAX_CHECK_HOURS = MAX_CHECK_HOURS
    OUTCOME_WIN = OUTCOME_WIN
    OUTCOME_PARTIAL_WIN = OUTCOME_PARTIAL_WIN
    OUTCOME_LOSS = OUTCOME_LOSS
    OUTCOME_NEVER_ENTERED_TP_HIT = OUTCOME_NEVER_ENTERED_TP_HIT
    OUTCOME_NEVER_ENTERED_SL_HIT = OUTCOME_NEVER_ENTERED_SL_HIT
    OUTCOME_NEVER_ENTERED_NO_HIT = OUTCOME_NEVER_ENTERED_NO_HIT
    OUTCOME_EXPIRED = OUTCOME_EXPIRED
    OUTCOME_PENDING = OUTCOME_PENDING
    ENABLE_DAILY_REPORTS = ENABLE_DAILY_REPORTS
    ENABLE_WEEKLY_REPORTS = ENABLE_WEEKLY_REPORTS
    ENABLE_MONTHLY_REPORTS = ENABLE_MONTHLY_REPORTS
    
    # Data Fetcher settings
    ENABLE_CACHING = ENABLE_CACHING
    PHASE5_BINANCE_API_KEY = PHASE5_BINANCE_API_KEY
    PHASE5_BINANCE_API_SECRET = PHASE5_BINANCE_API_SECRET
    PHASE5_USE_FUTURES = PHASE5_USE_FUTURES
    PHASE5_CACHE_TTL = PHASE5_CACHE_TTL
    
    # Directory paths
    BASE_DIR = BASE_DIR
    SIGNALS_DIR = SIGNALS_DIR
    LOGS_DIR = LOGS_DIR
    PERFORMANCE_DIR = PERFORMANCE_DIR
    SIGNALS_RAW_DIR = SIGNALS_RAW_DIR
    SIGNALS_CONFIRMED_DIR = SIGNALS_CONFIRMED_DIR
    SIGNALS_FINAL_DIR = SIGNALS_FINAL_DIR
    SIGNALS_SUMMARY_DIR = SIGNALS_SUMMARY_DIR
    PERFORMANCE_DAILY_DIR = PERFORMANCE_DAILY_DIR
    PERFORMANCE_MONTHLY_DIR = PERFORMANCE_MONTHLY_DIR
    PERFORMANCE_TRADES_DIR = PERFORMANCE_TRADES_DIR
    PERFORMANCE_EXPERTS_DIR = PERFORMANCE_EXPERTS_DIR


# ============================================================================
# VALIDATE ON IMPORT
# ============================================================================

_validation_issues = validate_config()
if _validation_issues:
    import warnings
    for issue in _validation_issues:
        warnings.warn(f"Config issue: {issue}")