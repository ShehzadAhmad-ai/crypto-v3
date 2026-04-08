# ta_config.py - Complete Advanced Technical Analyzer Configuration
"""
Complete configuration for Technical Analyzer Expert
Covers all 50+ indicators from technical_analyzer.py
All thresholds, periods, weights, and settings centralized here
"""

# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

MODULE_NAME = "technical_analyzer"
MODULE_VERSION = "2.0.0"

LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE_PATH = "logs/technical_analyzer.log"

# ============================================================================
# DATA SETTINGS
# ============================================================================

MAIN_TIMEFRAME = "5min"
MIN_DATA_CANDLES = 500
MAX_DATA_CANDLES = 1000
PREFERRED_DATA_CANDLES = 750
ALLOW_PARTIAL_INDICATORS = True
SKIP_INDICATORS_ON_ERROR = True

# ============================================================================
# HTF (HIGHER TIMEFRAME) SETTINGS
# ============================================================================

ACTIVE_HTFS = ["15min", "1h", "4h"]

HTF_WEIGHTS = {
    "10min": 0.05,
    "15min": 0.10,
    "30min": 0.15,
    "1h": 0.25,
    "4h": 0.30,
    "1d": 0.15,
}

HTF_ALIGNMENT_BOOST = 0.10
HTF_CONFLICT_PENALTY = 0.10
HTF_STRONG_ALIGNMENT_BOOST = 0.15
HTF_STRONG_CONFLICT_PENALTY = 0.15
HTF_TREND_STRONG_THRESHOLD = 0.70
HTF_TREND_WEAK_THRESHOLD = 0.40

# HTF Indicators (from technical_analyzer.py)
HTF_RSI_PERIOD = 28
HTF_ADX_PERIOD = 28
HTF_TREND_STRENGTH_MULTIPLIER = 1.5

# ============================================================================
# TRADE DECISION FILTERS
# ============================================================================

MIN_CONFIDENCE_TO_TRADE = 0.65
MIN_AGREEMENT = 0.55
MIN_VOLUME_RATIO = 0.8
MIN_RISK_REWARD = 1.5
MIN_TREND_STRENGTH = 0.30

ACTION_STRONG_ENTRY = 0.85
ACTION_ENTER_NOW = 0.70

# ============================================================================
# RSI (Relative Strength Index)
# ============================================================================

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERSOLD = 20
RSI_EXTREME_OVERBOUGHT = 80

RSI_BULL_TREND_OVERSOLD = 40
RSI_BULL_TREND_OVERBOUGHT = 85
RSI_BEAR_TREND_OVERSOLD = 20
RSI_BEAR_TREND_OVERBOUGHT = 60

RSI_TURNING_CANDLES = 2
RSI_DIVERGENCE_LOOKBACK = 20

# ============================================================================
# MACD (Moving Average Convergence Divergence)
# ============================================================================

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

MACD_TRENDING_FAST = 8
MACD_TRENDING_SLOW = 17
MACD_RANGING_FAST = 12
MACD_RANGING_SLOW = 26

MACD_HISTOGRAM_THRESHOLD = 0.0001
MACD_CROSSOVER_STRENGTH = 0.5

# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

EMA_FAST = 8
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_VERY_SLOW = 200
EMA_EXTRA = [34, 100, 144, 300]

EMA_STRONG_BULL_ALIGNMENT = "price > ema8 > ema21 > ema50 > ema200"
EMA_STRONG_BEAR_ALIGNMENT = "price < ema8 < ema21 < ema50 < ema200"
EMA_PARTIAL_ALIGNMENT_SCORE = 0.7
EMA_FAST_DISTANCE_BULLISH = 0.02
EMA_SLOW_DISTANCE_BEARISH = -0.02

# ============================================================================
# ADX (Average Directional Index)
# ============================================================================

ADX_PERIOD = 14
ADX_TRENDING = 25
ADX_STRONG_TREND = 40
ADX_WEAK_TREND = 20
ADX_EXTREME_TREND = 60
DMI_PLUS_MINUS_THRESHOLD = 5

# ============================================================================
# BOLLINGER BANDS
# ============================================================================

BB_PERIOD = 20
BB_STD = 2.0

BB_DISCOUNT_ZONE = 0.30
BB_PREMIUM_ZONE = 0.70
BB_EXTREME_DISCOUNT = 0.15
BB_EXTREME_PREMIUM = 0.85

BB_SQUEEZE_THRESHOLD = 0.04
BB_SQUEEZE_RELEASE_THRESHOLD = 0.06
BB_SQUEEZE_INTENSITY_MULTIPLIER = 2.0

BB_PERCENT_B_BULLISH = 0.2
BB_PERCENT_B_BEARISH = 0.8

# ============================================================================
# ATR (Average True Range)
# ============================================================================

ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.0

ATR_LOW_VOL_MULTIPLIER = 1.2
ATR_HIGH_VOL_MULTIPLIER = 2.0
ATR_EXTREME_VOL_MULTIPLIER = 2.5

ATR_PERCENT_LOW = 0.005
ATR_PERCENT_NORMAL = 0.015
ATR_PERCENT_HIGH = 0.03
ATR_PERCENT_EXTREME = 0.05

# ============================================================================
# VOLUME INDICATORS (Complete)
# ============================================================================

# Base Volume
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_THRESHOLD = 1.5
VOLUME_EXTREME_SPIKE = 2.5
VOLUME_CONFIRMATION = 1.2
VOLUME_DRYING_UP = 0.5
VOLUME_TREND_LOOKBACK = 10
VOLUME_TREND_STRONG_INCREASE = 1.3
VOLUME_TREND_STRONG_DECREASE = 0.7

# OBV (On-Balance Volume)
OBV_PERIOD = 20
OBV_MA_PERIOD = 20
OBV_DIVERGENCE_LOOKBACK = 20

# CMF (Chaikin Money Flow)
CMF_PERIOD = 20
CMF_BULLISH = 0.1
CMF_BEARISH = -0.1
CMF_STRONG_BULLISH = 0.2
CMF_STRONG_BEARISH = -0.2

# MFI (Money Flow Index)
MFI_PERIOD = 14
MFI_OVERBOUGHT = 80
MFI_OVERSOLD = 20
MFI_BULLISH = 40
MFI_BEARISH = 60

# ADL (Accumulation/Distribution Line)
ADL_SIGNAL_PERIOD = 9
ADL_BULLISH_CROSS = "above"
ADL_BEARISH_CROSS = "below"

# Klinger Oscillator
KLINGER_FAST = 34
KLINGER_SLOW = 55
KLINGER_SIGNAL = 13
KLINGER_BULLISH = 0
KLINGER_BEARISH = 0

# Chaikin Oscillator
CHAIKIN_FAST = 3
CHAIKIN_SLOW = 10
CHAIKIN_BULLISH = 0
CHAIKIN_BEARISH = 0

# Ease of Movement (EOM)
EOM_PERIOD = 14
EOM_SMOOTHING = 14
EOM_BULLISH = 0.1
EOM_BEARISH = -0.1

# Volume Price Trend (VPT)
VPT_SIGNAL_PERIOD = 13
VPT_DIVERGENCE_THRESHOLD = 0.03

# Negative Volume Index (NVI)
NVI_BASE = 1000
NVI_SIGNAL_PERIOD = 255
NVI_BULLISH = "nvi_above_signal"

# Positive Volume Index (PVI)
PVI_BASE = 1000
PVI_SIGNAL_PERIOD = 255

# Volume Oscillator
VOLUME_OSCILLATOR_FAST = 5
VOLUME_OSCILLATOR_SLOW = 20
VOLUME_OSCILLATOR_BULLISH = 20
VOLUME_OSCILLATOR_BEARISH = -20

# ============================================================================
# MOMENTUM INDICATORS (Complete)
# ============================================================================

# RSI (already above)
# MACD (already above)

# Stochastic
STOCH_K = 14
STOCH_D = 3
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20

# Stochastic RSI
STOCH_RSI_PERIOD = 14
STOCH_RSI_K = 3
STOCH_RSI_D = 3
STOCH_RSI_OVERBOUGHT = 80
STOCH_RSI_OVERSOLD = 20

# CCI (Commodity Channel Index)
CCI_PERIOD = 20
CCI_OVERBOUGHT = 100
CCI_OVERSOLD = -100
CCI_EXTREME = 200

# Williams %R
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80

# Ultimate Oscillator
ULTIMATE_OSCILLATOR_PERIODS = (7, 14, 28)
ULTIMATE_OSCILLATOR_OVERBOUGHT = 70
ULTIMATE_OSCILLATOR_OVERSOLD = 30

# Awesome Oscillator
AWESOME_OSCILLATOR_FAST = 5
AWESOME_OSCILLATOR_SLOW = 34
AWESOME_OSCILLATOR_SAUCER_THRESHOLD = 0.001

# ROC (Rate of Change)
ROC_FAST = 5
ROC_MEDIUM = 10
ROC_SLOW = 20
ROC_BULLISH_THRESHOLD = 0.02
ROC_BEARISH_THRESHOLD = -0.02

# TSI (True Strength Index)
TSI_FAST = 25
TSI_SLOW = 13
TSI_SIGNAL = 7
TSI_BULLISH = 0
TSI_BEARISH = 0
TSI_SIGNAL_CROSS_BULLISH = "tsi_above_signal"
TSI_SIGNAL_CROSS_BEARISH = "tsi_below_signal"

# KST (Know Sure Thing)
KST_ROC1 = 10
KST_ROC2 = 15
KST_ROC3 = 20
KST_ROC4 = 30
KST_SMA1 = 10
KST_SMA2 = 10
KST_SMA3 = 10
KST_SMA4 = 15
KST_SIGNAL = 9
KST_SIGNAL_CROSS_BULLISH = "kst_above_signal"
KST_SIGNAL_CROSS_BEARISH = "kst_below_signal"

# TRIX
TRIX_PERIOD = 15
TRIX_SIGNAL = 9
TRIX_BULLISH = 0
TRIX_BEARISH = 0
TRIX_SIGNAL_CROSS_BULLISH = "trix_above_signal"
TRIX_SIGNAL_CROSS_BEARISH = "trix_below_signal"

# Mass Index
MASS_INDEX_PERIOD = 25
MASS_INDEX_BULLISH = 26.5
MASS_INDEX_BEARISH = 27.0

# ============================================================================
# TREND INDICATORS (Complete)
# ============================================================================

# Ichimoku Cloud
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52
ICHIMOKU_CHIKOU_LOOKBACK = 26
ICHIMOKU_TWIST_STRENGTH = 0.8
ICHIMOKU_THICK_CLOUD_THRESHOLD = 0.02
ICHIMOKU_PRICE_POSITION_BULLISH = "price_above_cloud"
ICHIMOKU_PRICE_POSITION_BEARISH = "price_below_cloud"
ICHIMOKU_TWIST_BULLISH = "tenkan_above_kijun"
ICHIMOKU_TWIST_BEARISH = "tenkan_below_kijun"
ICHIMOKU_FUTURE_CLOUD_BULLISH = "senkou_a_above_senkou_b"

# Keltner Channels
KELTNER_PERIOD = 20
KELTNER_ATR_PERIOD = 10
KELTNER_MULTIPLIER = 2.0
KELTNER_BREAKOUT_THRESHOLD = 0.02
KELTNER_REVERSION_THRESHOLD = 0.01
KELTNER_BREAKOUT_BULLISH = "price_above_keltner_high"
KELTNER_BREAKOUT_BEARISH = "price_below_keltner_low"
KELTNER_REVERSION_BULLISH = "price_at_keltner_low"
KELTNER_REVERSION_BEARISH = "price_at_keltner_high"

# Donchian Channels
DONCHIAN_PERIOD = 20
DONCHIAN_BREAKOUT_BULLISH = "price_above_donchian_high"
DONCHIAN_BREAKOUT_BEARISH = "price_below_donchian_low"

# Parabolic SAR
PSAR_IAF = 0.02
PSAR_MAX_AF = 0.2
PSAR_BULLISH = "price_above_psar"
PSAR_BEARISH = "price_below_psar"
PSAR_FLIP_STRENGTH = 1.5

# Elder Ray Index
ELDER_EMA_PERIOD = 13
ELDER_BULL_POWER_BULLISH = "bull_power_positive"
ELDER_BEAR_POWER_BEARISH = "bear_power_negative"
ELDER_BUY_INDEX_BULLISH = "bull_power_positive_and_rising"

# Vortex Indicator
VORTEX_PERIOD = 14
VORTEX_BULLISH = 1.0
VORTEX_BEARISH = 1.0
VORTEX_STRONG_BULLISH = 1.2
VORTEX_STRONG_BEARISH = 0.8

# Vertical Horizontal Filter (VHF)
VHF_PERIOD = 28
VHF_TRENDING = 0.6
VHF_RANGING = 0.4

# Relative Vigor Index (RVI)
RVI_PERIOD = 10
RVI_SIGNAL_PERIOD = 4
RVI_BULLISH = 0.5
RVI_BEARISH = -0.5
RSI_BULLISH_MIN = 50      # RSI above this is considered bullish zone
RSI_BEARISH_MAX = 50

# Coppock Curve
COPPOCK_ROC1 = 14
COPPOCK_ROC2 = 11
COPPOCK_WMA = 10
COPPOCK_BULLISH = 0

# ============================================================================
# SUPPORT/RESISTANCE INDICATORS
# ============================================================================

SR_DETECTION_WINDOW = 20
SR_SENSITIVITY = 0.004
SR_MERGE_THRESHOLD = 0.001
SR_BREAKS_TO_CONFIRM = 2
SR_VOLUME_CONFIRMATION = 1.5
SR_NEAR_LEVEL_THRESHOLD = 0.01
SR_STRONG_LEVEL_THRESHOLD = 0.005

# Swing Points
SWING_WINDOW = 5
SWING_CONFIRMATION_CANDLES = 2

# Hidden Support/Resistance
HIDDEN_SR_BINS = 20
HIDDEN_SR_STRENGTH_THRESHOLD = 0.15

# Volume Profile
VOLUME_PROFILE_BINS = 50
VOLUME_PROFILE_WINDOW = 200
VALUE_AREA_VOLUME_PERCENT = 0.68
POC_STRENGTH_THRESHOLD = 0.10
VAH_VAL_DISTANCE_THRESHOLD = 0.02
VOLUME_PROFILE_UPDATE_FREQUENCY = 50

# ============================================================================
# CUMULATIVE DELTA
# ============================================================================

CUMULATIVE_DELTA_PERIOD = 20
DELTA_BULLISH = "cum_delta_above_ma"
DELTA_BEARISH = "cum_delta_below_ma"
DELTA_DIVERGENCE_THRESHOLD = 0.05

# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================

DIVERGENCE_LOOKBACK = 20
DIVERGENCE_MIN_SWING_DISTANCE = 5

DIVERGENCE_WEIGHTS = {
    "rsi": 1.0,
    "macd": 0.9,
    "obv": 0.8,
    "rsi_hidden": 0.7,
    "macd_hidden": 0.6,
    "stoch": 0.6,
    "cci": 0.5,
    "mfi": 0.5,
}

DIVERGENCE_WEAK = 0.3
DIVERGENCE_MODERATE = 0.5
DIVERGENCE_STRONG = 0.7
DIVERGENCE_VERY_STRONG = 0.9

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

REGIME_ADX_TRENDING = 30
REGIME_ADX_STRONG_TREND = 45
REGIME_BB_WIDTH_VOLATILE = 0.08
REGIME_BB_WIDTH_SQUEEZE = 0.04
REGIME_MIN_CONFIDENCE = 0.60
REGIME_STRONG_CONFIDENCE = 0.80

# Wyckoff Detection
WYCKOFF_SWING_ORDER = 10
WYCKOFF_VOLUME_THRESHOLD = 1.2
WYCKOFF_ACCUMULATION_PATTERN = ["spring", "test", "SOS", "LPS"]
WYCKOFF_DISTRIBUTION_PATTERN = ["upthrust", "test", "SOW", "LPSY"]

# Squeeze Detection
SQUEEZE_COMPARE_WITH_KELTNER = True
SQUEEZE_BB_KELTNER_RATIO = 0.8

# Volatility Regime
VOLATILITY_REGIME_LOOKBACK = 50
VOLATILITY_PERCENTILES = {
    "extreme_low": 10,
    "low": 25,
    "normal": 75,
    "high": 90,
    "extreme_high": 95,
}

# Liquidity Regime
LIQUIDITY_SCORE_WEIGHTS = {
    "volume_trend": 0.4,
    "volume_profile": 0.3,
    "spread": 0.3,
}

# ============================================================================
# SCORING & GRADING
# ============================================================================

SCORING_WEIGHTS = {
    "momentum": 0.25,
    "trend": 0.25,
    "volatility": 0.15,
    "volume": 0.20,
    "divergence": 0.15,
}

REGIME_WEIGHTS = {
    "TRENDING": {
        "momentum": 0.20,
        "trend": 0.40,
        "volatility": 0.15,
        "volume": 0.15,
        "divergence": 0.10,
    },
    "RANGING": {
        "momentum": 0.30,
        "trend": 0.15,
        "volatility": 0.20,
        "volume": 0.20,
        "divergence": 0.15,
    },
    "VOLATILE": {
        "momentum": 0.15,
        "trend": 0.20,
        "volatility": 0.35,
        "volume": 0.20,
        "divergence": 0.10,
    },
}

AGREEMENT_HIGH_THRESHOLD = 0.70
AGREEMENT_HIGH_BOOST = 0.08
AGREEMENT_LOW_THRESHOLD = 0.40
AGREEMENT_LOW_PENALTY = -0.10

GRADE_THRESHOLDS = {
    "A+": 0.92,
    "A": 0.85,
    "B+": 0.80,
    "B": 0.75,
    "B-": 0.70,
    "C+": 0.65,
    "C": 0.60,
    "D": 0.50,
    "F": 0.00,
}

POSITION_MULTIPLIERS = {
    "A+": 1.5,
    "A": 1.3,
    "B+": 1.1,
    "B": 1.0,
    "B-": 0.9,
    "C+": 0.75,
    "C": 0.5,
    "D": 0.25,
    "F": 0.0,
}

# ============================================================================
# ENTRY, STOP LOSS & TAKE PROFIT
# ============================================================================

ENTRY_PRIORITY = ["ema21", "vwap", "support_resistance", "bollinger_mid"]
EMA21_ENTRY_OFFSET = 0.002

SL_PRIORITY = ["swing", "ema_slow", "bollinger_opposite", "atr"]
SL_SWING_OFFSET_BUY = 0.995
SL_SWING_OFFSET_SELL = 1.005
SL_EMA_SLOW_OFFSET_BUY = 0.99
SL_EMA_SLOW_OFFSET_SELL = 1.01

TP_PRIORITY = ["resistance_support", "bollinger_opposite", "atr_multiple"]

DYNAMIC_RR = {
    "high_confidence": 1.2,
    "medium_confidence": 1.5,
    "low_confidence": 2.0,
}

MAX_STOP_PERCENT = 0.05
MIN_STOP_PERCENT = 0.002
MAX_TP_PERCENT = 0.10
MAX_CONFLICT_RATIO = 0.30

# ============================================================================
# INDICATOR ENABLE/DISABLE FLAGS
# ============================================================================

# Core (always enabled)
ENABLE_RSI = True
ENABLE_MACD = True
ENABLE_EMA = True
ENABLE_ADX = True
ENABLE_BOLLINGER = True
ENABLE_ATR = True
ENABLE_VOLUME = True

# Volume Indicators
ENABLE_OBV = True
ENABLE_CMF = True
ENABLE_MFI = True
ENABLE_ADL = True
ENABLE_KLINGER = True
ENABLE_CHAIKIN_OSC = True
ENABLE_EOM = True
ENABLE_VPT = True
ENABLE_NVI = True
ENABLE_PVI = True
ENABLE_VOLUME_OSCILLATOR = True

# Momentum Indicators
ENABLE_STOCHASTIC = True
ENABLE_STOCH_RSI = True
ENABLE_CCI = True
ENABLE_WILLIAMS_R = True
ENABLE_ULTIMATE_OSCILLATOR = True
ENABLE_AWESOME_OSCILLATOR = True
ENABLE_ROC = True
ENABLE_TSI = True
ENABLE_KST = True
ENABLE_TRIX = True
ENABLE_MASS_INDEX = True

# Trend Indicators
ENABLE_ICHIMOKU = True
ENABLE_KELTNER = True
ENABLE_DONCHIAN = True
ENABLE_PSAR = True
ENABLE_ELDER_RAY = True
ENABLE_VORTEX = True
ENABLE_VHF = True
ENABLE_RVI = True
ENABLE_COPPOCK = True

# Support/Resistance
ENABLE_SWING_POINTS = True
ENABLE_HIDDEN_SR = True
ENABLE_VOLUME_PROFILE = True

# Advanced
ENABLE_CUMULATIVE_DELTA = True
ENABLE_HTF_INDICATORS = True

# ============================================================================
# PERFORMANCE & DEBUGGING
# ============================================================================

USE_VECTORIZED = True
CACHE_INDICATORS = True
CACHE_SIZE = 100
CACHE_TTL = 60

DEBUG_MODE = False
DEBUG_OUTPUT_INDICATORS = False
DEBUG_OUTPUT_DIVERGENCES = False

# ============================================================================
# SYMBOL-SPECIFIC OVERRIDES
# ============================================================================

SYMBOL_OVERRIDES = {
    # "BTCUSDT": {
    #     "ATR_STOP_MULTIPLIER": 2.0,
    #     "MIN_CONFIDENCE_TO_TRADE": 0.70,
    # },
}