"""
SMC Expert V3 - Configuration (FIXED)
All ATR-based thresholds, weights, and settings
Lowered thresholds for better signal detection
Added HTF alignment weights
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SMCCONFIG:
    """Main configuration for SMC Expert"""
    
    # ============================================================
    # ATR-BASED THRESHOLDS (Dynamic, adapts to volatility)
    # ============================================================
    
    # Distance thresholds (in ATR multiples) - LOWERED for better detection
    POI_DISTANCE_MAX_ATR: float = 2.0      # 1.5 → 2.0 (more tolerant)
    LIQUIDITY_DISTANCE_MAX_ATR: float = 2.5  # 2.0 → 2.5
    SWEEP_DISTANCE_MAX_ATR: float = 0.8     # 0.5 → 0.8
    STOP_DISTANCE_MIN_ATR: float = 0.5      # 0.8 → 0.5 (tighter stops allowed)
    STOP_DISTANCE_MAX_ATR: float = 2.5      # 2.0 → 2.5
    TARGET_DISTANCE_MIN_ATR: float = 1.2    # 1.5 → 1.2
    MAX_SLIPPAGE_ATR: float = 0.5           # 0.3 → 0.5
    
    # Structure break thresholds - LOWERED
    BOS_MIN_MOVE_ATR: float = 0.6           # 1.0 → 0.6
    BOS_CONFIRMATION_ATR: float = 0.3       # 0.5 → 0.3
    MSS_MIN_MOVE_ATR: float = 0.5           # 0.8 → 0.5
    
    # ============================================================
    # SWING DETECTION
    # ============================================================
    
    SWING_WINDOW: int = 3                   # 5 → 3 (more sensitive)
    SWING_STRENGTH_THRESHOLD: float = 0.4   # 0.6 → 0.4
    MIN_SWING_DISTANCE_ATR: float = 0.5     # 0.8 → 0.5
    
    # ============================================================
    # ORDER BLOCK DETECTION - LOWERED THRESHOLDS
    # ============================================================
    
    OB_MIN_STRENGTH: float = 0.4            # 0.6 → 0.4
    OB_MIN_VOLUME_RATIO: float = 0.8        # 1.2 → 0.8
    OB_MAX_AGE_BARS: int = 100              # 50 → 100 (keep older OBs)
    OB_MIN_DISPLACEMENT_ATR: float = 0.5    # 1.0 → 0.5 (easier to detect)
    OB_MITIGATION_TOLERANCE_ATR: float = 0.2  # 0.1 → 0.2
    
    # ============================================================
    # FVG DETECTION - LOWERED THRESHOLDS
    # ============================================================
    
    FVG_MIN_GAP_ATR: float = 0.05           # 0.2 → 0.05 (detect smaller gaps)
    FVG_MAX_AGE_BARS: int = 50              # 30 → 50
    FVG_MIN_STRENGTH: float = 0.3           # 0.5 → 0.3
    
    # ============================================================
    # LIQUIDITY DETECTION
    # ============================================================
    
    LIQUIDITY_SWEEP_THRESHOLD_ATR: float = 0.5  # 1.0 → 0.5
    LIQUIDITY_CLUSTER_TOLERANCE_ATR: float = 0.5  # 0.3 → 0.5
    MIN_LIQUIDITY_TOUCHES: int = 2
    LIQUIDITY_STRENGTH_DECAY_BARS: int = 200
    
    # ============================================================
    # OTE & FIBONACCI LEVELS
    # ============================================================
    
    OTE_LEVEL_1: float = 0.705              # 70.5%
    OTE_LEVEL_2: float = 0.79               # 79%
    FIB_RETRACEMENTS: List[float] = field(default_factory=lambda: [0.382, 0.5, 0.618, 0.705, 0.786])
    FIB_EXTENSIONS: List[float] = field(default_factory=lambda: [1.272, 1.414, 1.618, 2.0])
    
    # ============================================================
    # SESSION TIMES (EST)
    # ============================================================
    
    ASIA_START: int = 20    # 8 PM
    ASIA_END: int = 1       # 1 AM
    LONDON_START: int = 3   # 3 AM
    LONDON_END: int = 7     # 7 AM
    NY_START: int = 8       # 8 AM
    NY_END: int = 12        # 12 PM
    
    KILL_ZONES: List[Dict] = field(default_factory=lambda: [
        {"session": "LONDON", "start": 3, "end": 4},
        {"session": "LONDON_NY", "start": 8, "end": 10},
        {"session": "NY", "start": 10, "end": 11},
        {"session": "NY", "start": 14, "end": 15},
    ])
    
    SESSION_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "ASIA": 0.5,                    # 0.6 → 0.5
        "LONDON": 0.8,                  # 0.9 → 0.8
        "LONDON_NY_OVERLAP": 1.0,
        "NY": 0.7,                      # 0.8 → 0.7
        "OUTSIDE": 0.3,                 # 0.4 → 0.3
    })
    
    # ============================================================
    # SCORING WEIGHTS (Meta scoring)
    # ============================================================
    
    SCORING_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'market_structure': 0.12,
        'order_block': 0.12,
        'fvg': 0.10,
        'liquidity': 0.12,
        'premium_discount': 0.08,
        'session': 0.08,
        'displacement': 0.05,
        'confluence': 0.10,
        'amd_phase': 0.08,
        'htf_alignment': 0.15,          # NEW: increased weight for HTF
    })
    
    # ============================================================
    # GRADE THRESHOLDS - LOWERED for more signals
    # ============================================================
    
    GRADE_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'A+': 0.90,     # 0.92 → 0.90
        'A': 0.82,      # 0.85 → 0.82
        'B+': 0.76,     # 0.80 → 0.76
        'B': 0.70,      # 0.75 → 0.70
        'B-': 0.65,     # 0.70 → 0.65
        'C+': 0.58,     # 0.65 → 0.58
        'C': 0.50,      # 0.60 → 0.50
        'D': 0.40,      # 0.50 → 0.40
        'F': 0.00,
    })
    
    POSITION_MULTIPLIERS: Dict[str, float] = field(default_factory=lambda: {
        'A+': 1.5,
        'A': 1.3,
        'B+': 1.1,
        'B': 1.0,
        'B-': 0.9,
        'C+': 0.7,
        'C': 0.5,
        'D': 0.3,
        'F': 0.0,
    })
    
    # ============================================================
    # ACTION THRESHOLDS - LOWERED
    # ============================================================
    
    STRONG_ENTRY_THRESHOLD: float = 0.70   # 0.85 → 0.80
    ENTER_NOW_THRESHOLD: float = 0.65      # 0.70 → 0.65
    WAIT_RETEST_THRESHOLD: float = 0.55    # 0.60 → 0.55
    
    # Kill zone requirement - RELAXED
    KILL_ZONE_REQUIRED_SCORE: float = 0.60  # 0.85 → 0.70
    KILL_ZONE_PENALTY: float = 0.85        # 0.8 → 0.85 (less penalty)
    
    # ============================================================
    # MULTI-TIMEFRAME (HTF) SETTINGS - NEW
    # ============================================================
    
    # HTF timeframes to analyze (in order of importance)
    HTF_TIMEFRAMES: List[str] = field(default_factory=lambda: ['15m', '1h', '4h', '1d'])
    
    # HTF alignment weights (higher = more important)
    HTF_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        '15m': 0.6,
        '1h': 0.8,
        '4h': 1.0,
        '1d': 1.2,
    })
    
    # HTF alignment scoring thresholds
    HTF_ALIGNMENT_BONUS: float = 1.3       # Max multiplier when all HTFs aligned
    HTF_ALIGNMENT_PENALTY: float = 0.7     # Min multiplier when all HTFs oppose
    
    # Minimum HTFs required for alignment check
    MIN_HTF_ALIGNMENT_COUNT: int = 2
    
    # ============================================================
    # RISK MANAGEMENT
    # ============================================================
    
    DYNAMIC_RR: Dict[str, Dict] = field(default_factory=lambda: {
        'high_confidence': {'min_rr': 1.2, 'desc': 'Confidence > 0.80'},
        'medium_confidence': {'min_rr': 1.5, 'desc': 'Confidence 0.65-0.80'},
        'low_confidence': {'min_rr': 2.0, 'desc': 'Confidence 0.50-0.65'},
        'trending': {'min_rr': 1.2, 'desc': 'Trending market'},
        'ranging': {'min_rr': 1.5, 'desc': 'Ranging market'},
        'volatile': {'min_rr': 2.0, 'desc': 'Volatile market'},
        'strong_poi': {'min_rr': 1.2, 'desc': '3+ confluences'},
        'moderate_poi': {'min_rr': 1.5, 'desc': '2 confluences'},
        'weak_poi': {'min_rr': 2.0, 'desc': '1 confluence'},
    })
    
    MAX_RISK_PER_TRADE: float = 0.02
    MAX_DAILY_TRADES: int = 8              # 6 → 8
    
    # ============================================================
    # REPLAY & LEARNING
    # ============================================================
    
    REPLAY_LOOKBACK: int = 100             # 50 → 100
    REPLAY_MIN_SAMPLES: int = 10           # 20 → 10
    SIMILARITY_THRESHOLD: float = 0.6      # 0.7 → 0.6
    
    # ============================================================
    # MISCELLANEOUS
    # ============================================================
    
    MIN_BARS_FOR_ANALYSIS: int = 100        # 100 → 50
    VOLATILITY_PERIOD: int = 20
    DISPLACEMENT_VOLUME_THRESHOLD: float = 1.2  # 1.5 → 1.2
    
    # ============================================================
    # AMD PHASE DETECTION
    # ============================================================
    
    ACCUMULATION_RANGE_ATR: float = 1.5    # 2.0 → 1.5
    ACCUMULATION_MIN_BARS: int = 8         # 10 → 8
    MANIPULATION_SWEEP_ATR: float = 0.5    # 1.0 → 0.5
    DISTRIBUTION_TREND_ATR: float = 1.5    # 2.0 → 1.5


# Singleton instance
CONFIG = SMCCONFIG()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_poi_distance_max(atr: float) -> float:
    """Get max POI distance based on ATR"""
    return CONFIG.POI_DISTANCE_MAX_ATR * atr


def get_liquidity_distance_max(atr: float) -> float:
    """Get max liquidity distance based on ATR"""
    return CONFIG.LIQUIDITY_DISTANCE_MAX_ATR * atr


def get_stop_distance_min(atr: float) -> float:
    """Get minimum stop distance based on ATR"""
    return max(CONFIG.STOP_DISTANCE_MIN_ATR * atr, atr * 0.3)


def get_min_rr(confidence: float, regime: str, poi_strength: str) -> float:
    """Get dynamic minimum RR based on multiple factors - LOWERED"""
    min_rr = 1.0  # Start lower
    
    # Adjust for confidence
    if confidence > 0.80:
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['high_confidence']['min_rr'])
    elif confidence > 0.65:
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['medium_confidence']['min_rr'])
    else:
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['low_confidence']['min_rr'])
    
    # Adjust for regime
    if regime == "TRENDING":
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['trending']['min_rr'])
    elif regime == "RANGING":
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['ranging']['min_rr'])
    else:
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['volatile']['min_rr'])
    
    # Adjust for POI strength
    if poi_strength == "STRONG":
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['strong_poi']['min_rr'])
    elif poi_strength == "MODERATE":
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['moderate_poi']['min_rr'])
    else:
        min_rr = max(min_rr, CONFIG.DYNAMIC_RR['weak_poi']['min_rr'])
    
    return round(min_rr, 2)


def get_htf_weight(timeframe: str) -> float:
    """Get weight for a specific HTF timeframe"""
    return CONFIG.HTF_WEIGHTS.get(timeframe, 0.5)