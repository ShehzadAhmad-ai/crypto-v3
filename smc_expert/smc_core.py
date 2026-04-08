"""
SMC Expert V3 - Core Data Structures (FIXED)
All base classes, enums, and data containers for the SMC system
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import pandas as pd
import numpy as np

# ============================================================
# ENUMS
# ============================================================

class Direction(Enum):
    """Trading direction"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class SessionType(Enum):
    """Market sessions"""
    ASIA = "ASIA"
    LONDON = "LONDON"
    NY = "NY"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    ASIA_LONDON_OVERLAP = "ASIA_LONDON_OVERLAP"
    OUTSIDE = "OUTSIDE"


class AMDPhase(Enum):
    """Accumulation, Manipulation, Distribution phases"""
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"
    UNKNOWN = "UNKNOWN"


class MitigationState(Enum):
    """Mitigation status for OBs and FVGs"""
    UNMITIGATED = "UNMITIGATED"
    PARTIAL = "PARTIAL"
    FULL = "FULL"
    INVALIDATED = "INVALIDATED"


class SwingType(Enum):
    """Swing point types"""
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low


class OBType(Enum):
    """Order block types"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    BREAKER_BULLISH = "BREAKER_BULLISH"
    BREAKER_BEARISH = "BREAKER_BEARISH"
    RECLAIMED_BULLISH = "RECLAIMED_BULLISH"
    RECLAIMED_BEARISH = "RECLAIMED_BEARISH"


class FVGType(Enum):
    """Fair Value Gap types"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    INVERSE = "INVERSE"


class ZoneType(Enum):
    """Premium/Discount zone types"""
    DEEP_DISCOUNT = "DEEP_DISCOUNT"
    DISCOUNT = "DISCOUNT"
    EQUILIBRIUM = "EQUILIBRIUM"
    PREMIUM = "PREMIUM"
    DEEP_PREMIUM = "DEEP_PREMIUM"


class ActionType(Enum):
    """Trading actions"""
    STRONG_ENTRY = "STRONG_ENTRY"
    ENTER_NOW = "ENTER_NOW"
    WAIT_RETEST = "WAIT_RETEST"
    SKIP = "SKIP"
    EXIT = "EXIT"
    FLIP = "FLIP"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Candle:
    """Individual candle data with derived features"""
    index: int
    timestamp: Union[datetime, int]
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Derived fields (calculated in __post_init__)
    body: float = 0.0
    upper_wick: float = 0.0
    lower_wick: float = 0.0
    range_: float = 0.0
    body_ratio: float = 0.0
    is_bullish: bool = False
    is_bearish: bool = False
    strength: float = 0.0
    
    def __post_init__(self):
        """Calculate derived values"""
        self.body = abs(self.close - self.open)
        self.range_ = self.high - self.low
        
        if self.range_ > 0:
            self.body_ratio = self.body / self.range_
        else:
            self.body_ratio = 0.5
        
        self.is_bullish = self.close > self.open
        self.is_bearish = self.close < self.open
        
        # Upper and lower wicks
        if self.is_bullish:
            self.upper_wick = self.high - self.close
            self.lower_wick = self.open - self.low
        else:
            self.upper_wick = self.high - self.open
            self.lower_wick = self.close - self.low
        
        # Strength based on body ratio
        self.strength = self.body_ratio


@dataclass
class Swing:
    """Swing point (HH/HL/LH/LL)"""
    index: int
    price: float
    type: SwingType
    strength: float
    timestamp: Union[datetime, int]
    confirmed: bool = True


@dataclass
class OrderBlock:
    """Order block structure"""
    type: OBType
    direction: Direction
    price: float
    stop: float
    strength: float
    age_bars: int
    timestamp: Union[datetime, int]
    mitigation_state: MitigationState = MitigationState.UNMITIGATED
    mitigation_price: Optional[float] = None
    volume_ratio: float = 1.0
    displacement_strength: float = 0.5


@dataclass
class FVG:
    """Fair Value Gap structure"""
    type: FVGType
    direction: Direction
    upper: float
    lower: float
    mid: float
    strength: float
    age_bars: int
    timestamp: Union[datetime, int]
    mitigation_state: MitigationState = MitigationState.UNMITIGATED
    mitigation_percent: float = 0.0


@dataclass
class LiquidityLevel:
    """Liquidity level (BSL or SSL)"""
    price: float
    type: str
    touches: int
    strength: float
    distance_pct: float
    distance_atr: float
    timestamp: Union[datetime, int]
    cluster_id: Optional[int] = None


@dataclass
class LiquiditySweep:
    """Liquidity sweep event"""
    price: float
    type: str
    target_level: float
    reversal_strength: float
    timestamp: Union[datetime, int]
    candle_index: int


@dataclass
class POI:
    """Point of Interest - combined confluence zone"""
    price: float
    type: str
    sub_type: str
    direction: Direction
    strength: float
    components: List[Dict]
    distance_atr: float
    is_active: bool = True


@dataclass
class SMCContext:
    """Complete market context for analysis"""
    current_structure: str = "NEUTRAL"
    structure_strength: float = 0.5
    swings: List[Swing] = field(default_factory=list)
    bos_points: List[Dict] = field(default_factory=list)
    choch_points: List[Dict] = field(default_factory=list)
    amd_phase: AMDPhase = AMDPhase.UNKNOWN
    amd_confidence: float = 0.0
    session: SessionType = SessionType.OUTSIDE
    is_kill_zone: bool = False
    session_weight: float = 0.5
    current_price: float = 0.0
    atr: float = 0.0
    volatility_regime: str = "NORMAL"
    zone_type: ZoneType = ZoneType.EQUILIBRIUM
    zone_percent: float = 0.5
    ote_levels: Dict[str, float] = field(default_factory=dict)
    htf_trend: str = "NEUTRAL"
    htf_alignment_score: float = 0.5
    recent_sweeps: List[LiquiditySweep] = field(default_factory=list)
    recent_traps: List[Dict] = field(default_factory=list)
    displacement_bars: List[int] = field(default_factory=list)
    
    # Multi-HTF support
    htf_analysis: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class SMCData:
    """Complete SMC analysis output"""
    symbol: str
    timeframe: str
    timestamp: datetime
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvgs: List[FVG] = field(default_factory=list)
    liquidity_levels: List[LiquidityLevel] = field(default_factory=list)
    liquidity_sweeps: List[LiquiditySweep] = field(default_factory=list)
    supply_zones: List[Dict] = field(default_factory=list)
    demand_zones: List[Dict] = field(default_factory=list)
    active_pois: List[POI] = field(default_factory=list)
    liquidity_map: Dict[str, Any] = field(default_factory=dict)
    next_target: Optional[Dict] = None
    context: Optional[SMCContext] = None
    signals: List[Dict] = field(default_factory=list)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR from dataframe - SAFE with fallback"""
    if df is None or len(df) < period + 1:
        return 0.001  # Small fallback to avoid division by zero
    
    try:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            return (df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) / 20
        
        return atr
    except Exception:
        return 0.001


def df_row_to_candle(row: pd.Series, index: int) -> Candle:
    """
    Convert DataFrame row to Candle object
    This FIXES the KeyError: 'body' issue
    """
    # Handle timestamp - could be datetime or integer index
    timestamp = row.name if hasattr(row, 'name') else index
    if not isinstance(timestamp, datetime):
        timestamp = datetime.now()
    
    # Get values with fallbacks
    open_price = float(row['open']) if 'open' in row else 0
    high_price = float(row['high']) if 'high' in row else 0
    low_price = float(row['low']) if 'low' in row else 0
    close_price = float(row['close']) if 'close' in row else 0
    volume = float(row['volume']) if 'volume' in row else 0
    
    candle = Candle(
        index=index,
        timestamp=timestamp,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume
    )
    
    return candle


def df_to_candle_list(df: pd.DataFrame) -> List[Candle]:
    """Convert entire DataFrame to list of Candle objects"""
    candles = []
    for i in range(len(df)):
        row = df.iloc[i]
        candle = df_row_to_candle(row, i)
        candles.append(candle)
    return candles


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def safe_get_swing_price(swing: Any) -> float:
    """Safely get price from either Swing object or dict"""
    if hasattr(swing, 'price'):
        return swing.price
    elif isinstance(swing, dict):
        return swing.get('price', 0)
    return 0


def safe_get_swing_type(swing: Any) -> str:
    """Safely get type from either Swing object or dict"""
    if hasattr(swing, 'type'):
        if hasattr(swing.type, 'value'):
            return swing.type.value
        return str(swing.type)
    elif isinstance(swing, dict):
        return swing.get('type', 'UNKNOWN')
    return 'UNKNOWN'