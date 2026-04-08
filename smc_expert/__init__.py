"""
SMC Expert V3 - Complete Smart Money Concepts Trading System
"""

from .smc_core import (
    Direction, SessionType, AMDPhase, MitigationState,
    Candle, Swing, OrderBlock, FVG, LiquidityLevel, POI,
    SMCContext, SMCData
)

from .smc_config import SMCCONFIG, CONFIG, get_min_rr
from .smc_factory import SMCFactory
from .smc_patterns import ICTPatternManager, SilverBulletDetector, PowerOfThreeDetector, TurtleSoupDetector
from .smc_confirmation import EntryConfirmation

__all__ = [
    'SMCFactory',
    'SMCCONFIG',
    'CONFIG',
    'Direction',
    'SessionType',
    'AMDPhase',
    'MitigationState',
    'Candle',
    'Swing',
    'OrderBlock',
    'FVG',
    'LiquidityLevel',
    'POI',
    'SMCContext',
    'SMCData',
    'get_min_rr',
    'ICTPatternManager',
    'SilverBulletDetector',
    'PowerOfThreeDetector',
    'TurtleSoupDetector',
    'EntryConfirmation'
]

__version__ = '3.0.0'