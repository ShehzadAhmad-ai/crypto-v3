"""
Strategy Expert - Strategies Package
All trading strategies are stored in this folder.

This package auto-loads all strategy classes from .py files in this directory.
Each strategy should inherit from BaseStrategy and implement:
    - generate_signal()
    - calculate_entry_sl_tp()

Available Strategies (Version 3 - Advanced):
    Core/Trend Strategies:
        - BollingerBandsStrategy
        - EMATrendStrategy
        - MACDCrossoverStrategy
        - RSIMomentumStrategy
        - SupportResistanceStrategy
        - VolumeSpikeStrategy
        - VWAPReversionStrategy
    
    Smart Money/SMC Strategies:
        - LiquiditySweepReversalStrategy
        - MarketMakerTrapExpansionStrategy
        - SmartOrderFlowMomentumStrategy
        - HiddenOrderAccumulationStrategy
        - SmartMoneyStrategy
    
    Advanced Multi-Layer Strategies:
        - MultiLayerPrePumpStrategy
        - AdaptiveAIPrePumpStrategy
        - DynamicPumpPredictionStrategy
        - VolatilityCompressionBreakoutStrategy
        - VolatilityExpansionMomentumStrategy
        - VolumetricAccumulationPressureStrategy
        - MultiTimeframeMomentumSqueezeStrategy
        - VSASmartZoneStrategy
    
    Institutional Strategies (S01-S10):
        - TrendMomentumStrategy (S01)
        - VolatilityBreakoutStrategy (S02)
        - LiquiditySweepStrategy (S03)
        - OrderBlockStrategy (S04)
        - MarketMakerStrategy (S05)
        - AIProbabilityStrategy (S06)
        - WhaleLiquidationStrategy (S07)
        - SmartMoneyFlowStrategy (S08)
        - MTFFusionStrategy (S09)
        - VSASmartZoneStrategy (S10)
    
    Pattern Recognition:
        - PatternRecognitionStrategy
"""

# Core/Trend Strategies
from strategy_expert.strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategy_expert.strategies.ema_trend import EMATrendStrategy
from strategy_expert.strategies.macd_crossover_strategy import MACDCrossoverStrategy
from strategy_expert.strategies.rsi_momentum_strategy import RSIMomentumStrategy
from strategy_expert.strategies.support_resistance_strategy import SupportResistanceStrategy
from strategy_expert.strategies.volume_spike_strategy import VolumeSpikeStrategy
from strategy_expert.strategies.vwap_reversion import VWAPReversionStrategy

# Smart Money/SMC Strategies
from strategy_expert.strategies.liquidity_sweep_reversal import LiquiditySweepReversalStrategy
from strategy_expert.strategies.market_maker_trap_expansion import MarketMakerTrapExpansionStrategy
from strategy_expert.strategies.smart_order_flow_momentum import SmartOrderFlowMomentumStrategy
from strategy_expert.strategies.hidden_order_accumulation import HiddenOrderAccumulationStrategy
from strategy_expert.strategies.smart_money_strategy import SmartMoneyStrategy

# Advanced Multi-Layer Strategies
from strategy_expert.strategies.multi_layer_pre_pump import MultiLayerPrePumpStrategy
from strategy_expert.strategies.adaptive_ai_pre_pump import AdaptiveAIPrePumpStrategy
from strategy_expert.strategies.dynamic_pump_prediction import DynamicPumpPredictionStrategy
from strategy_expert.strategies.volatility_compression_breakout import VolatilityCompressionBreakoutStrategy
from strategy_expert.strategies.volatility_expansion_momentum import VolatilityExpansionMomentumStrategy
from strategy_expert.strategies.volumetric_accumulation_pressure import VolumetricAccumulationPressureStrategy
from strategy_expert.strategies.multi_timeframe_momentum_squeeze import MultiTimeframeMomentumSqueezeStrategy

# Institutional Strategies (S01-S10)
from strategy_expert.strategies.s01_trend_momentum import TrendMomentumStrategy
from strategy_expert.strategies.s02_volatility_breakout import VolatilityBreakoutStrategy
from strategy_expert.strategies.s03_liquidity_sweep import LiquiditySweepStrategy
from strategy_expert.strategies.s04_order_block import OrderBlockStrategy
from strategy_expert.strategies.s05_market_maker import MarketMakerStrategy
from strategy_expert.strategies.s06_ai_probability import AIProbabilityStrategy
from strategy_expert.strategies.s07_whale_liquidation import WhaleLiquidationStrategy
from strategy_expert.strategies.s08_smart_money_flow import SmartMoneyFlowStrategy
from strategy_expert.strategies.s09_mtf_fusion import MTFFusionStrategy
from strategy_expert.strategies.s10_vsa_smart_zone import VSASmartZoneStrategy

# Pattern Recognition
from strategy_expert.strategies.pattern_recognition_strategy import PatternRecognitionStrategy

# Export all strategies
__all__ = [
    # Core/Trend
    'BollingerBandsStrategy',
    'EMATrendStrategy',
    'MACDCrossoverStrategy',
    'RSIMomentumStrategy',
    'SupportResistanceStrategy',
    'VolumeSpikeStrategy',
    'VWAPReversionStrategy',
    
    # Smart Money/SMC
    'LiquiditySweepReversalStrategy',
    'MarketMakerTrapExpansionStrategy',
    'SmartOrderFlowMomentumStrategy',
    'HiddenOrderAccumulationStrategy',
    'SmartMoneyStrategy',
    
    # Advanced Multi-Layer
    'MultiLayerPrePumpStrategy',
    'AdaptiveAIPrePumpStrategy',
    'DynamicPumpPredictionStrategy',
    'VolatilityCompressionBreakoutStrategy',
    'VolatilityExpansionMomentumStrategy',
    'VolumetricAccumulationPressureStrategy',
    'MultiTimeframeMomentumSqueezeStrategy',
    
    # Institutional
    'TrendMomentumStrategy',
    'VolatilityBreakoutStrategy',
    'LiquiditySweepStrategy',
    'OrderBlockStrategy',
    'MarketMakerStrategy',
    'AIProbabilityStrategy',
    'WhaleLiquidationStrategy',
    'SmartMoneyFlowStrategy',
    'MTFFusionStrategy',
    'VSASmartZoneStrategy',
    
    # Pattern Recognition
    'PatternRecognitionStrategy',
]

# Print initialization message (only in debug mode)
import os
if os.environ.get('STRATEGY_DEBUG', '0') == '1':
    print(f"📦 Strategy Expert Strategies Package loaded (Version 3 - Advanced)")
    print(f"   Available: {len(__all__)} strategies")