"""
Strategy Expert Configuration
Centralized configuration for all strategies and pipeline settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StrategyConfig:
    """Configuration for a single strategy"""
    enabled: bool = True
    base_weight: float = 1.0
    min_confidence: float = 0.6
    min_rr: float = 1.5
    params: Dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for the 8-step pipeline"""
    # Step 1: Quality Filter
    min_strategy_confidence: float = 0.55
    
    # Step 2: Max Strategies
    max_strategies_to_use: int = 13
    
    # Step 3 & 4: Voting & Agreement
    min_agreement_ratio: float = 0.55  # 55% minimum agreement
    
    # Step 6: Risk Filter
    min_risk_reward: float = 1.2
    
    # Step 7: Late Entry Filter
    late_entry_atr_multiplier: float = 1.0  # Skip if price moved > 1 ATR
    
    # Weight Manager
    weight_min: float = 0.5
    weight_max: float = 2.0
    weight_target_score: float = 0.75  # Target win_rate × avg_rr
    weight_history_length: int = 100
    
    # Scoring Engine
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'A': 0.85,   # Confidence ≥ 85%
        'B': 0.75,   # Confidence ≥ 75%
        'C': 0.65,   # Confidence ≥ 65%
        'D': 0.55,   # Confidence ≥ 55%
        'F': 0.0
    })
    
    # Output
    max_reasons: int = 5


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations"""
    # RSI
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_threshold: float = 0.04  # 4% width
    
    # ADX
    adx_period: int = 14
    adx_trend_threshold: int = 25
    adx_strong_threshold: int = 40
    
    # ATR
    atr_period: int = 14
    atr_expansion_threshold: float = 1.3
    atr_low_ratio: float = 0.7
    
    # Volume
    volume_ma_period: int = 20
    volume_spike_threshold: float = 1.5
    volume_breakout_threshold: float = 1.8
    volume_growth_threshold: float = 1.2
    
    # EMA
    ema_short: int = 8
    ema_mid: int = 21
    ema_long: int = 50
    ema_trend: int = 200
    
    # OBV
    obv_period: int = 20
    
    # Range
    tight_range_threshold: float = 0.04


@dataclass
class StrategyExpertConfig:
    """Main configuration container"""
    # Pipeline settings
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Indicator settings
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    
    # Strategy-specific settings
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    
    # Global overrides
    global_enabled: bool = True
    debug_mode: bool = False


# ============================================================================
# DEFAULT STRATEGY CONFIGURATIONS
# ============================================================================

def get_default_strategy_configs() -> Dict[str, StrategyConfig]:
    """Get default configurations for all strategies"""
    
    return {
        # Core Strategies
        'LiquiditySweepReversalStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.25,
            min_confidence=0.60,
            min_rr=1.5,
            params={
                'volume_spike_threshold': 1.6,
                'rsi_bullish_min': 40,
                'rsi_bearish_max': 60,
                'buy_confidence': 0.87,
                'sell_confidence': 0.85,
            }
        ),
        
        'MarketMakerTrapExpansionStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.40,
            min_confidence=0.60,
            min_rr=1.5,
            params={
                'volume_spike_threshold': 1.7,
                'rsi_bullish_min': 50,
                'rsi_bearish_max': 50,
                'buy_confidence': 0.88,
                'sell_confidence': 0.86,
            }
        ),
        
        'MultiLayerPrePumpStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.55,
            min_confidence=0.65,
            min_rr=2.0,
            params={
                'atr_expansion_threshold': 1.3,
                'volume_confirm_threshold': 1.5,
                'rsi_bullish_min': 55,
                'rsi_bearish_max': 45,
                'adx_trend_threshold': 25,
                'buy_confidence': 0.92,
                'sell_confidence': 0.90,
            }
        ),
        
        'MultiTimeframeMomentumSqueezeStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.45,
            min_confidence=0.60,
            min_rr=1.5,
            params={
                'squeeze_threshold': 0.04,
                'volume_spike': 1.5,
                'adx_strong': 25,
                'rsi_bullish': 55,
                'rsi_bearish': 45,
            }
        ),
        
        'SmartOrderFlowMomentumStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.50,
            min_confidence=0.60,
            min_rr=1.5,
            params={
                'cumulative_delta_ratio': 1.2,
                'volume_spike_threshold': 1.5,
                'rsi_bullish_min': 55,
                'rsi_bearish_max': 45,
            }
        ),
        
        'VolatilityCompressionBreakoutStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.30,
            min_confidence=0.60,
            min_rr=1.5,
            params={
                'bb_squeeze_threshold': 0.04,
                'atr_low_ratio': 0.7,
                'volume_breakout_threshold': 1.8,
                'rsi_bullish_min': 55,
                'rsi_bearish_max': 45,
            }
        ),
        
        'VolatilityExpansionMomentumStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.35,
            min_confidence=0.60,
            min_rr=1.5,
            params={
                'atr_expansion_threshold': 1.3,
                'adx_trend_threshold': 25,
                'adx_strong_threshold': 40,
                'volume_confirmation_threshold': 1.5,
                'rsi_bullish_min': 55,
                'rsi_bullish_max': 70,
                'rsi_bearish_max': 45,
            }
        ),
        
        'VolumetricAccumulationPressureStrategy': StrategyConfig(
            enabled=True,
            base_weight=1.35,
            min_confidence=0.60,
            min_rr=1.5,
            params={
                'tight_range_threshold': 0.04,
                'volume_growth_threshold': 1.2,
                'volume_spike_threshold': 1.5,
                'rsi_accumulation_min': 55,
                'rsi_distribution_max': 45,
                'obv_period': 20,
            }
        ),
        
        # Additional strategies can be added here...
    }


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class StrategyExpertConfigLoader:
    """Load and manage configuration for Strategy Expert"""
    
    def __init__(self, custom_config: Optional[Dict] = None):
        """
        Initialize configuration
        
        Args:
            custom_config: Optional dictionary with custom overrides
        """
        self.config = StrategyExpertConfig()
        
        # Load default strategy configs
        self.config.strategies = get_default_strategy_configs()
        
        # Apply custom overrides if provided
        if custom_config:
            self._apply_overrides(custom_config)
    
    def _apply_overrides(self, overrides: Dict):
        """Apply custom configuration overrides"""
        
        # Pipeline overrides
        if 'pipeline' in overrides:
            for key, value in overrides['pipeline'].items():
                if hasattr(self.config.pipeline, key):
                    setattr(self.config.pipeline, key, value)
        
        # Indicator overrides
        if 'indicators' in overrides:
            for key, value in overrides['indicators'].items():
                if hasattr(self.config.indicators, key):
                    setattr(self.config.indicators, key, value)
        
        # Strategy overrides
        if 'strategies' in overrides:
            for strat_name, strat_config in overrides['strategies'].items():
                if strat_name in self.config.strategies:
                    for key, value in strat_config.items():
                        if hasattr(self.config.strategies[strat_name], key):
                            setattr(self.config.strategies[strat_name], key, value)
        
        # Global overrides
        if 'global_enabled' in overrides:
            self.config.global_enabled = overrides['global_enabled']
        if 'debug_mode' in overrides:
            self.config.debug_mode = overrides['debug_mode']
    
    def get_strategy_config(self, strategy_name: str) -> StrategyConfig:
        """Get configuration for a specific strategy"""
        if strategy_name in self.config.strategies:
            return self.config.strategies[strategy_name]
        return StrategyConfig()  # Return default if not found
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled"""
        if not self.config.global_enabled:
            return False
        config = self.get_strategy_config(strategy_name)
        return config.enabled
    
    def get_enabled_strategies(self) -> List[str]:
        """Get list of all enabled strategy names"""
        return [
            name for name, config in self.config.strategies.items()
            if config.enabled and self.config.global_enabled
        ]
    
    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get base weight for a strategy"""
        config = self.get_strategy_config(strategy_name)
        return config.base_weight
    
    def update_strategy_config(self, strategy_name: str, updates: Dict):
        """Update a strategy's configuration dynamically"""
        if strategy_name in self.config.strategies:
            for key, value in updates.items():
                if hasattr(self.config.strategies[strategy_name], key):
                    setattr(self.config.strategies[strategy_name], key, value)
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        import dataclasses
        
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {
                    field.name: _to_dict(getattr(obj, field.name))
                    for field in dataclasses.fields(obj)
                }
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_dict(item) for item in obj]
            else:
                return obj
        
        return _to_dict(self.config)


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================

# Create default config instance
default_config = StrategyExpertConfigLoader()

# Helper functions for easy access
def get_config() -> StrategyExpertConfig:
    """Get the global configuration"""
    return default_config.config

def get_pipeline_config() -> PipelineConfig:
    """Get pipeline configuration"""
    return default_config.config.pipeline

def get_indicator_config() -> IndicatorConfig:
    """Get indicator configuration"""
    return default_config.config.indicators