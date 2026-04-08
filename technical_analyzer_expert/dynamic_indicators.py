# dynamic_indicators.py - ENHANCED FOR REGIME-BASED SETTINGS
"""
Dynamic Indicator Configuration - Enhanced Version
- Generates indicator settings based on market regime
- Accepts regime from market_regime detector
- Provides settings for technical_analyzer
- All thresholds configurable from config.py
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from config import Config
from logger import log


class RegimeCategory(Enum):
    """Market regime categories for indicator settings"""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    WEAK_BULL = "WEAK_BULL"
    NEUTRAL = "NEUTRAL"
    WEAK_BEAR = "WEAK_BEAR"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


@dataclass
class AssetProfile:
    """Asset-specific profile data"""
    symbol: str
    volatility_profile: str
    trend_profile: str
    volume_profile: str
    optimal_timeframes: List[str]
    indicator_settings: Dict[str, Any]
    regime_settings: Dict[str, Any] = field(default_factory=dict)


class DynamicIndicatorConfig:
    """
    Enhanced Dynamic Indicator Configuration
    Generates indicator settings based on market regime and asset profile
    """
    
    def __init__(self):
        self.asset_profiles = {}
        
        # Base default settings (from config)
        self.default_settings = {
            # RSI settings
            'rsi_period': getattr(Config, 'RSI_PERIOD', 14),
            'rsi_oversold': getattr(Config, 'RSI_OVERSOLD', 30),
            'rsi_overbought': getattr(Config, 'RSI_OVERBOUGHT', 70),
            
            # MACD settings
            'macd_fast': getattr(Config, 'MACD_FAST', 12),
            'macd_slow': getattr(Config, 'MACD_SLOW', 26),
            'macd_signal': getattr(Config, 'MACD_SIGNAL', 9),
            
            # EMA settings
            'ema_fast': getattr(Config, 'EMA_FAST', 8),
            'ema_medium': getattr(Config, 'EMA_MEDIUM', 21),
            'ema_slow': getattr(Config, 'EMA_SLOW', 50),
            'ema_very_slow': getattr(Config, 'EMA_VERY_SLOW', 200),
            
            # Bollinger Bands
            'bb_period': getattr(Config, 'BB_PERIOD', 20),
            'bb_std': getattr(Config, 'BB_STD', 2.0),
            
            # ATR
            'atr_period': getattr(Config, 'ATR_WINDOW', 14),
            
            # ADX
            'adx_period': getattr(Config, 'ADX_PERIOD', 14),
            'adx_trend_threshold': getattr(Config, 'TREND_REGIME_ADX_THRESHOLD', 25),
            
            # Volume
            'volume_ma_period': getattr(Config, 'VOLUME_MA_PERIOD', 20),
            'volume_spike_threshold': getattr(Config, 'VOLUME_SPIKE_THRESHOLD', 1.5),
            
            # Stochastic
            'stoch_k': 14,
            'stoch_d': 3,
            
            # ROC
            'roc_fast': 5,
            'roc_medium': 10,
            'roc_slow': 20,
            
            # Ichimoku
            'ichimoku_tenkan': 9,
            'ichimoku_kijun': 26,
            'ichimoku_senkou': 52,
            
            # Keltner Channels
            'keltner_period': 20,
            'keltner_atr_period': 10,
            'keltner_offset_atr': 2.0,
            
            # Donchian
            'donchian_period': 20,
            
            # Parabolic SAR
            'psar_iaf': 0.02,
            'psar_maxaf': 0.2,
            
            # ATR Trailing Stop
            'atr_trailing_multiplier': 3.0,
            
            # Elder Ray
            'elder_ema_period': 13,
            
            # Vortex
            'vortex_period': 14,
            
            # CMO
            'cmo_period': 9,
            
            # CMF
            'cmf_period': 20,
            
            # MFI
            'mfi_period': 14,
            
            # Klinger
            'klinger_window1': 34,
            'klinger_window2': 55,
            'klinger_window_signal': 13,
            
            # CCI
            'cci_period': 20,
            
            # Williams %R
            'williams_r_period': 14,
            
            # Stochastic RSI
            'stoch_rsi_period': 14,
            'stoch_rsi_smooth_k': 3,
            'stoch_rsi_smooth_d': 3,
            
            # Ultimate Oscillator
            'uo_window1': 7,
            'uo_window2': 14,
            'uo_window3': 28,
            
            # Volume Profile
            'vprofile_window': 200,
            'vprofile_bins': 40,
            
            # TSI
            'tsi_window_fast': 25,
            'tsi_window_slow': 13,
            
            # KST
            'kst_roc1': 10,
            'kst_roc2': 15,
            'kst_roc3': 20,
            'kst_roc4': 30,
            'kst_window1': 10,
            'kst_window2': 10,
            'kst_window3': 10,
            'kst_window4': 15,
            
            # Enable flags
            'enable_ichimoku': True,
            'enable_keltner': True,
            'enable_donchian': True,
            'enable_psar': True,
            'enable_elder_ray': True,
            'enable_vortex': True,
            'enable_cmo': True,
            'enable_cmf': True,
            'enable_obv': True,
            'enable_mfi': True,
            'enable_adl': True,
            'enable_klinger': True,
            'enable_cci': True,
            'enable_williams_r': True,
            'enable_stoch_rsi': True,
            'enable_uo': True,
            'enable_adx': True,
            'enable_vprofile': True,
            'enable_tsi': True,
            'enable_kst': True,
        }
        
        log.info("DynamicIndicatorConfig initialized")
    
    # ==================== MAIN PUBLIC METHODS ====================
    
    def get_indicator_settings(self, symbol: str, regime: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get indicator settings for a symbol
        If regime is provided, applies regime-based adjustments
        """
        # Start with asset-specific profile if exists
        if symbol in self.asset_profiles:
            settings = self.asset_profiles[symbol].indicator_settings.copy()
        else:
            settings = self.default_settings.copy()
        
        # Apply regime-based adjustments if provided
        if regime:
            settings = self._apply_regime_settings(settings, regime)
        
        return settings
    
    def analyze_asset(self, symbol: str, df: pd.DataFrame, 
                      regime: Optional[Dict] = None) -> AssetProfile:
        """
        Analyze asset and create profile with optional regime adjustment
        """
        if len(df) < 100:
            return self._create_default_profile(symbol)
        
        # Basic asset analysis
        daily_returns = df['close'].pct_change().dropna()
        std_dev = daily_returns.std()
        
        # Volatility profile
        if std_dev < 0.01:
            volatility = 'LOW'
        elif std_dev < 0.03:
            volatility = 'MEDIUM'
        elif std_dev < 0.06:
            volatility = 'HIGH'
        else:
            volatility = 'EXTREME'
        
        # Trend profile
        price = df['close'].values
        if len(price) > 0:
            trend = 'TRENDING' if price[-1] > price[int(len(price)*0.7)] else 'MEAN_REVERTING'
        else:
            trend = 'MEAN_REVERTING'
        
        # Volume profile
        avg_value = (df['volume'] * df['close']).mean()
        if avg_value > 10000000:
            volume_profile = 'VERY_LIQUID'
        elif avg_value > 1000000:
            volume_profile = 'LIQUID'
        elif avg_value > 100000:
            volume_profile = 'MODERATE'
        else:
            volume_profile = 'LOW'
        
        # Optimal timeframes based on volatility
        if volatility == 'LOW':
            timeframes = ['5m', '15m', '30m', '1h']
        elif volatility == 'MEDIUM':
            timeframes = ['5m', '15m', '30m']
        elif volatility == 'HIGH':
            timeframes = ['5m', '15m']
        else:
            timeframes = ['5m']
        
        # Generate base settings
        settings = self._generate_base_settings(volatility, trend, volume_profile)
        
        # Apply regime adjustments if provided
        if regime:
            settings = self._apply_regime_settings(settings, regime)
        
        profile = AssetProfile(
            symbol=symbol,
            volatility_profile=volatility,
            trend_profile=trend,
            volume_profile=volume_profile,
            optimal_timeframes=timeframes,
            indicator_settings=settings,
            regime_settings=regime if regime else {}
        )
        
        self.asset_profiles[symbol] = profile
        return profile
    
    # ==================== REGIME-BASED ADJUSTMENTS ====================
    
    def _apply_regime_settings(self, settings: Dict[str, Any], 
                               regime: Dict) -> Dict[str, Any]:
        """
        Apply regime-specific adjustments to indicator settings
        """
        regime_name = regime.get('regime', 'UNKNOWN')
        bias = regime.get('bias', 'NEUTRAL')
        bias_score = regime.get('bias_score', 0)
        trending_score = regime.get('trending_score', 0.5)
        volatility_state = regime.get('volatility_state', 'NORMAL')
        
        # Make a copy to avoid modifying original
        settings = settings.copy()
        
        # ===== 1. RSI ADJUSTMENTS =====
        if 'BULL' in regime_name or bias in ['BULLISH', 'STRONG_BULLISH']:
            # In bull market, higher RSI is normal
            settings['rsi_oversold'] = 40
            settings['rsi_overbought'] = 85
            settings['rsi_period'] = max(8, settings.get('rsi_period', 14) - 4)
        elif 'BEAR' in regime_name or bias in ['BEARISH', 'STRONG_BEARISH']:
            # In bear market, lower RSI is normal
            settings['rsi_oversold'] = 20
            settings['rsi_overbought'] = 65
            settings['rsi_period'] = max(8, settings.get('rsi_period', 14) - 4)
        elif 'RANGING' in regime_name:
            # In ranging market, use standard levels
            settings['rsi_oversold'] = 30
            settings['rsi_overbought'] = 70
            settings['rsi_period'] = 14
        
        # ===== 2. MACD ADJUSTMENTS =====
        if 'TREND' in regime_name or trending_score > 0.6:
            # Trending market - faster MACD
            settings['macd_fast'] = max(6, settings.get('macd_fast', 12) - 4)
            settings['macd_slow'] = max(20, settings.get('macd_slow', 26) - 6)
        elif 'RANGING' in regime_name:
            # Ranging market - slower MACD
            settings['macd_fast'] = 12
            settings['macd_slow'] = 26
        
        # ===== 3. EMA ADJUSTMENTS =====
        if 'TREND' in regime_name or trending_score > 0.7:
            # Strong trend - faster EMAs
            settings['ema_fast'] = max(5, settings.get('ema_fast', 8) - 3)
            settings['ema_medium'] = max(13, settings.get('ema_medium', 21) - 8)
            settings['ema_slow'] = max(34, settings.get('ema_slow', 50) - 16)
        elif 'RANGING' in regime_name:
            # Ranging - slower EMAs
            settings['ema_fast'] = 8
            settings['ema_medium'] = 21
            settings['ema_slow'] = 50
        
        # ===== 4. BOLLINGER BANDS ADJUSTMENTS =====
        # Ranging markets should have priority - wider bands for mean reversion
        if 'RANGING' in regime_name:
            settings['bb_std'] = 2.2
        elif volatility_state in ['HIGH', 'EXTREME_HIGH']:
            # High volatility - wider bands
            settings['bb_std'] = min(3.0, settings.get('bb_std', 2.0) + 0.5)
        elif volatility_state in ['LOW', 'EXTREME_LOW']:
            # Low volatility - tighter bands
            settings['bb_std'] = max(1.5, settings.get('bb_std', 2.0) - 0.3)
        
        # ===== 5. VOLUME THRESHOLD ADJUSTMENTS =====
        if volatility_state in ['HIGH', 'EXTREME_HIGH']:
            # High volatility - need higher volume to confirm
            settings['volume_spike_threshold'] = 2.0
        else:
            settings['volume_spike_threshold'] = 1.5
        
        # ===== 6. ADX THRESHOLD ADJUSTMENTS =====
        settings['adx_trend_threshold'] = 25
        if trending_score > 0.7:
            settings['adx_trend_threshold'] = 20  # Lower threshold in strong trends
        
        # ===== 7. ENABLE/DISABLE INDICATORS BASED ON REGIME =====
        if 'TREND' in regime_name:
            # Trending: enable trend-following indicators
            settings['enable_adx'] = True
            settings['enable_psar'] = True
            settings['enable_ichimoku'] = True
            settings['enable_elder_ray'] = True
        elif 'RANGING' in regime_name:
            # Ranging: enable mean reversion indicators
            settings['enable_cmf'] = True
            settings['enable_mfi'] = True
            settings['enable_cci'] = True
            settings['enable_williams_r'] = True
        elif 'VOLATILE' in regime_name:
            # Volatile: enable volatility indicators
            settings['enable_keltner'] = True
            settings['enable_donchian'] = True
            settings['enable_vortex'] = True
        
        return settings
    
    def _generate_base_settings(self, volatility: str, trend: str, 
                                 volume_profile: str) -> Dict[str, Any]:
        """
        Generate base indicator settings from asset profile
        """
        settings = self.default_settings.copy()
        
        # Adjust based on volatility
        if volatility == 'LOW':
            settings['rsi_period'] = 10
            settings['donchian_period'] = 15
            settings['cci_period'] = 15
            settings['bb_std'] = 1.8
        elif volatility == 'HIGH':
            settings['rsi_period'] = 18
            settings['donchian_period'] = 25
            settings['cci_period'] = 25
            settings['bb_std'] = 2.2
        elif volatility == 'EXTREME':
            settings['rsi_period'] = 21
            settings['donchian_period'] = 30
            settings['cci_period'] = 30
            settings['bb_std'] = 2.5
        
        # Adjust based on trend
        if trend == 'TRENDING':
            settings['macd_fast'] = 8
            settings['macd_slow'] = 17
            settings['enable_psar'] = True
            settings['enable_adx'] = True
            settings['enable_ichimoku'] = True
        else:
            settings['enable_cmf'] = True
            settings['enable_mfi'] = True
            settings['enable_cci'] = True
        
        # Adjust based on volume
        if volume_profile in ['VERY_LIQUID', 'LIQUID']:
            settings['volume_spike_threshold'] = 1.8
        elif volume_profile == 'LOW':
            settings['volume_spike_threshold'] = 1.2
        
        return settings
    
    # ==================== REGIME-SPECIFIC PRESETS ====================
    
    def get_regime_preset(self, regime_name: str) -> Dict[str, Any]:
        """
        Get preset indicator settings for a specific regime
        """
        presets = {
            'STRONG_BULL_TREND': {
                'rsi_oversold': 40,
                'rsi_overbought': 85,
                'rsi_period': 10,
                'macd_fast': 8,
                'macd_slow': 17,
                'ema_fast': 5,
                'ema_medium': 13,
                'ema_slow': 34,
                'bb_std': 2.0,
                'adx_trend_threshold': 20,
            },
            'BULL_TREND': {
                'rsi_oversold': 40,
                'rsi_overbought': 80,
                'rsi_period': 12,
                'macd_fast': 10,
                'macd_slow': 20,
                'ema_fast': 8,
                'ema_medium': 16,
                'ema_slow': 40,
                'bb_std': 2.0,
            },
            'RANGING_NEUTRAL': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'ema_fast': 8,
                'ema_medium': 21,
                'ema_slow': 50,
                'bb_std': 2.2,
            },
            'BEAR_TREND': {
                'rsi_oversold': 20,
                'rsi_overbought': 65,
                'rsi_period': 12,
                'macd_fast': 10,
                'macd_slow': 20,
                'ema_fast': 8,
                'ema_medium': 16,
                'ema_slow': 40,
                'bb_std': 2.0,
            },
            'STRONG_BEAR_TREND': {
                'rsi_oversold': 15,
                'rsi_overbought': 60,
                'rsi_period': 10,
                'macd_fast': 8,
                'macd_slow': 17,
                'ema_fast': 5,
                'ema_medium': 13,
                'ema_slow': 34,
                'bb_std': 2.0,
                'adx_trend_threshold': 20,
            },
            'VOLATILE_EXPANSION': {
                'rsi_period': 10,
                'bb_std': 2.5,
                'volume_spike_threshold': 2.0,
                'atr_period': 10,
                'enable_keltner': True,
                'enable_donchian': True,
            },
        }
        
        # Find matching preset
        for key in presets:
            if key in regime_name:
                return presets[key]
        
        return {}
    
    # ==================== UTILITY METHODS ====================
    
    def _create_default_profile(self, symbol: str) -> AssetProfile:
        """Create default profile for assets with insufficient data"""
        return AssetProfile(
            symbol=symbol,
            volatility_profile='MEDIUM',
            trend_profile='MEAN_REVERTING',
            volume_profile='MODERATE',
            optimal_timeframes=['5m', '15m', '30m'],
            indicator_settings=self.default_settings.copy(),
            regime_settings={}
        )
    
    def clear_cache(self):
        """Clear all cached asset profiles"""
        self.asset_profiles.clear()
        log.info("DynamicIndicatorConfig cache cleared")
    
    def get_asset_summary(self, symbol: str) -> Optional[Dict]:
        """Get summary of asset profile"""
        if symbol not in self.asset_profiles:
            return None
        
        profile = self.asset_profiles[symbol]
        return {
            'symbol': profile.symbol,
            'volatility': profile.volatility_profile,
            'trend': profile.trend_profile,
            'volume': profile.volume_profile,
            'timeframes': profile.optimal_timeframes,
            'indicator_count': len(profile.indicator_settings)
        }