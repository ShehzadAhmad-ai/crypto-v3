"""
Strategy Name: Multi-Layer Pre-Pump Detection (Advanced)
Description: Combines hidden accumulation, liquidity sweeps, volatility expansion, and momentum ignition
Logic:
- Layer 1: Hidden accumulation/distribution
- Layer 2: Liquidity sweep confirmation
- Layer 3: Volatility expansion
- Layer 4: Momentum ignition
- Layer 5: Volume confirmation
- All layers must align for signal
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class MultiLayerPrePumpStrategy(BaseStrategy):
    """
    Advanced Multi-Layer Pre-Pump Detection
    - 5-layer confirmation system
    - Dynamic scoring per layer
    - Only high-probability setups
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Multi-Layer Pre-Pump Detection"
        self.description = "Advanced 5-layer confirmation for high-probability pre-pump entries"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Layer thresholds
        self.accumulation_threshold = 1.2
        self.atr_expansion_threshold = 1.3
        self.volume_confirm_threshold = 1.5
        self.rsi_bullish_min = 55
        self.rsi_bearish_max = 45
        self.adx_trend_threshold = 25
        
        # Layer weights
        self.layer_weights = {
            'accumulation': 0.25,
            'liquidity_sweep': 0.20,
            'volatility': 0.15,
            'momentum': 0.15,
            'volume': 0.25
        }
        
        # Confidence thresholds
        self.buy_confidence = 0.92
        self.sell_confidence = 0.90
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'adx', 'atr', 'atr_avg', 'volume', 'volume_avg',
            'cum_buy_delta', 'cum_sell_delta', 'hidden_support', 'hidden_resistance'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate pre-pump signal with 5-layer confirmation
        """
        # ===== 1. ENSURE ALL REQUIRED INDICATORS ARE AVAILABLE =====
        indicators = self.calculator.calculate_missing(df, indicators, self.required_indicators)
        
        # ===== 2. EXTRACT REQUIRED VALUES =====
        current_price = indicators.get('price', 0)
        if current_price == 0:
            return None
        
        # ===== 3. GET REGIME CONTEXT =====
        regime = market_regime.get('regime', 'UNKNOWN')
        regime_bias = market_regime.get('bias_score', 0)
        
        # ===== 4. EXTRACT INDICATOR VALUES =====
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 20)
        atr = indicators.get('atr', 0)
        atr_avg = indicators.get('atr_avg', 0)
        volume = indicators.get('volume', 0)
        volume_avg = indicators.get('volume_avg', 0)
        cum_buy_delta = indicators.get('cum_buy_delta', 0)
        cum_sell_delta = indicators.get('cum_sell_delta', 0)
        hidden_support = indicators.get('hidden_support', 0)
        hidden_resistance = indicators.get('hidden_resistance', 0)
        
        # Get smart money data
        liquidity_sweep_low = False
        liquidity_sweep_high = False
        if module_signals:
            liquidity_sweep_low = module_signals.get('liquidity_sweep_low', False)
            liquidity_sweep_high = module_signals.get('liquidity_sweep_high', False)
        
        # ===== 5. LAYER 1: ACCUMULATION/DISTRIBUTION =====
        layer1_bullish = cum_buy_delta > cum_sell_delta * self.accumulation_threshold and current_price > hidden_support
        layer1_bearish = cum_sell_delta > cum_buy_delta * self.accumulation_threshold and current_price < hidden_resistance
        
        # ===== 6. LAYER 2: LIQUIDITY SWEEP =====
        layer2_bullish = liquidity_sweep_low
        layer2_bearish = liquidity_sweep_high
        
        # ===== 7. LAYER 3: VOLATILITY EXPANSION =====
        atr_expansion = atr > atr_avg * self.atr_expansion_threshold if atr_avg > 0 else False
        layer3_bullish = atr_expansion
        layer3_bearish = atr_expansion
        
        # ===== 8. LAYER 4: MOMENTUM =====
        layer4_bullish = rsi > self.rsi_bullish_min and adx > self.adx_trend_threshold
        layer4_bearish = rsi < self.rsi_bearish_max and adx > self.adx_trend_threshold
        
        # ===== 9. LAYER 5: VOLUME CONFIRMATION =====
        volume_confirm = volume > volume_avg * self.volume_confirm_threshold if volume_avg > 0 else False
        layer5_bullish = volume_confirm
        layer5_bearish = volume_confirm
        
        # ===== 10. CALCULATE LAYER SCORES =====
        bullish_layers = sum([
            layer1_bullish,
            layer2_bullish,
            layer3_bullish,
            layer4_bullish,
            layer5_bullish
        ])
        
        bearish_layers = sum([
            layer1_bearish,
            layer2_bearish,
            layer3_bearish,
            layer4_bearish,
            layer5_bearish
        ])
        
        # ===== 11. BUILD REASONS =====
        bullish_reasons = []
        bearish_reasons = []
        
        if layer1_bullish:
            bullish_reasons.append("Layer 1: Hidden accumulation detected")
        if layer2_bullish:
            bullish_reasons.append("Layer 2: Liquidity sweep below support")
        if layer3_bullish:
            bullish_reasons.append("Layer 3: ATR volatility expansion")
        if layer4_bullish:
            bullish_reasons.append(f"Layer 4: RSI={rsi:.1f}, ADX={adx:.1f} - bullish momentum")
        if layer5_bullish:
            bullish_reasons.append(f"Layer 5: Volume confirmation ({volume/volume_avg:.1f}x)")
        
        if layer1_bearish:
            bearish_reasons.append("Layer 1: Hidden distribution detected")
        if layer2_bearish:
            bearish_reasons.append("Layer 2: Liquidity sweep above resistance")
        if layer3_bearish:
            bearish_reasons.append("Layer 3: ATR volatility expansion")
        if layer4_bearish:
            bearish_reasons.append(f"Layer 4: RSI={rsi:.1f}, ADX={adx:.1f} - bearish momentum")
        if layer5_bearish:
            bearish_reasons.append(f"Layer 5: Volume confirmation ({volume/volume_avg:.1f}x)")
        
        # ===== 12. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_layers = 4  # Need at least 4 layers to fire
        
        if bullish_layers >= min_layers and bullish_layers > bearish_layers:
            action = 'BUY'
            confidence = self.buy_confidence
            reasons = bullish_reasons[:5]
        
        elif bearish_layers >= min_layers and bearish_layers > bullish_layers:
            action = 'SELL'
            confidence = self.sell_confidence
            reasons = bearish_reasons[:5]
        
        # ===== 13. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 14. CALCULATE ENTRY, SL, TP =====
        if action != 'HOLD':
            levels = self.calculate_entry_sl_tp(df, indicators, market_regime, action)
            
            return StrategyOutput(
                action=action,
                confidence=confidence,
                entry=levels['entry'],
                stop_loss=levels['stop_loss'],
                take_profit=levels['take_profit'],
                risk_reward=levels['risk_reward'],
                reasons=reasons[:5],
                strategy_name=self.name,
                indicators_used=['rsi', 'adx', 'atr', 'volume_ratio', 'cum_buy_delta', 'cum_sell_delta']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        hidden_support = indicators.get('hidden_support', current_price - atr)
        hidden_resistance = indicators.get('hidden_resistance', current_price + atr)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = hidden_support * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 3.0)  # Higher target for pre-pump
        
        else:  # SELL
            entry = current_price
            stop_loss = hidden_resistance * 1.01
            risk = stop_loss - entry
            take_profit = entry - (risk * 3.0)
        
        # Calculate risk/reward
        if action == 'BUY':
            risk = entry - stop_loss
            reward = take_profit - entry
        else:
            risk = stop_loss - entry
            reward = entry - take_profit
        
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }