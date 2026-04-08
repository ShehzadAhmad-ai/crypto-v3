"""
Strategy Name: Volumetric Accumulation Pressure (Advanced)
Description: Detects accumulation phases before explosive moves
Logic:
- Tight range detection with OBV divergence
- Volume growth analysis
- RSI momentum confirmation
- Accumulation pressure scoring
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class VolumetricAccumulationPressureStrategy(BaseStrategy):
    """
    Advanced Volumetric Accumulation Pressure Strategy
    - OBV divergence detection
    - Volume pressure analysis
    - Accumulation zone identification
    - Breakout probability scoring
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Volumetric Accumulation Pressure"
        self.description = "Advanced detection of accumulation phases before explosive moves"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Accumulation thresholds
        self.tight_range_threshold = 0.04
        self.volume_growth_threshold = 1.2
        self.volume_spike_threshold = 1.5
        
        # RSI thresholds
        self.rsi_accumulation_min = 55
        self.rsi_distribution_max = 45
        
        # Confidence thresholds
        self.buy_confidence = 0.89
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'volume', 'volume_avg', 'range_high', 'range_low',
            'obv', 'obv_ma', 'atr', 'volume_ratio'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate signal based on volumetric accumulation pressure
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
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        range_high = indicators.get('range_high', current_price * 1.02)
        range_low = indicators.get('range_low', current_price * 0.98)
        obv = indicators.get('obv', 0)
        obv_ma = indicators.get('obv_ma', 0)
        
        # ===== 5. CALCULATE RANGE SIZE =====
        range_size = (range_high - range_low) / current_price
        tight_range = range_size < self.tight_range_threshold
        
        # ===== 6. DETECT ACCUMULATION PRESSURE =====
        obv_rising = obv > obv_ma
        volume_growth = volume_ratio > self.volume_growth_threshold
        volume_spike = volume_ratio > self.volume_spike_threshold
        
        # ===== 7. BULLISH ACCUMULATION =====
        bullish_score = 0
        bullish_reasons = []
        
        if tight_range:
            bullish_score += 2
            bullish_reasons.append(f"Price in tight accumulation range ({range_size:.2%})")
        
        if obv_rising:
            bullish_score += 3
            bullish_reasons.append("OBV rising → buy pressure building")
        
        if volume_growth:
            bullish_score += 2
            bullish_reasons.append(f"Gradual volume expansion: {volume_ratio:.1f}x")
        
        if volume_spike and not volume_growth:
            bullish_score += 1
            bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
        
        if rsi > self.rsi_accumulation_min:
            bullish_score += 2
            bullish_reasons.append(f"RSI momentum ignition: {rsi:.1f}")
        
        # Check for accumulation phase ending
        if tight_range and obv_rising and volume_growth:
            bullish_score += 2
            bullish_reasons.append("Accumulation phase likely ending → breakout imminent")
        
        # ===== 8. BEARISH DISTRIBUTION =====
        bearish_score = 0
        bearish_reasons = []
        
        if tight_range:
            bearish_score += 2
            bearish_reasons.append(f"Price in tight distribution range ({range_size:.2%})")
        
        if not obv_rising:
            bearish_score += 3
            bearish_reasons.append("OBV falling → sell pressure building")
        
        if volume_growth:
            bearish_score += 2
            bearish_reasons.append(f"Volume growth: {volume_ratio:.1f}x")
        
        if volume_spike:
            bearish_score += 1
            bearish_reasons.append(f"Volume spike on downside: {volume_ratio:.1f}x")
        
        if rsi < self.rsi_distribution_max:
            bearish_score += 2
            bearish_reasons.append(f"RSI weakening: {rsi:.1f}")
        
        if tight_range and not obv_rising and volume_growth:
            bearish_score += 2
            bearish_reasons.append("Distribution phase detected → breakdown imminent")
        
        # ===== 9. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 5
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 14))
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (bearish_score / 14))
            reasons = bearish_reasons[:5]
        
        # ===== 10. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 11. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['rsi', 'volume_ratio', 'obv', 'range_high', 'range_low']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        range_high = indicators.get('range_high', current_price + atr)
        range_low = indicators.get('range_low', current_price - atr)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = range_low * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 3.0)
        
        else:  # SELL
            entry = current_price
            stop_loss = range_high * 1.01
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