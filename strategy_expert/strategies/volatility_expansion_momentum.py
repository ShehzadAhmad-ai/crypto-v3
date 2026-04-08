"""
Strategy Name: Volatility Expansion Momentum (Advanced)
Description: Trades strong moves when volatility and momentum expand together
Logic:
- ATR expansion detection
- ADX trend strength confirmation
- Volume confirmation
- Momentum alignment
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class VolatilityExpansionMomentumStrategy(BaseStrategy):
    """
    Advanced Volatility Expansion Momentum Strategy
    - ATR expansion with ADX confirmation
    - Volume expansion analysis
    - DMI directional confirmation
    - Momentum acceleration
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Volatility Expansion Momentum"
        self.description = "Advanced trading when volatility and momentum expand together"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Expansion thresholds
        self.atr_expansion_threshold = 1.3
        self.adx_trend_threshold = 25
        self.adx_strong_threshold = 40
        self.volume_confirmation_threshold = 1.5
        
        # RSI thresholds
        self.rsi_bullish_min = 55
        self.rsi_bullish_max = 70
        self.rsi_bearish_max = 45
        
        # Confidence thresholds
        self.buy_confidence = 0.87
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'adx', 'atr', 'atr_avg', 'volume_ratio',
            'dmi_plus', 'dmi_minus'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate signal based on volatility expansion and momentum
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
        adx = indicators.get('adx', 0)
        atr = indicators.get('atr', 0)
        atr_avg = indicators.get('atr_avg', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        dmi_plus = indicators.get('dmi_plus', 0)
        dmi_minus = indicators.get('dmi_minus', 0)
        
        # ===== 5. DETECT VOLATILITY EXPANSION =====
        atr_expansion = atr > atr_avg * self.atr_expansion_threshold if atr_avg > 0 else False
        
        # ===== 6. DETECT TREND STRENGTH =====
        trend_strong = adx > self.adx_trend_threshold
        trend_very_strong = adx > self.adx_strong_threshold
        
        # ===== 7. DETECT DIRECTIONAL EXPANSION =====
        bullish_directional = dmi_plus > dmi_minus
        bearish_directional = dmi_minus > dmi_plus
        
        # ===== 8. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        if atr_expansion:
            bullish_score += 3
            bullish_reasons.append(f"ATR volatility expansion: {atr:.4f} > {atr_avg * self.atr_expansion_threshold:.4f}")
        
        if trend_strong:
            bullish_score += 2
            bullish_reasons.append(f"Trend strength: ADX={adx:.1f}")
        
        if trend_very_strong:
            bullish_score += 1
            bullish_reasons.append(f"Strong trend: ADX={adx:.1f}")
        
        if volume_ratio > self.volume_confirmation_threshold:
            bullish_score += 2
            bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        if self.rsi_bullish_min < rsi < self.rsi_bullish_max:
            bullish_score += 2
            bullish_reasons.append(f"RSI bullish momentum zone: {rsi:.1f}")
        
        if bullish_directional:
            bullish_score += 2
            bullish_reasons.append("Directional expansion: DMI+ > DMI-")
        
        # ===== 9. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        if atr_expansion:
            bearish_score += 3
            bearish_reasons.append(f"Volatility expansion detected: ATR={atr:.4f}")
        
        if trend_strong:
            bearish_score += 2
            bearish_reasons.append(f"Trend strength: ADX={adx:.1f}")
        
        if trend_very_strong:
            bearish_score += 1
            bearish_reasons.append(f"Strong trend: ADX={adx:.1f}")
        
        if volume_ratio > self.volume_confirmation_threshold:
            bearish_score += 2
            bearish_reasons.append(f"Volume confirming selling pressure: {volume_ratio:.1f}x")
        
        if rsi < self.rsi_bearish_max:
            bearish_score += 2
            bearish_reasons.append(f"RSI bearish momentum: {rsi:.1f}")
        
        if bearish_directional:
            bearish_score += 2
            bearish_reasons.append("Directional expansion: DMI- > DMI+")
        
        # ===== 10. DETERMINE ACTION =====
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
        
        # ===== 11. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 12. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['rsi', 'adx', 'atr', 'volume_ratio', 'dmi_plus', 'dmi_minus']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        adx = indicators.get('adx', 20)
        
        # ADX-based multiplier (stronger trend = wider stops)
        trend_multiplier = 1.0 + (adx - 25) / 50 if adx > 25 else 1.0
        trend_multiplier = max(0.8, min(1.5, trend_multiplier))
        
        if action == 'BUY':
            entry = current_price
            stop_loss = entry - (atr * 1.5 * trend_multiplier)
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = entry + (atr * 1.5 * trend_multiplier)
            risk = stop_loss - entry
            take_profit = entry - (risk * 2.5)
        
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