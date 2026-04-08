"""
Strategy Name: Volatility Compression Breakout (Advanced)
Description: Detects explosive breakouts after volatility squeeze
Logic:
- Bollinger Band width compression detection
- ATR compression confirmation
- Volume expansion confirmation
- Momentum breakout confirmation
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class VolatilityCompressionBreakoutStrategy(BaseStrategy):
    """
    Advanced Volatility Compression Breakout Strategy
    - Multi-indicator squeeze detection
    - Breakout momentum confirmation
    - Volume expansion analysis
    - Dynamic breakout targets
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Volatility Compression Breakout"
        self.description = "Advanced detection of explosive breakouts after volatility squeeze"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Squeeze thresholds
        self.bb_squeeze_threshold = 0.04  # 4% width
        self.atr_low_ratio = 0.7
        self.volume_breakout_threshold = 1.8
        
        # RSI thresholds
        self.rsi_bullish_min = 55
        self.rsi_bearish_max = 45
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'bb_width', 'bb_upper', 'bb_lower', 'bb_middle',
            'atr', 'atr_avg', 'volume', 'volume_avg', 'volume_ratio'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate signal based on volatility compression breakout
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
        bb_width = indicators.get('bb_width', 0)
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        bb_middle = indicators.get('bb_middle', current_price)
        atr = indicators.get('atr', 0)
        atr_avg = indicators.get('atr_avg', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # ===== 5. DETECT SQUEEZE =====
        bb_squeeze = bb_width < self.bb_squeeze_threshold
        atr_compression = atr < atr_avg * self.atr_low_ratio if atr_avg > 0 else False
        squeeze = bb_squeeze or atr_compression
        
        # ===== 6. DETECT VOLUME EXPANSION =====
        volume_expansion = volume_ratio > self.volume_breakout_threshold
        
        # ===== 7. BULLISH BREAKOUT =====
        bullish_score = 0
        bullish_reasons = []
        
        if squeeze:
            bullish_score += 2
            bullish_reasons.append(f"Volatility squeeze detected (BB width: {bb_width:.4f})")
        
        if volume_expansion:
            bullish_score += 2
            bullish_reasons.append(f"Volume expansion: {volume_ratio:.1f}x")
        
        if current_price > bb_upper:
            bullish_score += 3
            bullish_reasons.append("Price breaking upper Bollinger Band")
        
        if rsi > self.rsi_bullish_min:
            bullish_score += 1
            bullish_reasons.append(f"RSI momentum shift: {rsi:.1f}")
        
        # ===== 8. BEARISH BREAKDOWN =====
        bearish_score = 0
        bearish_reasons = []
        
        if squeeze:
            bearish_score += 2
            bearish_reasons.append(f"Volatility squeeze detected (BB width: {bb_width:.4f})")
        
        if volume_expansion:
            bearish_score += 2
            bearish_reasons.append(f"Volume expansion: {volume_ratio:.1f}x")
        
        if current_price < bb_lower:
            bearish_score += 3
            bearish_reasons.append("Price breaking lower Bollinger Band")
        
        if rsi < self.rsi_bearish_max:
            bearish_score += 1
            bearish_reasons.append(f"RSI bearish momentum: {rsi:.1f}")
        
        # ===== 9. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 4
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 12))
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (bearish_score / 12))
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
                indicators_used=['bb_width', 'bb_upper', 'bb_lower', 'rsi', 'volume_ratio', 'atr']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        bb_middle = indicators.get('bb_middle', current_price)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = bb_lower * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = bb_upper * 1.01
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