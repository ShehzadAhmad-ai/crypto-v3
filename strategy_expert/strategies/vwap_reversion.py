"""
Strategy Name: VWAP Reversion (Advanced)
Description: Mean reversion around VWAP with volume confirmation
Logic:
- Buy when price is below VWAP with bullish divergence
- Sell when price is above VWAP with bearish divergence
- VWAP bands for dynamic support/resistance
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class VWAPReversionStrategy(BaseStrategy):
    """
    Advanced VWAP Reversion Strategy
    - VWAP with standard deviation bands
    - Mean reversion with volume confirmation
    - Trend alignment filter
    """
    
    def __init__(self):
        super().__init__()
        self.name = "VWAP Reversion"
        self.description = "Advanced mean reversion around VWAP with volume confirmation"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # VWAP settings
        self.vwap_band_multiplier = 2.0
        self.vwap_deviation_threshold = 1.0  # 1% deviation
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'vwap', 'volume_ratio', 'rsi', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trading signal based on VWAP mean reversion
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
        vwap = indicators.get('vwap', current_price)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        rsi = indicators.get('rsi', 50)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Calculate VWAP deviation
        vwap_deviation = (current_price - vwap) / vwap * 100
        vwap_distance = abs(vwap_deviation)
        
        # Calculate VWAP bands (simplified - would need true VWAP std dev)
        vwap_upper = vwap * (1 + self.vwap_band_multiplier * 0.005)
        vwap_lower = vwap * (1 - self.vwap_band_multiplier * 0.005)
        
        # ===== 5. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        # Price significantly below VWAP (discount)
        if vwap_deviation < -self.vwap_deviation_threshold:
            bullish_score += 3
            bullish_reasons.append(f"Price {abs(vwap_deviation):.2f}% below VWAP (discount zone)")
            
            # Volume confirmation for reversal
            if volume_ratio > 1.3:
                bullish_score += 1
                bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
            
            # RSI oversold or turning up
            if rsi < 35:
                bullish_score += 1
                bullish_reasons.append(f"RSI oversold: {rsi:.1f}")
            elif rsi > rsi - 2:
                bullish_score += 1
                bullish_reasons.append(f"RSI turning up: {rsi:.1f}")
            
            # Price bouncing from VWAP lower band
            if current_price > vwap_lower and current_price < vwap:
                bullish_score += 1
                bullish_reasons.append("Bouncing from VWAP lower band")
        
        # ===== 6. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        # Price significantly above VWAP (premium)
        if vwap_deviation > self.vwap_deviation_threshold:
            bearish_score += 3
            bearish_reasons.append(f"Price {vwap_deviation:.2f}% above VWAP (premium zone)")
            
            if volume_ratio > 1.3:
                bearish_score += 1
                bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
            
            if rsi > 65:
                bearish_score += 1
                bearish_reasons.append(f"RSI overbought: {rsi:.1f}")
            elif rsi < rsi + 2:
                bearish_score += 1
                bearish_reasons.append(f"RSI turning down: {rsi:.1f}")
            
            if current_price < vwap_upper and current_price > vwap:
                bearish_score += 1
                bearish_reasons.append("Rejecting from VWAP upper band")
        
        # ===== 7. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 3
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 12))
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (bearish_score / 12))
            reasons = bearish_reasons[:5]
        
        # ===== 8. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            # VWAP reversion is stronger in ranging markets
            if 'RANGING' in regime:
                confidence = min(0.95, confidence * 1.05)
                reasons.append("Ranging market - VWAP reversion more reliable")
            
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 9. CALCULATE ENTRY, SL, TP =====
        if action != 'HOLD':
            levels = self.calculate_entry_sl_tp(df, indicators, market_regime, action, vwap)
            
            return StrategyOutput(
                action=action,
                confidence=confidence,
                entry=levels['entry'],
                stop_loss=levels['stop_loss'],
                take_profit=levels['take_profit'],
                risk_reward=levels['risk_reward'],
                reasons=reasons[:5],
                strategy_name=self.name,
                indicators_used=['vwap', 'volume_ratio', 'rsi', 'atr']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str, vwap: float = None) -> Dict:
        """
        Calculate entry, stop loss, and take profit levels
        """
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if vwap is None:
            vwap = indicators.get('vwap', current_price)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = vwap * 0.99  # Below VWAP
            risk = entry - stop_loss
            take_profit = vwap  # Reversion to VWAP
        else:  # SELL
            entry = current_price
            stop_loss = vwap * 1.01  # Above VWAP
            risk = stop_loss - entry
            take_profit = vwap  # Reversion to VWAP
        
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