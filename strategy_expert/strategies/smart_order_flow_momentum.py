"""
Strategy Name: Smart Order Flow Momentum (Advanced)
Description: Detects pre-pump setups by analyzing order flow and momentum ignition
Logic:
- Hidden order flow detection (cumulative delta)
- Momentum ignition confirmation
- Volume spike validation
- Institutional order flow tracking
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class SmartOrderFlowMomentumStrategy(BaseStrategy):
    """
    Advanced Smart Order Flow Momentum Strategy
    - Cumulative delta analysis
    - Hidden order flow detection
    - Momentum ignition with volume
    - Institutional flow confirmation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Smart Order Flow Momentum"
        self.description = "Advanced order flow analysis for pre-pump detection"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Order flow thresholds
        self.delta_ratio_threshold = 1.2
        self.volume_spike_threshold = 1.5
        
        # RSI thresholds
        self.rsi_bullish_min = 55
        self.rsi_bearish_max = 45
        
        # Confidence thresholds
        self.buy_confidence = 0.90
        self.sell_confidence = 0.88
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'macd_histogram', 'volume', 'volume_avg',
            'cum_buy_delta', 'cum_sell_delta', 'hidden_support', 'hidden_resistance',
            'atr', 'volume_ratio'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate signal based on order flow and momentum ignition
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
        macd_hist = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        cum_buy_delta = indicators.get('cum_buy_delta', 0)
        cum_sell_delta = indicators.get('cum_sell_delta', 0)
        hidden_support = indicators.get('hidden_support', 0)
        hidden_resistance = indicators.get('hidden_resistance', 0)
        
        # ===== 5. ORDER FLOW ANALYSIS =====
        delta_ratio = cum_buy_delta / (cum_sell_delta + 1)
        bullish_flow = delta_ratio > self.delta_ratio_threshold and current_price > hidden_support
        bearish_flow = cum_sell_delta / (cum_buy_delta + 1) > self.delta_ratio_threshold and current_price < hidden_resistance
        
        # ===== 6. MOMENTUM IGNITION =====
        momentum_bullish = rsi > self.rsi_bullish_min and macd_hist > 0
        momentum_bearish = rsi < self.rsi_bearish_max and macd_hist < 0
        
        # ===== 7. VOLUME SPIKE =====
        volume_spike = volume_ratio > self.volume_spike_threshold
        
        # ===== 8. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        if bullish_flow:
            bullish_score += 3
            bullish_reasons.append(f"Bullish order flow (delta ratio: {delta_ratio:.2f})")
        
        if momentum_bullish:
            bullish_score += 3
            bullish_reasons.append(f"Momentum ignition: RSI={rsi:.1f}, MACD positive")
        
        if volume_spike:
            bullish_score += 2
            bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
        
        # Check for hidden support
        if current_price > hidden_support and hidden_support > 0:
            bullish_score += 1
            bullish_reasons.append(f"Price above hidden support: {hidden_support:.2f}")
        
        # ===== 9. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        if bearish_flow:
            bearish_score += 3
            bearish_reasons.append(f"Bearish order flow (delta ratio: {cum_sell_delta/cum_buy_delta:.2f})")
        
        if momentum_bearish:
            bearish_score += 3
            bearish_reasons.append(f"Momentum ignition: RSI={rsi:.1f}, MACD negative")
        
        if volume_spike:
            bearish_score += 2
            bearish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
        
        if current_price < hidden_resistance and hidden_resistance > 0:
            bearish_score += 1
            bearish_reasons.append(f"Price below hidden resistance: {hidden_resistance:.2f}")
        
        # ===== 10. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 5
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 12))
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (bearish_score / 12))
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
                indicators_used=['rsi', 'macd_histogram', 'volume_ratio', 'cum_buy_delta', 'cum_sell_delta']
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
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = hidden_resistance * 1.01
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