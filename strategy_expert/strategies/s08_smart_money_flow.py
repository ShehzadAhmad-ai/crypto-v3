"""
Strategy Name: Smart Money Flow (S08 - Advanced)
Description: Track institutional order flow and trade with smart money
Logic:
- Order flow imbalance detection
- Cumulative delta analysis
- Volume profile confirmation
- Institutional footprint detection
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class SmartMoneyFlowStrategy(BaseStrategy):
    """
    Advanced Smart Money Flow Strategy
    - Institutional order flow tracking
    - Cumulative delta analysis
    - Volume profile imbalance
    - Order block confirmation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Smart Money Flow"
        self.description = "Advanced institutional order flow tracking and smart money trading"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Order flow thresholds
        self.delta_imbalance_threshold = 1.5
        self.volume_confirmation_threshold = 1.3
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'vwap', 'rsi', 'macd_histogram', 'volume_ratio',
            'obv', 'cum_buy_delta', 'cum_sell_delta', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate smart money flow signal
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
        ema_200 = indicators.get('ema_200', current_price)
        vwap = indicators.get('vwap', current_price)
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        obv = indicators.get('obv', 0)
        obv_prev = indicators.get('obv_prev', obv)
        cum_buy_delta = indicators.get('cum_buy_delta', 0)
        cum_sell_delta = indicators.get('cum_sell_delta', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # ===== 5. ORDER FLOW ANALYSIS =====
        delta_ratio = cum_buy_delta / (cum_sell_delta + 1)
        delta_positive = delta_ratio > self.delta_imbalance_threshold
        
        # Order flow based on price action
        if len(df) >= 1:
            delta_price = volume_ratio > self.volume_confirmation_threshold and current_price > df['open'].iloc[-1]
            delta_positive = delta_positive or delta_price
        
        delta_negative = cum_sell_delta / (cum_buy_delta + 1) > self.delta_imbalance_threshold
        if len(df) >= 1:
            delta_negative = delta_negative or (volume_ratio > self.volume_confirmation_threshold and current_price < df['open'].iloc[-1])
        
        # ===== 6. BULLISH SETUP (Smart money buying) =====
        bullish_score = 0
        bullish_reasons = []
        
        if current_price > ema_200 and current_price > vwap:
            if delta_positive:
                bullish_score += 4
                bullish_reasons.append(f"Order flow: heavy buying (delta ratio: {delta_ratio:.2f})")
            
            if volume_ratio > self.volume_confirmation_threshold:
                bullish_score += 2
                bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
            
            if obv > obv_prev:
                bullish_score += 2
                bullish_reasons.append("OBV rising with price")
            
            if macd_hist > 0:
                bullish_score += 1
                bullish_reasons.append("MACD bullish")
            
            if rsi > 50:
                bullish_score += 1
                bullish_reasons.append(f"RSI bullish: {rsi:.1f}")
        
        # ===== 7. BEARISH SETUP (Smart money selling) =====
        bearish_score = 0
        bearish_reasons = []
        
        if current_price < ema_200 and current_price < vwap:
            if delta_negative:
                bearish_score += 4
                bearish_reasons.append(f"Order flow: heavy selling (delta ratio: {cum_sell_delta/cum_buy_delta:.2f})")
            
            if volume_ratio > self.volume_confirmation_threshold:
                bearish_score += 2
                bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
            
            if obv < obv_prev:
                bearish_score += 2
                bearish_reasons.append("OBV falling with price")
            
            if macd_hist < 0:
                bearish_score += 1
                bearish_reasons.append("MACD bearish")
            
            if rsi < 50:
                bearish_score += 1
                bearish_reasons.append(f"RSI bearish: {rsi:.1f}")
        
        # ===== 8. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 6
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.6 + (bullish_score - min_score) * 0.06)
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.6 + (bearish_score - min_score) * 0.06)
            reasons = bearish_reasons[:5]
        
        # ===== 9. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 10. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['cum_buy_delta', 'cum_sell_delta', 'volume_ratio', 'obv', 'rsi', 'macd_histogram']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        vwap = indicators.get('vwap', current_price)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = vwap * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = vwap * 1.01
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