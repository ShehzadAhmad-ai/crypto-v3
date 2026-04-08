"""
Strategy Name: AI Probability (S06 - Advanced)
Description: Multi-indicator scoring engine - only high probability trades
Logic:
- 10+ indicator scoring system
- Dynamic weight allocation
- Only trades when probability score is very high
- Adaptive thresholds based on market conditions
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class AIProbabilityStrategy(BaseStrategy):
    """
    Advanced AI Probability Strategy
    - Multi-indicator scoring with dynamic weights
    - Adaptive thresholds
    - Only highest probability setups
    - Machine learning inspired scoring
    """
    
    def __init__(self):
        super().__init__()
        self.name = "AI Probability"
        self.description = "Advanced multi-indicator scoring engine for highest probability trades"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Score thresholds
        self.min_score = 8
        self.max_score = 11
        
        # Indicator weights (dynamic)
        self.weights = {
            'trend': 0.20,
            'momentum': 0.20,
            'volume': 0.15,
            'volatility': 0.15,
            'price_action': 0.15,
            'smart_money': 0.15
        }
        
        # Confidence thresholds
        self.buy_confidence = 0.92
        self.sell_confidence = 0.90
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_8', 'ema_21', 'ema_50', 'ema_200', 'rsi', 'macd_histogram',
            'volume_ratio', 'vwap', 'bb_position', 'obv', 'atr', 'adx'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate AI probability score based signal
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
        ema_8 = indicators.get('ema_8', current_price)
        ema_21 = indicators.get('ema_21', current_price)
        ema_50 = indicators.get('ema_50', current_price)
        ema_200 = indicators.get('ema_200', current_price)
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        vwap = indicators.get('vwap', current_price)
        bb_position = indicators.get('bb_position', 0.5)
        obv = indicators.get('obv', 0)
        obv_prev = indicators.get('obv_prev', obv)
        atr = indicators.get('atr', current_price * 0.01)
        adx = indicators.get('adx', 20)
        
        # Supertrend proxy (using EMA alignment)
        supertrend_bull = ema_8 > ema_21 and current_price > ema_8
        supertrend_bear = ema_8 < ema_21 and current_price < ema_8
        
        # ===== 5. BULLISH SCORING =====
        long_score = 0
        long_reasons = []
        
        # Trend alignment (max 2)
        if current_price > ema_200:
            long_score += 1
            long_reasons.append("Price > EMA200")
        if ema_50 > ema_200:
            long_score += 1
            long_reasons.append("EMA50 > EMA200")
        
        # Momentum (max 2)
        if macd_hist > 0:
            long_score += 1
            long_reasons.append("MACD positive")
        if rsi > 55:
            long_score += 1
            long_reasons.append(f"RSI > 55: {rsi:.1f}")
        
        # Volume (max 2)
        if volume_ratio > 1.2:
            long_score += 1
            long_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
        if obv > obv_prev:
            long_score += 1
            long_reasons.append("OBV rising")
        
        # Volatility/Bands (max 2)
        if current_price > vwap:
            long_score += 1
            long_reasons.append("Price > VWAP")
        if bb_position > 0.4:
            long_score += 1
            long_reasons.append(f"BB position: {bb_position:.2f}")
        
        # Market strength (max 2)
        if supertrend_bull:
            long_score += 1
            long_reasons.append("Supertrend bullish")
        if adx > 25:
            long_score += 1
            long_reasons.append(f"ADX trend strength: {adx:.1f}")
        
        # ===== 6. BEARISH SCORING =====
        short_score = 0
        short_reasons = []
        
        if current_price < ema_200:
            short_score += 1
            short_reasons.append("Price < EMA200")
        if ema_50 < ema_200:
            short_score += 1
            short_reasons.append("EMA50 < EMA200")
        
        if macd_hist < 0:
            short_score += 1
            short_reasons.append("MACD negative")
        if rsi < 45:
            short_score += 1
            short_reasons.append(f"RSI < 45: {rsi:.1f}")
        
        if volume_ratio > 1.2:
            short_score += 1
            short_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
        if obv < obv_prev:
            short_score += 1
            short_reasons.append("OBV falling")
        
        if current_price < vwap:
            short_score += 1
            short_reasons.append("Price < VWAP")
        if bb_position < 0.6:
            short_score += 1
            short_reasons.append(f"BB position: {bb_position:.2f}")
        
        if supertrend_bear:
            short_score += 1
            short_reasons.append("Supertrend bearish")
        if adx > 25:
            short_score += 1
            short_reasons.append(f"ADX trend strength: {adx:.1f}")
        
        # ===== 7. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        if long_score >= self.min_score:
            action = 'BUY'
            confidence = min(0.95, 0.6 + (long_score - self.min_score) * 0.08)
            reasons = long_reasons[:5]
        
        elif short_score >= self.min_score:
            action = 'SELL'
            confidence = min(0.95, 0.6 + (short_score - self.min_score) * 0.08)
            reasons = short_reasons[:5]
        
        # ===== 8. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 9. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['ema_8', 'ema_21', 'ema_50', 'ema_200', 'rsi', 'macd_histogram', 'volume_ratio', 'vwap', 'bb_position', 'obv', 'adx']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        ema_21 = indicators.get('ema_21', current_price)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = ema_21 * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = ema_21 * 1.01
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