"""
Strategy Name: Trend Momentum (S01 - Advanced)
Description: Multi-layer trend + momentum + volume + volatility strategy
Logic:
- Trend: EMA200, EMA50 alignment
- Momentum: MACD, RSI confirmation
- Volume: VWAP, Volume Ratio
- Volatility: Bollinger Bands position
- Only trades with trend alignment
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class TrendMomentumStrategy(BaseStrategy):
    """
    Advanced Trend Momentum Strategy
    - Multi-layer confirmation system
    - Trend alignment filter
    - Dynamic pullback entries
    - ATR-based position sizing
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Trend Momentum"
        self.description = "Multi-layer trend + momentum + volume + volatility strategy"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # EMA periods
        self.ema_fast = 8
        self.ema_medium = 21
        self.ema_slow = 50
        self.ema_trend = 200
        
        # Score thresholds
        self.min_score = 5
        self.max_score = 10
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_8', 'ema_21', 'ema_50', 'ema_200', 'rsi', 'macd_histogram',
            'vwap', 'bb_position', 'volume_ratio', 'atr', 'adx'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trend momentum signal with multi-layer confirmation
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
        vwap = indicators.get('vwap', current_price)
        bb_position = indicators.get('bb_position', 0.5)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        adx = indicators.get('adx', 20)
        
        # ===== 5. MARKET STATE FILTER =====
        bullish_env = current_price > ema_200 and ema_21 > ema_200 and current_price > vwap
        bearish_env = current_price < ema_200 and ema_21 < ema_200 and current_price < vwap
        
        # ===== 6. BULLISH SETUP =====
        bullish_score = 0
        bullish_reasons = []
        
        if bullish_env:
            # Layer 1: Trend Alignment
            if current_price > ema_8 > ema_21:
                bullish_score += 2
                bullish_reasons.append("Bullish EMA stack")
            
            # Layer 2: Pullback to EMA50 or VWAP
            pullback_to_ema = abs(current_price - ema_50) / ema_50 < 0.005
            pullback_to_vwap = abs(current_price - vwap) / vwap < 0.005
            
            if pullback_to_ema or pullback_to_vwap:
                bullish_score += 2
                bullish_reasons.append("Pullback to EMA50/VWAP")
            
            # Layer 3: RSI in sweet spot
            if 45 <= rsi <= 60:
                bullish_score += 2
                bullish_reasons.append(f"RSI optimal: {rsi:.1f}")
            
            # Layer 4: MACD confirmation
            if macd_hist > 0:
                bullish_score += 2
                bullish_reasons.append("MACD bullish")
            
            # Layer 5: Price at middle BB
            if 0.45 <= bb_position <= 0.55:
                bullish_score += 1
                bullish_reasons.append("Price at middle BB")
            
            # Layer 6: Volume confirmation
            if volume_ratio > 1.2:
                bullish_score += 1
                bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            # Layer 7: ADX trend strength
            if adx > 25:
                bullish_score += 1
                bullish_reasons.append(f"ADX trend strength: {adx:.1f}")
        
        # ===== 7. BEARISH SETUP =====
        bearish_score = 0
        bearish_reasons = []
        
        if bearish_env:
            if current_price < ema_8 < ema_21:
                bearish_score += 2
                bearish_reasons.append("Bearish EMA stack")
            
            pullback_to_ema = abs(current_price - ema_50) / ema_50 < 0.005
            pullback_to_vwap = abs(current_price - vwap) / vwap < 0.005
            
            if pullback_to_ema or pullback_to_vwap:
                bearish_score += 2
                bearish_reasons.append("Pullback to EMA50/VWAP")
            
            if 40 <= rsi <= 55:
                bearish_score += 2
                bearish_reasons.append(f"RSI optimal: {rsi:.1f}")
            
            if macd_hist < 0:
                bearish_score += 2
                bearish_reasons.append("MACD bearish")
            
            if 0.45 <= bb_position <= 0.55:
                bearish_score += 1
                bearish_reasons.append("Price at middle BB")
            
            if volume_ratio > 1.2:
                bearish_score += 1
                bearish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if adx > 25:
                bearish_score += 1
                bearish_reasons.append(f"ADX trend strength: {adx:.1f}")
        
        # ===== 8. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        if bullish_score >= self.min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.6 + (bullish_score - self.min_score) * 0.05)
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= self.min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.6 + (bearish_score - self.min_score) * 0.05)
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
                indicators_used=['ema_8', 'ema_21', 'ema_50', 'ema_200', 'rsi', 'macd_histogram', 'vwap', 'bb_position', 'volume_ratio', 'adx']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        ema_21 = indicators.get('ema_21', current_price)
        vwap = indicators.get('vwap', current_price)
        
        if action == 'BUY':
            # Entry on pullback to EMA21 or VWAP
            entry = min(current_price, ema_21, vwap)
            stop_loss = entry - (atr * 1.5)
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = max(current_price, ema_21, vwap)
            stop_loss = entry + (atr * 1.5)
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