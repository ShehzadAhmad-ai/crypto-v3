"""
Strategy Name: Market Maker (S05 - Advanced)
Description: Detect fake breakouts and trade the reversal
Logic:
- Fake breakout detection with large wicks
- Reversal candle confirmation
- Volume spike validation
- VWAP and EMA confirmation
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class MarketMakerStrategy(BaseStrategy):
    """
    Advanced Market Maker Manipulation Strategy
    - Fake breakout detection with wick analysis
    - Volume confirmation
    - Reversal pattern detection
    - Multi-timeframe context
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Market Maker"
        self.description = "Advanced detection of fake breakouts and reversal trading"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Fake breakout thresholds
        self.wick_body_ratio = 2.0
        self.volume_spike_threshold = 1.5
        
        # Confidence thresholds
        self.buy_confidence = 0.87
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'vwap', 'rsi', 'volume_ratio', 'atr', 'close', 'open', 'high', 'low'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate market maker trap signal
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
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Need at least 3 candles for pattern detection
        if len(df) < 3:
            return None
        
        # Get recent candles
        c1 = df.iloc[-1]  # Current
        c2 = df.iloc[-2]  # Previous
        c3 = df.iloc[-3]  # Two bars ago
        
        # ===== 5. DETECT FAKE BREAKOUT PATTERNS =====
        recent_low = df['low'].iloc[-10:-1].min() if len(df) >= 10 else c3['low']
        recent_high = df['high'].iloc[-10:-1].max() if len(df) >= 10 else c3['high']
        
        # Fake breakdown detection
        fake_breakdown = False
        if c2['low'] < recent_low:
            wick_size = c2['low'] - min(c2['open'], c2['close'])
            body_size = abs(c2['close'] - c2['open'])
            if wick_size > body_size * self.wick_body_ratio:
                fake_breakdown = True
        
        # Fake breakout detection
        fake_breakout = False
        if c2['high'] > recent_high:
            wick_size = c2['high'] - max(c2['open'], c2['close'])
            body_size = abs(c2['close'] - c2['open'])
            if wick_size > body_size * self.wick_body_ratio:
                fake_breakout = True
        
        # Reversal confirmation
        reversal_up = c1['close'] > c2['close']
        reversal_down = c1['close'] < c2['close']
        
        # ===== 6. BULLISH SETUP (Fake breakdown) =====
        bullish_score = 0
        bullish_reasons = []
        
        if fake_breakdown:
            bullish_score += 3
            bullish_reasons.append("Fake breakdown with large wick (stop hunt)")
            
            if reversal_up:
                bullish_score += 2
                bullish_reasons.append("Reversal candle")
            
            if rsi < 35:
                bullish_score += 2
                bullish_reasons.append(f"RSI oversold: {rsi:.1f}")
            
            if volume_ratio > self.volume_spike_threshold:
                bullish_score += 2
                bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if c1['close'] > vwap:
                bullish_score += 1
                bullish_reasons.append("Closed above VWAP")
        
        # ===== 7. BEARISH SETUP (Fake breakout) =====
        bearish_score = 0
        bearish_reasons = []
        
        if fake_breakout:
            bearish_score += 3
            bearish_reasons.append("Fake breakout with large wick (stop hunt)")
            
            if reversal_down:
                bearish_score += 2
                bearish_reasons.append("Reversal candle")
            
            if rsi > 65:
                bearish_score += 2
                bearish_reasons.append(f"RSI overbought: {rsi:.1f}")
            
            if volume_ratio > self.volume_spike_threshold:
                bearish_score += 2
                bearish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if c1['close'] < vwap:
                bearish_score += 1
                bearish_reasons.append("Closed below VWAP")
        
        # ===== 8. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 6
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.6 + (bullish_score - min_score) * 0.05)
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.6 + (bearish_score - min_score) * 0.05)
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
                indicators_used=['rsi', 'volume_ratio', 'vwap', 'ema_200']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if len(df) >= 10:
            recent_low = df['low'].iloc[-10:-1].min()
            recent_high = df['high'].iloc[-10:-1].max()
        else:
            recent_low = current_price * 0.98
            recent_high = current_price * 1.02
        
        if action == 'BUY':
            entry = current_price
            stop_loss = recent_low * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = recent_high * 1.01
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