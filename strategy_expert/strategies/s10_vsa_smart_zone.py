"""
Strategy Name: VSA Smart Zone (S10 - Advanced)
Description: Volume Spread Analysis with smart supply/demand zones
Logic:
- VSA patterns (No Supply, No Demand, Testing)
- Supply/Demand zone identification
- Volume confirmation
- Trend alignment
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class VSASmartZoneStrategy(BaseStrategy):
    """
    Advanced VSA Smart Zone Strategy
    - Volume Spread Analysis patterns
    - Supply/Demand zone detection
    - Volume profile analysis
    - Smart money footprint identification
    """
    
    def __init__(self):
        super().__init__()
        self.name = "VSA Smart Zone"
        self.description = "Advanced Volume Spread Analysis with smart supply/demand zones"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # VSA thresholds
        self.low_volume_threshold = 0.7
        self.high_volume_threshold = 1.5
        
        # Confidence thresholds
        self.buy_confidence = 0.87
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'rsi', 'volume_ratio', 'atr', 'close', 'open', 'high', 'low'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate VSA smart zone signal
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
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Need at least 10 candles for VSA patterns
        if len(df) < 10:
            return None
        
        # Get recent candles for VSA analysis
        c1 = df.iloc[-1]  # Current
        c2 = df.iloc[-2]  # Previous
        c3 = df.iloc[-3]  # Two bars ago
        
        # Calculate volume average
        vol_avg = df['volume'].iloc[-20:].mean()
        
        # ===== 5. VSA PATTERN DETECTION =====
        # 1. No Supply (bullish)
        no_supply = (
            c2['close'] < c2['open'] and  # Down candle
            c2['volume'] < vol_avg * self.low_volume_threshold and  # Low volume
            c1['close'] > c1['open'] and  # Following up candle
            c1['close'] > c2['high']  # Breaks above
        )
        
        # 2. No Demand (bearish)
        no_demand = (
            c2['close'] > c2['open'] and  # Up candle
            c2['volume'] < vol_avg * self.low_volume_threshold and  # Low volume
            c1['close'] < c1['open'] and  # Following down candle
            c1['close'] < c2['low']  # Breaks below
        )
        
        # 3. Testing After Decline (bullish)
        recent_lows = df['low'].iloc[-10:-1].min()
        testing_decline = (
            c2['low'] < recent_lows and  # New low
            c2['volume'] < vol_avg * self.low_volume_threshold and  # Low volume
            c2['close'] > c2['low'] + (c2['high'] - c2['low']) * 0.7 and  # Closed high
            c1['close'] > c2['close']  # Follow through
        )
        
        # 4. Testing After Rally (bearish)
        recent_highs = df['high'].iloc[-10:-1].max()
        testing_rally = (
            c2['high'] > recent_highs and  # New high
            c2['volume'] < vol_avg * self.low_volume_threshold and  # Low volume
            c2['close'] < c2['high'] - (c2['high'] - c2['low']) * 0.7 and  # Closed low
            c1['close'] < c2['close']  # Follow through
        )
        
        # 5. Buying Climax (bearish reversal)
        buying_climax = (
            volume_ratio > self.high_volume_threshold and  # High volume
            c1['close'] > c1['open'] and  # Up candle
            c1['high'] - c1['close'] < (c1['high'] - c1['low']) * 0.3 and  # Small upper wick
            rsi > 70  # RSI overbought
        )
        
        # 6. Selling Climax (bullish reversal)
        selling_climax = (
            volume_ratio > self.high_volume_threshold and  # High volume
            c1['close'] < c1['open'] and  # Down candle
            c1['close'] - c1['low'] < (c1['high'] - c1['low']) * 0.3 and  # Small lower wick
            rsi < 30  # RSI oversold
        )
        
        # ===== 6. BULLISH VSA PATTERNS =====
        bullish_score = 0
        bullish_reasons = []
        
        if current_price > ema_200:
            if no_supply:
                bullish_score += 4
                bullish_reasons.append("VSA: No Supply")
            
            if testing_decline:
                bullish_score += 4
                bullish_reasons.append("VSA: Testing After Decline")
            
            if selling_climax:
                bullish_score += 3
                bullish_reasons.append("VSA: Selling Climax - Potential reversal")
            
            if rsi > 45 and rsi < 60:
                bullish_score += 1
                bullish_reasons.append(f"RSI optimal: {rsi:.1f}")
        
        # ===== 7. BEARISH VSA PATTERNS =====
        bearish_score = 0
        bearish_reasons = []
        
        if current_price < ema_200:
            if no_demand:
                bearish_score += 4
                bearish_reasons.append("VSA: No Demand")
            
            if testing_rally:
                bearish_score += 4
                bearish_reasons.append("VSA: Testing After Rally")
            
            if buying_climax:
                bearish_score += 3
                bearish_reasons.append("VSA: Buying Climax - Potential reversal")
            
            if rsi > 40 and rsi < 55:
                bearish_score += 1
                bearish_reasons.append(f"RSI optimal: {rsi:.1f}")
        
        # ===== 8. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 4
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.6 + (bullish_score - min_score) * 0.07)
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.6 + (bearish_score - min_score) * 0.07)
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
                indicators_used=['volume_ratio', 'rsi', 'ema_200']
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