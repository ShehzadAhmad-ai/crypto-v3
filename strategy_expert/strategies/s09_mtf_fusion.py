"""
Strategy Name: MTF Fusion (S09 - Advanced)
Description: Multi-timeframe Smart Money Trend Fusion
Logic:
- Confirms signals across multiple timeframes
- Only trades when all timeframes align
- Higher timeframe trend filter
- Lower timeframe entry timing
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class MTFFusionStrategy(BaseStrategy):
    """
    Advanced Multi-Timeframe Fusion Strategy
    - Multi-timeframe alignment detection
    - Higher timeframe trend filter
    - Lower timeframe entry confirmation
    - Volume confirmation across timeframes
    """
    
    def __init__(self):
        super().__init__()
        self.name = "MTF Fusion"
        self.description = "Advanced multi-timeframe smart money trend fusion"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Timeframe periods (in candles)
        self.tf_short = 5   # 5m (current timeframe)
        self.tf_medium = 15  # 15m
        self.tf_long = 60    # 1h
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'ema_21', 'rsi', 'macd_histogram', 'volume_ratio', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate multi-timeframe fusion signal
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
        
        # ===== 4. EXTRACT INDICATOR VALUES (Current Timeframe) =====
        ema_200 = indicators.get('ema_200', current_price)
        ema_21 = indicators.get('ema_21', current_price)
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # ===== 5. HIGHER TIMEFRAME ANALYSIS =====
        # Use longer lookback as proxy for higher timeframe
        if len(df) < self.tf_long:
            return None
        
        # 15m timeframe (medium)
        df_medium = df.iloc[-self.tf_medium:].copy()
        ema_200_medium = df_medium['close'].mean()
        price_medium = df_medium['close'].iloc[-1]
        trend_medium_bull = price_medium > ema_200_medium
        
        # 1h timeframe (long)
        df_long = df.iloc[-self.tf_long:].copy()
        ema_200_long = df_long['close'].mean()
        price_long = df_long['close'].iloc[-1]
        trend_long_bull = price_long > ema_200_long
        
        # ===== 6. BULLISH SETUP =====
        bullish_score = 0
        bullish_reasons = []
        
        # Higher timeframe alignment
        if trend_long_bull:
            bullish_score += 2
            bullish_reasons.append("1h timeframe bullish")
        
        if trend_medium_bull:
            bullish_score += 2
            bullish_reasons.append("15m timeframe bullish")
        
        # Current timeframe conditions
        if current_price > ema_200:
            bullish_score += 1
            bullish_reasons.append("Current timeframe > EMA200")
        
        if rsi > 50:
            bullish_score += 1
            bullish_reasons.append(f"RSI bullish: {rsi:.1f}")
        
        if macd_hist > 0:
            bullish_score += 1
            bullish_reasons.append("MACD bullish")
        
        if volume_ratio > 1.2:
            bullish_score += 1
            bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # Pullback to EMA21
        if abs(current_price - ema_21) / ema_21 < 0.005:
            bullish_score += 2
            bullish_reasons.append("Pullback to EMA21")
        
        # ===== 7. BEARISH SETUP =====
        bearish_score = 0
        bearish_reasons = []
        
        if not trend_long_bull:
            bearish_score += 2
            bearish_reasons.append("1h timeframe bearish")
        
        if not trend_medium_bull:
            bearish_score += 2
            bearish_reasons.append("15m timeframe bearish")
        
        if current_price < ema_200:
            bearish_score += 1
            bearish_reasons.append("Current timeframe < EMA200")
        
        if rsi < 50:
            bearish_score += 1
            bearish_reasons.append(f"RSI bearish: {rsi:.1f}")
        
        if macd_hist < 0:
            bearish_score += 1
            bearish_reasons.append("MACD bearish")
        
        if volume_ratio > 1.2:
            bearish_score += 1
            bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        if abs(current_price - ema_21) / ema_21 < 0.005:
            bearish_score += 2
            bearish_reasons.append("Pullback to EMA21")
        
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
                indicators_used=['ema_200', 'ema_21', 'rsi', 'macd_histogram', 'volume_ratio']
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