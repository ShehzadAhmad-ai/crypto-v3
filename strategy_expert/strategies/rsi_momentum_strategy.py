"""
Strategy Name: RSI Momentum (Advanced)
Description: Momentum trading using RSI oversold/overbought conditions with divergence detection
Logic:
- Buy when RSI oversold (<30) with bullish divergence or reversal pattern
- Sell when RSI overbought (>70) with bearish divergence or reversal pattern
- RSI divergence detection (hidden and regular)
- Momentum shift detection
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class RSIMomentumStrategy(BaseStrategy):
    """
    Advanced RSI Momentum Strategy
    - Divergence detection (price vs RSI)
    - Oversold/Overbought with confirmation
    - Dynamic RSI periods based on timeframe
    - Volume confirmation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "RSI Momentum"
        self.description = "Advanced momentum trading using RSI with divergence detection"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # RSI settings
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Divergence detection lookback
        self.divergence_lookback = 20
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'volume_ratio', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trading signal based on RSI momentum with divergence detection
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
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Get RSI history for divergence detection
        rsi_prev = indicators.get('rsi_prev', rsi)
        rsi_prev2 = indicators.get('rsi_prev2', rsi_prev)
        
        # ===== 5. DETECT DIVERGENCE =====
        bullish_divergence = False
        bearish_divergence = False
        
        if len(df) >= self.divergence_lookback:
            price_array = df['close'].iloc[-self.divergence_lookback:].values
            rsi_array = df['rsi'].iloc[-self.divergence_lookback:].values if 'rsi' in df else None
            
            if rsi_array is not None:
                # Bullish divergence: price lower low, RSI higher low
                price_lows = self._find_lows(price_array)
                rsi_lows = self._find_lows(rsi_array)
                
                if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                    if price_lows[-1] < price_lows[-2] and rsi_lows[-1] > rsi_lows[-2]:
                        bullish_divergence = True
                
                # Bearish divergence: price higher high, RSI lower high
                price_highs = self._find_highs(price_array)
                rsi_highs = self._find_highs(rsi_array)
                
                if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                    if price_highs[-1] > price_highs[-2] and rsi_highs[-1] < rsi_highs[-2]:
                        bearish_divergence = True
        
        # ===== 6. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        # Condition 1: RSI oversold
        if rsi < self.rsi_oversold:
            bullish_score += 3
            bullish_reasons.append(f"RSI oversold: {rsi:.1f} < {self.rsi_oversold}")
        
        # Condition 2: RSI turning up from oversold
        elif rsi > rsi_prev and rsi_prev < self.rsi_oversold:
            bullish_score += 2
            bullish_reasons.append(f"RSI turning up from oversold: {rsi:.1f} > {rsi_prev:.1f}")
        
        # Condition 3: Bullish divergence
        if bullish_divergence:
            bullish_score += 4
            bullish_reasons.append("Bullish RSI divergence detected")
        
        # Condition 4: Volume confirmation
        if volume_ratio > 1.5 and bullish_score > 0:
            bullish_score += 1
            bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 7. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        # Condition 1: RSI overbought
        if rsi > self.rsi_overbought:
            bearish_score += 3
            bearish_reasons.append(f"RSI overbought: {rsi:.1f} > {self.rsi_overbought}")
        
        # Condition 2: RSI turning down from overbought
        elif rsi < rsi_prev and rsi_prev > self.rsi_overbought:
            bearish_score += 2
            bearish_reasons.append(f"RSI turning down from overbought: {rsi:.1f} < {rsi_prev:.1f}")
        
        # Condition 3: Bearish divergence
        if bearish_divergence:
            bearish_score += 4
            bearish_reasons.append("Bearish RSI divergence detected")
        
        # Condition 4: Volume confirmation
        if volume_ratio > 1.5 and bearish_score > 0:
            bearish_score += 1
            bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 8. DETERMINE ACTION =====
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
                indicators_used=['rsi', 'volume_ratio', 'atr']
            )
        
        return None

    def _find_lows(self, array: np.ndarray) -> list:
        """Find local lows in array"""
        lows = []
        for i in range(1, len(array) - 1):
            if array[i] < array[i-1] and array[i] < array[i+1]:
                lows.append(array[i])
        return lows
    
    def _find_highs(self, array: np.ndarray) -> list:
        """Find local highs in array"""
        highs = []
        for i in range(1, len(array) - 1):
            if array[i] > array[i-1] and array[i] > array[i+1]:
                highs.append(array[i])
        return highs

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """
        Calculate entry, stop loss, and take profit levels
        """
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        rsi = indicators.get('rsi', 50)
        
        # RSI-based multipliers
        if action == 'BUY':
            rsi_distance = (self.rsi_oversold - rsi) / self.rsi_oversold
            risk_multiplier = max(0.8, min(1.5, 1.0 + rsi_distance * 2))
        else:
            rsi_distance = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            risk_multiplier = max(0.8, min(1.5, 1.0 + rsi_distance * 2))
        
        if action == 'BUY':
            entry = current_price
            stop_loss = entry - (atr * 1.5 * risk_multiplier)
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.0)
        
        else:  # SELL
            entry = current_price
            stop_loss = entry + (atr * 1.5 * risk_multiplier)
            risk = stop_loss - entry
            take_profit = entry - (risk * 2.0)
        
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