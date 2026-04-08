"""
Strategy Name: MACD Crossover (Advanced)
Description: Momentum trading using MACD crossovers with divergence detection
Logic:
- Buy when MACD histogram turns positive (bullish crossover) with volume confirmation
- Sell when MACD histogram turns negative (bearish crossover)
- MACD divergence detection (price makes lower low, MACD makes higher low)
- Momentum acceleration detection
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class MACDCrossoverStrategy(BaseStrategy):
    """
    Advanced MACD Crossover Strategy
    - Divergence detection (hidden and regular)
    - Momentum acceleration tracking
    - Volume confirmation
    - Dynamic exit based on momentum
    """
    
    def __init__(self):
        super().__init__()
        self.name = "MACD Crossover"
        self.description = "Advanced momentum trading using MACD crossovers with divergence detection"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # MACD settings
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Momentum threshold (acceleration factor)
        self.momentum_threshold = 1.5
        
        # Divergence detection lookback
        self.divergence_lookback = 20
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'volume_ratio', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trading signal based on MACD crossovers with advanced confirmation
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
        macd_hist = indicators.get('macd_histogram', 0)
        macd_line = indicators.get('macd', 0)
        macd_signal_line = indicators.get('macd_signal', 0)
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Get previous values
        macd_hist_prev = indicators.get('macd_histogram_prev', macd_hist)
        macd_hist_prev2 = indicators.get('macd_histogram_prev2', macd_hist_prev)
        macd_line_prev = indicators.get('macd_prev', macd_line)
        
        # ===== 5. DETECT DIVERGENCE =====
        bullish_divergence = False
        bearish_divergence = False
        
        if len(df) >= self.divergence_lookback:
            # Price and MACD arrays
            price_array = df['close'].iloc[-self.divergence_lookback:].values
            macd_array = df['macd_histogram'].iloc[-self.divergence_lookback:].values if 'macd_histogram' in df else None
            
            if macd_array is not None:
                # Bullish divergence: price lower low, MACD higher low
                price_lows = self._find_lows(price_array)
                macd_lows = self._find_lows(macd_array)
                
                if len(price_lows) >= 2 and len(macd_lows) >= 2:
                    if price_lows[-1] < price_lows[-2] and macd_lows[-1] > macd_lows[-2]:
                        bullish_divergence = True
                    
                    # Bearish divergence: price higher high, MACD lower high
                    price_highs = self._find_highs(price_array)
                    macd_highs = self._find_highs(macd_array)
                    
                    if len(price_highs) >= 2 and len(macd_highs) >= 2:
                        if price_highs[-1] > price_highs[-2] and macd_highs[-1] < macd_highs[-2]:
                            bearish_divergence = True
        
        # ===== 6. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        # Condition 1: Bullish crossover (histogram turns positive)
        if macd_hist > 0 and macd_hist_prev <= 0:
            bullish_score += 4
            bullish_reasons.append("MACD bullish crossover")
            
            # Stronger if histogram accelerating
            if macd_hist > abs(macd_hist_prev):
                bullish_score += 1
                bullish_reasons.append("MACD histogram accelerating bullish")
        
        # Condition 2: Strong bullish momentum
        elif macd_hist > 0 and abs(macd_hist) > abs(macd_hist_prev) * self.momentum_threshold:
            bullish_score += 3
            bullish_reasons.append("MACD strong bullish momentum")
        
        # Condition 3: MACD line above signal line in positive territory
        elif macd_line > macd_signal_line and macd_line > 0:
            bullish_score += 2
            bullish_reasons.append("MACD above signal line in positive territory")
        
        # Condition 4: Bullish divergence
        if bullish_divergence:
            bullish_score += 3
            bullish_reasons.append("Bullish MACD divergence detected")
        
        # Condition 5: Volume confirmation
        if volume_ratio > 1.5 and bullish_score > 0:
            bullish_score += 1
            bullish_reasons.append(f"High volume confirmation ({volume_ratio:.1f}x)")
        
        # ===== 7. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        # Condition 1: Bearish crossover
        if macd_hist < 0 and macd_hist_prev >= 0:
            bearish_score += 4
            bearish_reasons.append("MACD bearish crossover")
            
            if macd_hist < macd_hist_prev:
                bearish_score += 1
                bearish_reasons.append("MACD histogram accelerating bearish")
        
        # Condition 2: Strong bearish momentum
        elif macd_hist < 0 and abs(macd_hist) > abs(macd_hist_prev) * self.momentum_threshold:
            bearish_score += 3
            bearish_reasons.append("MACD strong bearish momentum")
        
        # Condition 3: MACD line below signal line
        elif macd_line < macd_signal_line and macd_line < 0:
            bearish_score += 2
            bearish_reasons.append("MACD below signal line in negative territory")
        
        # Condition 4: Bearish divergence
        if bearish_divergence:
            bearish_score += 3
            bearish_reasons.append("Bearish MACD divergence detected")
        
        # Condition 5: Volume confirmation
        if volume_ratio > 1.5 and bearish_score > 0:
            bearish_score += 1
            bearish_reasons.append(f"High volume confirmation ({volume_ratio:.1f}x)")
        
        # ===== 8. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 2
        
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
            if 'STRONG' in regime:
                confidence = min(0.95, confidence * 1.05)
                reasons.append("Strong trend regime - increased confidence")
            
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
                indicators_used=['macd', 'macd_histogram', 'macd_signal', 'volume_ratio', 'rsi']
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
        macd_hist = indicators.get('macd_histogram', 0)
        
        # Momentum-based multipliers
        momentum_strength = abs(macd_hist) / 100 if abs(macd_hist) > 0 else 0.5
        momentum_strength = min(1.5, max(0.5, momentum_strength))
        
        if action == 'BUY':
            entry = current_price
            stop_loss = entry - (atr * 1.5 * (1 / momentum_strength))
            risk = entry - stop_loss
            take_profit = entry + (risk * (2.0 + momentum_strength * 0.5))
        
        else:  # SELL
            entry = current_price
            stop_loss = entry + (atr * 1.5 * (1 / momentum_strength))
            risk = stop_loss - entry
            take_profit = entry - (risk * (2.0 + momentum_strength * 0.5))
        
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