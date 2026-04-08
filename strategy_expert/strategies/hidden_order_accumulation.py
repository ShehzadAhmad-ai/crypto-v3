"""
Strategy Name: Hidden Order Accumulation (Advanced)
Description: Detects quiet accumulation before explosive crypto moves
Logic:
- Tight consolidation detection
- Cumulative delta rising on low volume
- Volume spread analysis
- Hidden institutional buying
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class HiddenOrderAccumulationStrategy(BaseStrategy):
    """
    Advanced Hidden Order Accumulation Strategy
    - Tight consolidation detection
    - Cumulative delta analysis
    - Volume spread analysis
    - Smart money footprint detection
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Hidden Order Accumulation"
        self.description = "Advanced detection of quiet accumulation before explosive moves"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Accumulation thresholds
        self.tight_range_threshold = 0.04  # 4%
        self.volume_growth_threshold = 1.1
        self.delta_ma_ratio = 1.2
        
        # RSI thresholds
        self.rsi_bullish_min = 50
        self.rsi_bearish_max = 50
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'volume', 'volume_avg', 'range_high', 'range_low',
            'cum_delta', 'cum_delta_ma', 'atr', 'volume_ratio'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate signal based on hidden accumulation patterns
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
        volume = indicators.get('volume', 0)
        volume_avg = indicators.get('volume_avg', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        range_high = indicators.get('range_high', current_price * 1.02)
        range_low = indicators.get('range_low', current_price * 0.98)
        cum_delta = indicators.get('cum_delta', 0)
        cum_delta_ma = indicators.get('cum_delta_ma', 0)
        
        # ===== 5. CALCULATE RANGE TIGHTNESS =====
        range_pct = (range_high - range_low) / current_price
        tight_range = range_pct < self.tight_range_threshold
        
        # ===== 6. DETECT ACCUMULATION/DISTRIBUTION =====
        delta_rising = cum_delta > cum_delta_ma * self.delta_ma_ratio
        delta_falling = cum_delta < cum_delta_ma * (1 / self.delta_ma_ratio)
        
        volume_increasing = volume > volume_avg * self.volume_growth_threshold
        
        # ===== 7. BULLISH ACCUMULATION =====
        bullish_score = 0
        bullish_reasons = []
        
        if tight_range:
            bullish_score += 2
            bullish_reasons.append(f"Tight consolidation detected ({range_pct:.2%} range)")
        
        if delta_rising:
            bullish_score += 3
            bullish_reasons.append("Cumulative delta rising → accumulation")
        
        if volume_increasing and not volume_ratio > 1.5:
            bullish_score += 1
            bullish_reasons.append("Volume slowly increasing (quiet accumulation)")
        
        if rsi > self.rsi_bullish_min:
            bullish_score += 1
            bullish_reasons.append(f"RSI healthy momentum: {rsi:.1f}")
        
        # ===== 8. BEARISH DISTRIBUTION =====
        bearish_score = 0
        bearish_reasons = []
        
        if tight_range:
            bearish_score += 2
            bearish_reasons.append(f"Tight consolidation detected ({range_pct:.2%} range)")
        
        if delta_falling:
            bearish_score += 3
            bearish_reasons.append("Cumulative delta falling → distribution")
        
        if volume_increasing:
            bearish_score += 1
            bearish_reasons.append("Volume increasing on down moves")
        
        if rsi < self.rsi_bearish_max:
            bearish_score += 1
            bearish_reasons.append(f"RSI weakening: {rsi:.1f}")
        
        # ===== 9. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 4
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 12))
            reasons = bullish_reasons[:5]
            reasons.append("Potential pre-pump setup")
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (bearish_score / 12))
            reasons = bearish_reasons[:5]
        
        # ===== 10. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 11. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['rsi', 'volume_ratio', 'cum_delta', 'range_high', 'range_low']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        range_high = indicators.get('range_high', current_price + atr)
        range_low = indicators.get('range_low', current_price - atr)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = range_low * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 3.0)  # Higher target for accumulation breakouts
        
        else:  # SELL
            entry = current_price
            stop_loss = range_high * 1.01
            risk = stop_loss - entry
            take_profit = entry - (risk * 3.0)
        
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