"""
Strategy Name: EMA Trend (Advanced)
Description: Trend following using EMA alignment with dynamic thresholds
Logic: 
- Buy when price > fast EMA > medium EMA > slow EMA (bullish stack) with volume confirmation
- Sell when price < fast EMA < medium EMA < slow EMA (bearish stack)
- EMA crossovers with momentum confirmation
- Dynamic exit based on trend strength
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class EMATrendStrategy(BaseStrategy):
    """
    Advanced EMA Trend Following Strategy
    - Dynamic EMA periods based on timeframe
    - Trend strength detection with ADX
    - Volume confirmation for entries
    - ATR-based trailing stop
    """
    
    def __init__(self):
        super().__init__()
        self.name = "EMA Trend"
        self.description = "Advanced trend following using EMA alignment with dynamic thresholds"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # EMA periods (customizable)
        self.ema_fast = 8
        self.ema_medium = 21
        self.ema_slow = 50
        self.ema_trend = 200
        
        # Trend strength threshold (percentage separation)
        self.trend_strength_threshold = 0.5
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_8', 'ema_21', 'ema_50', 'ema_200',
            'rsi', 'adx', 'volume_ratio', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trading signal based on EMA alignment with advanced confirmation
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
        
        # ===== 4. ADAPT THRESHOLDS BASED ON REGIME =====
        strength_threshold = self.trend_strength_threshold
        
        if 'STRONG' in regime:
            strength_threshold = 1.0
        elif 'WEAK' in regime:
            strength_threshold = 0.3
        
        # ===== 5. EXTRACT INDICATOR VALUES =====
        ema_fast_val = indicators.get('ema_8', 0)
        ema_medium_val = indicators.get('ema_21', 0)
        ema_slow_val = indicators.get('ema_50', 0)
        ema_trend_val = indicators.get('ema_200', 0)
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 20)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Get previous values for crossover detection
        ema_fast_prev = indicators.get('ema_8_prev', ema_fast_val)
        ema_medium_prev = indicators.get('ema_21_prev', ema_medium_val)
        
        # ===== 6. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        # Perfect bullish EMA stack (all EMAs aligned)
        if current_price > ema_fast_val > ema_medium_val > ema_slow_val > ema_trend_val:
            bullish_score += 4
            bullish_reasons.append("Perfect bullish EMA alignment (all EMAs stacked)")
            
            # Calculate separation strength
            fast_to_medium = (ema_fast_val - ema_medium_val) / ema_medium_val * 100
            medium_to_slow = (ema_medium_val - ema_slow_val) / ema_slow_val * 100
            
            if fast_to_medium > strength_threshold and medium_to_slow > strength_threshold:
                bullish_score += 1
                bullish_reasons.append(f"Strong EMA separation ({fast_to_medium:.1f}%, {medium_to_slow:.1f}%)")
        
        # Bullish stack (price above fast and medium)
        elif current_price > ema_fast_val > ema_medium_val:
            bullish_score += 2
            bullish_reasons.append(f"Bullish: Price > EMA{self.ema_fast} > EMA{self.ema_medium}")
        
        # Bullish crossover - fast EMA crossing above medium
        elif ema_fast_val > ema_medium_val and ema_fast_prev <= ema_medium_prev:
            bullish_score += 3
            bullish_reasons.append(f"Bullish EMA crossover: EMA{self.ema_fast} crossed above EMA{self.ema_medium}")
        
        # ADX trend confirmation
        if adx > 25:
            bullish_score += 1
            bullish_reasons.append(f"ADX confirms trend: {adx:.1f}")
        
        # Volume confirmation
        if volume_ratio > 1.3 and bullish_score > 0:
            bullish_score += 1
            bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # RSI momentum confirmation
        if rsi > 55 and bullish_score > 0:
            bullish_score += 1
            bullish_reasons.append(f"RSI bullish momentum: {rsi:.1f}")
        
        # ===== 7. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        # Perfect bearish EMA stack
        if current_price < ema_fast_val < ema_medium_val < ema_slow_val < ema_trend_val:
            bearish_score += 4
            bearish_reasons.append("Perfect bearish EMA alignment (all EMAs stacked)")
            
            fast_to_medium = (ema_medium_val - ema_fast_val) / ema_fast_val * 100
            medium_to_slow = (ema_slow_val - ema_medium_val) / ema_medium_val * 100
            
            if fast_to_medium > strength_threshold and medium_to_slow > strength_threshold:
                bearish_score += 1
                bearish_reasons.append(f"Strong EMA separation ({fast_to_medium:.1f}%, {medium_to_slow:.1f}%)")
        
        # Partial bearish stack
        elif current_price < ema_fast_val < ema_medium_val:
            bearish_score += 2
            bearish_reasons.append(f"Bearish: Price < EMA{self.ema_fast} < EMA{self.ema_medium}")
        
        # Bearish crossover
        elif ema_fast_val < ema_medium_val and ema_fast_prev >= ema_medium_prev:
            bearish_score += 3
            bearish_reasons.append(f"Bearish EMA crossover: EMA{self.ema_fast} crossed below EMA{self.ema_medium}")
        
        if adx > 25:
            bearish_score += 1
            bearish_reasons.append(f"ADX confirms trend: {adx:.1f}")
        
        if volume_ratio > 1.3 and bearish_score > 0:
            bearish_score += 1
            bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        if rsi < 45 and bearish_score > 0:
            bearish_score += 1
            bearish_reasons.append(f"RSI bearish momentum: {rsi:.1f}")
        
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
            # Boost confidence if aligned with regime
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
                indicators_used=['ema_8', 'ema_21', 'ema_50', 'ema_200', 'rsi', 'adx', 'volume_ratio']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """
        Calculate advanced entry, stop loss, and take profit levels
        Uses dynamic pullback entries and ATR-based stops
        """
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        ema_fast = indicators.get('ema_8', current_price)
        ema_medium = indicators.get('ema_21', current_price)
        ema_slow = indicators.get('ema_50', current_price)
        
        # Trend strength for dynamic multipliers
        adx = indicators.get('adx', 20)
        
        if adx > 40:  # Very strong trend
            pullback_depth = 0.3
            risk_multiplier = 1.2
            reward_multiplier = 2.5
        elif adx > 25:  # Moderate trend
            pullback_depth = 0.5
            risk_multiplier = 1.0
            reward_multiplier = 2.0
        else:  # Weak trend
            pullback_depth = 0.7
            risk_multiplier = 0.8
            reward_multiplier = 1.8
        
        if action == 'BUY':
            # Entry on pullback to fast EMA
            if current_price > ema_fast:
                entry = ema_fast + (current_price - ema_fast) * pullback_depth
            else:
                entry = current_price
            
            # Dynamic stop loss based on trend strength
            stop_loss = min(ema_medium * 0.99, entry - (atr * 1.5 * risk_multiplier))
            
            # Take profit with trailing potential
            risk = entry - stop_loss
            take_profit = entry + (risk * reward_multiplier)
            
            # Cap at slow EMA for first target
            if ema_slow > entry:
                take_profit = min(take_profit, ema_slow)
        
        else:  # SELL
            if current_price < ema_fast:
                entry = ema_fast - (ema_fast - current_price) * pullback_depth
            else:
                entry = current_price
            
            stop_loss = max(ema_medium * 1.01, entry + (atr * 1.5 * risk_multiplier))
            
            risk = stop_loss - entry
            take_profit = entry - (risk * reward_multiplier)
            
            if ema_slow < entry:
                take_profit = max(take_profit, ema_slow)
        
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