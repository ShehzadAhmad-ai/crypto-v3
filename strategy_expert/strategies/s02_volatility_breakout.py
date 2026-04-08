"""
Strategy Name: Volatility Breakout (S02 - Advanced)
Description: Breakout after volatility squeeze with momentum confirmation
Logic:
- Bollinger Band squeeze detection
- Price breakout confirmation
- Volume spike validation
- Momentum alignment with OBV and RSI
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Advanced Volatility Breakout Strategy
    - BB squeeze detection with multi-bar confirmation
    - Volume profile analysis
    - OBV divergence confirmation
    - Dynamic breakout targets
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Volatility Breakout"
        self.description = "Advanced breakout after volatility squeeze with momentum confirmation"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Squeeze thresholds
        self.bb_squeeze_threshold = 0.05
        self.volume_breakout_threshold = 1.5
        
        # Confidence thresholds
        self.buy_confidence = 0.87
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'bb_upper', 'bb_lower', 'bb_width', 'volume_ratio',
            'obv', 'rsi', 'macd_histogram', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate volatility breakout signal
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
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        bb_width = indicators.get('bb_width', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        obv = indicators.get('obv', 0)
        obv_prev = indicators.get('obv_prev', obv)
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # ===== 5. DETECT SQUEEZE =====
        squeeze = bb_width < self.bb_squeeze_threshold
        
        # Check for squeeze in last 10 candles
        squeeze_detected = False
        if 'bb_width' in df.columns and len(df) >= 10:
            recent_widths = df['bb_width'].iloc[-10:].values
            squeeze_detected = all(w < self.bb_squeeze_threshold for w in recent_widths[-5:])
        
        # ===== 6. BULLISH BREAKOUT =====
        bullish_score = 0
        bullish_reasons = []
        
        if squeeze_detected:
            bullish_score += 2
            bullish_reasons.append("BB Squeeze detected (last 5 candles)")
        
        if current_price > ema_200 and current_price > bb_upper:
            bullish_score += 3
            bullish_reasons.append("Breakout above upper BB and EMA200")
            
            if macd_hist > 0:
                bullish_score += 1
                bullish_reasons.append("MACD bullish")
            
            if volume_ratio > self.volume_breakout_threshold:
                bullish_score += 2
                bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if obv > obv_prev:
                bullish_score += 1
                bullish_reasons.append("OBV rising")
            
            if rsi > 50:
                bullish_score += 1
                bullish_reasons.append(f"RSI bullish: {rsi:.1f}")
        
        # ===== 7. BEARISH BREAKDOWN =====
        bearish_score = 0
        bearish_reasons = []
        
        if squeeze_detected:
            bearish_score += 2
            bearish_reasons.append("BB Squeeze detected")
        
        if current_price < ema_200 and current_price < bb_lower:
            bearish_score += 3
            bearish_reasons.append("Breakout below lower BB and EMA200")
            
            if macd_hist < 0:
                bearish_score += 1
                bearish_reasons.append("MACD bearish")
            
            if volume_ratio > self.volume_breakout_threshold:
                bearish_score += 2
                bearish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if obv < obv_prev:
                bearish_score += 1
                bearish_reasons.append("OBV falling")
            
            if rsi < 50:
                bearish_score += 1
                bearish_reasons.append(f"RSI bearish: {rsi:.1f}")
        
        # ===== 8. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 5
        
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
                indicators_used=['bb_width', 'bb_upper', 'bb_lower', 'volume_ratio', 'obv', 'rsi', 'macd_histogram']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = bb_lower * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = bb_upper * 1.01
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