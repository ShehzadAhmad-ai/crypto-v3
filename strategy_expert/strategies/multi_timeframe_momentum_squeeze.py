"""
Strategy Name: Multi-Timeframe Momentum Squeeze (Advanced)
Description: Combines short-term squeeze with higher-timeframe momentum
Logic:
- Lower timeframe Bollinger Band squeeze detection
- Higher timeframe momentum (ADX + RSI)
- Volume confirmation on breakout
- Multi-timeframe alignment scoring
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class MultiTimeframeMomentumSqueezeStrategy(BaseStrategy):
    """
    Advanced Multi-Timeframe Momentum Squeeze Strategy
    - Multi-timeframe squeeze detection
    - Momentum alignment scoring
    - Volume breakout confirmation
    - Dynamic target levels
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Multi-Timeframe Momentum Squeeze"
        self.description = "Advanced multi-timeframe squeeze detection for explosive moves"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Squeeze thresholds
        self.squeeze_threshold = 0.04
        self.volume_spike = 1.5
        self.adx_strong = 25
        
        # RSI thresholds
        self.rsi_bullish = 55
        self.rsi_bearish = 45
        
        # Confidence thresholds
        self.buy_confidence = 0.90
        self.sell_confidence = 0.88
        
        # Required indicators
        self.required_indicators = [
            'price', 'bb_width', 'bb_upper', 'bb_lower', 'rsi_htf', 'adx_htf',
            'volume', 'volume_avg', 'atr', 'volume_ratio'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate signal based on multi-timeframe momentum squeeze
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
        rsi_htf = indicators.get('rsi_htf', 50)
        adx_htf = indicators.get('adx_htf', 20)
        bb_width = indicators.get('bb_width', 0)
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        atr = indicators.get('atr', current_price * 0.01)
        
        # ===== 5. ADAPT THRESHOLDS BASED ON REGIME =====
        rsi_bullish_adj = self.rsi_bullish
        rsi_bearish_adj = self.rsi_bearish
        
        if 'BULL' in regime:
            rsi_bullish_adj = 50
            rsi_bearish_adj = 40
        elif 'BEAR' in regime:
            rsi_bullish_adj = 60
            rsi_bearish_adj = 50
        
        # ===== 6. DETECT SQUEEZE =====
        squeeze = bb_width < self.squeeze_threshold
        
        # ===== 7. DETECT HIGHER TIMEFRAME MOMENTUM =====
        momentum_up = adx_htf > self.adx_strong and rsi_htf > rsi_bullish_adj
        momentum_down = adx_htf > self.adx_strong and rsi_htf < rsi_bearish_adj
        
        # ===== 8. VOLUME CONFIRMATION =====
        volume_confirm = volume_ratio > self.volume_spike
        
        # ===== 9. BULLISH SETUP =====
        bullish_score = 0
        bullish_reasons = []
        
        if squeeze:
            bullish_score += 2
            bullish_reasons.append(f"Lower timeframe squeeze detected (BB width: {bb_width:.4f})")
        
        if momentum_up:
            bullish_score += 3
            bullish_reasons.append(f"Higher timeframe bullish momentum (ADX={adx_htf:.1f}, RSI={rsi_htf:.1f})")
        
        if volume_confirm:
            bullish_score += 2
            bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        if current_price > bb_upper:
            bullish_score += 3
            bullish_reasons.append("Price broke upper squeeze band → pump likely")
        
        # ===== 10. BEARISH SETUP =====
        bearish_score = 0
        bearish_reasons = []
        
        if squeeze:
            bearish_score += 2
            bearish_reasons.append(f"Lower timeframe squeeze detected (BB width: {bb_width:.4f})")
        
        if momentum_down:
            bearish_score += 3
            bearish_reasons.append(f"Higher timeframe bearish momentum (ADX={adx_htf:.1f}, RSI={rsi_htf:.1f})")
        
        if volume_confirm:
            bearish_score += 2
            bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        if current_price < bb_lower:
            bearish_score += 3
            bearish_reasons.append("Price broke lower squeeze band → dump likely")
        
        # ===== 11. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 5
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 14))
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (bearish_score / 14))
            reasons = bearish_reasons[:5]
        
        # ===== 12. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            
            # Reduce confidence if squeeze is against the trend
            if action == 'BUY' and regime_bias < -0.3:
                confidence = confidence * 0.8
                reasons.append("Caution: Squeeze breakout against bearish regime")
            elif action == 'SELL' and regime_bias > 0.3:
                confidence = confidence * 0.8
                reasons.append("Caution: Squeeze breakdown against bullish regime")
        
        # ===== 13. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['bb_width', 'bb_upper', 'bb_lower', 'rsi_htf', 'adx_htf', 'volume_ratio']
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
            take_profit = entry + (risk * 3.0)
        
        else:  # SELL
            entry = current_price
            stop_loss = bb_upper * 1.01
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