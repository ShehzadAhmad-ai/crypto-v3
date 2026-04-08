"""
Strategy Name: Whale Liquidation (S07 - Advanced)
Description: Detect liquidation cascades and trade the momentum
Logic:
- Volume spike detection
- Price acceleration detection
- Momentum confirmation
- Whale activity identification
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class WhaleLiquidationStrategy(BaseStrategy):
    """
    Advanced Whale Liquidation Strategy
    - Liquidation cascade detection
    - Volume acceleration analysis
    - Momentum surge detection
    - Price velocity measurement
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Whale Liquidation"
        self.description = "Advanced detection of liquidation cascades and momentum trading"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Volume thresholds
        self.extreme_volume_threshold = 2.0
        self.volume_spike_threshold = 1.5
        
        # Momentum thresholds
        self.momentum_acceleration = 1.5
        
        # Confidence thresholds
        self.buy_confidence = 0.87
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'ema_21', 'rsi', 'macd_histogram', 'volume_ratio', 'obv', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate whale liquidation signal
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
        ema_21 = indicators.get('ema_21', current_price)
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        obv = indicators.get('obv', 0)
        obv_prev = indicators.get('obv_prev', obv)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Get previous MACD value
        macd_hist_prev = indicators.get('macd_histogram_prev', macd_hist)
        
        # Calculate price acceleration
        price_acceleration = 0
        if len(df) >= 3:
            price_change_1 = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            price_change_2 = (df['close'].iloc[-2] - df['close'].iloc[-3]) / df['close'].iloc[-3]
            price_acceleration = price_change_1 - price_change_2 if abs(price_change_2) > 0 else 0
        
        # ===== 5. BULLISH SETUP (Short liquidations) =====
        bullish_score = 0
        bullish_reasons = []
        
        if current_price > ema_200 and current_price > ema_21:
            # Extreme volume
            if volume_ratio > self.extreme_volume_threshold:
                bullish_score += 3
                bullish_reasons.append(f"Extreme volume: {volume_ratio:.1f}x (potential liquidation)")
            
            # Volume spike
            elif volume_ratio > self.volume_spike_threshold:
                bullish_score += 2
                bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            # MACD acceleration
            if macd_hist > 0 and macd_hist > macd_hist_prev * self.momentum_acceleration:
                bullish_score += 2
                bullish_reasons.append("MACD accelerating")
            
            # Price acceleration
            if price_acceleration > 0.01:
                bullish_score += 2
                bullish_reasons.append(f"Price surge acceleration: {price_acceleration:.2%}")
            
            # RSI strength
            if rsi > 60:
                bullish_score += 1
                bullish_reasons.append(f"RSI strong: {rsi:.1f}")
            
            # OBV surge
            if obv > obv_prev * 1.1:
                bullish_score += 1
                bullish_reasons.append("OBV surging")
        
        # ===== 6. BEARISH SETUP (Long liquidations) =====
        bearish_score = 0
        bearish_reasons = []
        
        if current_price < ema_200 and current_price < ema_21:
            if volume_ratio > self.extreme_volume_threshold:
                bearish_score += 3
                bearish_reasons.append(f"Extreme volume: {volume_ratio:.1f}x (potential liquidation)")
            elif volume_ratio > self.volume_spike_threshold:
                bearish_score += 2
                bearish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if macd_hist < 0 and macd_hist < macd_hist_prev * self.momentum_acceleration:
                bearish_score += 2
                bearish_reasons.append("MACD accelerating down")
            
            if price_acceleration < -0.01:
                bearish_score += 2
                bearish_reasons.append(f"Price drop acceleration: {abs(price_acceleration):.2%}")
            
            if rsi < 40:
                bearish_score += 1
                bearish_reasons.append(f"RSI weak: {rsi:.1f}")
            
            if obv < obv_prev * 0.9:
                bearish_score += 1
                bearish_reasons.append("OBV plunging")
        
        # ===== 7. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 5
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.6 + (bullish_score - min_score) * 0.06)
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.6 + (bearish_score - min_score) * 0.06)
            reasons = bearish_reasons[:5]
        
        # ===== 8. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 9. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['volume_ratio', 'macd_histogram', 'rsi', 'obv', 'ema_200', 'ema_21']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = entry - (atr * 1.2)
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = entry + (atr * 1.2)
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