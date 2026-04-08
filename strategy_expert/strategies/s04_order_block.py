"""
Strategy Name: Order Block (S04 - Advanced)
Description: Trade at institutional order blocks with confirmation
Logic:
- Order block identification
- Price retest confirmation
- Volume validation
- Momentum confirmation
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class OrderBlockStrategy(BaseStrategy):
    """
    Advanced Order Block Strategy
    - Dynamic order block detection
    - Retest confirmation with volume
    - Multiple timeframe alignment
    - ATR-based risk management
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Order Block"
        self.description = "Advanced institutional order block trading with confirmation"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Order block thresholds
        self.ob_proximity_threshold = 0.01  # 1%
        self.volume_confirmation_threshold = 1.3
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'rsi', 'macd_histogram', 'volume_ratio', 'obv', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate order block signal
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
        macd_hist = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        obv = indicators.get('obv', 0)
        obv_prev = indicators.get('obv_prev', obv)
        atr = indicators.get('atr', current_price * 0.01)
        
        # ===== 5. DETECT ORDER BLOCKS =====
        bullish_ob, bearish_ob = self._detect_order_blocks(df)
        
        # ===== 6. CHECK PROXIMITY TO ORDER BLOCKS =====
        bullish_ob_proximity = False
        bearish_ob_proximity = False
        ob_price = 0
        
        if bullish_ob:
            ob_price = bullish_ob
            distance_pct = abs(current_price - ob_price) / current_price
            bullish_ob_proximity = distance_pct < self.ob_proximity_threshold
        
        if bearish_ob:
            ob_price = bearish_ob
            distance_pct = abs(current_price - ob_price) / current_price
            bearish_ob_proximity = distance_pct < self.ob_proximity_threshold
        
        # ===== 7. BULLISH SETUP =====
        bullish_score = 0
        bullish_reasons = []
        
        if bullish_ob_proximity and current_price > ema_200:
            bullish_score += 4
            bullish_reasons.append(f"Bullish OB at {ob_price:.2f} - price retesting")
            
            if 40 <= rsi <= 55:
                bullish_score += 2
                bullish_reasons.append(f"RSI optimal: {rsi:.1f}")
            
            if macd_hist > 0:
                bullish_score += 1
                bullish_reasons.append("MACD bullish")
            
            if volume_ratio > self.volume_confirmation_threshold:
                bullish_score += 1
                bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if obv > obv_prev:
                bullish_score += 1
                bullish_reasons.append("OBV rising")
        
        # ===== 8. BEARISH SETUP =====
        bearish_score = 0
        bearish_reasons = []
        
        if bearish_ob_proximity and current_price < ema_200:
            bearish_score += 4
            bearish_reasons.append(f"Bearish OB at {ob_price:.2f} - price retesting")
            
            if 45 <= rsi <= 60:
                bearish_score += 2
                bearish_reasons.append(f"RSI optimal: {rsi:.1f}")
            
            if macd_hist < 0:
                bearish_score += 1
                bearish_reasons.append("MACD bearish")
            
            if volume_ratio > self.volume_confirmation_threshold:
                bearish_score += 1
                bearish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
            
            if obv < obv_prev:
                bearish_score += 1
                bearish_reasons.append("OBV falling")
        
        # ===== 9. DETERMINE ACTION =====
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
            levels = self.calculate_entry_sl_tp(df, indicators, market_regime, action, ob_price)
            
            return StrategyOutput(
                action=action,
                confidence=confidence,
                entry=levels['entry'],
                stop_loss=levels['stop_loss'],
                take_profit=levels['take_profit'],
                risk_reward=levels['risk_reward'],
                reasons=reasons[:5],
                strategy_name=self.name,
                indicators_used=['rsi', 'macd_histogram', 'volume_ratio', 'obv', 'ema_200']
            )
        
        return None

    def _detect_order_blocks(self, df: pd.DataFrame) -> tuple:
        """Detect bullish and bearish order blocks"""
        if len(df) < 10:
            return None, None
        
        bullish_ob = None
        bearish_ob = None
        
        # Look for strong candles that broke structure
        for i in range(-10, -2):
            candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish OB: strong up candle after consolidation
            if candle['close'] > candle['open'] and (candle['close'] - candle['open']) / candle['open'] > 0.01:
                if next_candle['close'] > candle['high']:
                    bullish_ob = candle['low']
            
            # Bearish OB: strong down candle after consolidation
            if candle['close'] < candle['open'] and (candle['open'] - candle['close']) / candle['open'] > 0.01:
                if next_candle['close'] < candle['low']:
                    bearish_ob = candle['high']
        
        return bullish_ob, bearish_ob

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str, ob_price: float = 0) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = ob_price * 0.99 if ob_price > 0 else entry - (atr * 1.5)
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = ob_price * 1.01 if ob_price > 0 else entry + (atr * 1.5)
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