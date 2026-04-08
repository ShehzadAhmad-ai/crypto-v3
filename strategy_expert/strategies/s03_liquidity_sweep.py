"""
Strategy Name: Liquidity Sweep (S03 - Advanced)
Description: Capture reversals after stop hunts and liquidity grabs
Logic:
- Liquidity sweep detection at key levels
- Reversal candle confirmation
- Volume spike validation
- Order block proximity check
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class LiquiditySweepStrategy(BaseStrategy):
    """
    Advanced Liquidity Sweep Strategy
    - Multi-level sweep detection
    - Reversal pattern confirmation
    - Order block proximity analysis
    - Volume absorption detection
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Liquidity Sweep"
        self.description = "Advanced capture of reversals after stop hunts and liquidity grabs"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Volume thresholds
        self.volume_spike_threshold = 1.5
        self.rsi_reversal_threshold = 35
        
        # Sweep detection lookback
        self.sweep_lookback = 10
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'rsi', 'volume_ratio', 'atr', 'close', 'open', 'high', 'low'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate liquidity sweep reversal signal
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
        
        # Get recent price data
        if len(df) < 5:
            return None
        
        recent_lows = df['low'].iloc[-self.sweep_lookback:-1].min()
        recent_highs = df['high'].iloc[-self.sweep_lookback:-1].max()
        current_low = df['low'].iloc[-1]
        current_high = df['high'].iloc[-1]
        
        # ===== 5. DETECT LIQUIDITY SWEEPS =====
        sweep_down = current_low < recent_lows
        sweep_up = current_high > recent_highs
        
        # ===== 6. DETECT REVERSAL CANDLES =====
        bullish_reversal = False
        bearish_reversal = False
        
        if len(df) >= 2:
            prev_close = df['close'].iloc[-2]
            prev_open = df['open'].iloc[-2]
            curr_close = df['close'].iloc[-1]
            curr_open = df['open'].iloc[-1]
            
            # Bullish engulfing
            if prev_close < prev_open and curr_close > curr_open and curr_close > prev_open:
                bullish_reversal = True
            
            # Bearish engulfing
            if prev_close > prev_open and curr_close < curr_open and curr_close < prev_open:
                bearish_reversal = True
        
        # ===== 7. BULLISH SETUP =====
        bullish_score = 0
        bullish_reasons = []
        
        if current_price > ema_200:
            if sweep_down:
                bullish_score += 3
                bullish_reasons.append("Liquidity sweep below detected")
                
                if rsi < self.rsi_reversal_threshold:
                    bullish_score += 2
                    bullish_reasons.append(f"RSI oversold reversal: {rsi:.1f}")
                
                if volume_ratio > self.volume_spike_threshold:
                    bullish_score += 2
                    bullish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
                
                if bullish_reversal:
                    bullish_score += 2
                    bullish_reasons.append("Bullish engulfing pattern")
        
        # ===== 8. BEARISH SETUP =====
        bearish_score = 0
        bearish_reasons = []
        
        if current_price < ema_200:
            if sweep_up:
                bearish_score += 3
                bearish_reasons.append("Liquidity sweep above detected")
                
                if rsi > (100 - self.rsi_reversal_threshold):
                    bearish_score += 2
                    bearish_reasons.append(f"RSI overbought reversal: {rsi:.1f}")
                
                if volume_ratio > self.volume_spike_threshold:
                    bearish_score += 2
                    bearish_reasons.append(f"Volume spike: {volume_ratio:.1f}x")
                
                if bearish_reversal:
                    bearish_score += 2
                    bearish_reasons.append("Bearish engulfing pattern")
        
        # ===== 9. DETERMINE ACTION =====
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
                indicators_used=['rsi', 'volume_ratio', 'ema_200', 'atr']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if len(df) >= self.sweep_lookback:
            sweep_level = df['low'].iloc[-self.sweep_lookback:-1].min() if action == 'BUY' else df['high'].iloc[-self.sweep_lookback:-1].max()
        else:
            sweep_level = current_price * 0.98 if action == 'BUY' else current_price * 1.02
        
        if action == 'BUY':
            entry = current_price
            stop_loss = sweep_level * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = sweep_level * 1.01
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