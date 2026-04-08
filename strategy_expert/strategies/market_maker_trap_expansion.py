"""
Strategy Name: Market Maker Trap Expansion (Advanced)
Description: Detects false breakouts created by market makers leading to strong directional expansions
Logic:
- Fake breakout above resistance with volume spike
- Fake breakdown below support with volume spike
- Trapped trader detection
- Expansion confirmation with momentum
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class MarketMakerTrapExpansionStrategy(BaseStrategy):
    """
    Advanced Market Maker Trap Strategy
    - False breakout/breakdown detection
    - Trapped volume analysis
    - Momentum expansion confirmation
    - Dynamic trap zones
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Market Maker Trap Expansion"
        self.description = "Advanced false breakout detection with expansion confirmation"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Volume thresholds
        self.volume_spike_threshold = 1.7
        self.trap_volume_threshold = 1.5
        
        # RSI thresholds
        self.rsi_bullish_min = 50
        self.rsi_bearish_max = 50
        
        # Trap detection lookback
        self.trap_lookback = 10
        
        # Confidence thresholds
        self.buy_confidence = 0.88
        self.sell_confidence = 0.86
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'volume', 'volume_avg', 'resistance', 'support',
            'high', 'low', 'close', 'atr', 'volume_ratio', 'adx'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate market maker trap reversal signal
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
        adx = indicators.get('adx', 20)
        
        # Get levels
        resistance = indicators.get('resistance', 0)
        support = indicators.get('support', 0)
        high = indicators.get('high', current_price)
        low = indicators.get('low', current_price)
        close = indicators.get('close', current_price)
        
        # ===== 5. DETECT FALSE BREAKOUT/BREAKDOWN =====
        false_breakout = False
        false_breakdown = False
        
        if resistance > 0:
            fake_breakout = high > resistance and close < resistance
            false_breakout = fake_breakout and volume_ratio > self.volume_spike_threshold
        
        if support > 0:
            fake_breakdown = low < support and close > support
            false_breakdown = fake_breakdown and volume_ratio > self.volume_spike_threshold
        
        # ===== 6. DETECT TRAPPED VOLUME =====
        trapped_bulls, trapped_bears = self._detect_trapped_volume(df, indicators)
        
        # ===== 7. BULLISH TRAP =====
        bullish_score = 0
        bullish_reasons = []
        
        if false_breakout:
            bullish_score += 4
            bullish_reasons.append("False breakout above resistance - market maker trap")
            
            if volume_ratio > self.trap_volume_threshold:
                bullish_score += 1
                bullish_reasons.append(f"Trapped volume detected: {volume_ratio:.1f}x")
            
            if rsi > self.rsi_bullish_min:
                bullish_score += 1
                bullish_reasons.append(f"RSI strength: {rsi:.1f}")
            
            if adx > 25:
                bullish_score += 1
                bullish_reasons.append(f"ADX confirms trend expansion: {adx:.1f}")
        
        if trapped_bulls:
            bullish_score += 2
            bullish_reasons.append("Trapped bulls identified - potential reversal")
        
        # ===== 8. BEARISH TRAP =====
        bearish_score = 0
        bearish_reasons = []
        
        if false_breakdown:
            bearish_score += 4
            bearish_reasons.append("False breakdown below support - bear trap")
            
            if volume_ratio > self.trap_volume_threshold:
                bearish_score += 1
                bearish_reasons.append(f"Trapped volume detected: {volume_ratio:.1f}x")
            
            if rsi < self.rsi_bearish_max:
                bearish_score += 1
                bearish_reasons.append(f"RSI weakness: {rsi:.1f}")
            
            if adx > 25:
                bearish_score += 1
                bearish_reasons.append(f"ADX confirms trend expansion: {adx:.1f}")
        
        if trapped_bears:
            bearish_score += 2
            bearish_reasons.append("Trapped bears identified - potential reversal")
        
        # ===== 9. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 4
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 12))
            reasons = bullish_reasons[:5]
        
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
                indicators_used=['rsi', 'volume_ratio', 'adx', 'atr', 'resistance', 'support']
            )
        
        return None

    def _detect_trapped_volume(self, df: pd.DataFrame, indicators: Dict) -> tuple:
        """Detect trapped bulls and bears based on volume patterns"""
        trapped_bulls = False
        trapped_bears = False
        
        if len(df) < 5:
            return trapped_bulls, trapped_bears
        
        # Check for trapped bulls (breakout failure)
        recent_highs = df['high'].iloc[-5:].max()
        current_close = df['close'].iloc[-1]
        
        if current_close < recent_highs * 0.99:
            # Price failed to hold breakout levels
            volume_spike = indicators.get('volume_ratio', 1.0) > 1.5
            if volume_spike:
                trapped_bulls = True
        
        # Check for trapped bears (breakdown failure)
        recent_lows = df['low'].iloc[-5:].min()
        if current_close > recent_lows * 1.01:
            volume_spike = indicators.get('volume_ratio', 1.0) > 1.5
            if volume_spike:
                trapped_bears = True
        
        return trapped_bulls, trapped_bears

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        resistance = indicators.get('resistance', current_price + atr)
        support = indicators.get('support', current_price - atr)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = resistance * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = support * 1.01
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