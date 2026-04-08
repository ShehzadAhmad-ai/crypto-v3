"""
Strategy Name: Volume Spike (Advanced)
Description: Detect significant volume spikes for price confirmation with divergence detection
Logic:
- High volume with price increase = bullish confirmation
- High volume with price decrease = bearish confirmation
- Volume divergence (price up, volume down) = warning
- Volume climax detection (exhaustion)
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class VolumeSpikeStrategy(BaseStrategy):
    """
    Advanced Volume Spike Strategy
    - Volume spike detection with multiple thresholds
    - Volume divergence detection
    - Volume climax identification
    - Price-volume correlation analysis
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Volume Spike"
        self.description = "Advanced volume spike detection with divergence analysis"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Volume thresholds
        self.volume_spike_threshold = 1.8
        self.volume_climax_threshold = 2.5
        self.volume_divergence_lookback = 10
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'volume_ratio', 'rsi', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trading signal based on volume spikes and divergence
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
        volume_ratio = indicators.get('volume_ratio', 1.0)
        rsi = indicators.get('rsi', 50)
        atr = indicators.get('atr', current_price * 0.01)
        
        # ===== 5. DETECT VOLUME PATTERNS =====
        volume_spike = volume_ratio > self.volume_spike_threshold
        volume_climax = volume_ratio > self.volume_climax_threshold
        
        # Price change
        price_change_pct = 0
        if len(df) >= 2:
            price_change_pct = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        
        # Volume divergence detection
        bullish_divergence, bearish_divergence = self._detect_volume_divergence(df)
        
        # ===== 6. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        # Volume spike with price increase
        if volume_spike and price_change_pct > 0.5:
            bullish_score += 4
            bullish_reasons.append(f"Volume spike with price up: {volume_ratio:.1f}x")
            
            if volume_climax:
                bullish_score += 1
                bullish_reasons.append("Volume climax - potential trend start")
        
        # Volume divergence (price down, volume down) = bullish
        elif bullish_divergence:
            bullish_score += 3
            bullish_reasons.append("Bullish volume divergence - price falling on decreasing volume")
        
        # Volume confirmation with RSI
        if volume_ratio > 1.3 and rsi > 50 and bullish_score > 0:
            bullish_score += 1
            bullish_reasons.append(f"Volume-RSI confirmation: {rsi:.1f}")
        
        # ===== 7. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        # Volume spike with price decrease
        if volume_spike and price_change_pct < -0.5:
            bearish_score += 4
            bearish_reasons.append(f"Volume spike with price down: {volume_ratio:.1f}x")
            
            if volume_climax:
                bearish_score += 1
                bearish_reasons.append("Volume climax - potential trend reversal")
        
        # Volume divergence (price up, volume down) = bearish
        elif bearish_divergence:
            bearish_score += 3
            bearish_reasons.append("Bearish volume divergence - price rising on decreasing volume")
        
        # Volume confirmation with RSI
        if volume_ratio > 1.3 and rsi < 50 and bearish_score > 0:
            bearish_score += 1
            bearish_reasons.append(f"Volume-RSI confirmation: {rsi:.1f}")
        
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
                indicators_used=['volume_ratio', 'rsi', 'atr']
            )
        
        return None

    def _detect_volume_divergence(self, df: pd.DataFrame) -> tuple:
        """Detect bullish and bearish volume divergence"""
        if len(df) < self.volume_divergence_lookback:
            return False, False
        
        price_5 = df['close'].iloc[-5:].values
        volume_5 = df['volume'].iloc[-5:].values
        
        # Bullish divergence: price down, volume down
        price_trend = price_5[-1] < price_5[0]
        volume_trend = volume_5[-1] < volume_5[0]
        bullish_div = price_trend and volume_trend and not (volume_5[-1] > volume_5[0] * 1.5)
        
        # Bearish divergence: price up, volume down
        price_trend_up = price_5[-1] > price_5[0]
        volume_trend_down = volume_5[-1] < volume_5[0]
        bearish_div = price_trend_up and volume_trend_down
        
        return bullish_div, bearish_div

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """
        Calculate entry, stop loss, and take profit levels
        """
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # Volume-based multiplier
        volume_multiplier = min(1.5, max(0.8, volume_ratio / 2))
        
        if action == 'BUY':
            entry = current_price
            stop_loss = entry - (atr * 1.5 * (1 / volume_multiplier))
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.0)
        
        else:  # SELL
            entry = current_price
            stop_loss = entry + (atr * 1.5 * (1 / volume_multiplier))
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