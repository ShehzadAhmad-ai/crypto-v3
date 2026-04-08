"""
Strategy Name: Bollinger Bands (Advanced)
Description: Mean reversion and breakout trading using Bollinger Bands with dynamic thresholds
Logic: 
- Buy when price touches lower band in uptrend (mean reversion) with RSI confirmation
- Sell when price touches upper band in downtrend (mean reversion) with RSI confirmation
- Breakout strategies from Bollinger Squeeze with volume confirmation
- Dynamic ATR-based stop loss and take profit
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class BollingerBandsStrategy(BaseStrategy):
    """
    Advanced Bollinger Bands Strategy
    - Dynamic thresholds based on market regime
    - ATR-based position sizing
    - Volume confirmation for breakouts
    - Self-calculates missing indicators
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Bollinger Bands"
        self.description = "Mean reversion and breakout trading using Bollinger Bands with dynamic thresholds"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Bollinger settings
        self.bb_period = 20
        self.bb_std = 2.0
        
        # Squeeze detection threshold (width in %)
        self.squeeze_threshold = 0.04  # 4% width
        
        # Band position thresholds (adaptive)
        self.lower_band_threshold = 0.2   # 20% from bottom
        self.upper_band_threshold = 0.8   # 80% from bottom
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'bb_position', 'bb_width', 'bb_upper', 'bb_lower', 
            'bb_middle', 'rsi', 'volume_ratio', 'atr'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trading signal based on Bollinger Bands with advanced logic
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
        regime_trend_strength = market_regime.get('trend_strength', 0)
        
        # ===== 4. ADAPT THRESHOLDS BASED ON REGIME =====
        lower_threshold = self.lower_band_threshold
        upper_threshold = self.upper_band_threshold
        
        # In trending markets, bands are less reliable for mean reversion
        if 'TREND' in regime and regime_trend_strength > 0.6:
            lower_threshold = 0.15  # More extreme needed
            upper_threshold = 0.85
        elif 'RANGING' in regime:
            lower_threshold = 0.25  # Less extreme in ranging markets
            upper_threshold = 0.75
        
        # ===== 5. EXTRACT INDICATOR VALUES =====
        bb_position = indicators.get('bb_position', 0.5)
        bb_width = indicators.get('bb_width', 0)
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        bb_middle = indicators.get('bb_middle', current_price)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        rsi = indicators.get('rsi', 50)
        atr = indicators.get('atr', current_price * 0.01)
        
        # Get previous values for momentum
        bb_position_prev = indicators.get('bb_position_prev', bb_position)
        
        # ===== 6. BULLISH CONDITIONS (Mean Reversion + Squeeze Breakout) =====
        bullish_score = 0
        bullish_reasons = []
        
        # Condition 1: Mean reversion - price at lower band
        if bb_position < lower_threshold:
            bullish_score += 3
            bullish_reasons.append(f"Price at lower Bollinger Band (position: {bb_position:.2f})")
            
            # Stronger if RSI confirms oversold
            if rsi < 30:
                bullish_score += 1
                bullish_reasons.append(f"RSI oversold confirmation: {rsi:.1f}")
            
            # Stronger if bouncing off lower band
            if bb_position > bb_position_prev:
                bullish_score += 1
                bullish_reasons.append("Bouncing off lower band")
        
        # Condition 2: Bollinger Squeeze breakout (bullish)
        elif bb_width < self.squeeze_threshold:
            # Check for bullish breakout with volume
            if len(df) >= 5:
                recent_high = df['close'].iloc[-5:].max()
                if recent_high > bb_upper:
                    bullish_score += 4
                    bullish_reasons.append("Breakout above upper Bollinger Band from squeeze")
                    
                    if volume_ratio > 1.8:
                        bullish_score += 2
                        bullish_reasons.append(f"Strong volume confirmation: {volume_ratio:.1f}x")
        
        # Condition 3: Volume confirmation for mean reversion
        if volume_ratio > 1.5 and bullish_score > 0:
            bullish_score += 1
            bullish_reasons.append(f"High volume confirmation ({volume_ratio:.1f}x)")
        
        # ===== 7. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        # Condition 1: Mean reversion - price at upper band
        if bb_position > upper_threshold:
            bearish_score += 3
            bearish_reasons.append(f"Price at upper Bollinger Band (position: {bb_position:.2f})")
            
            # Stronger if RSI confirms overbought
            if rsi > 70:
                bearish_score += 1
                bearish_reasons.append(f"RSI overbought confirmation: {rsi:.1f}")
            
            # Stronger if rejecting upper band
            if bb_position < bb_position_prev:
                bearish_score += 1
                bearish_reasons.append("Rejecting upper band")
        
        # Condition 2: Bollinger Squeeze breakout (bearish)
        elif bb_width < self.squeeze_threshold:
            if len(df) >= 5:
                recent_low = df['close'].iloc[-5:].min()
                if recent_low < bb_lower:
                    bearish_score += 4
                    bearish_reasons.append("Breakout below lower Bollinger Band from squeeze")
                    
                    if volume_ratio > 1.8:
                        bearish_score += 2
                        bearish_reasons.append(f"Strong volume confirmation: {volume_ratio:.1f}x")
        
        # Condition 3: Volume confirmation
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
            # In strong trends, mean reversion signals are weaker
            if 'TREND' in regime and 'squeeze' not in ' '.join(reasons).lower():
                confidence = confidence * 0.8
                reasons.append("Trend market - reducing mean reversion confidence")
            
            # Boost if aligned with regime
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
                indicators_used=['bb_position', 'bb_width', 'rsi', 'volume_ratio', 'atr']
            )
        
        return None

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """
        Calculate advanced entry, stop loss, and take profit levels
        Uses ATR-based volatility adjustment and Bollinger Band levels
        """
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        bb_middle = indicators.get('bb_middle', current_price)
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        bb_position = indicators.get('bb_position', 0.5)
        
        # Volatility multiplier based on ATR percentage
        atr_pct = atr / current_price
        if atr_pct > 0.05:  # High volatility
            risk_multiplier = 1.2
            reward_multiplier = 2.5
        elif atr_pct < 0.01:  # Low volatility
            risk_multiplier = 0.8
            reward_multiplier = 1.8
        else:
            risk_multiplier = 1.0
            reward_multiplier = 2.0
        
        if action == 'BUY':
            # Entry: Current price or slight premium
            entry = current_price
            
            # Dynamic Stop Loss based on band position
            if bb_position < 0.3:  # Mean reversion at lower band
                stop_loss = bb_lower * 0.99
            else:  # Squeeze breakout
                stop_loss = entry - (atr * 1.5 * risk_multiplier)
            
            # Take Profit: Dynamic based on volatility
            risk = entry - stop_loss
            take_profit = entry + (risk * reward_multiplier)
            
            # Cap at middle band for mean reversion
            if bb_position < 0.3 and bb_middle > entry:
                take_profit = min(take_profit, bb_middle)
        
        else:  # SELL
            entry = current_price
            
            if bb_position > 0.7:
                stop_loss = bb_upper * 1.01
            else:
                stop_loss = entry + (atr * 1.5 * risk_multiplier)
            
            risk = stop_loss - entry
            take_profit = entry - (risk * reward_multiplier)
            
            if bb_position > 0.7 and bb_middle < entry:
                take_profit = max(take_profit, bb_middle)
        
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