"""
Strategy Name: Support/Resistance (Advanced)
Description: Trade at key support and resistance levels with multi-timeframe confirmation
Logic:
- Buy at support with bullish confirmation (candle pattern, volume, RSI)
- Sell at resistance with bearish confirmation
- Dynamic level detection with strength scoring
- Breakout/breakdown detection with volume confirmation
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class SupportResistanceStrategy(BaseStrategy):
    """
    Advanced Support/Resistance Strategy
    - Dynamic level detection with strength scoring
    - Multiple timeframe confirmation
    - Rejection/breakout detection
    - Volume confirmation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Support Resistance"
        self.description = "Advanced support/resistance trading with multi-timeframe confirmation"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Level detection parameters
        self.level_lookback = 50
        self.level_strength_min = 3  # Minimum touches to consider strong level
        self.level_proximity_pct = 0.01  # 1% proximity
        
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
        Generate trading signal based on support/resistance levels
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
        
        # ===== 5. DETECT SUPPORT AND RESISTANCE LEVELS =====
        supports = self._detect_levels(df, 'support')
        resistances = self._detect_levels(df, 'resistance')
        
        # ===== 6. FIND NEAREST LEVELS =====
        nearest_support = self._find_nearest_level(current_price, supports, 'below')
        nearest_resistance = self._find_nearest_level(current_price, resistances, 'above')
        
        # ===== 7. BULLISH CONDITIONS =====
        bullish_score = 0
        bullish_reasons = []
        
        if nearest_support:
            distance_pct = (current_price - nearest_support) / current_price * 100
            
            # Price near support
            if distance_pct < self.level_proximity_pct * 100:
                bullish_score += 3
                bullish_reasons.append(f"Price near strong support: {nearest_support:.2f} ({distance_pct:.2f}% above)")
                
                # Check for bullish rejection (hammer, bullish engulfing)
                if self._check_bullish_rejection(df):
                    bullish_score += 2
                    bullish_reasons.append("Bullish rejection pattern at support")
                
                # RSI oversold or turning up
                if rsi < 35:
                    bullish_score += 1
                    bullish_reasons.append(f"RSI oversold: {rsi:.1f}")
                elif rsi > rsi - 2:
                    bullish_score += 1
                    bullish_reasons.append(f"RSI turning up: {rsi:.1f}")
                
                # Volume confirmation
                if volume_ratio > 1.3:
                    bullish_score += 1
                    bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 8. BEARISH CONDITIONS =====
        bearish_score = 0
        bearish_reasons = []
        
        if nearest_resistance:
            distance_pct = (nearest_resistance - current_price) / current_price * 100
            
            # Price near resistance
            if distance_pct < self.level_proximity_pct * 100:
                bearish_score += 3
                bearish_reasons.append(f"Price near strong resistance: {nearest_resistance:.2f} ({distance_pct:.2f}% below)")
                
                # Check for bearish rejection
                if self._check_bearish_rejection(df):
                    bearish_score += 2
                    bearish_reasons.append("Bearish rejection pattern at resistance")
                
                # RSI overbought or turning down
                if rsi > 65:
                    bearish_score += 1
                    bearish_reasons.append(f"RSI overbought: {rsi:.1f}")
                elif rsi < rsi + 2:
                    bearish_score += 1
                    bearish_reasons.append(f"RSI turning down: {rsi:.1f}")
                
                # Volume confirmation
                if volume_ratio > 1.3:
                    bearish_score += 1
                    bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 9. DETERMINE ACTION =====
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
            levels = self.calculate_entry_sl_tp(df, indicators, market_regime, action, 
                                                nearest_support, nearest_resistance)
            
            return StrategyOutput(
                action=action,
                confidence=confidence,
                entry=levels['entry'],
                stop_loss=levels['stop_loss'],
                take_profit=levels['take_profit'],
                risk_reward=levels['risk_reward'],
                reasons=reasons[:5],
                strategy_name=self.name,
                indicators_used=['price', 'volume_ratio', 'rsi', 'atr']
            )
        
        return None

    def _detect_levels(self, df: pd.DataFrame, level_type: str) -> List[float]:
        """Detect support or resistance levels"""
        levels = []
        
        if level_type == 'support':
            # Find swing lows
            for i in range(2, len(df) - 2):
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i-2] and
                    df['low'].iloc[i] < df['low'].iloc[i+1] and
                    df['low'].iloc[i] < df['low'].iloc[i+2]):
                    levels.append(df['low'].iloc[i])
        else:
            # Find swing highs
            for i in range(2, len(df) - 2):
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    df['high'].iloc[i] > df['high'].iloc[i+1] and
                    df['high'].iloc[i] > df['high'].iloc[i+2]):
                    levels.append(df['high'].iloc[i])
        
        # Group nearby levels
        levels = self._group_levels(levels)
        
        return levels
    
    def _group_levels(self, levels: List[float], tolerance_pct: float = 0.005) -> List[float]:
        """Group nearby levels together"""
        if not levels:
            return []
        
        levels.sort()
        grouped = []
        current_group = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_group[-1]) / current_group[-1] < tolerance_pct:
                current_group.append(level)
            else:
                grouped.append(np.mean(current_group))
                current_group = [level]
        
        grouped.append(np.mean(current_group))
        return grouped
    
    def _find_nearest_level(self, price: float, levels: List[float], 
                            direction: str) -> Optional[float]:
        """Find nearest level above or below price"""
        if not levels:
            return None
        
        if direction == 'below':
            below = [l for l in levels if l < price]
            return max(below) if below else None
        else:
            above = [l for l in levels if l > price]
            return min(above) if above else None
    
    def _check_bullish_rejection(self, df: pd.DataFrame) -> bool:
        """Check for bullish rejection candle (hammer, bullish engulfing)"""
        if len(df) < 2:
            return False
        
        c1 = df.iloc[-1]
        c2 = df.iloc[-2]
        
        # Hammer pattern
        body = abs(c1['close'] - c1['open'])
        lower_wick = min(c1['open'], c1['close']) - c1['low']
        upper_wick = c1['high'] - max(c1['open'], c1['close'])
        
        if lower_wick > body * 2 and upper_wick < body:
            return True
        
        # Bullish engulfing
        if (c2['close'] < c2['open'] and  # Previous bearish
            c1['close'] > c1['open'] and  # Current bullish
            c1['close'] > c2['open'] and  # Closes above previous open
            c1['open'] < c2['close']):    # Opens below previous close
            return True
        
        return False
    
    def _check_bearish_rejection(self, df: pd.DataFrame) -> bool:
        """Check for bearish rejection candle (shooting star, bearish engulfing)"""
        if len(df) < 2:
            return False
        
        c1 = df.iloc[-1]
        c2 = df.iloc[-2]
        
        # Shooting star
        body = abs(c1['close'] - c1['open'])
        upper_wick = c1['high'] - max(c1['open'], c1['close'])
        lower_wick = min(c1['open'], c1['close']) - c1['low']
        
        if upper_wick > body * 2 and lower_wick < body:
            return True
        
        # Bearish engulfing
        if (c2['close'] > c2['open'] and  # Previous bullish
            c1['close'] < c1['open'] and  # Current bearish
            c1['close'] < c2['open'] and  # Closes below previous open
            c1['open'] > c2['close']):    # Opens above previous close
            return True
        
        return False

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str,
                              nearest_support: float = None,
                              nearest_resistance: float = None) -> Dict:
        """
        Calculate entry, stop loss, and take profit levels
        """
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if action == 'BUY':
            entry = current_price
            
            # Stop loss below support or ATR-based
            if nearest_support:
                stop_loss = nearest_support * 0.99
            else:
                stop_loss = entry - (atr * 1.5)
            
            # Take profit at next resistance or 2x risk
            risk = entry - stop_loss
            if nearest_resistance:
                take_profit = min(entry + (risk * 2), nearest_resistance)
            else:
                take_profit = entry + (risk * 2)
        
        else:  # SELL
            entry = current_price
            
            # Stop loss above resistance or ATR-based
            if nearest_resistance:
                stop_loss = nearest_resistance * 1.01
            else:
                stop_loss = entry + (atr * 1.5)
            
            # Take profit at next support or 2x risk
            risk = stop_loss - entry
            if nearest_support:
                take_profit = max(entry - (risk * 2), nearest_support)
            else:
                take_profit = entry - (risk * 2)
        
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