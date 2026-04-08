"""
Strategy Name: Pattern Recognition (Advanced)
Description: Detect and trade based on chart patterns with advanced confirmation
Logic:
- Candlestick pattern detection (hammer, engulfing, etc.)
- Chart pattern detection (double top/bottom, head & shoulders)
- Pattern confidence scoring
- Volume confirmation
- Trend alignment filter
"""

from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class PatternRecognitionStrategy(BaseStrategy):
    """
    Advanced Pattern Recognition Strategy
    - Multi-pattern detection
    - Confidence scoring system
    - Volume confirmation
    - Trend alignment validation
    - Pattern failure detection
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Pattern Recognition"
        self.description = "Advanced chart pattern detection and trading with confirmation"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Pattern confidence thresholds
        self.min_pattern_confidence = 0.6
        self.high_confidence_threshold = 0.8
        
        # Volume confirmation thresholds
        self.volume_confirmation_threshold = 1.3
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Pattern lookback periods
        self.pattern_lookback = 50
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'rsi', 'volume_ratio', 'atr', 'close', 'open', 'high', 'low'
        ]
        
        # ===== PATTERN LISTS =====
        self.bullish_patterns = [
            'hammer', 'bullish_engulfing', 'morning_star', 'piercing_pattern',
            'three_white_soldiers', 'double_bottom', 'inverse_head_and_shoulders',
            'ascending_triangle', 'falling_wedge', 'cup_and_handle'
        ]
        
        self.bearish_patterns = [
            'shooting_star', 'bearish_engulfing', 'evening_star', 'dark_cloud_cover',
            'three_black_crows', 'double_top', 'head_and_shoulders',
            'descending_triangle', 'rising_wedge'
        ]
        
        # High confidence patterns
        self.high_confidence_bullish = [
            'inverse_head_and_shoulders', 'double_bottom', 'bullish_engulfing',
            'cup_and_handle', 'ascending_triangle'
        ]
        
        self.high_confidence_bearish = [
            'head_and_shoulders', 'double_top', 'bearish_engulfing',
            'descending_triangle', 'rising_wedge'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate signal based on pattern recognition with advanced confirmation
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
        
        # ===== 5. DETECT PATTERNS =====
        # Get patterns from module_signals or calculate directly
        pattern_dict = {}
        if module_signals and 'patterns' in module_signals:
            for name, conf in module_signals['patterns'].items():
                key = name.lower().replace(' ', '_')
                pattern_dict[key] = conf
        
        # Also detect basic candlestick patterns from price data
        if len(df) >= 2:
            self._detect_candlestick_patterns(df, pattern_dict)
        
        # ===== 6. DETECT CHART PATTERNS =====
        chart_patterns = self._detect_chart_patterns(df)
        pattern_dict.update(chart_patterns)
        
        # Sort patterns by confidence
        sorted_patterns = sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
        
        # ===== 7. BULLISH PATTERN SCORING =====
        bullish_score = 0
        bullish_reasons = []
        bullish_patterns_found = []
        
        for pattern_key, pattern_conf in sorted_patterns[:5]:
            if pattern_key in self.bullish_patterns:
                if pattern_conf < self.min_pattern_confidence:
                    continue
                
                # Base score from pattern confidence
                pattern_score = int(pattern_conf * 4)
                bullish_score += pattern_score
                
                # Additional weight for high confidence patterns
                if pattern_key in self.high_confidence_bullish:
                    bullish_score += 2
                    pattern_desc = pattern_key.replace('_', ' ').title()
                    bullish_reasons.append(f"High confidence {pattern_desc} pattern")
                else:
                    pattern_desc = pattern_key.replace('_', ' ').title()
                    bullish_reasons.append(f"{pattern_desc} pattern detected")
                
                bullish_patterns_found.append(pattern_key)
                
                if len(bullish_patterns_found) >= 3:
                    break
        
        # ===== 8. BEARISH PATTERN SCORING =====
        bearish_score = 0
        bearish_reasons = []
        bearish_patterns_found = []
        
        for pattern_key, pattern_conf in sorted_patterns[:5]:
            if pattern_key in self.bearish_patterns:
                if pattern_conf < self.min_pattern_confidence:
                    continue
                
                pattern_score = int(pattern_conf * 4)
                bearish_score += pattern_score
                
                if pattern_key in self.high_confidence_bearish:
                    bearish_score += 2
                    pattern_desc = pattern_key.replace('_', ' ').title()
                    bearish_reasons.append(f"High confidence {pattern_desc} pattern")
                else:
                    pattern_desc = pattern_key.replace('_', ' ').title()
                    bearish_reasons.append(f"{pattern_desc} pattern detected")
                
                bearish_patterns_found.append(pattern_key)
                
                if len(bearish_patterns_found) >= 3:
                    break
        
        # ===== 9. MULTIPLE PATTERN CONFIRMATION =====
        if len(bullish_patterns_found) > 1:
            bullish_score += 1
            bullish_reasons.append(f"{len(bullish_patterns_found)} confirming bullish patterns")
        
        if len(bearish_patterns_found) > 1:
            bearish_score += 1
            bearish_reasons.append(f"{len(bearish_patterns_found)} confirming bearish patterns")
        
        # ===== 10. VOLUME CONFIRMATION =====
        if volume_ratio > self.volume_confirmation_threshold:
            if bullish_score > bearish_score and bullish_score > 0:
                bullish_score += 1
                bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
            elif bearish_score > bullish_score and bearish_score > 0:
                bearish_score += 1
                bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 11. TREND ALIGNMENT =====
        trend_aligned_bull = current_price > ema_200 and regime_bias > 0
        trend_aligned_bear = current_price < ema_200 and regime_bias < 0
        
        if trend_aligned_bull and bullish_score > 0:
            bullish_score += 2
            bullish_reasons.append("Pattern aligned with bullish trend")
        elif trend_aligned_bear and bearish_score > 0:
            bearish_score += 2
            bearish_reasons.append("Pattern aligned with bearish trend")
        
        # ===== 12. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 4
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = min(0.95, 0.5 + (bullish_score / 14))
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = min(0.95, 0.5 + (bearish_score / 14))
            reasons = bearish_reasons[:5]
        
        # ===== 13. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            # Counter-trend patterns have lower confidence
            if action == 'BUY' and regime_bias < -0.3:
                confidence = confidence * 0.7
                reasons.append("Counter-trend pattern - reduced confidence")
            elif action == 'SELL' and regime_bias > 0.3:
                confidence = confidence * 0.7
                reasons.append("Counter-trend pattern - reduced confidence")
            
            # Boost if aligned with regime
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 14. CALCULATE ENTRY, SL, TP =====
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
                indicators_used=['patterns', 'volume_ratio', 'rsi', 'ema_200']
            )
        
        return None

    def _detect_candlestick_patterns(self, df: pd.DataFrame, pattern_dict: Dict):
        """Detect basic candlestick patterns"""
        if len(df) < 2:
            return
        
        c1 = df.iloc[-1]
        c2 = df.iloc[-2]
        
        # Calculate candle metrics
        body1 = abs(c1['close'] - c1['open'])
        body2 = abs(c2['close'] - c2['open'])
        upper_wick1 = c1['high'] - max(c1['open'], c1['close'])
        lower_wick1 = min(c1['open'], c1['close']) - c1['low']
        upper_wick2 = c2['high'] - max(c2['open'], c2['close'])
        lower_wick2 = min(c2['open'], c2['close']) - c2['low']
        
        # Hammer
        if lower_wick1 > body1 * 2 and upper_wick1 < body1 * 0.5:
            pattern_dict['hammer'] = 0.75
        
        # Shooting Star
        if upper_wick1 > body1 * 2 and lower_wick1 < body1 * 0.5:
            pattern_dict['shooting_star'] = 0.75
        
        # Bullish Engulfing
        if (c2['close'] < c2['open'] and  # Previous bearish
            c1['close'] > c1['open'] and  # Current bullish
            c1['close'] > c2['open'] and  # Closes above previous open
            c1['open'] < c2['close']):    # Opens below previous close
            pattern_dict['bullish_engulfing'] = 0.85
        
        # Bearish Engulfing
        if (c2['close'] > c2['open'] and  # Previous bullish
            c1['close'] < c1['open'] and  # Current bearish
            c1['close'] < c2['open'] and  # Closes below previous open
            c1['open'] > c2['close']):    # Opens above previous close
            pattern_dict['bearish_engulfing'] = 0.85
        
        # Morning Star (simplified)
        if len(df) >= 3:
            c3 = df.iloc[-3]
            if (c3['close'] < c3['open'] and  # First bearish
                body2 < body1 * 0.3 and  # Second small body
                c1['close'] > c1['open'] and  # Third bullish
                c1['close'] > (c3['open'] + c3['close']) / 2):  # Closes above midpoint
                pattern_dict['morning_star'] = 0.80
        
        # Evening Star (simplified)
        if len(df) >= 3:
            c3 = df.iloc[-3]
            if (c3['close'] > c3['open'] and  # First bullish
                body2 < body1 * 0.3 and  # Second small body
                c1['close'] < c1['open'] and  # Third bearish
                c1['close'] < (c3['open'] + c3['close']) / 2):  # Closes below midpoint
                pattern_dict['evening_star'] = 0.80

    def _detect_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect chart patterns (double top/bottom, head & shoulders)"""
        patterns = {}
        
        if len(df) < self.pattern_lookback:
            return patterns
        
        # Get recent highs and lows
        highs = df['high'].iloc[-self.pattern_lookback:].values
        lows = df['low'].iloc[-self.pattern_lookback:].values
        closes = df['close'].iloc[-self.pattern_lookback:].values
        
        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)
        
        # Double Top
        if len(peaks) >= 2:
            peak1 = peaks[-2]
            peak2 = peaks[-1]
            peak_diff = abs(peak1 - peak2) / peak1 * 100
            
            if peak_diff < 3:  # Within 3%
                # Check for neckline break
                neckline = min(highs[peaks[-2]:peaks[-1]]) if peaks[-2] < peaks[-1] else min(highs[peaks[-1]:peaks[-2]])
                if closes[-1] < neckline:
                    patterns['double_top'] = 0.75
        
        # Double Bottom
        if len(troughs) >= 2:
            trough1 = troughs[-2]
            trough2 = troughs[-1]
            trough_diff = abs(trough1 - trough2) / trough1 * 100
            
            if trough_diff < 3:
                neckline = max(lows[troughs[-2]:troughs[-1]]) if troughs[-2] < troughs[-1] else max(lows[troughs[-1]:troughs[-2]])
                if closes[-1] > neckline:
                    patterns['double_bottom'] = 0.75
        
        # Head and Shoulders
        if len(peaks) >= 3:
            left_shoulder = peaks[-3]
            head = peaks[-2]
            right_shoulder = peaks[-1]
            
            if head > left_shoulder and head > right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder * 100
                if shoulder_diff < 5:
                    # Check for neckline break
                    neckline = min(highs[peaks[-3]:peaks[-2]])
                    if closes[-1] < neckline:
                        patterns['head_and_shoulders'] = 0.85
        
        # Inverse Head and Shoulders
        if len(troughs) >= 3:
            left_shoulder = troughs[-3]
            head = troughs[-2]
            right_shoulder = troughs[-1]
            
            if head < left_shoulder and head < right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder * 100
                if shoulder_diff < 5:
                    neckline = max(lows[troughs[-3]:troughs[-2]])
                    if closes[-1] > neckline:
                        patterns['inverse_head_and_shoulders'] = 0.85
        
        return patterns

    def _find_peaks(self, array: np.ndarray, window: int = 5) -> List[int]:
        """Find local peaks in array"""
        peaks = []
        for i in range(window, len(array) - window):
            if all(array[i] >= array[i - j] for j in range(1, window + 1)) and \
               all(array[i] >= array[i + j] for j in range(1, window + 1)):
                peaks.append(i)
        return peaks

    def _find_troughs(self, array: np.ndarray, window: int = 5) -> List[int]:
        """Find local troughs in array"""
        troughs = []
        for i in range(window, len(array) - window):
            if all(array[i] <= array[i - j] for j in range(1, window + 1)) and \
               all(array[i] <= array[i + j] for j in range(1, window + 1)):
                troughs.append(i)
        return troughs

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if len(df) >= self.pattern_lookback:
            recent_low = df['low'].iloc[-self.pattern_lookback:].min()
            recent_high = df['high'].iloc[-self.pattern_lookback:].max()
        else:
            recent_low = current_price * 0.98
            recent_high = current_price * 1.02
        
        if action == 'BUY':
            entry = current_price
            stop_loss = recent_low * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
        
        else:  # SELL
            entry = current_price
            stop_loss = recent_high * 1.01
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