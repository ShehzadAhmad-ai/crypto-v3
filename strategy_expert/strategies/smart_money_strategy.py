"""
Strategy Name: Smart Money (Advanced)
Description: Institutional trading concepts: order blocks, FVGs, market structure
Logic:
- Order block identification and retest
- Fair Value Gap (FVG) detection
- Market structure analysis
- Break of Structure (BOS) and Change of Character (CHOCH)
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class SmartMoneyStrategy(BaseStrategy):
    """
    Advanced Smart Money Strategy
    - Order block detection with strength scoring
    - Fair Value Gap (FVG) identification
    - Market structure analysis (BOS/CHOCH)
    - Institutional order flow tracking
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Smart Money"
        self.description = "Advanced institutional trading concepts: order blocks, FVGs, market structure"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Order block thresholds
        self.ob_proximity_threshold = 0.01  # 1%
        self.fvg_proximity_threshold = 0.005  # 0.5%
        
        # Confidence thresholds
        self.buy_confidence = 0.85
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'ema_200', 'atr', 'volume_ratio'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate smart money signal
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
        atr = indicators.get('atr', current_price * 0.01)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # ===== 5. GET SMART MONEY DATA FROM MODULE SIGNALS =====
        order_blocks = []
        fvgs = []
        bos = False
        choch = False
        market_structure = {}
        
        if module_signals:
            order_blocks = module_signals.get('order_blocks', [])
            fvgs = module_signals.get('fair_value_gaps', [])
            bos = module_signals.get('bos', False)
            choch = module_signals.get('choch', False)
            market_structure = module_signals.get('market_structure', {})
        
        # ===== 6. DETECT SMART MONEY PATTERNS DIRECTLY FROM PRICE =====
        if len(df) >= 10:
            # Detect Order Blocks from price action
            detected_obs = self._detect_order_blocks(df)
            order_blocks.extend(detected_obs)
            
            # Detect Fair Value Gaps
            detected_fvgs = self._detect_fvgs(df)
            fvgs.extend(detected_fvgs)
            
            # Detect Market Structure
            structure = self._detect_market_structure(df)
            bos = bos or structure.get('bos', False)
            choch = choch or structure.get('choch', False)
        
        # ===== 7. CHECK ORDER BLOCKS FOR ENTRIES =====
        bullish_ob = None
        bearish_ob = None
        
        for block in order_blocks[-5:]:  # Last 5 order blocks
            if isinstance(block, dict):
                block_price = block.get('price', 0)
                block_type = block.get('type', '')
                distance_pct = abs(block_price - current_price) / current_price
                
                if distance_pct < self.ob_proximity_threshold:
                    if 'bullish' in block_type.lower():
                        bullish_ob = block
                    elif 'bearish' in block_type.lower():
                        bearish_ob = block
        
        # ===== 8. CHECK FVGS FOR ENTRIES =====
        bullish_fvg = None
        bearish_fvg = None
        
        for fvg in fvgs[-5:]:
            if isinstance(fvg, dict):
                fvg_direction = fvg.get('direction', '')
                fvg_low = fvg.get('low', 0)
                fvg_high = fvg.get('high', 0)
                
                if fvg_direction == 'bullish' and fvg_low < current_price < fvg_high:
                    bullish_fvg = fvg
                elif fvg_direction == 'bearish' and fvg_low < current_price < fvg_high:
                    bearish_fvg = fvg
        
        # ===== 9. BULLISH SETUP =====
        bullish_score = 0
        bullish_reasons = []
        
        # Check for bullish order block
        if bullish_ob:
            bullish_score += 3
            bullish_reasons.append(f"Bullish Order Block at {bullish_ob.get('price', 0):.2f}")
            if bullish_ob.get('strength', 0) > 0.7:
                bullish_score += 1
                bullish_reasons.append("High strength order block")
        
        # Check for bullish FVG
        if bullish_fvg:
            bullish_score += 2
            bullish_reasons.append(f"Price in Bullish FVG: {bullish_fvg.get('low', 0):.2f}-{bullish_fvg.get('high', 0):.2f}")
        
        # Check for Break of Structure
        if bos:
            if market_structure.get('trend') == 'bullish':
                bullish_score += 2
                bullish_reasons.append("Bullish Break of Structure (BOS)")
        
        # Check for Change of Character
        if choch:
            bullish_score += 1
            bullish_reasons.append("Change of Character (CHOCH) - trend reversal")
        
        # Volume confirmation
        if volume_ratio > 1.3 and bullish_score > 0:
            bullish_score += 1
            bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 10. BEARISH SETUP =====
        bearish_score = 0
        bearish_reasons = []
        
        if bearish_ob:
            bearish_score += 3
            bearish_reasons.append(f"Bearish Order Block at {bearish_ob.get('price', 0):.2f}")
            if bearish_ob.get('strength', 0) > 0.7:
                bearish_score += 1
                bearish_reasons.append("High strength order block")
        
        if bearish_fvg:
            bearish_score += 2
            bearish_reasons.append(f"Price in Bearish FVG: {bearish_fvg.get('low', 0):.2f}-{bearish_fvg.get('high', 0):.2f}")
        
        if bos:
            if market_structure.get('trend') == 'bearish':
                bearish_score += 2
                bearish_reasons.append("Bearish Break of Structure (BOS)")
        
        if choch:
            bearish_score += 1
            bearish_reasons.append("Change of Character (CHOCH) - trend reversal")
        
        if volume_ratio > 1.3 and bearish_score > 0:
            bearish_score += 1
            bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 11. DETERMINE ACTION =====
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
        
        # ===== 12. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.1)
                reasons.append(f"Aligned with {regime} regime")
        
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
                indicators_used=['smart_money', 'order_blocks', 'fvgs', 'market_structure']
            )
        
        return None

    def _detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect order blocks from price action"""
        order_blocks = []
        
        if len(df) < 5:
            return order_blocks
        
        for i in range(5, len(df)):
            # Look for strong momentum candles
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            # Bullish Order Block: Strong up candle after consolidation
            if (candle['close'] > candle['open'] and 
                (candle['close'] - candle['open']) / candle['open'] > 0.01):
                if prev_candle['close'] < prev_candle['open']:
                    order_blocks.append({
                        'type': 'bullish',
                        'price': candle['low'],
                        'strength': (candle['close'] - candle['open']) / candle['open'],
                        'timestamp': candle.name
                    })
            
            # Bearish Order Block: Strong down candle after consolidation
            if (candle['close'] < candle['open'] and 
                (candle['open'] - candle['close']) / candle['open'] > 0.01):
                if prev_candle['close'] > prev_candle['open']:
                    order_blocks.append({
                        'type': 'bearish',
                        'price': candle['high'],
                        'strength': (candle['open'] - candle['close']) / candle['open'],
                        'timestamp': candle.name
                    })
        
        return order_blocks[-5:]  # Return last 5 order blocks

    def _detect_fvgs(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Fair Value Gaps"""
        fvgs = []
        
        if len(df) < 3:
            return fvgs
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: Gap between candle1 high and candle3 low
            if candle1['high'] < candle3['low']:
                fvgs.append({
                    'direction': 'bullish',
                    'high': candle3['low'],
                    'low': candle1['high'],
                    'filled': candle2['low'] <= candle1['high']
                })
            
            # Bearish FVG: Gap between candle1 low and candle3 high
            if candle1['low'] > candle3['high']:
                fvgs.append({
                    'direction': 'bearish',
                    'high': candle1['low'],
                    'low': candle3['high'],
                    'filled': candle2['high'] >= candle1['low']
                })
        
        return fvgs[-5:]  # Return last 5 FVGs

    def _detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """Detect market structure (BOS/CHOCH)"""
        structure = {
            'bos': False,
            'choch': False,
            'trend': 'neutral'
        }
        
        if len(df) < 10:
            return structure
        
        # Find swing highs and lows
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))
        
        # Determine trend
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
                structure['trend'] = 'bullish'
            elif swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
                structure['trend'] = 'bearish'
        
        # Detect BOS
        if len(swing_highs) >= 2:
            if structure['trend'] == 'bullish' and swing_highs[-1][1] > swing_highs[-2][1]:
                structure['bos'] = True
            elif structure['trend'] == 'bearish' and swing_lows[-1][1] < swing_lows[-2][1]:
                structure['bos'] = True
        
        # Detect CHOCH
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            if (swing_highs[-1][1] < swing_highs[-2][1] and 
                swing_lows[-1][1] > swing_lows[-2][1] and
                structure['trend'] == 'bullish'):
                structure['choch'] = True
            elif (swing_highs[-1][1] > swing_highs[-2][1] and 
                  swing_lows[-1][1] < swing_lows[-2][1] and
                  structure['trend'] == 'bearish'):
                structure['choch'] = True
        
        return structure

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        
        if len(df) >= 10:
            recent_low = df['low'].iloc[-10:].min()
            recent_high = df['high'].iloc[-10:].max()
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