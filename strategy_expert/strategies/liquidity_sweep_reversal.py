"""
Strategy Name: Liquidity Sweep Reversal (Advanced)
Description: Trades reversals after stop hunts and liquidity grabs used by smart money
Logic:
- Detects sweep below support or above resistance with volume spikes
- Market structure shift confirmation (CHOCH/BOS)
- Volume profile analysis for absorption
- Dynamic entry with ATR-based stops
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class LiquiditySweepReversalStrategy(BaseStrategy):
    """
    Advanced Liquidity Sweep Reversal Strategy
    - Multiple liquidity sweep detection methods
    - Volume absorption analysis
    - Market structure shift confirmation
    - Dynamic entry zones
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Liquidity Sweep Reversal"
        self.description = "Advanced reversal trading after stop hunts and liquidity grabs"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Volume thresholds
        self.volume_spike_threshold = 1.6
        self.absorption_threshold = 1.3
        
        # RSI thresholds
        self.rsi_bullish_min = 40
        self.rsi_bearish_max = 60
        
        # Sweep detection lookback
        self.sweep_lookback = 10
        
        # Confidence thresholds
        self.buy_confidence = 0.87
        self.sell_confidence = 0.85
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'volume', 'volume_avg', 'previous_high', 'previous_low',
            'atr', 'volume_ratio', 'vwap'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate liquidity sweep reversal signal with full trade setup
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
        vwap = indicators.get('vwap', current_price)
        
        # Get previous levels
        prev_high = indicators.get('previous_high', 0)
        prev_low = indicators.get('previous_low', 0)
        
        # Get smart money data (from module_signals if available)
        bos = False
        choch = False
        if module_signals:
            bos = module_signals.get('bos', False)
            choch = module_signals.get('choch', False)
        
        # ===== 5. DETECT LIQUIDITY SWEEPS =====
        sweep_down, sweep_up = self._detect_liquidity_sweeps(df, indicators)
        
        # ===== 6. DETECT VOLUME ABSORPTION =====
        absorption = self._detect_volume_absorption(df, indicators)
        
        # ===== 7. BULLISH LIQUIDITY SWEEP =====
        bullish_score = 0
        bullish_reasons = []
        
        if sweep_down or (prev_low > 0 and current_price < prev_low):
            # Volume spike confirmation
            if volume_ratio > self.volume_spike_threshold:
                bullish_score += 3
                bullish_reasons.append(f"Liquidity sweep below with volume spike ({volume_ratio:.1f}x)")
            
            # Volume absorption (smart money buying)
            if absorption['bullish']:
                bullish_score += 2
                bullish_reasons.append("Volume absorption detected - smart money buying")
            
            # Market structure shift
            if choch or bos:
                bullish_score += 2
                bullish_reasons.append("Market structure shift (CHOCH/BOS) confirmed")
            
            # RSI recovery
            if rsi > self.rsi_bullish_min:
                bullish_score += 1
                bullish_reasons.append(f"RSI recovering: {rsi:.1f}")
            
            # VWAP support
            if current_price > vwap:
                bullish_score += 1
                bullish_reasons.append("Price above VWAP - bullish context")
        
        # ===== 8. BEARISH LIQUIDITY SWEEP =====
        bearish_score = 0
        bearish_reasons = []
        
        if sweep_up or (prev_high > 0 and current_price > prev_high):
            if volume_ratio > self.volume_spike_threshold:
                bearish_score += 3
                bearish_reasons.append(f"Liquidity sweep above with volume spike ({volume_ratio:.1f}x)")
            
            if absorption['bearish']:
                bearish_score += 2
                bearish_reasons.append("Volume absorption detected - smart money selling")
            
            if choch or bos:
                bearish_score += 2
                bearish_reasons.append("Market structure shift (CHOCH/BOS) confirmed")
            
            if rsi < self.rsi_bearish_max:
                bearish_score += 1
                bearish_reasons.append(f"RSI weakening: {rsi:.1f}")
            
            if current_price < vwap:
                bearish_score += 1
                bearish_reasons.append("Price below VWAP - bearish context")
        
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
                indicators_used=['rsi', 'volume_ratio', 'atr', 'vwap', 'previous_high', 'previous_low']
            )
        
        return None

    def _detect_liquidity_sweeps(self, df: pd.DataFrame, indicators: Dict) -> tuple:
        """Detect liquidity sweeps below and above"""
        sweep_down = False
        sweep_up = False
        
        if len(df) < self.sweep_lookback:
            return sweep_down, sweep_up
        
        recent_lows = df['low'].iloc[-self.sweep_lookback:-1].min()
        recent_highs = df['high'].iloc[-self.sweep_lookback:-1].max()
        
        current_low = df['low'].iloc[-1]
        current_high = df['high'].iloc[-1]
        
        if current_low < recent_lows:
            sweep_down = True
        
        if current_high > recent_highs:
            sweep_up = True
        
        return sweep_down, sweep_up
    
    def _detect_volume_absorption(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Detect volume absorption (smart money accumulation/distribution)"""
        volume = indicators.get('volume', 0)
        volume_avg = indicators.get('volume_avg', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        bullish_absorption = False
        bearish_absorption = False
        
        if len(df) >= 2:
            price_up = df['close'].iloc[-1] > df['close'].iloc[-2]
            price_down = df['close'].iloc[-1] < df['close'].iloc[-2]
            
            # Bullish absorption: price down on high volume, then reversal
            if price_down and volume_ratio > self.absorption_threshold:
                if len(df) >= 3 and df['close'].iloc[-2] > df['close'].iloc[-3]:
                    bullish_absorption = True
            
            # Bearish absorption: price up on high volume, then reversal
            if price_up and volume_ratio > self.absorption_threshold:
                if len(df) >= 3 and df['close'].iloc[-2] < df['close'].iloc[-3]:
                    bearish_absorption = True
        
        return {'bullish': bullish_absorption, 'bearish': bearish_absorption}

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        vwap = indicators.get('vwap', current_price)
        prev_low = indicators.get('previous_low', current_price - atr)
        prev_high = indicators.get('previous_high', current_price + atr)
        
        if action == 'BUY':
            entry = current_price
            
            # Stop loss below sweep level or ATR-based
            stop_loss = min(prev_low * 0.99, entry - (atr * 1.5))
            
            # Take profit at VWAP or 2x risk
            risk = entry - stop_loss
            if vwap > entry:
                take_profit = min(vwap, entry + (risk * 2))
            else:
                take_profit = entry + (risk * 2)
        
        else:  # SELL
            entry = current_price
            stop_loss = max(prev_high * 1.01, entry + (atr * 1.5))
            risk = stop_loss - entry
            
            if vwap < entry:
                take_profit = max(vwap, entry - (risk * 2))
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