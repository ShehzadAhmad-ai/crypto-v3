"""
Strategy Name: Dynamic Pump Prediction (Advanced)
Description: Combines accumulation, liquidity sweeps, volatility, momentum with dynamic weighting
Logic:
- Multi-layer dynamic scoring
- Adaptive weight allocation
- Only highest probability setups
- Real-time parameter optimization
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class DynamicPumpPredictionStrategy(BaseStrategy):
    """
    Advanced Dynamic Pump Prediction Strategy
    - Multi-layer dynamic scoring
    - Adaptive weight allocation
    - Real-time parameter optimization
    - High-probability setup detection
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Dynamic Pump Prediction"
        self.description = "High-probability pre-pump detection using multi-layer dynamic scoring"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Layer weights (dynamic)
        self.weights = {
            'accumulation': 0.25,
            'liquidity_sweep': 0.20,
            'volatility': 0.15,
            'momentum': 0.15,
            'volume': 0.25
        }
        
        # Thresholds
        self.atr_expansion_threshold = 1.3
        self.volume_spike_threshold = 1.5
        self.rsi_bullish_min = 55
        self.rsi_bearish_max = 45
        self.adx_trend_threshold = 25
        
        # Confidence thresholds
        self.buy_confidence = 0.90
        self.sell_confidence = 0.88
        
        # Required indicators
        self.required_indicators = [
            'price', 'rsi', 'adx', 'macd_histogram', 'atr', 'atr_avg', 'volume', 'volume_avg',
            'cum_buy_delta', 'cum_sell_delta', 'hidden_support', 'hidden_resistance',
            'bb_width', 'volume_ratio'
        ]
        
        # Calculator for missing indicators
        self.calculator = IndicatorCalculator()

    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate dynamic pump prediction signal
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
        
        # ===== 4. ADAPT WEIGHTS BASED ON REGIME =====
        self._adapt_weights(regime)
        
        # ===== 5. EXTRACT INDICATOR VALUES =====
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 20)
        macd_hist = indicators.get('macd_histogram', 0)
        atr = indicators.get('atr', 0)
        atr_avg = indicators.get('atr_avg', 0)
        bb_width = indicators.get('bb_width', 0)
        volume = indicators.get('volume', 0)
        volume_avg = indicators.get('volume_avg', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        cum_buy_delta = indicators.get('cum_buy_delta', 0)
        cum_sell_delta = indicators.get('cum_sell_delta', 0)
        hidden_support = indicators.get('hidden_support', 0)
        hidden_resistance = indicators.get('hidden_resistance', 0)
        
        # Get smart money data
        liquidity_sweep_low = False
        liquidity_sweep_high = False
        if module_signals:
            liquidity_sweep_low = module_signals.get('liquidity_sweep_low', False)
            liquidity_sweep_high = module_signals.get('liquidity_sweep_high', False)
        
        # ===== 6. LAYER SCORING =====
        layer_scores = {}
        
        # Layer 1: Accumulation / Distribution
        layer1_bullish = cum_buy_delta > cum_sell_delta and current_price > hidden_support
        layer1_bearish = cum_sell_delta > cum_buy_delta and current_price < hidden_resistance
        
        # Layer 2: Liquidity Sweep
        layer2_bullish = liquidity_sweep_low
        layer2_bearish = liquidity_sweep_high
        
        # Layer 3: Volatility Expansion
        atr_expansion = atr > atr_avg * self.atr_expansion_threshold if atr_avg > 0 else False
        bb_expansion = bb_width > 0.04
        volatility_expansion = atr_expansion or bb_expansion
        
        # Layer 4: Momentum
        bullish_momentum = rsi > self.rsi_bullish_min and adx > self.adx_trend_threshold and macd_hist > 0
        bearish_momentum = rsi < self.rsi_bearish_max and adx > self.adx_trend_threshold and macd_hist < 0
        
        # Layer 5: Volume Confirmation
        volume_confirmation = volume > volume_avg * self.volume_spike_threshold if volume_avg > 0 else False
        
        # ===== 7. CALCULATE SCORES =====
        bullish_score = 0
        bullish_reasons = []
        
        if layer1_bullish:
            bullish_score += self.weights['accumulation']
            bullish_reasons.append("Hidden accumulation detected")
        
        if layer2_bullish:
            bullish_score += self.weights['liquidity_sweep']
            bullish_reasons.append("Liquidity sweep below support")
        
        if volatility_expansion:
            bullish_score += self.weights['volatility']
            bullish_reasons.append("Volatility expansion detected")
        
        if bullish_momentum:
            bullish_score += self.weights['momentum']
            bullish_reasons.append(f"Bullish momentum: RSI={rsi:.1f}, ADX={adx:.1f}")
        
        if volume_confirmation:
            bullish_score += self.weights['volume']
            bullish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 8. BEARISH SCORING =====
        bearish_score = 0
        bearish_reasons = []
        
        if layer1_bearish:
            bearish_score += self.weights['accumulation']
            bearish_reasons.append("Hidden distribution detected")
        
        if layer2_bearish:
            bearish_score += self.weights['liquidity_sweep']
            bearish_reasons.append("Liquidity sweep above resistance")
        
        if volatility_expansion:
            bearish_score += self.weights['volatility']
            bearish_reasons.append("Volatility expansion detected")
        
        if bearish_momentum:
            bearish_score += self.weights['momentum']
            bearish_reasons.append(f"Bearish momentum: RSI={rsi:.1f}, ADX={adx:.1f}")
        
        if volume_confirmation:
            bearish_score += self.weights['volume']
            bearish_reasons.append(f"Volume confirmation: {volume_ratio:.1f}x")
        
        # ===== 9. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 0.85
        
        if bullish_score >= min_score and bullish_score > bearish_score:
            action = 'BUY'
            confidence = self.buy_confidence * min(1.2, bullish_score)
            reasons = bullish_reasons[:5]
        
        elif bearish_score >= min_score and bearish_score > bullish_score:
            action = 'SELL'
            confidence = self.sell_confidence * min(1.2, bearish_score)
            reasons = bearish_reasons[:5]
        
        # ===== 10. REGIME ADJUSTMENT =====
        if action != 'HOLD':
            if action == 'BUY' and regime_bias > 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
            elif action == 'SELL' and regime_bias < 0:
                confidence = min(0.95, confidence * 1.05)
                reasons.append(f"Aligned with {regime} regime")
        
        # ===== 11. CALCULATE ENTRY, SL, TP =====
        if action != 'HOLD':
            levels = self.calculate_entry_sl_tp(df, indicators, market_regime, action)
            
            return StrategyOutput(
                action=action,
                confidence=min(0.95, confidence),
                entry=levels['entry'],
                stop_loss=levels['stop_loss'],
                take_profit=levels['take_profit'],
                risk_reward=levels['risk_reward'],
                reasons=reasons[:5],
                strategy_name=self.name,
                indicators_used=['rsi', 'adx', 'macd_histogram', 'atr', 'volume_ratio', 'cum_buy_delta', 'cum_sell_delta']
            )
        
        return None

    def _adapt_weights(self, regime: str):
        """Adapt layer weights based on market regime"""
        
        if 'TRENDING' in regime:
            self.weights = {
                'accumulation': 0.20,
                'liquidity_sweep': 0.15,
                'volatility': 0.15,
                'momentum': 0.25,
                'volume': 0.25
            }
        elif 'RANGING' in regime:
            self.weights = {
                'accumulation': 0.30,
                'liquidity_sweep': 0.25,
                'volatility': 0.10,
                'momentum': 0.10,
                'volume': 0.25
            }
        elif 'HIGH_VOLATILITY' in regime:
            self.weights = {
                'accumulation': 0.15,
                'liquidity_sweep': 0.15,
                'volatility': 0.30,
                'momentum': 0.20,
                'volume': 0.20
            }
        else:
            self.weights = {
                'accumulation': 0.25,
                'liquidity_sweep': 0.20,
                'volatility': 0.15,
                'momentum': 0.15,
                'volume': 0.25
            }

    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = indicators.get('price', 0)
        atr = indicators.get('atr', current_price * 0.01)
        hidden_support = indicators.get('hidden_support', current_price - atr)
        hidden_resistance = indicators.get('hidden_resistance', current_price + atr)
        
        if action == 'BUY':
            entry = current_price
            stop_loss = hidden_support * 0.99
            risk = entry - stop_loss
            take_profit = entry + (risk * 3.0)
        
        else:  # SELL
            entry = current_price
            stop_loss = hidden_resistance * 1.01
            risk = stop_loss - entry
            take_profit = entry - (risk * 3.0)
        
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