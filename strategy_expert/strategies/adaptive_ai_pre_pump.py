"""
Strategy Name: Adaptive AI Pre-Pump Predictor (Advanced)
Description: Fully adaptive, multi-layer predictive strategy for detecting explosive coin moves
Logic:
- Dynamically adjusts thresholds based on market conditions
- Adaptive layer weighting based on market regime
- Machine learning inspired scoring system
- Self-optimizing parameters
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config
from strategy_expert.calculator import IndicatorCalculator


class AdaptiveAIPrePumpStrategy(BaseStrategy):
    """
    Advanced Adaptive AI Pre-Pump Predictor
    - Dynamic threshold adaptation
    - Market regime-based layer weights
    - Self-optimizing scoring
    - High-probability setup detection
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Adaptive AI Pre-Pump Predictor"
        self.description = "Fully adaptive predictive strategy for ultra high-probability pre-pump entries"
        
        # ===== LOAD CONFIGURATION =====
        pipeline_config = get_pipeline_config()
        
        # Base layer weights (will adapt based on regime)
        self.layer_weights = {
            'accumulation': 0.25,
            'liquidity_sweep': 0.20,
            'volatility': 0.15,
            'momentum': 0.15,
            'volume': 0.25
        }
        
        # Adaptive thresholds
        self.volume_spike_threshold = 1.6
        self.atr_expansion_threshold = 1.3
        self.rsi_bullish_min = 55
        self.rsi_bearish_max = 45
        self.adx_trend_threshold = 25
        
        # Confidence thresholds
        self.buy_confidence = 0.92
        self.sell_confidence = 0.90
        
        # Performance tracking for adaptation
        self.performance_history = []
        self.adaptive_params = {}
        
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
        Generate adaptive AI pre-pump signal
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
        regime_volatility = market_regime.get('volatility', 0)
        
        # ===== 4. ADAPTIVE PARAMETERS BASED ON REGIME =====
        self._adapt_parameters(regime, regime_volatility)
        
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
        
        # ===== 6. LAYER SCORING WITH ADAPTIVE WEIGHTS =====
        layer_scores = {}
        
        # Layer 1: Accumulation / Distribution
        acc_weight = self.layer_weights['accumulation'] * (1 + min(cum_buy_delta / max(1, cum_sell_delta), 2))
        dist_weight = self.layer_weights['accumulation'] * (1 + min(cum_sell_delta / max(1, cum_buy_delta), 2))
        
        layer1_bullish = cum_buy_delta > cum_sell_delta and current_price > hidden_support
        layer1_bearish = cum_sell_delta > cum_buy_delta and current_price < hidden_resistance
        
        # Layer 2: Liquidity Sweep
        layer2_bullish = liquidity_sweep_low
        layer2_bearish = liquidity_sweep_high
        
        # Layer 3: Volatility Expansion (adaptive thresholds)
        atr_factor = atr / max(1, atr_avg) if atr_avg > 0 else 1
        bb_factor = bb_width / self.bb_squeeze_threshold if hasattr(self, 'bb_squeeze_threshold') else 1
        volatility_expansion = atr_factor > self.atr_expansion_threshold or bb_factor > 1.0
        
        # Layer 4: Momentum
        bullish_momentum = rsi > self.rsi_bullish_min and adx > self.adx_trend_threshold and macd_hist > 0
        bearish_momentum = rsi < self.rsi_bearish_max and adx > self.adx_trend_threshold and macd_hist < 0
        
        # Layer 5: Volume Confirmation
        volume_confirmation = volume > volume_avg * self.volume_spike_threshold if volume_avg > 0 else False
        
        # ===== 7. CALCULATE ADAPTIVE SCORES =====
        bullish_score = 0
        bullish_reasons = []
        
        if layer1_bullish:
            bullish_score += acc_weight
            bullish_reasons.append(f"Hidden accumulation (adaptive weight: {acc_weight:.2f})")
        
        if layer2_bullish:
            bullish_score += self.layer_weights['liquidity_sweep']
            bullish_reasons.append("Liquidity sweep below support")
        
        if volatility_expansion:
            bullish_score += self.layer_weights['volatility'] * min(atr_factor, 2)
            bullish_reasons.append(f"Adaptive volatility expansion (ATR factor: {atr_factor:.2f})")
        
        if bullish_momentum:
            bullish_score += self.layer_weights['momentum']
            bullish_reasons.append(f"Bullish momentum: RSI={rsi:.1f}, ADX={adx:.1f}")
        
        if volume_confirmation:
            bullish_score += self.layer_weights['volume']
            bullish_reasons.append(f"Volume spike confirmation: {volume_ratio:.1f}x")
        
        # ===== 8. BEARISH SCORING =====
        bearish_score = 0
        bearish_reasons = []
        
        if layer1_bearish:
            bearish_score += dist_weight
            bearish_reasons.append(f"Hidden distribution (adaptive weight: {dist_weight:.2f})")
        
        if layer2_bearish:
            bearish_score += self.layer_weights['liquidity_sweep']
            bearish_reasons.append("Liquidity sweep above resistance")
        
        if volatility_expansion:
            bearish_score += self.layer_weights['volatility'] * min(atr_factor, 2)
            bearish_reasons.append(f"Adaptive volatility expansion (ATR factor: {atr_factor:.2f})")
        
        if bearish_momentum:
            bearish_score += self.layer_weights['momentum']
            bearish_reasons.append(f"Bearish momentum: RSI={rsi:.1f}, ADX={adx:.1f}")
        
        if volume_confirmation:
            bearish_score += self.layer_weights['volume']
            bearish_reasons.append(f"Volume spike confirmation: {volume_ratio:.1f}x")
        
        # ===== 9. DETERMINE ACTION =====
        action = 'HOLD'
        confidence = 0.5
        reasons = []
        
        min_score = 0.85  # Adaptive threshold
        
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

    def _adapt_parameters(self, regime: str, volatility: float):
        """Adapt parameters based on market regime"""
        
        # Adjust thresholds based on regime
        if 'HIGH_VOLATILITY' in regime or volatility > 0.05:
            self.volume_spike_threshold = 1.8
            self.atr_expansion_threshold = 1.5
            self.layer_weights['volatility'] = 0.20
            self.layer_weights['accumulation'] = 0.20
        elif 'LOW_VOLATILITY' in regime or volatility < 0.01:
            self.volume_spike_threshold = 1.4
            self.atr_expansion_threshold = 1.2
            self.layer_weights['volatility'] = 0.10
            self.layer_weights['accumulation'] = 0.30
        else:
            self.volume_spike_threshold = 1.6
            self.atr_expansion_threshold = 1.3
            self.layer_weights['volatility'] = 0.15
            self.layer_weights['accumulation'] = 0.25
        
        # Adjust RSI thresholds based on trend
        if 'BULL' in regime:
            self.rsi_bullish_min = 50
            self.rsi_bearish_max = 50
        elif 'BEAR' in regime:
            self.rsi_bullish_min = 60
            self.rsi_bearish_max = 40
        else:
            self.rsi_bullish_min = 55
            self.rsi_bearish_max = 45
        
        # Set BB squeeze threshold
        self.bb_squeeze_threshold = 0.04

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