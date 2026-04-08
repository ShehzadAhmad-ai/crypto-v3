# risk_analyzer.py - Advanced Risk Analyzer with Multi-Method Entry Zones
"""
Advanced Risk Analyzer for Phase 8
Calculates:
- Dynamic entry zones using SMC + Indicators + Structure + Volume
- Risk-managed stop loss (preserves original, adds new)
- Risk-managed take profit (preserves original, adds new)
- Uses advanced indicators for fallback (Keltner, Donchian, PSAR, ATR Trailing)
- NEVER overwrites original SL/TP from experts

Version: 3.0
Author: Risk Management Expert
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from config import Config
from logger import log

EPSILON = 1e-10


class StopPriority(Enum):
    """Priority hierarchy for stop loss placement"""
    ORDER_BLOCK = 1
    FAIR_VALUE_GAP = 2
    LIQUIDITY_SWEEP = 3
    SWING_POINT = 4
    SUPPORT_RESISTANCE = 5
    ADVANCED_INDICATOR = 6   # Keltner, Donchian, PSAR, ATR Trailing
    VOLUME_BASED = 7
    ATR_FALLBACK = 8


class TPPriority(Enum):
    """Priority hierarchy for take profit placement"""
    HTF_LEVEL = 1
    LIQUIDITY_CLUSTER = 2
    FIBONACCI_EXTENSION = 3
    MARKET_STRUCTURE = 4
    ADVANCED_INDICATOR = 5
    RR_BASED = 6


class EntryZoneMethod(Enum):
    """Methods for entry zone calculation"""
    SMC_ORDER_BLOCK = "SMC_ORDER_BLOCK"
    SMC_FAIR_VALUE_GAP = "SMC_FAIR_VALUE_GAP"
    SMC_LIQUIDITY_SWEEP = "SMC_LIQUIDITY_SWEEP"
    INDICATOR_BOLLINGER = "INDICATOR_BOLLINGER"
    INDICATOR_VWAP = "INDICATOR_VWAP"
    INDICATOR_EMA = "INDICATOR_EMA"
    INDICATOR_RSI = "INDICATOR_RSI"
    STRUCTURE_SR = "STRUCTURE_SR"
    STRUCTURE_SWING = "STRUCTURE_SWING"
    STRUCTURE_MTF = "STRUCTURE_MTF"
    VOLUME_PROFILE = "VOLUME_PROFILE"
    VOLUME_CVD = "VOLUME_CVD"
    VOLUME_ABSORPTION = "VOLUME_ABSORPTION"


@dataclass
class EntryZoneResult:
    """Complete entry zone calculation result"""
    # Buy zone (for BUY signals)
    buy_zone_low: float
    buy_zone_high: float
    
    # Sell zone (for SELL signals)
    sell_zone_low: float
    sell_zone_high: float
    
    # Which method was used
    method: str
    confidence: float
    width_pct: float
    reason: str
    
    # All contributing methods for debugging
    contributing_methods: Dict[str, Dict] = None


@dataclass
class StopLossResult:
    """Complete stop loss calculation result"""
    price: float
    method: str
    distance_pct: float
    atr_multiple: float
    confidence: float
    reason: str


@dataclass
class TakeProfitResult:
    """Complete take profit calculation result"""
    price: float
    method: str
    distance_pct: float
    atr_multiple: float
    risk_reward: float
    confidence: float
    reason: str


class RiskAnalyzer:
    """
    Advanced Risk Analyzer using ALL market structure data
    NEVER overwrites original SL/TP - only ADDS risk-managed versions
    """
    
    def __init__(self):
        # Load settings from config
        self.min_rr = getattr(Config, 'MIN_RISK_REWARD', 1.5)
        self.target_rr = getattr(Config, 'TARGET_RISK_REWARD', 2.5)
        
        # Entry zone weights
        self.entry_zone_weights = getattr(Config, 'ENTRY_ZONE_WEIGHTS', {
            'smc_order_block': 0.25,
            'smc_fvg': 0.20,
            'smc_liquidity': 0.15,
            'indicator_bollinger': 0.10,
            'indicator_vwap': 0.10,
            'structure_sr': 0.10,
            'volume_profile': 0.10
        })
        
        # Advanced indicators for stop loss
        self.enable_advanced_stop = getattr(Config, 'ENABLE_ADVANCED_STOP_INDICATORS', True)
        self.stop_indicators = getattr(Config, 'STOP_INDICATORS', ['keltner', 'donchian', 'psar', 'atr_trailing'])
        
        # Buffer percentages (0.2% default)
        self.buffer_pct = getattr(Config, 'STOP_BUFFER_PCT', 0.002)
        
        log.info("RiskAnalyzer initialized with multi-method entry zones")
    
    # =========================================================================
    # ENTRY ZONE CALCULATION (Multi-Method)
    # =========================================================================
    
    def calculate_entry_zone(self,
                             direction: str,
                             current_price: float,
                             atr: float,
                             df: pd.DataFrame,
                             order_blocks: List[Dict],
                             fair_value_gaps: List[Dict],
                             liquidity_sweeps: List[Dict],
                             supports: List[float],
                             resistances: List[float],
                             swing_points: List[Dict],
                             htf_levels: List[float],
                             volume_profile: Dict,
                             vwap: float,
                             rsi: float,
                             ema_fast: float,
                             ema_slow: float,
                             cvd_data: Dict = None,
                             absorption_data: Dict = None) -> EntryZoneResult:
        """
        Calculate entry zone using ALL available methods.
        Returns buy zone for BUY signals, sell zone for SELL signals.
        
        Args:
            direction: 'BUY' or 'SELL'
            current_price: Current market price
            atr: Average True Range
            df: OHLCV DataFrame
            order_blocks: List of SMC order blocks
            fair_value_gaps: List of FVGs
            liquidity_sweeps: List of liquidity sweeps
            supports: List of support levels
            resistances: List of resistance levels
            swing_points: List of swing points (HH/HL/LH/LL)
            htf_levels: Higher timeframe levels
            volume_profile: Volume profile data (POC, VAL, VAH)
            vwap: VWAP value
            rsi: Current RSI value
            ema_fast: Fast EMA value (e.g., EMA8)
            ema_slow: Slow EMA value (e.g., EMA21)
            cvd_data: CVD divergence data
            absorption_data: Absorption zone data
        
        Returns:
            EntryZoneResult with buy and sell zones
        """
        
        all_zones = []
        
        # ===== METHOD 1: SMC ORDER BLOCKS =====
        smc_zones = self._get_smc_order_block_zones(direction, current_price, order_blocks, atr)
        if smc_zones:
            all_zones.append({
                'method': EntryZoneMethod.SMC_ORDER_BLOCK,
                'zone': smc_zones,
                'confidence': 0.85,
                'weight': self.entry_zone_weights.get('smc_order_block', 0.25)
            })
        
        # ===== METHOD 2: SMC FAIR VALUE GAPS =====
        fvg_zones = self._get_smc_fvg_zones(direction, current_price, fair_value_gaps, atr)
        if fvg_zones:
            all_zones.append({
                'method': EntryZoneMethod.SMC_FAIR_VALUE_GAP,
                'zone': fvg_zones,
                'confidence': 0.80,
                'weight': self.entry_zone_weights.get('smc_fvg', 0.20)
            })
        
        # ===== METHOD 3: SMC LIQUIDITY SWEEPS =====
        sweep_zones = self._get_smc_liquidity_zones(direction, current_price, liquidity_sweeps, atr)
        if sweep_zones:
            all_zones.append({
                'method': EntryZoneMethod.SMC_LIQUIDITY_SWEEP,
                'zone': sweep_zones,
                'confidence': 0.75,
                'weight': self.entry_zone_weights.get('smc_liquidity', 0.15)
            })
        
        # ===== METHOD 4: BOLLINGER BANDS =====
        bb_zones = self._get_bollinger_zones(direction, current_price, df, atr)
        if bb_zones:
            all_zones.append({
                'method': EntryZoneMethod.INDICATOR_BOLLINGER,
                'zone': bb_zones,
                'confidence': 0.70,
                'weight': self.entry_zone_weights.get('indicator_bollinger', 0.10)
            })
        
        # ===== METHOD 5: VWAP DISCOUNT/PREMIUM ZONES =====
        vwap_zones = self._get_vwap_zones(direction, current_price, vwap, atr)
        if vwap_zones:
            all_zones.append({
                'method': EntryZoneMethod.INDICATOR_VWAP,
                'zone': vwap_zones,
                'confidence': 0.65,
                'weight': self.entry_zone_weights.get('indicator_vwap', 0.10)
            })
        
        # ===== METHOD 6: EMA PULLBACK ZONES =====
        ema_zones = self._get_ema_zones(direction, current_price, ema_fast, ema_slow, atr)
        if ema_zones:
            all_zones.append({
                'method': EntryZoneMethod.INDICATOR_EMA,
                'zone': ema_zones,
                'confidence': 0.65,
                'weight': 0.08  # Slightly lower weight
            })
        
        # ===== METHOD 7: RSI ZONES =====
        rsi_zones = self._get_rsi_zones(direction, current_price, rsi, atr)
        if rsi_zones:
            all_zones.append({
                'method': EntryZoneMethod.INDICATOR_RSI,
                'zone': rsi_zones,
                'confidence': 0.60,
                'weight': 0.05
            })
        
        # ===== METHOD 8: SUPPORT/RESISTANCE ZONES =====
        sr_zones = self._get_sr_zones(direction, current_price, supports, resistances, atr)
        if sr_zones:
            all_zones.append({
                'method': EntryZoneMethod.STRUCTURE_SR,
                'zone': sr_zones,
                'confidence': 0.70,
                'weight': self.entry_zone_weights.get('structure_sr', 0.10)
            })
        
        # ===== METHOD 9: SWING POINT ZONES =====
        swing_zones = self._get_swing_zones(direction, current_price, swing_points, atr)
        if swing_zones:
            all_zones.append({
                'method': EntryZoneMethod.STRUCTURE_SWING,
                'zone': swing_zones,
                'confidence': 0.70,
                'weight': 0.05
            })
        
        # ===== METHOD 10: MULTI-TIMEFRAME LEVEL ZONES =====
        mtf_zones = self._get_mtf_zones(direction, current_price, htf_levels, atr)
        if mtf_zones:
            all_zones.append({
                'method': EntryZoneMethod.STRUCTURE_MTF,
                'zone': mtf_zones,
                'confidence': 0.80,
                'weight': 0.08
            })
        
        # ===== METHOD 11: VOLUME PROFILE ZONES =====
        vp_zones = self._get_volume_profile_zones(direction, current_price, volume_profile, atr)
        if vp_zones:
            all_zones.append({
                'method': EntryZoneMethod.VOLUME_PROFILE,
                'zone': vp_zones,
                'confidence': 0.75,
                'weight': self.entry_zone_weights.get('volume_profile', 0.10)
            })
        
        # ===== METHOD 12: CVD DIVERGENCE ZONES =====
        if cvd_data:
            cvd_zones = self._get_cvd_zones(direction, current_price, cvd_data, atr)
            if cvd_zones:
                all_zones.append({
                    'method': EntryZoneMethod.VOLUME_CVD,
                    'zone': cvd_zones,
                    'confidence': 0.70,
                    'weight': 0.05
                })
        
        # ===== METHOD 13: ABSORPTION ZONES =====
        if absorption_data:
            abs_zones = self._get_absorption_zones(direction, current_price, absorption_data, atr)
            if abs_zones:
                all_zones.append({
                    'method': EntryZoneMethod.VOLUME_ABSORPTION,
                    'zone': abs_zones,
                    'confidence': 0.75,
                    'weight': 0.05
                })
        
        # ===== AGGREGATE ALL ZONES =====
        if not all_zones:
            # Fallback to ATR-based zone
            return self._fallback_entry_zone(direction, current_price, atr)
        
        # Collect all buy zones (for BUY direction) and sell zones (for SELL direction)
        buy_zone_lows = []
        buy_zone_highs = []
        sell_zone_lows = []
        sell_zone_highs = []
        weights = []
        
        for zone_data in all_zones:
            zone = zone_data['zone']
            weight = zone_data['weight'] * zone_data['confidence']
            method = zone_data['method'].value
            
            if direction == 'BUY':
                if zone.get('buy_low') and zone.get('buy_high'):
                    buy_zone_lows.append(zone['buy_low'])
                    buy_zone_highs.append(zone['buy_high'])
                    weights.append(weight)
            else:  # SELL
                if zone.get('sell_low') and zone.get('sell_high'):
                    sell_zone_lows.append(zone['sell_low'])
                    sell_zone_highs.append(zone['sell_high'])
                    weights.append(weight)
        
        # Calculate weighted median for buy zone
        if direction == 'BUY' and buy_zone_lows:
            # Use weighted median (simplified: average of weighted values)
            total_weight = sum(weights)
            if total_weight > 0:
                buy_zone_low = sum(l * w for l, w in zip(buy_zone_lows, weights)) / total_weight
                buy_zone_high = sum(h * w for h, w in zip(buy_zone_highs, weights)) / total_weight
            else:
                buy_zone_low = np.median(buy_zone_lows)
                buy_zone_high = np.median(buy_zone_highs)
            
            # Ensure zone is valid
            if buy_zone_low >= buy_zone_high:
                buy_zone_low = buy_zone_high * 0.995
            
            # Add buffer
            buy_zone_low = buy_zone_low * 0.998
            buy_zone_high = buy_zone_high * 1.002
            
            zone_width = (buy_zone_high - buy_zone_low) / buy_zone_low
            
            return EntryZoneResult(
                buy_zone_low=round(buy_zone_low, 6),
                buy_zone_high=round(buy_zone_high, 6),
                sell_zone_low=0,
                sell_zone_high=0,
                method=f"Weighted aggregation of {len(all_zones)} methods",
                confidence=min(0.95, sum(z['confidence'] * z['weight'] for z in all_zones) / sum(z['weight'] for z in all_zones)),
                width_pct=round(zone_width, 4),
                reason=f"Entry zone from {len(all_zones)} methods",
                contributing_methods={z['method'].value: {'zone': z['zone'], 'confidence': z['confidence']} for z in all_zones}
            )
        
        elif direction == 'SELL' and sell_zone_lows:
            total_weight = sum(weights)
            if total_weight > 0:
                sell_zone_low = sum(l * w for l, w in zip(sell_zone_lows, weights)) / total_weight
                sell_zone_high = sum(h * w for h, w in zip(sell_zone_highs, weights)) / total_weight
            else:
                sell_zone_low = np.median(sell_zone_lows)
                sell_zone_high = np.median(sell_zone_highs)
            
            if sell_zone_low >= sell_zone_high:
                sell_zone_high = sell_zone_low * 1.005
            
            sell_zone_low = sell_zone_low * 0.998
            sell_zone_high = sell_zone_high * 1.002
            
            zone_width = (sell_zone_high - sell_zone_low) / sell_zone_low
            
            return EntryZoneResult(
                buy_zone_low=0,
                buy_zone_high=0,
                sell_zone_low=round(sell_zone_low, 6),
                sell_zone_high=round(sell_zone_high, 6),
                method=f"Weighted aggregation of {len(all_zones)} methods",
                confidence=min(0.95, sum(z['confidence'] * z['weight'] for z in all_zones) / sum(z['weight'] for z in all_zones)),
                width_pct=round(zone_width, 4),
                reason=f"Entry zone from {len(all_zones)} methods",
                contributing_methods={z['method'].value: {'zone': z['zone'], 'confidence': z['confidence']} for z in all_zones}
            )
        
        # Fallback
        return self._fallback_entry_zone(direction, current_price, atr)
    
    # =========================================================================
    # STOP LOSS CALCULATION (Adds risk-managed stop, preserves original)
    # =========================================================================
    
    def calculate_risk_managed_stop_loss(self,
                                         direction: str,
                                         entry_price: float,
                                         current_price: float,
                                         atr: float,
                                         volatility_regime: str,
                                         market_regime: str,
                                         adx: float,
                                         bb_width: float,
                                         order_blocks: List[Dict],
                                         fair_value_gaps: List[Dict],
                                         liquidity_sweeps: List[Dict],
                                         swing_points: List[Dict],
                                         supports: List[float],
                                         resistances: List[float],
                                         df: pd.DataFrame,
                                         volume_profile: Dict = None) -> StopLossResult:
        """
        Calculate risk-managed stop loss using priority hierarchy.
        NEVER overwrites original stop loss - this is an ADDITIONAL stop.
        
        Uses advanced indicators for fallback:
        - Keltner Channels
        - Donchian Channels
        - Parabolic SAR
        - ATR Trailing Stop
        
        Returns:
            StopLossResult with stop price and metadata
        """
        
        # ===== PRIORITY 1: ORDER BLOCKS =====
        stop = self._get_ob_stop(direction, current_price, order_blocks, atr)
        if stop:
            return stop
        
        # ===== PRIORITY 2: FAIR VALUE GAPS =====
        stop = self._get_fvg_stop(direction, current_price, fair_value_gaps, atr)
        if stop:
            return stop
        
        # ===== PRIORITY 3: LIQUIDITY SWEEPS =====
        stop = self._get_sweep_stop(direction, current_price, liquidity_sweeps, atr)
        if stop:
            return stop
        
        # ===== PRIORITY 4: SWING POINTS =====
        stop = self._get_swing_stop(direction, current_price, swing_points, atr)
        if stop:
            return stop
        
        # ===== PRIORITY 5: SUPPORT/RESISTANCE =====
        stop = self._get_sr_stop(direction, current_price, supports, resistances, atr)
        if stop:
            return stop
        
        # ===== PRIORITY 6: ADVANCED INDICATORS =====
        if self.enable_advanced_stop:
            stop = self._get_advanced_indicator_stop(direction, current_price, atr, df)
            if stop:
                return stop
        
        # ===== PRIORITY 7: VOLUME-BASED STOP =====
        if volume_profile:
            stop = self._get_volume_stop(direction, current_price, volume_profile, atr)
            if stop:
                return stop
        
        # ===== PRIORITY 8: ADVANCED ATR FALLBACK =====
        return self._get_advanced_atr_stop(direction, entry_price, current_price, atr,
                                           volatility_regime, market_regime, adx, bb_width)
    
    # =========================================================================
    # TAKE PROFIT CALCULATION (Adds risk-managed TP, preserves original)
    # =========================================================================
    
    def calculate_risk_managed_take_profit(self,
                                           direction: str,
                                           entry_price: float,
                                           stop_loss: float,
                                           current_price: float,
                                           atr: float,
                                           resistances: List[float],
                                           supports: List[float],
                                           liquidity_clusters: List[Dict],
                                           htf_levels: List[float],
                                           fib_extensions: Dict,
                                           swing_points: List[Dict],
                                           df: pd.DataFrame,
                                           confidence: float,
                                           win_rate: float = 0.5) -> TakeProfitResult:
        """
        Calculate risk-managed take profit using priority hierarchy.
        NEVER overwrites original TP - this is an ADDITIONAL target.
        
        Returns:
            TakeProfitResult with TP price and metadata
        """
        
        risk = abs(entry_price - stop_loss)
        
        # ===== PRIORITY 1: HTF LEVELS =====
        tp = self._get_htf_tp(direction, current_price, htf_levels, risk)
        if tp and tp.risk_reward >= self.min_rr:
            return tp
        
        # ===== PRIORITY 2: LIQUIDITY CLUSTERS =====
        tp = self._get_liquidity_tp(direction, current_price, liquidity_clusters, risk)
        if tp and tp.risk_reward >= self.min_rr:
            return tp
        
        # ===== PRIORITY 3: FIBONACCI EXTENSIONS =====
        tp = self._get_fib_tp(direction, entry_price, stop_loss, fib_extensions, risk)
        if tp and tp.risk_reward >= self.min_rr:
            return tp
        
        # ===== PRIORITY 4: MARKET STRUCTURE TARGETS =====
        tp = self._get_structure_tp(direction, current_price, swing_points, risk)
        if tp and tp.risk_reward >= self.min_rr:
            return tp
        
        # ===== PRIORITY 5: ADVANCED INDICATOR TARGETS =====
        tp = self._get_indicator_tp(direction, current_price, df, risk)
        if tp and tp.risk_reward >= self.min_rr:
            return tp
        
        # ===== PRIORITY 6: DYNAMIC RR-BASED TARGET =====
        return self._get_rr_based_tp(direction, entry_price, stop_loss, risk,
                                     confidence, win_rate)
    
    # =========================================================================
    # PRIVATE METHODS - ENTRY ZONE CALCULATORS
    # =========================================================================
    
    def _get_smc_order_block_zones(self, direction, current_price, order_blocks, atr):
        """Get entry zone from SMC order blocks"""
        if not order_blocks:
            return None
        
        if direction == 'BUY':
            # Bullish order blocks below price
            bullish_obs = [ob for ob in order_blocks 
                          if ob.get('type') == 'BULLISH_OB' 
                          and ob.get('price', 0) < current_price]
            if bullish_obs:
                bullish_obs.sort(key=lambda x: x.get('price', 0), reverse=True)
                best_ob = bullish_obs[0]
                ob_price = best_ob.get('price', 0)
                
                return {
                    'buy_low': ob_price * 0.998,
                    'buy_high': ob_price,
                    'sell_low': 0,
                    'sell_high': 0
                }
        
        else:  # SELL
            bearish_obs = [ob for ob in order_blocks 
                          if ob.get('type') == 'BEARISH_OB' 
                          and ob.get('price', 0) > current_price]
            if bearish_obs:
                bearish_obs.sort(key=lambda x: x.get('price', 0))
                best_ob = bearish_obs[0]
                ob_price = best_ob.get('price', 0)
                
                return {
                    'buy_low': 0,
                    'buy_high': 0,
                    'sell_low': ob_price,
                    'sell_high': ob_price * 1.002
                }
        
        return None
    
    def _get_smc_fvg_zones(self, direction, current_price, fair_value_gaps, atr):
        """Get entry zone from SMC Fair Value Gaps"""
        if not fair_value_gaps:
            return None
        
        if direction == 'BUY':
            bullish_fvgs = [fvg for fvg in fair_value_gaps 
                           if fvg.get('direction') == 'BULLISH' 
                           and fvg.get('high', 0) < current_price]
            if bullish_fvgs:
                bullish_fvgs.sort(key=lambda x: x.get('high', 0), reverse=True)
                best_fvg = bullish_fvgs[0]
                fvg_low = best_fvg.get('low', 0)
                fvg_high = best_fvg.get('high', 0)
                
                return {
                    'buy_low': fvg_low,
                    'buy_high': fvg_high,
                    'sell_low': 0,
                    'sell_high': 0
                }
        
        else:  # SELL
            bearish_fvgs = [fvg for fvg in fair_value_gaps 
                           if fvg.get('direction') == 'BEARISH' 
                           and fvg.get('low', 0) > current_price]
            if bearish_fvgs:
                bearish_fvgs.sort(key=lambda x: x.get('low', 0))
                best_fvg = bearish_fvgs[0]
                fvg_low = best_fvg.get('low', 0)
                fvg_high = best_fvg.get('high', 0)
                
                return {
                    'buy_low': 0,
                    'buy_high': 0,
                    'sell_low': fvg_low,
                    'sell_high': fvg_high
                }
        
        return None
    
    def _get_smc_liquidity_zones(self, direction, current_price, liquidity_sweeps, atr):
        """Get entry zone from liquidity sweeps"""
        if not liquidity_sweeps:
            return None
        
        if direction == 'BUY':
            down_sweeps = [s for s in liquidity_sweeps 
                          if s.get('type') in ['DOWN_SWEEP', 'DOWN_SWEEP_FAILURE']
                          and s.get('price', 0) < current_price]
            if down_sweeps:
                down_sweeps.sort(key=lambda x: x.get('price', 0), reverse=True)
                best_sweep = down_sweeps[0]
                sweep_price = best_sweep.get('price', 0)
                
                return {
                    'buy_low': sweep_price * 0.998,
                    'buy_high': sweep_price,
                    'sell_low': 0,
                    'sell_high': 0
                }
        
        else:  # SELL
            up_sweeps = [s for s in liquidity_sweeps 
                        if s.get('type') in ['UP_SWEEP', 'UP_SWEEP_FAILURE']
                        and s.get('price', 0) > current_price]
            if up_sweeps:
                up_sweeps.sort(key=lambda x: x.get('price', 0))
                best_sweep = up_sweeps[0]
                sweep_price = best_sweep.get('price', 0)
                
                return {
                    'buy_low': 0,
                    'buy_high': 0,
                    'sell_low': sweep_price,
                    'sell_high': sweep_price * 1.002
                }
        
        return None
    
    def _get_bollinger_zones(self, direction, current_price, df, atr):
        """Get entry zone from Bollinger Bands"""
        if 'bb_lower' not in df.columns or 'bb_upper' not in df.columns:
            return None
        
        bb_lower = float(df['bb_lower'].iloc[-1])
        bb_upper = float(df['bb_upper'].iloc[-1])
        bb_middle = float(df['bb_middle'].iloc[-1]) if 'bb_middle' in df.columns else (bb_lower + bb_upper) / 2
        
        if direction == 'BUY':
            # Entry zone at lower band
            return {
                'buy_low': bb_lower * 0.998,
                'buy_high': bb_lower * 1.002,
                'sell_low': 0,
                'sell_high': 0
            }
        else:  # SELL
            return {
                'buy_low': 0,
                'buy_high': 0,
                'sell_low': bb_upper * 0.998,
                'sell_high': bb_upper * 1.002
            }
    
    def _get_vwap_zones(self, direction, current_price, vwap, atr):
        """Get entry zone from VWAP discount/premium zones"""
        if vwap <= 0:
            return None
        
        if direction == 'BUY':
            # Discount zone: VWAP × 0.97 to VWAP
            discount_low = vwap * 0.97
            discount_high = vwap
            
            return {
                'buy_low': discount_low,
                'buy_high': discount_high,
                'sell_low': 0,
                'sell_high': 0
            }
        else:  # SELL
            # Premium zone: VWAP to VWAP × 1.03
            premium_low = vwap
            premium_high = vwap * 1.03
            
            return {
                'buy_low': 0,
                'buy_high': 0,
                'sell_low': premium_low,
                'sell_high': premium_high
            }
    
    def _get_ema_zones(self, direction, current_price, ema_fast, ema_slow, atr):
        """Get entry zone from EMA pullback"""
        if ema_fast <= 0 or ema_slow <= 0:
            return None
        
        if direction == 'BUY':
            # Entry zone near EMA21 (pullback level)
            entry_level = ema_slow
            
            return {
                'buy_low': entry_level * 0.998,
                'buy_high': entry_level * 1.002,
                'sell_low': 0,
                'sell_high': 0
            }
        else:  # SELL
            entry_level = ema_slow
            
            return {
                'buy_low': 0,
                'buy_high': 0,
                'sell_low': entry_level * 0.998,
                'sell_high': entry_level * 1.002
            }
    
    def _get_rsi_zones(self, direction, current_price, rsi, atr):
        """Get entry zone from RSI oversold/overbought zones"""
        if rsi <= 0:
            return None
        
        if direction == 'BUY' and rsi < 30:
            # Oversold - entry zone with ATR buffer
            return {
                'buy_low': current_price - (atr * 0.5),
                'buy_high': current_price,
                'sell_low': 0,
                'sell_high': 0
            }
        elif direction == 'SELL' and rsi > 70:
            return {
                'buy_low': 0,
                'buy_high': 0,
                'sell_low': current_price,
                'sell_high': current_price + (atr * 0.5)
            }
        
        return None
    
    def _get_sr_zones(self, direction, current_price, supports, resistances, atr):
        """Get entry zone from support/resistance levels"""
        if direction == 'BUY' and supports:
            valid_supports = [s for s in supports if s < current_price]
            if valid_supports:
                nearest_support = max(valid_supports)
                return {
                    'buy_low': nearest_support * 0.998,
                    'buy_high': nearest_support,
                    'sell_low': 0,
                    'sell_high': 0
                }
        
        elif direction == 'SELL' and resistances:
            valid_resistances = [r for r in resistances if r > current_price]
            if valid_resistances:
                nearest_resistance = min(valid_resistances)
                return {
                    'buy_low': 0,
                    'buy_high': 0,
                    'sell_low': nearest_resistance,
                    'sell_high': nearest_resistance * 1.002
                }
        
        return None
    
    def _get_swing_zones(self, direction, current_price, swing_points, atr):
        """Get entry zone from swing points"""
        if not swing_points:
            return None
        
        if direction == 'BUY':
            # Find HL (Higher Low) or LL (Lower Low) below price
            swing_lows = [s for s in swing_points 
                         if s.get('type') in ['HL', 'LL'] 
                         and s.get('price', 0) < current_price]
            if swing_lows:
                swing_lows.sort(key=lambda x: x.get('price', 0), reverse=True)
                nearest_low = swing_lows[0].get('price', 0)
                return {
                    'buy_low': nearest_low * 0.998,
                    'buy_high': nearest_low,
                    'sell_low': 0,
                    'sell_high': 0
                }
        
        else:  # SELL
            swing_highs = [s for s in swing_points 
                          if s.get('type') in ['HH', 'LH'] 
                          and s.get('price', 0) > current_price]
            if swing_highs:
                swing_highs.sort(key=lambda x: x.get('price', 0))
                nearest_high = swing_highs[0].get('price', 0)
                return {
                    'buy_low': 0,
                    'buy_high': 0,
                    'sell_low': nearest_high,
                    'sell_high': nearest_high * 1.002
                }
        
        return None
    
    def _get_mtf_zones(self, direction, current_price, htf_levels, atr):
        """Get entry zone from higher timeframe levels"""
        if not htf_levels:
            return None
        
        if direction == 'BUY':
            # Find HTF support below price
            supports = [l for l in htf_levels if l < current_price]
            if supports:
                nearest = max(supports)
                return {
                    'buy_low': nearest * 0.998,
                    'buy_high': nearest,
                    'sell_low': 0,
                    'sell_high': 0
                }
        
        else:  # SELL
            resistances = [l for l in htf_levels if l > current_price]
            if resistances:
                nearest = min(resistances)
                return {
                    'buy_low': 0,
                    'buy_high': 0,
                    'sell_low': nearest,
                    'sell_high': nearest * 1.002
                }
        
        return None
    
    def _get_volume_profile_zones(self, direction, current_price, volume_profile, atr):
        """Get entry zone from volume profile (POC)"""
        if not volume_profile:
            return None
        
        poc = volume_profile.get('poc', 0)
        val = volume_profile.get('val', 0)
        vah = volume_profile.get('vah', 0)
        
        if poc <= 0:
            return None
        
        if direction == 'BUY' and poc < current_price:
            # Entry at POC (high volume node)
            return {
                'buy_low': poc * 0.998,
                'buy_high': poc,
                'sell_low': 0,
                'sell_high': 0
            }
        elif direction == 'SELL' and poc > current_price:
            return {
                'buy_low': 0,
                'buy_high': 0,
                'sell_low': poc,
                'sell_high': poc * 1.002
            }
        
        return None
    
    def _get_cvd_zones(self, direction, current_price, cvd_data, atr):
        """Get entry zone from CVD divergence"""
        divergence = cvd_data.get('divergence')
        if divergence == 'BULLISH' and direction == 'BUY':
            return {
                'buy_low': current_price - (atr * 0.5),
                'buy_high': current_price,
                'sell_low': 0,
                'sell_high': 0
            }
        elif divergence == 'BEARISH' and direction == 'SELL':
            return {
                'buy_low': 0,
                'buy_high': 0,
                'sell_low': current_price,
                'sell_high': current_price + (atr * 0.5)
            }
        
        return None
    
    def _get_absorption_zones(self, direction, current_price, absorption_data, atr):
        """Get entry zone from absorption zones"""
        if not absorption_data:
            return None
        
        absorption_price = absorption_data.get('price', 0)
        absorption_type = absorption_data.get('type', '')
        
        if absorption_type == 'BULLISH' and direction == 'BUY' and absorption_price < current_price:
            return {
                'buy_low': absorption_price * 0.998,
                'buy_high': absorption_price,
                'sell_low': 0,
                'sell_high': 0
            }
        elif absorption_type == 'BEARISH' and direction == 'SELL' and absorption_price > current_price:
            return {
                'buy_low': 0,
                'buy_high': 0,
                'sell_low': absorption_price,
                'sell_high': absorption_price * 1.002
            }
        
        return None
    
    def _fallback_entry_zone(self, direction, current_price, atr):
        """Fallback entry zone using ATR"""
        atr_mult = 0.5
        
        if direction == 'BUY':
            return EntryZoneResult(
                buy_zone_low=round(current_price - (atr * atr_mult), 6),
                buy_zone_high=round(current_price, 6),
                sell_zone_low=0,
                sell_zone_high=0,
                method="ATR Fallback",
                confidence=0.5,
                width_pct=round(atr_mult * atr / current_price, 4),
                reason="No structure found - using ATR fallback"
            )
        else:
            return EntryZoneResult(
                buy_zone_low=0,
                buy_zone_high=0,
                sell_zone_low=round(current_price, 6),
                sell_zone_high=round(current_price + (atr * atr_mult), 6),
                method="ATR Fallback",
                confidence=0.5,
                width_pct=round(atr_mult * atr / current_price, 4),
                reason="No structure found - using ATR fallback"
            )
    
    # =========================================================================
    # PRIVATE METHODS - STOP LOSS CALCULATORS
    # =========================================================================
    
    def _get_ob_stop(self, direction, current_price, order_blocks, atr):
        """Stop at order block"""
        if not order_blocks:
            return None
        
        if direction == 'BUY':
            bullish_obs = [ob for ob in order_blocks 
                          if ob.get('type') == 'BULLISH_OB' 
                          and ob.get('price', 0) < current_price]
            if bullish_obs:
                bullish_obs.sort(key=lambda x: x.get('price', 0), reverse=True)
                best_ob = bullish_obs[0]
                ob_price = best_ob.get('price', 0)
                stop_price = ob_price * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Order Block at {ob_price:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.85,
                    reason=f"Stop below bullish order block"
                )
        
        else:  # SELL
            bearish_obs = [ob for ob in order_blocks 
                          if ob.get('type') == 'BEARISH_OB' 
                          and ob.get('price', 0) > current_price]
            if bearish_obs:
                bearish_obs.sort(key=lambda x: x.get('price', 0))
                best_ob = bearish_obs[0]
                ob_price = best_ob.get('price', 0)
                stop_price = ob_price * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Order Block at {ob_price:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.85,
                    reason=f"Stop above bearish order block"
                )
        
        return None
    
    def _get_fvg_stop(self, direction, current_price, fair_value_gaps, atr):
        """Stop at fair value gap"""
        if not fair_value_gaps:
            return None
        
        if direction == 'BUY':
            bullish_fvgs = [fvg for fvg in fair_value_gaps 
                           if fvg.get('direction') == 'BULLISH' 
                           and fvg.get('high', 0) < current_price]
            if bullish_fvgs:
                bullish_fvgs.sort(key=lambda x: x.get('high', 0), reverse=True)
                best_fvg = bullish_fvgs[0]
                fvg_low = best_fvg.get('low', 0)
                stop_price = fvg_low * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"FVG at {fvg_low:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.80,
                    reason=f"Stop below bullish FVG"
                )
        
        else:  # SELL
            bearish_fvgs = [fvg for fvg in fair_value_gaps 
                           if fvg.get('direction') == 'BEARISH' 
                           and fvg.get('low', 0) > current_price]
            if bearish_fvgs:
                bearish_fvgs.sort(key=lambda x: x.get('low', 0))
                best_fvg = bearish_fvgs[0]
                fvg_high = best_fvg.get('high', 0)
                stop_price = fvg_high * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"FVG at {fvg_high:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.80,
                    reason=f"Stop above bearish FVG"
                )
        
        return None
    
    def _get_sweep_stop(self, direction, current_price, liquidity_sweeps, atr):
        """Stop at liquidity sweep"""
        if not liquidity_sweeps:
            return None
        
        if direction == 'BUY':
            down_sweeps = [s for s in liquidity_sweeps 
                          if s.get('type') in ['DOWN_SWEEP', 'DOWN_SWEEP_FAILURE']
                          and s.get('price', 0) < current_price]
            if down_sweeps:
                down_sweeps.sort(key=lambda x: x.get('price', 0), reverse=True)
                best_sweep = down_sweeps[0]
                sweep_price = best_sweep.get('price', 0)
                stop_price = sweep_price * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Liquidity Sweep at {sweep_price:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.75,
                    reason=f"Stop below liquidity sweep"
                )
        
        else:  # SELL
            up_sweeps = [s for s in liquidity_sweeps 
                        if s.get('type') in ['UP_SWEEP', 'UP_SWEEP_FAILURE']
                        and s.get('price', 0) > current_price]
            if up_sweeps:
                up_sweeps.sort(key=lambda x: x.get('price', 0))
                best_sweep = up_sweeps[0]
                sweep_price = best_sweep.get('price', 0)
                stop_price = sweep_price * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Liquidity Sweep at {sweep_price:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.75,
                    reason=f"Stop above liquidity sweep"
                )
        
        return None
    
    def _get_swing_stop(self, direction, current_price, swing_points, atr):
        """Stop at swing point"""
        if not swing_points:
            return None
        
        if direction == 'BUY':
            swing_lows = [s for s in swing_points 
                         if s.get('type') in ['HL', 'LL'] 
                         and s.get('price', 0) < current_price]
            if swing_lows:
                swing_lows.sort(key=lambda x: x.get('price', 0), reverse=True)
                nearest_low = swing_lows[0].get('price', 0)
                stop_price = nearest_low * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Swing Low at {nearest_low:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.70,
                    reason=f"Stop below swing low"
                )
        
        else:  # SELL
            swing_highs = [s for s in swing_points 
                          if s.get('type') in ['HH', 'LH'] 
                          and s.get('price', 0) > current_price]
            if swing_highs:
                swing_highs.sort(key=lambda x: x.get('price', 0))
                nearest_high = swing_highs[0].get('price', 0)
                stop_price = nearest_high * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Swing High at {nearest_high:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.70,
                    reason=f"Stop above swing high"
                )
        
        return None
    
    def _get_sr_stop(self, direction, current_price, supports, resistances, atr):
        """Stop at support/resistance"""
        if direction == 'BUY' and supports:
            valid_supports = [s for s in supports if s < current_price]
            if valid_supports:
                nearest_support = max(valid_supports)
                stop_price = nearest_support * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Support at {nearest_support:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.65,
                    reason=f"Stop below support"
                )
        
        elif direction == 'SELL' and resistances:
            valid_resistances = [r for r in resistances if r > current_price]
            if valid_resistances:
                nearest_resistance = min(valid_resistances)
                stop_price = nearest_resistance * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                
                return StopLossResult(
                    price=round(stop_price, 6),
                    method=f"Resistance at {nearest_resistance:.6f}",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.65,
                    reason=f"Stop above resistance"
                )
        
        return None
    
    def _get_advanced_indicator_stop(self, direction, current_price, atr, df):
        """Advanced indicator-based stop using Keltner, Donchian, PSAR, ATR Trailing"""
        
        # Try Keltner Channel
        if 'kc_low' in df.columns and 'kc_high' in df.columns and not pd.isna(df['kc_low'].iloc[-1]):
            kc_low = float(df['kc_low'].iloc[-1])
            kc_high = float(df['kc_high'].iloc[-1])
            
            if direction == 'BUY':
                stop_price = kc_low * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                return StopLossResult(
                    price=round(stop_price, 6),
                    method="Keltner Channel Lower Band",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.70,
                    reason="Stop at Keltner lower band"
                )
            else:
                stop_price = kc_high * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                return StopLossResult(
                    price=round(stop_price, 6),
                    method="Keltner Channel Upper Band",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.70,
                    reason="Stop at Keltner upper band"
                )
        
        # Try Donchian Channel
        if 'donchian_low' in df.columns and 'donchian_high' in df.columns and not pd.isna(df['donchian_low'].iloc[-1]):
            donchian_low = float(df['donchian_low'].iloc[-1])
            donchian_high = float(df['donchian_high'].iloc[-1])
            
            if direction == 'BUY':
                stop_price = donchian_low * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                return StopLossResult(
                    price=round(stop_price, 6),
                    method="Donchian Channel Low",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.68,
                    reason="Stop at Donchian low"
                )
            else:
                stop_price = donchian_high * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                return StopLossResult(
                    price=round(stop_price, 6),
                    method="Donchian Channel High",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.68,
                    reason="Stop at Donchian high"
                )
        
        # Try Parabolic SAR
        if 'psar' in df.columns and not pd.isna(df['psar'].iloc[-1]):
            psar = float(df['psar'].iloc[-1])
            
            if direction == 'BUY':
                if psar < current_price:
                    stop_price = psar * (1 - self.buffer_pct)
                    distance_pct = (current_price - stop_price) / current_price
                    return StopLossResult(
                        price=round(stop_price, 6),
                        method="Parabolic SAR",
                        distance_pct=round(distance_pct, 4),
                        atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                        confidence=0.65,
                        reason="Stop at Parabolic SAR"
                    )
            else:
                if psar > current_price:
                    stop_price = psar * (1 + self.buffer_pct)
                    distance_pct = (stop_price - current_price) / current_price
                    return StopLossResult(
                        price=round(stop_price, 6),
                        method="Parabolic SAR",
                        distance_pct=round(distance_pct, 4),
                        atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                        confidence=0.65,
                        reason="Stop at Parabolic SAR"
                    )
        
        # Try ATR Trailing Stop
        if 'atr_trailing_stop' in df.columns and not pd.isna(df['atr_trailing_stop'].iloc[-1]):
            atr_stop = float(df['atr_trailing_stop'].iloc[-1])
            
            if direction == 'BUY' and atr_stop < current_price:
                stop_price = atr_stop * (1 - self.buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                return StopLossResult(
                    price=round(stop_price, 6),
                    method="ATR Trailing Stop",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.60,
                    reason="Stop at ATR trailing stop"
                )
            elif direction == 'SELL' and atr_stop > current_price:
                stop_price = atr_stop * (1 + self.buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                return StopLossResult(
                    price=round(stop_price, 6),
                    method="ATR Trailing Stop",
                    distance_pct=round(distance_pct, 4),
                    atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                    confidence=0.60,
                    reason="Stop at ATR trailing stop"
                )
        
        return None
    
    def _get_volume_stop(self, direction, current_price, volume_profile, atr):
        """Volume-based stop at POC (Point of Control)"""
        poc = volume_profile.get('poc', 0)
        
        if poc <= 0:
            return None
        
        if direction == 'BUY' and poc < current_price:
            stop_price = poc * (1 - self.buffer_pct)
            distance_pct = (current_price - stop_price) / current_price
            return StopLossResult(
                price=round(stop_price, 6),
                method=f"Volume Profile POC at {poc:.6f}",
                distance_pct=round(distance_pct, 4),
                atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                confidence=0.60,
                reason="Stop below high volume node"
            )
        
        elif direction == 'SELL' and poc > current_price:
            stop_price = poc * (1 + self.buffer_pct)
            distance_pct = (stop_price - current_price) / current_price
            return StopLossResult(
                price=round(stop_price, 6),
                method=f"Volume Profile POC at {poc:.6f}",
                distance_pct=round(distance_pct, 4),
                atr_multiple=round(distance_pct / (atr / current_price) if atr > 0 else 1, 2),
                confidence=0.60,
                reason="Stop above high volume node"
            )
        
        return None
    
    def _get_advanced_atr_stop(self, direction, entry_price, current_price, atr,
                               volatility_regime, market_regime, adx, bb_width):
        """Advanced ATR-based stop with dynamic multiplier"""
        # Base multiplier
        base_multiplier = 1.5
        
        # Volatility factor (based on ATR percentage)
        atr_pct = atr / current_price if current_price > 0 else 0.02
        if atr_pct < 0.005:
            vol_factor = 1.2
        elif atr_pct < 0.015:
            vol_factor = 1.0
        elif atr_pct < 0.03:
            vol_factor = 0.8
        else:
            vol_factor = 0.6
        
        # Regime factor
        if market_regime in ['STRONG_BULL_TREND', 'STRONG_BEAR_TREND']:
            regime_factor = 1.2
        elif market_regime in ['BULL_TREND', 'BEAR_TREND']:
            regime_factor = 1.0
        elif 'RANGING' in market_regime:
            regime_factor = 0.8
        else:
            regime_factor = 0.7
        
        # ADX factor
        adx_factor = 1.0 + min(0.5, (adx - 25) / 100) if adx > 0 else 1.0
        
        # Bollinger width factor
        bb_factor = 1.0
        if bb_width > 0:
            if bb_width < 0.02:
                bb_factor = 1.1
            elif bb_width > 0.06:
                bb_factor = 0.8
        
        # Final multiplier
        final_multiplier = base_multiplier * vol_factor * regime_factor * adx_factor * bb_factor
        final_multiplier = max(0.8, min(2.5, final_multiplier))
        
        if direction == 'BUY':
            stop_price = entry_price - (atr * final_multiplier)
            distance_pct = (current_price - stop_price) / current_price
            return StopLossResult(
                price=round(stop_price, 6),
                method=f"Advanced ATR Stop ({final_multiplier:.1f}x)",
                distance_pct=round(distance_pct, 4),
                atr_multiple=round(final_multiplier, 2),
                confidence=0.55,
                reason=f"Fallback stop using {final_multiplier:.1f}x ATR"
            )
        else:
            stop_price = entry_price + (atr * final_multiplier)
            distance_pct = (stop_price - current_price) / current_price
            return StopLossResult(
                price=round(stop_price, 6),
                method=f"Advanced ATR Stop ({final_multiplier:.1f}x)",
                distance_pct=round(distance_pct, 4),
                atr_multiple=round(final_multiplier, 2),
                confidence=0.55,
                reason=f"Fallback stop using {final_multiplier:.1f}x ATR"
            )
    
    # =========================================================================
    # PRIVATE METHODS - TAKE PROFIT CALCULATORS
    # =========================================================================
    
    def _get_htf_tp(self, direction, current_price, htf_levels, risk):
        """Take profit at higher timeframe level"""
        if not htf_levels:
            return None
        
        if direction == 'BUY':
            levels = [l for l in htf_levels if l > current_price]
            if levels:
                tp_price = min(levels)
                reward = tp_price - current_price
                rr = reward / risk if risk > 0 else 0
                return TakeProfitResult(
                    price=round(tp_price, 6),
                    method="HTF Resistance Level",
                    distance_pct=round(reward / current_price, 4),
                    atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                    risk_reward=round(rr, 2),
                    confidence=0.80,
                    reason=f"Target at HTF resistance {tp_price:.6f}"
                )
        
        else:  # SELL
            levels = [l for l in htf_levels if l < current_price]
            if levels:
                tp_price = max(levels)
                reward = current_price - tp_price
                rr = reward / risk if risk > 0 else 0
                return TakeProfitResult(
                    price=round(tp_price, 6),
                    method="HTF Support Level",
                    distance_pct=round(reward / current_price, 4),
                    atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                    risk_reward=round(rr, 2),
                    confidence=0.80,
                    reason=f"Target at HTF support {tp_price:.6f}"
                )
        
        return None
    
    def _get_liquidity_tp(self, direction, current_price, liquidity_clusters, risk):
        """Take profit at liquidity cluster"""
        if not liquidity_clusters:
            return None
        
        if direction == 'BUY':
            clusters = [c for c in liquidity_clusters if c.get('price', 0) > current_price]
            if clusters:
                clusters.sort(key=lambda x: x.get('price', 0))
                tp_price = clusters[0].get('price', 0)
                reward = tp_price - current_price
                rr = reward / risk if risk > 0 else 0
                return TakeProfitResult(
                    price=round(tp_price, 6),
                    method="Liquidity Cluster",
                    distance_pct=round(reward / current_price, 4),
                    atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                    risk_reward=round(rr, 2),
                    confidence=0.75,
                    reason=f"Target at liquidity cluster {tp_price:.6f}"
                )
        
        else:  # SELL
            clusters = [c for c in liquidity_clusters if c.get('price', 0) < current_price]
            if clusters:
                clusters.sort(key=lambda x: x.get('price', 0), reverse=True)
                tp_price = clusters[0].get('price', 0)
                reward = current_price - tp_price
                rr = reward / risk if risk > 0 else 0
                return TakeProfitResult(
                    price=round(tp_price, 6),
                    method="Liquidity Cluster",
                    distance_pct=round(reward / current_price, 4),
                    atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                    risk_reward=round(rr, 2),
                    confidence=0.75,
                    reason=f"Target at liquidity cluster {tp_price:.6f}"
                )
        
        return None
    
    def _get_fib_tp(self, direction, entry_price, stop_loss, fib_extensions, risk):
        """Take profit at Fibonacci extension"""
        if not fib_extensions:
            return None
        
        if direction == 'BUY':
            for level_name in ['1.272', '1.618', '2.0', '2.618']:
                tp_price = fib_extensions.get(level_name, 0)
                if tp_price > entry_price:
                    reward = tp_price - entry_price
                    rr = reward / risk if risk > 0 else 0
                    if rr >= self.min_rr:
                        return TakeProfitResult(
                            price=round(tp_price, 6),
                            method=f"Fib Extension {level_name}",
                            distance_pct=round(reward / entry_price, 4),
                            atr_multiple=round(reward / (entry_price * 0.02) if entry_price > 0 else 1, 2),
                            risk_reward=round(rr, 2),
                            confidence=0.70,
                            reason=f"Target at Fibonacci {level_name} extension"
                        )
        
        else:  # SELL
            for level_name in ['0.618', '0.786', '0.886']:
                tp_price = fib_extensions.get(level_name, 0)
                if tp_price < entry_price and tp_price > 0:
                    reward = entry_price - tp_price
                    rr = reward / risk if risk > 0 else 0
                    if rr >= self.min_rr:
                        return TakeProfitResult(
                            price=round(tp_price, 6),
                            method=f"Fib Retracement {level_name}",
                            distance_pct=round(reward / entry_price, 4),
                            atr_multiple=round(reward / (entry_price * 0.02) if entry_price > 0 else 1, 2),
                            risk_reward=round(rr, 2),
                            confidence=0.70,
                            reason=f"Target at Fibonacci {level_name} retracement"
                        )
        
        return None
    
    def _get_structure_tp(self, direction, current_price, swing_points, risk):
        """Take profit at market structure target"""
        if not swing_points:
            return None
        
        if direction == 'BUY':
            # Find next swing high
            swing_highs = [s for s in swing_points 
                          if s.get('type') in ['HH', 'LH'] 
                          and s.get('price', 0) > current_price]
            if swing_highs:
                swing_highs.sort(key=lambda x: x.get('price', 0))
                tp_price = swing_highs[0].get('price', 0)
                reward = tp_price - current_price
                rr = reward / risk if risk > 0 else 0
                return TakeProfitResult(
                    price=round(tp_price, 6),
                    method="Next Swing High",
                    distance_pct=round(reward / current_price, 4),
                    atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                    risk_reward=round(rr, 2),
                    confidence=0.65,
                    reason=f"Target at next swing high {tp_price:.6f}"
                )
        
        else:  # SELL
            swing_lows = [s for s in swing_points 
                         if s.get('type') in ['HL', 'LL'] 
                         and s.get('price', 0) < current_price]
            if swing_lows:
                swing_lows.sort(key=lambda x: x.get('price', 0), reverse=True)
                tp_price = swing_lows[0].get('price', 0)
                reward = current_price - tp_price
                rr = reward / risk if risk > 0 else 0
                return TakeProfitResult(
                    price=round(tp_price, 6),
                    method="Next Swing Low",
                    distance_pct=round(reward / current_price, 4),
                    atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                    risk_reward=round(rr, 2),
                    confidence=0.65,
                    reason=f"Target at next swing low {tp_price:.6f}"
                )
        
        return None
    
    def _get_indicator_tp(self, direction, current_price, df, risk):
        """Take profit at opposite Bollinger/Keltner band"""
        
        # Try Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns and not pd.isna(df['bb_upper'].iloc[-1]):
            bb_upper = float(df['bb_upper'].iloc[-1])
            bb_lower = float(df['bb_lower'].iloc[-1])
            
            if direction == 'BUY' and bb_upper > current_price:
                reward = bb_upper - current_price
                rr = reward / risk if risk > 0 else 0
                if rr >= self.min_rr:
                    return TakeProfitResult(
                        price=round(bb_upper, 6),
                        method="Bollinger Upper Band",
                        distance_pct=round(reward / current_price, 4),
                        atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                        risk_reward=round(rr, 2),
                        confidence=0.60,
                        reason=f"Target at Bollinger upper band"
                    )
            
            elif direction == 'SELL' and bb_lower < current_price:
                reward = current_price - bb_lower
                rr = reward / risk if risk > 0 else 0
                if rr >= self.min_rr:
                    return TakeProfitResult(
                        price=round(bb_lower, 6),
                        method="Bollinger Lower Band",
                        distance_pct=round(reward / current_price, 4),
                        atr_multiple=round(reward / (current_price * 0.02) if current_price > 0 else 1, 2),
                        risk_reward=round(rr, 2),
                        confidence=0.60,
                        reason=f"Target at Bollinger lower band"
                    )
        
        return None
    
    def _get_rr_based_tp(self, direction, entry_price, stop_loss, risk, confidence, win_rate):
        """Dynamic RR-based take profit"""
        # Calculate optimal RR based on confidence and win rate
        if confidence > 0.85:
            optimal_rr = 3.0
        elif confidence > 0.75:
            optimal_rr = 2.5
        elif confidence > 0.65:
            optimal_rr = 2.0
        else:
            optimal_rr = 1.5
        
        # Adjust by win rate
        if win_rate > 0.65:
            optimal_rr *= 1.1
        elif win_rate < 0.45:
            optimal_rr *= 0.9
        
        optimal_rr = max(self.min_rr, min(4.0, optimal_rr))
        
        if direction == 'BUY':
            tp_price = entry_price + (risk * optimal_rr)
            reward = tp_price - entry_price
            return TakeProfitResult(
                price=round(tp_price, 6),
                method=f"Dynamic RR {optimal_rr:.1f}:1",
                distance_pct=round(reward / entry_price, 4),
                atr_multiple=round(reward / (entry_price * 0.02) if entry_price > 0 else 1, 2),
                risk_reward=round(optimal_rr, 2),
                confidence=0.50,
                reason=f"Target based on {optimal_rr:.1f}:1 risk/reward"
            )
        else:
            tp_price = entry_price - (risk * optimal_rr)
            reward = entry_price - tp_price
            return TakeProfitResult(
                price=round(tp_price, 6),
                method=f"Dynamic RR {optimal_rr:.1f}:1",
                distance_pct=round(reward / entry_price, 4),
                atr_multiple=round(reward / (entry_price * 0.02) if entry_price > 0 else 1, 2),
                risk_reward=round(optimal_rr, 2),
                confidence=0.50,
                reason=f"Target based on {optimal_rr:.1f}:1 risk/reward"
            )