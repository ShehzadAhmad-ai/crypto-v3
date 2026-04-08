# mtf_pullback.py - Multi-Timeframe Pullback Detector
"""
Multi-Timeframe Pullback Detector
Detects when price is pulling back to higher timeframe support/resistance
Used to identify optimal entry points in trending markets
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Try to import config, with fallback
try:
    from config import Config
except ImportError:
    class Config:
        pass

from logger import log


@dataclass
class PullbackResult:
    """Complete pullback analysis result"""
    # Core result
    is_pullback: bool
    direction: str  # BULLISH (pullback to support) or BEARISH (pullback to resistance)
    confidence: float
    
    # Pullback details
    pullback_to_tf: str  # Which timeframe support/resistance
    pullback_level: float  # Price level of support/resistance
    pullback_depth: float  # Distance from current price to level (percentage)
    current_price: float
    
    # Health metrics
    volume_decreasing: bool  # Healthy pullback = volume decreasing
    has_reversal_pattern: bool  # Bullish pattern at support
    rsi_reset: bool  # RSI pulled back from overbought/oversold
    
    # Entry zone
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    take_profit: float
    
    # Score components
    score: float
    
    # Human-readable
    summary: str
    reasons: List[str]


class MTFPullbackDetector:
    """
    Detects pullbacks to higher timeframe support/resistance
    Identifies optimal entry zones for trend continuation trades
    """
    
    def __init__(self):
        """Initialize pullback detector with config settings"""
        # Load settings from config
        self.max_pullback_depth = getattr(Config, 'MTF_PULLBACK_DEPTH_MAX', 0.05)  # 5%
        self.volume_threshold = getattr(Config, 'MTF_PULLBACK_VOLUME_THRESHOLD', 0.8)  # 80% of average
        self.rsi_reset_threshold = getattr(Config, 'MTF_PULLBACK_RSI_THRESHOLD', 10)  # RSI moved 10 points
        
        log.debug("MTFPullbackDetector initialized")
    
    def _get_avg_volume(self, df: pd.DataFrame, period: int = 20) -> float:
        """Get average volume over specified period"""
        try:
            if df is None or df.empty:
                return 1.0
            if len(df) >= period:
                return float(df['volume'].iloc[-period:].mean())
            return float(df['volume'].mean())
        except Exception:
            return 1.0
    
    def detect_pullback(
        self,
        primary_df: pd.DataFrame,
        primary_price: float,
        primary_volume: float,
        primary_rsi: float,
        timeframe_results: Dict[str, Dict[str, Any]],
        direction: str  # Expected direction from primary analysis
    ) -> PullbackResult:
        """
        Detect if price is pulling back to higher timeframe support/resistance
        
        Args:
            primary_df: Primary timeframe DataFrame
            primary_price: Current price on primary timeframe
            primary_volume: Current volume on primary timeframe
            primary_rsi: Current RSI on primary timeframe
            timeframe_results: Results from analyzing higher timeframes
            direction: Expected direction (BUY or SELL)
        
        Returns:
            PullbackResult with complete analysis
        """
        if direction == 'BUY':
            return self._detect_bullish_pullback(
                primary_df, primary_price, primary_volume, primary_rsi,
                timeframe_results
            )
        else:
            return self._detect_bearish_pullback(
                primary_df, primary_price, primary_volume, primary_rsi,
                timeframe_results
            )
    
    def _detect_bullish_pullback(
        self,
        primary_df: pd.DataFrame,
        primary_price: float,
        primary_volume: float,
        primary_rsi: float,
        timeframe_results: Dict[str, Dict]
    ) -> PullbackResult:
        """
        Detect bullish pullback to higher timeframe support
        """
        reasons = []
        score = 0.0
        pullback_to_tf = None
        pullback_level = None
        pullback_depth = None
        
        # ===== STEP 1: Find nearest higher timeframe support =====
        supports = []
        for tf, result in timeframe_results.items():
            # Try different possible keys for support level
            support = result.get('support') or result.get('closest_support')
            if support and support < primary_price:
                # Calculate distance
                distance = (primary_price - support) / primary_price
                if distance < self.max_pullback_depth:
                    supports.append({
                        'timeframe': tf,
                        'level': support,
                        'distance': distance,
                        'strength': self._get_level_strength(result)
                    })
        
        if not supports:
            return self._no_pullback_result('BUY', primary_price)
        
        # Find closest support
        closest = min(supports, key=lambda x: x['distance'])
        pullback_to_tf = closest['timeframe']
        pullback_level = closest['level']
        pullback_depth = closest['distance']
        
        reasons.append(f"Price at {pullback_to_tf} support (${pullback_level:.4f})")
        score += 0.3
        
        # ===== STEP 2: Check volume on pullback =====
        avg_volume = self._get_avg_volume(primary_df)
        volume_ratio = primary_volume / avg_volume if avg_volume > 0 else 1.0
        volume_decreasing = volume_ratio < self.volume_threshold
        
        if volume_decreasing:
            reasons.append(f"Volume decreasing ({volume_ratio:.2f}x) - healthy pullback")
            score += 0.2
        else:
            reasons.append(f"Volume increasing ({volume_ratio:.2f}x) - caution")
        
        # ===== STEP 3: Check RSI reset =====
        # RSI should be pulling back from overbought (70) to neutral (50-60)
        rsi_reset = 40 < primary_rsi < 60
        if rsi_reset:
            reasons.append(f"RSI reset to {primary_rsi:.1f} - healthy")
            score += 0.15
        else:
            reasons.append(f"RSI at {primary_rsi:.1f} - not reset yet")
        
        # ===== STEP 4: Check for reversal patterns =====
        has_reversal_pattern = self._check_bullish_reversal_pattern(primary_df)
        if has_reversal_pattern:
            reasons.append("Bullish reversal pattern detected at support")
            score += 0.25
        
        # ===== STEP 5: Check higher timeframe trend =====
        htf_trend_bullish = self._check_htf_trend(timeframe_results, 'BULLISH')
        if htf_trend_bullish:
            reasons.append("Higher timeframes are bullish")
            score += 0.1
        
        # ===== STEP 6: Calculate entry zone =====
        entry_zone_low = pullback_level * 0.998
        entry_zone_high = pullback_level
        
        # Stop loss below support
        stop_loss = pullback_level * 0.99
        
        # Take profit at next resistance
        take_profit = self._get_next_resistance(primary_price, timeframe_results)
        
        # ===== STEP 7: Calculate final confidence =====
        confidence = min(0.95, max(0.1, score))
        is_pullback = score >= 0.5
        
        # ===== STEP 8: Build summary =====
        if is_pullback:
            summary = f"Bullish pullback to {pullback_to_tf} support at ${pullback_level:.4f} ({pullback_depth:.1%} depth). {len(reasons)} confirmations."
        else:
            summary = f"Weak pullback signal - score {score:.1%}"
        
        return PullbackResult(
            is_pullback=is_pullback,
            direction='BULLISH',
            confidence=confidence,
            pullback_to_tf=pullback_to_tf,
            pullback_level=pullback_level,
            pullback_depth=round(pullback_depth, 4),
            current_price=primary_price,
            volume_decreasing=volume_decreasing,
            has_reversal_pattern=has_reversal_pattern,
            rsi_reset=rsi_reset,
            entry_zone_low=round(entry_zone_low, 6),
            entry_zone_high=round(entry_zone_high, 6),
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            score=round(score, 3),
            summary=summary,
            reasons=reasons
        )
    
    def _detect_bearish_pullback(
        self,
        primary_df: pd.DataFrame,
        primary_price: float,
        primary_volume: float,
        primary_rsi: float,
        timeframe_results: Dict[str, Dict]
    ) -> PullbackResult:
        """
        Detect bearish pullback to higher timeframe resistance
        """
        reasons = []
        score = 0.0
        pullback_to_tf = None
        pullback_level = None
        pullback_depth = None
        
        # ===== STEP 1: Find nearest higher timeframe resistance =====
        resistances = []
        for tf, result in timeframe_results.items():
            # Try different possible keys for resistance level
            resistance = result.get('resistance') or result.get('closest_resistance')
            if resistance and resistance > primary_price:
                # Calculate distance
                distance = (resistance - primary_price) / primary_price
                if distance < self.max_pullback_depth:
                    resistances.append({
                        'timeframe': tf,
                        'level': resistance,
                        'distance': distance,
                        'strength': self._get_level_strength(result)
                    })
        
        if not resistances:
            return self._no_pullback_result('SELL', primary_price)
        
        # Find closest resistance
        closest = min(resistances, key=lambda x: x['distance'])
        pullback_to_tf = closest['timeframe']
        pullback_level = closest['level']
        pullback_depth = closest['distance']
        
        reasons.append(f"Price at {pullback_to_tf} resistance (${pullback_level:.4f})")
        score += 0.3
        
        # ===== STEP 2: Check volume on pullback =====
        avg_volume = self._get_avg_volume(primary_df)
        volume_ratio = primary_volume / avg_volume if avg_volume > 0 else 1.0
        volume_decreasing = volume_ratio < self.volume_threshold
        
        if volume_decreasing:
            reasons.append(f"Volume decreasing ({volume_ratio:.2f}x) - healthy pullback")
            score += 0.2
        else:
            reasons.append(f"Volume increasing ({volume_ratio:.2f}x) - caution")
        
        # ===== STEP 3: Check RSI reset =====
        # RSI should be pulling back from oversold (30) to neutral (40-50)
        rsi_reset = 40 < primary_rsi < 60
        if rsi_reset:
            reasons.append(f"RSI reset to {primary_rsi:.1f} - healthy")
            score += 0.15
        else:
            reasons.append(f"RSI at {primary_rsi:.1f} - not reset yet")
        
        # ===== STEP 4: Check for reversal patterns =====
        has_reversal_pattern = self._check_bearish_reversal_pattern(primary_df)
        if has_reversal_pattern:
            reasons.append("Bearish reversal pattern detected at resistance")
            score += 0.25
        
        # ===== STEP 5: Check higher timeframe trend =====
        htf_trend_bearish = self._check_htf_trend(timeframe_results, 'BEARISH')
        if htf_trend_bearish:
            reasons.append("Higher timeframes are bearish")
            score += 0.1
        
        # ===== STEP 6: Calculate entry zone =====
        entry_zone_low = pullback_level
        entry_zone_high = pullback_level * 1.002
        
        # Stop loss above resistance
        stop_loss = pullback_level * 1.01
        
        # Take profit at next support
        take_profit = self._get_next_support(primary_price, timeframe_results)
        
        # ===== STEP 7: Calculate final confidence =====
        confidence = min(0.95, max(0.1, score))
        is_pullback = score >= 0.5
        
        # ===== STEP 8: Build summary =====
        if is_pullback:
            summary = f"Bearish pullback to {pullback_to_tf} resistance at ${pullback_level:.4f} ({pullback_depth:.1%} depth). {len(reasons)} confirmations."
        else:
            summary = f"Weak pullback signal - score {score:.1%}"
        
        return PullbackResult(
            is_pullback=is_pullback,
            direction='BEARISH',
            confidence=confidence,
            pullback_to_tf=pullback_to_tf,
            pullback_level=pullback_level,
            pullback_depth=round(pullback_depth, 4),
            current_price=primary_price,
            volume_decreasing=volume_decreasing,
            has_reversal_pattern=has_reversal_pattern,
            rsi_reset=rsi_reset,
            entry_zone_low=round(entry_zone_low, 6),
            entry_zone_high=round(entry_zone_high, 6),
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            score=round(score, 3),
            summary=summary,
            reasons=reasons
        )
    
    def _get_level_strength(self, result: Dict) -> float:
        """Calculate strength of a support/resistance level"""
        strength = 0.5
        
        # Check if level is from higher timeframe
        tf = result.get('timeframe', '')
        if tf in ['4h', '1d', '1w']:
            strength += 0.2
        
        # Check volume at level
        volume = result.get('volume', {})
        if volume.get('has_spike'):
            strength += 0.2
        
        # Check if level has been tested multiple times
        supports = result.get('supports', [])
        resistances = result.get('resistances', [])
        if len(supports) > 2 or len(resistances) > 2:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _check_bullish_reversal_pattern(self, df: pd.DataFrame) -> bool:
        """Check for bullish reversal patterns on primary timeframe"""
        try:
            if df is None or df.empty or len(df) < 3:
                return False
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and
                last['close'] > last['open'] and
                last['close'] > prev['open'] and
                last['open'] < prev['close']):
                return True
            
            # Hammer
            body = abs(last['close'] - last['open'])
            lower_wick = min(last['open'], last['close']) - last['low']
            if lower_wick > body * 2 and last['close'] > last['open']:
                return True
            
            # Morning Star
            if (prev2['close'] < prev2['open'] and
                abs(prev['close'] - prev['open']) < abs(prev2['close'] - prev2['open']) * 0.5 and
                last['close'] > last['open'] and
                last['close'] > prev2['close']):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _check_bearish_reversal_pattern(self, df: pd.DataFrame) -> bool:
        """Check for bearish reversal patterns on primary timeframe"""
        try:
            if df is None or df.empty or len(df) < 3:
                return False
            
            last = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            # Bearish Engulfing
            if (prev['close'] > prev['open'] and
                last['close'] < last['open'] and
                last['close'] < prev['open'] and
                last['open'] > prev['close']):
                return True
            
            # Shooting Star
            body = abs(last['close'] - last['open'])
            upper_wick = last['high'] - max(last['open'], last['close'])
            if upper_wick > body * 2 and last['close'] < last['open']:
                return True
            
            # Evening Star
            if (prev2['close'] > prev2['open'] and
                abs(prev['close'] - prev['open']) < abs(prev2['close'] - prev2['open']) * 0.5 and
                last['close'] < last['open'] and
                last['close'] < prev2['close']):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _check_htf_trend(self, timeframe_results: Dict, direction: str) -> bool:
        """Check if higher timeframes align with direction"""
        bullish_count = 0
        bearish_count = 0
        total = 0
        
        for tf, result in timeframe_results.items():
            tf_direction = result.get('direction', 'NEUTRAL')
            if tf_direction == 'BULLISH' or tf_direction == 'BUY':
                bullish_count += 1
            elif tf_direction == 'BEARISH' or tf_direction == 'SELL':
                bearish_count += 1
            total += 1
        
        if total == 0:
            return False
        
        if direction == 'BULLISH':
            return bullish_count > bearish_count
        else:
            return bearish_count > bullish_count
    
    def _get_next_resistance(self, current_price: float, timeframe_results: Dict) -> float:
        """Get the next resistance level above current price"""
        resistances = []
        for tf, result in timeframe_results.items():
            resistance = result.get('resistance') or result.get('closest_resistance')
            if resistance and resistance > current_price:
                resistances.append(resistance)
            # Also add top resistances list
            for r in result.get('resistances', []):
                if r > current_price:
                    resistances.append(r)
        
        if resistances:
            return min(resistances) * 0.998  # Slightly below resistance
        else:
            # Default: 4% target
            return current_price * 1.04
    
    def _get_next_support(self, current_price: float, timeframe_results: Dict) -> float:
        """Get the next support level below current price"""
        supports = []
        for tf, result in timeframe_results.items():
            support = result.get('support') or result.get('closest_support')
            if support and support < current_price:
                supports.append(support)
            for s in result.get('supports', []):
                if s < current_price:
                    supports.append(s)
        
        if supports:
            return max(supports) * 1.002  # Slightly above support
        else:
            # Default: 4% target
            return current_price * 0.96
    
    def _no_pullback_result(self, direction: str, current_price: float) -> PullbackResult:
        """Return result when no pullback detected"""
        if direction == 'BUY':
            entry_zone_low = current_price - (current_price * 0.005)
            entry_zone_high = current_price
            stop_loss = current_price - (current_price * 0.01)
            take_profit = current_price + (current_price * 0.03)
        else:
            entry_zone_low = current_price
            entry_zone_high = current_price + (current_price * 0.005)
            stop_loss = current_price + (current_price * 0.01)
            take_profit = current_price - (current_price * 0.03)
        
        return PullbackResult(
            is_pullback=False,
            direction=direction,
            confidence=0.3,
            pullback_to_tf='NONE',
            pullback_level=0,
            pullback_depth=0,
            current_price=current_price,
            volume_decreasing=False,
            has_reversal_pattern=False,
            rsi_reset=False,
            entry_zone_low=round(entry_zone_low, 6),
            entry_zone_high=round(entry_zone_high, 6),
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            score=0,
            summary="No pullback detected",
            reasons=["Price not near higher timeframe support/resistance"]
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def detect_pullback(
    primary_df: pd.DataFrame,
    primary_price: float,
    primary_volume: float,
    primary_rsi: float,
    timeframe_results: Dict[str, Dict],
    direction: str
) -> PullbackResult:
    """
    Convenience function to detect pullback
    
    Args:
        primary_df: Primary timeframe DataFrame
        primary_price: Current price
        primary_volume: Current volume
        primary_rsi: Current RSI
        timeframe_results: Results from higher timeframe analysis
        direction: Expected direction (BUY or SELL)
    
    Returns:
        PullbackResult with complete analysis
    """
    detector = MTFPullbackDetector()
    return detector.detect_pullback(
        primary_df, primary_price, primary_volume, primary_rsi,
        timeframe_results, direction
    )