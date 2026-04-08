"""
pattern_intelligence.py - Enhanced Context Scoring for Pattern V4

Adds to each pattern:
- Real context analysis (trend, location, volatility, momentum)
- SMC concepts (liquidity sweeps, BOS, order blocks)
- Dynamic weights based on market regime
- Volume profile strength

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .pattern_core import (
    PatternV4, PatternDirection, ContextScore, LiquidityAnalysis, 
    TrapAnalysis, TrapType, PatternStage
)
from .pattern_config import CONFIG, get_dynamic_context_weights, get_pattern_regime_validity


# ============================================================================
# CONTEXT ANALYZER - Enhanced with SMC Concepts
# ============================================================================

class ContextAnalyzerV4:
    """
    Analyzes market context for each pattern with SMC concepts:
    - Trend alignment (pattern vs market trend)
    - Support/Resistance reaction (key levels)
    - Volume confirmation (volume supports move)
    - Volatility condition (optimal range)
    - Liquidity sweep presence (SMC concept)
    - Volume profile strength (POC analysis)
    
    Uses DYNAMIC WEIGHTS based on market regime.
    """
    
    def __init__(self):
        self.config = CONFIG
        self.dynamic_weights = CONFIG.dynamic_context_weights
    
    def analyze(self, pattern: Dict, df: pd.DataFrame, 
                swings: List, liquidity_data: Dict = None,
                regime: str = "NEUTRAL") -> ContextScore:
        """
        Calculate context score for a pattern.
        Returns ContextScore object with all components.
        """
        
        direction = pattern.get('direction', 'NEUTRAL')
        
        # Get dynamic weights for this regime
        weights = get_dynamic_context_weights(regime)
        
        # Calculate each component
        trend_score = self._calculate_trend_alignment(direction, swings, df)
        sr_score = self._calculate_support_resistance(df, pattern, liquidity_data)
        volume_score = self._calculate_volume_confirmation(pattern, df)
        volatility_score = self._calculate_volatility_condition(df)
        liquidity_score = self._calculate_liquidity_sweep_score(liquidity_data, direction)
        volume_profile_score = self._calculate_volume_profile_strength(df, pattern)
        
        # Calculate weighted total
        total = (
            weights.get('trend_alignment', 0.30) * trend_score +
            weights.get('support_resistance', 0.25) * sr_score +
            weights.get('volume_confirmation', 0.20) * volume_score +
            weights.get('volatility_condition', 0.15) * volatility_score +
            weights.get('liquidity_sweep', 0.10) * liquidity_score +
            weights.get('volume_profile', 0.00) * volume_profile_score
        )
        
        # Create ContextScore object
        context = ContextScore(
            total=total,
            trend_alignment=trend_score,
            support_resistance=sr_score,
            volume_confirmation=volume_score,
            volatility_condition=volatility_score,
            liquidity_sweep=liquidity_score,
            volume_profile=volume_profile_score,
            applied_weights=weights,
            applied_regime=regime
        )
        
        # Set SMC flags from liquidity data
        if liquidity_data:
            context.sweep_detected = liquidity_data.get('has_down_sweep', False) or liquidity_data.get('has_up_sweep', False)
            context.bos_detected = liquidity_data.get('bos_detected', False)
            context.order_block_nearby = liquidity_data.get('order_block_nearby', False)
            
            # Determine support/resistance based on sweeps
            if direction == 'BUY' and liquidity_data.get('has_down_sweep', False):
                context.at_support = True
            if direction == 'SELL' and liquidity_data.get('has_up_sweep', False):
                context.at_resistance = True
        
        return context
    
    def _calculate_trend_alignment(self, direction: str, swings: List, 
                                    df: pd.DataFrame) -> float:
        """
        Calculate trend alignment score (0-1).
        Higher score = pattern aligns with trend.
        """
        if len(swings) < 6:
            if len(swings) >= 3:
                # Use available swings (but still return default for now)
                return 0.5
            else:
                return 0.5
        
        # Analyze recent swing structure
        recent = swings[-6:]
        bullish_swings = sum(1 for s in recent if s.type in ['HH', 'HL'])
        bearish_swings = sum(1 for s in recent if s.type in ['LH', 'LL'])
        
        total_swings = bullish_swings + bearish_swings
        if total_swings == 0:
            return 0.5
        
        # Calculate trend strength and direction
        if bullish_swings > bearish_swings:
            trend_direction = 'BULLISH'
            trend_strength = bullish_swings / total_swings
        else:
            trend_direction = 'BEARISH'
            trend_strength = bearish_swings / total_swings
        
        # Calculate alignment
        if direction == 'BUY':
            if trend_direction == 'BULLISH':
                return 0.5 + (trend_strength * 0.5)  # 0.5 to 1.0
            elif trend_direction == 'BEARISH':
                return 0.5 - (trend_strength * 0.3)  # 0.2 to 0.5
            else:
                return 0.5
        else:  # SELL
            if trend_direction == 'BEARISH':
                return 0.5 + (trend_strength * 0.5)
            elif trend_direction == 'BULLISH':
                return 0.5 - (trend_strength * 0.3)
            else:
                return 0.5
    
    def _calculate_support_resistance(self, df: pd.DataFrame, pattern: Dict,
                                       liquidity_data: Dict) -> float:
        """
        Calculate support/resistance reaction score.
        Higher score = price at favorable zone.
        """
        current_price = float(df['close'].iloc[-1])
        pattern_direction = pattern.get('direction', 'NEUTRAL')
        
        # Get key levels from liquidity data
        equal_highs = []
        equal_lows = []
        if liquidity_data:
            equal_highs = liquidity_data.get('equal_highs', [])
            equal_lows = liquidity_data.get('equal_lows', [])
        
        # Check proximity to key levels
        near_resistance = False
        near_support = False
        
        for level in equal_highs:
            if isinstance(level, dict):
                level_price = level.get('price', 0)
            else:
                level_price = level
            if abs(current_price - level_price) / current_price < 0.01:
                near_resistance = True
                break
        
        for level in equal_lows:
            if isinstance(level, dict):
                level_price = level.get('price', 0)
            else:
                level_price = level
            if abs(current_price - level_price) / current_price < 0.01:
                near_support = True
                break
        
        # Score based on pattern direction
        if pattern_direction == 'BUY':
            # BUY wants support (discount zone)
            if near_support:
                return 0.9
            elif near_resistance:
                return 0.3
            else:
                return 0.5
        else:  # SELL
            # SELL wants resistance (premium zone)
            if near_resistance:
                return 0.9
            elif near_support:
                return 0.3
            else:
                return 0.5
    
    def _calculate_volume_confirmation(self, pattern: Dict, df: pd.DataFrame) -> float:
        """
        Calculate volume confirmation score.
        Higher score = volume supports the pattern.
        """
        start_idx = pattern.get('start_idx', 0)
        end_idx = pattern.get('end_idx', len(df) - 1)
        
        if start_idx >= end_idx or start_idx < 5:
            return 0.5
        
        # Volume during pattern formation
        formation_vol = df['volume'].iloc[start_idx:end_idx].mean()
        prior_vol = df['volume'].iloc[max(0, start_idx-20):start_idx].mean()
        
        # Volume on breakout (if available)
        if end_idx + 3 < len(df):
            breakout_vol = df['volume'].iloc[end_idx:end_idx+3].mean()
            avg_vol = df['volume'].iloc[max(0, end_idx-20):end_idx].mean()
            breakout_ratio = breakout_vol / max(avg_vol, 1)
        else:
            breakout_ratio = 1.0
        
        # Contraction score (formation < prior = good)
        if prior_vol > 0:
            contraction = max(0.0, 1.0 - (formation_vol / prior_vol))
        else:
            contraction = 0.5
        
        # Expansion score (breakout > avg = good)
        expansion = min(1.0, max(0.0, (breakout_ratio - 1.0) / 1.0))
        
        return (contraction * 0.5 + expansion * 0.5)
    
    def _calculate_volatility_condition(self, df: pd.DataFrame) -> float:
        """
        Calculate volatility condition score.
        Optimal volatility: 1-2% ATR.
        """
        if 'atr' not in df.columns:
            return 0.5
        
        current_price = float(df['close'].iloc[-1])
        atr = float(df['atr'].iloc[-1])
        atr_pct = atr / current_price if current_price > 0 else 0.02
        
        # Optimal range: 1-2%
        if 0.01 <= atr_pct <= 0.02:
            return 0.9
        elif 0.005 <= atr_pct <= 0.03:
            return 0.7
        elif atr_pct < 0.005:
            return 0.4  # Too low, no movement
        elif atr_pct > 0.04:
            return 0.3  # Too high, risky
        else:
            return 0.5
    
    def _calculate_liquidity_sweep_score(self, liquidity_data: Dict, 
                                          direction: str) -> float:
        """
        Calculate liquidity sweep score (SMC concept).
        Higher score = sweep confirms pattern direction.
        """
        if not liquidity_data:
            return 0.5
        
        score = 0.5
        
        # Sweep confirmation
        if direction == 'BUY' and liquidity_data.get('has_down_sweep', False):
            score += 0.3
        elif direction == 'SELL' and liquidity_data.get('has_up_sweep', False):
            score += 0.3
        
        # Stop hunt confirmation
        stop_hunt = liquidity_data.get('stop_hunt_probability', 0)
        if stop_hunt > 0.6:
            score += 0.2
        
        # BOS (Break of Structure) confirmation
        if liquidity_data.get('bos_detected', False):
            score += 0.15
        
        # Order block nearby
        if liquidity_data.get('order_block_nearby', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_volume_profile_strength(self, df: pd.DataFrame, 
                                            pattern: Dict) -> float:
        """
        Calculate volume profile strength at key levels.
        Higher score = key levels have high volume concentration.
        """
        neckline = pattern.get('neckline')
        if neckline is None:
            return 0.5
        
        lookback = getattr(CONFIG, 'volume_profile_config', {}).get('lookback_bars', 100)
        if len(df) < lookback:
            return 0.5
        
        recent = df.iloc[-lookback:]
        
        # Create volume profile bins
        price_min = recent['low'].min()
        price_max = recent['high'].max()
        bins = np.linspace(price_min, price_max, 20)
        
        volume_by_price = {}
        for i in range(len(bins) - 1):
            mask = (recent['close'] >= bins[i]) & (recent['close'] < bins[i+1])
            if mask.any():
                volume = recent.loc[mask, 'volume'].sum()
                volume_by_price[(bins[i] + bins[i+1]) / 2] = volume
        
        if not volume_by_price:
            return 0.5
        
        # Find POC (highest volume node)
        poc_price = max(volume_by_price, key=volume_by_price.get)
        
        # Check if neckline is near POC
        distance_pct = abs(neckline - poc_price) / neckline
        
        if distance_pct < 0.005:  # Within 0.5%
            return 0.9
        elif distance_pct < 0.01:  # Within 1%
            return 0.7
        elif distance_pct < 0.02:  # Within 2%
            return 0.5
        else:
            return 0.3


# ============================================================================
# LIQUIDITY ANALYZER - Integration with existing liquidity module
# ============================================================================

class LiquidityAnalyzerV4:
    """
    Analyzes liquidity data for each pattern.
    Integrates with existing liquidity.py module.
    """
    
    def __init__(self):
        self.liquidity_module = None
        self._try_import_liquidity()
        self.enabled = CONFIG.features.get('enable_liquidity_analysis', True)
    
    def _try_import_liquidity(self):
        """Try to import existing liquidity module"""
        try:
            from .liquidity import LiquidityIntelligence
            self.liquidity_module = LiquidityIntelligence()
        except ImportError:
            pass
    
    def analyze(self, pattern: Dict, df: pd.DataFrame) -> LiquidityAnalysis:
        """
        Add liquidity analysis to pattern.
        Returns LiquidityAnalysis object.
        """
        if not self.enabled:
            return LiquidityAnalysis(score=0.5)
        
        try:
            if self.liquidity_module:
                summary = self.liquidity_module.get_liquidity_summary(df)
                return self._map_liquidity_data(summary)
            else:
                return self._analyze_liquidity_internal(pattern, df)
        except Exception as e:
            return LiquidityAnalysis(score=0.5)
    
    def _map_liquidity_data(self, summary: Dict) -> LiquidityAnalysis:
        """Map external liquidity data to LiquidityAnalysis"""
        return LiquidityAnalysis(
            score=summary.get('score', 0.5),
            net_bias=summary.get('net_bias', 0),
            direction=summary.get('direction', 'NEUTRAL'),
            has_down_sweep=summary.get('has_down_sweep', False),
            has_up_sweep=summary.get('has_up_sweep', False),
            stop_hunt_probability=summary.get('stop_hunt_probability', 0),
            inducement_count=summary.get('inducement_count', 0),
            sweeps=summary.get('sweeps', [])
        )
    
    def _analyze_liquidity_internal(self, pattern: Dict, df: pd.DataFrame) -> LiquidityAnalysis:
        """Internal liquidity analysis (fallback)"""
        if 'atr' not in df.columns:
            return LiquidityAnalysis(score=0.5)
        
        atr = df['atr'].iloc[-1]
        recent = df.iloc[-10:]
        
        sweeps = []
        has_down_sweep = False
        has_up_sweep = False
        
        for i in range(len(recent)):
            row = recent.iloc[i]
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            
            if upper_wick > atr * 1.5:
                sweeps.append({'type': 'UP_SWEEP', 'strength': min(1.0, upper_wick / atr)})
                has_up_sweep = True
            if lower_wick > atr * 1.5:
                sweeps.append({'type': 'DOWN_SWEEP', 'strength': min(1.0, lower_wick / atr)})
                has_down_sweep = True
        
        # Calculate score based on pattern direction
        direction = pattern.get('direction', 'NEUTRAL')
        score = 0.5
        
        if direction == 'BUY' and has_down_sweep:
            score += 0.2
        elif direction == 'SELL' and has_up_sweep:
            score += 0.2
        
        return LiquidityAnalysis(
            score=min(1.0, score),
            sweeps=sweeps,
            has_down_sweep=has_down_sweep,
            has_up_sweep=has_up_sweep
        )


# ============================================================================
# TRAP DETECTOR
# ============================================================================

class TrapDetectorV4:
    """
    Detects and classifies traps:
    - Bull trap (false breakout up)
    - Bear trap (false breakout down)
    - Liquidity grab (sweep that reverses)
    - Inducement (fake breakout)
    - Sweep failure (sweep that doesn't hold)
    """
    
    def __init__(self):
        self.enabled = CONFIG.features.get('enable_trap_detection', True)
        self.convert_traps = CONFIG.trap_config.get('convert_traps', True)
        self.min_strength = CONFIG.trap_config.get('min_trap_strength', 0.6)
    
    def detect(self, pattern: Dict, df: pd.DataFrame) -> TrapAnalysis:
        """
        Detect traps for a pattern.
        Returns TrapAnalysis object.
        """
        if not self.enabled:
            return TrapAnalysis(detected=False)
        
        direction = pattern.get('direction', 'NEUTRAL')
        current_price = float(df['close'].iloc[-1])
        recent = df.iloc[-15:]
        
        trap = TrapAnalysis()
        
        # Detect based on pattern direction
        if direction == 'BUY':
            self._detect_bull_trap(trap, recent, current_price)
        else:
            self._detect_bear_trap(trap, recent, current_price)
        
        # Also check for inducements
        if not trap.detected:
            self._detect_inducement(trap, df)
        
        # Check for sweep failures
        if not trap.detected:
            self._detect_sweep_failure(trap, df)
        
        # Convert trap if applicable
        if trap.detected and self.convert_traps and trap.strength >= self.min_strength:
            self._convert_trap(trap, direction)
        
        return trap
    
    def _detect_bull_trap(self, trap: TrapAnalysis, recent: pd.DataFrame, 
                          current_price: float):
        """Detect bull trap (false breakout up)"""
        recent_high = recent['high'].max()
        previous_high = recent.iloc[:-5]['high'].max() if len(recent) > 5 else recent_high
        
        if current_price < recent_high and recent_high > previous_high:
            trap.detected = True
            trap.trap_type = TrapType.BULL_TRAP
            trap.strength = min(1.0, (recent_high - previous_high) / previous_high * 10)
            trap.details = {'break_level': previous_high, 'fake_high': recent_high}
    
    def _detect_bear_trap(self, trap: TrapAnalysis, recent: pd.DataFrame,
                          current_price: float):
        """Detect bear trap (false breakout down)"""
        recent_low = recent['low'].min()
        previous_low = recent.iloc[:-5]['low'].min() if len(recent) > 5 else recent_low
        
        if current_price > recent_low and recent_low < previous_low:
            trap.detected = True
            trap.trap_type = TrapType.BEAR_TRAP
            trap.strength = min(1.0, (previous_low - recent_low) / previous_low * 10)
            trap.details = {'break_level': previous_low, 'fake_low': recent_low}
    
    def _detect_inducement(self, trap: TrapAnalysis, df: pd.DataFrame):
        """Detect inducement patterns"""
        if len(df) < 20:
            return
        
        recent = df.iloc[-20:]
        
        for i in range(2, len(recent) - 2):
            prev_high = recent['high'].iloc[:i].max()
            prev_low = recent['low'].iloc[:i].min()
            curr = recent.iloc[i]
            next1 = recent.iloc[i+1]
            
            # Upside inducement
            if curr['high'] > prev_high and next1['close'] < curr['close']:
                trap.detected = True
                trap.trap_type = TrapType.INDUCEMENT
                trap.strength = min(1.0, (curr['high'] - prev_high) / prev_high)
                trap.details['inducement_type'] = 'UPSIDE'
                return
            
            # Downside inducement
            if curr['low'] < prev_low and next1['close'] > curr['close']:
                trap.detected = True
                trap.trap_type = TrapType.INDUCEMENT
                trap.strength = min(1.0, (prev_low - curr['low']) / prev_low)
                trap.details['inducement_type'] = 'DOWNSIDE'
                return
    
    def _detect_sweep_failure(self, trap: TrapAnalysis, df: pd.DataFrame):
        """Detect sweep failures"""
        if len(df) < 15:
            return
        
        recent = df.iloc[-15:]
        
        for i in range(2, len(recent) - 2):
            prev_low = recent['low'].iloc[:i].min()
            prev_high = recent['high'].iloc[:i].max()
            curr = recent.iloc[i]
            next1 = recent.iloc[i+1]
            
            # Bullish sweep failure
            if curr['low'] < prev_low and curr['close'] > prev_low:
                if next1['close'] < next1['open']:
                    trap.detected = True
                    trap.trap_type = TrapType.SWEEP_FAILURE
                    trap.details['sweep_type'] = 'BULLISH_FAILURE'
                    return
            
            # Bearish sweep failure
            if curr['high'] > prev_high and curr['close'] < prev_high:
                if next1['close'] > next1['open']:
                    trap.detected = True
                    trap.trap_type = TrapType.SWEEP_FAILURE
                    trap.details['sweep_type'] = 'BEARISH_FAILURE'
                    return
    
    def _convert_trap(self, trap: TrapAnalysis, original_direction: str):
        """Convert trap to opposite signal"""
        trap.converted = True
        trap.confidence = 0.6 + trap.strength * 0.3
        
        if trap.trap_type == TrapType.BULL_TRAP:
            trap.convert_to = PatternDirection.SELL
        elif trap.trap_type == TrapType.BEAR_TRAP:
            trap.convert_to = PatternDirection.BUY
        elif trap.trap_type == TrapType.INDUCEMENT:
            if trap.details.get('inducement_type') == 'UPSIDE':
                trap.convert_to = PatternDirection.SELL
            else:
                trap.convert_to = PatternDirection.BUY
        elif trap.trap_type == TrapType.SWEEP_FAILURE:
            if trap.details.get('sweep_type') == 'BULLISH_FAILURE':
                trap.convert_to = PatternDirection.SELL
            else:
                trap.convert_to = PatternDirection.BUY


# ============================================================================
# QUALITY CALCULATOR - Pattern Quality Scoring
# ============================================================================

class QualityCalculatorV4:
    """
    Calculates pattern quality based on:
    - Structure clarity
    - Swing alternation
    - Volume pattern
    - Formation completeness
    """
    
    def __init__(self):
        pass
    
    def calculate(self, pattern: Dict, df: pd.DataFrame, 
                  swings: List) -> float:
        """
        Calculate overall pattern quality score (0-1).
        """
        # Get components from pattern similarity
        components = pattern.get('components', {})
        
        # Extract quality indicators
        structure_clarity = components.get('structure_clarity', 0.5)
        volume_pattern = components.get('volume_pattern', 0.5)
        neckline_quality = components.get('neckline_quality', 0.5)
        
        # Calculate swing health
        swing_health = self._calculate_swing_health(swings, pattern)
        
        # Calculate formation quality
        formation_quality = self._calculate_formation_quality(pattern, df)
        
        # Weighted average
        quality = (
            structure_clarity * 0.30 +
            volume_pattern * 0.25 +
            neckline_quality * 0.20 +
            swing_health * 0.15 +
            formation_quality * 0.10
        )
        
        return min(1.0, quality)
    
    def _calculate_swing_health(self, swings: List, pattern: Dict) -> float:
        """Calculate health of swings used in pattern"""
        start_idx = pattern.get('start_idx', 0)
        end_idx = pattern.get('end_idx', len(swings) - 1)
        
        relevant_swings = [s for s in swings if start_idx <= s.index <= end_idx]
        
        if len(relevant_swings) < 3:
            return 0.5
        
        # Check alternation
        types = [s.type for s in relevant_swings]
        alternations = sum(1 for i in range(1, len(types)) if types[i] != types[i-1])
        alternation_rate = alternations / (len(types) - 1)
        
        return min(1.0, alternation_rate * 1.2)
    
    def _calculate_formation_quality(self, pattern: Dict, df: pd.DataFrame) -> float:
        """Calculate formation quality based on bar count and noise"""
        start_idx = pattern.get('start_idx', 0)
        end_idx = pattern.get('end_idx', len(df) - 1)
        
        formation_bars = end_idx - start_idx
        
        if formation_bars < 5:
            return 0.3
        elif formation_bars < 10:
            return 0.6
        elif formation_bars < 30:
            return 0.8
        elif formation_bars < 50:
            return 0.7
        else:
            return 0.5


# ============================================================================
# MAIN INTELLIGENCE ENGINE
# ============================================================================

class PatternIntelligenceEngineV4:
    """
    Main intelligence engine that orchestrates all analyzers.
    Adds context, liquidity, traps, and quality to each pattern.
    """
    
    def __init__(self):
        self.context_analyzer = ContextAnalyzerV4()
        self.liquidity_analyzer = LiquidityAnalyzerV4()
        self.trap_detector = TrapDetectorV4()
        self.quality_calculator = QualityCalculatorV4()
    
    def analyze_pattern(self, pattern: Dict, df: pd.DataFrame,
                        swings: List = None,
                        regime_data: Dict = None,
                        liquidity_data: Dict = None) -> Dict:
        """
        Apply all intelligence layers to a single pattern.
        Returns enhanced pattern with context, liquidity, trap scores.
        """
        pattern = pattern.copy() if pattern else {}
        
        regime = regime_data.get('regime', 'NEUTRAL') if regime_data else 'NEUTRAL'
        
        # 1. Context analysis
        context = self.context_analyzer.analyze(
             pattern, df, swings, liquidity_data, regime
        )
        
        pattern['context_score'] = context.total
        pattern['context_components'] = context.to_dict()
        
        # 2. Liquidity analysis
        liquidity = self.liquidity_analyzer.analyze(pattern, df)
        pattern['liquidity'] = liquidity.to_dict()
        
        # 3. Trap detection
        trap = self.trap_detector.detect(pattern, df)
        pattern['trap'] = trap.to_dict()
        
        # 4. Quality calculation
        quality = self.quality_calculator.calculate(pattern, df, swings or [])
        
        # Apply trap conversion if needed
        if trap.converted and trap.convert_to:
            pattern['direction'] = trap.convert_to.value
            pattern['trap_converted'] = True
        
        # Apply regime validity multiplier to context score
        pattern_name = pattern.get('pattern_name', 'DEFAULT')
        regime_validity = get_pattern_regime_validity(pattern_name, regime)
        pattern['context_score'] = min(1.0, pattern['context_score'] * regime_validity)
        pattern['regime_validity'] = regime_validity
        
        return pattern
    
    def analyze_batch(self, patterns: List[Dict], df: pd.DataFrame,
                      
                      swings: List = None,
                      regime_data: Dict = None,
                      liquidity_data: Dict = None) -> List[Dict]:
        """
        Apply intelligence to multiple patterns.
        """
        enhanced = []
        
        for pattern in patterns:
            try:
                enhanced_pattern = self.analyze_pattern(
                    pattern, df, swings, regime_data, liquidity_data
                )
                enhanced.append(enhanced_pattern)
            except Exception as e:
                # Log the error so we can see what's failing
                import logging
                logging.warning(f"Intelligence failed for pattern {pattern.get('pattern_name', '?')}: {e}")
                # Pass the original pattern through unenhanced instead of dropping it
                enhanced.append(pattern)
        
        return enhanced


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ContextAnalyzerV4',
    'LiquidityAnalyzerV4',
    'TrapDetectorV4',
    'QualityCalculatorV4',
    'PatternIntelligenceEngineV4',
]




