# mtf_aggregator.py - Clean Multi-Timeframe Aggregator
"""
Multi-Timeframe Aggregator
Aggregates results from multiple higher timeframes into a single MTF score
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from config import Config
from logger import log


@dataclass
class MTFAggregatedResult:
    """Complete aggregated MTF result"""
    # Core result
    confirmed: bool
    score: float
    bias: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float
    
    # Alignment stats
    bullish_timeframes: int
    bearish_timeframes: int
    neutral_timeframes: int
    total_timeframes: int
    alignment_percentage: float
    alignment_quality: str  # STRONG_ALIGNMENT, ALIGNED, MIXED, CONFLICT
    
    # Weighted results
    weighted_score: float
    unweighted_score: float
    
    # Pullback analysis
    pullback: Dict[str, Any]
    
    # Key levels across timeframes
    key_levels: Dict[str, List[float]]
    
    # Conflict resolution
    conflict: Dict[str, Any]
    
    # Confidence boost/penalty
    confidence_boost: float
    
    # Human-readable
    story: str
    reasons: List[str]
    
    # Raw data
    timeframe_results: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MTFAggregator:
    """
    Aggregates results from multiple timeframe analyses
    
    Features:
    - Weighted aggregation by timeframe importance
    - Alignment analysis (how many timeframes agree)
    - Conflict resolution (higher timeframes win)
    - MTF score calculation
    - Human-readable MTF story
    """
    
    def __init__(self):
        # Load weights from config
        self.weights = getattr(Config, 'MTF_WEIGHTS', {
            '15m': 0.5,
            '30m': 0.7,
            '1h': 1.0,
            '4h': 1.5,
            '1d': 2.0,
            '1w': 2.5
        })
        
        # Load thresholds
        self.min_alignment = getattr(Config, 'MTF_MIN_ALIGNMENT', 0.60)
        self.min_score = getattr(Config, 'MTF_MIN_SCORE', 0.70)
        self.confidence_boost_max = getattr(Config, 'MTF_CONFIDENCE_BOOST_MAX', 0.15)
        self.confidence_penalty_max = getattr(Config, 'MTF_CONFIDENCE_PENALTY_MAX', -0.15)
        
        log.debug("MTFAggregator initialized")
    
    def aggregate(
        self,
        timeframe_results: Dict[str, Dict[str, Any]],
        primary_direction: Optional[str] = None
    ) -> MTFAggregatedResult:
        """
        Aggregate results from all timeframes
        
        Args:
            timeframe_results: Dictionary of timeframe -> analysis result
            primary_direction: Direction from primary timeframe (for comparison)
        
        Returns:
            MTFAggregatedResult with complete MTF analysis
        """
        if not timeframe_results:
            return self._empty_result("No timeframe data")
        
        # ===== STEP 1: COUNT DIRECTIONS =====
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        weighted_score_sum = 0
        total_weight = 0
        unweighted_score_sum = 0
        
        for tf, result in timeframe_results.items():
            direction = result.get('direction', 'NEUTRAL')
            score = result.get('confidence', 0.5)  # Use confidence from expert signals
            weight = self.weights.get(tf, 1.0)
            
            if direction == 'BULLISH' or direction == 'BUY':
                bullish_count += 1
            elif direction == 'BEARISH' or direction == 'SELL':
                bearish_count += 1
            else:
                neutral_count += 1
            
            weighted_score_sum += score * weight
            total_weight += weight
            unweighted_score_sum += score
        
        total_timeframes = len(timeframe_results)
        
        # ===== STEP 2: CALCULATE ALIGNMENT =====
        if total_timeframes > 0:
            alignment_percentage = max(bullish_count, bearish_count) / total_timeframes
        else:
            alignment_percentage = 0
        
        # Determine alignment quality
        if alignment_percentage >= 0.8:
            alignment_quality = 'STRONG_ALIGNMENT'
        elif alignment_percentage >= 0.6:
            alignment_quality = 'ALIGNED'
        elif alignment_percentage >= 0.4:
            alignment_quality = 'MIXED'
        else:
            alignment_quality = 'CONFLICT'
        
        # ===== STEP 3: CALCULATE WEIGHTED SCORES =====
        weighted_score = weighted_score_sum / total_weight if total_weight > 0 else 0.5
        unweighted_score = unweighted_score_sum / total_timeframes if total_timeframes > 0 else 0.5
        
        # ===== STEP 4: DETERMINE OVERALL BIAS =====
        if bullish_count > bearish_count:
            bias = 'BULLISH'
            bias_strength = bullish_count / total_timeframes if total_timeframes > 0 else 0
        elif bearish_count > bullish_count:
            bias = 'BEARISH'
            bias_strength = bearish_count / total_timeframes if total_timeframes > 0 else 0
        else:
            bias = 'NEUTRAL'
            bias_strength = 0
        
        # ===== STEP 5: CONFLICT RESOLUTION =====
        conflict_result = self._resolve_conflicts(timeframe_results)
        
        # ===== STEP 6: CALCULATE CONFIDENCE BOOST =====
        confidence_boost = self._calculate_confidence_boost(
            alignment_percentage, conflict_result, bias_strength
        )
        
        # ===== STEP 7: CHECK CONFIRMATION =====
        if primary_direction:
            # Check if MTF aligns with primary direction
            confirmed = (bias == primary_direction and 
                        weighted_score >= self.min_score and 
                        alignment_percentage >= self.min_alignment)
        else:
            confirmed = weighted_score >= self.min_score and alignment_percentage >= self.min_alignment
        
        # ===== STEP 8: EXTRACT KEY LEVELS =====
        key_levels = self._extract_key_levels(timeframe_results)
        
        # ===== STEP 9: ANALYZE PULLBACK =====
        pullback_result = self._analyze_pullback(timeframe_results, primary_direction)
        
        # ===== STEP 10: BUILD STORY =====
        story, reasons = self._build_story(
            bias, alignment_quality, weighted_score,
            bullish_count, bearish_count, neutral_count,
            conflict_result, pullback_result, confirmed, primary_direction
        )
        
        return MTFAggregatedResult(
            confirmed=confirmed,
            score=round(weighted_score, 3),
            bias=bias,
            confidence=round(weighted_score, 3),
            bullish_timeframes=bullish_count,
            bearish_timeframes=bearish_count,
            neutral_timeframes=neutral_count,
            total_timeframes=total_timeframes,
            alignment_percentage=round(alignment_percentage, 3),
            alignment_quality=alignment_quality,
            weighted_score=round(weighted_score, 3),
            unweighted_score=round(unweighted_score, 3),
            pullback=pullback_result,
            key_levels=key_levels,
            conflict=conflict_result,
            confidence_boost=round(confidence_boost, 3),
            story=story,
            reasons=reasons,
            timeframe_results=timeframe_results
        )
    
    def _resolve_conflicts(self, timeframe_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Resolve conflicts between timeframes (higher timeframes win)"""
        try:
            # Sort timeframes by weight (higher first)
            sorted_tfs = sorted(
                timeframe_results.keys(),
                key=lambda x: self.weights.get(x, 1.0),
                reverse=True
            )
            
            if not sorted_tfs:
                return {'exists': False, 'resolution': 'NO_DATA'}
            
            # Get highest timeframe direction
            highest_tf = sorted_tfs[0]
            highest_result = timeframe_results[highest_tf]
            highest_direction = highest_result.get('direction', 'NEUTRAL')
            highest_confidence = highest_result.get('confidence', 0.5)
            
            # Check for conflicts
            conflicts = []
            for tf in sorted_tfs[1:]:
                tf_direction = timeframe_results[tf].get('direction', 'NEUTRAL')
                if tf_direction != 'NEUTRAL' and tf_direction != highest_direction:
                    conflicts.append({
                        'timeframe': tf,
                        'direction': tf_direction,
                        'conflicts_with': highest_tf
                    })
            
            if conflicts:
                return {
                    'exists': True,
                    'resolution': 'FOLLOW_HIGHEST',
                    'highest_timeframe': highest_tf,
                    'highest_direction': highest_direction,
                    'highest_confidence': highest_confidence,
                    'conflicts': conflicts,
                    'confidence_impact': -0.05 * len(conflicts)
                }
            else:
                return {
                    'exists': False,
                    'resolution': 'ALIGNED',
                    'highest_direction': highest_direction,
                    'confidence_impact': 0.05
                }
                
        except Exception as e:
            log.debug(f"Error resolving conflicts: {e}")
            return {'exists': False, 'resolution': 'ERROR', 'confidence_impact': 0}
    
    def _calculate_confidence_boost(
        self,
        alignment_percentage: float,
        conflict_result: Dict,
        bias_strength: float
    ) -> float:
        """Calculate confidence boost based on alignment"""
        boost = 0.0
        
        # Base alignment boost
        if alignment_percentage >= 0.8:
            boost += 0.10
        elif alignment_percentage >= 0.6:
            boost += 0.05
        elif alignment_percentage <= 0.3:
            boost -= 0.05
        
        # Bias strength boost
        if bias_strength >= 0.8:
            boost += 0.05
        elif bias_strength >= 0.6:
            boost += 0.02
        
        # Conflict impact
        boost += conflict_result.get('confidence_impact', 0)
        
        # Clamp to limits
        return max(self.confidence_penalty_max, min(self.confidence_boost_max, boost))
    
    def _extract_key_levels(self, timeframe_results: Dict[str, Dict]) -> Dict[str, List[float]]:
        """Extract support and resistance levels across all timeframes"""
        all_supports = []
        all_resistances = []
        
        for tf, result in timeframe_results.items():
            support = result.get('support')
            resistance = result.get('resistance')
            supports = result.get('supports', [])
            resistances = result.get('resistances', [])
            
            if support:
                all_supports.append(support)
            if resistance:
                all_resistances.append(resistance)
            
            all_supports.extend(supports)
            all_resistances.extend(resistances)
        
        # Cluster nearby levels (within 0.5%)
        clustered_supports = self._cluster_levels(all_supports)
        clustered_resistances = self._cluster_levels(all_resistances)
        
        return {
            'supports': clustered_supports[:3],
            'resistances': clustered_resistances[:3]
        }
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.005) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _analyze_pullback(self, timeframe_results: Dict[str, Dict], 
                          primary_direction: Optional[str]) -> Dict[str, Any]:
        """Analyze if price is pulling back to higher timeframe support/resistance"""
        if not primary_direction:
            return {'is_pullback': False, 'score': 0}
        
        # Get highest timeframe support/resistance
        highest_tf = max(timeframe_results.keys(), 
                        key=lambda x: self.weights.get(x, 1.0))
        highest_result = timeframe_results.get(highest_tf, {})
        
        if primary_direction == 'BUY' or primary_direction == 'BULLISH':
            support = highest_result.get('support')
            if support:
                return {
                    'is_pullback': True,
                    'direction': 'BULLISH',
                    'pullback_to_tf': highest_tf,
                    'pullback_level': support,
                    'score': 0.5
                }
        else:
            resistance = highest_result.get('resistance')
            if resistance:
                return {
                    'is_pullback': True,
                    'direction': 'BEARISH',
                    'pullback_to_tf': highest_tf,
                    'pullback_level': resistance,
                    'score': 0.5
                }
        
        return {'is_pullback': False, 'score': 0}
    
    def _build_story(
        self,
        bias: str,
        alignment_quality: str,
        weighted_score: float,
        bullish_count: int,
        bearish_count: int,
        neutral_count: int,
        conflict_result: Dict,
        pullback_result: Dict,
        confirmed: bool,
        primary_direction: Optional[str]
    ) -> tuple:
        """Build human-readable MTF story"""
        reasons = []
        story_parts = []
        
        # Overall bias
        if bias == 'BULLISH':
            story_parts.append(f"MTF bullish bias with {weighted_score:.0%} confidence")
        elif bias == 'BEARISH':
            story_parts.append(f"MTF bearish bias with {weighted_score:.0%} confidence")
        else:
            story_parts.append(f"MTF neutral bias with {weighted_score:.0%} confidence")
        
        # Alignment quality
        if alignment_quality == 'STRONG_ALIGNMENT':
            story_parts.append(f"Strong alignment: {bullish_count} bullish, {bearish_count} bearish")
            reasons.append(f"Strong alignment: {bullish_count} timeframes agree on {bias.lower()} bias")
        elif alignment_quality == 'ALIGNED':
            story_parts.append(f"Moderate alignment: {bullish_count} bullish, {bearish_count} bearish")
            reasons.append(f"Moderate alignment: majority ({max(bullish_count, bearish_count)}) {bias.lower()}")
        elif alignment_quality == 'MIXED':
            story_parts.append(f"Mixed signals: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral")
            reasons.append(f"Mixed signals across timeframes")
        else:
            story_parts.append(f"Conflicting signals across timeframes")
            reasons.append(f"Conflicting timeframes - reduce confidence")
        
        # Conflict resolution
        if conflict_result.get('exists'):
            highest = conflict_result.get('highest_timeframe', 'higher')
            highest_dir = conflict_result.get('highest_direction', 'neutral')
            story_parts.append(f"Following {highest} timeframe bias ({highest_dir})")
            reasons.append(f"Following {highest} timeframe bias ({highest_dir})")
        
        # Pullback
        if pullback_result.get('is_pullback'):
            pullback_to = pullback_result.get('pullback_to_tf', 'support')
            story_parts.append(f"Price pulling back to {pullback_to} support")
            reasons.append(f"Price at {pullback_to} support - potential entry")
        
        # Confirmation with primary
        if primary_direction:
            if confirmed:
                story_parts.append(f"MTF confirms {primary_direction} signal")
                reasons.append(f"MTF confirms {primary_direction} signal")
            else:
                story_parts.append(f"MTF does not confirm {primary_direction} signal")
                reasons.append(f"MTF does not confirm - consider waiting")
        
        story = " | ".join(story_parts)
        
        return story, reasons
    
    def _empty_result(self, reason: str) -> MTFAggregatedResult:
        """Return empty result when aggregation fails"""
        return MTFAggregatedResult(
            confirmed=False,
            score=0.5,
            bias='NEUTRAL',
            confidence=0.5,
            bullish_timeframes=0,
            bearish_timeframes=0,
            neutral_timeframes=0,
            total_timeframes=0,
            alignment_percentage=0,
            alignment_quality='NO_DATA',
            weighted_score=0.5,
            unweighted_score=0.5,
            pullback={'is_pullback': False, 'score': 0},
            key_levels={'supports': [], 'resistances': []},
            conflict={'exists': False},
            confidence_boost=0,
            story=f"MTF analysis failed: {reason}",
            reasons=[reason],
            timeframe_results={}
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def aggregate_mtf_results(
    timeframe_results: Dict[str, Dict[str, Any]],
    primary_direction: Optional[str] = None
) -> MTFAggregatedResult:
    """
    Convenience function to aggregate MTF results
    
    Args:
        timeframe_results: Dictionary of timeframe -> analysis result
        primary_direction: Direction from primary timeframe
    
    Returns:
        MTFAggregatedResult with complete MTF analysis
    """
    aggregator = MTFAggregator()
    return aggregator.aggregate(timeframe_results, primary_direction)