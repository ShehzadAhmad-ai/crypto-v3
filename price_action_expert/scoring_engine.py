"""
scoring_engine.py
Layer 7: Scoring Engine for Price Action Expert V3.5

Combines all components with dynamic weights:
- Pattern quality and alignment
- Trap severity
- Sequence confidence
- MTF alignment
- Support/Resistance context
- Volume confirmation
- Liquidity confirmation
- Regime alignment

Outputs final confidence score, grade, and position multiplier
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import configuration
from .price_action_config import (
    BASE_SCORING_WEIGHTS,
    DYNAMIC_WEIGHT_ADJUSTMENTS,
    GRADE_THRESHOLDS,
    GRADE_POSITION_MULTIPLIER,
    MIN_TRADE_CONFIDENCE
)


class SignalGrade(Enum):
    """Signal grades with position multipliers"""
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    D = "D"
    F = "F"


@dataclass
class ScoringResult:
    """
    Complete scoring result
    """
    # Final scores
    total_score: float                  # 0-1 raw score before sigmoid
    confidence: float                   # 0-1 final confidence after sigmoid
    grade: SignalGrade
    position_multiplier: float          # 0.5-1.5 based on grade
    
    # Component scores (for debugging)
    component_scores: Dict[str, float]
    component_weights: Dict[str, float]
    
    # Adjusted scores
    pattern_score: float
    trap_score: float
    sequence_score: float
    context_score: float
    
    # Decision
    is_tradeable: bool
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            'confidence': round(self.confidence, 3),
            'grade': self.grade.value,
            'position_multiplier': self.position_multiplier,
            'component_scores': {k: round(v, 3) for k, v in self.component_scores.items()},
            'is_tradeable': self.is_tradeable,
            'reasons': self.reasons[:5]
        }


class ScoringEngine:
    """
    Advanced scoring engine with dynamic weights
    
    Features:
    - Weighted combination of all components
    - Dynamic weight adjustment based on market conditions
    - Non-linear confidence calculation (sigmoid)
    - Grade assignment with position multipliers
    - Tradeability filtering
    """
    
    def __init__(self):
        """Initialize the scoring engine"""
        self.scoring_history: List[ScoringResult] = []
    
    # =========================================================
    # COMPONENT SCORING
    # =========================================================
    
    def score_pattern(self, pattern_quality: str, pattern_strength: float,
                      pattern_alignment: int = 1, total_patterns: int = 1) -> float:
        """
        Score the pattern component
        
        Args:
            pattern_quality: 'A', 'B', or 'C'
            pattern_strength: 0-1 pattern strength
            pattern_alignment: Number of patterns aligned
            total_patterns: Total patterns detected
        
        Returns:
            Score 0-1
        """
        # Base score from pattern quality
        quality_scores = {'A': 0.9, 'B': 0.7, 'C': 0.5}
        base_score = quality_scores.get(pattern_quality, 0.5)
        
        # Adjust by pattern strength
        strength_score = base_score * pattern_strength
        
        # Alignment bonus (multiple patterns in same direction)
        if total_patterns > 1:
            alignment_ratio = pattern_alignment / total_patterns
            alignment_bonus = alignment_ratio * 0.15
            strength_score = min(0.95, strength_score + alignment_bonus)
        
        return strength_score
    
    def score_trap(self, trap_severity: str, trap_score: float,
                   liquidity_sweep: bool = False) -> float:
        """
        Score the trap component
        
        Args:
            trap_severity: 'extreme', 'strong', 'medium', 'minor', or 'none'
            trap_score: 0-1 trap severity score
            liquidity_sweep: Whether liquidity sweep occurred
        
        Returns:
            Score 0-1
        """
        if trap_severity == 'none':
            return 0.0
        
        # Base score from severity
        severity_scores = {
            'extreme': 0.95,
            'strong': 0.85,
            'medium': 0.70,
            'minor': 0.50
        }
        base_score = severity_scores.get(trap_severity, 0.5)
        
        # Adjust by trap score
        trap_adjusted = base_score * trap_score
        
        # Liquidity sweep boost
        if liquidity_sweep:
            trap_adjusted = min(0.95, trap_adjusted + 0.10)
        
        return trap_adjusted
    
    def score_sequence(self, sequence_confidence: float, sequence_type: str,
                       momentum_score: float) -> float:
        """
        Score the sequence component
        
        Args:
            sequence_confidence: 0-1 sequence confidence
            sequence_type: Type of sequence (reversal, momentum, etc.)
            momentum_score: 0-1 momentum score
        
        Returns:
            Score 0-1
        """
        # Base from sequence confidence
        base_score = sequence_confidence
        
        # Type adjustments
        type_boosts = {
            'reversal': 0.10,
            'momentum': 0.10,
            'compression_breakout': 0.15,
            'liquidity_run': 0.20,
            'exhaustion': -0.10,
            'indecision': -0.15
        }
        
        boost = type_boosts.get(sequence_type, 0.0)
        base_score = min(0.95, max(0.05, base_score + boost))
        
        # Combine with momentum
        combined = (base_score + momentum_score) / 2
        
        return combined
    
    def score_context(self, mtf_alignment: float, at_key_level: bool,
                      key_level_strength: float, structure_bias: str,
                      structure_strength: float, volatility_state: str,
                      session_weight: float) -> float:
        """
        Score the context component
        
        Args:
            mtf_alignment: 0-1 MTF alignment score
            at_key_level: Whether at key level
            key_level_strength: 0-1 key level strength
            structure_bias: 'bullish', 'bearish', 'neutral'
            structure_strength: 0-1 structure strength
            volatility_state: 'low', 'normal', 'high', 'extreme'
            session_weight: 0-1 session weight
        
        Returns:
            Score 0-1
        """
        scores = []
        weights = []
        
        # MTF alignment
        scores.append(mtf_alignment)
        weights.append(0.30)
        
        # Key level
        if at_key_level:
            level_score = 0.5 + (key_level_strength * 0.5)
        else:
            level_score = 0.3
        scores.append(level_score)
        weights.append(0.25)
        
        # Structure bias
        if structure_bias != 'neutral':
            structure_score = 0.5 + (structure_strength * 0.5)
        else:
            structure_score = 0.4
        scores.append(structure_score)
        weights.append(0.20)
        
        # Volatility
        volatility_scores = {
            'normal': 0.7,
            'low': 0.5,
            'high': 0.6,
            'extreme': 0.4
        }
        scores.append(volatility_scores.get(volatility_state, 0.5))
        weights.append(0.15)
        
        # Session
        scores.append(session_weight)
        weights.append(0.10)
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return weighted_score
    
    def score_volume(self, volume_ratio: float, volume_trend: str,
                     has_divergence: bool = False) -> float:
        """
        Score the volume component
        
        Args:
            volume_ratio: Current volume / average volume
            volume_trend: 'INCREASING', 'DECREASING', 'STABLE'
            has_divergence: Whether volume-price divergence exists
        
        Returns:
            Score 0-1
        """
        base_score = 0.5
        
        # Volume ratio adjustment
        if volume_ratio >= 2.0:
            base_score += 0.20
        elif volume_ratio >= 1.5:
            base_score += 0.15
        elif volume_ratio >= 1.2:
            base_score += 0.08
        elif volume_ratio <= 0.5:
            base_score -= 0.15
        
        # Volume trend adjustment
        if volume_trend == 'INCREASING':
            base_score += 0.10
        elif volume_trend == 'DECREASING':
            base_score -= 0.05
        
        # Divergence adjustment
        if has_divergence:
            base_score += 0.10
        
        return min(0.95, max(0.05, base_score))
    
    def score_liquidity(self, has_sweep: bool, sweep_count: int,
                        cascade_risk: float = 0.0) -> float:
        """
        Score the liquidity component
        
        Args:
            has_sweep: Whether liquidity sweep occurred
            sweep_count: Number of levels swept
            cascade_risk: 0-1 cascade risk score
        
        Returns:
            Score 0-1
        """
        if not has_sweep:
            return 0.3
        
        base_score = 0.6
        
        # Sweep count boost
        if sweep_count >= 3:
            base_score += 0.20
        elif sweep_count >= 2:
            base_score += 0.12
        elif sweep_count >= 1:
            base_score += 0.05
        
        # Cascade risk boost
        if cascade_risk > 0.7:
            base_score += 0.15
        elif cascade_risk > 0.4:
            base_score += 0.08
        
        return min(0.95, base_score)
    
    def score_regime(self, regime_bias: str, regime_score: float,
                     trend_stage: str) -> float:
        """
        Score the regime component
        
        Args:
            regime_bias: 'BULLISH', 'BEARISH', 'NEUTRAL'
            regime_score: -1 to 1 regime bias score
            trend_stage: 'early', 'mid', 'late', 'exhaustion', 'consolidation'
        
        Returns:
            Score 0-1
        """
        # Convert regime bias to 0-1
        if regime_bias == 'BULLISH':
            base_score = 0.5 + (abs(regime_score) * 0.5)
        elif regime_bias == 'BEARISH':
            base_score = 0.5 + (abs(regime_score) * 0.5)
        else:
            base_score = 0.4
        
        # Trend stage adjustment
        stage_boosts = {
            'early': 0.15,
            'mid': 0.08,
            'late': -0.05,
            'exhaustion': -0.15,
            'consolidation': 0.0
        }
        
        boost = stage_boosts.get(trend_stage, 0.0)
        base_score = min(0.95, max(0.2, base_score + boost))
        
        return base_score
    
    # =========================================================
    # DYNAMIC WEIGHT ADJUSTMENT
    # =========================================================
    
    def get_dynamic_weights(self, volatility_state: str, trend_stage: str,
                            has_trap: bool, sequence_type: str) -> Dict[str, float]:
        """
        Get dynamically adjusted weights based on market conditions
        
        Args:
            volatility_state: 'low', 'normal', 'high', 'extreme'
            trend_stage: 'early', 'mid', 'late', 'exhaustion', 'consolidation'
            has_trap: Whether trap detected
            sequence_type: Type of sequence
        
        Returns:
            Adjusted weights dictionary
        """
        # Start with base weights
        weights = BASE_SCORING_WEIGHTS.copy()
        
        # Volatility adjustments
        if volatility_state == 'high':
            weights['trap_severity'] = min(0.25, weights.get('trap_severity', 0.15) + 0.05)
            weights['pattern_quality'] = max(0.15, weights.get('pattern_quality', 0.20) - 0.05)
        elif volatility_state == 'low':
            weights['sequence_confidence'] = min(0.20, weights.get('sequence_confidence', 0.12) + 0.05)
        
        # Trend stage adjustments
        if trend_stage == 'early':
            weights['sequence_confidence'] = min(0.20, weights.get('sequence_confidence', 0.12) + 0.05)
            weights['regime_alignment'] = min(0.20, weights.get('regime_alignment', 0.12) + 0.05)
        elif trend_stage == 'late':
            weights['trap_severity'] = min(0.25, weights.get('trap_severity', 0.15) + 0.08)
            weights['pattern_quality'] = max(0.15, weights.get('pattern_quality', 0.20) - 0.05)
        elif trend_stage == 'exhaustion':
            weights['trap_severity'] = min(0.30, weights.get('trap_severity', 0.15) + 0.10)
        
        # Trap adjustments
        if has_trap:
            weights['trap_severity'] = min(0.30, weights.get('trap_severity', 0.15) + 0.10)
            weights['pattern_quality'] = max(0.15, weights.get('pattern_quality', 0.20) - 0.05)
        
        # Sequence type adjustments
        if sequence_type in ['liquidity_run', 'reversal']:
            weights['trap_severity'] = min(0.25, weights.get('trap_severity', 0.15) + 0.05)
            weights['sequence_confidence'] = min(0.20, weights.get('sequence_confidence', 0.12) + 0.05)
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    # =========================================================
    # CONFIDENCE CALCULATION
    # =========================================================
    
    def sigmoid(self, x: float, steepness: float = 10.0, midpoint: float = 0.5) -> float:
        """
        Sigmoid function for non-linear confidence mapping
        
        Args:
            x: Raw score (0-1)
            steepness: Steepness of the curve
            midpoint: Midpoint of the curve
        
        Returns:
            Confidence score (0-1)
        """
        # Center around midpoint
        x_centered = x - midpoint
        # Apply sigmoid
        result = 1 / (1 + np.exp(-steepness * x_centered))
        return result
    
    def calculate_confidence(self, raw_score: float, component_variance: float = 0.1) -> float:
        """
        Calculate final confidence from raw score
        
        Args:
            raw_score: Weighted sum of component scores (0-1)
            component_variance: Variance between components (penalty for disagreement)
        
        Returns:
            Confidence score (0-1)
        """
        # Apply sigmoid for non-linear mapping
        confidence = self.sigmoid(raw_score, steepness=10.0, midpoint=0.5)
        
        # Apply variance penalty
        confidence = confidence * (1 - min(0.3, component_variance))
        
        return min(0.98, max(0.02, confidence))
    
    def get_grade_and_multiplier(self, confidence: float) -> Tuple[SignalGrade, float]:
        """
        Get grade and position multiplier based on confidence
        
        Args:
            confidence: Final confidence score (0-1)
        
        Returns:
            (grade, position_multiplier)
        """
        for grade_name, threshold in sorted(GRADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if confidence >= threshold:
                grade = SignalGrade(grade_name)
                multiplier = GRADE_POSITION_MULTIPLIER.get(grade_name, 1.0)
                return grade, multiplier
        
        return SignalGrade.F, 0.0
    
    # =========================================================
    # COMPLETE SCORING
    # =========================================================
    
    def score(self,
              pattern_quality: str,
              pattern_strength: float,
              pattern_alignment: int,
              total_patterns: int,
              trap_severity: str,
              trap_score: float,
              liquidity_sweep: bool,
              sequence_confidence: float,
              sequence_type: str,
              momentum_score: float,
              mtf_alignment: float,
              at_key_level: bool,
              key_level_strength: float,
              structure_bias: str,
              structure_strength: float,
              volatility_state: str,
              session_weight: float,
              volume_ratio: float,
              volume_trend: str,
              has_volume_divergence: bool,
              has_liquidity_sweep: bool,
              sweep_count: int,
              cascade_risk: float,
              regime_bias: str,
              regime_score: float,
              trend_stage: str) -> ScoringResult:
        """
        Complete scoring with all components
        
        Args:
            All component scores (see above)
        
        Returns:
            ScoringResult with final confidence and grade
        """
        # Calculate component scores
        pattern_score = self.score_pattern(
            pattern_quality, pattern_strength, pattern_alignment, total_patterns
        )
        
        trap_score_val = self.score_trap(trap_severity, trap_score, liquidity_sweep)
        
        sequence_score = self.score_sequence(
            sequence_confidence, sequence_type, momentum_score
        )
        
        context_score = self.score_context(
            mtf_alignment, at_key_level, key_level_strength,
            structure_bias, structure_strength, volatility_state, session_weight
        )
        
        volume_score = self.score_volume(volume_ratio, volume_trend, has_volume_divergence)
        
        liquidity_score = self.score_liquidity(has_liquidity_sweep, sweep_count, cascade_risk)
        
        regime_score_val = self.score_regime(regime_bias, regime_score, trend_stage)
        
        # Store component scores
        component_scores = {
            'pattern': pattern_score,
            'trap': trap_score_val,
            'sequence': sequence_score,
            'context': context_score,
            'volume': volume_score,
            'liquidity': liquidity_score,
            'regime': regime_score_val
        }
        
        # Get dynamic weights
        has_trap = trap_severity != 'none'
        weights = self.get_dynamic_weights(volatility_state, trend_stage, has_trap, sequence_type)
        
        # Calculate weighted raw score
        raw_score = 0.0
        component_weights = {}
        
        # Map components to weights
        weight_mapping = {
            'pattern': 'pattern_quality',
            'trap': 'trap_severity',
            'sequence': 'sequence_confidence',
            'context': 'sr_alignment',
            'volume': 'volume_confirmation',
            'liquidity': 'liquidity_confirmation',
            'regime': 'regime_alignment'
        }
        
        for comp, score in component_scores.items():
            weight_key = weight_mapping.get(comp, comp)
            weight = weights.get(weight_key, 0.05)
            component_weights[comp] = weight
            raw_score += score * weight
        
        # Calculate component variance (penalty for disagreement)
        score_values = list(component_scores.values())
        if len(score_values) > 1:
            component_variance = np.var(score_values)
        else:
            component_variance = 0.0
        
        # Calculate final confidence
        confidence = self.calculate_confidence(raw_score, component_variance)
        
        # Get grade and multiplier
        grade, multiplier = self.get_grade_and_multiplier(confidence)
        
        # Determine tradeability
        is_tradeable = confidence >= MIN_TRADE_CONFIDENCE
        
        # Build reasons
        reasons = []
        if pattern_score >= 0.7:
            reasons.append(f"Strong pattern quality ({pattern_quality} grade)")
        if trap_score_val >= 0.7:
            reasons.append(f"Strong trap detection ({trap_severity} severity)")
        if sequence_score >= 0.7:
            reasons.append(f"High confidence sequence ({sequence_type})")
        if context_score >= 0.7:
            reasons.append("Strong market context")
        if mtf_alignment >= 0.8:
            reasons.append("Multi-timeframe alignment")
        if at_key_level:
            reasons.append(f"At key {key_level_strength:.0%} strength level")
        if confidence >= 0.85:
            reasons.append(f"Excellent confidence ({confidence:.0%})")
        elif confidence >= 0.70:
            reasons.append(f"Good confidence ({confidence:.0%})")
        
        if not is_tradeable:
            reasons.append(f"Confidence below threshold ({MIN_TRADE_CONFIDENCE:.0%})")
        
        result = ScoringResult(
            total_score=raw_score,
            confidence=confidence,
            grade=grade,
            position_multiplier=multiplier,
            component_scores=component_scores,
            component_weights=component_weights,
            pattern_score=pattern_score,
            trap_score=trap_score_val,
            sequence_score=sequence_score,
            context_score=context_score,
            is_tradeable=is_tradeable,
            reasons=reasons
        )
        
        # Store in history
        self.scoring_history.append(result)
        if len(self.scoring_history) > 100:
            self.scoring_history.pop(0)
        
        return result
    
    def score_from_components(self, components: Dict[str, Any]) -> ScoringResult:
        """
        Score from a dictionary of components (convenience method)
        
        Args:
            components: Dictionary with all component scores
        
        Returns:
            ScoringResult
        """
        return self.score(
            pattern_quality=components.get('pattern_quality', 'C'),
            pattern_strength=components.get('pattern_strength', 0.5),
            pattern_alignment=components.get('pattern_alignment', 1),
            total_patterns=components.get('total_patterns', 1),
            trap_severity=components.get('trap_severity', 'none'),
            trap_score=components.get('trap_score', 0.0),
            liquidity_sweep=components.get('liquidity_sweep', False),
            sequence_confidence=components.get('sequence_confidence', 0.5),
            sequence_type=components.get('sequence_type', 'indecision'),
            momentum_score=components.get('momentum_score', 0.5),
            mtf_alignment=components.get('mtf_alignment', 0.5),
            at_key_level=components.get('at_key_level', False),
            key_level_strength=components.get('key_level_strength', 0.5),
            structure_bias=components.get('structure_bias', 'neutral'),
            structure_strength=components.get('structure_strength', 0.5),
            volatility_state=components.get('volatility_state', 'normal'),
            session_weight=components.get('session_weight', 0.7),
            volume_ratio=components.get('volume_ratio', 1.0),
            volume_trend=components.get('volume_trend', 'STABLE'),
            has_volume_divergence=components.get('has_volume_divergence', False),
            has_liquidity_sweep=components.get('has_liquidity_sweep', False),
            sweep_count=components.get('sweep_count', 0),
            cascade_risk=components.get('cascade_risk', 0.0),
            regime_bias=components.get('regime_bias', 'NEUTRAL'),
            regime_score=components.get('regime_score', 0.0),
            trend_stage=components.get('trend_stage', 'consolidation')
        )


# ==================== CONVENIENCE FUNCTIONS ====================

def calculate_score(components: Dict[str, Any]) -> ScoringResult:
    """
    Convenience function to calculate score from components
    
    Args:
        components: Dictionary with all component scores
    
    Returns:
        ScoringResult
    """
    engine = ScoringEngine()
    return engine.score_from_components(components)


def get_confidence(components: Dict[str, Any]) -> float:
    """
    Convenience function to get confidence only
    
    Args:
        components: Dictionary with component scores
    
    Returns:
        Confidence score (0-1)
    """
    result = calculate_score(components)
    return result.confidence


def is_tradeable(components: Dict[str, Any]) -> bool:
    """
    Convenience function to check if signal is tradeable
    
    Args:
        components: Dictionary with component scores
    
    Returns:
        True if tradeable
    """
    result = calculate_score(components)
    return result.is_tradeable


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Test with a strong setup
    strong_components = {
        'pattern_quality': 'A',
        'pattern_strength': 0.85,
        'pattern_alignment': 3,
        'total_patterns': 3,
        'trap_severity': 'strong',
        'trap_score': 0.82,
        'liquidity_sweep': True,
        'sequence_confidence': 0.85,
        'sequence_type': 'reversal',
        'momentum_score': 0.78,
        'mtf_alignment': 0.85,
        'at_key_level': True,
        'key_level_strength': 0.8,
        'structure_bias': 'bullish',
        'structure_strength': 0.75,
        'volatility_state': 'normal',
        'session_weight': 0.9,
        'volume_ratio': 1.8,
        'volume_trend': 'INCREASING',
        'has_volume_divergence': True,
        'has_liquidity_sweep': True,
        'sweep_count': 2,
        'cascade_risk': 0.6,
        'regime_bias': 'BULLISH',
        'regime_score': 0.7,
        'trend_stage': 'early'
    }
    
    # Test with a weak setup
    weak_components = {
        'pattern_quality': 'C',
        'pattern_strength': 0.45,
        'pattern_alignment': 1,
        'total_patterns': 1,
        'trap_severity': 'none',
        'trap_score': 0.0,
        'liquidity_sweep': False,
        'sequence_confidence': 0.4,
        'sequence_type': 'indecision',
        'momentum_score': 0.45,
        'mtf_alignment': 0.4,
        'at_key_level': False,
        'key_level_strength': 0.3,
        'structure_bias': 'neutral',
        'structure_strength': 0.4,
        'volatility_state': 'low',
        'session_weight': 0.6,
        'volume_ratio': 0.8,
        'volume_trend': 'DECREASING',
        'has_volume_divergence': False,
        'has_liquidity_sweep': False,
        'sweep_count': 0,
        'cascade_risk': 0.0,
        'regime_bias': 'NEUTRAL',
        'regime_score': 0.0,
        'trend_stage': 'consolidation'
    }
    
    engine = ScoringEngine()
    
    print("=" * 70)
    print("STRONG SETUP SCORING")
    print("=" * 70)
    result = engine.score_from_components(strong_components)
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Grade: {result.grade.value}")
    print(f"Position Multiplier: {result.position_multiplier:.1f}x")
    print(f"Tradeable: {result.is_tradeable}")
    print("\nComponent Scores:")
    for comp, score in result.component_scores.items():
        weight = result.component_weights.get(comp, 0)
        print(f"  {comp:12s}: {score:.2f} (weight: {weight:.0%})")
    print("\nReasons:")
    for r in result.reasons:
        print(f"  ✓ {r}")
    
    print("\n" + "=" * 70)
    print("WEAK SETUP SCORING")
    print("=" * 70)
    result = engine.score_from_components(weak_components)
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Grade: {result.grade.value}")
    print(f"Position Multiplier: {result.position_multiplier:.1f}x")
    print(f"Tradeable: {result.is_tradeable}")
    print("\nComponent Scores:")
    for comp, score in result.component_scores.items():
        weight = result.component_weights.get(comp, 0)
        print(f"  {comp:12s}: {score:.2f} (weight: {weight:.0%})")
    print("\nReasons:")
    for r in result.reasons:
        print(f"  {r}")
    print("=" * 70)