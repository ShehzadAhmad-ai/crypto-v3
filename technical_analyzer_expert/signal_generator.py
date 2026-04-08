# signal_generator.py - Advanced Signal Generation Engine
"""
Signal Generation Expert
- Combines all indicator scores and divergences to generate BUY/SELL signal
- Calculates bullish and bearish scores using weighted averages
- Checks signal conflicts (if conflict ratio >30%, returns NEUTRAL)
- Applies HTF alignment boost/penalty
- Applies agreement boost (if >70%) or penalty (if <40%)
- Applies minimum filters BEFORE generating signal
- Generates one-line decision reason
- Returns TASignal object (without trade setup)
- All thresholds from ta_config.py
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import configuration and core classes
from .ta_config import *
from .ta_core import (
    TASignal, CategoryScore, IndicatorSignal, DivergenceResult,
    SignalDirection, SignalAction, create_skip_signal
)

# Try to import logger
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class SignalGenerator:
    """
    Advanced Signal Generation Engine
    Combines all analysis to generate final trading signals
    """
    
    def __init__(self):
        """Initialize the signal generator with thresholds from config"""
        self.min_confidence = MIN_CONFIDENCE_TO_TRADE
        self.min_agreement = MIN_AGREEMENT
        self.min_volume_ratio = MIN_VOLUME_RATIO
        self.min_trend_strength = MIN_TREND_STRENGTH
        self.max_conflict_ratio = MAX_CONFLICT_RATIO
        
        # HTF adjustments
        self.htf_alignment_boost = HTF_ALIGNMENT_BOOST
        self.htf_conflict_penalty = HTF_CONFLICT_PENALTY
        
        # Agreement adjustments
        self.agreement_high_threshold = AGREEMENT_HIGH_THRESHOLD
        self.agreement_high_boost = AGREEMENT_HIGH_BOOST
        self.agreement_low_threshold = AGREEMENT_LOW_THRESHOLD
        self.agreement_low_penalty = AGREEMENT_LOW_PENALTY
        
        log.info("SignalGenerator initialized")
    
    # ============================================================================
    # MAIN SIGNAL GENERATION METHODS
    # ============================================================================
    
    def generate_signal(self,
                        symbol: str,
                        category_scores: Dict[str, CategoryScore],
                        divergences: List[DivergenceResult],
                        htf_alignment_score: float,
                        volume_ratio: float,
                        trend_strength: float,
                        regime_bias: float,
                        indicator_count: int = 0) -> TASignal:
        """
        Generate final trading signal from all analysis components
        
        Args:
            symbol: Trading symbol
            category_scores: Dictionary of category scores
            divergences: List of detected divergences
            htf_alignment_score: HTF alignment score (0-1)
            volume_ratio: Current volume ratio
            trend_strength: Trend strength from regime (0-1)
            regime_bias: Regime bias (-1 to 1, positive = bullish)
            indicator_count: Number of indicators analyzed
        
        Returns:
            TASignal object with complete signal information
        """
        try:
            # Step 1: Calculate raw bullish and bearish scores
            bullish_score, bearish_score, net_score = self._calculate_raw_scores(category_scores)
            
            # Step 2: Calculate agreement score
            agreement_score = self._calculate_agreement(category_scores)
            
            # Step 3: Check for conflicts
            conflict_ratio = self._calculate_conflict_ratio(category_scores)
            if conflict_ratio > self.max_conflict_ratio:
                log.debug(f"High conflict ratio ({conflict_ratio:.2f}) - returning neutral signal")
                return create_skip_signal(symbol, f"High indicator conflict: {conflict_ratio:.1%}")
            
            # Step 4: Apply HTF alignment adjustment
            bullish_score, bearish_score, htf_boost = self._apply_htf_adjustment(
                bullish_score, bearish_score, htf_alignment_score, net_score
            )
            
            # Step 5: Apply agreement adjustment
            bullish_score, bearish_score = self._apply_agreement_adjustment(
                bullish_score, bearish_score, agreement_score
            )
            
            # Step 6: Apply divergence adjustment
            divergence_bias = self._calculate_divergence_bias(divergences)
            bullish_score, bearish_score = self._apply_divergence_adjustment(
                bullish_score, bearish_score, divergence_bias
            )
            
            # Step 7: Apply regime bias
            bullish_score, bearish_score = self._apply_regime_bias(
                bullish_score, bearish_score, regime_bias
            )
            
            # Step 8: Calculate final confidence
            confidence = self._calculate_confidence(
                bullish_score, bearish_score, agreement_score,
                volume_ratio, trend_strength, htf_alignment_score
            )
            
            # Step 9: Apply minimum filters
            min_filters_passed, filter_reason = self._check_minimum_filters(
                confidence, agreement_score, volume_ratio
            )
            
            if not min_filters_passed:
                return create_skip_signal(symbol, filter_reason)
            
            # Step 10: Determine direction
            direction = self._determine_direction(bullish_score, bearish_score, net_score)
            
            if direction == "NEUTRAL":
                return create_skip_signal(symbol, "No clear direction (bullish/bearish scores too close)")
            
            # Step 11: Generate decision reason
            decision_reason = self._generate_decision_reason(
                direction, bullish_score, bearish_score, confidence,
                agreement_score, divergences, htf_alignment_score
            )
            
            # Step 12: Create signal (without trade setup)
            signal = TASignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                grade="PENDING",  # Will be set by scoring engine
                position_multiplier=0.0,  # Will be set by scoring engine
                action="PENDING",  # Will be set by scoring engine
                decision_reason=decision_reason,
                raw_bullish_score=bullish_score,
                raw_bearish_score=bearish_score,
                agreement_score=agreement_score,
                divergence_count=len(divergences),
                indicator_count=indicator_count,
                htf_aligned=htf_alignment_score > 0.6
            )
            
            log.info(f"Signal generated: {direction} with confidence {confidence:.2%}")
            
            return signal
            
        except Exception as e:
            log.error(f"Error generating signal: {e}")
            return create_skip_signal(symbol, f"Signal generation error: {e}")
    
    # ============================================================================
    # SCORE CALCULATION METHODS
    # ============================================================================
    
    def _calculate_raw_scores(self, category_scores: Dict[str, CategoryScore]) -> Tuple[float, float, float]:
        """
        Calculate raw bullish and bearish scores from categories
        
        Returns:
            Tuple of (bullish_score, bearish_score, net_score)
        """
        bullish_scores = []
        bearish_scores = []
        
        for category, score in category_scores.items():
            if score.net_score > 0:
                bullish_scores.append(score.net_score)
            else:
                bearish_scores.append(abs(score.net_score))
        
        bullish_score = np.mean(bullish_scores) if bullish_scores else 0.3
        bearish_score = np.mean(bearish_scores) if bearish_scores else 0.3
        net_score = bullish_score - bearish_score
        
        return bullish_score, bearish_score, net_score
    
    def _calculate_agreement(self, category_scores: Dict[str, CategoryScore]) -> float:
        """
        Calculate agreement score across categories
        
        Returns:
            Agreement score between 0 and 1
        """
        if not category_scores:
            return 0.0
        
        # Count bullish and bearish categories
        bullish_count = 0
        bearish_count = 0
        
        for score in category_scores.values():
            if score.net_score > 0.1:
                bullish_count += 1
            elif score.net_score < -0.1:
                bearish_count += 1
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.5
        
        # Agreement is the percentage of categories agreeing on dominant direction
        agreement = max(bullish_count, bearish_count) / total
        
        return round(agreement, 3)
    
    def _calculate_conflict_ratio(self, category_scores: Dict[str, CategoryScore]) -> float:
        """
        Calculate conflict ratio (percentage of conflicting categories)
        
        Returns:
            Conflict ratio between 0 and 1
        """
        if not category_scores:
            return 0.0
        
        # Count conflicting categories (bullish and bearish both present)
        bullish_count = sum(1 for s in category_scores.values() if s.net_score > 0.1)
        bearish_count = sum(1 for s in category_scores.values() if s.net_score < -0.1)
        
        total_categories = len(category_scores)
        
        if total_categories == 0:
            return 0.0
        
        # Conflict is when both bullish and bearish exist
        if bullish_count > 0 and bearish_count > 0:
            # Conflict ratio = min(bullish, bearish) / total
            conflict_ratio = min(bullish_count, bearish_count) / total_categories
        else:
            conflict_ratio = 0.0
        
        return conflict_ratio
    
    def _calculate_divergence_bias(self, divergences: List[DivergenceResult]) -> float:
        """
        Calculate net divergence bias
        
        Returns:
            Bias between -1 (bearish) and +1 (bullish)
        """
        if not divergences:
            return 0.0
        
        bullish_strength = sum(d.strength for d in divergences if d.type == "BULLISH")
        bearish_strength = sum(d.strength for d in divergences if d.type == "BEARISH")
        total_strength = bullish_strength + bearish_strength
        
        if total_strength > 0:
            return (bullish_strength - bearish_strength) / total_strength
        
        return 0.0
    
    def _calculate_confidence(self, bullish_score: float, bearish_score: float,
                              agreement_score: float, volume_ratio: float,
                              trend_strength: float, htf_alignment: float) -> float:
        """
        Calculate overall confidence score
        
        Returns:
            Confidence score between 0 and 1
        """
        # Direction strength
        direction_strength = abs(bullish_score - bearish_score)
        
        # Agreement multiplier
        agreement_multiplier = 0.7 + (agreement_score * 0.3)
        
        # Volume multiplier
        volume_multiplier = min(1.5, max(0.7, volume_ratio))
        
        # Trend strength multiplier
        trend_multiplier = 0.8 + (trend_strength * 0.4)
        
        # HTF alignment multiplier
        htf_multiplier = 0.8 + (htf_alignment * 0.4)
        
        # Calculate confidence
        confidence = direction_strength * agreement_multiplier * volume_multiplier * trend_multiplier * htf_multiplier
        
        # Clamp to 0-1
        confidence = max(0.0, min(1.0, confidence))
        
        return round(confidence, 3)
    
    # ============================================================================
    # ADJUSTMENT METHODS
    # ============================================================================
    
    def _apply_htf_adjustment(self, bullish_score: float, bearish_score: float,
                               htf_alignment: float, net_score: float) -> Tuple[float, float, float]:
        """
        Apply HTF alignment adjustment
        
        Returns:
            Adjusted (bullish_score, bearish_score, boost_applied)
        """
        if htf_alignment > 0.7:
            # Strong alignment - boost dominant direction
            boost = self.htf_alignment_boost * 1.5
            if net_score > 0:
                bullish_score = min(1.0, bullish_score + boost)
            else:
                bearish_score = min(1.0, bearish_score + boost)
            return bullish_score, bearish_score, boost
            
        elif htf_alignment > 0.6:
            # Moderate alignment
            boost = self.htf_alignment_boost
            if net_score > 0:
                bullish_score = min(1.0, bullish_score + boost)
            else:
                bearish_score = min(1.0, bearish_score + boost)
            return bullish_score, bearish_score, boost
            
        elif htf_alignment < 0.4:
            # Conflict - penalize dominant direction
            penalty = self.htf_conflict_penalty
            if net_score > 0:
                bullish_score = max(0.0, bullish_score - penalty)
            else:
                bearish_score = max(0.0, bearish_score - penalty)
            return bullish_score, bearish_score, -penalty
        
        return bullish_score, bearish_score, 0.0
    
    def _apply_agreement_adjustment(self, bullish_score: float, bearish_score: float,
                                     agreement_score: float) -> Tuple[float, float]:
        """
        Apply agreement-based adjustment
        """
        if agreement_score >= self.agreement_high_threshold:
            # High agreement - boost dominant direction
            if bullish_score > bearish_score:
                bullish_score = min(1.0, bullish_score + self.agreement_high_boost)
            else:
                bearish_score = min(1.0, bearish_score + self.agreement_high_boost)
                
        elif agreement_score <= self.agreement_low_threshold:
            # Low agreement - penalize both
            bullish_score = max(0.0, bullish_score + self.agreement_low_penalty)
            bearish_score = max(0.0, bearish_score + self.agreement_low_penalty)
        
        return bullish_score, bearish_score
    
    def _apply_divergence_adjustment(self, bullish_score: float, bearish_score: float,
                                      divergence_bias: float) -> Tuple[float, float]:
        """
        Apply divergence-based adjustment
        """
        if divergence_bias > 0.3:
            # Bullish divergence dominant
            bullish_score = min(1.0, bullish_score + divergence_bias * 0.2)
        elif divergence_bias < -0.3:
            # Bearish divergence dominant
            bearish_score = min(1.0, bearish_score + abs(divergence_bias) * 0.2)
        
        return bullish_score, bearish_score
    
    def _apply_regime_bias(self, bullish_score: float, bearish_score: float,
                            regime_bias: float) -> Tuple[float, float]:
        """
        Apply regime bias adjustment
        """
        if regime_bias > 0.3:
            # Bullish regime
            bullish_score = min(1.0, bullish_score + regime_bias * 0.15)
        elif regime_bias < -0.3:
            # Bearish regime
            bearish_score = min(1.0, bearish_score + abs(regime_bias) * 0.15)
        
        return bullish_score, bearish_score
    
    # ============================================================================
    # FILTER METHODS
    # ============================================================================
    
    def _check_minimum_filters(self, confidence: float, agreement: float,
                                volume_ratio: float) -> Tuple[bool, str]:
        """
        Check minimum filters before generating signal
        
        Returns:
            Tuple of (passed, reason)
        """
        if confidence < self.min_confidence:
            return False, f"Confidence below threshold: {confidence:.2%} < {self.min_confidence:.0%}"
        
        if agreement < self.min_agreement:
            return False, f"Agreement below threshold: {agreement:.2%} < {self.min_agreement:.0%}"
        
        if volume_ratio < self.min_volume_ratio:
            return False, f"Volume ratio below threshold: {volume_ratio:.2f}x < {self.min_volume_ratio:.1f}x"
        
        return True, "Filters passed"
    
    def _determine_direction(self, bullish_score: float, bearish_score: float,
                              net_score: float) -> str:
        """
        Determine signal direction
        
        Returns:
            "BUY", "SELL", or "NEUTRAL"
        """
        # Need minimum spread to avoid whipsaws
        spread_threshold = 0.1
        
        if net_score > spread_threshold and bullish_score > bearish_score:
            return "BUY"
        elif net_score < -spread_threshold and bearish_score > bullish_score:
            return "SELL"
        else:
            return "NEUTRAL"
    
    # ============================================================================
    # REASON GENERATION
    # ============================================================================
    
    def _generate_decision_reason(self, direction: str, bullish_score: float,
                                   bearish_score: float, confidence: float,
                                   agreement: float, divergences: List[DivergenceResult],
                                   htf_alignment: float) -> str:
        """
        Generate human-readable decision reason
        """
        reasons = []
        
        # Direction and strength
        if direction == "BUY":
            reasons.append(f"BUY signal with {confidence:.1%} confidence")
        else:
            reasons.append(f"SELL signal with {confidence:.1%} confidence")
        
        # Score difference
        score_diff = abs(bullish_score - bearish_score)
        if score_diff > 0.4:
            reasons.append("strong directional bias")
        elif score_diff > 0.2:
            reasons.append("moderate directional bias")
        
        # Agreement
        if agreement > 0.8:
            reasons.append("strong indicator agreement")
        elif agreement > 0.6:
            reasons.append("good indicator agreement")
        
        # Divergences
        if divergences:
            bullish_divs = [d for d in divergences if d.type == "BULLISH"]
            bearish_divs = [d for d in divergences if d.type == "BEARISH"]
            
            if direction == "BUY" and bullish_divs:
                reasons.append(f"{len(bullish_divs)} bullish divergence(s)")
            elif direction == "SELL" and bearish_divs:
                reasons.append(f"{len(bearish_divs)} bearish divergence(s)")
        
        # HTF alignment
        if htf_alignment > 0.7:
            reasons.append("HTF aligned")
        elif htf_alignment < 0.4:
            reasons.append("HTF conflict")
        
        # Generate reason string
        if reasons:
            return " | ".join(reasons)
        else:
            return f"{direction} signal generated"
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_signal_summary(self, signal: TASignal) -> Dict[str, Any]:
        """
        Get human-readable summary of a signal
        """
        return {
            "direction": signal.direction,
            "confidence": f"{signal.confidence:.1%}",
            "grade": signal.grade,
            "action": signal.action,
            "reason": signal.decision_reason,
            "agreement": f"{signal.agreement_score:.1%}",
            "bullish_score": f"{signal.raw_bullish_score:.2f}",
            "bearish_score": f"{signal.raw_bearish_score:.2f}",
            "divergences": signal.divergence_count,
            "indicators": signal.indicator_count,
            "htf_aligned": signal.htf_aligned
        }
    
    def validate_signal(self, signal: TASignal) -> bool:
        """
        Validate if a signal is valid for trading
        """
        if not signal:
            return False
        
        if signal.direction == "NEUTRAL":
            return False
        
        if signal.action not in ["STRONG_ENTRY", "ENTER_NOW"]:
            return False
        
        if signal.confidence < self.min_confidence:
            return False
        
        return True


# ============================================================================
# SIMPLE WRAPPER FUNCTIONS
# ============================================================================

def generate_technical_signal(symbol: str,
                              category_scores: Dict[str, CategoryScore],
                              divergences: List[DivergenceResult],
                              htf_alignment_score: float,
                              volume_ratio: float,
                              trend_strength: float,
                              regime_bias: float,
                              indicator_count: int = 0) -> TASignal:
    """
    Simple wrapper function for signal generation
    """
    generator = SignalGenerator()
    return generator.generate_signal(
        symbol, category_scores, divergences, htf_alignment_score,
        volume_ratio, trend_strength, regime_bias, indicator_count
    )


def get_signal_direction(signal: TASignal) -> str:
    """
    Get signal direction as string
    """
    if signal:
        return signal.direction
    return "NEUTRAL"


def is_signal_valid(signal: TASignal) -> bool:
    """
    Check if signal is valid for trading
    """
    generator = SignalGenerator()
    return generator.validate_signal(signal)