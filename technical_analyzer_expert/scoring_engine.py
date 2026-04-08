# scoring_engine.py - Advanced Scoring Engine for Technical Analysis
"""
Scoring Engine Expert
- Calculates final signal scores using regime-specific weights
- Applies agreement adjustments (boost/penalty)
- Blends with signal confidence for final score
- Assigns grades (A+ to F) based on final score
- Determines action (STRONG_ENTRY, ENTER_NOW, SKIP)
- All thresholds from ta_config.py
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import configuration and core classes
from .ta_config import *
from .ta_core import Grade, SignalAction, CategoryScore, IndicatorSignal

# Try to import logger
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class ScoringEngine:
    """
    Advanced Scoring Engine
    Calculates final signal scores with regime-specific weighting
    """
    
    def __init__(self):
        """Initialize the scoring engine with thresholds from config"""
        self.base_weights = SCORING_WEIGHTS.copy()
        self.regime_weights = REGIME_WEIGHTS.copy()
        self.grade_thresholds = GRADE_THRESHOLDS.copy()
        self.position_multipliers = POSITION_MULTIPLIERS.copy()
        
        # Action thresholds
        self.strong_entry_threshold = ACTION_STRONG_ENTRY
        self.enter_now_threshold = ACTION_ENTER_NOW
        
        # Agreement adjustments
        self.agreement_high_threshold = AGREEMENT_HIGH_THRESHOLD
        self.agreement_high_boost = AGREEMENT_HIGH_BOOST
        self.agreement_low_threshold = AGREEMENT_LOW_THRESHOLD
        self.agreement_low_penalty = AGREEMENT_LOW_PENALTY
        
        log.info("ScoringEngine initialized")
    
    # ============================================================================
    # MAIN SCORING METHODS
    # ============================================================================
    
    def calculate_final_score(self, 
                              category_scores: Dict[str, CategoryScore],
                              signal_confidence: float,
                              agreement_score: float,
                              regime: Optional[str] = None,
                              htf_boost: float = 0.0,
                              divergence_bias: float = 0.0) -> Tuple[float, float, float]:
        """
        Calculate final bullish and bearish scores
        
        Args:
            category_scores: Dictionary of category scores (momentum, trend, volatility, volume)
            signal_confidence: Raw signal confidence from signal generator
            agreement_score: Agreement between indicators (0-1)
            regime: Current market regime (TRENDING, RANGING, VOLATILE)
            htf_boost: Boost from HTF alignment (0-0.15)
            divergence_bias: Bias from divergences (-1 to 1)
        
        Returns:
            Tuple of (bullish_score, bearish_score, final_score)
            final_score = bullish_score - bearish_score (range -1 to 1)
        """
        try:
            # Step 1: Get regime-specific weights
            weights = self._get_regime_weights(regime)
            
            # Step 2: Calculate weighted bullish and bearish scores
            bullish_score = 0.0
            bearish_score = 0.0
            total_weight = 0.0
            
            for category, score_data in category_scores.items():
                weight = weights.get(category, 0.15)
                
                # Use net score (positive = bullish, negative = bearish)
                net_score = score_data.net_score
                
                if net_score > 0:
                    bullish_score += net_score * weight
                else:
                    bearish_score += abs(net_score) * weight
                
                total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                bullish_score = bullish_score / total_weight
                bearish_score = bearish_score / total_weight
            else:
                bullish_score = 0.5
                bearish_score = 0.5
            
            # Step 3: Apply agreement adjustment
            bullish_score, bearish_score = self._apply_agreement_adjustment(
                bullish_score, bearish_score, agreement_score
            )
            
            # Step 4: Apply HTF boost/penalty
            bullish_score, bearish_score = self._apply_htf_adjustment(
                bullish_score, bearish_score, htf_boost
            )
            
            # Step 5: Apply divergence bias
            bullish_score, bearish_score = self._apply_divergence_adjustment(
                bullish_score, bearish_score, divergence_bias
            )
            
            # Step 6: Blend with signal confidence
            final_bullish = (bullish_score * 0.6) + (signal_confidence * 0.4)
            final_bearish = (bearish_score * 0.6) + ((1 - signal_confidence) * 0.4)
            
            # Step 7: Calculate final net score
            final_score = final_bullish - final_bearish
            
            # Clamp to -1 to 1 range
            final_score = max(-1.0, min(1.0, final_score))
            
            log.debug(f"Final scores - Bullish: {final_bullish:.3f}, Bearish: {final_bearish:.3f}, Net: {final_score:.3f}")
            
            return final_bullish, final_bearish, final_score
            
        except Exception as e:
            log.error(f"Error calculating final score: {e}")
            return 0.5, 0.5, 0.0
    
    def calculate_confidence_score(self, 
                                   bullish_score: float, 
                                   bearish_score: float,
                                   agreement_score: float,
                                   volume_ratio: float,
                                   trend_strength: float) -> float:
        """
        Calculate overall confidence score (0-1)
        
        Args:
            bullish_score: Bullish score (0-1)
            bearish_score: Bearish score (0-1)
            agreement_score: Agreement between indicators (0-1)
            volume_ratio: Current volume ratio (1 = average)
            trend_strength: Trend strength from regime (0-1)
        
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Base confidence from direction strength
            direction_strength = abs(bullish_score - bearish_score)
            base_confidence = direction_strength
            
            # Apply agreement multiplier
            agreement_multiplier = 0.7 + (agreement_score * 0.3)
            confidence = base_confidence * agreement_multiplier
            
            # Apply volume multiplier
            volume_multiplier = min(1.5, max(0.7, volume_ratio))
            confidence *= volume_multiplier
            
            # Apply trend strength multiplier
            trend_multiplier = 0.8 + (trend_strength * 0.4)
            confidence *= trend_multiplier
            
            # Clamp to 0-1
            confidence = max(0.0, min(1.0, confidence))
            
            return round(confidence, 3)
            
        except Exception as e:
            log.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_grade_and_multiplier(self, final_score: float) -> Tuple[str, float]:
        """
        Get grade and position multiplier based on final score
        
        Args:
            final_score: Final net score (-1 to 1)
        
        Returns:
            Tuple of (grade, position_multiplier)
        """
        try:
            # Convert -1 to 1 range to 0-1 range for grading
            normalized_score = (final_score + 1) / 2
            
            # Find grade
            grade = "F"
            for g, threshold in sorted(self.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
                if normalized_score >= threshold:
                    grade = g
                    break
            
            # Get position multiplier
            multiplier = self.position_multipliers.get(grade, 0.0)
            
            return grade, multiplier
            
        except Exception as e:
            log.error(f"Error getting grade: {e}")
            return "F", 0.0
    
    def determine_action(self, confidence: float, final_score: float, 
                         risk_reward: float = 0.0, min_filters_passed: bool = True) -> Tuple[str, str]:
        """
        Determine action based on confidence, final score, and filters
        
        Args:
            confidence: Overall confidence score (0-1)
            final_score: Final net score (-1 to 1)
            risk_reward: Risk/reward ratio (if calculated)
            min_filters_passed: Whether minimum filters passed
        
        Returns:
            Tuple of (action, reason)
        """
        try:
            # Check minimum filters first
            if not min_filters_passed:
                return "SKIP", "Minimum filters not met"
            
            # Check confidence threshold
            if confidence < MIN_CONFIDENCE_TO_TRADE:
                return "SKIP", f"Confidence below threshold: {confidence:.2f} < {MIN_CONFIDENCE_TO_TRADE}"
            
            # Check final score direction strength
            if abs(final_score) < MIN_TREND_STRENGTH:
                return "SKIP", f"Direction strength too weak: {abs(final_score):.2f} < {MIN_TREND_STRENGTH}"
            
            # Check risk/reward if provided
            if risk_reward > 0 and risk_reward < MIN_RISK_REWARD:
                return "SKIP", f"Risk/Reward below threshold: {risk_reward:.2f} < {MIN_RISK_REWARD}"
            
            # Determine action based on confidence
            if confidence >= self.strong_entry_threshold:
                return "STRONG_ENTRY", f"Strong signal with {confidence:.2%} confidence"
            elif confidence >= self.enter_now_threshold:
                return "ENTER_NOW", f"Good signal with {confidence:.2%} confidence"
            else:
                return "SKIP", f"Confidence too low: {confidence:.2%} < {self.enter_now_threshold:.0%}"
                
        except Exception as e:
            log.error(f"Error determining action: {e}")
            return "SKIP", "Error in action determination"
    
    # ============================================================================
    # ADJUSTMENT METHODS
    # ============================================================================
    
    def _get_regime_weights(self, regime: Optional[str]) -> Dict[str, float]:
        """
        Get scoring weights based on market regime
        """
        if not regime:
            return self.base_weights.copy()
        
        # Find matching regime category
        for regime_category, weights in self.regime_weights.items():
            if regime_category in regime:
                log.debug(f"Using {regime_category} weights for regime {regime}")
                return weights.copy()
        
        # Default to base weights
        return self.base_weights.copy()
    
    def _apply_agreement_adjustment(self, bullish_score: float, bearish_score: float,
                                     agreement_score: float) -> Tuple[float, float]:
        """
        Apply agreement-based adjustment to scores
        
        High agreement: Boost the dominant direction
        Low agreement: Penalize both directions
        """
        if agreement_score >= self.agreement_high_threshold:
            # High agreement: Boost dominant direction
            if bullish_score > bearish_score:
                bullish_score = min(1.0, bullish_score + self.agreement_high_boost)
            else:
                bearish_score = min(1.0, bearish_score + self.agreement_high_boost)
            
            log.debug(f"High agreement ({agreement_score:.2f}) - boosting dominant direction")
            
        elif agreement_score <= self.agreement_low_threshold:
            # Low agreement: Penalize both
            bullish_score = max(0.0, bullish_score + self.agreement_low_penalty)
            bearish_score = max(0.0, bearish_score + self.agreement_low_penalty)
            
            log.debug(f"Low agreement ({agreement_score:.2f}) - penalizing both directions")
        
        return bullish_score, bearish_score
    
    def _apply_htf_adjustment(self, bullish_score: float, bearish_score: float,
                               htf_boost: float) -> Tuple[float, float]:
        """
        Apply HTF alignment adjustment
        """
        if htf_boost > 0:
            # HTF aligned - boost the current direction
            if bullish_score > bearish_score:
                bullish_score = min(1.0, bullish_score + htf_boost)
            else:
                bearish_score = min(1.0, bearish_score + htf_boost)
            
            log.debug(f"HTF alignment boost: +{htf_boost:.2f}")
        
        return bullish_score, bearish_score
    
    def _apply_divergence_adjustment(self, bullish_score: float, bearish_score: float,
                                      divergence_bias: float) -> Tuple[float, float]:
        """
        Apply divergence-based adjustment
        
        divergence_bias: -1 (bearish) to +1 (bullish)
        """
        if divergence_bias > 0.3:
            # Bullish divergence dominant
            bullish_score = min(1.0, bullish_score + divergence_bias * 0.2)
            log.debug(f"Bullish divergence adjustment: +{divergence_bias * 0.2:.2f}")
            
        elif divergence_bias < -0.3:
            # Bearish divergence dominant
            bearish_score = min(1.0, bearish_score + abs(divergence_bias) * 0.2)
            log.debug(f"Bearish divergence adjustment: +{abs(divergence_bias) * 0.2:.2f}")
        
        return bullish_score, bearish_score
    
    # ============================================================================
    # AGGREGATE SCORING METHODS
    # ============================================================================
    
    def score_signal(self, 
                     category_scores: Dict[str, CategoryScore],
                     signal_confidence: float,
                     agreement_score: float,
                     volume_ratio: float,
                     trend_strength: float,
                     regime: Optional[str] = None,
                     htf_boost: float = 0.0,
                     divergence_bias: float = 0.0,
                     risk_reward: float = 0.0,
                     min_filters_passed: bool = True) -> Dict[str, Any]:
        """
        Complete signal scoring pipeline
        
        Returns:
            Dictionary with all scoring results
        """
        try:
            # Step 1: Calculate final scores
            bullish_score, bearish_score, final_score = self.calculate_final_score(
                category_scores, signal_confidence, agreement_score,
                regime, htf_boost, divergence_bias
            )
            
            # Step 2: Calculate confidence
            confidence = self.calculate_confidence_score(
                bullish_score, bearish_score, agreement_score,
                volume_ratio, trend_strength
            )
            
            # Step 3: Get grade and multiplier
            grade, position_multiplier = self.get_grade_and_multiplier(final_score)
            
            # Step 4: Determine action
            action, action_reason = self.determine_action(
                confidence, final_score, risk_reward, min_filters_passed
            )
            
            # Step 5: Prepare result
            result = {
                "bullish_score": round(bullish_score, 3),
                "bearish_score": round(bearish_score, 3),
                "final_score": round(final_score, 3),
                "confidence": confidence,
                "grade": grade,
                "position_multiplier": position_multiplier,
                "action": action,
                "action_reason": action_reason,
                "direction": "BULLISH" if final_score > 0.1 else "BEARISH" if final_score < -0.1 else "NEUTRAL"
            }
            
            log.info(f"Scoring complete: Grade={grade}, Confidence={confidence:.2%}, Action={action}")
            
            return result
            
        except Exception as e:
            log.error(f"Error in signal scoring: {e}")
            return {
                "bullish_score": 0.5,
                "bearish_score": 0.5,
                "final_score": 0.0,
                "confidence": 0.0,
                "grade": "F",
                "position_multiplier": 0.0,
                "action": "SKIP",
                "action_reason": f"Scoring error: {e}",
                "direction": "NEUTRAL"
            }
    
    def calculate_category_agreement(self, category_scores: Dict[str, CategoryScore]) -> float:
        """
        Calculate overall agreement across categories
        
        Returns:
            Agreement score between 0 and 1
        """
        if not category_scores:
            return 0.0
        
        # Collect net scores from each category
        net_scores = [score.net_score for score in category_scores.values() if score.net_score != 0]
        
        if not net_scores:
            return 0.0
        
        # Count bullish and bearish
        bullish_count = sum(1 for s in net_scores if s > 0)
        bearish_count = sum(1 for s in net_scores if s < 0)
        total = len(net_scores)
        
        if total == 0:
            return 0.0
        
        # Agreement is the percentage of categories agreeing on the dominant direction
        agreement = max(bullish_count, bearish_count) / total
        
        return round(agreement, 3)
    
    def calculate_weighted_score(self, signals: List[IndicatorSignal]) -> float:
        """
        Calculate weighted score from individual indicator signals
        
        Returns:
            Net score (-1 to 1)
        """
        if not signals:
            return 0.0
        
        bullish_weighted = 0.0
        bearish_weighted = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = signal.weight
            
            if signal.signal.value in ["BULLISH", "CROSS_ABOVE", "DIVERGENCE_BULLISH"]:
                bullish_weighted += signal.strength * weight
            elif signal.signal.value in ["BEARISH", "CROSS_BELOW", "DIVERGENCE_BEARISH"]:
                bearish_weighted += signal.strength * weight
            
            total_weight += weight
        
        if total_weight > 0:
            net_score = (bullish_weighted - bearish_weighted) / total_weight
            return max(-1.0, min(1.0, net_score))
        
        return 0.0
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_grade_description(self, grade: str) -> str:
        """
        Get human-readable description of a grade
        """
        descriptions = {
            "A+": "Exceptional signal - very high probability setup",
            "A": "Excellent signal - high probability setup",
            "B+": "Very good signal - above average probability",
            "B": "Good signal - solid setup with clear edge",
            "B-": "Decent signal - moderate probability, acceptable risk",
            "C+": "Fair signal - average setup, careful position sizing",
            "C": "Weak signal - below average probability",
            "D": "Poor signal - high risk, avoid or very small position",
            "F": "No signal - skip this trade"
        }
        return descriptions.get(grade, "Unknown grade")
    
    def get_action_description(self, action: str) -> str:
        """
        Get human-readable description of an action
        """
        descriptions = {
            "STRONG_ENTRY": "Strong entry signal - consider full position size",
            "ENTER_NOW": "Good entry signal - consider standard position",
            "SKIP": "Skip this trade - conditions not met",
            "WAIT": "Wait for confirmation - monitor for better entry"
        }
        return descriptions.get(action, "Unknown action")
    
    def validate_score(self, score: float) -> bool:
        """
        Validate if a score is within acceptable range
        """
        return 0.0 <= score <= 1.0
    
    def get_score_quality(self, confidence: float, agreement: float) -> str:
        """
        Get qualitative assessment of signal quality
        """
        if confidence >= 0.85 and agreement >= 0.7:
            return "EXCELLENT"
        elif confidence >= 0.75 and agreement >= 0.6:
            return "GOOD"
        elif confidence >= 0.65 and agreement >= 0.5:
            return "FAIR"
        elif confidence >= MIN_CONFIDENCE_TO_TRADE:
            return "MARGINAL"
        else:
            return "POOR"


# ============================================================================
# SIMPLE WRAPPER FUNCTIONS
# ============================================================================

def score_technical_signal(category_scores: Dict[str, CategoryScore],
                           signal_confidence: float,
                           agreement_score: float,
                           volume_ratio: float,
                           trend_strength: float,
                           regime: Optional[str] = None,
                           htf_boost: float = 0.0,
                           divergence_bias: float = 0.0,
                           risk_reward: float = 0.0,
                           min_filters_passed: bool = True) -> Dict[str, Any]:
    """
    Simple wrapper function for signal scoring
    """
    engine = ScoringEngine()
    return engine.score_signal(
        category_scores, signal_confidence, agreement_score,
        volume_ratio, trend_strength, regime, htf_boost,
        divergence_bias, risk_reward, min_filters_passed
    )


def calculate_agreement(category_scores: Dict[str, CategoryScore]) -> float:
    """
    Calculate agreement across categories
    """
    engine = ScoringEngine()
    return engine.calculate_category_agreement(category_scores)