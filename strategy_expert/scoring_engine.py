"""
Scoring Engine for Strategy Expert
Grades final signal and calculates position multiplier

Features:
- Grade assignment (A/B/C/D/F) based on confidence
- Position multiplier based on signal quality
- Comprehensive scoring across multiple dimensions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from strategy_expert.trade_combiner import CombinedTrade
from strategy_expert.voting_engine import VotingStats, VoteResult
from strategy_expert.strategy_config import get_pipeline_config, PipelineConfig


class Grade(Enum):
    """Signal grades"""
    A = "A"      # Excellent: confidence ≥ 85%
    B = "B"      # Good: confidence ≥ 75%
    C = "C"      # Average: confidence ≥ 65%
    D = "D"      # Poor: confidence ≥ 55%
    F = "F"      # Very Poor: confidence < 55%
    
    @property
    def description(self) -> str:
        """Get grade description"""
        descriptions = {
            'A': 'Excellent signal - High conviction trade',
            'B': 'Good signal - Strong setup',
            'C': 'Average signal - Acceptable setup',
            'D': 'Poor signal - Consider reducing size',
            'F': 'Very poor signal - Skip or minimal size'
        }
        return descriptions.get(self.value, 'Unknown grade')
    
    @property
    def recommended_multiplier(self) -> float:
        """Get recommended position multiplier based on grade"""
        multipliers = {
            'A': 1.5,
            'B': 1.2,
            'C': 1.0,
            'D': 0.7,
            'F': 0.5
        }
        return multipliers.get(self.value, 1.0)


@dataclass
class ScoreComponents:
    """Individual components that contribute to final score"""
    confidence: float = 0.0          # Base confidence from voting
    agreement_ratio: float = 0.0     # How many strategies agreed
    weighted_score_ratio: float = 0.0 # Buy/Sell score ratio
    risk_reward: float = 0.0         # Risk/reward ratio
    strategy_count: int = 0          # Number of strategies that agreed
    strategy_quality: float = 0.0    # Average weight of agreeing strategies
    regime_alignment: float = 0.0    # How well aligned with market regime
    
    def calculate_total_score(self) -> float:
        """
        Calculate total score (0-1)
        
        Formula:
            total = (confidence × 0.30) +
                    (agreement_ratio × 0.20) +
                    (weighted_score_ratio × 0.20) +
                    (min(rr/3, 1) × 0.10) +
                    (min(strategy_count/5, 1) × 0.10) +
                    (strategy_quality × 0.10)
        """
        # Risk/reward contribution (capped at 3.0)
        rr_contribution = min(self.risk_reward / 3.0, 1.0)
        
        # Strategy count contribution (max 5)
        count_contribution = min(self.strategy_count / 5.0, 1.0)
        
        total = (
            self.confidence * 0.30 +
            self.agreement_ratio * 0.20 +
            self.weighted_score_ratio * 0.20 +
            rr_contribution * 0.10 +
            count_contribution * 0.10 +
            self.strategy_quality * 0.10
        )
        
        return min(1.0, max(0.0, total))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'confidence': round(self.confidence, 3),
            'agreement_ratio': round(self.agreement_ratio, 3),
            'weighted_score_ratio': round(self.weighted_score_ratio, 3),
            'risk_reward': round(self.risk_reward, 2),
            'strategy_count': self.strategy_count,
            'strategy_quality': round(self.strategy_quality, 3),
            'regime_alignment': round(self.regime_alignment, 3),
            'total_score': round(self.calculate_total_score(), 3)
        }


@dataclass
class SignalGrade:
    """Complete signal grading result"""
    grade: Grade
    score: float
    components: ScoreComponents
    position_multiplier: float
    recommendation: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for output"""
        return {
            'grade': self.grade.value,
            'score': round(self.score, 3),
            'position_multiplier': round(self.position_multiplier, 2),
            'recommendation': self.recommendation,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'components': self.components.to_dict()
        }


class ScoringEngine:
    """
    Scores and grades final signals
    
    Features:
    - Multi-dimensional scoring
    - Grade assignment (A-F)
    - Position multiplier calculation
    - Strength/weakness analysis
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize Scoring Engine
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or get_pipeline_config()
        self.grade_thresholds = self.config.grade_thresholds
    
    def score_signal(self, combined_trade: CombinedTrade,
                    vote_result: VoteResult,
                    vote_stats: VotingStats,
                    regime_bias: float = 0.0,
                    strategy_weights: Dict[str, float] = None) -> ScoreComponents:
        """
        Calculate score components for a signal
        
        Args:
            combined_trade: Combined trade from TradeCombiner
            vote_result: Voting result (BUY/SELL)
            vote_stats: Voting statistics
            regime_bias: Market regime bias (-1 to 1)
            strategy_weights: Dictionary of strategy weights
        
        Returns:
            ScoreComponents with all calculated scores
        """
        # 1. Confidence (from voting)
        confidence = self._calculate_confidence(vote_result, vote_stats)
        
        # 2. Agreement ratio (percentage of strategies that agreed)
        total_strategies = vote_stats.total_strategies
        if vote_result == VoteResult.BUY:
            agreement_ratio = vote_stats.buy_count / total_strategies if total_strategies > 0 else 0
        else:
            agreement_ratio = vote_stats.sell_count / total_strategies if total_strategies > 0 else 0
        
        # 3. Weighted score ratio
        total_score = vote_stats.buy_score + vote_stats.sell_score
        if total_score > 0:
            if vote_result == VoteResult.BUY:
                weighted_score_ratio = vote_stats.buy_score / total_score
            else:
                weighted_score_ratio = vote_stats.sell_score / total_score
        else:
            weighted_score_ratio = 0
        
        # 4. Risk/reward (from combined trade)
        risk_reward = combined_trade.risk_reward if combined_trade else 0
        
        # 5. Strategy count (number of agreeing strategies)
        if vote_result == VoteResult.BUY:
            strategy_count = vote_stats.buy_count
        else:
            strategy_count = vote_stats.sell_count
        
        # 6. Strategy quality (average weight of agreeing strategies)
        if strategy_weights and strategy_count > 0:
            if vote_result == VoteResult.BUY:
                weights = [strategy_weights.get(name, 1.0) for name in vote_stats.buy_strategies]
            else:
                weights = [strategy_weights.get(name, 1.0) for name in vote_stats.sell_strategies]
            
            # Average weight, normalized to 0-1 range (assuming max weight is 2.0)
            avg_weight = sum(weights) / len(weights) if weights else 1.0
            strategy_quality = min(1.0, avg_weight / 2.0)  # Normalize: weight 2.0 = quality 1.0
        else:
            strategy_quality = 0.5  # Default middle
        
        # 7. Regime alignment
        regime_alignment = self._calculate_regime_alignment(vote_result, regime_bias)
        
        return ScoreComponents(
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            weighted_score_ratio=weighted_score_ratio,
            risk_reward=risk_reward,
            strategy_count=strategy_count,
            strategy_quality=strategy_quality,
            regime_alignment=regime_alignment
        )
    
    def _calculate_confidence(self, vote_result: VoteResult,
                             vote_stats: VotingStats) -> float:
        """
        Calculate overall confidence from voting results
        
        Formula: (winning_score / total_score) × agreement_ratio
        """
        if vote_result == VoteResult.BUY:
            winning_score = vote_stats.buy_score
        elif vote_result == VoteResult.SELL:
            winning_score = vote_stats.sell_score
        else:
            return 0.0
        
        total_score = vote_stats.buy_score + vote_stats.sell_score
        
        if total_score == 0:
            return 0.0
        
        score_ratio = winning_score / total_score
        agreement_ratio = max(vote_stats.buy_count, vote_stats.sell_count) / vote_stats.total_strategies
        
        confidence = score_ratio * agreement_ratio
        
        return min(1.0, confidence)
    
    def _calculate_regime_alignment(self, vote_result: VoteResult,
                                   regime_bias: float) -> float:
        """
        Calculate regime alignment score (0-1)
        
        BUY with bullish regime = high alignment
        SELL with bearish regime = high alignment
        """
        if vote_result == VoteResult.BUY:
            if regime_bias > 0:
                return min(1.0, regime_bias + 0.5)  # Bullish regime
            elif regime_bias < 0:
                return max(0.0, 1.0 - abs(regime_bias))  # Bearish regime (against)
            else:
                return 0.5  # Neutral
        elif vote_result == VoteResult.SELL:
            if regime_bias < 0:
                return min(1.0, abs(regime_bias) + 0.5)  # Bearish regime
            elif regime_bias > 0:
                return max(0.0, 1.0 - regime_bias)  # Bullish regime (against)
            else:
                return 0.5  # Neutral
        else:
            return 0.0
    
    def assign_grade(self, score: float) -> Grade:
        """
        Assign grade based on score
        
        Args:
            score: Total score (0-1)
        
        Returns:
            Grade (A-F)
        """
        if score >= self.grade_thresholds['A']:
            return Grade.A
        elif score >= self.grade_thresholds['B']:
            return Grade.B
        elif score >= self.grade_thresholds['C']:
            return Grade.C
        elif score >= self.grade_thresholds['D']:
            return Grade.D
        else:
            return Grade.F
    
    def calculate_position_multiplier(self, grade: Grade, 
                                     components: ScoreComponents) -> float:
        """
        Calculate final position multiplier
        
        Base multiplier from grade, then adjusted by components
        """
        # Base from grade
        multiplier = grade.recommended_multiplier
        
        # Adjust by risk/reward (higher RR = higher multiplier)
        if components.risk_reward >= 3.0:
            multiplier *= 1.2
        elif components.risk_reward >= 2.5:
            multiplier *= 1.1
        elif components.risk_reward >= 2.0:
            multiplier *= 1.05
        elif components.risk_reward < 1.5:
            multiplier *= 0.9
        
        # Adjust by agreement ratio
        if components.agreement_ratio >= 0.8:
            multiplier *= 1.1
        elif components.agreement_ratio >= 0.7:
            multiplier *= 1.05
        elif components.agreement_ratio < 0.6:
            multiplier *= 0.9
        
        # Adjust by strategy quality
        if components.strategy_quality >= 0.8:
            multiplier *= 1.05
        elif components.strategy_quality < 0.5:
            multiplier *= 0.9
        
        # Clamp to reasonable range
        return max(0.5, min(2.0, multiplier))
    
    def analyze_strengths_weaknesses(self, components: ScoreComponents,
                                     grade: Grade) -> Tuple[List[str], List[str]]:
        """
        Analyze strengths and weaknesses of the signal
        
        Returns:
            Tuple of (strengths, weaknesses) lists
        """
        strengths = []
        weaknesses = []
        
        # Confidence
        if components.confidence >= 0.8:
            strengths.append(f"High confidence: {components.confidence:.1%}")
        elif components.confidence < 0.6:
            weaknesses.append(f"Low confidence: {components.confidence:.1%}")
        
        # Agreement
        if components.agreement_ratio >= 0.7:
            strengths.append(f"Strong agreement: {components.agreement_ratio:.0%} of strategies")
        elif components.agreement_ratio < 0.6:
            weaknesses.append(f"Weak agreement: {components.agreement_ratio:.0%} of strategies")
        
        # Risk/Reward
        if components.risk_reward >= 2.5:
            strengths.append(f"Excellent risk/reward: {components.risk_reward:.1f}")
        elif components.risk_reward >= 2.0:
            strengths.append(f"Good risk/reward: {components.risk_reward:.1f}")
        elif components.risk_reward < 1.5:
            weaknesses.append(f"Poor risk/reward: {components.risk_reward:.1f}")
        
        # Strategy count
        if components.strategy_count >= 4:
            strengths.append(f"Multiple confirmations: {components.strategy_count} strategies")
        elif components.strategy_count <= 2:
            weaknesses.append(f"Few confirmations: only {components.strategy_count} strategies")
        
        # Strategy quality
        if components.strategy_quality >= 0.8:
            strengths.append("High-quality strategies agreed")
        elif components.strategy_quality < 0.5:
            weaknesses.append("Low-quality strategies agreed")
        
        # Grade-based
        if grade == Grade.A:
            strengths.append("Top-tier signal - high probability setup")
        elif grade == Grade.F:
            weaknesses.append("Poor signal quality - consider skipping")
        
        return strengths[:3], weaknesses[:3]  # Limit to 3 each
    
    def grade_signal(self, combined_trade: CombinedTrade,
                    vote_result: VoteResult,
                    vote_stats: VotingStats,
                    regime_bias: float = 0.0,
                    strategy_weights: Dict[str, float] = None) -> SignalGrade:
        """
        Complete signal grading process
        
        Args:
            combined_trade: Combined trade from TradeCombiner
            vote_result: Voting result (BUY/SELL)
            vote_stats: Voting statistics
            regime_bias: Market regime bias (-1 to 1)
            strategy_weights: Dictionary of strategy weights
        
        Returns:
            Complete SignalGrade with all information
        """
        # Calculate components
        components = self.score_signal(
            combined_trade, vote_result, vote_stats,
            regime_bias, strategy_weights
        )
        
        # Calculate total score
        total_score = components.calculate_total_score()
        
        # Assign grade
        grade = self.assign_grade(total_score)
        
        # Calculate position multiplier
        position_multiplier = self.calculate_position_multiplier(grade, components)
        
        # Analyze strengths and weaknesses
        strengths, weaknesses = self.analyze_strengths_weaknesses(components, grade)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(grade, components)
        
        return SignalGrade(
            grade=grade,
            score=total_score,
            components=components,
            position_multiplier=position_multiplier,
            recommendation=recommendation,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _generate_recommendation(self, grade: Grade,
                                 components: ScoreComponents) -> str:
        """
        Generate trading recommendation based on grade and components
        """
        if grade == Grade.A:
            return "STRONG_ENTRY - High probability setup with excellent metrics"
        elif grade == Grade.B:
            return "ENTRY - Good setup with solid confirmation"
        elif grade == Grade.C:
            return "CAUTIOUS_ENTRY - Acceptable setup, consider normal position size"
        elif grade == Grade.D:
            return "REDUCED_ENTRY - Poor metrics, use reduced position size"
        else:
            return "SKIP - Signal quality too low, wait for better setup"
    
    def get_grade_summary(self, signal_grade: SignalGrade) -> Dict:
        """
        Get formatted summary of the grade
        
        Args:
            signal_grade: SignalGrade object
        
        Returns:
            Dictionary with formatted summary
        """
        return {
            'grade': signal_grade.grade.value,
            'score': f"{signal_grade.score:.1%}",
            'multiplier': f"{signal_grade.position_multiplier:.1f}x",
            'recommendation': signal_grade.recommendation,
            'strengths': signal_grade.strengths,
            'weaknesses': signal_grade.weaknesses,
            'components': {
                'confidence': f"{signal_grade.components.confidence:.1%}",
                'agreement': f"{signal_grade.components.agreement_ratio:.0%}",
                'risk_reward': f"{signal_grade.components.risk_reward:.1f}",
                'strategies': signal_grade.components.strategy_count,
            }
        }


class EnhancedScoringEngine(ScoringEngine):
    """
    Enhanced scoring engine with additional features:
    - Market condition adjustments
    - Time-based scoring
    - Volatility normalization
    """
    
    def __init__(self, config: PipelineConfig = None,
                 volatility_adjustment: bool = True):
        super().__init__(config)
        self.volatility_adjustment = volatility_adjustment
    
    def score_with_market_context(self, combined_trade: CombinedTrade,
                                  vote_result: VoteResult,
                                  vote_stats: VotingStats,
                                  market_regime: Dict,
                                  strategy_weights: Dict[str, float] = None) -> SignalGrade:
        """
        Score signal with market context adjustments
        
        Args:
            combined_trade: Combined trade object
            vote_result: Voting result
            vote_stats: Voting statistics
            market_regime: Market regime data
            strategy_weights: Strategy weights dictionary
        
        Returns:
            SignalGrade with market-adjusted scoring
        """
        regime_name = market_regime.get('regime', 'UNKNOWN')
        regime_bias = market_regime.get('bias_score', 0)
        volatility = market_regime.get('volatility', 0)
        
        # Base grading
        grade = self.grade_signal(
            combined_trade, vote_result, vote_stats,
            regime_bias, strategy_weights
        )
        
        # Adjust for market conditions
        if self.volatility_adjustment and volatility > 0:
            # Reduce score in high volatility
            if volatility > 0.05:  # 5% volatility
                adjustment = 0.9
                grade.score *= adjustment
                grade.position_multiplier *= adjustment
                grade.weaknesses.append(f"High volatility ({volatility:.1%}) - reduced position")
            
            # Increase score in low volatility (breakout potential)
            elif volatility < 0.01:  # 1% volatility
                if 'squeeze' in str(market_regime).lower():
                    adjustment = 1.1
                    grade.score = min(1.0, grade.score * adjustment)
                    grade.position_multiplier = min(2.0, grade.position_multiplier * adjustment)
                    grade.strengths.append("Low volatility squeeze - breakout potential")
        
        # Adjust for regime alignment (already in components, but add comment)
        if grade.grade in [Grade.A, Grade.B]:
            if (vote_result == VoteResult.BUY and 'BULL' in regime_name) or \
               (vote_result == VoteResult.SELL and 'BEAR' in regime_name):
                grade.strengths.append(f"Aligned with {regime_name} regime")
        
        return grade


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_grade(confidence: float, risk_reward: float,
                agreement_ratio: float = 0.6) -> Grade:
    """
    Quick grade calculation without full scoring engine
    
    Args:
        confidence: Signal confidence (0-1)
        risk_reward: Risk/reward ratio
        agreement_ratio: Percentage of strategies that agreed
    
    Returns:
        Grade (A-F)
    """
    # Simple scoring formula
    score = (
        confidence * 0.4 +
        min(risk_reward / 3.0, 1.0) * 0.3 +
        agreement_ratio * 0.3
    )
    
    if score >= 0.85:
        return Grade.A
    elif score >= 0.75:
        return Grade.B
    elif score >= 0.65:
        return Grade.C
    elif score >= 0.55:
        return Grade.D
    else:
        return Grade.F


def calculate_position_size(grade: Grade, base_size: float) -> float:
    """
    Calculate position size based on grade
    
    Args:
        grade: Signal grade
        base_size: Base position size
    
    Returns:
        Adjusted position size
    """
    multipliers = {
        Grade.A: 1.5,
        Grade.B: 1.2,
        Grade.C: 1.0,
        Grade.D: 0.7,
        Grade.F: 0.5
    }
    
    return base_size * multipliers.get(grade, 1.0)