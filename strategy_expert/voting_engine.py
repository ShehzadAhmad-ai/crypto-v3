"""
Voting Engine for Strategy Expert
Handles weighted voting and agreement checks (Steps 3-4 of pipeline)

Step 3: Weighted Voting
    buy_score = Σ(confidence × weight)
    sell_score = Σ(confidence × weight)
    Higher score wins direction

Step 4: Agreement Check
    agreement_ratio = max(buy_count, sell_count) / total_strategies
    if agreement_ratio < MIN_AGREEMENT_RATIO → HOLD
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from strategy_expert.base_strategy import StrategyOutput
from strategy_expert.strategy_config import get_pipeline_config, PipelineConfig


class VoteResult(Enum):
    """Possible voting results"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    TIE = "TIE"
    INSUFFICIENT_AGREEMENT = "INSUFFICIENT_AGREEMENT"


@dataclass
class VotingStats:
    """Statistics from the voting process"""
    buy_count: int = 0
    sell_count: int = 0
    buy_score: float = 0.0
    sell_score: float = 0.0
    total_strategies: int = 0
    agreement_ratio: float = 0.0
    buy_strategies: List[str] = None
    sell_strategies: List[str] = None
    buy_confidences: List[float] = None
    sell_confidences: List[float] = None
    buy_weights: List[float] = None
    sell_weights: List[float] = None
    
    def __post_init__(self):
        if self.buy_strategies is None:
            self.buy_strategies = []
        if self.sell_strategies is None:
            self.sell_strategies = []
        if self.buy_confidences is None:
            self.buy_confidences = []
        if self.sell_confidences is None:
            self.sell_confidences = []
        if self.buy_weights is None:
            self.buy_weights = []
        if self.sell_weights is None:
            self.sell_weights = []
    
    @property
    def total_votes(self) -> int:
        return self.buy_count + self.sell_count
    
    @property
    def is_buy_winning(self) -> bool:
        return self.buy_score > self.sell_score
    
    @property
    def is_sell_winning(self) -> bool:
        return self.sell_score > self.buy_score
    
    @property
    def is_tie(self) -> bool:
        return abs(self.buy_score - self.sell_score) < 0.001
    
    @property
    def score_diff(self) -> float:
        return abs(self.buy_score - self.sell_score)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/debugging"""
        return {
            'buy_count': self.buy_count,
            'sell_count': self.sell_count,
            'buy_score': round(self.buy_score, 3),
            'sell_score': round(self.sell_score, 3),
            'total_strategies': self.total_strategies,
            'agreement_ratio': round(self.agreement_ratio, 3),
            'buy_strategies': self.buy_strategies,
            'sell_strategies': self.sell_strategies,
        }


class VotingEngine:
    """
    Weighted voting engine for combining multiple strategy signals
    
    Features:
    - Weighted scoring (confidence × weight)
    - Agreement ratio check
    - Detailed voting statistics
    - Configurable thresholds
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize Voting Engine
        
        Args:
            config: Pipeline configuration (uses default if not provided)
        """
        self.config = config or get_pipeline_config()
        self.min_agreement_ratio = self.config.min_agreement_ratio
    
    def vote(self, strategy_outputs: List[StrategyOutput]) -> Tuple[VoteResult, VotingStats]:
        """
        Execute weighted voting on strategy outputs
        
        Args:
            strategy_outputs: List of strategy outputs (already filtered by quality)
        
        Returns:
            Tuple of (VoteResult, VotingStats)
        """
        # Initialize stats
        stats = VotingStats()
        stats.total_strategies = len(strategy_outputs)
        
        # Separate BUY and SELL signals
        for output in strategy_outputs:
            if output.action == 'BUY':
                stats.buy_count += 1
                stats.buy_score += output.confidence * output.weight
                stats.buy_strategies.append(output.strategy_name)
                stats.buy_confidences.append(output.confidence)
                stats.buy_weights.append(output.weight)
            
            elif output.action == 'SELL':
                stats.sell_count += 1
                stats.sell_score += output.confidence * output.weight
                stats.sell_strategies.append(output.strategy_name)
                stats.sell_confidences.append(output.confidence)
                stats.sell_weights.append(output.weight)
        
        # Calculate agreement ratio
        max_votes = max(stats.buy_count, stats.sell_count)
        stats.agreement_ratio = max_votes / stats.total_strategies if stats.total_strategies > 0 else 0
        
        # Check if we have any signals
        if stats.total_votes == 0:
            return VoteResult.HOLD, stats
        
        # Check agreement threshold
        if stats.agreement_ratio < self.min_agreement_ratio:
            return VoteResult.INSUFFICIENT_AGREEMENT, stats
        
        # Determine winner based on weighted scores
        if stats.buy_score > stats.sell_score:
            return VoteResult.BUY, stats
        elif stats.sell_score > stats.buy_score:
            return VoteResult.SELL, stats
        else:
            return VoteResult.TIE, stats
    
    def get_winning_strategies(self, result: VoteResult, 
                               stats: VotingStats) -> List[str]:
        """
        Get list of strategies that voted for the winning side
        
        Args:
            result: The vote result
            stats: Voting statistics
        
        Returns:
            List of strategy names
        """
        if result == VoteResult.BUY:
            return stats.buy_strategies
        elif result == VoteResult.SELL:
            return stats.sell_strategies
        else:
            return []
    
    def get_winning_strategies_with_weights(self, result: VoteResult,
                                            stats: VotingStats) -> List[Tuple[str, float, float]]:
        """
        Get winning strategies with their confidence and weight
        
        Args:
            result: The vote result
            stats: Voting statistics
        
        Returns:
            List of (strategy_name, confidence, weight)
        """
        if result == VoteResult.BUY:
            return list(zip(stats.buy_strategies, 
                           stats.buy_confidences, 
                           stats.buy_weights))
        elif result == VoteResult.SELL:
            return list(zip(stats.sell_strategies, 
                           stats.sell_confidences, 
                           stats.sell_weights))
        else:
            return []
    
    def calculate_confidence_score(self, result: VoteResult,
                                   stats: VotingStats) -> float:
        """
        Calculate overall confidence score based on voting results
        
        Formula:
            confidence = (winning_score / total_score) × agreement_ratio
        
        Where:
            winning_score = max(buy_score, sell_score)
            total_score = buy_score + sell_score
            agreement_ratio = max(buy_count, sell_count) / total_strategies
        
        Returns:
            Confidence score between 0 and 1
        """
        if result not in [VoteResult.BUY, VoteResult.SELL]:
            return 0.0
        
        winning_score = max(stats.buy_score, stats.sell_score)
        total_score = stats.buy_score + stats.sell_score
        
        if total_score == 0:
            return 0.0
        
        # Base confidence from score ratio
        score_confidence = winning_score / total_score
        
        # Multiply by agreement ratio
        final_confidence = score_confidence * stats.agreement_ratio
        
        return min(1.0, final_confidence)
    
    def generate_vote_reason(self, result: VoteResult, 
                             stats: VotingStats) -> str:
        """
        Generate human-readable reason for the vote result
        
        Args:
            result: The vote result
            stats: Voting statistics
        
        Returns:
            Human-readable reason string
        """
        if result == VoteResult.HOLD:
            return "No strategies generated signals"
        
        elif result == VoteResult.INSUFFICIENT_AGREEMENT:
            return (f"Insufficient agreement: {stats.agreement_ratio:.1%} "
                   f"(needs {self.min_agreement_ratio:.0%})")
        
        elif result == VoteResult.TIE:
            return f"Tie vote: BUY={stats.buy_score:.2f} vs SELL={stats.sell_score:.2f}"
        
        elif result == VoteResult.BUY:
            return (f"BUY wins: {stats.buy_score:.2f} vs {stats.sell_score:.2f} "
                   f"({stats.buy_count}/{stats.total_strategies} strategies, "
                   f"{stats.agreement_ratio:.1%} agreement)")
        
        elif result == VoteResult.SELL:
            return (f"SELL wins: {stats.sell_score:.2f} vs {stats.buy_score:.2f} "
                   f"({stats.sell_count}/{stats.total_strategies} strategies, "
                   f"{stats.agreement_ratio:.1%} agreement)")
        
        return "Unknown vote result"
    
    def get_vote_summary(self, strategy_outputs: List[StrategyOutput]) -> Dict:
        """
        Get a complete summary of the voting process
        
        Args:
            strategy_outputs: List of strategy outputs
        
        Returns:
            Dictionary with complete voting analysis
        """
        result, stats = self.vote(strategy_outputs)
        
        summary = {
            'result': result.value,
            'result_detail': self.generate_vote_reason(result, stats),
            'stats': stats.to_dict(),
            'confidence': self.calculate_confidence_score(result, stats),
            'winning_strategies': self.get_winning_strategies(result, stats),
            'winning_strategies_detail': self.get_winning_strategies_with_weights(result, stats),
        }
        
        return summary


class WeightedVotingEngine(VotingEngine):
    """
    Extended voting engine with additional features:
    - Confidence threshold per strategy
    - Minimum vote count requirement
    - Dynamic threshold adjustment based on market regime
    """
    
    def __init__(self, config: PipelineConfig = None, 
                 min_votes: int = 2):
        """
        Initialize Weighted Voting Engine
        
        Args:
            config: Pipeline configuration
            min_votes: Minimum number of votes required (not just ratio)
        """
        super().__init__(config)
        self.min_votes = min_votes
    
    def vote_with_min_votes(self, strategy_outputs: List[StrategyOutput]) -> Tuple[VoteResult, VotingStats]:
        """
        Execute voting with minimum vote requirement
        
        Args:
            strategy_outputs: List of strategy outputs
        
        Returns:
            Tuple of (VoteResult, VotingStats)
        """
        result, stats = self.vote(strategy_outputs)
        
        # Check minimum vote count
        if result in [VoteResult.BUY, VoteResult.SELL]:
            winning_count = stats.buy_count if result == VoteResult.BUY else stats.sell_count
            
            if winning_count < self.min_votes:
                return VoteResult.INSUFFICIENT_AGREEMENT, stats
        
        return result, stats
    
    def vote_with_regime_adjustment(self, strategy_outputs: List[StrategyOutput],
                                    regime_bias: float) -> Tuple[VoteResult, VotingStats]:
        """
        Execute voting with regime-based adjustment
        
        When regime bias is strong, reduce agreement threshold
        
        Args:
            strategy_outputs: List of strategy outputs
            regime_bias: Regime bias score (-1 to 1)
        
        Returns:
            Tuple of (VoteResult, VotingStats)
        """
        # Adjust agreement threshold based on regime
        original_threshold = self.min_agreement_ratio
        
        if abs(regime_bias) > 0.5:
            # Strong regime: lower agreement requirement
            self.min_agreement_ratio = max(0.4, original_threshold * 0.8)
        elif abs(regime_bias) > 0.3:
            # Moderate regime: slight adjustment
            self.min_agreement_ratio = max(0.45, original_threshold * 0.9)
        else:
            # Neutral: use original
            self.min_agreement_ratio = original_threshold
        
        result, stats = self.vote(strategy_outputs)
        
        # Restore original threshold
        self.min_agreement_ratio = original_threshold
        
        return result, stats


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_vote_result_from_signals(buy_signals: List[StrategyOutput],
                                    sell_signals: List[StrategyOutput],
                                    min_agreement_ratio: float = 0.55) -> Tuple[VoteResult, VotingStats]:
    """
    Create vote result from pre-separated buy and sell signals
    
    Useful when strategies are already filtered by quality
    
    Args:
        buy_signals: List of BUY strategy outputs
        sell_signals: List of SELL strategy outputs
        min_agreement_ratio: Minimum agreement required
    
    Returns:
        Tuple of (VoteResult, VotingStats)
    """
    total_strategies = len(buy_signals) + len(sell_signals)
    
    if total_strategies == 0:
        return VoteResult.HOLD, VotingStats()
    
    stats = VotingStats(
        buy_count=len(buy_signals),
        sell_count=len(sell_signals),
        buy_score=sum(s.confidence * s.weight for s in buy_signals),
        sell_score=sum(s.confidence * s.weight for s in sell_signals),
        total_strategies=total_strategies,
        buy_strategies=[s.strategy_name for s in buy_signals],
        sell_strategies=[s.strategy_name for s in sell_signals],
        buy_confidences=[s.confidence for s in buy_signals],
        sell_confidences=[s.confidence for s in sell_signals],
        buy_weights=[s.weight for s in buy_signals],
        sell_weights=[s.weight for s in sell_signals],
    )
    
    max_votes = max(stats.buy_count, stats.sell_count)
    stats.agreement_ratio = max_votes / total_strategies if total_strategies > 0 else 0
    
    if stats.agreement_ratio < min_agreement_ratio:
        return VoteResult.INSUFFICIENT_AGREEMENT, stats
    
    if stats.buy_score > stats.sell_score:
        return VoteResult.BUY, stats
    elif stats.sell_score > stats.buy_score:
        return VoteResult.SELL, stats
    else:
        return VoteResult.TIE, stats


def calculate_weighted_score(strategy_outputs: List[StrategyOutput], 
                            action: str) -> float:
    """
    Calculate weighted score for a specific action
    
    Args:
        strategy_outputs: List of strategy outputs
        action: 'BUY' or 'SELL'
    
    Returns:
        Total weighted score
    """
    return sum(
        s.confidence * s.weight 
        for s in strategy_outputs 
        if s.action == action
    )