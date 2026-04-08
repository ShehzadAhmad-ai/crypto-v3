"""
Trade Combiner for Strategy Expert
Averages entry, stop loss, and take profit from multiple strategies (Step 5 of pipeline)

Step 5: Weighted Average
    entry = Σ(entry × weight) / Σ(weight)
    stop_loss = Σ(sl × weight) / Σ(weight)
    take_profit = Σ(tp × weight) / Σ(weight)
    
    Strong strategies have more influence on final price levels
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from strategy_expert.base_strategy import StrategyOutput
from strategy_expert.voting_engine import VoteResult, VotingStats


@dataclass
class CombinedTrade:
    """Result of combining multiple trade setups"""
    direction: str                  # 'BUY' or 'SELL'
    entry: float                    # Weighted average entry
    stop_loss: float                # Weighted average stop loss
    take_profit: float              # Weighted average take profit
    risk_reward: float              # Calculated risk/reward ratio
    risk_amount: float              # Risk amount in price terms
    reward_amount: float            # Reward amount in price terms
    strategies_used: List[str]      # List of strategies used in combination
    strategies_count: int           # Number of strategies used
    total_weight: float             # Sum of all weights used
    
    def validate(self) -> bool:
        """Validate that the combined trade makes sense"""
        if self.direction == 'BUY':
            valid = (self.stop_loss < self.entry < self.take_profit)
        else:  # SELL
            valid = (self.take_profit < self.entry < self.stop_loss)
        
        if not valid:
            return False
        
        # RR should be positive
        return self.risk_reward > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for output"""
        return {
            'direction': self.direction,
            'entry': round(self.entry, 4),
            'stop_loss': round(self.stop_loss, 4),
            'take_profit': round(self.take_profit, 4),
            'risk_reward': round(self.risk_reward, 2),
            'risk_amount': round(self.risk_amount, 4),
            'reward_amount': round(self.reward_amount, 4),
            'strategies_used': self.strategies_used,
            'strategies_count': self.strategies_count,
            'total_weight': round(self.total_weight, 3)
        }


class TradeCombiner:
    """
    Combines multiple strategy outputs into a single trade setup
    
    Features:
    - Weighted average for entry, SL, TP
    - Outlier detection and removal
    - Min/max bounds for safety
    - Risk/reward recalculation
    """
    
    def __init__(self, outlier_threshold: float = 2.0,
                 max_sl_distance_pct: float = 0.05,
                 min_entry_distance_pct: float = 0.001):
        """
        Initialize Trade Combiner
        
        Args:
            outlier_threshold: Z-score threshold for outlier removal (default 2.0)
            max_sl_distance_pct: Maximum SL distance as percentage of entry (default 5%)
            min_entry_distance_pct: Minimum distance between entry and SL/TP (default 0.1%)
        """
        self.outlier_threshold = outlier_threshold
        self.max_sl_distance_pct = max_sl_distance_pct
        self.min_entry_distance_pct = min_entry_distance_pct
    
    def combine(self, strategy_outputs: List[StrategyOutput],
                vote_result: VoteResult, vote_stats: VotingStats) -> Optional[CombinedTrade]:
        """
        Combine trade setups from winning strategies
        
        Args:
            strategy_outputs: List of all strategy outputs
            vote_result: The voting result (BUY or SELL)
            vote_stats: Voting statistics
        
        Returns:
            CombinedTrade object or None if not enough valid setups
        """
        if vote_result not in [VoteResult.BUY, VoteResult.SELL]:
            return None
        
        # Get winning strategies based on vote result
        if vote_result == VoteResult.BUY:
            winning_outputs = [s for s in strategy_outputs if s.action == 'BUY']
        else:
            winning_outputs = [s for s in strategy_outputs if s.action == 'SELL']
        
        if not winning_outputs:
            return None
        
        # Extract data for combination
        entries = []
        stop_losses = []
        take_profits = []
        weights = []
        strategy_names = []
        
        for output in winning_outputs:
            entries.append(output.entry)
            stop_losses.append(output.stop_loss)
            take_profits.append(output.take_profit)
            weights.append(output.weight)
            strategy_names.append(output.strategy_name)
        
        # Remove outliers (optional, improves stability)
        entries, stop_losses, take_profits, weights, strategy_names = self._remove_outliers(
            entries, stop_losses, take_profits, weights, strategy_names
        )
        
        if len(entries) == 0:
            return None
        
        # Calculate weighted averages
        total_weight = sum(weights)
        
        if total_weight == 0:
            # Fallback to simple average if all weights are zero
            avg_entry = np.mean(entries)
            avg_sl = np.mean(stop_losses)
            avg_tp = np.mean(take_profits)
        else:
            avg_entry = sum(e * w for e, w in zip(entries, weights)) / total_weight
            avg_sl = sum(sl * w for sl, w in zip(stop_losses, weights)) / total_weight
            avg_tp = sum(tp * w for tp, w in zip(take_profits, weights)) / total_weight
        
        # Apply safety bounds
        avg_entry, avg_sl, avg_tp = self._apply_safety_bounds(
            avg_entry, avg_sl, avg_tp, vote_result.value
        )
        
        # Calculate risk/reward
        if vote_result == VoteResult.BUY:
            risk = avg_entry - avg_sl
            reward = avg_tp - avg_entry
        else:  # SELL
            risk = avg_sl - avg_entry
            reward = avg_entry - avg_tp
        
        risk_reward = reward / risk if risk > 0 else 0
        
        return CombinedTrade(
            direction=vote_result.value,
            entry=avg_entry,
            stop_loss=avg_sl,
            take_profit=avg_tp,
            risk_reward=risk_reward,
            risk_amount=risk,
            reward_amount=reward,
            strategies_used=strategy_names,
            strategies_count=len(strategy_names),
            total_weight=total_weight
        )
    
    def _remove_outliers(self, entries: List[float], stop_losses: List[float],
                         take_profits: List[float], weights: List[float],
                         names: List[str]) -> Tuple[List[float], List[float], 
                                                    List[float], List[float], List[str]]:
        """
        Remove outliers using z-score method
        
        Args:
            entries, stop_losses, take_profits: Price lists
            weights: Strategy weights
            names: Strategy names
        
        Returns:
            Filtered lists with outliers removed
        """
        if len(entries) < 3:
            return entries, stop_losses, take_profits, weights, names
        
        # Calculate z-scores for entries
        entry_mean = np.mean(entries)
        entry_std = np.std(entries)
        
        if entry_std == 0:
            return entries, stop_losses, take_profits, weights, names
        
        # Keep only entries within threshold
        keep_indices = []
        for i, entry in enumerate(entries):
            z_score = abs(entry - entry_mean) / entry_std
            if z_score <= self.outlier_threshold:
                keep_indices.append(i)
        
        # If we removed too many, keep all
        if len(keep_indices) < max(2, len(entries) // 2):
            return entries, stop_losses, take_profits, weights, names
        
        # Filter lists
        filtered_entries = [entries[i] for i in keep_indices]
        filtered_sl = [stop_losses[i] for i in keep_indices]
        filtered_tp = [take_profits[i] for i in keep_indices]
        filtered_weights = [weights[i] for i in keep_indices]
        filtered_names = [names[i] for i in keep_indices]
        
        return filtered_entries, filtered_sl, filtered_tp, filtered_weights, filtered_names
    
    def _apply_safety_bounds(self, entry: float, sl: float, tp: float, 
                            direction: str) -> Tuple[float, float, float]:
        """
        Apply safety bounds to ensure SL and TP are reasonable
        
        Args:
            entry: Entry price
            sl: Stop loss price
            tp: Take profit price
            direction: 'BUY' or 'SELL'
        
        Returns:
            Adjusted (entry, sl, tp) with bounds applied
        """
        if direction == 'BUY':
            # Ensure SL is below entry
            if sl >= entry:
                sl = entry * (1 - self.min_entry_distance_pct)
            
            # Ensure TP is above entry
            if tp <= entry:
                tp = entry * (1 + self.min_entry_distance_pct)
            
            # Limit max SL distance
            max_sl_distance = entry * self.max_sl_distance_pct
            min_sl = entry - max_sl_distance
            if sl < min_sl:
                sl = min_sl
            
        else:  # SELL
            # Ensure SL is above entry
            if sl <= entry:
                sl = entry * (1 + self.min_entry_distance_pct)
            
            # Ensure TP is below entry
            if tp >= entry:
                tp = entry * (1 - self.min_entry_distance_pct)
            
            # Limit max SL distance
            max_sl_distance = entry * self.max_sl_distance_pct
            max_sl = entry + max_sl_distance
            if sl > max_sl:
                sl = max_sl
        
        return entry, sl, tp
    
    def combine_simple(self, strategy_outputs: List[StrategyOutput],
                       direction: str) -> Optional[CombinedTrade]:
        """
        Simple combination without outlier detection (faster)
        
        Args:
            strategy_outputs: List of strategy outputs (all same direction)
            direction: 'BUY' or 'SELL'
        
        Returns:
            CombinedTrade object
        """
        if not strategy_outputs:
            return None
        
        # Filter by direction
        filtered = [s for s in strategy_outputs if s.action == direction]
        
        if not filtered:
            return None
        
        # Simple weighted average
        total_weight = sum(s.weight for s in filtered)
        
        if total_weight == 0:
            avg_entry = np.mean([s.entry for s in filtered])
            avg_sl = np.mean([s.stop_loss for s in filtered])
            avg_tp = np.mean([s.take_profit for s in filtered])
        else:
            avg_entry = sum(s.entry * s.weight for s in filtered) / total_weight
            avg_sl = sum(s.stop_loss * s.weight for s in filtered) / total_weight
            avg_tp = sum(s.take_profit * s.weight for s in filtered) / total_weight
        
        # Calculate risk/reward
        if direction == 'BUY':
            risk = avg_entry - avg_sl
            reward = avg_tp - avg_entry
        else:
            risk = avg_sl - avg_entry
            reward = avg_entry - avg_tp
        
        risk_reward = reward / risk if risk > 0 else 0
        
        return CombinedTrade(
            direction=direction,
            entry=avg_entry,
            stop_loss=avg_sl,
            take_profit=avg_tp,
            risk_reward=risk_reward,
            risk_amount=risk,
            reward_amount=reward,
            strategies_used=[s.strategy_name for s in filtered],
            strategies_count=len(filtered),
            total_weight=total_weight
        )


class WeightedTradeCombiner(TradeCombiner):
    """
    Enhanced trade combiner with additional features:
    - Dynamic weighting based on strategy confidence
    - Convergence check (how close strategies agree)
    - Position sizing recommendations
    """
    
    def __init__(self, outlier_threshold: float = 2.0,
                 max_sl_distance_pct: float = 0.05,
                 min_entry_distance_pct: float = 0.001,
                 min_convergence_ratio: float = 0.7):
        """
        Initialize Weighted Trade Combiner
        
        Args:
            outlier_threshold: Z-score threshold for outlier removal
            max_sl_distance_pct: Maximum SL distance as percentage
            min_entry_distance_pct: Minimum distance between entry and SL/TP
            min_convergence_ratio: Minimum agreement among entries (0-1)
        """
        super().__init__(outlier_threshold, max_sl_distance_pct, min_entry_distance_pct)
        self.min_convergence_ratio = min_convergence_ratio
    
    def combine_with_convergence(self, strategy_outputs: List[StrategyOutput],
                                 vote_result: VoteResult) -> Tuple[Optional[CombinedTrade], float]:
        """
        Combine trades with convergence check
        
        Returns:
            Tuple of (CombinedTrade, convergence_ratio)
        """
        if vote_result not in [VoteResult.BUY, VoteResult.SELL]:
            return None, 0.0
        
        # Get winning strategies
        if vote_result == VoteResult.BUY:
            winning_outputs = [s for s in strategy_outputs if s.action == 'BUY']
        else:
            winning_outputs = [s for s in strategy_outputs if s.action == 'SELL']
        
        if not winning_outputs:
            return None, 0.0
        
        # Calculate convergence (how close entries are)
        entries = [s.entry for s in winning_outputs]
        entry_mean = np.mean(entries)
        entry_std = np.std(entries) if len(entries) > 1 else 0
        
        # Convergence ratio: 1 - (std/mean)
        if entry_mean > 0:
            convergence = 1 - min(1.0, entry_std / entry_mean)
        else:
            convergence = 0.5
        
        # Check if convergence is sufficient
        if convergence < self.min_convergence_ratio:
            return None, convergence
        
        # Combine trades
        combined = self.combine(strategy_outputs, vote_result, None)
        
        return combined, convergence
    
    def get_position_size_multiplier(self, combined_trade: CombinedTrade,
                                     convergence: float) -> float:
        """
        Calculate position size multiplier based on trade quality
        
        Factors:
        - Risk/Reward ratio (higher RR = larger position)
        - Convergence (higher agreement = larger position)
        - Number of strategies (more confirmation = larger position)
        
        Returns:
            Multiplier between 0.5 and 1.5
        """
        # Base multiplier
        multiplier = 1.0
        
        # Adjust by RR (1.5 RR = 1x, 2.0 RR = 1.1x, 3.0 RR = 1.2x)
        if combined_trade.risk_reward > 2.0:
            multiplier *= 1 + (combined_trade.risk_reward - 2.0) * 0.1
            multiplier = min(1.2, multiplier)
        elif combined_trade.risk_reward < 1.5:
            multiplier *= 0.8
        
        # Adjust by convergence
        multiplier *= 0.8 + (convergence * 0.4)  # 0.8 to 1.2
        
        # Adjust by number of strategies
        if combined_trade.strategies_count >= 4:
            multiplier *= 1.1
        elif combined_trade.strategies_count <= 2:
            multiplier *= 0.9
        
        # Clamp to range
        return max(0.5, min(1.5, multiplier))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_median_trade(strategy_outputs: List[StrategyOutput]) -> Optional[CombinedTrade]:
    """
    Calculate median trade setup (alternative to weighted average)
    More robust against outliers but less influenced by weights
    
    Args:
        strategy_outputs: List of strategy outputs (same direction)
    
    Returns:
        CombinedTrade with median values
    """
    if not strategy_outputs:
        return None
    
    direction = strategy_outputs[0].action
    
    entries = [s.entry for s in strategy_outputs]
    stop_losses = [s.stop_loss for s in strategy_outputs]
    take_profits = [s.take_profit for s in strategy_outputs]
    
    median_entry = np.median(entries)
    median_sl = np.median(stop_losses)
    median_tp = np.median(take_profits)
    
    # Calculate risk/reward
    if direction == 'BUY':
        risk = median_entry - median_sl
        reward = median_tp - median_entry
    else:
        risk = median_sl - median_entry
        reward = median_entry - median_tp
    
    risk_reward = reward / risk if risk > 0 else 0
    
    return CombinedTrade(
        direction=direction,
        entry=median_entry,
        stop_loss=median_sl,
        take_profit=median_tp,
        risk_reward=risk_reward,
        risk_amount=risk,
        reward_amount=reward,
        strategies_used=[s.strategy_name for s in strategy_outputs],
        strategies_count=len(strategy_outputs),
        total_weight=len(strategy_outputs)  # Not weighted
    )


def calculate_trimmed_mean_trade(strategy_outputs: List[StrategyOutput],
                                 trim_percent: float = 0.1) -> Optional[CombinedTrade]:
    """
    Calculate trimmed mean trade (remove top/bottom x%)
    
    Args:
        strategy_outputs: List of strategy outputs (same direction)
        trim_percent: Percentage to trim from each end (0-0.5)
    
    Returns:
        CombinedTrade with trimmed mean
    """
    if not strategy_outputs or len(strategy_outputs) < 3:
        return calculate_median_trade(strategy_outputs)
    
    direction = strategy_outputs[0].action
    
    entries = sorted([s.entry for s in strategy_outputs])
    stop_losses = sorted([s.stop_loss for s in strategy_outputs])
    take_profits = sorted([s.take_profit for s in strategy_outputs])
    
    trim_count = int(len(entries) * trim_percent)
    
    if trim_count > 0:
        entries = entries[trim_count:-trim_count]
        stop_losses = stop_losses[trim_count:-trim_count]
        take_profits = take_profits[trim_count:-trim_count]
    
    trimmed_entry = np.mean(entries)
    trimmed_sl = np.mean(stop_losses)
    trimmed_tp = np.mean(take_profits)
    
    # Calculate risk/reward
    if direction == 'BUY':
        risk = trimmed_entry - trimmed_sl
        reward = trimmed_tp - trimmed_entry
    else:
        risk = trimmed_sl - trimmed_entry
        reward = trimmed_entry - trimmed_tp
    
    risk_reward = reward / risk if risk > 0 else 0
    
    return CombinedTrade(
        direction=direction,
        entry=trimmed_entry,
        stop_loss=trimmed_sl,
        take_profit=trimmed_tp,
        risk_reward=risk_reward,
        risk_amount=risk,
        reward_amount=reward,
        strategies_used=[s.strategy_name for s in strategy_outputs],
        strategies_count=len(strategy_outputs),
        total_weight=len(strategy_outputs)
    )