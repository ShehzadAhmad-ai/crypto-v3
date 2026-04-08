"""
Weight Manager for Strategy Expert
Dynamically adjusts strategy weights based on historical performance
Weight formula: weight = base_weight × (win_rate × avg_rr) / target_score
Range: 0.5 to 2.0
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque

from strategy_expert.strategy_config import get_pipeline_config, StrategyExpertConfigLoader


@dataclass
class TradeRecord:
    """Record of a single trade executed by a strategy"""
    strategy_name: str
    timestamp: datetime
    action: str                    # 'BUY' or 'SELL'
    entry: float
    exit_price: float
    pnl: float                     # Profit/Loss in percent
    risk_reward: float             # Actual RR achieved
    won: bool                      # True if profitable
    confidence_at_entry: float
    signal_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'entry': self.entry,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'risk_reward': self.risk_reward,
            'won': self.won,
            'confidence_at_entry': self.confidence_at_entry,
            'signal_id': self.signal_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary"""
        return cls(
            strategy_name=data['strategy_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            action=data['action'],
            entry=data['entry'],
            exit_price=data['exit_price'],
            pnl=data['pnl'],
            risk_reward=data['risk_reward'],
            won=data['won'],
            confidence_at_entry=data['confidence_at_entry'],
            signal_id=data.get('signal_id', '')
        )


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy"""
    strategy_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_rr: float = 0.0
    win_rate: float = 0.0
    current_weight: float = 1.0
    base_weight: float = 1.0
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, trade: TradeRecord):
        """Update performance with a new trade"""
        self.total_trades += 1
        self.total_pnl += trade.pnl
        self.avg_pnl = self.total_pnl / self.total_trades
        
        if trade.won:
            self.wins += 1
        else:
            self.losses += 1
        
        self.win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0
        
        # Update average risk/reward (only for winning trades that achieved target)
        if trade.won and trade.risk_reward > 0:
            # Rolling average of last 100 RR values
            self.recent_trades.append(trade.risk_reward)
            self.avg_rr = sum(self.recent_trades) / len(self.recent_trades)
        elif not trade.won and len(self.recent_trades) > 0:
            # Keep existing avg_rr, don't include losing trades
            pass
    
    def calculate_score(self, target_score: float = 0.75) -> float:
        """Calculate performance score (win_rate × avg_rr)"""
        if self.total_trades < 5:
            # Not enough data, return base score
            return target_score
        
        # Use avg_rr from recent trades or default to 1.5 if no winning trades
        rr = self.avg_rr if self.avg_rr > 0 else 1.5
        
        # Score = win_rate × avg_rr
        score = self.win_rate * rr
        
        return max(0.3, min(1.5, score))  # Clamp between 0.3 and 1.5
    
    def get_new_weight(self, target_score: float = 0.75, 
                       min_weight: float = 0.5, 
                       max_weight: float = 2.0) -> float:
        """Calculate new weight based on performance"""
        if self.total_trades < 5:
            # Not enough data, use base weight
            return self.base_weight
        
        score = self.calculate_score(target_score)
        
        # weight = base_weight × (score / target_score)
        new_weight = self.base_weight * (score / target_score)
        
        # Clamp to range
        new_weight = max(min_weight, min(max_weight, new_weight))
        
        self.current_weight = new_weight
        return new_weight
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'strategy_name': self.strategy_name,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'total_pnl': self.total_pnl,
            'avg_pnl': self.avg_pnl,
            'avg_rr': self.avg_rr,
            'win_rate': self.win_rate,
            'current_weight': self.current_weight,
            'base_weight': self.base_weight
        }


class WeightManager:
    """
    Manages dynamic weights for all strategies
    Weights are updated based on historical performance (win_rate × avg_rr)
    """
    
    def __init__(self, config_loader: StrategyExpertConfigLoader = None,
                 save_file: str = "strategy_weights.json"):
        """
        Initialize Weight Manager
        
        Args:
            config_loader: Configuration loader instance
            save_file: File path to persist weight data
        """
        self.config = config_loader or StrategyExpertConfigLoader()
        self.pipeline_config = self.config.config.pipeline
        
        # Weight limits
        self.min_weight = self.pipeline_config.weight_min
        self.max_weight = self.pipeline_config.weight_max
        self.target_score = self.pipeline_config.weight_target_score
        self.history_length = self.pipeline_config.weight_history_length
        
        # Performance tracking
        self.performance: Dict[str, StrategyPerformance] = {}
        self.trade_history: List[TradeRecord] = []
        self.save_file = save_file
        
        # Initialize with base weights from config
        self._initialize_weights()
        
        # Load persisted data if exists
        self.load()
    
    def _initialize_weights(self):
        """Initialize performance tracking for all strategies"""
        for strategy_name in self.config.get_enabled_strategies():
            base_weight = self.config.get_strategy_weight(strategy_name)
            
            self.performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name,
                base_weight=base_weight,
                current_weight=base_weight
            )
    
    def record_trade(self, trade: TradeRecord):
        """
        Record a completed trade for a strategy
        
        Args:
            trade: TradeRecord with trade outcome
        """
        # Add to performance
        if trade.strategy_name in self.performance:
            self.performance[trade.strategy_name].update(trade)
        
        # Add to global history
        self.trade_history.append(trade)
        
        # Trim history if needed
        if len(self.trade_history) > self.history_length * 10:
            self.trade_history = self.trade_history[-self.history_length * 10:]
        
        # Update weights after recording
        self.update_all_weights()
        
        # Auto-save
        self.save()
    
    def update_weight(self, strategy_name: str) -> float:
        """
        Update weight for a single strategy based on performance
        
        Returns:
            New weight value
        """
        if strategy_name not in self.performance:
            base_weight = self.config.get_strategy_weight(strategy_name)
            self.performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name,
                base_weight=base_weight,
                current_weight=base_weight
            )
        
        perf = self.performance[strategy_name]
        new_weight = perf.get_new_weight(
            target_score=self.target_score,
            min_weight=self.min_weight,
            max_weight=self.max_weight
        )
        
        return new_weight
    
    def update_all_weights(self) -> Dict[str, float]:
        """
        Update weights for all strategies
        
        Returns:
            Dict of strategy_name -> new_weight
        """
        updated_weights = {}
        
        for strategy_name in self.performance:
            new_weight = self.update_weight(strategy_name)
            updated_weights[strategy_name] = new_weight
        
        return updated_weights
    
    def get_weight(self, strategy_name: str) -> float:
        """
        Get current weight for a strategy
        
        Args:
            strategy_name: Name of the strategy
        
        Returns:
            Current weight (default 1.0 if not found)
        """
        if strategy_name in self.performance:
            return self.performance[strategy_name].current_weight
        
        # Fallback to config base weight
        return self.config.get_strategy_weight(strategy_name)
    
    def get_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy"""
        return self.performance.get(strategy_name)
    
    def get_all_performance(self) -> Dict[str, StrategyPerformance]:
        """Get performance metrics for all strategies"""
        return self.performance
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all strategy performances"""
        summary = {
            'total_strategies': len(self.performance),
            'total_trades': len(self.trade_history),
            'strategies': {}
        }
        
        for name, perf in self.performance.items():
            summary['strategies'][name] = perf.to_dict()
        
        return summary
    
    def get_top_strategies(self, limit: int = None) -> List[Tuple[str, float]]:
        """
        Get top performing strategies by weight
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of (strategy_name, weight) sorted by weight descending
        """
        if limit is None:
            limit = self.pipeline_config.max_strategies_to_use
        
        sorted_strategies = sorted(
            self.performance.items(),
            key=lambda x: x[1].current_weight,
            reverse=True
        )
        
        return [(name, perf.current_weight) for name, perf in sorted_strategies[:limit]]
    
    def get_bottom_strategies(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get worst performing strategies by weight"""
        sorted_strategies = sorted(
            self.performance.items(),
            key=lambda x: x[1].current_weight
        )
        
        return [(name, perf.current_weight) for name, perf in sorted_strategies[:limit]]
    
    def reset_performance(self, strategy_name: str = None):
        """
        Reset performance data for a strategy or all strategies
        
        Args:
            strategy_name: Specific strategy to reset, or None for all
        """
        if strategy_name:
            if strategy_name in self.performance:
                base_weight = self.config.get_strategy_weight(strategy_name)
                self.performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    base_weight=base_weight,
                    current_weight=base_weight
                )
        else:
            self._initialize_weights()
            self.trade_history = []
    
    def save(self):
        """Persist weight and performance data to file"""
        try:
            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'target_score': self.target_score,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight,
                'performance': {},
                'trade_history': [t.to_dict() for t in self.trade_history[-1000:]]
            }
            
            for name, perf in self.performance.items():
                data['performance'][name] = {
                    'total_trades': perf.total_trades,
                    'wins': perf.wins,
                    'losses': perf.losses,
                    'total_pnl': perf.total_pnl,
                    'avg_rr': perf.avg_rr,
                    'win_rate': perf.win_rate,
                    'current_weight': perf.current_weight,
                    'base_weight': perf.base_weight,
                    'recent_trades': list(perf.recent_trades)
                }
            
            # Write to file
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving weight data: {e}")
    
    def load(self):
        """Load persisted weight and performance data from file"""
        if not os.path.exists(self.save_file):
            return
        
        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
            
            # Load performance data
            for name, perf_data in data.get('performance', {}).items():
                if name in self.performance:
                    perf = self.performance[name]
                    perf.total_trades = perf_data.get('total_trades', 0)
                    perf.wins = perf_data.get('wins', 0)
                    perf.losses = perf_data.get('losses', 0)
                    perf.total_pnl = perf_data.get('total_pnl', 0.0)
                    perf.avg_rr = perf_data.get('avg_rr', 0.0)
                    perf.win_rate = perf_data.get('win_rate', 0.0)
                    perf.current_weight = perf_data.get('current_weight', perf.base_weight)
                    perf.recent_trades = deque(perf_data.get('recent_trades', []), maxlen=self.history_length)
            
            # Load trade history
            self.trade_history = [
                TradeRecord.from_dict(t) for t in data.get('trade_history', [])
            ]
            
        except Exception as e:
            print(f"Error loading weight data: {e}")
    
    def print_summary(self):
        """Print performance summary to console"""
        print("\n" + "="*70)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("="*70)
        print(f"{'Strategy':<35} {'Trades':<8} {'Win%':<8} {'Avg RR':<8} {'Weight':<8}")
        print("-"*70)
        
        for name, perf in sorted(self.performance.items(), 
                                 key=lambda x: x[1].current_weight, 
                                 reverse=True):
            print(f"{name[:34]:<35} {perf.total_trades:<8} "
                  f"{perf.win_rate*100:.1f}%{'':<4} "
                  f"{perf.avg_rr:<8.2f} "
                  f"{perf.current_weight:<8.2f}")
        
        print("="*70)
        print(f"Total Trades: {len(self.trade_history)}")
        print(f"Active Strategies: {len(self.performance)}")
        print("="*70)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_trade_pnl(entry: float, exit_price: float, action: str) -> float:
    """
    Calculate PnL percentage for a trade
    
    Args:
        entry: Entry price
        exit_price: Exit price
        action: 'BUY' or 'SELL'
    
    Returns:
        PnL percentage (positive for profit, negative for loss)
    """
    if action == 'BUY':
        return ((exit_price - entry) / entry) * 100
    else:  # SELL
        return ((entry - exit_price) / entry) * 100


def calculate_achieved_rr(entry: float, exit_price: float, 
                          stop_loss: float, action: str) -> float:
    """
    Calculate achieved risk/reward ratio for a completed trade
    
    Args:
        entry: Entry price
        exit_price: Exit price
        stop_loss: Stop loss price
        action: 'BUY' or 'SELL'
    
    Returns:
        Achieved risk/reward ratio (0 if stopped out)
    """
    if action == 'BUY':
        risk = entry - stop_loss
        reward = exit_price - entry
    else:  # SELL
        risk = stop_loss - entry
        reward = entry - exit_price
    
    if risk <= 0:
        return 0.0
    
    # If stopped out, reward is negative (loss)
    if reward <= 0:
        return -abs(reward) / risk
    
    return reward / risk