"""
expert_weight_manager.py
Dynamic Weight Management for 5 Experts

Features:
- Track performance of each expert (wins/losses, RR achieved)
- Calculate performance score = win_rate × avg_rr
- Update weights based on performance relative to target
- Persist weights to file for recovery
- Weight decay to prevent extreme values
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExpertPerformance:
    """Performance tracking for a single expert"""
    expert_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_rr_achieved: float = 0.0
    avg_rr: float = 0.0
    win_rate: float = 0.0
    performance_score: float = 0.0
    
    # Recent performance (last 20 trades for faster adaptation)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))

    def sync_with_config(self):
        """Sync initial weights from config.py"""
        try:
            from config import Config
            if hasattr(Config, 'EXPERT_INITIAL_WEIGHTS'):
                self.config.initial_weights = Config.EXPERT_INITIAL_WEIGHTS
                self.weights = Config.EXPERT_INITIAL_WEIGHTS.copy()
                self._init_performance()
        except ImportError:
            pass
    
    def update(self, won: bool, risk_reward: float):
        """Update performance with a new trade outcome"""
        self.total_trades += 1
        self.total_rr_achieved += risk_reward
        
        if won:
            self.wins += 1
            self.recent_trades.append(1)  # Win
        else:
            self.losses += 1
            self.recent_trades.append(0)  # Loss
        
        # Recalculate metrics
        self.win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0.0
        self.avg_rr = self.total_rr_achieved / self.wins if self.wins > 0 else 0.0
        
        # Calculate performance score
        self.performance_score = self.win_rate * self.avg_rr if self.avg_rr > 0 else self.win_rate
    
    def get_recent_win_rate(self, lookback: int = 20) -> float:
        """Get win rate from recent trades only"""
        recent = list(self.recent_trades)[-lookback:]
        if not recent:
            return self.win_rate
        return sum(recent) / len(recent)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'expert_name': self.expert_name,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(self.win_rate, 3),
            'avg_rr': round(self.avg_rr, 2),
            'performance_score': round(self.performance_score, 3),
            'recent_trades': list(self.recent_trades)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpertPerformance':
        """Create from dictionary"""
        perf = cls(
            expert_name=data['expert_name'],
            total_trades=data['total_trades'],
            wins=data['wins'],
            losses=data['losses'],
            total_rr_achieved=data.get('total_rr_achieved', 0),
            avg_rr=data.get('avg_rr', 0),
            win_rate=data.get('win_rate', 0),
            performance_score=data.get('performance_score', 0)
        )
        perf.recent_trades = deque(data.get('recent_trades', []), maxlen=20)
        return perf


@dataclass
class WeightManagerConfig:
    """Configuration for weight manager"""
    
    # Initial weights
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        'pattern_v3': 1.25,
        'price_action': 1.20,
        'smc': 1.30,
        'technical': 1.15,
        'strategy': 1.10
    })
    
    # Weight limits
    min_weight: float = 0.5
    max_weight: float = 2.0
    
    # Target performance score (win_rate × avg_rr)
    target_score: float = 0.75
    
    # Update settings
    update_after_trades: int = 10          # Minimum trades before updating
    update_interval_hours: int = 24         # Force update every X hours
    
    # Weight smoothing (to prevent wild swings)
    smoothing_factor: float = 0.3          # New weight = (1 - α) × old + α × calculated
    
    # Weight decay (gradually return to base)
    decay_rate: float = 0.01               # 1% per update toward initial
    
    # Performance history
    max_history_per_expert: int = 100
    
    # Persistence
    save_file: str = "expert_weights.json"
    auto_save: bool = True


# ============================================================================
# MAIN WEIGHT MANAGER CLASS
# ============================================================================

class ExpertWeightManager:
    """
    Manages dynamic weights for 5 experts based on performance
    
    Weight Formula:
        performance_score = win_rate × avg_rr
        weight_factor = performance_score / target_score
        new_weight = clamp(weight_factor × base_weight)
        smoothed = (1 - α) × old_weight + α × new_weight
    
    Features:
    - Tracks each expert's performance
    - Updates weights after each trade (with smoothing)
    - Persists to file for recovery
    - Provides weight recommendations
    """
    
    def __init__(self, config: WeightManagerConfig = None):
        """Initialize weight manager"""
        self.config = config or WeightManagerConfig()
        
        # Performance tracking
        self.performance: Dict[str, ExpertPerformance] = {}
        self._init_performance()
        
        # Current weights (start with initial)
        self.weights: Dict[str, float] = self.config.initial_weights.copy()
        
        # Last update timestamp
        self.last_update: datetime = datetime.now()
        
        # Load persisted data if exists
        self.load()
        
        # Track if any updates were made
        self._updated_since_last_save = False
    
    def _init_performance(self):
        """Initialize performance tracking for all experts"""
        for name in self.config.initial_weights.keys():
            self.performance[name] = ExpertPerformance(expert_name=name)
    
    # ========================================================================
    # CORE METHODS
    # ========================================================================
    
    def record_trade(self, expert_name: str, won: bool, risk_reward: float):
        """
        Record a trade outcome for an expert
        
        Args:
            expert_name: Name of the expert
            won: Whether the trade was profitable
            risk_reward: Achieved risk/reward ratio
        """
        if expert_name not in self.performance:
            return
        
        # Update performance
        self.performance[expert_name].update(won, risk_reward)
        
        # Update weight based on new performance
        self._update_single_weight(expert_name)
        
        self._updated_since_last_save = True
        
        # Auto-save if enabled
        if self.config.auto_save:
            self.save()
    
    def record_batch_trades(self, trades: List[Tuple[str, bool, float]]):
        """
        Record multiple trades at once
        
        Args:
            trades: List of (expert_name, won, risk_reward)
        """
        for expert_name, won, rr in trades:
            self.record_trade(expert_name, won, rr)
    
    def get_weight(self, expert_name: str) -> float:
        """Get current weight for an expert"""
        return self.weights.get(expert_name, 1.0)
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get all current weights"""
        return self.weights.copy()
    
    def get_performance(self, expert_name: str) -> Optional[ExpertPerformance]:
        """Get performance data for an expert"""
        return self.performance.get(expert_name)
    
    def get_all_performance(self) -> Dict[str, ExpertPerformance]:
        """Get all performance data"""
        return self.performance.copy()
    
    def update_all_weights(self):
        """Update weights for all experts based on current performance"""
        for expert_name in self.weights.keys():
            self._update_single_weight(expert_name)
        
        self.last_update = datetime.now()
        self._updated_since_last_save = True
    
    def _update_single_weight(self, expert_name: str):
        """Update weight for a single expert"""
        perf = self.performance.get(expert_name)
        if not perf:
            return
        
        # Only update after minimum trades
        if perf.total_trades < self.config.update_after_trades:
            return
        
        # Calculate performance score
        score = perf.performance_score
        
        # Calculate target weight
        if self.config.target_score > 0:
            target_factor = score / self.config.target_score
        else:
            target_factor = 1.0
        
        base_weight = self.config.initial_weights.get(expert_name, 1.0)
        calculated_weight = base_weight * target_factor
        
        # Clamp to limits
        calculated_weight = max(self.config.min_weight, 
                               min(self.config.max_weight, calculated_weight))
        
        # Apply smoothing
        old_weight = self.weights.get(expert_name, base_weight)
        smoothed_weight = (1 - self.config.smoothing_factor) * old_weight + \
                          self.config.smoothing_factor * calculated_weight
        
        # Apply decay toward initial weight
        decayed_weight = smoothed_weight * (1 - self.config.decay_rate) + \
                         base_weight * self.config.decay_rate
        
        # Final weight
        new_weight = max(self.config.min_weight, 
                        min(self.config.max_weight, decayed_weight))
        
        self.weights[expert_name] = round(new_weight, 3)
    
    # ========================================================================
    # WEIGHT RECOMMENDATIONS
    # ========================================================================
    
    def get_recommendations(self) -> Dict[str, Dict]:
        """
        Get weight recommendations for all experts
        Returns dict with current weight, recommended weight, and reasoning
        """
        recommendations = {}
        
        for expert_name, perf in self.performance.items():
            current = self.weights.get(expert_name, 1.0)
            
            # Calculate recommendation
            if perf.total_trades < self.config.update_after_trades:
                recommended = current
                reason = f"Insufficient data ({perf.total_trades} trades)"
            else:
                score = perf.performance_score
                base = self.config.initial_weights.get(expert_name, 1.0)
                
                if score > self.config.target_score * 1.2:
                    recommended = min(self.config.max_weight, current * 1.1)
                    reason = f"Overperforming (score: {score:.2f} > target)"
                elif score < self.config.target_score * 0.8:
                    recommended = max(self.config.min_weight, current * 0.9)
                    reason = f"Underperforming (score: {score:.2f} < target)"
                else:
                    recommended = current
                    reason = "Performance meets target"
            
            recommendations[expert_name] = {
                'current_weight': round(current, 3),
                'recommended_weight': round(recommended, 3),
                'performance_score': round(perf.performance_score, 3),
                'win_rate': round(perf.win_rate, 3),
                'avg_rr': round(perf.avg_rr, 2),
                'total_trades': perf.total_trades,
                'reason': reason
            }
        
        return recommendations
    
    def apply_recommendations(self):
        """Apply recommended weight adjustments"""
        recommendations = self.get_recommendations()
        for expert_name, rec in recommendations.items():
            self.weights[expert_name] = rec['recommended_weight']
        
        self.last_update = datetime.now()
        self._updated_since_last_save = True
    
    # ========================================================================
    # WEIGHT VALIDATION & NORMALIZATION
    # ========================================================================
    
    def normalize_weights(self) -> Dict[str, float]:
        """
        Normalize weights so they sum to a target (optional)
        This ensures total influence is consistent
        """
        total = sum(self.weights.values())
        if total == 0:
            return self.weights
        
        # Normalize to sum of initial weights
        target_total = sum(self.config.initial_weights.values())
        factor = target_total / total
        
        normalized = {k: round(v * factor, 3) for k, v in self.weights.items()}
        
        # Re-clamp to limits
        for k, v in normalized.items():
            normalized[k] = max(self.config.min_weight, 
                               min(self.config.max_weight, v))
        
        self.weights = normalized
        return self.weights
    
    def reset_weights(self):
        """Reset all weights to initial values"""
        self.weights = self.config.initial_weights.copy()
        self._updated_since_last_save = True
    
    def reset_performance(self, expert_name: str = None):
        """Reset performance data for an expert or all experts"""
        if expert_name:
            if expert_name in self.performance:
                self.performance[expert_name] = ExpertPerformance(expert_name=expert_name)
                self._update_single_weight(expert_name)
        else:
            self._init_performance()
            self.update_all_weights()
        
        self._updated_since_last_save = True
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self, filepath: str = None):
        """Save weights and performance to file"""
        if filepath is None:
            filepath = self.config.save_file
        
        try:
            data = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'weights': self.weights,
                'config': {
                    'min_weight': self.config.min_weight,
                    'max_weight': self.config.max_weight,
                    'target_score': self.config.target_score
                },
                'performance': {
                    name: perf.to_dict() 
                    for name, perf in self.performance.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self._updated_since_last_save = False
            
        except Exception as e:
            print(f"Error saving expert weights: {e}")
    
    def load(self, filepath: str = None):
        """Load weights and performance from file"""
        if filepath is None:
            filepath = self.config.save_file
        
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load weights
            if 'weights' in data:
                self.weights = data['weights']
            
            # Load performance
            if 'performance' in data:
                for name, perf_data in data['performance'].items():
                    if name in self.performance:
                        self.performance[name] = ExpertPerformance.from_dict(perf_data)
            
            self.last_update = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
            
        except Exception as e:
            print(f"Error loading expert weights: {e}")
    
    # ========================================================================
    # STATISTICS & REPORTING
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights,
            'total_trades': sum(p.total_trades for p in self.performance.values()),
            'experts': {}
        }
        
        for name, perf in self.performance.items():
            stats['experts'][name] = {
                'total_trades': perf.total_trades,
                'wins': perf.wins,
                'losses': perf.losses,
                'win_rate': round(perf.win_rate, 3),
                'avg_rr': round(perf.avg_rr, 2),
                'performance_score': round(perf.performance_score, 3),
                'current_weight': self.weights.get(name, 1.0)
            }
        
        return stats
    
    def get_performance_summary(self) -> str:
        """Get human-readable performance summary"""
        lines = []
        lines.append("=" * 70)
        lines.append("EXPERT PERFORMANCE SUMMARY")
        lines.append("=" * 70)
        lines.append(f"{'Expert':<20} {'Trades':<8} {'Win%':<8} {'Avg RR':<8} {'Weight':<8}")
        lines.append("-" * 70)
        
        for name, perf in sorted(self.performance.items(), 
                                 key=lambda x: x[1].performance_score, 
                                 reverse=True):
            lines.append(f"{name:<20} {perf.total_trades:<8} "
                        f"{perf.win_rate*100:.0f}%{'':<4} "
                        f"{perf.avg_rr:<8.2f} "
                        f"{self.weights.get(name, 1.0):<8.2f}")
        
        lines.append("-" * 70)
        lines.append(f"Total Trades: {sum(p.total_trades for p in self.performance.values())}")
        lines.append(f"Weight Range: {self.config.min_weight} - {self.config.max_weight}")
        lines.append(f"Target Score: {self.config.target_score}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print performance summary to console"""
        print(self.get_performance_summary())


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_weight_manager(initial_weights: Dict[str, float] = None,
                         min_weight: float = 0.5,
                         max_weight: float = 2.0,
                         target_score: float = 0.75) -> ExpertWeightManager:
    """Create a weight manager with custom settings"""
    config = WeightManagerConfig(
        initial_weights=initial_weights or {
            'pattern_v3': 1.25,
            'price_action': 1.20,
            'smc': 1.30,
            'technical': 1.15,
            'strategy': 1.10
        },
        min_weight=min_weight,
        max_weight=max_weight,
        target_score=target_score
    )
    return ExpertWeightManager(config)


def get_initial_weights() -> Dict[str, float]:
    """Get default initial weights"""
    return {
        'pattern_v3': 1.25,
        'price_action': 1.20,
        'smc': 1.30,
        'technical': 1.15,
        'strategy': 1.10
    }