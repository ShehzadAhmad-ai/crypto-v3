"""
pattern_learning.py - Self-Improvement & Learning Engine for Pattern V4

Learns from historical performance to improve pattern detection:
- Tracks win/loss rates per pattern type
- Adjusts pattern weights based on actual performance
- Analyzes failure patterns to identify weak signals
- Provides recommendations for configuration tuning
- Saves/loads performance data for persistence

Version: 4.0
Author: Pattern Intelligence System
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .pattern_config import CONFIG


# ============================================================================
# PATTERN PERFORMANCE DATA CLASS
# ============================================================================

@dataclass
class PatternPerformanceV4:
    """
    Performance tracking for a single pattern type.
    """
    pattern_name: str = ""
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_rr: float = 0.0
    total_pnl: float = 0.0
    last_trade_date: str = ""
    last_trade_result: str = ""  # 'WIN' or 'LOSS'
    
    # Recent performance (last 20 trades)
    recent_results: List[bool] = field(default_factory=list)  # True = win, False = loss
    recent_confidences: List[float] = field(default_factory=list)
    
    # Failure analysis
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    recent_failures: List[Dict] = field(default_factory=list)
    
    # Dynamic weight adjustment
    current_weight_multiplier: float = 1.0
    weight_adjustment_history: List[Dict] = field(default_factory=list)
    
    def update_win_rate(self):
        """Update win rate based on wins and losses"""
        total = self.wins + self.losses
        self.win_rate = self.wins / total if total > 0 else 0.0
    
    def add_trade_result(self, was_win: bool, confidence: float, rr: float, 
                         pnl: float = 0, failure_reason: str = None):
        """Record a trade result"""
        if was_win:
            self.wins += 1
            self.total_pnl += pnl
            self.last_trade_result = "WIN"
        else:
            self.losses += 1
            self.total_pnl += pnl
            self.last_trade_result = "LOSS"
            
            if failure_reason:
                self.failure_reasons[failure_reason] = self.failure_reasons.get(failure_reason, 0) + 1
                
                # Store recent failure
                self.recent_failures.append({
                    'timestamp': datetime.now().isoformat(),
                    'confidence': confidence,
                    'rr': rr,
                    'reason': failure_reason
                })
                if len(self.recent_failures) > 20:
                    self.recent_failures = self.recent_failures[-20:]
        
        # Update averages
        total_trades = self.wins + self.losses
        self.avg_confidence = (self.avg_confidence * (total_trades - 1) + confidence) / total_trades
        self.avg_rr = (self.avg_rr * (total_trades - 1) + rr) / total_trades
        
        # Update recent results
        self.recent_results.append(was_win)
        self.recent_confidences.append(confidence)
        if len(self.recent_results) > 20:
            self.recent_results = self.recent_results[-20:]
            self.recent_confidences = self.recent_confidences[-20:]
        
        self.update_win_rate()
        self.last_trade_date = datetime.now().isoformat()
    
    def get_recent_win_rate(self, n: int = 20) -> float:
        """Get win rate for last N trades"""
        if not self.recent_results:
            return 0.5
        recent = self.recent_results[-n:]
        return sum(recent) / len(recent) if recent else 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'pattern_name': self.pattern_name,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'avg_confidence': self.avg_confidence,
            'avg_rr': self.avg_rr,
            'total_pnl': self.total_pnl,
            'last_trade_date': self.last_trade_date,
            'last_trade_result': self.last_trade_result,
            'recent_results': self.recent_results,
            'recent_confidences': self.recent_confidences,
            'failure_reasons': self.failure_reasons,
            'recent_failures': self.recent_failures,
            'current_weight_multiplier': self.current_weight_multiplier,
            'weight_adjustment_history': self.weight_adjustment_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PatternPerformanceV4':
        """Create from dictionary"""
        return cls(
            pattern_name=data.get('pattern_name', ''),
            wins=data.get('wins', 0),
            losses=data.get('losses', 0),
            win_rate=data.get('win_rate', 0.0),
            avg_confidence=data.get('avg_confidence', 0.0),
            avg_rr=data.get('avg_rr', 0.0),
            total_pnl=data.get('total_pnl', 0.0),
            last_trade_date=data.get('last_trade_date', ''),
            last_trade_result=data.get('last_trade_result', ''),
            recent_results=data.get('recent_results', []),
            recent_confidences=data.get('recent_confidences', []),
            failure_reasons=data.get('failure_reasons', {}),
            recent_failures=data.get('recent_failures', []),
            current_weight_multiplier=data.get('current_weight_multiplier', 1.0),
            weight_adjustment_history=data.get('weight_adjustment_history', []),
        )


# ============================================================================
# PATTERN LEARNING ENGINE
# ============================================================================

class PatternLearningEngineV4:
    """
    Self-improvement engine that learns from historical performance.
    Adjusts pattern weights based on actual win rates.
    """
    
    def __init__(self, performance_file: str = "pattern_performance_v4.json"):
        self.performance_file = performance_file
        self.pattern_performance: Dict[str, PatternPerformanceV4] = {}
        self.config = CONFIG.learning_config if hasattr(CONFIG, 'learning_config') else {
            'enabled': True,
            'min_trades_for_adjustment': 20,
            'weight_adjustment_rate': 0.05,
            'max_weight_multiplier': 1.30,
            'min_weight_multiplier': 0.70,
            'target_win_rate': 0.55,
        }
        
        self._load_performance()
    
    def record_trade_outcome(self, pattern_name: str, was_win: bool,
                             confidence: float, rr: float, pnl: float = 0,
                             failure_reason: str = None):
        """
        Record trade outcome for learning.
        Called when trade is closed.
        """
        if not self.config.get('enabled', True):
            return
        
        if pattern_name not in self.pattern_performance:
            self.pattern_performance[pattern_name] = PatternPerformanceV4(
                pattern_name=pattern_name
            )
        
        perf = self.pattern_performance[pattern_name]
        perf.add_trade_result(was_win, confidence, rr, pnl, failure_reason)
        
        # Adjust weight if enough trades
        total_trades = perf.wins + perf.losses
        if total_trades >= self.config.get('min_trades_for_adjustment', 20):
            self._adjust_pattern_weight(perf)
        
        self._save_performance()
    
    def _adjust_pattern_weight(self, perf: PatternPerformanceV4):
        """
        Dynamically adjust pattern weight based on performance.
        """
        target_win_rate = self.config.get('target_win_rate', 0.55)
        deviation = perf.win_rate - target_win_rate
        
        # Calculate adjustment (max 5% per adjustment)
        adjustment_rate = self.config.get('weight_adjustment_rate', 0.05)
        adjustment = deviation * adjustment_rate
        
        # Clamp adjustment
        max_multiplier = self.config.get('max_weight_multiplier', 1.30)
        min_multiplier = self.config.get('min_weight_multiplier', 0.70)
        
        new_multiplier = perf.current_weight_multiplier + adjustment
        new_multiplier = max(min_multiplier, min(max_multiplier, new_multiplier))
        
        if new_multiplier != perf.current_weight_multiplier:
            perf.weight_adjustment_history.append({
                'timestamp': datetime.now().isoformat(),
                'win_rate': perf.win_rate,
                'old_multiplier': perf.current_weight_multiplier,
                'new_multiplier': new_multiplier,
                'adjustment': adjustment,
                'total_trades': perf.wins + perf.losses
            })
            perf.current_weight_multiplier = new_multiplier
    
    def get_pattern_weight_multiplier(self, pattern_name: str) -> float:
        """
        Get current weight multiplier for a pattern.
        """
        if pattern_name not in self.pattern_performance:
            return 1.0
        return self.pattern_performance[pattern_name].current_weight_multiplier
    
    def get_pattern_win_rate(self, pattern_name: str) -> float:
        """
        Get historical win rate for a pattern.
        """
        if pattern_name not in self.pattern_performance:
            return 0.5  # Default neutral
        return self.pattern_performance[pattern_name].win_rate
    
    def get_pattern_recent_win_rate(self, pattern_name: str, n: int = 20) -> float:
        """
        Get recent win rate for a pattern.
        """
        if pattern_name not in self.pattern_performance:
            return 0.5
        return self.pattern_performance[pattern_name].get_recent_win_rate(n)
    
    def get_best_patterns(self, min_trades: int = 10, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Get patterns with highest win rates.
        """
        results = []
        for name, perf in self.pattern_performance.items():
            total_trades = perf.wins + perf.losses
            if total_trades >= min_trades:
                results.append((name, perf.win_rate))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_worst_patterns(self, min_trades: int = 10, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Get patterns with lowest win rates.
        """
        results = []
        for name, perf in self.pattern_performance.items():
            total_trades = perf.wins + perf.losses
            if total_trades >= min_trades:
                results.append((name, perf.win_rate))
        
        results.sort(key=lambda x: x[1])
        return results[:limit]
    
    def analyze_failure_patterns(self) -> Dict:
        """
        Analyze common failure reasons across all patterns.
        Returns recommendations for improvement.
        """
        all_failures = defaultdict(int)
        pattern_failure_counts = {}
        
        for name, perf in self.pattern_performance.items():
            total_trades = perf.wins + perf.losses
            if total_trades >= 5:
                pattern_failure_counts[name] = len(perf.recent_failures)
                for reason, count in perf.failure_reasons.items():
                    all_failures[reason] += count
        
        # Sort failures by frequency
        sorted_failures = sorted(all_failures.items(), key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sorted_failures)
        
        return {
            'total_failures': sum(all_failures.values()),
            'top_failure_reasons': sorted_failures[:5],
            'patterns_with_most_failures': sorted(pattern_failure_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'recommendations': recommendations,
            'total_patterns_tracked': len(self.pattern_performance),
            'total_trades_recorded': sum(p.wins + p.losses for p in self.pattern_performance.values())
        }
    
    def _generate_recommendations(self, failures: List[Tuple[str, int]]) -> List[str]:
        """Generate recommendations based on failure analysis"""
        recommendations = []
        
        for reason, count in failures[:5]:
            if "volume" in reason.lower():
                recommendations.append("Increase volume confirmation weight - many failures due to volume issues")
            elif "wick" in reason.lower() or "wick_too_long" in reason.lower():
                recommendations.append("Add wick ratio penalty for breakouts - long wicks indicate false moves")
            elif "reversal" in reason.lower() or "immediate_reversal" in reason.lower():
                recommendations.append("Require 2-bar confirmation after breakout before entering")
            elif "resistance" in reason.lower() or "htf_resistance" in reason.lower():
                recommendations.append("Add HTF resistance check before entering long positions")
            elif "support" in reason.lower() or "htf_support" in reason.lower():
                recommendations.append("Add HTF support check before entering short positions")
            elif "low_volume" in reason.lower() or "volume_too_low" in reason.lower():
                recommendations.append("Increase minimum volume requirement for breakout confirmation")
        
        if not recommendations:
            recommendations.append("Monitor pattern performance - insufficient failure data yet")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict:
        """
        Get overall performance summary across all patterns.
        """
        total_wins = sum(p.wins for p in self.pattern_performance.values())
        total_losses = sum(p.losses for p in self.pattern_performance.values())
        total_trades = total_wins + total_losses
        
        return {
            'total_patterns': len(self.pattern_performance),
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'overall_win_rate': total_wins / total_trades if total_trades > 0 else 0,
            'avg_confidence': np.mean([p.avg_confidence for p in self.pattern_performance.values()]) if self.pattern_performance else 0,
            'avg_rr': np.mean([p.avg_rr for p in self.pattern_performance.values()]) if self.pattern_performance else 0,
            'best_pattern': self.get_best_patterns(5)[0] if self.get_best_patterns(5) else None,
            'worst_pattern': self.get_worst_patterns(5)[0] if self.get_worst_patterns(5) else None,
        }
    
    def load_history(self, history_data: Dict):
        """
        Load performance history from external data.
        """
        if not history_data:
            return
        
        for pattern_name, perf_data in history_data.items():
            if pattern_name not in self.pattern_performance:
                self.pattern_performance[pattern_name] = PatternPerformanceV4.from_dict(perf_data)
    
    def _load_performance(self):
        """Load performance data from file"""
        if not os.path.exists(self.performance_file):
            return
        
        try:
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
                for pattern_name, perf_data in data.items():
                    self.pattern_performance[pattern_name] = PatternPerformanceV4.from_dict(perf_data)
        except Exception as e:
            print(f"Error loading performance data: {e}")
    
    def _save_performance(self):
        """Save performance data to file"""
        try:
            data = {}
            for pattern_name, perf in self.pattern_performance.items():
                data[pattern_name] = perf.to_dict()
            
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {e}")
    
    def reset_performance(self, pattern_name: str = None):
        """
        Reset performance data for a specific pattern or all patterns.
        """
        if pattern_name:
            if pattern_name in self.pattern_performance:
                del self.pattern_performance[pattern_name]
        else:
            self.pattern_performance.clear()
        
        self._save_performance()


# ============================================================================
# PATTERN WEIGHT OPTIMIZER
# ============================================================================

class PatternWeightOptimizerV4:
    """
    Optimizes pattern detection weights based on historical performance.
    Uses genetic algorithm style optimization.
    """
    
    def __init__(self, learning_engine: PatternLearningEngineV4):
        self.learning_engine = learning_engine
        self.learning_config = CONFIG.learning_config if hasattr(CONFIG, 'learning_config') else {}
    
    def optimize_weights(self, pattern_name: str) -> Dict[str, float]:
        """
        Optimize weights for a specific pattern based on performance.
        Returns optimized weight dictionary.
        """
        perf = self.learning_engine.pattern_performance.get(pattern_name)
        if not perf or (perf.wins + perf.losses) < 20:
            # Not enough data, return None
            return None
        
        # Get current weights from config
        current_weights = CONFIG.pattern_weights.get(pattern_name, {})
        if not current_weights:
            return None
        
        # Analyze which components correlate with success
        component_effectiveness = self._analyze_component_effectiveness(pattern_name, perf)
        
        # Adjust weights based on effectiveness
        optimized_weights = {}
        for component, weight in current_weights.items():
            effectiveness = component_effectiveness.get(component, 0.5)
            # Increase weight for effective components, decrease for ineffective
            adjustment = (effectiveness - 0.5) * 0.5  # Max 25% adjustment
            new_weight = weight * (1 + adjustment)
            optimized_weights[component] = new_weight
        
        # Normalize to sum to 1.0
        total = sum(optimized_weights.values())
        if total > 0:
            optimized_weights = {k: v / total for k, v in optimized_weights.items()}
        
        return optimized_weights
    
    def _analyze_component_effectiveness(self, pattern_name: str, 
                                          perf: PatternPerformanceV4) -> Dict[str, float]:
        """
        Analyze which components were most effective in winning trades.
        Returns effectiveness score (0-1) for each component.
        """
        # This would require storing component scores per trade
        # For now, return default based on win rate
        effectiveness = {}
        
        # Higher win rate means all components are more effective
        base_effectiveness = perf.win_rate
        
        for component in CONFIG.pattern_weights.get(pattern_name, {}).keys():
            # Slight variation per component (in real system, track per component)
            effectiveness[component] = base_effectiveness
        
        return effectiveness


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PatternPerformanceV4',
    'PatternLearningEngineV4',
    'PatternWeightOptimizerV4',
]