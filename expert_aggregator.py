"""
expert_aggregator.py - Combines 5 Expert Signals into 1 Combined Signal

Phase 4 Pipeline:
1. Check consensus (minimum experts to agree, from config)
2. Build TP levels (collect all TPs, sort, cluster, assign percentages)
3. Calculate weighted averages (entry, stop loss)
4. Calculate final confidence
5. Output CombinedSignal

Version: 3.0 (Complete Rewrite)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from expert_weight_manager import ExpertWeightManager

from expert_interface import (
    ExpertSignal, CombinedSignal, TPLevel, ConsensusResult,
    is_tradeable_signal, is_full_signal
)


# ============================================================================
# DEFAULT CONFIGURATION (Read from config.py in production)
# ============================================================================

DEFAULT_CONFIG = {
    # Consensus settings
    'min_experts_to_agree': 3,          # Minimum experts that must agree on direction
    'min_experts_with_signal': 3,       # Minimum experts that must generate a signal
    
    # TP settings
    'tp_percentages': [0.25, 0.35, 0.25, 0.10, 0.05],  # TP1, TP2, TP3, TP4, TP5
    'tp_descriptions': [
        "Conservative target (partial)",
        "Primary target (partial)",
        "Main target (partial)",
        "Extended target (partial)",
        "Aggressive target (partial)"
    ],
    'tp_cluster_tolerance': 0.005,      # 0.5% tolerance for clustering
    'max_tp_distance_pct': 0.15,        # 15% max distance from entry
    
    # Weight limits
    'min_weight': 0.5,
    'max_weight': 2.0,
    
    # Validation
    'min_confidence_to_trade': 0.60,
    'min_risk_reward': 1.5,
    
    # Debug
    'debug': False
}


class ExpertAggregator:
    """
    Aggregates 5 expert signals into 1 combined signal
    
    Features:
    - Configurable consensus thresholds (from config.py)
    - TP level clustering and sorting
    - Weighted averages for entry/SL
    - Handles direction-only signals
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize aggregator
        
        Args:
            config: Configuration dictionary (loads from config.py if not provided)
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Try to load from config.py
        self._load_from_config()
        
        # Aggregation history
        self.aggregation_history: List[CombinedSignal] = []
        
        if self.config['debug']:
            print(f"[ExpertAggregator] Initialized with min_agreement={self.config['min_experts_to_agree']}")




        self.weight_manager = ExpertWeightManager()
        self.weight_manager.sync_with_config()
    
    def get_expert_weights(self):
        return self.weight_manager.get_all_weights()   
    def _load_from_config(self):
        """Load configuration from config.py if available"""
        try:
            from config import Config
            
            if hasattr(Config, 'MIN_EXPERTS_TO_AGREE'):
                self.config['min_experts_to_agree'] = Config.MIN_EXPERTS_TO_AGREE
            if hasattr(Config, 'MIN_EXPERTS_WITH_SIGNAL'):
                self.config['min_experts_with_signal'] = Config.MIN_EXPERTS_WITH_SIGNAL
            if hasattr(Config, 'TP_PERCENTAGES'):
                self.config['tp_percentages'] = Config.TP_PERCENTAGES
            if hasattr(Config, 'TP_CLUSTER_TOLERANCE'):
                self.config['tp_cluster_tolerance'] = Config.TP_CLUSTER_TOLERANCE
            if hasattr(Config, 'MAX_TP_DISTANCE_PCT'):
                self.config['max_tp_distance_pct'] = Config.MAX_TP_DISTANCE_PCT
                
        except ImportError:
            pass
    
    # ============================================================================
    # MAIN AGGREGATION METHOD
    # ============================================================================
    
    def aggregate(self, expert_signals: List[ExpertSignal]) -> Optional[CombinedSignal]:
        """
        Aggregate 5 expert signals into one combined signal
        
        Args:
            expert_signals: List of 5 ExpertSignal objects (one per expert)
        
        Returns:
            CombinedSignal if consensus reached, else None
        """
        if self.config['debug']:
            print(f"[ExpertAggregator] Processing {len(expert_signals)} expert signals")
        
        # ===== STEP 1: CHECK CONSENSUS =====
        consensus = self._check_consensus(expert_signals)
        
        if not consensus.reached:
            if self.config['debug']:
                print(f"[ExpertAggregator] ❌ Consensus NOT reached: {consensus.details}")
            return None
        
        direction = consensus.direction
        
        if self.config['debug']:
            print(f"[ExpertAggregator] ✅ Consensus reached: {direction} "
                  f"({consensus.agreement_ratio:.0%} agreement)")
        
        # ===== STEP 2: FILTER SIGNALS BY DIRECTION =====
        # Only use signals that agree with consensus direction
        agreeing_signals = [
            s for s in expert_signals 
            if s.direction == direction and is_tradeable_signal(s)
        ]
        
        # Also include direction-only signals that agree
        direction_only_signals = [
            s for s in expert_signals 
            if s.is_direction_only and s.direction == direction
        ]
        
        # Full signals (with entry/SL/TP)
        full_signals = [s for s in agreeing_signals if is_full_signal(s)]
        
        if not full_signals and not direction_only_signals:
            if self.config['debug']:
                print("[ExpertAggregator] No agreeing signals with valid data")
            return None
        
        # ===== STEP 3: BUILD TP LEVELS =====
        tp_levels = self._build_tp_levels(full_signals, direction)
        
        if not tp_levels:
            if self.config['debug']:
                print("[ExpertAggregator] No valid TP levels after clustering")
            return None
        
        # ===== STEP 4: CALCULATE WEIGHTED AVERAGES =====
        entry = self._weighted_average(full_signals, 'entry')
        stop_loss = self._weighted_average(full_signals, 'stop_loss')
        
        if entry <= 0 or stop_loss <= 0:
            if self.config['debug']:
                print("[ExpertAggregator] Invalid entry or stop loss")
            return None
        
        # ===== STEP 5: CALCULATE CONFIDENCE =====
        confidence = self._calculate_confidence(agreeing_signals, direction_only_signals, consensus)
        
        # ===== STEP 6: VALIDATE SETUP =====
        if not self._validate_setup(entry, stop_loss, tp_levels[0].price, direction):
            if self.config['debug']:
                print("[ExpertAggregator] Invalid setup (SL/TP on wrong side)")
            return None
        
        # ===== STEP 7: CHECK MINIMUM THRESHOLDS =====
        if confidence < self.config['min_confidence_to_trade']:
            if self.config['debug']:
                print(f"[ExpertAggregator] Confidence too low: {confidence:.1%}")
            return None
        
        risk_reward = self._calculate_risk_reward(entry, stop_loss, tp_levels[0].price, direction)
        if risk_reward < self.config['min_risk_reward']:
            if self.config['debug']:
                print(f"[ExpertAggregator] RR too low: {risk_reward:.2f}")
            return None
        
        # ===== STEP 8: BUILD EXPERT DETAILS =====
        expert_details = self._build_expert_details(expert_signals, direction, consensus)
        
        # ===== STEP 9: DETERMINE GRADE =====
        grade = self._determine_grade(confidence, consensus.agreement_ratio, risk_reward)
        
        # ===== STEP 10: CREATE COMBINED SIGNAL =====
        combined = CombinedSignal(
            direction=direction,
            entry=round(entry, 6),
            stop_loss=round(stop_loss, 6),
            tp_levels=tp_levels,
            confidence=round(confidence, 3),
            grade=grade,
            risk_reward=risk_reward,
            total_experts=len(expert_signals),
            agreeing_experts=len(agreeing_signals),
            opposing_experts=len(consensus.opposing_experts),
            neutral_experts=len(consensus.neutral_experts),
            agreement_ratio=consensus.agreement_ratio,
            consensus_reached=True,
            expert_details=expert_details,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.aggregation_history.append(combined)
        if len(self.aggregation_history) > 100:
            self.aggregation_history = self.aggregation_history[-100:]
        
        if self.config['debug']:
            print(f"[ExpertAggregator] ✅ Combined signal: {direction} "
                  f"conf={confidence:.1%}, RR={risk_reward:.2f}, grade={grade}")
        
        return combined
    
    # ============================================================================
    # CONSENSUS CHECK
    # ============================================================================
    
    def _check_consensus(self, signals: List[ExpertSignal]) -> ConsensusResult:
        """
        Check if there is sufficient consensus among experts
        
        Rules:
        1. Must have at least MIN_EXPERTS_WITH_SIGNAL experts that generated signals
        2. Majority direction must have at least MIN_EXPERTS_TO_AGREE experts
        """
        # Count tradeable signals (not HOLD/NEUTRAL)
        tradeable = [s for s in signals if is_tradeable_signal(s)]
        
        # Check minimum experts with signal
        if len(tradeable) < self.config['min_experts_with_signal']:
            return ConsensusResult(
                reached=False,
                direction="HOLD",
                agreeing_experts=[],
                opposing_experts=[],
                neutral_experts=[s.expert_name for s in signals if not is_tradeable_signal(s)],
                agreement_ratio=len(tradeable) / len(signals),
                details={'reason': f'Only {len(tradeable)} experts generated signals'}
            )
        
        # Count directions
        buy_experts = [s.expert_name for s in tradeable if s.direction == 'BUY']
        sell_experts = [s.expert_name for s in tradeable if s.direction == 'SELL']
        neutral_experts = [s.expert_name for s in signals if s.direction in ['NEUTRAL', 'HOLD']]
        
        buy_count = len(buy_experts)
        sell_count = len(sell_experts)
        
        # Determine majority
        if buy_count > sell_count:
            majority_direction = 'BUY'
            majority_count = buy_count
            agreeing = buy_experts
            opposing = sell_experts
        elif sell_count > buy_count:
            majority_direction = 'SELL'
            majority_count = sell_count
            agreeing = sell_experts
            opposing = buy_experts
        else:
            # Tie
            return ConsensusResult(
                reached=False,
                direction="HOLD",
                agreeing_experts=[],
                opposing_experts=buy_experts + sell_experts,
                neutral_experts=neutral_experts,
                agreement_ratio=0.5,
                details={'reason': f'Tie: {buy_count} BUY vs {sell_count} SELL'}
            )
        
        # Check if majority meets minimum agreement threshold
        if majority_count >= self.config['min_experts_to_agree']:
            return ConsensusResult(
                reached=True,
                direction=majority_direction,
                agreeing_experts=agreeing,
                opposing_experts=opposing,
                neutral_experts=neutral_experts,
                agreement_ratio=majority_count / len(signals),
                details={'buy_count': buy_count, 'sell_count': sell_count}
            )
        else:
            return ConsensusResult(
                reached=False,
                direction=majority_direction,
                agreeing_experts=agreeing,
                opposing_experts=opposing,
                neutral_experts=neutral_experts,
                agreement_ratio=majority_count / len(signals),
                details={'reason': f'Only {majority_count} experts agree, need {self.config["min_experts_to_agree"]}'}
            )
    
    # ============================================================================
    # TP LEVEL BUILDING
    # ============================================================================
    
    def _build_tp_levels(self, signals: List[ExpertSignal], direction: str) -> List[TPLevel]:
        """
        Build TP levels from all expert signals
        
        Process:
        1. Collect all TP prices (from take_profit and tp_levels)
        2. Filter by distance from entry
        3. Cluster nearby levels
        4. Sort by distance (closest first for direction)
        5. Assign percentages
        """
        # Collect all TP prices
        all_tp_prices = []
        
        for signal in signals:
            # Add single take_profit if valid
            if signal.take_profit > 0:
                all_tp_prices.append(signal.take_profit)
            
            # Add from tp_levels if available
            if signal.tp_levels:
                for tp in signal.tp_levels:
                    if tp.price > 0:
                        all_tp_prices.append(tp.price)
        
        if not all_tp_prices:
            return []
        
        # Filter by distance
        filtered = self._filter_tp_by_distance(all_tp_prices, direction)
        
        if not filtered:
            return []
        
        # Cluster nearby levels
        clustered = self._cluster_price_levels(filtered)
        
        if not clustered:
            return []
        
        # Sort by distance (closest first for direction)
        if direction == 'BUY':
            sorted_tp = sorted(clustered)
        else:
            sorted_tp = sorted(clustered, reverse=True)
        
        # Limit to max 5 levels
        sorted_tp = sorted_tp[:5]
        
        # Ensure minimum 3 levels (interpolate if needed)
        if len(sorted_tp) < 3:
            sorted_tp = self._interpolate_tp_levels(sorted_tp, direction)
        
        # Create TPLevel objects with percentages
        tp_levels = []
        for i, price in enumerate(sorted_tp[:5]):
            percentage = self.config['tp_percentages'][i] if i < len(self.config['tp_percentages']) else 0.1
            description = self.config['tp_descriptions'][i] if i < len(self.config['tp_descriptions']) else f"Target {i+1}"
            
            tp_levels.append(TPLevel(
                price=round(price, 6),
                percentage=percentage,
                description=description
            ))
        
        return tp_levels
    
    def _filter_tp_by_distance(self, tp_prices: List[float], direction: str) -> List[float]:
        """Remove TP levels that are too far from a reasonable reference"""
        # Use median as reference
        if not tp_prices:
            return []
        
        # For filtering, we need a reference price
        # Use the median of all TPs as reference
        median_tp = np.median(tp_prices)
        
        filtered = []
        for price in tp_prices:
            distance = abs(price - median_tp) / median_tp
            if distance <= self.config['max_tp_distance_pct']:
                filtered.append(price)
        
        return filtered if filtered else tp_prices[:5]
    
    def _cluster_price_levels(self, prices: List[float]) -> List[float]:
        """Cluster nearby price levels"""
        if not prices:
            return []
        
        sorted_prices = sorted(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            # Check if within tolerance
            if abs(price - current_cluster[-1]) / current_cluster[-1] <= self.config['tp_cluster_tolerance']:
                current_cluster.append(price)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [price]
        
        # Add last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def _interpolate_tp_levels(self, current_levels: List[float], direction: str) -> List[float]:
        """Interpolate additional TP levels if we have too few"""
        if len(current_levels) >= 3:
            return current_levels
        
        if len(current_levels) == 0:
            return []
        
        if len(current_levels) == 1:
            # Create two more levels based on the single level
            base = current_levels[0]
            if direction == 'BUY':
                step = base * 0.02  # 2% steps
                return [base, base + step, base + step * 2]
            else:
                step = base * 0.02
                return [base, base - step, base - step * 2]
        
        if len(current_levels) == 2:
            # Interpolate one more level
            if direction == 'BUY':
                step = (current_levels[1] - current_levels[0]) * 0.5
                return [current_levels[0], current_levels[1], current_levels[1] + step]
            else:
                step = (current_levels[0] - current_levels[1]) * 0.5
                return [current_levels[0], current_levels[1], current_levels[1] - step]
        
        return current_levels
    
    # ============================================================================
    # WEIGHTED AVERAGE & CONFIDENCE
    # ============================================================================
    
    def _weighted_average(self, signals: List[ExpertSignal], field: str) -> float:
        """Calculate weighted average of a field"""
        if not signals:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for s in signals:
            weight = s.weight
            if weight <= 0:
                weight = 1.0
            
            value = getattr(s, field, 0)
            if value <= 0:
                continue
            
            total_weight += weight
            weighted_sum += value * weight
        
        if total_weight == 0:
            return sum(getattr(s, field) for s in signals) / len(signals)
        
        return weighted_sum / total_weight
    
    def _calculate_confidence(self, agreeing_signals: List[ExpertSignal],
                              direction_only_signals: List[ExpertSignal],
                              consensus: ConsensusResult) -> float:
        """
        Calculate final confidence
        
        Formula: 
        confidence = (agreement_ratio × 0.4) + 
                    (avg_confidence_of_agreeing × 0.4) +
                    (direction_only_boost × 0.2)
        """
        # Factor 1: Agreement ratio
        agreement_factor = consensus.agreement_ratio
        
        # Factor 2: Average confidence of agreeing experts
        if agreeing_signals:
            avg_confidence = sum(s.confidence for s in agreeing_signals) / len(agreeing_signals)
        else:
            avg_confidence = 0.5
        
        # Factor 3: Direction-only signals boost
        if direction_only_signals:
            direction_boost = 0.10 * min(1.0, len(direction_only_signals) / 3)
        else:
            direction_boost = 0.0
        
        confidence = (agreement_factor * 0.4) + (avg_confidence * 0.4) + (direction_boost * 0.2)
        
        return min(0.95, max(0.30, confidence))
    
    def _calculate_risk_reward(self, entry: float, stop_loss: float, 
                                first_tp: float, direction: str) -> float:
        """Calculate risk/reward ratio"""
        if direction == 'BUY':
            risk = entry - stop_loss
            reward = first_tp - entry
        else:
            risk = stop_loss - entry
            reward = entry - first_tp
        
        return reward / risk if risk > 0 else 0
    
    def _validate_setup(self, entry: float, stop_loss: float, first_tp: float, direction: str) -> bool:
        """Validate that the trade setup is valid"""
        if direction == 'BUY':
            return stop_loss < entry < first_tp
        else:
            return first_tp < entry < stop_loss
    
    def _determine_grade(self, confidence: float, agreement_ratio: float, risk_reward: float) -> str:
        """Determine overall grade for the combined signal"""
        # Base score
        score = (confidence * 0.5) + (agreement_ratio * 0.3) + (min(risk_reward / 3.0, 1.0) * 0.2)
        
        if score >= 0.90:
            return 'A+'
        elif score >= 0.85:
            return 'A'
        elif score >= 0.78:
            return 'B+'
        elif score >= 0.72:
            return 'B'
        elif score >= 0.65:
            return 'B-'
        elif score >= 0.60:
            return 'C+'
        elif score >= 0.55:
            return 'C'
        else:
            return 'D'
    
    def _build_expert_details(self, signals: List[ExpertSignal], 
                               direction: str, consensus: ConsensusResult) -> Dict[str, Dict]:
        """Build expert_details dictionary for output"""
        details = {}
        
        for s in signals:
            details[s.expert_name] = {
                'direction': s.direction,
                'confidence': round(s.confidence, 3),
                'weight': round(s.weight, 3),
                'grade': s.grade,
                'agreed': s.expert_name in consensus.agreeing_experts,
                'is_direction_only': s.is_direction_only
            }
            
            # Add entry/SL/TP if available
            if s.entry > 0:
                details[s.expert_name]['entry'] = round(s.entry, 6)
            if s.stop_loss > 0:
                details[s.expert_name]['stop_loss'] = round(s.stop_loss, 6)
            if s.take_profit > 0:
                details[s.expert_name]['take_profit'] = round(s.take_profit, 6)
        
        return details
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        if not self.aggregation_history:
            return {'total_aggregations': 0}
        
        return {
            'total_aggregations': len(self.aggregation_history),
            'avg_confidence': np.mean([s.confidence for s in self.aggregation_history]),
            'avg_risk_reward': np.mean([s.risk_reward for s in self.aggregation_history]),
            'direction_counts': {
                'BUY': sum(1 for s in self.aggregation_history if s.direction == 'BUY'),
                'SELL': sum(1 for s in self.aggregation_history if s.direction == 'SELL')
            },
            'avg_tp_levels': np.mean([len(s.tp_levels) for s in self.aggregation_history]),
            'config': self.config
        }
    
    def update_config(self, key: str, value: Any):
        """Update configuration dynamically"""
        if key in self.config:
            self.config[key] = value
            if self.config['debug']:
                print(f"[ExpertAggregator] Updated config: {key} = {value}")
    
    def set_debug(self, enabled: bool):
        """Enable/disable debug logging"""
        self.config['debug'] = enabled
    
    def reset_history(self):
        """Reset aggregation history"""
        self.aggregation_history = []


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def aggregate_expert_signals(expert_signals: List[ExpertSignal], 
                             config: Dict = None) -> Optional[CombinedSignal]:
    """
    Convenience function to aggregate expert signals
    
    Args:
        expert_signals: List of 5 ExpertSignal objects
        config: Optional aggregator configuration
    
    Returns:
        CombinedSignal if consensus reached, else None
    """
    aggregator = ExpertAggregator(config)
    return aggregator.aggregate(expert_signals)