"""
pattern_clustering.py - Advanced Pattern Clustering Engine for Pattern V4

Groups patterns by:
- Direction alignment
- Price proximity
- Formation time

Provides:
- Cluster score calculation
- Feedback loop to enhance individual patterns
- Conflict resolution
- Position multiplier based on cluster strength

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import uuid

from .pattern_config import CONFIG


# ============================================================================
# CLUSTERING CONFIGURATION
# ============================================================================

class ClusteringConfigV4:
    """Configuration for clustering engine"""
    
    def __init__(self):
        clustering_config = CONFIG.clustering_config if hasattr(CONFIG, 'clustering_config') else {}
        
        self.enabled = clustering_config.get('enabled', True)
        self.min_cluster_size = clustering_config.get('min_cluster_size', 2)
        self.distance_threshold = clustering_config.get('distance_threshold', 0.02)  # 2%
        self.cluster_boost_per_pattern = clustering_config.get('cluster_boost_per_pattern', 0.05)
        self.max_cluster_boost = clustering_config.get('max_cluster_boost', 0.15)
        self.conflict_resolution = clustering_config.get('conflict_resolution', 'strongest_wins')
        self.conflict_threshold = clustering_config.get('conflict_threshold', 0.20)
        self.time_window_bars = clustering_config.get('time_window_bars', 20)


# ============================================================================
# PATTERN CLUSTER DATA CLASS
# ============================================================================

@dataclass
class PatternClusterV4:
    """
    A group of patterns that align in direction and proximity.
    """
    cluster_id: str
    direction: str  # 'BUY' or 'SELL'
    patterns: List[Dict]
    center_price: float
    price_range: Tuple[float, float]
    formation_start: int
    formation_end: int
    raw_score: float
    boosted_score: float
    
    @property
    def size(self) -> int:
        return len(self.patterns)
    
    @property
    def avg_confidence(self) -> float:
        if not self.patterns:
            return 0.0
        return sum(p.get('final_confidence', p.get('similarity', 0.5)) for p in self.patterns) / len(self.patterns)
    
    @property
    def avg_similarity(self) -> float:
        if not self.patterns:
            return 0.0
        return sum(p.get('similarity', 0.5) for p in self.patterns) / len(self.patterns)
    
    def get_pattern_names(self) -> List[str]:
        return [p.get('pattern_name', 'Unknown') for p in self.patterns]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'direction': self.direction,
            'size': self.size,
            'patterns': self.get_pattern_names(),
            'center_price': self.center_price,
            'raw_score': round(self.raw_score, 3),
            'boosted_score': round(self.boosted_score, 3),
            'avg_confidence': round(self.avg_confidence, 3),
            'avg_similarity': round(self.avg_similarity, 3),
        }


# ============================================================================
# CLUSTERING ENGINE
# ============================================================================

class PatternClusteringEngineV4:
    """
    Groups patterns by direction and price proximity.
    Provides feedback loop to enhance individual patterns.
    """
    
    def __init__(self, config: ClusteringConfigV4 = None):
        self.config = config or ClusteringConfigV4()
        self.clusters: List[PatternClusterV4] = []
    
    def cluster_patterns(self, patterns: List[Dict], 
                         current_price: float,
                         current_idx: int) -> List[Dict]:
        """
        Cluster patterns and apply feedback loop.
        Returns enhanced patterns with cluster data.
        """
        
        if not self.config.enabled or len(patterns) < 2:
            # No clustering, just return patterns unchanged
            for p in patterns:
                p['cluster_size'] = 1
                p['cluster_score'] = 0.0
                p['cluster_id'] = None
            return patterns
        
        # Step 1: Separate by direction
        bullish_patterns = [p for p in patterns if p.get('direction') == 'BUY']
        bearish_patterns = [p for p in patterns if p.get('direction') == 'SELL']
        
        # Step 2: Cluster each direction separately
        bullish_clusters = self._cluster_by_price(bullish_patterns, current_price, current_idx)
        bearish_clusters = self._cluster_by_price(bearish_patterns, current_price, current_idx)
        
        # Step 3: Combine all clusters
        all_clusters = bullish_clusters + bearish_clusters
        
        # Step 4: Detect conflicts between clusters
        conflicts = self._detect_conflicts(bullish_clusters, bearish_clusters, current_price)
        
        # Step 5: Apply feedback loop to enhance individual patterns
        enhanced_patterns = self._apply_feedback_loop(patterns, all_clusters, conflicts)
        
        # Step 6: Resolve conflicts for patterns in conflicting clusters
        enhanced_patterns = self._resolve_conflicts(enhanced_patterns, conflicts)
        
        return enhanced_patterns
    
    def _cluster_by_price(self, patterns: List[Dict], 
                          current_price: float,
                          current_idx: int) -> List[PatternClusterV4]:
        """
        Group patterns by price proximity.
        Uses DBSCAN-like approach with distance threshold.
        """
        
        if not patterns:
            return []
        
        # Sort patterns by price
        sorted_patterns = sorted(patterns, key=lambda p: self._get_pattern_price(p))
        
        clusters = []
        used = set()
        
        for i, p1 in enumerate(sorted_patterns):
            if i in used:
                continue
            
            # Start new cluster
            cluster_patterns = [p1]
            used.add(i)
            price1 = self._get_pattern_price(p1)
            
            # Find all patterns within distance threshold
            for j, p2 in enumerate(sorted_patterns):
                if j in used:
                    continue
                
                price2 = self._get_pattern_price(p2)
                distance = abs(price1 - price2) / max(price1, price2)
                
                # Also check time proximity (within time window)
                time_diff = abs(self._get_pattern_time(p1) - self._get_pattern_time(p2))
                
                if distance <= self.config.distance_threshold and time_diff <= self.config.time_window_bars:
                    cluster_patterns.append(p2)
                    used.add(j)
            
            # Create cluster if minimum size met
            if len(cluster_patterns) >= self.config.min_cluster_size:
                cluster = self._create_cluster(
                    cluster_patterns, current_price, current_idx
                )
                clusters.append(cluster)
        
        return clusters
    
    def _get_pattern_price(self, pattern: Dict) -> float:
        """Get the key price for clustering (entry or neckline)"""
        return pattern.get('entry', pattern.get('neckline', pattern.get('similarity', 0)))
    
    def _get_pattern_time(self, pattern: Dict) -> int:
        """Get the formation time for clustering"""
        return pattern.get('end_idx', pattern.get('start_idx', 0))
    
    def _create_cluster(self, patterns: List[Dict], 
                        current_price: float,
                        current_idx: int) -> PatternClusterV4:
        """
        Create a cluster from a list of patterns.
        Calculates cluster score and metrics.
        """
        
        # Determine cluster direction (all should be same)
        direction = patterns[0].get('direction', 'NEUTRAL')
        
        # Calculate center price (median of pattern prices)
        prices = [self._get_pattern_price(p) for p in patterns]
        center_price = np.median(prices)
        price_range = (min(prices), max(prices))
        
        # Formation range
        start_idx = min(p.get('start_idx', current_idx) for p in patterns)
        end_idx = max(p.get('end_idx', 0) for p in patterns)
        
        # Calculate raw cluster score
        raw_score = self._calculate_cluster_score(patterns)
        
        # Calculate boosted score (with cluster size bonus)
        boost = min(
            self.config.max_cluster_boost,
            (len(patterns) - 1) * self.config.cluster_boost_per_pattern
        )
        boosted_score = min(1.0, raw_score + boost)
        
        cluster = PatternClusterV4(
            cluster_id=str(uuid.uuid4())[:8],
            direction=direction,
            patterns=patterns,
            center_price=center_price,
            price_range=price_range,
            formation_start=start_idx,
            formation_end=end_idx,
            raw_score=raw_score,
            boosted_score=boosted_score
        )
        
        return cluster
    
    def _calculate_cluster_score(self, patterns: List[Dict]) -> float:
        """
        Calculate cluster score based on:
        - Number of patterns
        - Average confidence
        - Average similarity
        - Pattern diversity (different types)
        """
        
        if not patterns:
            return 0.0
        
        # Factor 1: Size factor (more patterns = higher score)
        size_factor = min(1.0, len(patterns) / 10)  # Cap at 10 patterns
        
        # Factor 2: Average confidence
        avg_confidence = sum(p.get('final_confidence', p.get('similarity', 0.5)) for p in patterns) / len(patterns)
        
        # Factor 3: Average similarity
        avg_similarity = sum(p.get('similarity', 0.5) for p in patterns) / len(patterns)
        
        # Factor 4: Pattern diversity (different types = stronger)
        unique_types = len(set(p.get('pattern_name', 'Unknown') for p in patterns))
        diversity_factor = min(1.0, unique_types / len(patterns))
        
        # Weighted combination
        score = (
            size_factor * 0.25 +
            avg_confidence * 0.35 +
            avg_similarity * 0.30 +
            diversity_factor * 0.10
        )
        
        return min(1.0, score)
    
    def _detect_conflicts(self, bullish_clusters: List[PatternClusterV4],
                          bearish_clusters: List[PatternClusterV4],
                          current_price: float) -> List[Dict]:
        """
        Detect conflicts between bullish and bearish clusters.
        Returns list of conflicts with severity.
        """
        
        conflicts = []
        
        for bull_cluster in bullish_clusters:
            for bear_cluster in bearish_clusters:
                # Check if clusters are close in price
                price_distance = abs(bull_cluster.center_price - bear_cluster.center_price)
                price_distance_pct = price_distance / max(bull_cluster.center_price, bear_cluster.center_price)
                
                if price_distance_pct <= self.config.distance_threshold * 2:
                    # Conflict detected
                    strength_diff = abs(bull_cluster.boosted_score - bear_cluster.boosted_score)
                    severity = min(1.0, (1 - strength_diff) * 2)
                    
                    conflicts.append({
                        'bullish_cluster': bull_cluster,
                        'bearish_cluster': bear_cluster,
                        'price_distance': price_distance_pct,
                        'strength_diff': strength_diff,
                        'severity': severity,
                        'winner': 'bullish' if bull_cluster.boosted_score > bear_cluster.boosted_score else 'bearish'
                    })
        
        return conflicts
    
    def _apply_feedback_loop(self, patterns: List[Dict],
                              clusters: List[PatternClusterV4],
                              conflicts: List[Dict]) -> List[Dict]:
        """
        Apply feedback loop to enhance individual patterns based on clusters.
        Patterns in strong clusters get confidence boosts.
        """
        
        # Create mapping from pattern index to cluster (using pattern name + similarity as key)
        pattern_to_cluster = {}
        for cluster in clusters:
            for pattern in cluster.patterns:
                # Create a unique key for the pattern
                key = (pattern.get('pattern_name', 'Unknown'), pattern.get('similarity', 0))
                pattern_to_cluster[key] = cluster
        
        # Apply boosts
        for pattern in patterns:
            key = (pattern.get('pattern_name', 'Unknown'), pattern.get('similarity', 0))
            cluster = pattern_to_cluster.get(key)
            
            if cluster:
                # Pattern is in a cluster
                pattern['cluster_id'] = cluster.cluster_id
                pattern['cluster_size'] = cluster.size
                pattern['cluster_score'] = cluster.boosted_score
                pattern['aligned_patterns'] = cluster.get_pattern_names()
                
                # Apply boost to confidence
                boost = cluster.boosted_score - cluster.raw_score
                old_confidence = pattern.get('final_confidence', pattern.get('similarity', 0.5))
                pattern['final_confidence'] = min(0.85, old_confidence + boost)
                
                # Calculate position multiplier based on cluster
                pattern['cluster_multiplier'] = self._calculate_cluster_multiplier(cluster)
            else:
                # Pattern is alone
                pattern['cluster_size'] = 1
                pattern['cluster_score'] = 0.0
                pattern['cluster_multiplier'] = 1.0
                pattern['cluster_id'] = None
        
        return patterns
    
    def _calculate_cluster_multiplier(self, cluster: PatternClusterV4) -> float:
        """
        Calculate position multiplier based on cluster strength.
        Range: 1.0 to 1.5
        """
        
        # Base on cluster score
        if cluster.boosted_score >= 0.85:
            multiplier = 1.5
        elif cluster.boosted_score >= 0.75:
            multiplier = 1.3
        elif cluster.boosted_score >= 0.65:
            multiplier = 1.15
        else:
            multiplier = 1.0
        
        # Adjust for cluster size
        if cluster.size >= 4:
            multiplier = min(1.5, multiplier + 0.1)
        elif cluster.size >= 3:
            multiplier = min(1.5, multiplier + 0.05)
        
        return multiplier
    
    def _resolve_conflicts(self, patterns: List[Dict],
                           conflicts: List[Dict]) -> List[Dict]:
        """
        Resolve conflicts by marking conflicting patterns or reducing confidence.
        """
        
        if not conflicts:
            return patterns
        
        # Get patterns involved in conflicts
        conflicted_pattern_keys = set()
        
        for conflict in conflicts:
            if conflict['severity'] > self.config.conflict_threshold:
                # Severe conflict - mark both clusters as conflicted
                for pattern in conflict['bullish_cluster'].patterns:
                    key = (pattern.get('pattern_name', 'Unknown'), pattern.get('similarity', 0))
                    conflicted_pattern_keys.add(key)
                for pattern in conflict['bearish_cluster'].patterns:
                    key = (pattern.get('pattern_name', 'Unknown'), pattern.get('similarity', 0))
                    conflicted_pattern_keys.add(key)
        
        # Apply penalties to conflicted patterns
        for pattern in patterns:
            key = (pattern.get('pattern_name', 'Unknown'), pattern.get('similarity', 0))
            if key in conflicted_pattern_keys:
                pattern['conflict_detected'] = True
                # Reduce confidence for conflicted patterns
                old_confidence = pattern.get('final_confidence', pattern.get('similarity', 0.5))
                pattern['final_confidence'] = max(0.40, old_confidence * 0.85)
                pattern['conflict_resolution'] = "Conflict with opposite direction patterns"
        
        return patterns


# ============================================================================
# CONFLUENCE SCORER
# ============================================================================

class ConfluenceScorerV4:
    """
    Calculates confluence score for patterns based on:
    - Multiple pattern agreement
    - Technical indicator alignment
    - Liquidity confirmation
    """
    
    def __init__(self):
        self.weights = {
            'pattern_count': 0.30,
            'indicator_alignment': 0.25,
            'liquidity_confirmation': 0.25,
            'volume_confirmation': 0.20
        }
    
    def calculate_confluence(self, pattern: Dict, 
                             all_patterns: List[Dict],
                             df: pd.DataFrame) -> float:
        """
        Calculate confluence score for a pattern.
        Higher score = more factors agree.
        """
        
        scores = []
        
        # 1. Pattern count (how many patterns agree with this direction)
        pattern_count_score = self._calc_pattern_count_score(pattern, all_patterns)
        scores.append(pattern_count_score * self.weights['pattern_count'])
        
        # 2. Indicator alignment
        indicator_score = self._calc_indicator_alignment(pattern, df)
        scores.append(indicator_score * self.weights['indicator_alignment'])
        
        # 3. Liquidity confirmation
        liquidity_score = pattern.get('liquidity', {}).get('score', 0.5)
        scores.append(liquidity_score * self.weights['liquidity_confirmation'])
        
        # 4. Volume confirmation
        volume_score = pattern.get('components', {}).get('volume_pattern', 0.5)
        scores.append(volume_score * self.weights['volume_confirmation'])
        
        return sum(scores)
    
    def _calc_pattern_count_score(self, pattern: Dict,
                                   all_patterns: List[Dict]) -> float:
        """Calculate score based on how many patterns agree"""
        
        direction = pattern.get('direction', 'NEUTRAL')
        
        same_direction = [
            p for p in all_patterns 
            if p.get('direction') == direction
        ]
        
        total = len(all_patterns)
        if total <= 1:
            return 0.5
        
        agreement_ratio = len(same_direction) / total
        
        # Convert to score: 0.5 at 50% agreement, 1.0 at 100% agreement
        return 0.5 + (agreement_ratio * 0.5)
    
    def _calc_indicator_alignment(self, pattern: Dict, 
                                   df: pd.DataFrame) -> float:
        """Calculate how well indicators align with pattern direction"""
        
        if len(df) < 20:
            return 0.5
        
        direction = pattern.get('direction', 'NEUTRAL')
        current_price = float(df['close'].iloc[-1])
        scores = []
        
        # RSI alignment
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if direction == 'BUY':
                if rsi < 40:
                    scores.append(0.9)
                elif rsi < 50:
                    scores.append(0.7)
                else:
                    scores.append(0.4)
            else:
                if rsi > 60:
                    scores.append(0.9)
                elif rsi > 50:
                    scores.append(0.7)
                else:
                    scores.append(0.4)
        
        # MACD alignment (if available)
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            signal = df['macd_signal'].iloc[-1]
            if direction == 'BUY':
                if macd > signal:
                    scores.append(0.8)
                else:
                    scores.append(0.4)
            else:
                if macd < signal:
                    scores.append(0.8)
                else:
                    scores.append(0.4)
        
        # EMA alignment
        if 'ema_8' in df.columns and 'ema_21' in df.columns:
            ema8 = df['ema_8'].iloc[-1]
            ema21 = df['ema_21'].iloc[-1]
            if direction == 'BUY':
                if ema8 > ema21:
                    scores.append(0.7)
                else:
                    scores.append(0.4)
            else:
                if ema8 < ema21:
                    scores.append(0.7)
                else:
                    scores.append(0.4)
        
        if not scores:
            return 0.5
        
        return sum(scores) / len(scores)


# ============================================================================
# MAIN CLUSTERING ENGINE (Orchestrator)
# ============================================================================

class PatternClusteringEngineV4Orchestrator:
    """
    Main clustering engine that orchestrates all clustering components.
    """
    
    def __init__(self):
        self.clustering_engine = PatternClusteringEngineV4()
        self.confluence_scorer = ConfluenceScorerV4()
    
    def process(self, patterns: List[Dict], df: pd.DataFrame,
                current_idx: int) -> List[Dict]:
        """
        Process all patterns through clustering and confluence.
        Returns enhanced patterns.
        """
        
        if not patterns:
            return patterns
        
        current_price = float(df['close'].iloc[-1])
        
        # Step 1: Cluster patterns
        enhanced_patterns = self.clustering_engine.cluster_patterns(
            patterns, current_price, current_idx
        )
        
        # Step 2: Calculate confluence scores
        for pattern in enhanced_patterns:
            confluence = self.confluence_scorer.calculate_confluence(
                pattern, enhanced_patterns, df
            )
            pattern['confluence_score'] = confluence
        
        # Step 3: Sort by cluster score (strongest first)
        enhanced_patterns.sort(
            key=lambda p: p.get('cluster_score', p.get('final_confidence', 0)), 
            reverse=True
        )
        
        return enhanced_patterns
    
    def get_best_cluster(self, patterns: List[Dict]) -> Optional[PatternClusterV4]:
        """Get the strongest cluster from patterns"""
        
        if not patterns:
            return None
        
        # Find patterns that are in clusters
        clustered_patterns = [p for p in patterns if p.get('cluster_size', 1) > 1]
        
        if not clustered_patterns:
            return None
        
        # Group by cluster_id
        cluster_map = {}
        for p in clustered_patterns:
            cluster_id = p.get('cluster_id')
            if cluster_id:
                if cluster_id not in cluster_map:
                    cluster_map[cluster_id] = []
                cluster_map[cluster_id].append(p)
        
        # Find strongest cluster
        strongest_cluster = None
        strongest_score = 0.0
        
        for cluster_id, cluster_patterns in cluster_map.items():
            avg_score = sum(p.get('cluster_score', 0) for p in cluster_patterns) / len(cluster_patterns)
            if avg_score > strongest_score:
                strongest_score = avg_score
                # Reconstruct cluster (simplified)
                strongest_cluster = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_patterns),
                    'patterns': [p.get('pattern_name', 'Unknown') for p in cluster_patterns],
                    'score': avg_score
                }
        
        return strongest_cluster
    
    def get_cluster_summary(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Get summary of all clusters"""
        
        clusters = []
        for p in patterns:
            if p.get('cluster_size', 1) > 1:
                clusters.append({
                    'cluster_id': p.get('cluster_id'),
                    'size': p.get('cluster_size', 1),
                    'patterns': p.get('aligned_patterns', []),
                    'score': p.get('cluster_score', 0)
                })
        
        # Deduplicate clusters
        unique_clusters = {}
        for c in clusters:
            if c['cluster_id'] not in unique_clusters:
                unique_clusters[c['cluster_id']] = c
        
        clusters_list = list(unique_clusters.values())
        
        return {
            'total_clusters': len(clusters_list),
            'clusters': clusters_list,
            'strongest_cluster': max(clusters_list, key=lambda x: x['score']) if clusters_list else None
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ClusteringConfigV4',
    'PatternClusterV4',
    'PatternClusteringEngineV4',
    'ConfluenceScorerV4',
    'PatternClusteringEngineV4Orchestrator',
]