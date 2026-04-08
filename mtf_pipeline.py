# mtf_pipeline.py - Clean Multi-Timeframe Confirmation Pipeline
"""
Multi-Timeframe Confirmation Pipeline - Phase 5

This pipeline:
1. Takes CombinedSignal from Phase 4 (Expert Aggregator)
2. Fetches higher timeframe data
3. Runs ALL 5 experts on each higher timeframe
4. Aggregates HTF results into an MTF score
5. Applies boost/penalty to the signal confidence
6. Filters out signals that don't meet MTF threshold

All settings are controlled from config.py
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from config import Config
from logger import log
from expert_aggregator import ExpertAggregator, DEFAULT_CONFIG
from expert_executor import ExpertExecutor
from unified_signal import UnifiedSignal, TPLevel
from expert_interface import ExpertSignal, ExpertName

# ============================================================================
# MTF CONFIGURATION
# ============================================================================

class MTFConfig:
    """Configuration for MTF Pipeline - loaded from Config"""
    
    def __init__(self):
        # Higher timeframes to analyze
        self.higher_timeframes = getattr(Config, 'MTF_HIGHER_TIMEFRAMES', ['15m', '1h', '4h'])
        
        # Minimum MTF score to confirm signal (0-1)
        self.min_mtf_score = getattr(Config, 'MTF_MIN_SCORE', 0.70)
        
        # Boost/penalty limits
        self.max_boost = getattr(Config, 'MTF_MAX_BOOST', 0.15)
        self.max_penalty = getattr(Config, 'MTF_MAX_PENALTY', -0.15)
        
        # Alignment thresholds
        self.strong_alignment = getattr(Config, 'MTF_STRONG_ALIGNMENT', 0.80)
        self.weak_alignment = getattr(Config, 'MTF_WEAK_ALIGNMENT', 0.50)
        
        # Whether to enable pullback detection
        self.enable_pullback = getattr(Config, 'MTF_ENABLE_PULLBACK', True)
        
        # Timeframe weights for aggregation
        self.timeframe_weights = getattr(Config, 'MTF_TIMEFRAME_WEIGHTS', {
            '15m': 0.5,
            '1h': 1.0,
            '4h': 1.5,
            '1d': 2.0
        })
        
        # Cache TTL in seconds
        self.cache_ttl = getattr(Config, 'MTF_CACHE_TTL', 60)


# ============================================================================
# HTF ANALYSIS RESULT
# ============================================================================

class HTFAnalysisResult:
    """Result of analyzing a single higher timeframe"""
    
    def __init__(self, timeframe: str, unified_signal: UnifiedSignal, 
                expert_signals: List[ExpertSignal]):
        self.timeframe = timeframe
        self.unified_signal = unified_signal
        self.expert_signals = expert_signals
        
        # Calculate alignment with primary signal
        self.alignment_score = 0.0
        self.alignment_quality = "NEUTRAL"
        
    @property
    def direction(self) -> str:
        return self.unified_signal.direction if self.unified_signal else "NEUTRAL"

    @property
    def confidence(self) -> float:
        return self.unified_signal.confidence if self.unified_signal else 0.0

    @property
    def has_signal(self) -> bool:
        return self.unified_signal is not None
    
    def calculate_alignment(self, primary_direction: str, 
                            primary_confidence: float) -> float:
        """Calculate how well this HTF aligns with primary signal"""
        if not self.has_signal:
            return 0.0
        
        if self.direction != primary_direction:
            return 0.0
        
        # Alignment strength = HTF confidence × (1 - distance penalty)
        # Higher confidence on HTF = stronger alignment
        return self.confidence
    
    def to_dict(self) -> Dict:
        return {
            'timeframe': self.timeframe,
            'direction': self.direction,
            'confidence': self.confidence,
            'has_signal': self.has_signal,
            'alignment_score': self.alignment_score
        }


# ============================================================================
# MTF AGGREGATED RESULT
# ============================================================================

class MTFAggregatedResult:
    """Aggregated result from all higher timeframes"""
    
    def __init__(self):
        self.timeframe_results: Dict[str, HTFAnalysisResult] = {}
        
        # Aggregated metrics
        self.bullish_count = 0
        self.bearish_count = 0
        self.neutral_count = 0
        self.total_timeframes = 0
        
        self.alignment_score = 0.0
        self.weighted_alignment = 0.0
        self.confidence_boost = 0.0
        
        self.is_confirmed = False
        self.pullback_detected = False
        self.pullback_data = None
        
        self.story = ""
        self.reasons = []
        self.confirming_timeframes = []   # ADD THIS
        self.conflicting_timeframes = []
    
    def add_result(self, timeframe: str, result: HTFAnalysisResult):
        self.timeframe_results[timeframe] = result
    
    def calculate_aggregation(self, primary_direction: str, 
                              primary_confidence: float,
                              config: MTFConfig):
        """Calculate aggregated MTF metrics"""
        
        self.total_timeframes = len(self.timeframe_results)
        
        # Count directions
        for tf, res in self.timeframe_results.items():
            if not res.has_signal:
                self.neutral_count += 1
            elif res.direction == 'BUY':
                self.bullish_count += 1
            elif res.direction == 'SELL':
                self.bearish_count += 1

        self.confirming_timeframes = [
            tf for tf, res in self.timeframe_results.items()
            if res.has_signal and res.direction == primary_direction
        ]
        self.conflicting_timeframes = [
            tf for tf, res in self.timeframe_results.items()
            if res.has_signal and res.direction != primary_direction
        ]
        
        # Calculate weighted alignment
        total_weight = 0.0
        weighted_sum = 0.0
        
        for tf, res in self.timeframe_results.items():
            weight = config.timeframe_weights.get(tf, 1.0)
            if res.has_signal and res.direction == primary_direction:
                alignment = res.confidence
            else:
                alignment = 0.0
            
            weighted_sum += alignment * weight
            total_weight += weight
        
        self.weighted_alignment = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate boost/penalty
        if self.weighted_alignment >= config.strong_alignment:
            self.confidence_boost = config.max_boost
            self.is_confirmed = True
            self.alignment_quality = "STRONG"
        elif self.weighted_alignment >= config.weak_alignment:
            self.confidence_boost = config.max_boost * (self.weighted_alignment - config.weak_alignment) / (config.strong_alignment - config.weak_alignment)
            self.is_confirmed = True
            self.alignment_quality = "MODERATE"
        elif self.weighted_alignment >= config.min_mtf_score:
            self.confidence_boost = 0.0
            self.is_confirmed = True
            self.alignment_quality = "WEAK"
        else:
            self.confidence_boost = config.max_penalty
            self.is_confirmed = False
            self.alignment_quality = "CONFLICT"
        
        # Build story
        self._build_story(primary_direction, config)
    
    def _build_story(self, primary_direction: str, config: MTFConfig):
        """Build human-readable MTF story"""
        
        if not self.timeframe_results:
            self.story = "No higher timeframe data available"
            self.reasons = ["Insufficient HTF data"]
            return
        
        parts = []
        reasons = []
        
        # Alignment summary
        if self.is_confirmed:
            parts.append(f"MTF confirms {primary_direction} signal")
            reasons.append(f"MTF confirmation: {self.weighted_alignment:.0%} alignment")
            
            if self.alignment_quality == "STRONG":
                parts.append(f"Strong alignment across {self.bullish_count + self.bearish_count} timeframes")
                reasons.append(f"Strong alignment - {self.bullish_count} bullish, {self.bearish_count} bearish")
            elif self.alignment_quality == "MODERATE":
                parts.append(f"Moderate alignment across timeframes")
        else:
            parts.append(f"MTF conflicts with {primary_direction} signal")
            reasons.append(f"MTF conflict: only {self.weighted_alignment:.0%} alignment")
        
        # Timeframe breakdown
        agreeing = [tf for tf, res in self.timeframe_results.items() 
                   if res.has_signal and res.direction == primary_direction]
        conflicting = [tf for tf, res in self.timeframe_results.items() 
                      if res.has_signal and res.direction != primary_direction]
        
        if agreeing:
            parts.append(f"Confirming: {', '.join(agreeing)}")
            reasons.append(f"Confirming TFs: {', '.join(agreeing)}")
        
        if conflicting:
            parts.append(f"Conflicting: {', '.join(conflicting)}")
            reasons.append(f"Conflicting TFs: {', '.join(conflicting)}")
        
        self.story = " | ".join(parts)
        self.reasons = reasons
    
    def apply_to_signal(self, signal: UnifiedSignal) -> UnifiedSignal:
        """Apply MTF boost/penalty to signal confidence"""
        if not self.is_confirmed:
            new_confidence = signal.confidence * (1 + self.confidence_boost)
        else:
            new_confidence = signal.confidence * (1 + self.confidence_boost)
        
        signal.confidence = min(0.95, max(0.05, new_confidence))
        signal.weighted_confidence = signal.confidence
        signal.mtf_score = self.weighted_alignment
        signal.mtf_aligned = self.is_confirmed
        signal.mtf_alignment_quality = self.alignment_quality
        signal.confirming_timeframes = [tf for tf, res in self.timeframe_results.items() 
                                        if res.has_signal and res.direction == signal.direction]
        signal.conflicting_timeframes = [tf for tf, res in self.timeframe_results.items() 
                                        if res.has_signal and res.direction != signal.direction]
        
        # Add MTF metadata
        if not hasattr(signal, 'metadata'):
            signal.metadata = {}
        
        signal.metadata['mtf'] = {
            'alignment_score': self.weighted_alignment,
            'confidence_boost': self.confidence_boost,
            'is_confirmed': self.is_confirmed,
            'alignment_quality': self.alignment_quality,
            'bullish_htf_count': self.bullish_count,
            'bearish_htf_count': self.bearish_count,
            'neutral_htf_count': self.neutral_count,
            'confirming_timeframes': signal.confirming_timeframes,
            'conflicting_timeframes': signal.conflicting_timeframes,
            'story': self.story,
            'reasons': self.reasons
        }
        
        return signal
    
    def to_dict(self) -> Dict:
        return {
            'confirmed': self.is_confirmed,
            'alignment_score': round(self.weighted_alignment, 3),
            'confidence_boost': round(self.confidence_boost, 3),
            'bullish_count': self.bullish_count,
            'bearish_count': self.bearish_count,
            'neutral_count': self.neutral_count,
            'total_timeframes': self.total_timeframes,
            'alignment_quality': self.alignment_quality,
            'story': self.story,
            'reasons': self.reasons,
            'timeframes': {tf: res.to_dict() for tf, res in self.timeframe_results.items()}
        }


# ============================================================================
# MAIN MTF PIPELINE
# ============================================================================

class MTFPipeline:
    """
    Multi-Timeframe Confirmation Pipeline - Phase 5
    
    Takes a CombinedSignal from Phase 4, runs all experts on higher timeframes,
    and applies confirmation boost/penalty.
    """
    
    def __init__(self, data_fetcher, expert_executor: ExpertExecutor = None):
        """
        Initialize MTF Pipeline
        
        Args:
            data_fetcher: DataFetcher instance for fetching HTF data
            expert_executor: Optional ExpertExecutor (creates new if not provided)
        """
        self.data_fetcher = data_fetcher
        self.config = MTFConfig()
        
        # Initialize expert executor
        self.expert_executor = expert_executor or ExpertExecutor(debug=False)
        
        # Initialize aggregator for HTF results
        self.aggregator = ExpertAggregator(config={'debug': False})
        
        # Cache for HTF analysis results
        self.cache: Dict[str, tuple] = {}  # key -> (timestamp, result)
        
        log.info("MTFPipeline initialized")
        log.info(f"  Higher Timeframes: {self.config.higher_timeframes}")
        log.info(f"  Min MTF Score: {self.config.min_mtf_score}")
        log.info(f"  Max Boost/Penalty: {self.config.max_boost}/{self.config.max_penalty}")
    
    def confirm(self, signal: UnifiedSignal, symbol: str, 
                timeframe: str, df: pd.DataFrame) -> Optional[UnifiedSignal]:
        """
        Run MTF confirmation on a combined signal
        
        Args:
            signal: CombinedSignal from Phase 4 (Expert Aggregator)
            symbol: Trading pair symbol
            timeframe: Primary timeframe
            df: Primary timeframe DataFrame
        
        Returns:
            Updated CombinedSignal with MTF metadata, or None if filtered
        """

        

        if not signal or not signal.consensus_reached:
            log.debug(f"MTF: No consensus signal for {symbol}, skipping")
            return None
        
        log.info(f"MTF: Confirming {signal.direction} signal for {symbol} on {timeframe}")
        
        try:
            # ===== STEP 1: ANALYZE EACH HIGHER TIMEFRAME =====
            htf_results: Dict[str, HTFAnalysisResult] = {}
            
            for htf in self.config.higher_timeframes:
                # Skip if not actually higher
                if not self._is_higher_tf(htf, timeframe):
                    continue
                
                # Fetch HTF data
                htf_df = self.data_fetcher.fetch_ohlcv(symbol, htf, limit=500)
                if htf_df is None or htf_df.empty or len(htf_df) < 50:
                    log.debug(f"  {symbol}: No data for {htf}, skipping")
                    continue
                
                # Run all experts on this HTF
                # expert_signals = self.expert_executor.run_all_experts
                expert_signals = self.expert_executor.execute_for_coin(
                    symbol=symbol,
                    df=htf_df,
                    timeframe=htf,
                    htf_data=None, 
                    market_regime=None,
                    structure_data=None,
                    sr_data=None,
                    liquidity_data=None,
                    indicators=None

                )
                
                # Check if all experts produced signals
                if len(expert_signals) < 5:
                    log.debug(f"  {symbol}: Only {len(expert_signals)} experts on {htf}")
                    continue
                
                # Aggregate HTF signals
                htf_combined = self.aggregator.aggregate(expert_signals)
                
                if not htf_combined or not htf_combined.consensus_reached:
                    log.debug(f"  {symbol}: No consensus on {htf}")
                    htf_results[htf] = HTFAnalysisResult(htf, None, expert_signals)
                else:
                    log.debug(f"  {symbol}: {htf} -> {htf_combined.direction} "
                             f"(conf={htf_combined.confidence:.2%})")
                    htf_results[htf] = HTFAnalysisResult(htf, htf_combined, expert_signals)
            
            if not htf_results:
                log.info(f"MTF: No higher timeframe data for {symbol}")
                return signal  # No penalty, just pass through
            
            # ===== STEP 2: AGGREGATE HTF RESULTS =====
            aggregated = MTFAggregatedResult()
            for tf, result in htf_results.items():
                # Calculate alignment with primary signal
                if result.has_signal:
                    result.alignment_score = result.calculate_alignment(
                        signal.direction, signal.confidence
                    )
                aggregated.add_result(tf, result)
            
            aggregated.calculate_aggregation(
                signal.direction, signal.confidence, self.config
            )
            
            log.info(f"MTF: Alignment = {aggregated.weighted_alignment:.1%}, "
                    f"Boost = {aggregated.confidence_boost:+.1%}, "
                    f"Confirmed = {aggregated.is_confirmed}")
            
            # ===== STEP 3: APPLY BOOST/PENALTY =====
            if aggregated.weighted_alignment >= self.config.min_mtf_score:
                # MTF confirms - apply boost
                old_confidence = signal.confidence
                signal = aggregated.apply_to_signal(signal)
                log.info(f"MTF: Confirmed! Confidence: {old_confidence:.1%} → {signal.confidence:.1%} (+{aggregated.confidence_boost:+.1%})")
            else:
                # MTF conflicts - filter out
                log.info(f"MTF: REJECTED - Alignment {aggregated.weighted_alignment:.1%} < {self.config.min_mtf_score:.0%}")
                return None
            
            # ===== STEP 4: ADD MTF DATA TO SIGNAL =====
            signal.metadata['mtf'] = {
                'alignment_score': aggregated.weighted_alignment,
                'confidence_boost': aggregated.confidence_boost,
                'is_confirmed': aggregated.is_confirmed,
                'alignment_quality': aggregated.alignment_quality,
                'bullish_count': aggregated.bullish_count,
                'bearish_count': aggregated.bearish_count,
                'neutral_count': aggregated.neutral_count,
                'confirming_timeframes': aggregated.confirming_timeframes,
                'conflicting_timeframes': aggregated.conflicting_timeframes,
                'story': aggregated.story,
                'reasons': aggregated.reasons
            }
            
            # Log MTF story
            log.info(f"MTF Story: {aggregated.story}")
            
            return signal
            
        except Exception as e:
            log.error(f"MTF: Error for {symbol}: {e}", exc_info=True)
            # On error, pass through (no penalty)
            return signal
    
    def _is_higher_tf(self, tf1: str, tf2: str) -> bool:
        """Check if tf1 is higher timeframe than tf2"""
        order = {
            '1m': 1, '5m': 2, '15m': 3, '30m': 4,
            '1h': 5, '4h': 6, '1d': 7, '1w': 8
        }
        return order.get(tf1, 0) > order.get(tf2, 0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'config': {
                'higher_timeframes': self.config.higher_timeframes,
                'min_mtf_score': self.config.min_mtf_score,
                'max_boost': self.config.max_boost,
                'max_penalty': self.config.max_penalty
            },
            'available_experts': self.expert_executor.get_available_experts(),
            'cache_size': len(self.cache)
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def confirm_mtf_signal(signal: UnifiedSignal, symbol: str, 
                       timeframe: str, df: pd.DataFrame,
                       data_fetcher) -> Optional[UnifiedSignal]:
    """
    Convenience function to run MTF confirmation
    
    Args:
        signal: CombinedSignal from Phase 4
        symbol: Trading pair symbol
        timeframe: Primary timeframe
        df: Primary timeframe DataFrame
        data_fetcher: DataFetcher instance
    
    Returns:
        Updated CombinedSignal or None if filtered
    """
    pipeline = MTFPipeline(data_fetcher)
    return pipeline.confirm(signal, symbol, timeframe, df)
