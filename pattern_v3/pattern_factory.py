# """
# pattern_factory.py - Main Pattern Pipeline Orchestrator for Pattern V4

# Ties together all components:
# - Detection Engine (23 pattern types with similarity scoring)
# - Intelligence Engine (context, liquidity, traps)
# - Execution Engine (draft/final trade setup)
# - Clustering Engine (pattern grouping & confluence)
# - Scoring Engine (confidence calculation)
# - Learning Engine (self-improvement)
# - MTF Engine (multi-timeframe confluence)

# Applies fail-fast at each step with detailed logging.
# Outputs clean PatternDecision objects for technical_pipeline / expert_aggregator.

# Version: 4.0
# Author: Pattern Intelligence System
# """

# import numpy as np
# import pandas as pd
# from typing import Dict, List, Optional, Tuple, Any
# from datetime import datetime
# import time
# import uuid

# from .pattern_debug import get_debugger_v4, quick_log, log_error
# from .pattern_core import PatternV4, PatternDirection, PatternStage, ActionType, Grade
# from .pattern_config import CONFIG, get_config, validate_config
# from .pattern_detection import PatternDetectionEngineV4
# from .pattern_intelligence import PatternIntelligenceEngineV4
# from .pattern_execution import PatternExecutionEngineV4
# from .pattern_clustering import PatternClusteringEngineV4Orchestrator
# from .pattern_scoring import PatternScoringEngineV4
# from .pattern_learning import PatternLearningEngineV4
# from .pattern_mtf import MultiTimeframeAnalyzerV4


# # ============================================================================
# # FAIL FAST SYSTEM
# # ============================================================================

# class FailFastSystemV4:
#     """
#     Early exit at each step to save computation.
#     Tracks failures and provides detailed reports.
    
#     USAGE:
#         if fail_fast.check(patterns is not None and len(patterns) > 0, 1, "No patterns"):
#             # This will NOT trigger because condition is TRUE (good)
            
#         if fail_fast.check(len(patterns) > 0, 1, "No patterns"):
#             # This WILL trigger if len(patterns) > 0 (bad)
#     """
    
#     def __init__(self):
#         self.fail_reasons = []
#         self.current_step = 0
#         self.step_names = {
#             1: "Detection",
#             2: "Intelligence",
#             3: "Lifecycle",
#             4: "Draft Execution",
#             5: "Clustering",
#             6: "Context Scoring",
#             7: "Scoring",
#             8: "Final Execution",
#             9: "Risk Filter",
#             10: "Trade Filter",
#         }
#         self.failure_count = 0
#         self.max_failures = 5
    
#     def check(self, condition: bool, step: int, reason: str, 
#               pattern_name: str = None) -> bool:
#         """
#         Check condition - if condition is TRUE (good), return False (continue)
#         If condition is FALSE (bad), record and return True (fail)
        
#         Returns:
#             True if FAILED (should exit)
#             False if PASSED (should continue)
#         """
#         if condition:
#             # Condition is TRUE - this is GOOD, continue
#             return False  # PASS - continue
#         else:
#             # Condition is FALSE - this is BAD, record failure
#             self.fail_reasons.append({
#                 'step': step,
#                 'step_name': self.step_names.get(step, f"Step{step}"),
#                 'reason': reason,
#                 'pattern_name': pattern_name,
#                 'timestamp': datetime.now().isoformat()
#             })
#             self.current_step = step
#             self.failure_count += 1
#             return True  # FAIL - exit
    
#     def is_failed(self) -> bool:
#         """Check if any failure occurred"""
#         return len(self.fail_reasons) > 0
    
#     def get_failure_report(self) -> Dict:
#         """Get detailed failure report"""
#         if not self.fail_reasons:
#             return {'failed': False}
        
#         last_failure = self.fail_reasons[-1]
#         return {
#             'failed': True,
#             'step': last_failure['step'],
#             'step_name': last_failure['step_name'],
#             'reason': last_failure['reason'],
#             'message': f"{last_failure['step_name']}: {last_failure['reason']}",
#             'all_failures': self.fail_reasons[-5:]
#         }
    
#     def reset(self):
#         """Reset for next pattern"""
#         self.fail_reasons = []
#         self.current_step = 0
    
#     def should_skip_symbol(self, symbol: str) -> bool:
#         """Check if symbol has too many failures"""
#         symbol_failures = [f for f in self.fail_reasons if f.get('symbol') == symbol]
#         return len(symbol_failures) >= self.max_failures


# # ============================================================================
# # PATTERN FACTORY V4 - MAIN PIPELINE
# # ============================================================================

# class PatternFactoryV4:
#     """
#     Main pattern pipeline orchestrator for V4.
#     Processes patterns through all stages with fail-fast and detailed logging.
#     """
    
#     def __init__(self, debug_mode: bool = True):
#         # Initialize debugger
#         self.debugger = get_debugger_v4()
#         self.debug_mode = debug_mode
        
#         # Initialize all engines
#         self.detection_engine = PatternDetectionEngineV4()
#         self.intelligence_engine = PatternIntelligenceEngineV4()
#         self.execution_engine = PatternExecutionEngineV4()
#         self.clustering_engine = PatternClusteringEngineV4Orchestrator()
#         self.scoring_engine = PatternScoringEngineV4()
#         self.learning_engine = PatternLearningEngineV4()
#         self.mtf_analyzer = MultiTimeframeAnalyzerV4()
        
#         # Fail fast system
#         self.fail_fast = FailFastSystemV4()
        
#         # Performance tracking
#         self.processing_times = []
#         self.patterns_processed = 0
#         self.patterns_accepted = 0
        
#         # Validate configuration
#         issues = validate_config() if hasattr(CONFIG, 'validate_config') else []
#         if issues:
#             self.debugger.log(f"⚠️ Config issues: {issues}", "WARNING")
        
#         self.debugger.log("✅ Pattern Factory V4 initialized")
#         self.debugger.log(f"   Debug Mode: {debug_mode}")
#         self.debugger.log(f"   Features: Liquidity={CONFIG.features.get('enable_liquidity_analysis', True) if hasattr(CONFIG, 'features') else True}, "
#                          f"Traps={CONFIG.features.get('enable_trap_detection', True) if hasattr(CONFIG, 'features') else True}, "
#                          f"Clustering={CONFIG.clustering_config.get('enabled', True) if hasattr(CONFIG, 'clustering_config') else True}")
    
#     def process_symbol(self, df: pd.DataFrame, symbol: str, 
#                        timeframe: str, htf_data: Dict[str, pd.DataFrame] = None,
#                        regime_data: Dict = None,
#                        liquidity_data: Dict = None,
#                        performance_history: Dict = None) -> List[Dict]:
#         """
#         Process all patterns for a single symbol.
#         Returns list of clean PatternDecision dicts for technical_pipeline.
#         """
        
#         start_time = time.time()
#         self.fail_fast.reset()
        
#         decisions = []
#         self.debugger.start_session(symbol, timeframe, df.shape)
        
#         # Load performance history if provided
#         if performance_history:
#             self.learning_engine.load_history(performance_history)
        
#         # ====================================================================
#         # STEP 1: DETECTION (23 pattern types with similarity scoring)
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"🔍 STEP 1: PATTERN DETECTION")
#         self.debugger.log(f"{'='*60}")
        
#         patterns, swings = self.detection_engine.detect_all_patterns(df, symbol, regime_data)



#         # ========== START OF DEBUG BLOCK - ADD THESE LINES ==========
#         self.debugger.log(f"   🔍 DEBUG: patterns type = {type(patterns)}")
#         self.debugger.log(f"   🔍 DEBUG: patterns length = {len(patterns)}")
#         self.debugger.log(f"   🔍 DEBUG: patterns memory id = {id(patterns)}")
#         self.debugger.log(f"   🔍 DEBUG: swings type = {type(swings)}")
#         self.debugger.log(f"   🔍 DEBUG: swings length = {len(swings)}")
        
#         # Log each pattern in detail
#         if patterns:
#             for i, p in enumerate(patterns):
#                 self.debugger.log(f"   🔍 DEBUG: pattern[{i}] = {p.get('pattern_name', 'Unknown')} - similarity: {p.get('similarity', 0):.3f} - direction: {p.get('direction', '?')}")
#         else:
#             self.debugger.log(f"   🔍 DEBUG: patterns list is EMPTY", "WARNING")

                
#         self.debugger.log(f"✅ Detection completed in {(time.time()-step_start)*1000:.1f}ms")
#         self.debugger.log(f"   Raw patterns found: {len(patterns)}")

        
#         # Log first few patterns
#         if patterns:
#             self.debugger.log(f"   First 5 patterns:")
#             for i, p in enumerate(patterns[:5]):
#                 self.debugger.log(f"      {i+1}. {p.get('pattern_name', 'Unknown')} ({p.get('direction', '?')}) - similarity: {p.get('similarity', 0):.1%}")
#                 self.debugger.log(f"   🔍 DEBUG: About to check fail-fast - patterns length = {len(patterns)}")
        
#         if self.fail_fast.check(len(patterns) > 0, 1, f"No patterns detected for {symbol}"):
#             self.debugger.log(f"❌ FAIL_FAST: No patterns detected", "WARNING")
#             self.debugger.log(f"   🔍 DEBUG: patterns was {len(patterns)} but fail-fast triggered!", "ERROR")
#             self.debugger.end_session(0)
#             return decisions
        
#         # ====================================================================
#         # STEP 2: INTELLIGENCE (Context, Liquidity, Traps)
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"🧠 STEP 2: PATTERN INTELLIGENCE")
#         self.debugger.log(f"{'='*60}")
        
#         enhanced_patterns = self.intelligence_engine.analyze_batch(
#             patterns, df, swings, regime_data, liquidity_data
#         )
        
#         self.debugger.log(f"✅ Intelligence completed in {(time.time()-step_start)*1000:.1f}ms")
#         self.debugger.log(f"   Patterns passed intelligence: {len(enhanced_patterns)}/{len(patterns)}")
        
#         if self.fail_fast.check(len(enhanced_patterns) > 0, 2, f"No patterns passed intelligence for {symbol}"):
#             self.debugger.log(f"❌ FAIL_FAST: No patterns passed intelligence", "WARNING")
#             self.debugger.end_session(0)
#             return decisions
#         # ====================================================================
#         # STEP 3: DRAFT EXECUTION (Early filter - RR must be >= 1.0)
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"📝 STEP 3: DRAFT EXECUTION")
#         self.debugger.log(f"{'='*60}")
        
#         patterns_with_draft = []
#         for pattern in enhanced_patterns:
#             pattern = self.execution_engine.draft_engine.generate_draft(pattern, df)
            
#             draft_rr = pattern.get('draft_rr', 0)
#             if draft_rr < 1.0:
#                 if draft_rr == 0:
#                     self.debugger.log(f"   ❌ {pattern.get('pattern_name', 'Unknown')}: Draft RR is 0 - generate_draft may have failed", "WARNING")
#                 else:
#                     self.debugger.log(f"   ❌ {pattern.get('pattern_name', 'Unknown')}: Poor draft RR ({draft_rr:.2f}) - skipping", "DEBUG")
#                 continue
            
#             patterns_with_draft.append(pattern)
#             self.debugger.log(f"   ✅ {pattern.get('pattern_name', 'Unknown')}: Draft RR = {pattern.get('draft_rr', 0):.2f}", "DEBUG")
        
#         self.debugger.log(f"✅ Draft execution completed in {(time.time()-step_start)*1000:.1f}ms")
#         self.debugger.log(f"   Patterns with valid draft: {len(patterns_with_draft)}/{len(enhanced_patterns)}")
        
#         if self.fail_fast.check(len(patterns_with_draft) > 0, 3, f"No patterns passed draft execution"):
#             self.debugger.log(f"❌ FAIL_FAST: No patterns passed draft execution", "WARNING")
#             self.debugger.end_session(0)
#             return decisions
        
#         # ====================================================================
#         # STEP 4: CLUSTERING (Group patterns by price and direction)
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"🔗 STEP 4: PATTERN CLUSTERING")
#         self.debugger.log(f"{'='*60}")
        
#         clustered_patterns = self.clustering_engine.process(
#             patterns_with_draft, df, len(df) - 1
#         )
        
#         self.debugger.log(f"✅ Clustering completed in {(time.time()-step_start)*1000:.1f}ms")
#         self.debugger.log(f"   Clustered patterns: {len(clustered_patterns)}")
        
#         # Log cluster summary
#         cluster_summary = self.clustering_engine.get_cluster_summary(clustered_patterns)
#         if cluster_summary.get('total_clusters', 0) > 0:
#             self.debugger.log(f"   Clusters found: {cluster_summary['total_clusters']}")
#             if cluster_summary.get('strongest_cluster'):
#                 sc = cluster_summary['strongest_cluster']
#                 self.debugger.log(f"   Strongest cluster: {sc.get('size', 0)} patterns, score: {sc.get('score', 0):.2f}")
        
#         # ====================================================================
#         # STEP 5: MTF CONFLUENCE (Multi-timeframe analysis)
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"📈 STEP 5: MTF CONFLUENCE")
#         self.debugger.log(f"{'='*60}")
        
#         for pattern in clustered_patterns:
#             if htf_data:
#                 mtf_result = self.mtf_analyzer.analyze_confluence(
#                     pattern, htf_data, timeframe
#                 )
#                 pattern['htf_confluence'] = mtf_result
#             else:
#                 pattern['htf_confluence'] = {'boost_factor': 1.0, 'weighted_score': 0.5}
        
#         self.debugger.log(f"✅ MTF analysis completed in {(time.time()-step_start)*1000:.1f}ms")
        
#         # ====================================================================
#         # STEP 6: SCORING (Context scores already calculated in intelligence)
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"📊 STEP 6: PATTERN SCORING")
#         self.debugger.log(f"{'='*60}")
        
#         regime = regime_data.get('regime', 'NEUTRAL') if regime_data else 'NEUTRAL'
#         scored_patterns = []
        
#         for pattern in clustered_patterns:
#             context_score = pattern.get('context_score', 0.5)
#             htf_confluence = pattern.get('htf_confluence', {})
            
#             # Get learning multiplier
#             pattern_name = pattern.get('pattern_name', 'Unknown')
#             learning_multiplier = self.learning_engine.get_pattern_weight_multiplier(pattern_name)
            
#             # Score the pattern
#             scored_pattern = self.scoring_engine.score_pattern(
#                 pattern, df, context_score, htf_confluence, 1.0, learning_multiplier
#             )
            
#             self.debugger.log(f"\n   {scored_pattern.get('pattern_name', 'Unknown')} ({scored_pattern.get('direction', '?')}):")
#             self.debugger.log(f"      Similarity: {scored_pattern.get('similarity', 0):.1%}")
#             self.debugger.log(f"      Context: {scored_pattern.get('context_score', 0):.1%}")
#             self.debugger.log(f"      Final Confidence: {scored_pattern.get('final_confidence', 0):.1%}")
#             self.debugger.log(f"      Grade: {scored_pattern.get('grade', 'F')}")
            
#             if scored_pattern.get('final_confidence', 0) < 0.25:
#                 self.debugger.log(f"      ❌ Confidence too low ({scored_pattern.get('final_confidence', 0):.1%}) - skipping")
#                 continue
            
#             scored_patterns.append(scored_pattern)
        
#         self.debugger.log(f"\n✅ Scoring completed in {(time.time()-step_start)*1000:.1f}ms")
#         self.debugger.log(f"   Patterns passed scoring: {len(scored_patterns)}/{len(clustered_patterns)}")
        
#         if self.fail_fast.check(len(scored_patterns) > 0, 7, f"No patterns passed scoring for {symbol}"):
#             self.debugger.log(f"❌ FAIL_FAST: No patterns passed scoring", "WARNING")
#             self.debugger.end_session(0)
#             return decisions

       
       
#                # ====================================================================
#         # STEP 7: FINAL EXECUTION (Refine entry/stop/target)
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"🎯 STEP 7: FINAL EXECUTION")
#         self.debugger.log(f"{'='*60}")
        
#         final_patterns = []
#         for pattern in scored_patterns:
#             # Get cluster score (from clustering step)
#             cluster_score = pattern.get('cluster_score', 0)
#             liquidity_score = pattern.get('liquidity', {}).get('score', 0.5)
#             regime = regime_data.get('regime', 'NEUTRAL') if regime_data else 'NEUTRAL'
#             final_confidence = pattern.get('final_confidence', 0.5)
            
#             # Generate final setup
#             pattern = self.execution_engine.generate_setup(
#                 pattern, df, final_confidence, cluster_score, 
#                 regime, liquidity_score, len(df) - 1
#             )
            
#             self.debugger.log(f"\n   {pattern.get('pattern_name', 'Unknown')}:")
#             self.debugger.log(f"      Entry: {pattern.get('entry', 0):.4f}")
#             self.debugger.log(f"      Stop: {pattern.get('stop_loss', 0):.4f}")
#             self.debugger.log(f"      Target: {pattern.get('take_profit', 0):.4f}")
#             self.debugger.log(f"      RR: {pattern.get('risk_reward', 0):.2f}")
#             self.debugger.log(f"      Action: {pattern.get('action', 'SKIP')}")
            
#             # REMOVED: RR check here - now handled in is_tradeable()
#             # if pattern.get('risk_reward', 0) < 1.2:
#             #     self.debugger.log(f"      ❌ RR too low ({pattern.get('risk_reward', 0):.2f}) - skipping")
#             #     continue
            
#             # Check tradeability (now only checks confidence and RR from config)
#             is_tradeable, reason = self.scoring_engine.is_tradeable(pattern)
#             if not is_tradeable:
#                 self.debugger.log(f"      ❌ Not tradeable: {reason}")
#                 continue
            
#             final_patterns.append(pattern)
        
#         # ====================================================================
#         # STEP 8: PATTERN COMPETITION & RANKING
#         # ====================================================================
#         step_start = time.time()
#         self.debugger.log(f"\n{'='*60}")
#         self.debugger.log(f"🏆 STEP 8: PATTERN COMPETITION")
#         self.debugger.log(f"{'='*60}")
        
#         # Rank by confidence
#         ranked_patterns = sorted(final_patterns, key=lambda p: p.get('final_confidence', 0), reverse=True)
        
#         # Log competition results
#         self.debugger.log(f"   Ranked patterns:")
#         for i, p in enumerate(ranked_patterns[:5], 1):
#             self.debugger.log(f"      {i}. {p.get('pattern_name', 'Unknown')} ({p.get('direction', '?')}): {p.get('final_confidence', 0):.1%}")
        
#         # Select best pattern
#         best_pattern = ranked_patterns[0] if ranked_patterns else None
        
#         # Boost if top 2 patterns agree on direction
#         if len(ranked_patterns) >= 2:
#             second = ranked_patterns[1]
#             if second.get('direction') == best_pattern.get('direction'):
#                 boost = 0.05
#                 best_pattern['final_confidence'] = min(0.95, best_pattern.get('final_confidence', 0) + boost)
#                 best_pattern['position_multiplier'] = min(1.5, best_pattern.get('position_multiplier', 1.0) + 0.1)
#                 best_pattern['reasons'].append(f"Agreement with {second.get('pattern_name', 'Unknown')}: +{boost:.0%}")
#                 self.debugger.log(f"\n   ✅ Boost applied: +{boost:.0%} for agreement with {second.get('pattern_name', 'Unknown')}")
        
#         self.debugger.log(f"\n✅ Competition completed in {(time.time()-step_start)*1000:.1f}ms")
#         self.debugger.log(f"   Selected: {best_pattern.get('pattern_name', 'Unknown')} ({best_pattern.get('direction', '?')}) with {best_pattern.get('final_confidence', 0):.1%} confidence")
        
#         # ====================================================================
#         # STEP 9: BUILD DECISION OUTPUT
#         # ====================================================================
#         if best_pattern:
#             decision = self._build_decision(best_pattern, symbol, timeframe)
#             decisions.append(decision)
#             self.patterns_accepted += 1
            
#             # Log the final decision
#             self.debugger.log_pattern_detection(best_pattern)
#             self.debugger.log_context_score(
#                 best_pattern.get('context_score', 0),
#                 best_pattern.get('context_components', {}),
#                 regime
#             )
#             self.debugger.log_mtf_confluence(best_pattern.get('htf_confluence', {}))
#             self.debugger.log_evolution(best_pattern.get('evolution', {}))
#             self.debugger.log_false_breakout_risk({
#                 'risk_score': best_pattern.get('false_breakout_risk', 0),
#                 'features': best_pattern.get('false_breakout_features', {})
#             })
#             self.debugger.log_final_confidence(
#                 best_pattern.get('final_confidence', 0),
#                 best_pattern.get('grade', 'F'),
#                 best_pattern.get('action', 'SKIP')
#             )
#             self.debugger.log_trade_setup(best_pattern)
#             self.debugger.log_decision(decision)
        
#         # ====================================================================
#         # SUMMARY
#         # ====================================================================
#         elapsed = time.time() - start_time
#         self.processing_times.append(elapsed)
#         self.patterns_processed += len(patterns)
        
#         # Calculate statistics
#         avg_similarity = np.mean([p.get('similarity', 0) for p in patterns]) if patterns else 0
#         avg_context = np.mean([p.get('context_score', 0) for p in enhanced_patterns]) if enhanced_patterns else 0
        
#         summary_stats = {
#             'total_detected': len(patterns),
#             'after_intelligence': len(enhanced_patterns),
#             'after_draft': len(patterns_with_draft),
#             'after_clustering': len(clustered_patterns),
#             'after_scoring': len(scored_patterns),
#             'tradeable': len(final_patterns),
#             'selected': len(decisions),
#             'processing_time_ms': elapsed * 1000,
#             'avg_similarity': avg_similarity,
#             'avg_context_score': avg_context,
#         }
        
#         self.debugger.log_summary(summary_stats)
        
#         # Log learning insights
#         learning_insights = self.learning_engine.analyze_failure_patterns() if hasattr(self.learning_engine, 'analyze_failure_patterns') else {}
#         if learning_insights:
#             self.debugger.log_learning_insights(learning_insights)
        
#         self.debugger.end_session(len(decisions))
        
#         # Keep only last 100 processing times
#         if len(self.processing_times) > 100:
#             self.processing_times = self.processing_times[-100:]
        
#         return decisions
    
#     def _build_decision(self, pattern: Dict, symbol: str, timeframe: str) -> Dict:
#         """
#         Build clean decision output for technical_pipeline / expert_aggregator.
#         This is the EXTERNAL output format.
#         """
        
#         # Build decision dictionary
#         decision = {
#             # Core identification
#             'pattern_id': str(uuid.uuid4())[:8],
#             'pattern_name': pattern.get('pattern_name', 'Unknown'),
#             'symbol': symbol,
#             'timeframe': timeframe,
#             'direction': pattern.get('direction', 'NEUTRAL'),
            
#             # Action
#             'action': pattern.get('action', 'SKIP'),
#             'action_detail': pattern.get('action_detail', ''),
            
#             # Trade setup
#             'entry': pattern.get('entry'),
#             'stop_loss': pattern.get('stop_loss'),
#             'take_profit': pattern.get('take_profit'),
#             'risk_reward': pattern.get('risk_reward', 0),
            
#             # Confidence and grade
#             'confidence': pattern.get('final_confidence', 0.5),
#             'grade': pattern.get('grade', 'F'),
            
#             # Position sizing
#             'position_multiplier': pattern.get('position_multiplier', 1.0),
            
#             # Decision reason
#             'decision_reason': pattern.get('decision_reason', self._build_reason(pattern)),
            
#             # For retest patterns
#             'retest_level': pattern.get('retest_level'),
#             'retest_confirmed': pattern.get('retest_confirmed', False),
            
#             # For active trade management
#             'stage': pattern.get('stage', 'FORMING'),
#             'completion_pct': pattern.get('completion_pct', 1.0),
#             'age_bars': pattern.get('age_bars', 0),
            
#             # Detailed scores (for logging/debugging)
#             'similarity': pattern.get('similarity', 0),
#             'context_score': pattern.get('context_score', 0),
#             'similarity_components': pattern.get('components', {}),
#             'context_components': pattern.get('context_components', {}),
            
#             # Reasons list
#             'reasons': pattern.get('reasons', []),
            
#             # Timestamp
#             'timestamp': datetime.now().isoformat(),
#             'decision_id': str(uuid.uuid4()),
#         }
        
#         return decision
    
#     def _build_reason(self, pattern: Dict) -> str:
#         """Build human-readable reason for decision"""
        
#         confidence = pattern.get('final_confidence', 0)
#         grade = pattern.get('grade', 'F')
#         pattern_name = pattern.get('pattern_name', 'Unknown')
#         direction = pattern.get('direction', 'NEUTRAL')
#         rr = pattern.get('risk_reward', 0)
        
#         if grade == 'A+':
#             return f"Exceptional setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - strong confluence across all factors"
#         elif grade == 'A':
#             return f"Excellent setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - high probability pattern"
#         elif grade == 'B+':
#             return f"Very good setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - solid confluence"
#         elif grade == 'B':
#             return f"Good setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - meets all criteria"
#         elif grade == 'B-':
#             return f"Above average setup: {pattern_name} ({direction}) with {confidence:.0%} confidence - acceptable risk"
#         else:
#             return f"{pattern_name} ({direction}) - {pattern.get('action_detail', 'Pattern detected')}"
    
#     def process_batch(self, dataframes: Dict[str, pd.DataFrame], 
#                       symbol: str, timeframe: str,
#                       htf_data: Dict[str, pd.DataFrame] = None,
#                       regime_data: Dict = None,
#                       liquidity_data: Dict = None) -> List[Dict]:
#         """
#         Process multiple timeframes or multiple symbols.
#         """
        
#         all_decisions = []
        
#         for tf, df in dataframes.items():
#             decisions = self.process_symbol(
#                 df, symbol, tf, htf_data, regime_data, liquidity_data
#             )
#             all_decisions.extend(decisions)
        
#         return all_decisions
    
#     def get_performance_stats(self) -> Dict:
#         """Get performance statistics"""
        
#         avg_time = np.mean(self.processing_times) if self.processing_times else 0
        
#         return {
#             'total_patterns_processed': self.patterns_processed,
#             'total_patterns_accepted': self.patterns_accepted,
#             'acceptance_rate': self.patterns_accepted / max(1, self.patterns_processed),
#             'avg_processing_time_ms': avg_time * 1000,
#             'fail_fast_failures': len(self.fail_fast.fail_reasons),
#             'last_failure': self.fail_fast.get_failure_report() if self.fail_fast.is_failed() else None
#         }
    
#     def reset_stats(self):
#         """Reset performance statistics"""
#         self.processing_times = []
#         self.patterns_processed = 0
#         self.patterns_accepted = 0
#         self.fail_fast.fail_reasons = []
    
#     def print_debug_summary(self):
#         """Print debug summary to console"""
#         self.debugger.print_summary()


# # ============================================================================
# # LEGACY COMPATIBILITY WRAPPER
# # ============================================================================

# class PatternFactoryLegacyV4:
#     """
#     Wrapper for backward compatibility with existing enhanced_patterns.py.
#     Converts legacy patterns to PatternV4 and processes them.
#     """
    
#     def __init__(self):
#         self.factory = PatternFactoryV4()
    
#     def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
#         """
#         Legacy method - returns patterns as dicts.
#         """
#         decisions = self.factory.process_symbol(df, "UNKNOWN", "unknown")
#         return decisions
    
#     def get_pattern_score(self, df: pd.DataFrame, direction: str = 'BUY') -> float:
#         """
#         Legacy method - returns aggregate score.
#         """
#         decisions = self.factory.process_symbol(df, "UNKNOWN", "unknown")
        
#         if not decisions:
#             return 0.5
        
#         if direction == 'BUY':
#             buy_decisions = [d for d in decisions if d['direction'] == 'BUY' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW']]
#             if buy_decisions:
#                 return max(d['confidence'] for d in buy_decisions)
#         else:
#             sell_decisions = [d for d in decisions if d['direction'] == 'SELL' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW']]
#             if sell_decisions:
#                 return max(d['confidence'] for d in sell_decisions)
        
#         return 0.5
    
#     def get_net_bias(self, df: pd.DataFrame) -> float:
#         """
#         Legacy method - returns net bias.
#         """
#         decisions = self.factory.process_symbol(df, "UNKNOWN", "unknown")
        
#         if not decisions:
#             return 0.0
        
#         buy_confidence = sum(d['confidence'] for d in decisions if d['direction'] == 'BUY' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW'])
#         sell_confidence = sum(d['confidence'] for d in decisions if d['direction'] == 'SELL' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW'])
        
#         total = buy_confidence + sell_confidence
#         if total == 0:
#             return 0.0
        
#         return (buy_confidence - sell_confidence) / total


# # ============================================================================
# # QUICK TEST FUNCTION
# # ============================================================================

# def test_pattern_factory_v4(symbol: str = "BTC/USDT", 
#                             timeframe: str = "1h",
#                             lookback: int = 200):
#     """
#     Test Pattern Factory V4 with sample data.
#     """
#     print("\n" + "="*80)
#     print(f"🔬 PATTERN V4 FACTORY TEST")
#     print("="*80)
#     print(f"Symbol: {symbol}")
#     print(f"Timeframe: {timeframe}")
#     print(f"Lookback: {lookback} candles")
#     print("="*80)
    
#     # Create sample data
#     dates = pd.date_range(start='2024-01-01', periods=lookback, freq='1h')
#     np.random.seed(42)
    
#     # Create trending data with patterns
#     trend = np.linspace(0, 0.1, lookback)
#     noise = np.random.randn(lookback) * 0.005
#     close = 100 * (1 + trend + noise)
    
#     df = pd.DataFrame({
#         'open': close * (1 + np.random.randn(lookback) * 0.002),
#         'high': close * (1 + abs(np.random.randn(lookback) * 0.005)),
#         'low': close * (1 - abs(np.random.randn(lookback) * 0.005)),
#         'close': close,
#         'volume': np.random.randint(1000, 10000, lookback)
#     }, index=dates)
    
#     # Ensure high is highest, low is lowest
#     df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(lookback) * 0.5)
#     df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(lookback) * 0.5)
    
#     # Add indicators
#     df['atr'] = (df['high'] - df['low']).rolling(14).mean()
#     df['rsi'] = 50 + np.random.randn(lookback) * 10
    
#     print(f"\n✅ Sample data created: {len(df)} candles")
#     print(f"   Current price: {df['close'].iloc[-1]:.2f}")
    
#     # Initialize factory
#     factory = PatternFactoryV4(debug_mode=True)
    
#     # Process
#     print("\n🔄 Processing patterns...")
#     decisions = factory.process_symbol(df, symbol, timeframe)
    
#     # Print results
#     print("\n" + "="*80)
#     print("📊 RESULTS")
#     print("="*80)
    
#     if decisions:
#         print(f"✅ Generated {len(decisions)} trading decisions:\n")
#         for i, d in enumerate(decisions, 1):
#             print(f"{i}. {d['pattern_name']} ({d['direction']})")
#             print(f"   Action: {d['action']}")
#             print(f"   Confidence: {d['confidence']:.1%}")
#             print(f"   Grade: {d['grade']}")
#             print(f"   Entry: {d['entry']:.4f}")
#             print(f"   Stop: {d['stop_loss']:.4f}")
#             print(f"   Target: {d['take_profit']:.4f}")
#             print(f"   RR: {d['risk_reward']:.2f}")
#             print(f"   Reason: {d['decision_reason'][:100]}...")
#             print()
#     else:
#         print("❌ No trading decisions generated")
    
#     # Print performance stats
#     stats = factory.get_performance_stats()
#     print("\n📈 PERFORMANCE STATS:")
#     print(f"   Patterns processed: {stats['total_patterns_processed']}")
#     print(f"   Patterns accepted: {stats['total_patterns_accepted']}")
#     print(f"   Acceptance rate: {stats['acceptance_rate']:.1%}")
#     print(f"   Avg processing time: {stats['avg_processing_time_ms']:.1f}ms")
    
#     factory.print_debug_summary()
    
#     return decisions


# # ============================================================================
# # EXPORTS
# # ============================================================================

# __all__ = [
#     'FailFastSystemV4',
#     'PatternFactoryV4',
#     'PatternFactoryLegacyV4',
#     'test_pattern_factory_v4',
# ]























"""
pattern_factory.py - Main Pattern Pipeline Orchestrator for Pattern V4

Ties together all components:
- Detection Engine (23 pattern types with similarity scoring)
- Intelligence Engine (context, liquidity, traps)
- Execution Engine (draft/final trade setup)
- Clustering Engine (pattern grouping & confluence)
- Scoring Engine (confidence calculation)
- Learning Engine (self-improvement)
- MTF Engine (multi-timeframe confluence)

Applies fail-fast at each step with detailed logging.
Outputs clean PatternDecision objects for technical_pipeline / expert_aggregator.

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import time
import uuid

from .pattern_debug import get_debugger_v4, quick_log, log_error
from .pattern_core import PatternV4, PatternDirection, PatternStage, ActionType, Grade
from .pattern_config import CONFIG, get_config, validate_config
from .pattern_detection import PatternDetectionEngineV4
from .pattern_intelligence import PatternIntelligenceEngineV4
from .pattern_execution import PatternExecutionEngineV4
from .pattern_clustering import PatternClusteringEngineV4Orchestrator
from .pattern_scoring import PatternScoringEngineV4
from .pattern_learning import PatternLearningEngineV4
from .pattern_mtf import MultiTimeframeAnalyzerV4


# ============================================================================
# FAIL FAST SYSTEM
# ============================================================================

class FailFastSystemV4:
    """
    Early exit at each step to save computation.
    Tracks failures and provides detailed reports.
    
    USAGE:
        if fail_fast.check(patterns is not None and len(patterns) > 0, 1, "No patterns"):
            # This will NOT trigger because condition is TRUE (good)
            
        if fail_fast.check(len(patterns) > 0, 1, "No patterns"):
            # This WILL trigger if len(patterns) > 0 (bad)
    """
    
    def __init__(self):
        self.fail_reasons = []
        self.current_step = 0
        self.step_names = {
            1: "Detection",
            2: "Intelligence",
            3: "Lifecycle",
            4: "Draft Execution",
            5: "Clustering",
            6: "Context Scoring",
            7: "Scoring",
            8: "Final Execution",
            9: "Risk Filter",
            10: "Trade Filter",
        }
        self.failure_count = 0
        self.max_failures = 5
    
    def check(self, condition: bool, step: int, reason: str, 
              pattern_name: str = None) -> bool:
        """
        Check condition - if condition is TRUE (good), return False (continue)
        If condition is FALSE (bad), record and return True (fail)
        
        Returns:
            True if FAILED (should exit)
            False if PASSED (should continue)
        """
        if condition:
            # Condition is TRUE - this is GOOD, continue
            return False  # PASS - continue
        else:
            # Condition is FALSE - this is BAD, record failure
            self.fail_reasons.append({
                'step': step,
                'step_name': self.step_names.get(step, f"Step{step}"),
                'reason': reason,
                'pattern_name': pattern_name,
                'timestamp': datetime.now().isoformat()
            })
            self.current_step = step
            self.failure_count += 1
            return True  # FAIL - exit
    
    def is_failed(self) -> bool:
        """Check if any failure occurred"""
        return len(self.fail_reasons) > 0
    
    def get_failure_report(self) -> Dict:
        """Get detailed failure report"""
        if not self.fail_reasons:
            return {'failed': False}
        
        last_failure = self.fail_reasons[-1]
        return {
            'failed': True,
            'step': last_failure['step'],
            'step_name': last_failure['step_name'],
            'reason': last_failure['reason'],
            'message': f"{last_failure['step_name']}: {last_failure['reason']}",
            'all_failures': self.fail_reasons[-5:]
        }
    
    def reset(self):
        """Reset for next pattern"""
        self.fail_reasons = []
        self.current_step = 0
    
    def should_skip_symbol(self, symbol: str) -> bool:
        """Check if symbol has too many failures"""
        symbol_failures = [f for f in self.fail_reasons if f.get('symbol') == symbol]
        return len(symbol_failures) >= self.max_failures


# ============================================================================
# PATTERN FACTORY V4 - MAIN PIPELINE
# ============================================================================

class PatternFactoryV4:
    """
    Main pattern pipeline orchestrator for V4.
    Processes patterns through all stages with fail-fast and detailed logging.
    """
    
    def __init__(self, debug_mode: bool = True):
        # Initialize debugger
        self.debugger = get_debugger_v4()
        self.debug_mode = debug_mode
        
        # Initialize all engines
        self.detection_engine = PatternDetectionEngineV4()
        self.intelligence_engine = PatternIntelligenceEngineV4()
        self.execution_engine = PatternExecutionEngineV4()
        self.clustering_engine = PatternClusteringEngineV4Orchestrator()
        self.scoring_engine = PatternScoringEngineV4()
        self.learning_engine = PatternLearningEngineV4()
        self.mtf_analyzer = MultiTimeframeAnalyzerV4()
        
        # Fail fast system
        self.fail_fast = FailFastSystemV4()
        
        # Performance tracking
        self.processing_times = []
        self.patterns_processed = 0
        self.patterns_accepted = 0
        
        # Validate configuration
        issues = validate_config() if hasattr(CONFIG, 'validate_config') else []
        if issues:
            self.debugger.log(f"⚠️ Config issues: {issues}", "WARNING")
        
        self.debugger.log("✅ Pattern Factory V4 initialized")
        self.debugger.log(f"   Debug Mode: {debug_mode}")
        self.debugger.log(f"   Features: Liquidity={CONFIG.features.get('enable_liquidity_analysis', True) if hasattr(CONFIG, 'features') else True}, "
                         f"Traps={CONFIG.features.get('enable_trap_detection', True) if hasattr(CONFIG, 'features') else True}, "
                         f"Clustering={CONFIG.clustering_config.get('enabled', True) if hasattr(CONFIG, 'clustering_config') else True}")
    
    def process_symbol(self, df: pd.DataFrame, symbol: str, 
                       timeframe: str, htf_data: Dict[str, pd.DataFrame] = None,
                       regime_data: Dict = None,
                       liquidity_data: Dict = None,
                       performance_history: Dict = None) -> List[Dict]:
        """
        Process all patterns for a single symbol.
        Returns list of clean PatternDecision dicts for technical_pipeline.
        """
        
        start_time = time.time()
        self.fail_fast.reset()
        
        decisions = []
        self.debugger.start_session(symbol, timeframe, df.shape)
        
        # Load performance history if provided
        if performance_history:
            self.learning_engine.load_history(performance_history)
        
        # ====================================================================
        # STEP 1: DETECTION (23 pattern types with similarity scoring)
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"🔍 STEP 1: PATTERN DETECTION")
        self.debugger.log(f"{'='*60}")
        
        patterns, swings = self.detection_engine.detect_all_patterns(df, symbol, regime_data)



        # ========== START OF DEBUG BLOCK - ADD THESE LINES ==========
        self.debugger.log(f"   🔍 DEBUG: patterns type = {type(patterns)}")
        self.debugger.log(f"   🔍 DEBUG: patterns length = {len(patterns)}")
        self.debugger.log(f"   🔍 DEBUG: patterns memory id = {id(patterns)}")
        self.debugger.log(f"   🔍 DEBUG: swings type = {type(swings)}")
        self.debugger.log(f"   🔍 DEBUG: swings length = {len(swings)}")
        
        # Log each pattern in detail
        if patterns:
            for i, p in enumerate(patterns):
                self.debugger.log(f"   🔍 DEBUG: pattern[{i}] = {p.get('pattern_name', 'Unknown')} - similarity: {p.get('similarity', 0):.3f} - direction: {p.get('direction', '?')}")
        else:
            self.debugger.log(f"   🔍 DEBUG: patterns list is EMPTY", "WARNING")

                
        self.debugger.log(f"✅ Detection completed in {(time.time()-step_start)*1000:.1f}ms")
        self.debugger.log(f"   Raw patterns found: {len(patterns)}")

        
        # Log first few patterns
        if patterns:
            self.debugger.log(f"   First 5 patterns:")
            for i, p in enumerate(patterns[:5]):
                self.debugger.log(f"      {i+1}. {p.get('pattern_name', 'Unknown')} ({p.get('direction', '?')}) - similarity: {p.get('similarity', 0):.1%}")
                self.debugger.log(f"   🔍 DEBUG: About to check fail-fast - patterns length = {len(patterns)}")
        
        if self.fail_fast.check(len(patterns) > 0, 1, f"No patterns detected for {symbol}"):
            self.debugger.log(f"❌ FAIL_FAST: No patterns detected", "WARNING")
            self.debugger.log(f"   🔍 DEBUG: patterns was {len(patterns)} but fail-fast triggered!", "ERROR")
            self.debugger.end_session(0)
            return decisions
        
        # ====================================================================
        # STEP 2: INTELLIGENCE (Context, Liquidity, Traps)
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"🧠 STEP 2: PATTERN INTELLIGENCE")
        self.debugger.log(f"{'='*60}")
        
        enhanced_patterns = self.intelligence_engine.analyze_batch(
            patterns, df, swings, regime_data, liquidity_data
        )
        
        self.debugger.log(f"✅ Intelligence completed in {(time.time()-step_start)*1000:.1f}ms")
        self.debugger.log(f"   Patterns passed intelligence: {len(enhanced_patterns)}/{len(patterns)}")
        
        if self.fail_fast.check(len(enhanced_patterns) == 0, 2, f"No patterns passed intelligence for {symbol}"):
            self.debugger.log(f"❌ FAIL_FAST: No patterns passed intelligence", "WARNING")
            self.debugger.end_session(0)
            return decisions
        # ====================================================================
        # STEP 3: DRAFT EXECUTION (Early filter - RR must be >= 1.0)
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"📝 STEP 3: DRAFT EXECUTION")
        self.debugger.log(f"{'='*60}")
        
        patterns_with_draft = []
        for pattern in enhanced_patterns:
            pattern = self.execution_engine.draft_engine.generate_draft(pattern, df)
            
            draft_rr = pattern.get('draft_rr', 0)
            if draft_rr < 1.0:
                if draft_rr == 0:
                    self.debugger.log(f"   ❌ {pattern.get('pattern_name', 'Unknown')}: Draft RR is 0 - generate_draft may have failed", "WARNING")
                else:
                    self.debugger.log(f"   ❌ {pattern.get('pattern_name', 'Unknown')}: Poor draft RR ({draft_rr:.2f}) - skipping", "DEBUG")
                continue
            
            patterns_with_draft.append(pattern)
            self.debugger.log(f"   ✅ {pattern.get('pattern_name', 'Unknown')}: Draft RR = {pattern.get('draft_rr', 0):.2f}", "DEBUG")
        
        self.debugger.log(f"✅ Draft execution completed in {(time.time()-step_start)*1000:.1f}ms")
        self.debugger.log(f"   Patterns with valid draft: {len(patterns_with_draft)}/{len(enhanced_patterns)}")
        
        if self.fail_fast.check(len(patterns_with_draft) == 0, 3, f"No patterns passed draft execution"):
            self.debugger.log(f"❌ FAIL_FAST: No patterns passed draft execution", "WARNING")
            self.debugger.end_session(0)
            return decisions
        
        # ====================================================================
        # STEP 4: CLUSTERING (Group patterns by price and direction)
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"🔗 STEP 4: PATTERN CLUSTERING")
        self.debugger.log(f"{'='*60}")
        
        clustered_patterns = self.clustering_engine.process(
            patterns_with_draft, df, len(df) - 1
        )
        
        self.debugger.log(f"✅ Clustering completed in {(time.time()-step_start)*1000:.1f}ms")
        self.debugger.log(f"   Clustered patterns: {len(clustered_patterns)}")
        
        # Log cluster summary
        cluster_summary = self.clustering_engine.get_cluster_summary(clustered_patterns)
        if cluster_summary.get('total_clusters', 0) > 0:
            self.debugger.log(f"   Clusters found: {cluster_summary['total_clusters']}")
            if cluster_summary.get('strongest_cluster'):
                sc = cluster_summary['strongest_cluster']
                self.debugger.log(f"   Strongest cluster: {sc.get('size', 0)} patterns, score: {sc.get('score', 0):.2f}")
        
        # ====================================================================
        # STEP 5: MTF CONFLUENCE (Multi-timeframe analysis)
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"📈 STEP 5: MTF CONFLUENCE")
        self.debugger.log(f"{'='*60}")
        
        for pattern in clustered_patterns:
            if htf_data:
                mtf_result = self.mtf_analyzer.analyze_confluence(
                    pattern, htf_data, timeframe
                )
                pattern['htf_confluence'] = mtf_result
            else:
                pattern['htf_confluence'] = {'boost_factor': 1.0, 'weighted_score': 0.5}
        
        self.debugger.log(f"✅ MTF analysis completed in {(time.time()-step_start)*1000:.1f}ms")
        
        # ====================================================================
        # STEP 6: SCORING (Context scores already calculated in intelligence)
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"📊 STEP 6: PATTERN SCORING")
        self.debugger.log(f"{'='*60}")
        
        regime = regime_data.get('regime', 'NEUTRAL') if regime_data else 'NEUTRAL'
        scored_patterns = []
        
        for pattern in clustered_patterns:
            context_score = pattern.get('context_score', 0.5)
            htf_confluence = pattern.get('htf_confluence', {})
            
            # Get learning multiplier
            pattern_name = pattern.get('pattern_name', 'Unknown')
            learning_multiplier = self.learning_engine.get_pattern_weight_multiplier(pattern_name)
            
            # Score the pattern
            scored_pattern = self.scoring_engine.score_pattern(
                pattern, df, context_score, htf_confluence, 1.0, learning_multiplier
            )
            
            self.debugger.log(f"\n   {scored_pattern.get('pattern_name', 'Unknown')} ({scored_pattern.get('direction', '?')}):")
            self.debugger.log(f"      Similarity: {scored_pattern.get('similarity', 0):.1%}")
            self.debugger.log(f"      Context: {scored_pattern.get('context_score', 0):.1%}")
            self.debugger.log(f"      Final Confidence: {scored_pattern.get('final_confidence', 0):.1%}")
            self.debugger.log(f"      Grade: {scored_pattern.get('grade', 'F')}")
            
            if scored_pattern.get('final_confidence', 0) < CONFIG.min_trade_confidence:
                self.debugger.log(f"      ❌ Confidence too low ({scored_pattern.get('final_confidence', 0):.1%} < {CONFIG.min_trade_confidence:.0%}) - skipping")
                continue
            
            scored_patterns.append(scored_pattern)
        
        self.debugger.log(f"\n✅ Scoring completed in {(time.time()-step_start)*1000:.1f}ms")
        self.debugger.log(f"   Patterns passed scoring: {len(scored_patterns)}/{len(clustered_patterns)}")
        
        if self.fail_fast.check(len(scored_patterns) == 0, 7, f"No patterns passed scoring for {symbol}"):
            self.debugger.log(f"❌ FAIL_FAST: No patterns passed scoring", "WARNING")
            self.debugger.end_session(0)
            return decisions

       
       
               # ====================================================================
        # STEP 7: FINAL EXECUTION (Refine entry/stop/target)
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"🎯 STEP 7: FINAL EXECUTION")
        self.debugger.log(f"{'='*60}")
        
        final_patterns = []
        for pattern in scored_patterns:
            # Get cluster score (from clustering step)
            cluster_score = pattern.get('cluster_score', 0)
            liquidity_score = pattern.get('liquidity', {}).get('score', 0.5)
            regime = regime_data.get('regime', 'NEUTRAL') if regime_data else 'NEUTRAL'
            final_confidence = pattern.get('final_confidence', 0.5)
            
            # Generate final setup
            pattern = self.execution_engine.generate_setup(
                pattern, df, final_confidence, cluster_score, 
                regime, liquidity_score, len(df) - 1
            )
            
            self.debugger.log(f"\n   {pattern.get('pattern_name', 'Unknown')}:")
            self.debugger.log(f"      Entry: {pattern.get('entry', 0):.4f}")
            self.debugger.log(f"      Stop: {pattern.get('stop_loss', 0):.4f}")
            self.debugger.log(f"      Target: {pattern.get('take_profit', 0):.4f}")
            self.debugger.log(f"      RR: {pattern.get('risk_reward', 0):.2f}")
            self.debugger.log(f"      Action: {pattern.get('action', 'SKIP')}")
            
            # REMOVED: RR check here - now handled in is_tradeable()
            # if pattern.get('risk_reward', 0) < 1.2:
            #     self.debugger.log(f"      ❌ RR too low ({pattern.get('risk_reward', 0):.2f}) - skipping")
            #     continue
            
            # Check tradeability (now only checks confidence and RR from config)
            is_tradeable, reason = self.scoring_engine.is_tradeable(pattern)
            if not is_tradeable:
                self.debugger.log(f"      ❌ Not tradeable: {reason}")
                continue
            
            final_patterns.append(pattern)
        
        # ====================================================================
        # STEP 8: PATTERN COMPETITION & RANKING
        # ====================================================================
        step_start = time.time()
        self.debugger.log(f"\n{'='*60}")
        self.debugger.log(f"🏆 STEP 8: PATTERN COMPETITION")
        self.debugger.log(f"{'='*60}")
        
        # Rank by confidence
        ranked_patterns = sorted(final_patterns, key=lambda p: p.get('final_confidence', 0), reverse=True)
        
        # Log competition results
        self.debugger.log(f"   Ranked patterns:")
        for i, p in enumerate(ranked_patterns[:5], 1):
            self.debugger.log(f"      {i}. {p.get('pattern_name', 'Unknown')} ({p.get('direction', '?')}): {p.get('final_confidence', 0):.1%}")
        
        # Select best pattern
        best_pattern = ranked_patterns[0] if ranked_patterns else None
        
        # Boost if top 2 patterns agree on direction
        if len(ranked_patterns) >= 2:
            second = ranked_patterns[1]
            if second.get('direction') == best_pattern.get('direction'):
                boost = 0.05
                best_pattern['final_confidence'] = min(0.95, best_pattern.get('final_confidence', 0) + boost)
                best_pattern['position_multiplier'] = min(1.5, best_pattern.get('position_multiplier', 1.0) + 0.1)
                best_pattern['reasons'].append(f"Agreement with {second.get('pattern_name', 'Unknown')}: +{boost:.0%}")
                self.debugger.log(f"\n   ✅ Boost applied: +{boost:.0%} for agreement with {second.get('pattern_name', 'Unknown')}")
        
        self.debugger.log(f"\n✅ Competition completed in {(time.time()-step_start)*1000:.1f}ms")
        self.debugger.log(f"   Selected: {best_pattern.get('pattern_name', 'Unknown')} ({best_pattern.get('direction', '?')}) with {best_pattern.get('final_confidence', 0):.1%} confidence")
        
        # ====================================================================
        # STEP 9: BUILD DECISION OUTPUT
        # ====================================================================
        if best_pattern:
            decision = self._build_decision(best_pattern, symbol, timeframe)
            decisions.append(decision)
            self.patterns_accepted += 1
            
            # Log the final decision
            self.debugger.log_pattern_detection(best_pattern)
            self.debugger.log_context_score(
                best_pattern.get('context_score', 0),
                best_pattern.get('context_components', {}),
                regime
            )
            self.debugger.log_mtf_confluence(best_pattern.get('htf_confluence', {}))
            self.debugger.log_evolution(best_pattern.get('evolution', {}))
            self.debugger.log_false_breakout_risk({
                'risk_score': best_pattern.get('false_breakout_risk', 0),
                'features': best_pattern.get('false_breakout_features', {})
            })
            self.debugger.log_final_confidence(
                best_pattern.get('final_confidence', 0),
                best_pattern.get('grade', 'F'),
                best_pattern.get('action', 'SKIP')
            )
            self.debugger.log_trade_setup(best_pattern)
            self.debugger.log_decision(decision)
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        elapsed = time.time() - start_time
        self.processing_times.append(elapsed)
        self.patterns_processed += len(patterns)
        
        # Calculate statistics
        avg_similarity = np.mean([p.get('similarity', 0) for p in patterns]) if patterns else 0
        avg_context = np.mean([p.get('context_score', 0) for p in enhanced_patterns]) if enhanced_patterns else 0
        
        summary_stats = {
            'total_detected': len(patterns),
            'after_intelligence': len(enhanced_patterns),
            'after_draft': len(patterns_with_draft),
            'after_clustering': len(clustered_patterns),
            'after_scoring': len(scored_patterns),
            'tradeable': len(final_patterns),
            'selected': len(decisions),
            'processing_time_ms': elapsed * 1000,
            'avg_similarity': avg_similarity,
            'avg_context_score': avg_context,
        }
        
        self.debugger.log_summary(summary_stats)
        
        # Log learning insights
        learning_insights = self.learning_engine.analyze_failure_patterns() if hasattr(self.learning_engine, 'analyze_failure_patterns') else {}
        if learning_insights:
            self.debugger.log_learning_insights(learning_insights)
        
        self.debugger.end_session(len(decisions))
        
        # Keep only last 100 processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        return decisions
    
    def _build_decision(self, pattern: Dict, symbol: str, timeframe: str) -> Dict:
        """
        Build clean decision output for technical_pipeline / expert_aggregator.
        This is the EXTERNAL output format.
        """
        
        # Build decision dictionary
        decision = {
            # Core identification
            'pattern_id': str(uuid.uuid4())[:8],
            'pattern_name': pattern.get('pattern_name', 'Unknown'),
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': pattern.get('direction', 'NEUTRAL'),
            
            # Action
            'action': pattern.get('action', 'SKIP'),
            'action_detail': pattern.get('action_detail', ''),
            
            # Trade setup
            'entry': pattern.get('entry'),
            'stop_loss': pattern.get('stop_loss'),
            'take_profit': pattern.get('take_profit'),
            'risk_reward': pattern.get('risk_reward', 0),
            
            # Confidence and grade
            'confidence': pattern.get('final_confidence', 0.5),
            'grade': pattern.get('grade', 'F'),
            
            # Position sizing
            'position_multiplier': pattern.get('position_multiplier', 1.0),
            
            # Decision reason
            'decision_reason': pattern.get('decision_reason', self._build_reason(pattern)),
            
            # For retest patterns
            'retest_level': pattern.get('retest_level'),
            'retest_confirmed': pattern.get('retest_confirmed', False),
            
            # For active trade management
            'stage': pattern.get('stage', 'FORMING'),
            'completion_pct': pattern.get('completion_pct', 1.0),
            'age_bars': pattern.get('age_bars', 0),
            
            # Detailed scores (for logging/debugging)
            'similarity': pattern.get('similarity', 0),
            'context_score': pattern.get('context_score', 0),
            'similarity_components': pattern.get('components', {}),
            'context_components': pattern.get('context_components', {}),
            
            # Reasons list
            'reasons': pattern.get('reasons', []),
            
            # Timestamp
            'timestamp': datetime.now().isoformat(),
            'decision_id': str(uuid.uuid4()),
        }
        
        return decision
    
    def _build_reason(self, pattern: Dict) -> str:
        """Build human-readable reason for decision"""
        
        confidence = pattern.get('final_confidence', 0)
        grade = pattern.get('grade', 'F')
        pattern_name = pattern.get('pattern_name', 'Unknown')
        direction = pattern.get('direction', 'NEUTRAL')
        rr = pattern.get('risk_reward', 0)
        
        if grade == 'A+':
            return f"Exceptional setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - strong confluence across all factors"
        elif grade == 'A':
            return f"Excellent setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - high probability pattern"
        elif grade == 'B+':
            return f"Very good setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - solid confluence"
        elif grade == 'B':
            return f"Good setup: {pattern_name} ({direction}) with {confidence:.0%} confidence, {rr:.2f} RR - meets all criteria"
        elif grade == 'B-':
            return f"Above average setup: {pattern_name} ({direction}) with {confidence:.0%} confidence - acceptable risk"
        else:
            return f"{pattern_name} ({direction}) - {pattern.get('action_detail', 'Pattern detected')}"
    
    def process_batch(self, dataframes: Dict[str, pd.DataFrame], 
                      symbol: str, timeframe: str,
                      htf_data: Dict[str, pd.DataFrame] = None,
                      regime_data: Dict = None,
                      liquidity_data: Dict = None) -> List[Dict]:
        """
        Process multiple timeframes or multiple symbols.
        """
        
        all_decisions = []
        
        for tf, df in dataframes.items():
            decisions = self.process_symbol(
                df, symbol, tf, htf_data, regime_data, liquidity_data
            )
            all_decisions.extend(decisions)
        
        return all_decisions
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'total_patterns_processed': self.patterns_processed,
            'total_patterns_accepted': self.patterns_accepted,
            'acceptance_rate': self.patterns_accepted / max(1, self.patterns_processed),
            'avg_processing_time_ms': avg_time * 1000,
            'fail_fast_failures': len(self.fail_fast.fail_reasons),
            'last_failure': self.fail_fast.get_failure_report() if self.fail_fast.is_failed() else None
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.processing_times = []
        self.patterns_processed = 0
        self.patterns_accepted = 0
        self.fail_fast.fail_reasons = []
    
    def print_debug_summary(self):
        """Print debug summary to console"""
        self.debugger.print_summary()


# ============================================================================
# LEGACY COMPATIBILITY WRAPPER
# ============================================================================

class PatternFactoryLegacyV4:
    """
    Wrapper for backward compatibility with existing enhanced_patterns.py.
    Converts legacy patterns to PatternV4 and processes them.
    """
    
    def __init__(self):
        self.factory = PatternFactoryV4()
    
    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Legacy method - returns patterns as dicts.
        """
        decisions = self.factory.process_symbol(df, "UNKNOWN", "unknown")
        return decisions
    
    def get_pattern_score(self, df: pd.DataFrame, direction: str = 'BUY') -> float:
        """
        Legacy method - returns aggregate score.
        """
        decisions = self.factory.process_symbol(df, "UNKNOWN", "unknown")
        
        if not decisions:
            return 0.5
        
        if direction == 'BUY':
            buy_decisions = [d for d in decisions if d['direction'] == 'BUY' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW']]
            if buy_decisions:
                return max(d['confidence'] for d in buy_decisions)
        else:
            sell_decisions = [d for d in decisions if d['direction'] == 'SELL' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW']]
            if sell_decisions:
                return max(d['confidence'] for d in sell_decisions)
        
        return 0.5
    
    def get_net_bias(self, df: pd.DataFrame) -> float:
        """
        Legacy method - returns net bias.
        """
        decisions = self.factory.process_symbol(df, "UNKNOWN", "unknown")
        
        if not decisions:
            return 0.0
        
        buy_confidence = sum(d['confidence'] for d in decisions if d['direction'] == 'BUY' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW'])
        sell_confidence = sum(d['confidence'] for d in decisions if d['direction'] == 'SELL' and d['action'] in ['STRONG_ENTRY', 'ENTER_NOW'])
        
        total = buy_confidence + sell_confidence
        if total == 0:
            return 0.0
        
        return (buy_confidence - sell_confidence) / total


# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def test_pattern_factory_v4(symbol: str = "BTC/USDT", 
                            timeframe: str = "1h",
                            lookback: int = 200):
    """
    Test Pattern Factory V4 with sample data.
    """
    print("\n" + "="*80)
    print(f"🔬 PATTERN V4 FACTORY TEST")
    print("="*80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Lookback: {lookback} candles")
    print("="*80)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=lookback, freq='1h')
    np.random.seed(42)
    
    # Create trending data with patterns
    trend = np.linspace(0, 0.1, lookback)
    noise = np.random.randn(lookback) * 0.005
    close = 100 * (1 + trend + noise)
    
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(lookback) * 0.002),
        'high': close * (1 + abs(np.random.randn(lookback) * 0.005)),
        'low': close * (1 - abs(np.random.randn(lookback) * 0.005)),
        'close': close,
        'volume': np.random.randint(1000, 10000, lookback)
    }, index=dates)
    
    # Ensure high is highest, low is lowest
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(lookback) * 0.5)
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(lookback) * 0.5)
    
    # Add indicators
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['rsi'] = 50 + np.random.randn(lookback) * 10
    
    print(f"\n✅ Sample data created: {len(df)} candles")
    print(f"   Current price: {df['close'].iloc[-1]:.2f}")
    
    # Initialize factory
    factory = PatternFactoryV4(debug_mode=True)
    
    # Process
    print("\n🔄 Processing patterns...")
    decisions = factory.process_symbol(df, symbol, timeframe)
    
    # Print results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    if decisions:
        print(f"✅ Generated {len(decisions)} trading decisions:\n")
        for i, d in enumerate(decisions, 1):
            print(f"{i}. {d['pattern_name']} ({d['direction']})")
            print(f"   Action: {d['action']}")
            print(f"   Confidence: {d['confidence']:.1%}")
            print(f"   Grade: {d['grade']}")
            print(f"   Entry: {d['entry']:.4f}")
            print(f"   Stop: {d['stop_loss']:.4f}")
            print(f"   Target: {d['take_profit']:.4f}")
            print(f"   RR: {d['risk_reward']:.2f}")
            print(f"   Reason: {d['decision_reason'][:100]}...")
            print()
    else:
        print("❌ No trading decisions generated")
    
    # Print performance stats
    stats = factory.get_performance_stats()
    print("\n📈 PERFORMANCE STATS:")
    print(f"   Patterns processed: {stats['total_patterns_processed']}")
    print(f"   Patterns accepted: {stats['total_patterns_accepted']}")
    print(f"   Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"   Avg processing time: {stats['avg_processing_time_ms']:.1f}ms")
    
    factory.print_debug_summary()
    
    return decisions


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'FailFastSystemV4',
    'PatternFactoryV4',
    'PatternFactoryLegacyV4',
    'test_pattern_factory_v4',
]