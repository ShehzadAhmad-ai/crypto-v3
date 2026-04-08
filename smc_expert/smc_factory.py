"""
SMC Expert V3 - Main Orchestrator (COMPLETE REWRITE)
Factory that ties all components together and generates trading signals
ADDED: Multi-HTF analysis, top-down bias, HTF alignment scoring
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings
import time

from .smc_core import (
    Direction, SMCContext, SMCData, POI, Candle, Swing,
    calculate_atr, normalize, df_to_candle_list, safe_get_swing_price
)
from .smc_config import CONFIG, get_htf_weight
from .smc_structure import MarketStructureAnalyzer
from .smc_blocks import OrderBlockDetector, MitigationTracker
from .smc_gaps import FVGManager
from .smc_liquidity import LiquidityManager
from .smc_zones import ZoneManager
from .smc_intelligence import ContextAnalyzer, SessionDetector
from .smc_execution import ExecutionManager
from .smc_scoring import MetaScoringEngine, ConfidenceCalibrator, GradeAssigner, NarrativeBuilder
from .smc_patterns import ICTPatternManager
from .smc_confirmation import EntryConfirmation
from .smc_debug import get_debugger, debug_log


class SMCFactory:
    """
    Main orchestrator for SMC Expert V3
    Ties together all components and generates trading signals
    ADDED: Multi-HTF analysis for top-down confirmation
    """
    
    def __init__(self):
        # Initialize debugger
        self.debug = get_debugger()
        self.debug.log("SMCFactory initialized with Multi-HTF support", "INFO")
        
        # Core analyzers
        self.structure_analyzer = MarketStructureAnalyzer()
        self.order_block_detector = OrderBlockDetector()
        self.fvg_manager = FVGManager()
        self.liquidity_manager = LiquidityManager()
        self.zone_manager = ZoneManager()
        self.context_analyzer = ContextAnalyzer()
        self.session_detector = SessionDetector()
        self.ict_patterns = ICTPatternManager()
        self.entry_confirmation = EntryConfirmation()
        
        # Execution engines
        self.execution_manager = ExecutionManager()
        
        # Scoring engines
        self.scoring_engine = MetaScoringEngine()
        self.calibrator = ConfidenceCalibrator()
        self.grade_assigner = GradeAssigner()
        self.narrative_builder = NarrativeBuilder()
        
        # HTF Analysis storage
        self.htf_analysis_cache: Dict[str, Dict] = {}
        
        # Replay history
        self.replay_history: List[Dict] = []
    
    @debug_log("SMCFactory")
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN",
                 timeframe: str = "1h", htf_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[Dict]:
        """
        Main analysis pipeline with Multi-HTF support
        
        Args:
            df: Current timeframe OHLCV data
            symbol: Trading pair symbol
            timeframe: Current timeframe (e.g., "5m", "15m", "1h")
            htf_data: Dictionary of higher timeframe DataFrames
                      Example: {'15m': df_15m, '1h': df_1h, '4h': df_4h, '1d': df_1d}
        """
        debug = get_debugger(symbol)
        debug.start_analysis(symbol, timeframe, df.shape)
        
        try:
            # ===== STEP 1: VALIDATE INPUT =====
            debug.log_step("Input Validation")
            if len(df) < CONFIG.MIN_BARS_FOR_ANALYSIS:
                debug.log_rejection(f"Insufficient bars: {len(df)} < {CONFIG.MIN_BARS_FOR_ANALYSIS}", "input_validation")
                debug.end_analysis(signal_generated=False)
                return None
            debug.log_step_complete("Input Validation", f"passed ({len(df)} bars)")
            
            # ===== STEP 2: PREPARE DATAFRAME =====
            debug.log_step("DataFrame Preparation")
            df = self._prepare_dataframe(df)
            debug.log_step_complete("DataFrame Preparation", f"{df.shape[0]} rows")
            
            # ===== STEP 2.5: MULTI-HTF ANALYSIS (NEW) =====
            debug.log_step("Multi-HTF Analysis")
            htf_analysis = self._analyze_multiple_htf(htf_data, symbol)
            debug.log_step_complete("Multi-HTF Analysis", f"Analyzed {len(htf_analysis)} timeframes")
            
            # ===== STEP 3: MARKET STRUCTURE ANALYSIS =====
            debug.log_step("Market Structure Analysis")
            start_time = time.time()
            structure_data = self.structure_analyzer.analyze(df)
            duration = (time.time() - start_time) * 1000
            
            swings = structure_data.get('swings', [])
            bos_points = structure_data.get('bos_points', [])
            choch_points = structure_data.get('choch_points', [])
            current_structure = structure_data.get('current_structure', 'NEUTRAL')
            
            debug.log_component_result("MarketStructureAnalyzer", len(swings) > 0, 
                                       f"{len(swings)} swings", duration)
            
            if len(swings) < 2:
                debug.log_rejection(f"Only {len(swings)} swings detected", "market_structure")
                debug.end_analysis(signal_generated=False)
                return None
            
            debug.log_step_complete("Market Structure Analysis", f"Structure: {current_structure}")
            
            # ===== STEP 4: LIQUIDITY ANALYSIS =====
            debug.log_step("Liquidity Analysis")
            start_time = time.time()
            liquidity_data = self.liquidity_manager.analyze(df, swings)
            duration = (time.time() - start_time) * 1000
            
            sweeps = liquidity_data.get('sweeps', [])
            liquidity_map = liquidity_data.get('liquidity_map', {})
            
            debug.log_component_result("LiquidityManager", len(sweeps) > 0, 
                                       f"{len(sweeps)} sweeps", duration)
            debug.log_step_complete("Liquidity Analysis", f"{len(sweeps)} sweeps detected")
            
            # ===== STEP 5: ORDER BLOCKS =====
            debug.log_step("Order Block Detection")
            start_time = time.time()
            ob_data = self.order_block_detector.detect_all(df, swings)
            duration = (time.time() - start_time) * 1000
            
            order_blocks = ob_data.get('order_blocks', [])
            debug.log_component_result("OrderBlockDetector", len(order_blocks) > 0, 
                                       f"{len(order_blocks)} OBs", duration)
            debug.log_step_complete("Order Block Detection", f"{len(order_blocks)} blocks found")
            
            # ===== STEP 6: FVG ANALYSIS =====
            debug.log_step("FVG Analysis")
            start_time = time.time()
            fvg_data = self.fvg_manager.analyze(df)
            duration = (time.time() - start_time) * 1000
            
            fvgs = fvg_data.get('fvgs', [])
            debug.log_component_result("FVGManager", len(fvgs) > 0, 
                                       f"{len(fvgs)} FVGs", duration)
            debug.log_step_complete("FVG Analysis", f"{len(fvgs)} gaps found")
            
            # ===== STEP 7: ZONE ANALYSIS =====
            debug.log_step("Zone Analysis")
            range_high = df['high'].tail(50).max()
            range_low = df['low'].tail(50).min()
            zone_direction = Direction.BUY if current_structure == "BULLISH" else Direction.SELL
            
            start_time = time.time()
            zone_data = self.zone_manager.analyze(df, zone_direction, range_high, range_low)
            duration = (time.time() - start_time) * 1000
            
            debug.log_step_complete("Zone Analysis", "completed")
            
            # ===== STEP 8: BUILD POIs =====
            debug.log_step("Building Points of Interest")
            pois = self._build_pois(order_blocks, fvgs, sweeps, zone_data, df)
            
            # Add HTF-based POIs if available
            if htf_analysis:
                htf_pois = self._build_htf_pois(htf_analysis, df)
                pois.extend(htf_pois)
                debug.log(f"Added {len(htf_pois)} HTF-based POIs", "INFO")
            
            debug.log_step_complete("Building Points of Interest", f"{len(pois)} POIs created")
            
            # ===== STEP 9: ICT PATTERNS DETECTION =====
            debug.log_step("ICT Patterns Detection")
            start_time = time.time()
            ict_patterns = self.ict_patterns.detect_all(
                df, sweeps, order_blocks, fvgs, bos_points
            )
            duration = (time.time() - start_time) * 1000
            
            silver_bullets = ict_patterns.get('silver_bullet', [])
            turtle_soups = ict_patterns.get('turtle_soup', [])
            
            debug.log_component_result("ICTPatternManager", len(silver_bullets) > 0, 
                                       f"{len(silver_bullets)} Silver Bullets, {len(turtle_soups)} Turtle Soups", duration)
            
            # Prioritize Silver Bullet if detected
            if silver_bullets:
                for pattern in silver_bullets:
                    silver_poi = POI(
                        price=pattern.get('entry_price', df['close'].iloc[-1]),
                        type='SILVER_BULLET',
                        sub_type=pattern.get('type', 'UNKNOWN'),
                        direction=pattern.get('direction', Direction.NEUTRAL),
                        strength=pattern.get('strength', 0.7),
                        components=[{'type': 'SILVER_BULLET', 'data': pattern}],
                        distance_atr=0,
                        is_active=True
                    )
                    pois.insert(0, silver_poi)
                debug.log("Silver Bullet POI prioritized", "INFO")
            
            # Add Turtle Soup POIs
            for pattern in turtle_soups:
                turtle_poi = POI(
                    price=pattern.get('entry_price', df['close'].iloc[-1]),
                    type='TURTLE_SOUP',
                    sub_type=pattern.get('type', 'UNKNOWN'),
                    direction=pattern.get('direction', Direction.NEUTRAL),
                    strength=pattern.get('strength', 0.6),
                    components=[{'type': 'TURTLE_SOUP', 'data': pattern}],
                    distance_atr=0,
                    is_active=True
                )
                pois.append(turtle_poi)
            
            debug.log_step_complete("ICT Patterns Detection", f"{len(silver_bullets)} patterns found")
            
            # ===== STEP 10: HTF ALIGNMENT SCORE (ENHANCED) =====
            debug.log_step("HTF Alignment Scoring")
            htf_alignment_score, htf_bias = self._calculate_htf_alignment_score(htf_analysis, current_structure)
            debug.log_step_complete("HTF Alignment Scoring", f"Score: {htf_alignment_score:.2f}, Bias: {htf_bias}")
            
            # ===== STEP 11: BUILD CONTEXT =====
            debug.log_step("Context Building")
            context = self.context_analyzer.analyze(
                df, swings, bos_points, choch_points, sweeps,
                zone_data['premium_discount']['zone'], htf_bias
            )
            # Add HTF analysis to context
            context.htf_analysis = htf_analysis
            context.htf_alignment_score = htf_alignment_score
            debug.log_step_complete("Context Building", f"AMD Phase: {context.amd_phase.value}")
            
            # ===== STEP 12: PROCESS ENTRIES =====
            debug.log_step("Entry Processing")
            best_entry = None
            best_poi = None
            best_confirmation = None

            for idx, poi in enumerate(pois[:15]):  # Check top 15 POIs
                debug.log(f"Evaluating POI {idx+1}: {poi.type} at {poi.price:.5f}", "DEBUG")
                entry = self.execution_manager.process_entry(poi, df, context)
                if entry:
                    confirmation = self.entry_confirmation.confirm(entry, df, context)
                    
                    # FIXED: Less strict confirmation requirement
                    if confirmation.get('confirmed', False) or confirmation.get('action') in ['STRONG_ENTRY', 'ENTER_NOW', 'WAIT_FOR_RETEST']:
                        best_entry = entry
                        best_poi = poi
                        best_confirmation = confirmation
                        debug.log(f"Selected POI: {poi.type} with action: {confirmation.get('action', 'UNKNOWN')}", "INFO")
                        break

            if not best_entry:
                debug.log_rejection("No valid entry found after processing all POIs", "entry_processing")
                debug.end_analysis(signal_generated=False)
                return None
            
            debug.log_step_complete("Entry Processing", f"Entry type: {best_entry.get('type', 'unknown')}")
            
            # ===== STEP 13: CALCULATE RISK PARAMETERS =====
            debug.log_step("Risk Calculation")
            next_target = self.liquidity_manager.get_next_target(
                best_entry['direction'], context.current_price
            )
            
            risk_params = self.execution_manager.calculate_risk_parameters(
                best_entry, context, next_target
            )
            
            poi_strength = self._get_poi_strength(best_poi)
            min_rr = self.execution_manager.risk_manager.get_min_rr(
                0.7, context, poi_strength
            )
            self.execution_manager.risk_manager.min_rr = min_rr
            
            rr_valid, rr_reason = self.execution_manager.risk_manager.validate_rr(
                risk_params['rr'], min_rr
            )
            
            if not rr_valid:
                debug.log_rejection(f"RR validation failed: {rr_reason}", "risk_management")
                debug.end_analysis(signal_generated=False)
                return None
            
            debug.log_step_complete("Risk Calculation", f"RR: {risk_params['rr']:.2f}")
            
            # ===== STEP 14: SCORE THE SETUP =====
            debug.log_step("Scoring")
            confluence_score = self.scoring_engine.calculate_confluence_score(
                context, best_entry, pois
            )
            component_breakdown = self.scoring_engine.get_component_breakdown()
            debug.log_step_complete("Scoring", f"Raw score: {confluence_score:.2f}")
            
            # ===== STEP 15: REPLAY ANALYSIS =====
            debug.log_step("Replay Analysis")
            replay_analysis = self._analyze_similar_setups(best_entry, context)
            debug.log_step_complete("Replay Analysis", f"Samples: {replay_analysis.get('samples_found', 0)}")
            
            # ===== STEP 16: CALIBRATE CONFIDENCE =====
            debug.log_step("Confidence Calibration")
            raw_confidence = confluence_score
            
            # Apply HTF alignment boost/penalty
            if htf_alignment_score > 0.7:
                raw_confidence *= min(1.3, 1.0 + (htf_alignment_score - 0.7))
                debug.log(f"HTF alignment BOOST: {htf_alignment_score:.2f} -> {raw_confidence:.2f}", "INFO")
            elif htf_alignment_score < 0.3:
                raw_confidence *= max(0.7, 1.0 - (0.3 - htf_alignment_score))
                debug.log(f"HTF alignment PENALTY: {htf_alignment_score:.2f} -> {raw_confidence:.2f}", "INFO")
            
            calibrated_confidence = self.calibrator.calibrate(
                raw_confidence, context, replay_analysis
            )
            debug.log_step_complete("Confidence Calibration", f"Final: {calibrated_confidence:.2f}")
            
            # ===== STEP 17: ASSIGN GRADE =====
            debug.log_step("Grade Assignment")
            grade, position_multiplier = self.grade_assigner.assign(calibrated_confidence)
            debug.log_step_complete("Grade Assignment", f"Grade: {grade}")
            
            # ===== STEP 18: CHECK EXECUTION LATENCY =====
            debug.log_step("Latency Check")
            slippage = abs(context.current_price - best_entry['entry']) / best_entry['entry'] if best_entry['entry'] > 0 else 0
            max_slippage = CONFIG.MAX_SLIPPAGE_ATR * context.atr / best_entry['entry'] if best_entry['entry'] > 0 else 0.01
            
            if slippage > max_slippage:
                debug.log_rejection(f"Slippage too high: {slippage:.2%} > {max_slippage:.2%}", "execution")
                debug.end_analysis(signal_generated=False)
                return None
            
            debug.log_step_complete("Latency Check", f"Slippage: {slippage:.4%}")
            
            # ===== STEP 19: GENERATE SIGNAL =====
            debug.log_step("Signal Generation")
            decision_reason = self._build_decision_reason(context, best_entry, component_breakdown, htf_alignment_score)
            
            signal = self.execution_manager.generate_signal(
                best_entry, risk_params, calibrated_confidence, grade,
                context, decision_reason
            )
            
            if not signal or signal['action'] == 'SKIP':
                debug.log_rejection("Signal generation returned SKIP", "signal_generation")
                debug.end_analysis(signal_generated=False)
                return None
            
            debug.log_step_complete("Signal Generation", f"Action: {signal['action']}")
            
            # ===== STEP 20: BUILD NARRATIVE =====
            debug.log_step("Narrative Building")
            narrative = self.narrative_builder.build_story(
                context, best_entry, calibrated_confidence, grade
            )
            # Add HTF insight to narrative
            if htf_analysis:
                narrative['htf_insight'] = self._build_htf_insight(htf_analysis, htf_alignment_score)
            debug.log_step_complete("Narrative Building", "completed")
            
            # ===== STEP 21: RECORD FOR REPLAY =====
            self._record_setup(signal, context)
            
            # ===== STEP 22: FORMAT OUTPUT =====
            result = self._format_output(
                signal, context, best_entry, best_poi,
                liquidity_map, narrative, replay_analysis,
                symbol, timeframe, htf_analysis
            )
            
            debug.log_signal_generated(signal)
            debug.end_analysis(signal_generated=True)
            
            return result
            
        except Exception as e:
            debug.log_error(e, "SMCFactory.analyze")
            debug.end_analysis(signal_generated=False)
            warnings.warn(f"Analysis error: {str(e)}")
            return None
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe with required columns"""
        df = df.copy()
        
        # Add derived columns if not present
        body = abs(df['close'] - df['open'])
        high_low = df['high'] - df['low']
        
        if 'body_ratio' not in df.columns:
            df['body_ratio'] = body / high_low.replace(0, np.nan)
            df['body_ratio'] = df['body_ratio'].fillna(0.5)
        
        if 'is_bullish' not in df.columns:
            df['is_bullish'] = df['close'] > df['open']
        
        if 'is_bearish' not in df.columns:
            df['is_bearish'] = df['close'] < df['open']
        
        if 'upper_wick' not in df.columns:
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        if 'lower_wick' not in df.columns:
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        if 'body' not in df.columns:
            df['body'] = body
        
        if 'range_' not in df.columns:
            df['range_'] = high_low
        
        return df
    
    def _analyze_multiple_htf(self, htf_data: Optional[Dict[str, pd.DataFrame]], symbol: str) -> Dict[str, Dict]:
        """
        Analyze multiple higher timeframes
        Returns dict with analysis for each timeframe
        """
        if not htf_data:
            return {}
        
        htf_analysis = {}
        
        for tf, tf_df in htf_data.items():
            if tf_df is None or len(tf_df) < 30:
                continue
            
            try:
                # Prepare HTF dataframe
                tf_df = self._prepare_dataframe(tf_df)
                
                # Analyze HTF structure
                tf_structure = self.structure_analyzer.analyze(tf_df)
                
                # Detect HTF Order Blocks
                tf_swings = tf_structure.get('swings', [])
                tf_obs = self.order_block_detector.detect_all(tf_df, tf_swings)
                
                # Detect HTF FVGs
                tf_fvgs = self.fvg_manager.analyze(tf_df)
                
                # Determine HTF bias
                tf_bias = tf_structure.get('current_structure', 'NEUTRAL')
                tf_strength = tf_structure.get('structure_strength', 0.5)
                
                # Get nearest HTF OB and FVG
                current_price = tf_df['close'].iloc[-1]
                atr = calculate_atr(tf_df)
                
                htf_analysis[tf] = {
                    'bias': tf_bias,
                    'strength': tf_strength,
                    'order_blocks': tf_obs.get('order_blocks', [])[:5],
                    'fvgs': tf_fvgs.get('fvgs', [])[:5],
                    'current_price': current_price,
                    'atr': atr,
                    'weight': get_htf_weight(tf)
                }
                
            except Exception as e:
                self.debug.log(f"Error analyzing HTF {tf}: {e}", "WARNING")
                continue
        
        return htf_analysis
    
    def _calculate_htf_alignment_score(self, htf_analysis: Dict[str, Dict], ltf_bias: str) -> Tuple[float, str]:
        """
        Calculate HTF alignment score based on multiple timeframes
        Returns (score, aggregated_bias)
        """
        if not htf_analysis:
            return 0.5, "NEUTRAL"
        
        total_score = 0
        total_weight = 0
        bullish_count = 0
        bearish_count = 0
        
        for tf, analysis in htf_analysis.items():
            weight = analysis.get('weight', 0.5)
            bias = analysis.get('bias', 'NEUTRAL')
            strength = analysis.get('strength', 0.5)
            
            if bias == 'BULLISH':
                bullish_count += 1
                tf_score = strength * weight
            elif bias == 'BEARISH':
                bearish_count += 1
                tf_score = strength * weight
            else:
                tf_score = 0.5 * weight
            
            total_score += tf_score
            total_weight += weight
        
        # Determine aggregated HTF bias
        if bullish_count > bearish_count:
            aggregated_bias = "BULLISH"
        elif bearish_count > bullish_count:
            aggregated_bias = "BEARISH"
        else:
            aggregated_bias = "NEUTRAL"
        
        # Calculate alignment with LTF
        if aggregated_bias == ltf_bias:
            alignment_multiplier = 1.2
        elif aggregated_bias == "NEUTRAL":
            alignment_multiplier = 1.0
        else:
            alignment_multiplier = 0.7
        
        final_score = min(1.0, (total_score / total_weight) * alignment_multiplier if total_weight > 0 else 0.5)
        
        return final_score, aggregated_bias
    
    def _build_htf_pois(self, htf_analysis: Dict[str, Dict], df: pd.DataFrame) -> List[POI]:
        """Build POIs from higher timeframe analysis"""
        pois = []
        current_price = df['close'].iloc[-1]
        atr = calculate_atr(df)
        
        for tf, analysis in htf_analysis.items():
            # Add HTF Order Blocks as POIs
            for ob in analysis.get('order_blocks', []):
                distance = abs(ob.price - current_price) / atr if atr > 0 else 0
                if distance <= CONFIG.POI_DISTANCE_MAX_ATR:
                    # Boost strength for higher timeframes
                    boosted_strength = min(1.0, ob.strength * (1 + analysis.get('weight', 0.5) * 0.3))
                    
                    poi = POI(
                        price=ob.price,
                        type='OB',
                        sub_type=f"HTF_{tf}_{ob.type.value}",
                        direction=ob.direction,
                        strength=boosted_strength,
                        components=[{'type': 'HTF_OB', 'timeframe': tf, 'data': ob}],
                        distance_atr=distance,
                        is_active=True
                    )
                    pois.append(poi)
            
            # Add HTF FVGs as POIs
            for fvg in analysis.get('fvgs', []):
                distance = abs(fvg.mid - current_price) / atr if atr > 0 else 0
                if distance <= CONFIG.POI_DISTANCE_MAX_ATR:
                    boosted_strength = min(1.0, fvg.strength * (1 + analysis.get('weight', 0.5) * 0.2))
                    
                    poi = POI(
                        price=fvg.mid,
                        type='FVG',
                        sub_type=f"HTF_{tf}_{fvg.type.value}",
                        direction=fvg.direction,
                        strength=boosted_strength,
                        components=[{'type': 'HTF_FVG', 'timeframe': tf, 'data': fvg}],
                        distance_atr=distance,
                        is_active=True
                    )
                    pois.append(poi)
        
        # Sort by strength
        pois.sort(key=lambda x: x.strength, reverse=True)
        
        return pois[:10]  # Limit to top 10 HTF POIs
    
    def _build_htf_insight(self, htf_analysis: Dict[str, Dict], alignment_score: float) -> str:
        """Build human-readable HTF insight"""
        if not htf_analysis:
            return "No higher timeframe data available"
        
        insights = []
        for tf, analysis in htf_analysis.items():
            bias = analysis.get('bias', 'NEUTRAL')
            strength = analysis.get('strength', 0.5)
            insights.append(f"{tf}: {bias} ({strength:.0%})")
        
        alignment_text = "aligned" if alignment_score > 0.6 else "misaligned" if alignment_score < 0.4 else "neutral"
        
        return f"HTF bias: {', '.join(insights)} | Alignment: {alignment_text} ({alignment_score:.0%})"
    
    def _build_pois(self, order_blocks: List, fvgs: List, sweeps: List,
                     zone_data: Dict, df: pd.DataFrame) -> List[POI]:
        """Build Points of Interest from all components"""
        pois = []
        current_price = df['close'].iloc[-1]
        atr = calculate_atr(df)
        
        # Order blocks as POIs
        for ob in order_blocks:
            distance = abs(ob.price - current_price) / atr if atr > 0 else 0
            if distance <= CONFIG.POI_DISTANCE_MAX_ATR:
                poi = POI(
                    price=ob.price,
                    type='OB',
                    sub_type=ob.type.value,
                    direction=ob.direction,
                    strength=ob.strength,
                    components=[{'type': 'OB', 'data': ob}],
                    distance_atr=distance,
                    is_active=ob.mitigation_state.value == 'UNMITIGATED'
                )
                pois.append(poi)
        
        # FVGs as POIs
        for fvg in fvgs:
            distance = abs(fvg.mid - current_price) / atr if atr > 0 else 0
            if distance <= CONFIG.POI_DISTANCE_MAX_ATR:
                poi = POI(
                    price=fvg.mid,
                    type='FVG',
                    sub_type=fvg.type.value,
                    direction=fvg.direction,
                    strength=fvg.strength,
                    components=[{'type': 'FVG', 'data': fvg}],
                    distance_atr=distance,
                    is_active=fvg.mitigation_state.value == 'UNMITIGATED'
                )
                pois.append(poi)
        
        # Sweeps as POIs
        for sweep in sweeps:
            distance = abs(sweep.price - current_price) / atr if atr > 0 else 0
            if distance <= CONFIG.POI_DISTANCE_MAX_ATR and sweep.reversal_strength > 0.5:
                direction = Direction.BUY if sweep.type == 'SSL_SWEEP' else Direction.SELL
                poi = POI(
                    price=sweep.price,
                    type='SWEEP',
                    sub_type=sweep.type,
                    direction=direction,
                    strength=sweep.reversal_strength,
                    components=[{'type': 'SWEEP', 'data': sweep}],
                    distance_atr=distance,
                    is_active=True
                )
                pois.append(poi)
        
        # OTE as POI
        ote_levels = zone_data.get('ote_levels', {})
        for ote_key in ['ote_1', 'ote_2']:
            if ote_key in ote_levels:
                ote_price = ote_levels[ote_key]
                distance = abs(ote_price - current_price) / atr if atr > 0 else 0
                if distance <= CONFIG.POI_DISTANCE_MAX_ATR:
                    poi = POI(
                        price=ote_price,
                        type='OTE',
                        sub_type=ote_key,
                        direction=zone_data.get('direction', Direction.NEUTRAL),
                        strength=0.65,
                        components=[{'type': 'OTE', 'data': ote_levels}],
                        distance_atr=distance,
                        is_active=True
                    )
                    pois.append(poi)
        
        # Sort by distance and strength
        pois.sort(key=lambda x: (x.distance_atr, -x.strength))
        
        return pois
    
    def _get_poi_strength(self, poi: POI) -> str:
        """Get POI strength category"""
        if poi.strength >= 0.75:
            return "STRONG"
        elif poi.strength >= 0.55:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _build_decision_reason(self, context: SMCContext, entry: Dict,
                                component_breakdown: Dict, htf_score: float) -> str:
        """Build human-readable decision reason"""
        reasons = []
        
        reasons.append(f"{context.current_structure} structure")
        reasons.append(f"{context.amd_phase.value} phase")
        reasons.append(f"{entry.get('type', 'setup')} entry")
        
        if context.is_kill_zone:
            reasons.append("kill zone")
        
        reasons.append(f"{context.zone_type.value}")
        
        if htf_score > 0.7:
            reasons.append("HTF aligned")
        elif htf_score < 0.3:
            reasons.append("HTF caution")
        
        return " + ".join(reasons[:6])
    
    def _analyze_similar_setups(self, entry: Dict, context: SMCContext) -> Dict:
        """Analyze similar setups from replay history"""
        if len(self.replay_history) < CONFIG.REPLAY_MIN_SAMPLES:
            return {'insufficient_data': True, 'samples_found': len(self.replay_history)}
        
        similar = [s for s in self.replay_history if s.get('entry_type') == entry.get('type')]
        
        if len(similar) < CONFIG.REPLAY_MIN_SAMPLES:
            return {'insufficient_data': True, 'samples_found': len(similar)}
        
        wins = [s for s in similar if s.get('outcome') == 'win']
        win_rate = len(wins) / len(similar) if similar else 0
        avg_rr = np.mean([s.get('rr_achieved', 0) for s in similar]) if similar else 0
        
        return {
            'insufficient_data': False,
            'samples_found': len(similar),
            'win_rate': round(win_rate, 2),
            'avg_rr_achieved': round(avg_rr, 2),
            'setup_similarity': 0.75
        }
    
    def _record_setup(self, signal: Dict, context: SMCContext):
        """Record setup for replay analysis"""
        if len(self.replay_history) >= CONFIG.REPLAY_LOOKBACK:
            self.replay_history.pop(0)
        
        self.replay_history.append({
            'direction': signal['direction'],
            'entry_type': signal.get('entry_details', {}).get('type'),
            'confidence': signal['confidence'],
            'rr': signal['risk_reward'],
            'session': context.session.value,
            'amd_phase': context.amd_phase.value,
            'timestamp': datetime.now()
        })
    
    def _format_output(self, signal: Dict, context: SMCContext, entry: Dict,
                        poi: POI, liquidity_map: Dict, narrative: Dict,
                        replay_analysis: Dict, symbol: str, timeframe: str,
                        htf_analysis: Dict) -> Dict:
        """Format final output"""
        next_target = self.liquidity_manager.get_next_target(
            Direction.BUY if signal['direction'] == 'BUY' else Direction.SELL,
            context.current_price
        )
        
        return {
            "module": "smc",
            "signal_id": self._generate_signal_id(symbol, signal['direction']),
            "pattern_name": f"SMC_{entry.get('type', 'UNKNOWN')}_{signal['direction']}",
            "direction": signal['direction'],
            "entry": signal['entry'],
            "stop_loss": signal['stop_loss'],
            "take_profit": signal['take_profit'],
            "risk_reward": signal['risk_reward'],
            "confidence": signal['confidence'],
            "grade": signal['grade'],
            "position_multiplier": CONFIG.POSITION_MULTIPLIERS.get(signal['grade'], 1.0),
            "action": signal['action'],
            "decision_reason": signal['decision_reason'],
            "liquidity_map": {
                "above": liquidity_map.get('above', [])[:3],
                "below": liquidity_map.get('below', [])[:3],
                "next_target": next_target
            },
            "narrative": narrative,
            "replay_analysis": replay_analysis,
            "context_summary": self.context_analyzer.get_context_summary(),
            "htf_analysis": {
                "available_timeframes": list(htf_analysis.keys()),
                "alignment_score": context.htf_alignment_score,
                "details": {tf: {'bias': a.get('bias'), 'strength': a.get('strength')} 
                           for tf, a in htf_analysis.items()}
            },
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe
        }
    
    def _generate_signal_id(self, symbol: str, direction: str) -> str:
        """Generate unique signal ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"smc_{symbol.replace('/', '_')}_{direction}_{timestamp}"
    
    def clear_cache(self):
        """Clear HTF cache and replay history"""
        self.htf_analysis_cache.clear()
        self.replay_history.clear()
        self.debug.log("Cache cleared", "INFO")