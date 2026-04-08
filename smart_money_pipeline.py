# smart_money_pipeline.py - ENHANCED WITH ALL SMART MONEY COMPONENTS
"""
Smart Money Confirmation Pipeline - Enhanced Version
Runs all smart money components:
- Liquidity Intelligence (sweeps, inducements, sweep failures)
- Order Flow Intelligence (CVD, absorption, exhaustion)
- Market Structure (order blocks, mitigation, breaker blocks)
- Microstructure (FVGs, displacement, liquidity zones)
- Liquidation Intelligence (real liquidation data)
- Market Maker Intelligence (positioning, inventory)
- Regime Alignment (with Technical Phase)

Uses order book if available, falls back to OHLCV only
"""
from typing import Dict, Optional, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

from config import Config
from logger import log
from signal_model import Signal, SignalStatus

# Import all smart money modules
from liquidity import LiquidityIntelligence
from orderflow import AdvancedOrderFlow
from microstructure import MicrostructureEngine
from market_maker_model import MarketMakerModel, MarketMakerActivity, SmartMoneyFootprint
from liquidation_intelligence import LiquidationIntelligence
from smart_money_scoring import SmartMoneyScoring, SmartMoneyResult
from smart_money_story import SmartMoneyStoryBuilder

# Phase5 data fetcher for real liquidation data
from phase5_data_fetcher import Phase5DataFetcher


class SmartMoneyPipeline:
    """
    Smart Money Confirmation Pipeline - Enhanced
    - Runs all enabled smart money modules
    - Uses order book when available (falls back to OHLCV)
    - Calculates smart_money_score using weighted components
    - Generates human-readable story
    - Filters signals below SMART_MONEY_MIN_SCORE
    """
    
    def __init__(self):
        # Initialize all modules
        self.liquidity = LiquidityIntelligence()
        self.orderflow = AdvancedOrderFlow()
        self.microstructure = MicrostructureEngine()
        self.market_maker = MarketMakerModel()
        self.liquidation = LiquidationIntelligence()
        self.scoring = SmartMoneyScoring()
        self.story_builder = SmartMoneyStoryBuilder()
        
        # Phase5 data fetcher for real liquidation data
        self.phase5_fetcher = Phase5DataFetcher()
        self.liquidation.set_fetcher(self.phase5_fetcher)
        
        # Load settings from config
        self.min_score = getattr(Config, 'SMART_MONEY_MIN_SCORE', 0.60)
        
        log.info("=" * 60)
        log.info("Smart Money Pipeline initialized (Enhanced)")
        log.info("=" * 60)
        log.info(f"  Liquidity Analysis: {Config.ENABLE_LIQUIDITY_ANALYSIS}")
        log.info(f"  Order Flow Analysis: {Config.ENABLE_ORDERFLOW_ANALYSIS}")
        log.info(f"  Microstructure Analysis: {Config.ENABLE_MICROSTRUCTURE_ANALYSIS}")
        log.info(f"  Market Maker Analysis: {Config.ENABLE_MARKET_MAKER_ANALYSIS}")
        log.info(f"  Liquidation Analysis: {getattr(Config, 'SM_ENABLE_LIQUIDATION_ANALYSIS', True)}")
        log.info(f"  Regime Alignment: {getattr(Config, 'SM_ENABLE_REGIME_ALIGNMENT', True)}")
        log.info(f"  Min Score Threshold: {self.min_score}")
        log.info("=" * 60)
    
    def confirm(self, signal: Signal, df: pd.DataFrame, 
                order_book: Optional[Dict] = None,
                regime_data: Optional[Dict] = None) -> Optional[Signal]:
        """
        Run smart money confirmation on a signal that passed MTF
        
        Args:
            signal: Signal that passed Phase 3 (MTF)
            df: OHLCV DataFrame
            order_book: Optional order book data (if available)
            regime_data: Optional regime data from Technical Phase
        
        Returns:
            Updated signal or None if filtered out
        """
        # Only run on signals that passed MTF
        if signal.status != SignalStatus.MTF_PASSED:
            return None
        
        try:
            # Store component results
            component_results = {}
            
            # ===== 1. LIQUIDITY ANALYSIS =====
            if Config.ENABLE_LIQUIDITY_ANALYSIS:
                liquidity_result = self.liquidity.get_liquidity_summary(df)
                component_results['liquidity'] = liquidity_result
                log.debug(f"[SM] Liquidity score: {liquidity_result.get('score', 0.5):.2f} | Bias: {liquidity_result.get('direction', 'NEUTRAL')}")
            
            # ===== 2. ORDER FLOW ANALYSIS =====
            if Config.ENABLE_ORDERFLOW_ANALYSIS:
                orderflow_result = self.orderflow.get_orderflow_summary(df)
                component_results['orderflow'] = orderflow_result
                log.debug(f"[SM] Order Flow score: {orderflow_result.get('score', 0.5):.2f} | Bias: {orderflow_result.get('direction', 'NEUTRAL')}")
            
            # ===== 3. MARKET STRUCTURE ANALYSIS =====
            # Using market_structure.py functions (already imported via microstructure)
            from market_structure import get_structure_summary, get_structure_bias, get_structure_score
            
            market_structure_result = get_structure_summary(df)
            component_results['market_structure'] = market_structure_result
            log.debug(f"[SM] Market Structure score: {market_structure_result.get('structure_score', 0.5):.2f} | Trend: {market_structure_result.get('trend', 'NEUTRAL')}")
            
            # ===== 4. MICROSTRUCTURE ANALYSIS =====
            if Config.ENABLE_MICROSTRUCTURE_ANALYSIS:
                microstructure_result = self.microstructure.get_microstructure_summary(df)
                component_results['microstructure'] = microstructure_result
                log.debug(f"[SM] Microstructure score: {microstructure_result.get('score', 0.5):.2f} | Bias: {microstructure_result.get('direction', 'NEUTRAL')}")
            
            # ===== 5. LIQUIDATION ANALYSIS =====
            if getattr(Config, 'SM_ENABLE_LIQUIDATION_ANALYSIS', True):
                liquidation_result = self.liquidation.analyze(signal.symbol, df)
                component_results['liquidations'] = liquidation_result
                log.debug(f"[SM] Liquidations score: {liquidation_result.get('score', 0.5):.2f} | Bias: {liquidation_result.get('bias', 'NEUTRAL')}")
            
            # ===== 6. MARKET MAKER ANALYSIS =====
            if Config.ENABLE_MARKET_MAKER_ANALYSIS:
                market_maker_result = self._analyze_market_maker(df, order_book)
                component_results['market_maker'] = market_maker_result
                log.debug(f"[SM] Market Maker score: {market_maker_result.get('score', 0.5):.2f} | Activity: {market_maker_result.get('activity', 'NEUTRAL')}")
            
            # ===== 7. REGIME ALIGNMENT =====
            if getattr(Config, 'SM_ENABLE_REGIME_ALIGNMENT', True) and regime_data:
                regime_alignment_result = self._calculate_regime_alignment(
                    regime_data, component_results
                )
                component_results['regime_alignment'] = regime_alignment_result
                log.debug(f"[SM] Regime Alignment score: {regime_alignment_result.get('score', 0.5):.2f}")
            
            # ===== 8. AGGREGATE SCORES =====
            scoring_result = self.scoring.aggregate_scores(component_results)
            
            # ===== 9. APPLY SMART MONEY FILTER =====
            if scoring_result.total_score < self.min_score:
                log.debug(f"{signal.symbol}: Smart money score {scoring_result.total_score:.2f} < {self.min_score}")
                return None
            
            # ===== 10. BUILD STORY =====
            story = self.story_builder.build_story(component_results, scoring_result)
            
            # ===== 11. UPDATE SIGNAL =====
            signal.smart_money_score = scoring_result.total_score
            signal.orderflow_bias = scoring_result.dominant_direction
            signal.status = SignalStatus.SMART_MONEY_PASSED
            
            # Store all results in metadata
            if not hasattr(signal, 'metadata'):
                signal.metadata = {}
            
            # Store smart money results
            signal.metadata['smart_money'] = {
                'score': scoring_result.total_score,
                'bias': scoring_result.dominant_direction,
                'net_bias': scoring_result.net_bias,
                'confidence': scoring_result.confidence,
                'agreement': scoring_result.agreement_score,
                'components': self.scoring.get_component_summary(scoring_result.component_scores),
                'reasons': scoring_result.reasons,
                'confidence_boost': scoring_result.confidence_boost
            }
            
            # Store story
            signal.metadata['smart_money_story'] = story.full_story
            signal.metadata['smart_money_summary'] = story.summary
            signal.metadata['smart_money_key_points'] = story.key_points
            signal.metadata['smart_money_warnings'] = story.risk_warnings
            
            # Store individual component results for strategies
            signal.metadata['liquidity_data'] = component_results.get('liquidity', {})
            signal.metadata['orderflow_data'] = component_results.get('orderflow', {})
            signal.metadata['structure_data'] = component_results.get('market_structure', {})
            signal.metadata['microstructure_data'] = component_results.get('microstructure', {})
            signal.metadata['liquidations_data'] = component_results.get('liquidations', {})
            signal.metadata['market_maker_data'] = component_results.get('market_maker', {})
            
            # Store liquidity flags for quick access (for strategies)
            liquidity_data = component_results.get('liquidity', {})
            signal.metadata['has_down_sweep'] = liquidity_data.get('has_down_sweep', False)
            signal.metadata['has_up_sweep'] = liquidity_data.get('has_up_sweep', False)
            signal.metadata['stop_hunt_prob'] = liquidity_data.get('stop_hunt_probability', 0)
            signal.metadata['inducements'] = liquidity_data.get('inducement_count', 0)
            signal.metadata['sweeps'] = liquidity_data.get('sweeps', [])
            
            # Store liquidation data
            liquidation_data = component_results.get('liquidations', {})
            signal.metadata['liquidations'] = {
                'short_liquidations': liquidation_data.get('short_liq_value', 0),
                'long_liquidations': liquidation_data.get('long_liq_value', 0),
                'total_liq_value': liquidation_data.get('total_liq_value', 0),
                'cascade_risk': liquidation_data.get('cascade_risk', 0),
                'dominant': liquidation_data.get('dominant', 'NONE')
            }
            
            # Store microstructure data
            microstructure_data = component_results.get('microstructure', {})
            signal.metadata['fair_value_gaps'] = microstructure_data.get('fvg', {}).get('unfilled_bullish', 0) + microstructure_data.get('fvg', {}).get('unfilled_bearish', 0)
            signal.metadata['bos_detected'] = microstructure_data.get('displacement', {}).get('count', 0) > 0
            signal.metadata['order_blocks'] = microstructure_data.get('order_blocks', {}).get('count', 0)
            
            # Terminal output
            self._print_smart_money_output(signal, scoring_result, story)
            
            return signal
            
        except Exception as e:
            log.error(f"Error in smart money pipeline for {signal.symbol}: {e}", exc_info=True)
            return None
    
    def _analyze_market_maker(self, df: pd.DataFrame, 
                               order_book: Optional[Dict]) -> Dict[str, Any]:
        """Analyze market maker activity - uses order book if available"""
        try:
            details = {}
            
            # Try to use order book if available
            if order_book and Config.ENABLE_MARKET_MAKER_ANALYSIS:
                mm_position = self.market_maker.analyze_market_maker_activity(df, order_book)
                
                details = {
                    'activity': mm_position.activity_type.value if mm_position.activity_type else 'NEUTRAL',
                    'net_position': mm_position.net_position,
                    'inventory_imbalance': mm_position.inventory_imbalance,
                    'activity_level': mm_position.activity_level,
                    'confidence': mm_position.confidence,
                    'score': 0.5,
                    'bias': 'NEUTRAL',
                    'data_source': 'order_book'
                }
                
                # Score based on market maker positioning
                score = 0.5
                bias = 0
                
                if mm_position.activity_type in [MarketMakerActivity.AGGRESSIVE_BUYING, 
                                                MarketMakerActivity.PASSIVE_BUYING]:
                    score += 0.2 * mm_position.confidence
                    bias += 0.3 * mm_position.confidence
                elif mm_position.activity_type in [MarketMakerActivity.AGGRESSIVE_SELLING,
                                                  MarketMakerActivity.PASSIVE_SELLING]:
                    score += 0.2 * mm_position.confidence
                    bias -= 0.3 * mm_position.confidence
                
                # Add inventory imbalance impact
                score += abs(mm_position.inventory_imbalance) * 0.2
                bias += mm_position.inventory_imbalance * 0.3
                
                details['score'] = min(0.99, score)
                details['bias'] = 'BULLISH' if bias > 0.1 else ('BEARISH' if bias < -0.1 else 'NEUTRAL')
                
            else:
                # Fallback to OHLCV-only analysis
                log.debug("Order book unavailable, using OHLCV proxy for market maker analysis")
                proxy = self.market_maker.proxy_options_flow(df)
                
                score = 0.5
                bias = 0
                
                # Unusual activity = smart money
                if proxy.unusual_activity_score > 0.6:
                    score += 0.15
                    bias += 0.2 if proxy.dealer_hedge_flow > 0 else -0.2
                
                # Dealer flow direction
                if abs(proxy.dealer_hedge_flow) > 0.3:
                    score += 0.1
                    bias += proxy.dealer_hedge_flow * 0.3
                
                # Put/call ratio extremes
                if proxy.put_call_ratio < 0.7:
                    score += 0.1
                    bias += 0.15
                elif proxy.put_call_ratio > 1.3:
                    score += 0.1
                    bias -= 0.15
                
                details = {
                    'score': min(0.99, score),
                    'bias': 'BULLISH' if bias > 0.1 else ('BEARISH' if bias < -0.1 else 'NEUTRAL'),
                    'put_call_ratio': proxy.put_call_ratio,
                    'implied_volatility': proxy.implied_volatility,
                    'dealer_flow': proxy.dealer_hedge_flow,
                    'unusual_activity': proxy.unusual_activity_score,
                    'data_source': 'ohlcv_proxy'
                }
            
            return details
            
        except Exception as e:
            log.debug(f"Market maker analysis error: {e}")
            return {'score': 0.5, 'bias': 'NEUTRAL', 'error': str(e)}
    
    def _calculate_regime_alignment(self, regime_data: Dict, 
                                     component_results: Dict) -> Dict[str, Any]:
        """Calculate alignment between Smart Money and market regime"""
        try:
            # Get regime bias
            regime_bias = regime_data.get('bias_score', 0)
            regime_name = regime_data.get('regime', 'UNKNOWN')
            
            # Calculate composite Smart Money bias from components
            sm_bias = 0
            sm_weights = 0
            
            # Liquidity bias
            liquidity = component_results.get('liquidity', {})
            if liquidity.get('net_bias'):
                sm_bias += liquidity.get('net_bias', 0) * 0.25
                sm_weights += 0.25
            
            # Order flow bias
            orderflow = component_results.get('orderflow', {})
            if orderflow.get('net_bias'):
                sm_bias += orderflow.get('net_bias', 0) * 0.25
                sm_weights += 0.25
            
            # Market structure bias
            market_structure = component_results.get('market_structure', {})
            structure_bias = market_structure.get('structure_bias', 0)
            if structure_bias != 0:
                sm_bias += structure_bias * 0.25
                sm_weights += 0.25
            
            # Microstructure bias
            microstructure = component_results.get('microstructure', {})
            if microstructure.get('net_bias'):
                sm_bias += microstructure.get('net_bias', 0) * 0.25
                sm_weights += 0.25
            
            if sm_weights > 0:
                sm_bias = sm_bias / sm_weights
            
            # Calculate alignment
            if sm_bias * regime_bias > 0:
                # Same direction
                alignment = min(1.0, abs(sm_bias * regime_bias) * 2)
                score = 0.5 + (alignment * 0.4)
            else:
                # Opposite direction
                alignment = 0
                score = 0.3
            
            return {
                'score': min(0.99, score),
                'bias': sm_bias,
                'regime_bias': regime_bias,
                'regime': regime_name,
                'alignment': alignment,
                'confidence': 0.7,
                'reasons': [
                    f"Smart Money bias: {sm_bias:+.2f}",
                    f"Regime bias: {regime_bias:+.2f}",
                    f"Alignment: {alignment:.0%}"
                ]
            }
            
        except Exception as e:
            log.debug(f"Regime alignment error: {e}")
            return {'score': 0.5, 'bias': 0, 'alignment': 0, 'reasons': ['Error in alignment']}
    
    def _print_smart_money_output(self, signal: Signal, 
                                   scoring_result: SmartMoneyResult,
                                   story) -> None:
        """Print formatted smart money output"""
        log.info("\n" + "=" * 70)
        log.info(f"[SMART MONEY] {signal.symbol} {signal.timeframe}")
        log.info("=" * 70)
        log.info(f"Score: {scoring_result.total_score:.2f}")
        log.info(f"Net Bias: {scoring_result.net_bias:+.2f} ({scoring_result.dominant_direction})")
        log.info(f"Confidence: {scoring_result.confidence:.1%}")
        log.info(f"Agreement: {scoring_result.agreement_score:.0%}")
        log.info(f"Status: {'PASSED' if scoring_result.total_score >= self.min_score else 'FAILED'}")
        log.info("-" * 70)
        
        # Component scores
        log.info("Component Scores:")
        for cs in scoring_result.component_scores:
            bias_icon = "+" if cs.bias > 0 else ("-" if cs.bias < 0 else " ")
            log.info(f"  {cs.name:18s}: {cs.score:.2f} (bias: {bias_icon}{cs.bias:+.2f})")
        
        log.info("-" * 70)
        log.info("Key Points:")
        for point in story.key_points[:6]:
            log.info(f"  - {point}")
        
        if story.risk_warnings:
            log.info("-" * 70)
            log.info("Warnings:")
            for warning in story.risk_warnings[:3]:
                log.info(f"  - {warning}")
        
        log.info("=" * 70)