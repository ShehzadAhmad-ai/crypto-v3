# risk_pipeline.py - Complete Risk Management Pipeline (Phase 8)
"""
Complete Risk Management Pipeline - Phase 8

This pipeline:
1. Takes CombinedSignal from Phase 7 (Light Confirm)
2. Calculates ENTRY ZONES using multi-method analysis (SMC + Indicators + Structure + Volume)
3. Calculates RISK-MANAGED STOP LOSS (ADDED - preserves original)
4. Calculates RISK-MANAGED TAKE PROFIT (ADDED - preserves original)
5. Calculates POSITION SIZE based on confidence, win rate, regime, volatility, grade
6. Creates PROTECTION PLAN (trailing stops, partial TPs, breakeven)
7. Validates all risk parameters
8. Outputs ENHANCED CombinedSignal with all risk-managed fields

All settings controlled from config.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime

from config import Config
from logger import log
from expert_interface import CombinedSignal, TPLevel, ExpertSignal
from signal_model import Signal, SignalStatus, EntryType

# Import risk components
from risk_analyzer import RiskAnalyzer, StopLossResult, TakeProfitResult, EntryZoneResult
from risk_manager import (
    RiskProfile, create_risk_profile, PositionSizingCalculator,
    TradeProtectionEngine, PortfolioHeatManager, CorrelationRiskFilter,
    validate_risk_reward, SignalQuality, calculate_partial_take_profits,
    calculate_breakeven_stop, TradeProtectionPlan, PositionSizeResult, PortfolioHeat
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class RiskPipelineConfig:
    """Configuration for Risk Pipeline - loaded from Config"""
    
    def __init__(self):
        # Risk/Reward thresholds
        self.min_risk_reward = getattr(Config, 'MIN_RISK_REWARD', 1.5)
        self.target_risk_reward = getattr(Config, 'TARGET_RISK_REWARD', 2.5)
        
        # Position sizing
        self.max_risk_per_trade = getattr(Config, 'MAX_RISK_PER_TRADE', 0.02)
        self.portfolio_value = getattr(Config, 'PORTFOLIO_VALUE', 10000)
        self.max_position_concentration = getattr(Config, 'MAX_POSITION_CONCENTRATION', 0.25)
        
        # Kelly Criterion
        self.enable_kelly = getattr(Config, 'ENABLE_KELLY', True)
        self.kelly_fraction = getattr(Config, 'KELLY_FRACTION', 0.25)
        
        # Entry zone
        self.entry_zone_atr_multiplier = getattr(Config, 'ENTRY_ZONE_ATR_MULTIPLIER', 0.5)
        
        # TP refinement
        self.max_tp_distance_pct = getattr(Config, 'MAX_TP_DISTANCE_PCT', 0.15)
        self.tp_cluster_tolerance = getattr(Config, 'TP_CLUSTER_TOLERANCE', 0.005)
        
        # Stop loss refinement
        self.enable_stop_refinement = getattr(Config, 'ENABLE_STOP_REFINEMENT', True)
        self.stop_buffer_pct = getattr(Config, 'STOP_BUFFER_PCT', 0.002)
        
        # Portfolio heat
        self.max_portfolio_heat = getattr(Config, 'MAX_PORTFOLIO_HEAT', 80)
        
        # Correlation risk
        self.correlation_threshold = getattr(Config, 'CORRELATION_THRESHOLD', 0.7)
        
        # Protection plans
        self.enable_trailing_stop = getattr(Config, 'ENABLE_TRAILING_STOP', True)
        self.enable_partial_tp = getattr(Config, 'ENABLE_PARTIAL_TP', True)
        self.enable_breakeven = getattr(Config, 'ENABLE_BREAKEVEN', True)
        self.enable_time_exit = getattr(Config, 'ENABLE_TIME_EXIT', True)
        
        # Time exit
        self.max_hold_bars = getattr(Config, 'MAX_HOLD_BARS', 48)
        
        # Historical stats for Kelly
        self.historical_win_rate = getattr(Config, 'HISTORICAL_WIN_RATE', 0.55)
        self.historical_avg_win = getattr(Config, 'HISTORICAL_AVG_WIN', 2.0)
        self.historical_avg_loss = getattr(Config, 'HISTORICAL_AVG_LOSS', 1.0)


# ============================================================================
# RISK PIPELINE
# ============================================================================

class RiskPipeline:
    """
    Complete Risk Management Pipeline - Phase 8
    
    ADDS risk-managed fields to CombinedSignal:
    - risk_managed_stop_loss
    - risk_managed_take_profit
    - entry_zones (buy_zone_low/high, sell_zone_low/high)
    - position_size_usd, position_size_units, risk_amount_usd, risk_percent
    - protection_plan
    - risk_profile
    """
    
    def __init__(self, portfolio_value: float = None):
        """Initialize risk pipeline"""
        self.config = RiskPipelineConfig()
        
        if portfolio_value is not None:
            self.config.portfolio_value = portfolio_value
        
        # Initialize risk components
        self.risk_analyzer = RiskAnalyzer()
        self.position_sizing = PositionSizingCalculator()
        self.trade_protection = TradeProtectionEngine()
        self.portfolio_heat_manager = PortfolioHeatManager()
        self.correlation_filter = CorrelationRiskFilter()
        
        # Track open positions (updated from performance tracker)
        self.open_positions: List[Dict] = []
        
        # Historical stats for Kelly
        self.historical_stats = {
            'win_rate': self.config.historical_win_rate,
            'avg_win': self.config.historical_avg_win,
            'avg_loss': self.config.historical_avg_loss
        }
        
        log.info("=" * 70)
        log.info("RISK MANAGEMENT PIPELINE (Phase 8) - INITIALIZED")
        log.info("=" * 70)
        log.info(f"  Portfolio Value: ${self.config.portfolio_value:,.2f}")
        log.info(f"  Max Risk/Trade: {self.config.max_risk_per_trade:.1%}")
        log.info(f"  Min RR: {self.config.min_risk_reward}")
        log.info(f"  Target RR: {self.config.target_risk_reward}")
        log.info(f"  Kelly Enabled: {self.config.enable_kelly}")
        log.info(f"  Kelly Fraction: {self.config.kelly_fraction:.0%}")
        log.info(f"  Historical Win Rate: {self.config.historical_win_rate:.1%}")
        log.info(f"  Stop Refinement: {self.config.enable_stop_refinement}")
        log.info("=" * 70)
    
    # ========================================================================
    # MAIN PROCESSING METHOD
    # ========================================================================
    
    def process_signal(self, signal: CombinedSignal, df: pd.DataFrame,
                       market_regime: Dict, structure_data: Dict,
                       sr_data: Dict, volume_data: Dict = None,
                       liquidity_data: Dict = None,
                       orderflow_data: Dict = None,
                       layer_scores: Dict = None) -> Optional[CombinedSignal]:
        """
        Process a signal through risk management (Phase 8)
        
        ADDS risk-managed fields - NEVER overwrites original SL/TP
        
        Args:
            signal: CombinedSignal from Phase 7 (Light Confirm)
            df: OHLCV DataFrame
            market_regime: Market regime data
            structure_data: Market structure data (order blocks, swings, etc.)
            sr_data: Support/resistance data
            volume_data: Volume analysis data
            liquidity_data: Liquidity analysis data
            orderflow_data: Order flow data
            layer_scores: Layer scores from previous phases
        
        Returns:
            Enhanced CombinedSignal with all risk-managed fields
        """
        # Only run on signals that passed Light Confirm
        if not signal or not signal.consensus_reached:
            return None
        
        try:
            current_price = float(df['close'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
            
            # Extract market regime data
            volatility_state = self._get_volatility_state(market_regime, atr, current_price)
            market_regime_type = market_regime.get('regime', 'UNKNOWN') if market_regime else 'UNKNOWN'
            
            # Extract structure data safely
            order_blocks = structure_data.get('order_blocks', []) if structure_data else []
            fair_value_gaps = structure_data.get('fair_value_gaps', []) if structure_data else []
            liquidity_sweeps = liquidity_data.get('sweeps', []) if liquidity_data else []
            swing_points = structure_data.get('swings', []) if structure_data else []
            
            # Get supports and resistances
            supports = sr_data.get('supports', []) if sr_data else []
            resistances = sr_data.get('resistances', []) if sr_data else []
            vwap = sr_data.get('vwap', current_price) if sr_data else current_price
            
            # Get higher timeframe levels
            htf_levels = []
            if 'mtf' in signal.metadata:
                htf_levels = signal.metadata.get('mtf', {}).get('key_levels', {}).get('resistances', []) + \
                             signal.metadata.get('mtf', {}).get('key_levels', {}).get('supports', [])
            
            # Get volume profile
            volume_profile = volume_data.get('volume_profile', {}) if volume_data else {}
            
            # Get technical indicators for entry zones
            rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df else 50
            ema_fast = float(df['ema_8'].iloc[-1]) if 'ema_8' in df else current_price
            ema_slow = float(df['ema_21'].iloc[-1]) if 'ema_21' in df else current_price
            
            # Get ADX and BB width for advanced ATR stop
            adx = float(df['adx'].iloc[-1]) if 'adx' in df else 20
            bb_width = float(df['bb_width'].iloc[-1]) if 'bb_width' in df else 0.04
            
            # Get CVD and absorption data
            cvd_data = orderflow_data.get('cvd', {}) if orderflow_data else None
            absorption_data = orderflow_data.get('absorption', [{}])[0] if orderflow_data else None
            
            # =================================================================
            # STEP 1: CALCULATE ENTRY ZONES
            # =================================================================
            entry_zone = self.risk_analyzer.calculate_entry_zone(
                direction=signal.direction,
                current_price=current_price,
                atr=atr,
                df=df,
                order_blocks=order_blocks,
                fair_value_gaps=fair_value_gaps,
                liquidity_sweeps=liquidity_sweeps,
                supports=supports,
                resistances=resistances,
                swing_points=swing_points,
                htf_levels=htf_levels,
                volume_profile=volume_profile,
                vwap=vwap,
                rsi=rsi,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                cvd_data=cvd_data,
                absorption_data=absorption_data
            )
            
            # =================================================================
            # STEP 2: CALCULATE RISK-MANAGED STOP LOSS
            # =================================================================
            risk_managed_stop = self.risk_analyzer.calculate_risk_managed_stop_loss(
                direction=signal.direction,
                entry_price=signal.entry,
                current_price=current_price,
                atr=atr,
                volatility_regime=volatility_state,
                market_regime=market_regime_type,
                adx=adx,
                bb_width=bb_width,
                order_blocks=order_blocks,
                fair_value_gaps=fair_value_gaps,
                liquidity_sweeps=liquidity_sweeps,
                swing_points=swing_points,
                supports=supports,
                resistances=resistances,
                df=df,
                volume_profile=volume_profile
            )
            
            # =================================================================
            # STEP 3: CALCULATE RISK-MANAGED TAKE PROFIT
            # =================================================================
            # Get liquidity clusters for TP
            liquidity_clusters = []
            if liquidity_data:
                liquidity_clusters = liquidity_data.get('clusters', [])
            
            # Get Fibonacci extensions (simplified)
            fib_extensions = {
                '1.272': signal.entry + (abs(signal.entry - risk_managed_stop.price) * 1.272),
                '1.618': signal.entry + (abs(signal.entry - risk_managed_stop.price) * 1.618),
                '2.0': signal.entry + (abs(signal.entry - risk_managed_stop.price) * 2.0),
                '2.618': signal.entry + (abs(signal.entry - risk_managed_stop.price) * 2.618)
            } if risk_managed_stop else {}
            
            risk_managed_tp = self.risk_analyzer.calculate_risk_managed_take_profit(
                direction=signal.direction,
                entry_price=signal.entry,
                stop_loss=risk_managed_stop.price if risk_managed_stop else signal.stop_loss,
                current_price=current_price,
                atr=atr,
                resistances=resistances,
                supports=supports,
                liquidity_clusters=liquidity_clusters,
                htf_levels=htf_levels,
                fib_extensions=fib_extensions,
                swing_points=swing_points,
                df=df,
                confidence=signal.confidence,
                win_rate=self.config.historical_win_rate
            )
            
            # =================================================================
            # STEP 4: CREATE RISK PROFILE
            # =================================================================
            if layer_scores is None:
                layer_scores = {
                    'pattern': signal.metadata.get('pattern_confidence', 0.5),
                    'price_action': signal.metadata.get('price_action_confidence', 0.5),
                    'smc': signal.metadata.get('smc_confidence', 0.5),
                    'technical': signal.metadata.get('technical_confidence', 0.5),
                    'strategy': signal.metadata.get('strategy_confidence', 0.5)
                }
            
            microstructure_aligned = signal.metadata.get('smart_money_score', 0.5) > 0.6
            htf_aligned = signal.metadata.get('mtf_score', 0.5) > 0.6
            
            risk_profile = create_risk_profile(
                layer_scores=layer_scores,
                microstructure_aligned=microstructure_aligned,
                htf_aligned=htf_aligned,
                market_regime=market_regime_type,
                volatility_state=volatility_state,
                historical_stats=self.historical_stats
            )
            
            # =================================================================
            # STEP 5: CHECK CORRELATION RISK
            # =================================================================
            corr_allowed, corr_reason, max_corr, corr_reduction = self.correlation_filter.check_correlation_risk(
                symbol=signal.symbol if hasattr(signal, 'symbol') else "UNKNOWN",
                correlation_data=structure_data,
                existing_positions=self.open_positions
            )
            
            if not corr_allowed:
                log.debug(f"[RISK] Correlation risk - {corr_reason}")
            
            # =================================================================
            # STEP 6: CALCULATE PORTFOLIO HEAT
            # =================================================================
            portfolio_heat = self.portfolio_heat_manager.calculate_portfolio_heat(
                portfolio_value=self.config.portfolio_value,
                positions=self.open_positions,
                correlation_data=structure_data
            )
            
            # =================================================================
            # STEP 7: CALCULATE POSITION SIZE
            # =================================================================
            position_result = self.position_sizing.calculate(
                confidence=signal.confidence,
                grade=signal.grade if hasattr(signal, 'grade') else "B",
                market_regime=market_regime_type,
                volatility_state=volatility_state,
                win_rate=self.config.historical_win_rate,
                avg_win=self.config.historical_avg_win,
                avg_loss=self.config.historical_avg_loss,
                correlation_reduction=corr_reduction,
                portfolio_heat=portfolio_heat.heat_score,
                entry_price=entry_zone.buy_zone_high if signal.direction == 'BUY' else entry_zone.sell_zone_low,
                stop_loss=risk_managed_stop.price if risk_managed_stop else None
            )
            
            # =================================================================
            # STEP 8: GET PROTECTION PLAN
            # =================================================================
            protection_plan = self.trade_protection.get_protection_plan(risk_profile.signal_quality)
            
            # =================================================================
            # STEP 9: VALIDATE RISK/REWARD
            # =================================================================
            risk_reward = risk_managed_tp.risk_reward if risk_managed_tp else 0
            rr_valid, rr_message = validate_risk_reward(risk_reward, self.config.min_risk_reward)
            
            # =================================================================
            # STEP 10: ENHANCE COMBINED SIGNAL (ADD risk-managed fields)
            # =================================================================
            # Add entry zones
            signal.buy_zone_low = entry_zone.buy_zone_low
            signal.buy_zone_high = entry_zone.buy_zone_high
            signal.sell_zone_low = entry_zone.sell_zone_low
            signal.sell_zone_high = entry_zone.sell_zone_high
            signal.entry_zone_method = entry_zone.method
            signal.entry_zone_confidence = entry_zone.confidence
            
            # Add risk-managed SL/TP (PRESERVE original)
            signal.risk_managed_stop_loss = risk_managed_stop.price if risk_managed_stop else None
            signal.risk_managed_take_profit = risk_managed_tp.price if risk_managed_tp else None
            
            # Add position sizing
            signal.position_size_usd = position_result.position_size_usd
            signal.position_size_units = position_result.position_size_units
            signal.risk_amount_usd = position_result.risk_amount_usd
            signal.risk_percent = position_result.risk_percent
            signal.position_multiplier = position_result.position_multiplier
            
            # Add protection plan
            signal.protection_plan = {
                'trailing_stop': protection_plan.trailing_stop,
                'partial_tp_levels': protection_plan.partial_tp_levels,
                'breakeven_trigger': protection_plan.breakeven_trigger,
                'time_exit_bars': protection_plan.time_exit_bars,
                'adverse_excursion_limit': protection_plan.adverse_excursion_limit
            }
            
            # Add risk profile
            signal.risk_profile = {
                'signal_quality': risk_profile.signal_quality.value,
                'probability': risk_profile.probability,
                'kelly_fraction': risk_profile.kelly_fraction,
                'risk_of_ruin': risk_profile.risk_of_ruin,
                'layer_scores': risk_profile.layer_scores
            }
            
            # =================================================================
            # STEP 11: ADD METADATA (for debugging)
            # =================================================================
            if not hasattr(signal, 'risk_metadata'):
                signal.risk_metadata = {}
            
            signal.risk_metadata = {
                'stop_loss': {
                    'method': risk_managed_stop.method if risk_managed_stop else 'original',
                    'confidence': risk_managed_stop.confidence if risk_managed_stop else 0.5,
                    'distance_pct': risk_managed_stop.distance_pct if risk_managed_stop else 0,
                    'atr_multiple': risk_managed_stop.atr_multiple if risk_managed_stop else 0,
                    'reason': risk_managed_stop.reason if risk_managed_stop else 'Using original stop'
                },
                'take_profit': {
                    'method': risk_managed_tp.method if risk_managed_tp else 'original',
                    'confidence': risk_managed_tp.confidence if risk_managed_tp else 0.5,
                    'risk_reward': risk_managed_tp.risk_reward if risk_managed_tp else 0,
                    'reason': risk_managed_tp.reason if risk_managed_tp else 'Using original TP'
                },
                'entry_zone': {
                    'method': entry_zone.method,
                    'confidence': entry_zone.confidence,
                    'width_pct': entry_zone.width_pct,
                    'contributing_methods': entry_zone.contributing_methods
                },
                'position_sizing': position_result.factor_breakdown,
                'portfolio_heat': {
                    'score': portfolio_heat.heat_score,
                    'status': portfolio_heat.status,
                    'total_exposure': portfolio_heat.total_exposure,
                    'sector_exposure': portfolio_heat.sector_exposure
                },
                'correlation_risk': {
                    'max_correlation': max_corr,
                    'reduction_factor': corr_reduction,
                    'reason': corr_reason
                },
                'risk_profile': {
                    'signal_quality': risk_profile.signal_quality.value,
                    'probability': risk_profile.probability,
                    'microstructure_aligned': risk_profile.microstructure_aligned,
                    'htf_aligned': risk_profile.htf_aligned,
                    'kelly_fraction': risk_profile.kelly_fraction,
                    'risk_of_ruin': risk_profile.risk_of_ruin
                }
            }
            
            # =================================================================
            # STEP 12: ADD WARNING FLAGS
            # =================================================================
            signal.warning_flags = []
            
            if not rr_valid:
                signal.warning_flags.append(f"Low RR: {risk_reward:.2f} < {self.config.min_risk_reward}")
            
            if position_result.risk_percent < 0.005:
                signal.warning_flags.append(f"Very small position size ({position_result.risk_percent:.2%} risk)")
            
            if risk_profile.risk_of_ruin > 0.1:
                signal.warning_flags.append(f"Risk of ruin: {risk_profile.risk_of_ruin:.1%}")
            
            if portfolio_heat.heat_score > self.config.max_portfolio_heat:
                signal.warning_flags.append(f"High portfolio heat: {portfolio_heat.heat_score:.0f}")
            
            if entry_zone.confidence < 0.5:
                signal.warning_flags.append(f"Low entry zone confidence: {entry_zone.confidence:.0%}")
            
            # =================================================================
            # STEP 13: UPDATE STATUS
            # =================================================================
            signal.status = 'RISK_PASSED'
            signal.timestamp = datetime.now()
            
            # =================================================================
            # STEP 14: PRINT OUTPUT
            # =================================================================
            self._print_risk_output(signal, risk_managed_stop, risk_managed_tp, 
                                    entry_zone, position_result, portfolio_heat, risk_profile)
            
            return signal
            
        except Exception as e:
            log.error(f"Error in risk pipeline: {e}", exc_info=True)
            return signal  # Return original signal on error
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _get_volatility_state(self, market_regime: Dict, atr: float, price: float) -> str:
        """Determine volatility regime"""
        if market_regime and 'volatility_state' in market_regime:
            return market_regime.get('volatility_state', 'NORMAL')
        
        atr_pct = atr / price if price > 0 else 0.02
        
        if atr_pct < 0.005:
            return 'LOW'
        elif atr_pct < 0.015:
            return 'NORMAL'
        elif atr_pct < 0.03:
            return 'HIGH'
        else:
            return 'EXTREME'
    
    # ========================================================================
    # PORTFOLIO MANAGEMENT
    # ========================================================================
    
    def update_portfolio(self, positions: List[Dict]):
        """Update open positions from performance tracker"""
        self.open_positions = positions
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value"""
        self.config.portfolio_value = new_value
    
    def update_historical_stats(self, win_rate: float, avg_win: float, avg_loss: float):
        """Update historical stats for Kelly calculation"""
        self.historical_stats = {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        self.config.historical_win_rate = win_rate
        self.config.historical_avg_win = avg_win
        self.config.historical_avg_loss = avg_loss
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def _print_risk_output(self, signal: CombinedSignal, 
                           risk_managed_stop: Optional[StopLossResult],
                           risk_managed_tp: Optional[TakeProfitResult],
                           entry_zone: EntryZoneResult,
                           position_result: PositionSizeResult,
                           portfolio_heat: PortfolioHeat,
                           risk_profile: RiskProfile):
        """Print formatted risk output"""
        log.info("\n" + "=" * 80)
        log.info(f"[RISK MANAGEMENT] {signal.symbol if hasattr(signal, 'symbol') else 'UNKNOWN'} - {signal.direction}")
        log.info("=" * 80)
        
        # Signal Quality
        log.info(f"\n📊 SIGNAL QUALITY: {risk_profile.signal_quality.value}")
        log.info(f"   Confidence: {signal.confidence:.1%}")
        log.info(f"   Probability: {risk_profile.probability:.1%}")
        log.info(f"   Kelly Fraction: {risk_profile.kelly_fraction:.1%}")
        log.info(f"   Risk of Ruin: {risk_profile.risk_of_ruin:.1%}")
        
        # Entry Zone
        log.info(f"\n🎯 ENTRY ZONE:")
        if signal.direction == 'BUY':
            log.info(f"   BUY Zone: {entry_zone.buy_zone_low:.6f} - {entry_zone.buy_zone_high:.6f}")
        else:
            log.info(f"   SELL Zone: {entry_zone.sell_zone_low:.6f} - {entry_zone.sell_zone_high:.6f}")
        log.info(f"   Method: {entry_zone.method}")
        log.info(f"   Confidence: {entry_zone.confidence:.0%}")
        log.info(f"   Width: {entry_zone.width_pct:.2%}")
        
        # Original vs Risk-Managed Stop Loss
        log.info(f"\n🛑 STOP LOSS:")
        log.info(f"   Original: {signal.stop_loss:.6f}")
        if risk_managed_stop:
            log.info(f"   Risk-Managed: {risk_managed_stop.price:.6f} ({risk_managed_stop.method})")
            log.info(f"   Distance: {risk_managed_stop.distance_pct:.2%} ({risk_managed_stop.atr_multiple:.1f}x ATR)")
            log.info(f"   Confidence: {risk_managed_stop.confidence:.0%}")
        
        # Original vs Risk-Managed Take Profit
        log.info(f"\n🎯 TAKE PROFIT:")
        log.info(f"   Original: {signal.take_profit:.6f}")
        if risk_managed_tp:
            log.info(f"   Risk-Managed: {risk_managed_tp.price:.6f} ({risk_managed_tp.method})")
            log.info(f"   Risk/Reward: {risk_managed_tp.risk_reward:.2f}:1")
            log.info(f"   Confidence: {risk_managed_tp.confidence:.0%}")
        
        # Position Sizing
        log.info(f"\n💰 POSITION SIZING:")
        log.info(f"   Position Size: ${position_result.position_size_usd:,.2f}")
        log.info(f"   Position Units: {position_result.position_size_units:.4f}")
        log.info(f"   Risk Amount: ${position_result.risk_amount_usd:,.2f}")
        log.info(f"   Risk Percent: {position_result.risk_percent:.2%}")
        log.info(f"   Position Multiplier: {position_result.position_multiplier:.2f}x")
        
        # Factor Breakdown
        log.info(f"\n📈 FACTOR BREAKDOWN:")
        for factor, value in position_result.factor_breakdown.items():
            log.info(f"   {factor}: {value:.3f}")
        
        # Portfolio Heat
        log.info(f"\n🔥 PORTFOLIO HEAT: {portfolio_heat.heat_score:.0f} ({portfolio_heat.status})")
        log.info(f"   Total Exposure: {portfolio_heat.total_exposure:.1%}")
        log.info(f"   Correlation Risk: {portfolio_heat.correlation_risk:.1%}")
        log.info(f"   Concentration Risk: {portfolio_heat.concentration_risk:.1%}")
        
        # Protection Plan
        log.info(f"\n🛡️ PROTECTION PLAN:")
        log.info(f"   Trailing Stop: {signal.protection_plan.get('trailing_stop', {})}")
        log.info(f"   Partial TP Levels: {len(signal.protection_plan.get('partial_tp_levels', []))} levels")
        log.info(f"   Breakeven Trigger: {signal.protection_plan.get('breakeven_trigger', 0):.1%}")
        log.info(f"   Time Exit: {signal.protection_plan.get('time_exit_bars', 0)} bars")
        
        # Warning Flags
        if signal.warning_flags:
            log.info(f"\n⚠️ WARNINGS:")
            for warning in signal.warning_flags[:5]:
                log.info(f"   • {warning}")
        
        log.info("=" * 80)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def apply_risk_management(signal: CombinedSignal, df: pd.DataFrame,
                          market_regime: Dict, structure_data: Dict,
                          sr_data: Dict, volume_data: Dict = None,
                          liquidity_data: Dict = None,
                          orderflow_data: Dict = None,
                          portfolio_value: float = None) -> Optional[CombinedSignal]:
    """
    Convenience function to apply risk management to a signal
    
    Args:
        signal: CombinedSignal from Phase 7
        df: OHLCV DataFrame
        market_regime: Market regime data
        structure_data: Market structure data
        sr_data: Support/resistance data
        volume_data: Volume analysis data
        liquidity_data: Liquidity analysis data
        orderflow_data: Order flow data
        portfolio_value: Optional portfolio value override
    
    Returns:
        Enhanced CombinedSignal with risk-managed fields
    """
    pipeline = RiskPipeline(portfolio_value)
    return pipeline.process_signal(
        signal=signal,
        df=df,
        market_regime=market_regime,
        structure_data=structure_data,
        sr_data=sr_data,
        volume_data=volume_data,
        liquidity_data=liquidity_data,
        orderflow_data=orderflow_data
    )