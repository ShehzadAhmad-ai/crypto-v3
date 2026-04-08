# main.py - COMPLETE TRADING SYSTEM V3 ORCHESTRATOR
"""
╔═════════════════════════════════════════════════════════════════���══════════════╗
║                      TRADING SYSTEM V3 - MAIN ORCHESTRATOR                     ║
║           13-Phase Multi-Expert Consensus Trading System with ML               ║
║                      Production-Ready Deployment Version                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

COMPLETE SYSTEM FLOW:
    Phase 1:  Initialization & Coin Selection (Dynamic + Target coins)
    Phase 2:  Data Fetching (OHLCV, HTF, Order Book, Liquidations)
    Phase 3:  5 Expert Modules IN PARALLEL (Pattern, Price Action, SMC, Technical, Strategy)
    Phase 4:  Expert Aggregator (Consensus & TP Level Building)
    Phase 5:  Multi-Timeframe Confirmation (Alignment scoring, Boost/Penalty)
    Phase 6:  Smart Money Filter (Liquidity, Traps, Sweeps)
    Phase 7:  Light Confirmations (Cross-Asset, Funding, OI, Sentiment)
    Phase 8:  Risk Management (SL/TP Priority, Entry Zones, Position Sizing)
    Phase 9:  Final Scoring (Weighted Ensemble, Grade Assignment)
    Phase 10: Signal Validator (Fakeouts, Traps, Cooldown, Expiry)
    Phase 11: Timing Predictor (Entry Timing with ML Ensemble)
    Phase 12: Signal Output & Storage (JSON, CSV, TXT)
    Phase 13: Performance Tracking (Outcome Analysis, Weight Updates)

Author: Trading System V3
Version: 3.1 - Production Ready
Last Updated: 2026-04-08
"""

import sys
import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION IMPORTS
# ============================================================================

from config import (
    DEBUG, TRADING_MODE, LOG_LEVEL, PRIMARY_TIMEFRAME, 
    TARGET_COINS, PORTFOLIO_VALUE, EXPERT_NAMES,
    SIGNALS_RAW_DIR, SIGNALS_FINAL_DIR, SIGNALS_SUMMARY_DIR,
    PERFORMANCE_DAILY_DIR, PERFORMANCE_MONTHLY_DIR, PERFORMANCE_TRADES_DIR,
    LOGS_DIR, get_config_summary, validate_config,
    MIN_EXPERTS_TO_AGREE, MIN_EXPERTS_WITH_SIGNAL,
    MIN_EXPERT_CONSENSUS_CONFIDENCE, FINAL_MIN_PROBABILITY,
    MAX_SIGNAL_AGE_MINUTES, SYMBOL_COOLDOWN_MINUTES,
    TRADING_MODE, MTF_HIGHER_TIMEFRAMES
)

# ============================================================================
# CORE MODULE IMPORTS
# ============================================================================

from directory_manager import DirectoryManager
from logger import setup_logger
from data_fetcher import DataFetcher
from cooldown_manager import CooldownManager
from performance_tracker import PerformanceTracker
from dynamic_coin_selector import DynamicCoinSelector

# ============================================================================
# PHASE PIPELINE IMPORTS
# ============================================================================

from expert_executor import ExpertExecutor
from expert_aggregator import ExpertAggregator
from mtf_pipeline import MTFPipeline
from smart_money_pipeline import SmartMoneyPipeline
from light_confirm_pipeline import LightConfirmPipeline
from risk_pipeline import RiskPipeline
from final_scoring_pipeline import FinalScoringPipeline
from signal_validator import SignalValidator
from entry_timing_predictor import EntryTimingPredictor
from signal_exporter import SignalExporter
from signal_summary import SignalSummary
from trade_outcome_analyzer import TradeOutcomeAnalyzer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_banner(text: str, char: str = "=", width: int = 85):
    """Print formatted banner"""
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_phase_header(phase_num: int, phase_name: str, width: int = 85):
    """Print phase header"""
    text = f"PHASE {phase_num}: {phase_name}"
    print(f"\n{'─' * width}")
    print(f"▶ {text}")
    print(f"{'─' * width}\n")


# ============================================================================
# MAIN TRADING SYSTEM CLASS
# ============================================================================

class TradingSystemV3:
    """
    Complete Trading System V3 - Production Ready
    
    Orchestrates all 13 phases with:
    - Parallel expert execution
    - Complete zone entry tracking
    - Proper outcome classification
    - Machine learning timing prediction
    - Dynamic weight updates
    """
    
    def __init__(self, config_override: Dict = None):
        """
        Initialize the trading system
        
        Args:
            config_override: Optional config overrides for testing
        """
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.session_id = self.start_time.strftime("%Y-%m-%d")
        
        # ===== INITIALIZE CORE COMPONENTS =====
        self.dir_manager = DirectoryManager()
        self.logger = setup_logger(
            name="TradingSystemV3",
            log_level=LOG_LEVEL,
            log_dir=LOGS_DIR
        )
        
        self.data_fetcher = DataFetcher()
        self.cooldown_manager = CooldownManager()
        self.performance_tracker = PerformanceTracker()
        
        # ===== INITIALIZE PHASE MODULES =====
        self.coin_selector = DynamicCoinSelector(self.data_fetcher, self.logger)
        self.expert_executor = ExpertExecutor(self.data_fetcher, self.logger)
        self.expert_aggregator = ExpertAggregator(self.logger)
        self.mtf_pipeline = MTFPipeline(self.data_fetcher, self.logger)
        self.smart_money_pipeline = SmartMoneyPipeline(self.data_fetcher, self.logger)
        self.light_confirm_pipeline = LightConfirmPipeline(self.data_fetcher, self.logger)
        self.risk_pipeline = RiskPipeline(self.logger)
        self.final_scoring = FinalScoringPipeline(self.logger)
        self.signal_validator = SignalValidator(self.cooldown_manager, self.logger)
        self.timing_predictor = EntryTimingPredictor(self.logger)
        self.signal_exporter = SignalExporter(self.logger)
        self.signal_summary = SignalSummary(self.logger)
        self.trade_analyzer = TradeOutcomeAnalyzer(self.logger)
        
        # ===== INITIALIZE STATISTICS =====
        self.cycle_stats = {
            'symbols_analyzed': 0,
            'signals_generated': 0,
            'signals_from_phase_4': 0,
            'signals_after_mtf': 0,
            'signals_after_sm': 0,
            'signals_after_lc': 0,
            'signals_after_risk': 0,
            'signals_after_scoring': 0,
            'signals_after_validation': 0,
            'final_signals': 0,
            'errors': 0,
        }
        
        self.all_signals = []
        self.final_signals = []
        self.rejected_signals = []
        
        self.daily_summary = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'run_id': self.run_id,
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': None,
            'total_symbols_analyzed': 0,
            'total_signals_generated': 0,
            'final_signals': 0,
            'by_direction': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'by_grade': {},
            'by_symbol': {},
            'signals_by_phase': self.cycle_stats.copy(),
            'performance_stats': {},
            'previous_day_performance': None,
            'processing_time_seconds': 0,
            'errors': []
        }
        
        # Print initialization banner
        print_banner("🚀 TRADING SYSTEM V3 - INITIALIZED")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Session: {self.session_id}")
        self.logger.info(f"Trading Mode: {TRADING_MODE}")
        self.logger.info(f"Portfolio Value: ${PORTFOLIO_VALUE:,.2f}")
        self.logger.info(f"Min Experts to Agree: {MIN_EXPERTS_TO_AGREE}/5")
        self.logger.info(f"Min Consensus Confidence: {MIN_EXPERT_CONSENSUS_CONFIDENCE:.0%}")
        self.logger.info(f"Primary Timeframe: {PRIMARY_TIMEFRAME}")
    
    
    # ========================================================================
    # PHASE 1: INITIALIZATION & COIN SELECTION
    # ========================================================================
    
    def phase_1_initialization_and_coin_selection(self) -> List[str]:
        """
        PHASE 1: Initialization & Coin Selection
        
        Steps:
        1.1: Validate configuration
        1.2: Load previous day's performance
        1.3: Dynamic coin selection
        
        Returns:
            List of symbols to analyze
        """
        print_phase_header(1, "INITIALIZATION & COIN SELECTION")
        phase_start = time.time()
        
        try:
            # Step 1.1: Validate configuration
            self.logger.info("📋 STEP 1.1: Validating configuration...")
            config_issues = validate_config()
            
            if config_issues:
                self.logger.warning(f"⚠️  {len(config_issues)} configuration issues found:")
                for issue in config_issues:
                    self.logger.warning(f"   • {issue}")
            else:
                self.logger.info("✅ Configuration validated successfully")
            
            # Display config summary
            config_summary = get_config_summary()
            self.logger.info("📊 Configuration Summary:")
            self.logger.info(f"   Mode: {config_summary['system']['trading_mode']}")
            self.logger.info(f"   Portfolio: ${config_summary['system']['portfolio_value']:,.0f}")
            self.logger.info(f"   Max Positions: {config_summary['system']['max_positions']}")
            self.logger.info(f"   Expert Agreement: {config_summary['expert_agreement']['min_experts_to_agree']}/5")
            
            # Step 1.2: Load and analyze previous day's performance
            self.logger.info("\n📈 STEP 1.2: Analyzing previous day's performance...")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            try:
                yesterday_perf = self._load_previous_performance(yesterday)
                if yesterday_perf:
                    self.daily_summary['previous_day_performance'] = yesterday_perf
                    self.logger.info(f"✅ Previous day ({yesterday}) loaded:")
                    self.logger.info(f"   Total Signals: {yesterday_perf.get('total_signals', 0)}")
                    self.logger.info(f"   Wins: {yesterday_perf.get('wins', 0)}")
                    self.logger.info(f"   Losses: {yesterday_perf.get('losses', 0)}")
                    
                    win_rate = yesterday_perf.get('win_rate', 0)
                    profit_factor = yesterday_perf.get('profit_factor', 0)
                    
                    self.logger.info(f"   Win Rate: {win_rate:.1%}")
                    self.logger.info(f"   Profit Factor: {profit_factor:.2f}")
                    self.logger.info(f"   Avg R:R: {yesterday_perf.get('avg_rr', 0):.2f}")
                else:
                    self.logger.info("ℹ️  No previous day performance found (first run)")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not load previous performance: {e}")
            
            # Step 1.3: Dynamic coin selection
            self.logger.info("\n🎯 STEP 1.3: Selecting coins to analyze...")
            symbols = self.coin_selector.select_coins()
            
            self.logger.info(f"✅ Selected {len(symbols)} coins")
            self.logger.info(f"   Target coins: {TARGET_COINS}")
            self.logger.info(f"   First 10 coins: {symbols[:10]}")
            
            phase_time = time.time() - phase_start
            self.logger.info(f"\n⏱️  Phase 1 completed in {format_time(phase_time)}")
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"❌ Phase 1 failed: {e}")
            self.logger.error(traceback.format_exc())
            self.daily_summary['errors'].append(f"Phase 1: {str(e)}")
            raise
    
    
    def _load_previous_performance(self, date_str: str) -> Optional[Dict]:
        """Load previous day's performance report"""
        report_path = os.path.join(PERFORMANCE_DAILY_DIR, f"{date_str}.json")
        
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.debug(f"Could not load performance report: {e}")
        
        return None
    
    
    # ========================================================================
    # PHASE 2: DATA FETCHING
    # ========================================================================
    
    def phase_2_data_fetching(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        PHASE 2: Data Fetching
        
        For each coin, fetch:
        2.1: Primary timeframe OHLCV
        2.2: Higher timeframes (15m, 1h, 4h)
        2.3: Order book data
        2.4: Liquidation data
        2.5: Market regime & indicators
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            Complete data dictionary or None on error
        """
        try:
            self.logger.debug(f"📊 Fetching data for {symbol}...")
            
            # Fetch primary OHLCV
            df = self.data_fetcher.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME)
            if df is None or len(df) < 200:
                self.logger.debug(f"⚠️  Insufficient data for {symbol}")
                return None
            
            # Record fetch time for later outcome tracking
            fetch_time = datetime.now()
            
            # Fetch HTF data
            htf_data = self.data_fetcher.fetch_higher_timeframes(symbol)
            
            # Fetch order book
            order_book = self.data_fetcher.fetch_order_book(symbol)
            
            # Fetch liquidations
            liquidations = self.data_fetcher.fetch_liquidations(symbol)
            
            # Calculate market regime
            market_regime = self.data_fetcher.calculate_market_regime(df)
            
            # Calculate all indicators (50+)
            indicators = self.data_fetcher.calculate_all_indicators(df)
            
            # Get current price
            current_price = float(df['close'].iloc[-1])
            
            # Build complete data package
            data = {
                'symbol': symbol,
                'df': df,
                'htf_data': htf_data,
                'order_book': order_book,
                'liquidations': liquidations,
                'market_regime': market_regime,
                'indicators': indicators,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'fetch_time': fetch_time,  # Track when data was fetched
                'last_close_time': df.index[-1],
                'atr': indicators.get('atr', 0) if indicators else 0,
            }
            
            self.logger.debug(f"✅ Data fetched for {symbol} (Price: ${current_price:.2f})")
            return data
            
        except Exception as e:
            self.logger.debug(f"❌ Data fetch error for {symbol}: {e}")
            self.daily_summary['errors'].append(f"Data fetch {symbol}: {str(e)}")
            return None
    
    
    # ========================================================================
    # PHASE 3: EXPERT MODULES (PARALLEL EXECUTION)
    # ========================================================================
    
    def phase_3_run_experts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 3: Run 5 Expert Modules in Parallel
        
        Experts:
        3.1: Pattern V3 (Harmonic, Structure, Volume patterns)
        3.2: Price Action V3.5 (Candlestick patterns, Sequences)
        3.3: SMC V3 (Market Structure, Order Blocks, FVG, Liquidity)
        3.4: Technical Analyzer V2.0 (50+ Indicators, Divergences)
        3.5: Strategy Expert (15+ Strategies, Consensus voting)
        
        All experts run in parallel for performance
        
        Args:
            data: Market data from Phase 2
            
        Returns:
            Dictionary with signals from all 5 experts
        """
        try:
            symbol = data['symbol']
            self.logger.debug(f"🔄 Running 5 experts in PARALLEL for {symbol}...")
            
            expert_signals = {}
            
            # Run experts in parallel
            expert_signals = self.expert_executor.run_all_experts(
                data,
                symbol=symbol,
                timeframe=PRIMARY_TIMEFRAME,
                regime_data=data.get('market_regime', {}),
                htf_data=data.get('htf_data', {}),
                order_book=data.get('order_book'),
                liquidations=data.get('liquidations'),
                indicators=data.get('indicators', {})
            )
            
            if expert_signals:
                # Save raw signals for analysis
                self.signal_exporter.save_raw_signals(symbol, expert_signals)
                
                # Count signals generated
                signals_count = sum(1 for s in expert_signals.values() 
                                  if s and s.get('direction') != 'HOLD')
                self.logger.debug(f"✅ {signals_count}/5 experts generated signals for {symbol}")
            else:
                self.logger.debug(f"⚠️  No expert signals for {symbol}")
            
            return expert_signals
            
        except Exception as e:
            self.logger.debug(f"❌ Expert execution error for {symbol}: {e}")
            self.daily_summary['errors'].append(f"Experts {symbol}: {str(e)}")
            return {}
    
    
    # ========================================================================
    # PHASE 4: EXPERT AGGREGATOR (CONSENSUS)
    # ========================================================================
    
    def phase_4_expert_aggregator(self, expert_signals: Dict[str, Any],
                                 data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        PHASE 4: Expert Aggregator - Consensus & Combination
        
        Steps:
        4.1: Check expert consensus (min experts to agree)
        4.2: Build take profit levels from all experts
        4.3: Aggregate confidence, entry, SL, reasoning
        
        Args:
            expert_signals: Signals from all 5 experts
            data: Market data
            
        Returns:
            Combined signal or None if no consensus
        """
        try:
            symbol = data['symbol']
            self.logger.debug(f"🤝 PHASE 4: Checking expert consensus for {symbol}...")
            
            # Check signal count
            signal_count = sum(1 for s in expert_signals.values() 
                             if s and s.get('direction') != 'HOLD')
            
            if signal_count < MIN_EXPERTS_WITH_SIGNAL:
                self.logger.debug(f"⚠️  Only {signal_count}/{MIN_EXPERTS_WITH_SIGNAL} experts generated signals")
                return None
            
            # Aggregate signals
            combined_signal = self.expert_aggregator.aggregate_signals(
                expert_signals=expert_signals,
                data=data,
                symbol=symbol,
                current_price=data['current_price'],
                atr=data.get('atr', 0)
            )
            
            if combined_signal and combined_signal.get('consensus_reached'):
                self.cycle_stats['signals_from_phase_4'] += 1
                self.logger.debug(f"✅ Consensus reached for {symbol}")
                self.logger.debug(f"   Direction: {combined_signal.get('direction')}")
                self.logger.debug(f"   Confidence: {combined_signal.get('confidence', 0):.1%}")
                self.logger.debug(f"   Experts agreed: {combined_signal.get('experts_agreed', 0)}")
                return combined_signal
            else:
                self.logger.debug(f"❌ No consensus for {symbol}")
                return None
                
        except Exception as e:
            self.logger.debug(f"❌ Aggregation error for {symbol}: {e}")
            self.daily_summary['errors'].append(f"Aggregator {symbol}: {str(e)}")
            return None
    
    
    # ========================================================================
    # PHASE 5: MTF CONFIRMATION
    # ========================================================================
    
    def phase_5_mtf_confirmation(self, combined_signal: Dict[str, Any],
                                data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 5: Multi-Timeframe Confirmation
        
        Steps:
        5.1: Analyze higher timeframes (15m, 1h, 4h)
        5.2: Calculate alignment score
        5.3: Apply confidence boost/penalty
        5.4: Detect pullback to HTF levels
        
        Args:
            combined_signal: Signal from Phase 4
            data: Market data
            
        Returns:
            Updated signal with MTF adjustments
        """
        try:
            symbol = data['symbol']
            original_confidence = combined_signal.get('confidence', 0)
            
            self.logger.debug(f"🔄 PHASE 5: MTF Confirmation for {symbol}...")
            
            mtf_signal = self.mtf_pipeline.apply_mtf_confirmation(
                combined_signal=combined_signal,
                data=data,
                htf_data=data.get('htf_data', {})
            )
            
            new_confidence = mtf_signal.get('confidence', original_confidence)
            confidence_change = new_confidence - original_confidence
            
            self.cycle_stats['signals_after_mtf'] += 1
            
            self.logger.debug(f"✅ MTF confirmation applied")
            self.logger.debug(f"   Confidence: {original_confidence:.1%} → {new_confidence:.1%} ({confidence_change:+.1%})")
            self.logger.debug(f"   HTF Alignment: {mtf_signal.get('mtf_alignment_score', 0):.1%}")
            
            return mtf_signal
            
        except Exception as e:
            self.logger.debug(f"❌ MTF confirmation error: {e}")
            self.daily_summary['errors'].append(f"MTF {symbol}: {str(e)}")
            return combined_signal
    
    
    # ========================================================================
    # PHASE 6: SMART MONEY FILTER
    # ========================================================================
    
    def phase_6_smart_money_filter(self, mtf_signal: Dict[str, Any],
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 6: Smart Money Filter
        
        Steps:
        6.1: Liquidity analysis (sweeps, inducements)
        6.2: Trap detection (bull/bear traps, stop hunts)
        6.3: Apply trap adjustments
        
        Args:
            mtf_signal: Signal from Phase 5
            data: Market data
            
        Returns:
            Updated signal with smart money analysis
        """
        try:
            symbol = data['symbol']
            original_confidence = mtf_signal.get('confidence', 0)
            
            self.logger.debug(f"💰 PHASE 6: Smart Money Filter for {symbol}...")
            
            sm_signal = self.smart_money_pipeline.apply_smart_money_filter(
                signal=mtf_signal,
                data=data,
                order_book=data.get('order_book'),
                liquidations=data.get('liquidations')
            )
            
            new_confidence = sm_signal.get('confidence', original_confidence)
            confidence_change = new_confidence - original_confidence
            
            self.cycle_stats['signals_after_sm'] += 1
            
            self.logger.debug(f"✅ Smart Money filter applied")
            self.logger.debug(f"   Confidence: {original_confidence:.1%} → {new_confidence:.1%} ({confidence_change:+.1%})")
            self.logger.debug(f"   SM Score: {sm_signal.get('smart_money_score', 0):.1%}")
            
            return sm_signal
            
        except Exception as e:
            self.logger.debug(f"❌ Smart money filter error: {e}")
            self.daily_summary['errors'].append(f"SmartMoney {symbol}: {str(e)}")
            return mtf_signal
    
    
    # ========================================================================
    # PHASE 7: LIGHT CONFIRMATIONS
    # ========================================================================
    
    def phase_7_light_confirmations(self, sm_signal: Dict[str, Any],
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 7: Light Confirmations
        
        Steps:
        7.1: Cross-asset analysis (BTC dominance, sector rotation)
        7.2: Funding & OI analysis
        7.3: Sentiment analysis (Fear & Greed)
        7.4: Apply confirmation score
        
        Args:
            sm_signal: Signal from Phase 6
            data: Market data
            
        Returns:
            Updated signal with light confirmations
        """
        try:
            symbol = data['symbol']
            original_confidence = sm_signal.get('confidence', 0)
            
            self.logger.debug(f"💡 PHASE 7: Light Confirmations for {symbol}...")
            
            lc_signal = self.light_confirm_pipeline.apply_light_confirmations(
                signal=sm_signal,
                data=data,
                symbol=symbol
            )
            
            new_confidence = lc_signal.get('confidence', original_confidence)
            confidence_change = new_confidence - original_confidence
            
            self.cycle_stats['signals_after_lc'] += 1
            
            self.logger.debug(f"✅ Light confirmations applied")
            self.logger.debug(f"   Confidence: {original_confidence:.1%} → {new_confidence:.1%} ({confidence_change:+.1%})")
            self.logger.debug(f"   LC Score: {lc_signal.get('light_confirm_score', 0):.1%}")
            
            return lc_signal
            
        except Exception as e:
            self.logger.debug(f"❌ Light confirmations error: {e}")
            self.daily_summary['errors'].append(f"LightConfirm {symbol}: {str(e)}")
            return sm_signal
    
    
    # ========================================================================
    # PHASE 8: RISK MANAGEMENT
    # ========================================================================
    
    def phase_8_risk_management(self, lc_signal: Dict[str, Any],
                               data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 8: Risk Management
        
        Steps:
        8.1: Dynamic stop loss (priority hierarchy)
        8.2: Dynamic entry zone (multiple methods)
        8.3: Dynamic take profit levels
        8.4: Position sizing (Kelly criterion, heat management)
        
        Args:
            lc_signal: Signal from Phase 7
            data: Market data
            
        Returns:
            Signal with complete risk management
        """
        try:
            symbol = data['symbol']
            
            self.logger.debug(f"⚠️  PHASE 8: Risk Management for {symbol}...")
            
            risk_signal = self.risk_pipeline.apply_risk_management(
                signal=lc_signal,
                data=data,
                portfolio_value=PORTFOLIO_VALUE,
                current_price=data['current_price'],
                atr=data.get('atr', 0)
            )
            
            self.cycle_stats['signals_after_risk'] += 1
            
            self.logger.debug(f"✅ Risk management applied")
            self.logger.debug(f"   Entry: ${risk_signal.get('entry', 0):.2f}")
            self.logger.debug(f"   Entry Zone: ${risk_signal.get('entry_zone_low', 0):.2f} - ${risk_signal.get('entry_zone_high', 0):.2f}")
            self.logger.debug(f"   Stop Loss: ${risk_signal.get('stop_loss', 0):.2f}")
            self.logger.debug(f"   Position Size: ${risk_signal.get('position_size', 0):.2f}")
            self.logger.debug(f"   Risk/Reward: {risk_signal.get('risk_reward', 0):.2f}")
            
            return risk_signal
            
        except Exception as e:
            self.logger.debug(f"❌ Risk management error: {e}")
            self.daily_summary['errors'].append(f"RiskMgmt {symbol}: {str(e)}")
            return lc_signal
    
    
    # ========================================================================
    # PHASE 9: FINAL SCORING
    # ========================================================================
    
    def phase_9_final_scoring(self, risk_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 9: Final Scoring
        
        Steps:
        9.1: Weighted ensemble of all phase scores
        9.2: Calculate final confidence & probability
        9.3: Assign grade (A+ to F)
        
        Args:
            risk_signal: Signal from Phase 8
            
        Returns:
            Signal with final score and grade
        """
        try:
            symbol = risk_signal.get('symbol', 'Unknown')
            
            self.logger.debug(f"🏆 PHASE 9: Final Scoring for {symbol}...")
            
            final_signal = self.final_scoring.calculate_final_score(risk_signal)
            
            self.cycle_stats['signals_after_scoring'] += 1
            
            self.logger.debug(f"✅ Final scoring applied")
            self.logger.debug(f"   Final Score: {final_signal.get('final_score', 0):.1%}")
            self.logger.debug(f"   Grade: {final_signal.get('grade', 'N/A')}")
            self.logger.debug(f"   Probability: {final_signal.get('probability', 0):.1%}")
            
            return final_signal
            
        except Exception as e:
            self.logger.debug(f"❌ Final scoring error: {e}")
            self.daily_summary['errors'].append(f"FinalScore: {str(e)}")
            return risk_signal
    
    
    # ========================================================================
    # PHASE 10: SIGNAL VALIDATOR
    # ========================================================================
    
    def phase_10_signal_validator(self, final_signal: Dict[str, Any],
                                 data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        PHASE 10: Signal Validator
        
        Steps:
        10.1: Fakeout detection
        10.2: Liquidity trap detection
        10.3: Stop hunt detection
        10.4: Final checks (probability, R:R, sides, cooldown, age)
        
        Args:
            final_signal: Signal from Phase 9
            data: Market data
            
        Returns:
            (is_valid, validated_signal)
        """
        try:
            symbol = data['symbol']
            
            self.logger.debug(f"✔️  PHASE 10: Signal Validation for {symbol}...")
            
            is_valid, validated_signal = self.signal_validator.validate_signal(
                signal=final_signal,
                data=data,
                current_price=data['current_price']
            )
            
            if is_valid:
                self.cycle_stats['signals_after_validation'] += 1
                self.logger.debug(f"✅ Signal VALIDATED")
                self.logger.debug(f"   Status: READY FOR ENTRY")
                return True, validated_signal
            else:
                self.logger.debug(f"❌ Signal REJECTED")
                self.logger.debug(f"   Reason: {validated_signal.get('rejection_reason', 'Unknown')}")
                self.rejected_signals.append(validated_signal)
                return False, validated_signal
            
        except Exception as e:
            self.logger.debug(f"❌ Signal validation error: {e}")
            self.daily_summary['errors'].append(f"Validator: {str(e)}")
            return False, final_signal
    
    
    # ========================================================================
    # PHASE 11: TIMING PREDICTOR
    # ========================================================================
    
    def phase_11_timing_predictor(self, validated_signal: Dict[str, Any],
                                 data: Dict[str, Any]) -> Dict[str, Any]:
        """
        PHASE 11: Timing Predictor
        
        Steps:
        11.1: Calculate entry timing using ML ensemble
        11.2: Predict expected_minutes_to_entry
        11.3: Calculate confidence range
        
        Args:
            validated_signal: Signal from Phase 10
            data: Market data
            
        Returns:
            Signal with timing predictions
        """
        try:
            symbol = data['symbol']
            
            self.logger.debug(f"⏰ PHASE 11: Timing Prediction for {symbol}...")
            
            timing_signal = self.timing_predictor.predict_entry_timing(
                signal=validated_signal,
                data=data,
                current_price=data['current_price']
            )
            
            self.logger.debug(f"✅ Timing prediction applied")
            self.logger.debug(f"   Expected Entry: {timing_signal.get('expected_minutes_to_entry', 0):.1f} minutes")
            self.logger.debug(f"   Timing Confidence: {timing_signal.get('timing_confidence', 0):.1%}")
            
            return timing_signal
            
        except Exception as e:
            self.logger.debug(f"❌ Timing prediction error: {e}")
            self.daily_summary['errors'].append(f"Timing: {str(e)}")
            return validated_signal
    
    
    # ========================================================================
    # PHASE 12: SIGNAL OUTPUT & STORAGE
    # ========================================================================
    
    def phase_12_signal_output(self, timing_signal: Dict[str, Any]) -> bool:
        """
        PHASE 12: Signal Output & Storage
        
        Steps:
        12.1: Save final signal to JSON
        12.2: Update daily summary
        12.3: Register cooldown
        12.4: Log to summary
        
        Args:
            timing_signal: Signal from Phase 11
            
        Returns:
            True if saved successfully
        """
        try:
            symbol = timing_signal.get('symbol', 'Unknown')
            
            self.logger.debug(f"💾 PHASE 12: Saving signal for {symbol}...")
            
            # Add metadata
            timing_signal['saved_at'] = datetime.now().isoformat()
            timing_signal['status'] = 'FINAL'
            
            # Save to JSON
            signal_file = os.path.join(
                SIGNALS_FINAL_DIR,
                f"{self.session_id}.json"
            )
            
            self.signal_exporter.save_final_signal(timing_signal, signal_file)
            
            # Register cooldown
            direction = timing_signal.get('direction', 'HOLD')
            self.cooldown_manager.register_signal(
                symbol=symbol,
                direction=direction,
                duration_minutes=SYMBOL_COOLDOWN_MINUTES
            )
            
            # Update statistics
            self.cycle_stats['final_signals'] += 1
            self.final_signals.append(timing_signal)
            
            # Update daily summary
            direction_key = direction if direction in ['BUY', 'SELL'] else 'HOLD'
            self.daily_summary['by_direction'][direction_key] += 1
            self.daily_summary['final_signals'] += 1
            
            grade = timing_signal.get('grade', 'N/A')
            if grade not in self.daily_summary['by_grade']:
                self.daily_summary['by_grade'][grade] = 0
            self.daily_summary['by_grade'][grade] += 1
            
            if symbol not in self.daily_summary['by_symbol']:
                self.daily_summary['by_symbol'][symbol] = 0
            self.daily_summary['by_symbol'][symbol] += 1
            
            self.logger.debug(f"✅ Signal saved: {symbol} {direction} Grade {grade}")
            return True
            
        except Exception as e:
            self.logger.debug(f"❌ Signal export error: {e}")
            self.daily_summary['errors'].append(f"Output {symbol}: {str(e)}")
            return False
    
    
    # ========================================================================
    # PHASE 13: PERFORMANCE TRACKING
    # ========================================================================
    
    def phase_13_performance_tracking(self):
        """
        PHASE 13: Performance Tracking
        
        Steps:
        13.1: Load previous day signals
        13.2: Check SL/TP outcomes (AFTER signal time)
        13.3: Check if price entered entry zone
        13.4: Update expert weights based on performance
        13.5: Generate performance reports
        
        Returns:
            Performance summary dictionary
        """
        print_phase_header(13, "PERFORMANCE TRACKING & ANALYSIS")
        
        try:
            self.logger.info("📊 Loading previous day's signals...")
            
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            previous_signals_file = os.path.join(SIGNALS_FINAL_DIR, f"{yesterday}.json")
            
            if not os.path.exists(previous_signals_file):
                self.logger.info("ℹ️  No previous signals to analyze (first run)")
                return None
            
            # Load previous signals
            try:
                with open(previous_signals_file, 'r') as f:
                    previous_signals = json.load(f)
            except json.JSONDecodeError:
                # Handle file with multiple signals
                previous_signals = []
                with open(previous_signals_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            previous_signals.append(json.loads(line))
            
            if not isinstance(previous_signals, list):
                previous_signals = [previous_signals] if previous_signals else []
            
            self.logger.info(f"✅ Loaded {len(previous_signals)} signals from {yesterday}")
            
            # Analyze outcomes
            self.logger.info("\n📈 Analyzing trade outcomes...")
            
            performance = self.trade_analyzer.analyze_trade_outcomes(
                signals=previous_signals,
                symbol=None  # Analyze all symbols
            )
            
            # Log performance summary
            self.logger.info(f"📊 Performance Report for {yesterday}:")
            self.logger.info(f"   Total Signals: {performance.get('total_signals', 0)}")
            self.logger.info(f"   Wins: {performance.get('wins', 0)}")
            self.logger.info(f"   Losses: {performance.get('losses', 0)}")
            self.logger.info(f"   Win Rate: {performance.get('win_rate', 0):.1%}")
            self.logger.info(f"   Profit Factor: {performance.get('profit_factor', 0):.2f}")
            self.logger.info(f"   Avg R:R: {performance.get('avg_rr', 0):.2f}")
            self.logger.info(f"   Total PnL: {performance.get('total_pnl', 0):.2f}")
            
            # Log by expert performance
            if performance.get('by_expert'):
                self.logger.info(f"\n👥 Expert Performance:")
                for expert_name, expert_stats in performance['by_expert'].items():
                    self.logger.info(f"   {expert_name}:")
                    self.logger.info(f"      Win Rate: {expert_stats.get('win_rate', 0):.1%}")
                    self.logger.info(f"      Trades: {expert_stats.get('count', 0)}")
            
            # Save performance report
            report_file = os.path.join(PERFORMANCE_DAILY_DIR, f"{yesterday}.json")
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=2)
            
            self.logger.info(f"\n✅ Performance report saved: {report_file}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ Phase 13 error: {e}")
            self.logger.error(traceback.format_exc())
            self.daily_summary['errors'].append(f"Phase 13: {str(e)}")
            return None
    
    
    # ========================================================================
    # MAIN EXECUTION: PROCESS SINGLE SYMBOL
    # ========================================================================
    
    def run_single_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Run all 13 phases for a single symbol
        
        Returns:
            Final signal or None if rejected
        """
        try:
            self.logger.debug(f"\n{'='*80}")
            self.logger.debug(f"Processing: {symbol}")
            self.logger.debug(f"{'='*80}")
            
            # Phase 2: Data Fetching
            data = self.phase_2_data_fetching(symbol)
            if data is None:
                return None
            
            # Phase 3: Run 5 Experts
            expert_signals = self.phase_3_run_experts(data)
            if not expert_signals:
                return None
            
            self.cycle_stats['signals_generated'] += 1
            
            # Phase 4: Expert Aggregator
            combined_signal = self.phase_4_expert_aggregator(expert_signals, data)
            if combined_signal is None:
                return None
            
            # Phase 5: MTF Confirmation
            mtf_signal = self.phase_5_mtf_confirmation(combined_signal, data)
            
            # Phase 6: Smart Money Filter
            sm_signal = self.phase_6_smart_money_filter(mtf_signal, data)
            
            # Phase 7: Light Confirmations
            lc_signal = self.phase_7_light_confirmations(sm_signal, data)
            
            # Phase 8: Risk Management
            risk_signal = self.phase_8_risk_management(lc_signal, data)
            
            # Phase 9: Final Scoring
            final_signal = self.phase_9_final_scoring(risk_signal)
            
            # Phase 10: Signal Validator
            is_valid, validated_signal = self.phase_10_signal_validator(final_signal, data)
            
            if not is_valid:
                return None
            
            # Phase 11: Timing Predictor
            timing_signal = self.phase_11_timing_predictor(validated_signal, data)
            
            # Phase 12: Signal Output & Storage
            self.phase_12_signal_output(timing_signal)
            
            return timing_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            self.cycle_stats['errors'] += 1
            self.daily_summary['errors'].append(f"Symbol {symbol}: {str(e)}")
            return None
    
    
    # ========================================================================
    # MAIN EXECUTION: COMPLETE CYCLE
    # ========================================================================
    
    def run_cycle(self, max_workers: int = 5, symbols_override: List[str] = None):
        """
        Run complete trading system cycle
        
        Args:
            max_workers: Number of parallel threads for symbol processing
            symbols_override: Override coin selection (for testing)
        """
        cycle_start = time.time()
        
        try:
            print_banner("🚀 STARTING TRADING SYSTEM CYCLE")
            self.logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # ===== PHASE 1: INITIALIZATION & COIN SELECTION =====
            symbols = self.phase_1_initialization_and_coin_selection()
            
            # Override symbols if provided (for testing)
            if symbols_override:
                self.logger.info(f"⚠️  Using override symbols: {symbols_override}")
                symbols = symbols_override
            
            # Limit symbols in debug mode
            if DEBUG and len(symbols) > 20:
                self.logger.info(f"🔍 DEBUG MODE: Limiting to 20 symbols")
                symbols = symbols[:20]
            
            self.daily_summary['total_symbols_analyzed'] = len(symbols)
            
            # ===== PHASES 2-12: PROCESS SYMBOLS =====
            print_phase_header(2, "PROCESSING SYMBOLS (PHASES 2-12)")
            
            self.logger.info(f"\n📊 Processing {len(symbols)} symbols...\n")
            
            # Process symbols with parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_single_symbol, symbol): symbol
                    for symbol in symbols
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    symbol = futures[future]
                    
                    try:
                        result = future.result()
                        if result:
                            status = "✅ FINAL"
                        else:
                            status = "⏭️  SKIPPED"
                    except Exception as e:
                        status = f"❌ ERROR: {str(e)[:30]}"
                    
                    # Progress indicator
                    progress = f"[{completed}/{len(symbols)}]"
                    self.logger.info(f"{progress} {symbol:<15} {status}")
            
            self.cycle_stats['symbols_analyzed'] = len(symbols)
            
            # ===== SUMMARY =====
            print_phase_header(0, "PHASE SUMMARY")
            
            self.logger.info("📊 Signals Generated by Phase:")
            self.logger.info(f"   Phase 3 (Experts):        {self.cycle_stats['signals_generated']}")
            self.logger.info(f"   Phase 4 (Aggregator):     {self.cycle_stats['signals_from_phase_4']}")
            self.logger.info(f"   Phase 5 (MTF):            {self.cycle_stats['signals_after_mtf']}")
            self.logger.info(f"   Phase 6 (Smart Money):    {self.cycle_stats['signals_after_sm']}")
            self.logger.info(f"   Phase 7 (Light Confirm):  {self.cycle_stats['signals_after_lc']}")
            self.logger.info(f"   Phase 8 (Risk Mgmt):      {self.cycle_stats['signals_after_risk']}")
            self.logger.info(f"   Phase 9 (Scoring):        {self.cycle_stats['signals_after_scoring']}")
            self.logger.info(f"   Phase 10 (Validator):     {self.cycle_stats['signals_after_validation']}")
            self.logger.info(f"   Phase 12 (Final):         {self.cycle_stats['final_signals']}")
            
            # ===== PHASE 13: PERFORMANCE TRACKING =====
            self.phase_13_performance_tracking()
            
            # ===== FINAL SUMMARY =====
            print_banner("📊 FINAL SUMMARY REPORT")
            
            cycle_time = time.time() - cycle_start
            self.daily_summary['processing_time_seconds'] = cycle_time
            self.daily_summary['end_time'] = datetime.now().isoformat()
            self.daily_summary['total_signals_generated'] = self.cycle_stats['signals_generated']
            self.daily_summary['signals_by_phase'] = self.cycle_stats.copy()
            
            self.logger.info(f"Date: {self.daily_summary['date']}")
            self.logger.info(f"Session ID: {self.daily_summary['session_id']}")
            self.logger.info(f"\nSymbols & Signals:")
            self.logger.info(f"   Symbols Analyzed: {len(symbols)}")
            self.logger.info(f"   Signals Generated: {self.cycle_stats['signals_generated']}")
            self.logger.info(f"   Final Signals: {self.cycle_stats['final_signals']}")
            
            self.logger.info(f"\nDirection Breakdown:")
            self.logger.info(f"   BUY:  {self.daily_summary['by_direction']['BUY']}")
            self.logger.info(f"   SELL: {self.daily_summary['by_direction']['SELL']}")
            self.logger.info(f"   HOLD: {self.daily_summary['by_direction']['HOLD']}")
            
            self.logger.info(f"\nGrade Distribution:")
            for grade, count in sorted(self.daily_summary['by_grade'].items()):
                self.logger.info(f"   Grade {grade}: {count}")
            
            self.logger.info(f"\nTop Symbols:")
            top_symbols = sorted(
                self.daily_summary['by_symbol'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for symbol, count in top_symbols:
                self.logger.info(f"   {symbol}: {count} signals")
            
            self.logger.info(f"\nPerformance:")
            self.logger.info(f"   Processing Time: {format_time(cycle_time)}")
            self.logger.info(f"   Avg Time per Symbol: {(cycle_time/len(symbols)*1000):.0f}ms")
            
            if self.daily_summary['errors']:
                self.logger.info(f"\nErrors Encountered: {len(self.daily_summary['errors'])}")
                for i, error in enumerate(self.daily_summary['errors'][:5], 1):
                    self.logger.info(f"   {i}. {error[:60]}")
                if len(self.daily_summary['errors']) > 5:
                    self.logger.info(f"   ... and {len(self.daily_summary['errors']) - 5} more")
            
            # Save daily summary
            summary_file = os.path.join(
                SIGNALS_SUMMARY_DIR,
                f"{self.session_id}_summary.json"
            )
            with open(summary_file, 'w') as f:
                json.dump(self.daily_summary, f, indent=2)
            
            self.logger.info(f"\n✅ Daily summary saved: {summary_file}")
            
            print_banner("✅ CYCLE COMPLETE - SYSTEM READY")
            self.logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total Duration: {format_time(cycle_time)}\n")
            
        except Exception as e:
            self.logger.error(f"❌ Cycle failed: {e}")
            self.logger.error(traceback.format_exc())
            self.daily_summary['errors'].append(f"Cycle: {str(e)}")
            raise


# ============================================================================
# ENTRY POINT & MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for the trading system"""
    
    try:
        # Create system instance
        system = TradingSystemV3()
        
        # Run complete cycle
        system.run_cycle(
            max_workers=5,  # Parallel threads
            symbols_override=None  # Use None for dynamic selection
        )
        
        print("\n✅ Trading System V3 - Ready for deployment!\n")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏸️  System interrupted by user\n")
        return 130
        
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)