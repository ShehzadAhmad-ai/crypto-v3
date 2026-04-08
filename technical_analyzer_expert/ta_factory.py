# ta_factory.py - Main Orchestrator for Technical Analyzer Expert
"""
Technical Analyzer Expert - Main Factory
Orchestrates all components to generate complete trading signals
- Step 1: Calculate all indicators
- Step 2: Detect market regime (decides direction)
- Step 3: Analyze indicators with regime settings
- Step 4: Detect divergences
- Step 5: Get HTF alignment (if HTF data provided)
- Step 6: Generate signal
- Step 7: Apply minimum filters
- Step 8: Calculate entry, stop loss, take profit
- Step 9: Score and grade
- Step 10: Format output matching Pattern Expert
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import configuration
from .ta_config import *

# Import core classes
from .ta_core import (
    TASignal, RegimeResult, HTFAnalysisResult, TradeSetup,
    CategoryScore, IndicatorSignal, DivergenceResult,
    create_skip_signal
)

# Import all components
from .market_regime_advanced import MarketRegimeAdvanced
from .htf_analyzer import HTFAnalyzer
from .indicator_analyzer import IndicatorAnalyzer
from .divergence_detector import DivergenceDetector
from .signal_generator import SignalGenerator
from .entry_sl_tp_engine import EntrySLTPEngine
from .scoring_engine import ScoringEngine

# Import existing technical analyzer for advanced indicators
try:
    from .technical_analyzer import TechnicalAnalyzer
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYZER_AVAILABLE = False

# Try to import logger
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class TechnicalAnalyzerFactory:
    """
    Main Orchestrator for Technical Analyzer Expert
    Brings together all components to generate trading signals
    """
    
    def __init__(self):
        """Initialize all components"""
        log.info("Initializing TechnicalAnalyzerFactory...")
        
        # Initialize all components
        self.regime_detector = MarketRegimeAdvanced()
        self.htf_analyzer = HTFAnalyzer()
        self.indicator_analyzer = IndicatorAnalyzer()
        self.divergence_detector = DivergenceDetector()
        self.signal_generator = SignalGenerator()
        self.entry_engine = EntrySLTPEngine()
        self.scoring_engine = ScoringEngine()
        
        # Initialize technical analyzer if available
        self.technical_analyzer = None
        if TECHNICAL_ANALYZER_AVAILABLE:
            self.technical_analyzer = TechnicalAnalyzer()
            log.info("TechnicalAnalyzer loaded for advanced indicators")
        
        # Cache for performance
        self.last_analysis_cache = {}
        
        log.info("TechnicalAnalyzerFactory initialized successfully")
    
    # ============================================================================
    # MAIN PUBLIC METHOD
    # ============================================================================
    
    def analyze(self, 
                df: pd.DataFrame,
                symbol: str,
                htf_data: Optional[Dict[str, pd.DataFrame]] = None,
                regime_override: Optional[Dict[str, Any]] = None) -> TASignal:
        """
        Complete technical analysis pipeline
        
        Args:
            df: Main timeframe OHLCV data (5min default)
            symbol: Trading symbol (e.g., "BTCUSDT")
            htf_data: Dictionary of timeframe -> DataFrame (optional)
            regime_override: Override regime detection (for testing)
        
        Returns:
            TASignal object with complete analysis
        """
        start_time = datetime.now()
        
        try:
            # ====================================================================
            # STEP 1: VALIDATE INPUT DATA
            # ====================================================================
            if df is None or df.empty:
                log.error(f"No data provided for {symbol}")
                return create_skip_signal(symbol, "No data provided")
            
            if len(df) < MIN_DATA_CANDLES:
                log.warning(f"Insufficient data for {symbol}: {len(df)} < {MIN_DATA_CANDLES}")
                return create_skip_signal(symbol, f"Insufficient data: {len(df)} candles")
            
            log.info(f"Starting technical analysis for {symbol} with {len(df)} candles")
            
            # ====================================================================
            # STEP 2: DETECT MARKET REGIME (DECIDES DIRECTION)
            # ====================================================================
            if regime_override:
                # Use override for testing
                regime = RegimeResult(
                    regime=regime_override.get('regime', 'UNKNOWN'),
                    regime_type=regime_override.get('regime_type', 'UNKNOWN'),
                    confidence=regime_override.get('confidence', 0.5),
                    bias=regime_override.get('bias', 'NEUTRAL'),
                    bias_score=regime_override.get('bias_score', 0.0),
                    trend_strength=regime_override.get('trend_strength', 0.5),
                    volatility_state=regime_override.get('volatility_state', 'NORMAL'),
                    liquidity_state=regime_override.get('liquidity_state', 'NORMAL'),
                    adx=regime_override.get('adx', 20.0),
                    atr_pct=regime_override.get('atr_pct', 0.02),
                    slope=regime_override.get('slope', 0.0),
                    is_squeeze=regime_override.get('is_squeeze', False),
                    squeeze_intensity=regime_override.get('squeeze_intensity', 0.0),
                    wyckoff_phase=regime_override.get('wyckoff_phase', 'UNKNOWN'),
                    wyckoff_confidence=regime_override.get('wyckoff_confidence', 0.0),
                    recommended_strategies=regime_override.get('recommended_strategies', []),
                    avoid_strategies=regime_override.get('avoid_strategies', []),
                    indicator_settings=regime_override.get('indicator_settings', {}),
                    summary=regime_override.get('summary', 'Regime override'),
                    details=regime_override.get('details', {})
                )
            else:
                # Detect regime from data
                regime = self.regime_detector.detect_regime(df, symbol)
            
            log.info(f"Regime: {regime.regime} | Bias: {regime.bias} ({regime.bias_score:.2f}) | Trend Strength: {regime.trend_strength:.2f}")
            
            # ====================================================================
            # STEP 3: GET DYNAMIC INDICATOR SETTINGS
            # ====================================================================
            indicator_settings = regime.indicator_settings
            
            # ====================================================================
            # STEP 4: CALCULATE AND ANALYZE ALL INDICATORS
            # ====================================================================
            # First, calculate all indicators using TechnicalAnalyzer if available
            if self.technical_analyzer:
                try:
                    # Apply regime settings
                    self.technical_analyzer.set_regime_settings(indicator_settings)
                    # Calculate indicators
                    df_indicators = self.technical_analyzer.calculate_all_indicators(df, symbol)
                    # Use for analysis
                    analysis_df = df_indicators
                except Exception as e:
                    log.warning(f"TechnicalAnalyzer failed: {e}, using fallback")
                    analysis_df = df
            else:
                analysis_df = df
            
            # Analyze indicators with regime settings
            indicator_signals, category_scores = self.indicator_analyzer.analyze_indicators(
                analysis_df, indicator_settings, symbol
            )
            
            log.debug(f"Analyzed {len(indicator_signals)} indicators across {len(category_scores)} categories")
            
            # ====================================================================
            # STEP 5: DETECT DIVERGENCES
            # ====================================================================
            divergences, divergence_score, has_bullish_div, has_bearish_div = \
                self.divergence_detector.detect_all_divergences(analysis_df)
            
            log.debug(f"Detected {len(divergences)} divergences (Bullish: {has_bullish_div}, Bearish: {has_bearish_div})")
            
            # ====================================================================
            # STEP 6: HTF ANALYSIS (if data provided)
            # ====================================================================
            htf_result = None
            htf_alignment_score = 0.5  # Neutral default
            htf_boost = 0.0
            
            if htf_data and len(htf_data) > 0:
                htf_result = self.htf_analyzer.analyze_htfs(df, htf_data, MAIN_TIMEFRAME, symbol)
                htf_alignment_score = htf_result.overall_alignment
                htf_boost = htf_result.alignment_boost
                log.debug(f"HTF Analysis: {htf_result.bullish_htf_count}B / {htf_result.bearish_htf_count}B | Alignment: {htf_alignment_score:.2f}")
            
            # ====================================================================
            # STEP 7: CALCULATE AGREEMENT AND VOLUME
            # ====================================================================
            agreement_score = self.scoring_engine.calculate_category_agreement(category_scores)
            
            # Get current volume ratio
            volume_ratio = 1.0
            if 'volume_ratio' in analysis_df.columns and not pd.isna(analysis_df['volume_ratio'].iloc[-1]):
                volume_ratio = float(analysis_df['volume_ratio'].iloc[-1])
            
            log.debug(f"Agreement: {agreement_score:.2f} | Volume Ratio: {volume_ratio:.2f}x")
            
            # ====================================================================
            # STEP 8: GENERATE SIGNAL
            # ====================================================================
            signal = self.signal_generator.generate_signal(
                symbol=symbol,
                category_scores=category_scores,
                divergences=divergences,
                htf_alignment_score=htf_alignment_score,
                volume_ratio=volume_ratio,
                trend_strength=regime.trend_strength,
                regime_bias=regime.bias_score,
                indicator_count=len(indicator_signals)
            )
            
            # If signal is skip, return early
            if signal.action == "SKIP":
                log.info(f"Signal skipped for {symbol}: {signal.decision_reason}")
                return signal
            
            log.info(f"Signal generated: {signal.direction} with confidence {signal.confidence:.2%}")
            
            # ====================================================================
            # STEP 9: CALCULATE TRADE SETUP
            # ====================================================================
            current_price = float(analysis_df['close'].iloc[-1])
            
            trade_setup = self.entry_engine.calculate_trade_setup(
                df=analysis_df,
                direction=signal.direction,
                confidence=signal.confidence,
                regime=regime,
                current_price=current_price
            )
            
            # Update signal with trade setup
            if trade_setup.is_valid:
                signal.entry = trade_setup.entry
                signal.stop_loss = trade_setup.stop_loss
                signal.take_profit = trade_setup.take_profit
                signal.risk_reward = trade_setup.risk_reward
                log.debug(f"Trade setup: Entry={signal.entry:.4f}, SL={signal.stop_loss:.4f}, TP={signal.take_profit:.4f}, RR={signal.risk_reward:.2f}")
            else:
                log.warning(f"Invalid trade setup: {trade_setup.validation_errors}")
                return create_skip_signal(symbol, f"Invalid trade setup: {', '.join(trade_setup.validation_errors[:2])}")
            
            # ====================================================================
            # STEP 10: SCORE AND GRADE THE SIGNAL
            # ====================================================================
            scoring_result = self.scoring_engine.score_signal(
                category_scores=category_scores,
                signal_confidence=signal.confidence,
                agreement_score=agreement_score,
                volume_ratio=volume_ratio,
                trend_strength=regime.trend_strength,
                regime=regime.regime_type,
                htf_boost=htf_boost,
                divergence_bias=self.divergence_detector.get_divergence_bias(divergences),
                risk_reward=signal.risk_reward if signal.risk_reward else 0,
                min_filters_passed=True
            )
            
            # Update signal with scoring results
            signal.grade = scoring_result['grade']
            signal.position_multiplier = scoring_result['position_multiplier']
            signal.action = scoring_result['action']
            
            # If scoring says skip, update reason
            if signal.action == "SKIP":
                signal.decision_reason = scoring_result['action_reason']
                log.info(f"Signal scored as SKIP: {signal.decision_reason}")
                return signal
            
            # ====================================================================
            # STEP 11: FINAL VALIDATION
            # ====================================================================
            # Check risk/reward again with scoring
            if signal.risk_reward and signal.risk_reward < MIN_RISK_REWARD:
                return create_skip_signal(symbol, f"Risk/Reward too low: {signal.risk_reward:.2f} < {MIN_RISK_REWARD:.1f}")
            
            # Check position multiplier
            if signal.position_multiplier <= 0:
                return create_skip_signal(symbol, f"Position multiplier zero: grade {signal.grade}")
            
            # ====================================================================
            # STEP 12: ENRICH SIGNAL WITH METADATA
            # ====================================================================
            signal.raw_bullish_score = scoring_result['bullish_score']
            signal.raw_bearish_score = scoring_result['bearish_score']
            signal.agreement_score = agreement_score
            signal.divergence_count = len(divergences)
            signal.indicator_count = len(indicator_signals)
            signal.htf_aligned = htf_alignment_score > 0.6
            signal.regime = regime.regime
            
            # Generate final decision reason if not already set
            if not signal.decision_reason or signal.decision_reason == "":
                signal.decision_reason = self._generate_final_reason(
                    signal, regime, trade_setup, divergences, htf_alignment_score
                )
            
            # Generate signal ID
            signal.signal_id = signal.generate_signal_id()
            signal.timestamp = datetime.now().isoformat()
            
            # ====================================================================
            # STEP 13: LOG AND RETURN
            # ====================================================================
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            log.info(f"Analysis complete for {symbol}: {signal.direction} | Grade: {signal.grade} | "
                    f"Confidence: {signal.confidence:.1%} | Action: {signal.action} | "
                    f"RR: {signal.risk_reward:.2f} | Time: {elapsed_ms:.0f}ms")
            
            return signal
            
        except Exception as e:
            log.error(f"Error in technical analysis for {symbol}: {e}", exc_info=True)
            return create_skip_signal(symbol, f"Analysis error: {str(e)[:100]}")
    
    # ============================================================================
    # BATCH ANALYSIS METHODS
    # ============================================================================
    
    def analyze_batch(self, 
                      data_dict: Dict[str, pd.DataFrame],
                      htf_data_dict: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None) -> Dict[str, TASignal]:
        """
        Analyze multiple symbols in batch
        
        Args:
            data_dict: Dictionary of symbol -> main timeframe DataFrame
            htf_data_dict: Optional dictionary of symbol -> HTF data dict
        
        Returns:
            Dictionary of symbol -> TASignal
        """
        results = {}
        
        for symbol, df in data_dict.items():
            try:
                htf_data = htf_data_dict.get(symbol) if htf_data_dict else None
                signal = self.analyze(df, symbol, htf_data)
                results[symbol] = signal
            except Exception as e:
                log.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = create_skip_signal(symbol, f"Batch error: {e}")
        
        return results
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _generate_final_reason(self, signal: TASignal, regime: RegimeResult,
                                trade_setup: TradeSetup, divergences: List[DivergenceResult],
                                htf_alignment: float) -> str:
        """Generate comprehensive final decision reason"""
        parts = []
        
        # Direction and grade
        parts.append(f"{signal.direction} signal - Grade {signal.grade}")
        
        # Confidence
        if signal.confidence >= 0.85:
            parts.append("very high confidence")
        elif signal.confidence >= 0.75:
            parts.append("high confidence")
        elif signal.confidence >= 0.65:
            parts.append("good confidence")
        
        # Regime context
        if "TREND" in regime.regime:
            parts.append(f"{regime.bias.lower()} trend regime")
        elif "RANGING" in regime.regime:
            parts.append("ranging market")
        
        # Divergences
        if divergences:
            bullish_divs = [d for d in divergences if d.type == "BULLISH"]
            bearish_divs = [d for d in divergences if d.type == "BEARISH"]
            
            if signal.direction == "BUY" and bullish_divs:
                parts.append(f"{len(bullish_divs)} bullish divergence(s)")
            elif signal.direction == "SELL" and bearish_divs:
                parts.append(f"{len(bearish_divs)} bearish divergence(s)")
        
        # HTF alignment
        if htf_alignment > 0.7:
            parts.append("HTF aligned")
        
        # Risk/Reward
        if trade_setup.risk_reward:
            parts.append(f"RR {trade_setup.risk_reward:.1f}")
        
        return " | ".join(parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        return {
            "technical_analyzer_available": TECHNICAL_ANALYZER_AVAILABLE,
            "regime_detector": "initialized",
            "htf_analyzer": "initialized",
            "indicator_analyzer": "initialized",
            "divergence_detector": "initialized",
            "signal_generator": "initialized",
            "entry_engine": "initialized",
            "scoring_engine": "initialized",
            "cache_size": len(self.last_analysis_cache)
        }
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.last_analysis_cache.clear()
        log.info("Analysis cache cleared")


# ============================================================================
# SIMPLE WRAPPER FUNCTION
# ============================================================================

def analyze_technical(symbol: str, 
                      df: pd.DataFrame,
                      htf_data: Optional[Dict[str, pd.DataFrame]] = None) -> TASignal:
    """
    Simple wrapper function for quick technical analysis
    
    Args:
        symbol: Trading symbol
        df: Main timeframe OHLCV data
        htf_data: Optional HTF data dictionary
    
    Returns:
        TASignal object
    """
    factory = TechnicalAnalyzerFactory()
    return factory.analyze(df, symbol, htf_data)


# ============================================================================
# COMMAND LINE INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    print("Technical Analyzer Expert - Test Mode")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=500, freq='5min')
    np.random.seed(42)
    
    # Generate sample price data
    price = 50000
    prices = []
    for i in range(500):
        price = price * (1 + np.random.randn() * 0.002)
        prices.append(price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.001)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.001)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100, 1000, 500)
    }, index=dates)
    
    # Run analysis
    factory = TechnicalAnalyzerFactory()
    signal = factory.analyze(df, "TESTUSDT")
    
    # Print results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Signal ID: {signal.signal_id}")
    print(f"Direction: {signal.direction}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Grade: {signal.grade}")
    print(f"Position Multiplier: {signal.position_multiplier}")
    print(f"Action: {signal.action}")
    print(f"Entry: {signal.entry}")
    print(f"Stop Loss: {signal.stop_loss}")
    print(f"Take Profit: {signal.take_profit}")
    print(f"Risk/Reward: {signal.risk_reward}")
    print(f"Decision Reason: {signal.decision_reason}")
    print(f"Timestamp: {signal.timestamp}")
    
    if signal.raw_bullish_score:
        print(f"\nRaw Scores:")
        print(f"  Bullish: {signal.raw_bullish_score:.3f}")
        print(f"  Bearish: {signal.raw_bearish_score:.3f}")
        print(f"  Agreement: {signal.agreement_score:.2%}")
        print(f"  Divergences: {signal.divergence_count}")
        print(f"  Indicators: {signal.indicator_count}")