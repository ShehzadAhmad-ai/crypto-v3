"""
expert_price_action.py
COMPLETE EXPERT PRICE ACTION ENGINE V3.5
Main Orchestrator - Integrates All 8 Layers

This is a standalone expert trader that uses ONLY price action:
- Reads candle psychology (bodies, wicks, close position)
- Detects 35+ candlestick patterns with confidence scoring
- Identifies traps (bull traps, bear traps, stop hunts, liquidity sweeps)
- Analyzes sequences (micro-patterns, story building)
- Uses multi-timeframe context from candle structure
- Calculates structure-based stops and targets
- Outputs standardized signals matching Pattern System format

CAN RUN COMPLETELY ALONE - NO INDICATORS NEEDED
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all layers
from .price_action_config import MIN_TRADE_CONFIDENCE, MIN_RISK_REWARD
from .candle_analyzer import CandleAnalyzer, CandleData, get_candle_summary
from .pattern_detector import PatternDetector, DetectedPattern, PatternDirection
from .context_engine import ContextEngine, MarketContext, get_market_context
from .trap_engine import TrapEngine, DetectedTrap, TrapSeverity, get_best_trap
from .sequence_analyzer import SequenceAnalyzer, CandleStory, analyze_sequence
from .sl_tp_engine import SLTPEngine, TradeSetup, calculate_trade_setup
from .scoring_engine import ScoringEngine, ScoringResult, calculate_score
from .signal_formatter import SignalFormatter, FormattedSignal, format_signal, format_skip, SignalAction
from .price_action_config import *
class ExpertPriceAction:
    """
    Complete Expert Price Action Engine V3.5
    
    Integrates all layers:
    - Layer 1: Candle Analyzer
    - Layer 2: Pattern Detector (35+ patterns)
    - Layer 3: Context Engine (MTF, trend stage, session)
    - Layer 4: Trap Engine (bull/bear traps, stop hunts)
    - Layer 5: Sequence Analyzer (micro-patterns, story)
    - Layer 6: SL/TP Engine (structure-based stops)
    - Layer 7: Scoring Engine (dynamic weights)
    - Layer 8: Signal Formatter (standardized output)
    
    Features:
    - Standalone - no external dependencies
    - Pure price action - no indicators
    - Expert-level trap detection
    - Human-readable story building
    - Structure-based risk management
    - Standardized output matching Pattern System
    """
    
    def __init__(self):
        """Initialize all layers"""
        self.candle_analyzer = CandleAnalyzer()
        self.pattern_detector = PatternDetector()
        self.context_engine = ContextEngine()
        self.trap_engine = TrapEngine()
        self.sequence_analyzer = SequenceAnalyzer()
        self.sl_tp_engine = SLTPEngine()
        self.scoring_engine = ScoringEngine()
        self.signal_formatter = SignalFormatter()
        
        # Statistics
        self.signals_generated = 0
        self.signals_skipped = 0
        
    # =========================================================
    # MAIN ANALYSIS METHOD
    # =========================================================
    
    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str,
                htf_data: Optional[Dict[str, pd.DataFrame]] = None,
                regime_data: Optional[Dict] = None,
                structure_data: Optional[Dict] = None,
                sr_data: Optional[Dict] = None,
                liquidity_data: Optional[Dict] = None) -> FormattedSignal:
        """
        Complete price action analysis
        
        Args:
            df: Primary timeframe OHLCV DataFrame
            symbol: Trading pair symbol
            timeframe: Timeframe string (e.g., '5m', '1h')
            htf_data: Optional dict of higher timeframe DataFrames
            regime_data: Optional regime data from market_regime.py
            structure_data: Optional structure data from market_structure.py
            sr_data: Optional support/resistance data from price_location.py
            liquidity_data: Optional liquidity data from liquidity.py
        
        Returns:
            FormattedSignal ready for output
        """
        # ===== VALIDATION =====
        if df is None or df.empty or len(df) < 20:
            return self._skip_signal(symbol, timeframe, "Insufficient data")
        
        # ===== LAYER 1: CANDLE ANALYSIS =====
        candles = self.candle_analyzer.analyze_all_candles(df)
        if not candles:
            return self._skip_signal(symbol, timeframe, "Failed to analyze candles")
        
        latest_candle = candles[-1]
        current_price = latest_candle.close
        atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
        
        # ===== LAYER 2: PATTERN DETECTION =====
        patterns = self.pattern_detector.detect_all_patterns(df, regime_data, sr_data)
        
        if not patterns:
            return self._skip_signal(symbol, timeframe, "No patterns detected")
        
        # Get best pattern
        best_pattern = patterns[0]
        
        # ===== LAYER 3: CONTEXT ANALYSIS =====
        context = self.context_engine.analyze(df, htf_data, regime_data, structure_data, sr_data)
        
        # ===== LAYER 4: TRAP DETECTION =====
        traps = self.trap_engine.detect_all_traps(df, sr_data, structure_data, liquidity_data)
        best_trap = traps[0] if traps else None
        
        # ===== LAYER 5: SEQUENCE ANALYSIS =====
        sequence = self.sequence_analyzer.analyze_sequence(df)
        
        # ===== LAYER 6: SL/TP CALCULATION =====
        # Get pattern index
        pattern_index = best_pattern.candle_index if hasattr(best_pattern, 'candle_index') else -1
        
        trade_setup = self.sl_tp_engine.calculate_setup(
            direction=best_pattern.direction.value,
            current_price=current_price,
            atr=atr,
            pattern_name=best_pattern.name,
            candles=candles,
            pattern_index=pattern_index,
            structure_data=structure_data,
            sr_data=sr_data,
            htf_levels=htf_data
        )
        
        # ===== LAYER 7: SCORING =====
        # Prepare components for scoring
        components = self._prepare_components(
            best_pattern=best_pattern,
            context=context,
            best_trap=best_trap,
            sequence=sequence,
            trade_setup=trade_setup,
            sr_data=sr_data,
            liquidity_data=liquidity_data,
            regime_data=regime_data
        )
        
        scoring_result = self.scoring_engine.score_from_components(components)
        
        # ===== CHECK TRADEABILITY =====
        if not scoring_result.is_tradeable:
            return self._skip_signal(symbol, timeframe, 
                                     f"Low confidence: {scoring_result.confidence:.0%}")
        
        if trade_setup.risk_reward < MIN_RISK_REWARD:
            return self._skip_signal(symbol, timeframe, 
                                     f"Poor risk/reward: {trade_setup.risk_reward:.1f}")
        
        # ===== LAYER 8: SIGNAL FORMATTING =====
        # Determine if this is a trap signal
        is_trap = best_trap is not None and best_trap.severity != TrapSeverity.NONE
        
        signal = self.signal_formatter.format_signal(
            symbol=symbol,
            timeframe=timeframe,
            pattern_name=best_pattern.name,
            direction=best_pattern.direction.value,
            confidence=scoring_result.confidence,
            grade=scoring_result.grade.value,
            position_multiplier=scoring_result.position_multiplier * context.risk_multiplier,
            trade_setup={
                'entry': trade_setup.entry_price,
                'stop_loss': trade_setup.stop_loss,
                'take_profit': trade_setup.take_profit,
                'risk_reward': trade_setup.risk_reward
            },
            pattern_index=pattern_index,
            pattern_age=0,
            retest_level=context.key_level_price if context.at_key_level else None,
            retest_confirmed=context.at_key_level,
            trap_type=best_trap.type.value if best_trap else None,
            trap_severity=best_trap.severity.value if best_trap else None,
            mtf_alignment=context.mtf_alignment_score,
            at_key_level=context.at_key_level,
            sequence_type=sequence.sequence_type.value,
            additional_reasons=self._build_reasons(best_pattern, context, best_trap, sequence),
            is_trap=is_trap
        )
        
        # Update statistics
        self.signals_generated += 1
        
        return signal
    
    # =========================================================
    # HELPER METHODS
    # =========================================================
    
    def _prepare_components(self, best_pattern: DetectedPattern,
                            context: MarketContext,
                            best_trap: Optional[DetectedTrap],
                            sequence: CandleStory,
                            trade_setup: TradeSetup,
                            sr_data: Optional[Dict],
                            liquidity_data: Optional[Dict],
                            regime_data: Optional[Dict]) -> Dict[str, Any]:
        """
        Prepare component scores for scoring engine
        """
        # Pattern quality mapping
        quality_map = {'A': 'A', 'B': 'B', 'C': 'C'}
        pattern_quality = quality_map.get(best_pattern.pattern_quality, 'C')
        
        # Trap data
        trap_severity = best_trap.severity.value if best_trap else 'none'
        trap_score = best_trap.severity_score if best_trap else 0.0
        liquidity_sweep = best_trap.liquidity_sweep if best_trap else False
        sweep_count = best_trap.sweep_count if best_trap else 0
        
        # Volume data
        volume_ratio = 1.0
        volume_trend = 'STABLE'
        has_volume_divergence = False
        
        if sr_data:
            volume_ratio = sr_data.get('volume_ratio', 1.0)
            volume_trend = sr_data.get('volume_trend', 'STABLE')
            has_volume_divergence = sr_data.get('has_volume_divergence', False)
        
        # Liquidity data
        has_liquidity_sweep = liquidity_sweep or False
        cascade_risk = 0.0
        if liquidity_data:
            cascade_risk = liquidity_data.get('cascade_risk', 0.0)
        
        # Regime data
        regime_bias = 'NEUTRAL'
        regime_score = 0.0
        if regime_data:
            regime_bias = regime_data.get('bias', 'NEUTRAL')
            regime_score = regime_data.get('bias_score', 0.0)
        
        return {
            'pattern_quality': pattern_quality,
            'pattern_strength': best_pattern.strength,
            'pattern_alignment': 1,
            'total_patterns': 1,
            'trap_severity': trap_severity,
            'trap_score': trap_score,
            'liquidity_sweep': liquidity_sweep,
            'sequence_confidence': sequence.sequence_confidence,
            'sequence_type': sequence.sequence_type.value,
            'momentum_score': sequence.momentum_score,
            'mtf_alignment': context.mtf_alignment_score,
            'at_key_level': context.at_key_level,
            'key_level_strength': context.key_level_strength,
            'structure_bias': context.structure_bias,
            'structure_strength': context.structure_strength,
            'volatility_state': context.volatility_state,
            'session_weight': context.session_weight,
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'has_volume_divergence': has_volume_divergence,
            'has_liquidity_sweep': has_liquidity_sweep,
            'sweep_count': sweep_count,
            'cascade_risk': cascade_risk,
            'regime_bias': regime_bias,
            'regime_score': regime_score,
            'trend_stage': context.trend_stage
        }
    
    def _build_reasons(self, pattern: DetectedPattern,
                       context: MarketContext,
                       trap: Optional[DetectedTrap],
                       sequence: CandleStory) -> List[str]:
        """
        Build list of reasons for the signal
        """
        reasons = []
        
        # Pattern reason
        if pattern.reasons:
            reasons.extend(pattern.reasons[:2])
        
        # Context reasons
        if context.trend_stage != 'consolidation':
            reasons.append(f"Trend stage: {context.trend_stage}")
        
        if context.mtf_alignment_score >= 0.7:
            reasons.append(f"MTF alignment: {context.mtf_alignment_score:.0%}")
        
        if context.at_key_level:
            reasons.append(f"At {context.key_level_type} level")
        
        # Trap reasons
        if trap and trap.reasons:
            reasons.extend(trap.reasons[:2])
        
        # Sequence reasons
        if sequence.key_moments:
            reasons.append(sequence.key_moments[0])
        
        return reasons[:5]
    
    def _skip_signal(self, symbol: str, timeframe: str, reason: str) -> FormattedSignal:
        """Generate a SKIP signal"""
        self.signals_skipped += 1
        return self.signal_formatter.format_skip(symbol, timeframe, reason)
    
    # =========================================================
    # BATCH ANALYSIS
    # =========================================================
    
    def analyze_batch(self, dataframes: Dict[str, pd.DataFrame],
                      symbol: str, timeframe: str,
                      htf_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[FormattedSignal]:
        """
        Analyze multiple dataframes (e.g., for different coins)
        
        Args:
            dataframes: Dict of symbol -> DataFrame
            symbol: Base symbol
            timeframe: Timeframe
            htf_data: Optional higher timeframe data
        
        Returns:
            List of formatted signals
        """
        signals = []
        
        for sym, df in dataframes.items():
            if df is not None and not df.empty:
                signal = self.analyze(df, sym, timeframe, htf_data)
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    # =========================================================
    # STATISTICS
    # =========================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return {
            'signals_generated': self.signals_generated,
            'signals_skipped': self.signals_skipped,
            'total_analyzed': self.signals_generated + self.signals_skipped
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.signals_generated = 0
        self.signals_skipped = 0


# ==================== CONVENIENCE FUNCTIONS ====================

def analyze_price_action(df: pd.DataFrame, symbol: str, timeframe: str,
                         htf_data: Optional[Dict[str, pd.DataFrame]] = None,
                         regime_data: Optional[Dict] = None,
                         structure_data: Optional[Dict] = None,
                         sr_data: Optional[Dict] = None) -> FormattedSignal:
    """
    Convenience function to analyze price action
    
    Args:
        df: Primary timeframe OHLCV DataFrame
        symbol: Trading pair symbol
        timeframe: Timeframe string
        htf_data: Optional higher timeframe DataFrames
        regime_data: Optional regime data
        structure_data: Optional structure data
        sr_data: Optional support/resistance data
    
    Returns:
        FormattedSignal
    """
    engine = ExpertPriceAction()
    return engine.analyze(df, symbol, timeframe, htf_data, regime_data, structure_data, sr_data)


def get_best_setup(df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[FormattedSignal]:
    """
    Convenience function to get best setup only
    
    Args:
        df: OHLCV DataFrame
        symbol: Trading pair symbol
        timeframe: Timeframe string
    
    Returns:
        Best signal or None
    """
    signal = analyze_price_action(df, symbol, timeframe)
    if signal.action in [SignalAction.ENTER_NOW, SignalAction.STRONG_ENTRY, 
                          SignalAction.FLIP_TO_BUY, SignalAction.FLIP_TO_SELL]:
        return signal
    return None


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
    
    np.random.seed(42)
    base_price = 50000
    prices = [base_price]
    
    # Create a reversal pattern sequence
    for i in range(199):
        if i < 60:
            change = np.random.randn() * 30 - 10  # Downtrend
        elif i < 70:
            change = np.random.randn() * 10       # Indecision
        elif i < 100:
            change = np.random.randn() * 30 + 15  # Uptrend
        else:
            change = np.random.randn() * 20
        
        prices.append(prices[-1] + change)
    
    data = []
    for i, close in enumerate(prices):
        open_p = close - np.random.randn() * 20
        high = max(open_p, close) + abs(np.random.randn() * 30)
        low = min(open_p, close) - abs(np.random.randn() * 30)
        volume = abs(np.random.randn() * 10000) + 5000
        
        # Create a morning star pattern
        if 70 <= i <= 72:
            if i == 70:  # First bearish candle
                open_p = close + 50
                close = close - 30
                high = open_p + 20
                low = close - 10
            elif i == 71:  # Doji
                open_p = close - 5
                close = close + 5
                high = close + 15
                low = open_p - 15
                volume = 8000
            elif i == 72:  # Bullish candle
                open_p = close - 20
                close = close + 60
                high = close + 10
                low = open_p - 5
                volume = 15000
        
        data.append({
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Calculate ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Create higher timeframe data
    df_1h = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'atr': 'mean'
    }).dropna()
    
    htf_data = {'1h': df_1h}
    
    # Run analysis
    print("=" * 70)
    print("EXPERT PRICE ACTION ENGINE V3.5 - TEST RUN")
    print("=" * 70)
    
    engine = ExpertPriceAction()
    signal = engine.analyze(df, "BTC/USDT", "5m", htf_data)
    
    print(f"\nSymbol: {signal.symbol}")
    print(f"Timeframe: {signal.timeframe}")
    print(f"Pattern: {signal.pattern_name}")
    print(f"Direction: {signal.direction}")
    print(f"Action: {signal.action}")
    print(f"Action Detail: {signal.action_detail}")
    print(f"\nTrade Setup:")
    print(f"  Entry: {signal.entry}")
    print(f"  Stop Loss: {signal.stop_loss}")
    print(f"  Take Profit: {signal.take_profit}")
    print(f"  Risk/Reward: {signal.risk_reward}")
    print(f"\nConfidence: {signal.confidence:.1%}")
    print(f"Grade: {signal.grade}")
    print(f"Position Multiplier: {signal.position_multiplier:.1f}x")
    print(f"\nDecision Reason:")
    print(f"  {signal.decision_reason}")
    print(f"\nStage: {signal.stage}")
    print(f"Retest Level: {signal.retest_level}")
    print(f"Retest Confirmed: {signal.retest_confirmed}")
    
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(engine.get_stats())
    print("=" * 70)