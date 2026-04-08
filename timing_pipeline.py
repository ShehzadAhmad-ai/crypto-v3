# timing_pipeline.py - SIMPLIFIED VERSION (Entry Only)
"""
Timing Prediction Pipeline (Phase 7) - Simplified
Predicts expected time to entry only (no TP timing)
"""
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from config import Config
from logger import log
from signal_model import Signal, SignalStatus
from entry_timing_predictor import EntryTimingPredictor, TimingPrediction


class TimingPipeline:
    """
    Timing Prediction Pipeline - Simplified
    - Predicts candles/minutes until entry zone is reached
    - Updates signal with entry timing only
    """
    
    def __init__(self):
        self.predictor = EntryTimingPredictor()
        log.info("Timing Pipeline initialized (Simplified - Entry Only)")
    
    def process_signal(self, signal: Signal, df: pd.DataFrame) -> Optional[Signal]:
        """
        Process a signal through timing prediction
        
        Args:
            signal: Signal that passed Phase 9 (Validation)
            df: OHLCV DataFrame
        
        Returns:
            Updated signal with entry timing prediction
        """
        # Only run on signals that passed Validation (FINAL)
        if signal.status != SignalStatus.FINAL:
            return None
        
        try:
            # Convert signal to dict for predictor
            signal_dict = {
                'direction': signal.direction,
                'entry_zone_low': signal.entry_zone_low,
                'entry_zone_high': signal.entry_zone_high,
                'take_profit': None  # No TP timing
            }
            
            # Get entry timing prediction only
            prediction = self.predictor.predict_entry_timing(df, signal_dict)
            
            # Update signal with entry timing only
            signal.expected_candles_to_entry = prediction.expected_candles_to_entry
            signal.expected_minutes_to_entry = prediction.expected_minutes_to_entry
            
            # Store TP timing as 0 (not used)
            signal.expected_minutes_to_tp = 0
            
            # Add timing reasons to metadata
            if not hasattr(signal, 'metadata'):
                signal.metadata = {}
            
            signal.metadata['entry_timing'] = {
                'confidence': prediction.entry_confidence,
                'confidence_level': prediction.entry_confidence_level.value,
                'range_low': prediction.entry_time_range_low,
                'range_high': prediction.entry_time_range_high,
                'estimated_time': prediction.estimated_entry_time.isoformat() if prediction.estimated_entry_time else None,
                'reasons': prediction.reasons
            }
            
            # Add to confirmation reasons
            if prediction.reasons:
                signal.confirmation_reasons.extend(prediction.reasons[:2])
            
            # Terminal output
            self._print_timing_output(signal, prediction)
            
            return signal
            
        except Exception as e:
            log.error(f"Error in timing pipeline for {signal.symbol}: {e}")
            return signal  # Return original signal on error
    
    def _print_timing_output(self, signal: Signal, prediction: TimingPrediction):
        """Print formatted timing output"""
        log.info("\n" + "=" * 60)
        log.info(f"[TIMING] {signal.symbol} {signal.timeframe}")
        log.info("-" * 40)
        log.info(f"ENTRY TIMING:")
        log.info(f"  Expected: {prediction.expected_candles_to_entry} candles | {prediction.expected_minutes_to_entry} minutes")
        log.info(f"  Confidence: {prediction.entry_confidence:.1%} ({prediction.entry_confidence_level.value})")
        log.info(f"  Estimated: {prediction.estimated_entry_time.strftime('%H:%M:%S') if prediction.estimated_entry_time else 'N/A'}")
        log.info(f"  Window: {prediction.entry_time_range_low}-{prediction.entry_time_range_high} minutes")
        log.info("-" * 40)
        log.info(f"Market Conditions:")
        log.info(f"  Avg Candle Speed: {prediction.avg_candle_speed}% of price")
        log.info(f"  Volatility: {prediction.volatility_regime}")
        log.info(f"  Congestion: {prediction.congestion_level:.1%}")
        log.info("-" * 40)
        if prediction.reasons:
            log.info("Reasons:")
            for reason in prediction.reasons[:3]:
                log.info(f"  • {reason}")
        log.info("=" * 60)