# entry_timing_predictor.py - SIMPLIFIED VERSION (Entry Only)
"""
Entry Timing Prediction Engine - Simplified
Predicts expected time to reach entry zone
- Multi-model ensemble prediction
- Market regime-based adjustments
- Volatility-weighted predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from collections import deque

EPSILON = 1e-10


class TimingConfidence(Enum):
    VERY_HIGH = "VERY_HIGH"  # >80% confidence
    HIGH = "HIGH"            # 60-80% confidence
    MODERATE = "MODERATE"    # 40-60% confidence
    LOW = "LOW"              # 20-40% confidence
    VERY_LOW = "VERY_LOW"    # <20% confidence


@dataclass
class TimingPrediction:
    """Simplified timing prediction result"""
    # Entry timing only
    expected_candles_to_entry: int
    expected_minutes_to_entry: int
    entry_confidence: float
    entry_confidence_level: TimingConfidence
    entry_time_range_low: int
    entry_time_range_high: int
    
    # Context metrics
    avg_candle_speed: float
    volatility_regime: str
    congestion_level: float
    
    # Time estimates
    estimated_entry_time: Optional[datetime] = None
    entry_window_start: Optional[datetime] = None
    entry_window_end: Optional[datetime] = None
    
    # Reasons
    reasons: List[str] = None


class EntryTimingPredictor:
    """
    Simplified entry timing predictor
    Only predicts time to reach entry zone
    """
    
    def __init__(self):
        self.prediction_history = deque(maxlen=1000)
        
        # Parameters
        self.atr_lookback = 20
        self.velocity_lookback = 10
        self.pattern_lookback = 50
        self.volume_lookback = 20
        
        # Cache
        self._cached_minutes_per_candle = 5
        
        logging.info("EntryTimingPredictor (Simplified) initialized")
    
    def predict_entry_timing(self, df: pd.DataFrame, signal: Dict) -> TimingPrediction:
        """
        Predict time until price reaches entry zone
        
        Args:
            df: OHLCV DataFrame
            signal: Signal dict with direction and entry zone
        
        Returns:
            TimingPrediction with entry timing only
        """
        try:
            current_price = float(df['close'].iloc[-1])
            direction = signal.get('direction', 'BUY')
            entry_zone_low = signal.get('entry_zone_low', current_price * 0.99)
            entry_zone_high = signal.get('entry_zone_high', current_price * 1.01)
            
            # ===== 1. CORE METRICS =====
            atr = self._calculate_atr(df)
            atr_pct = atr / current_price if current_price > 0 else 0.02
            
            velocity = self._calculate_velocity(df)
            vol_regime = self._detect_volatility_regime(df)
            congestion = self._calculate_congestion(df)
            volume_profile = self._calculate_volume_profile(df)
            
            # ===== 2. DISTANCE TO ENTRY ZONE =====
            entry_distance = self._calculate_distance(
                current_price, direction, entry_zone_low, entry_zone_high
            )
            
            # Already in zone
            if entry_distance <= 0:
                return TimingPrediction(
                    expected_candles_to_entry=0,
                    expected_minutes_to_entry=0,
                    entry_confidence=0.95,
                    entry_confidence_level=TimingConfidence.VERY_HIGH,
                    entry_time_range_low=0,
                    entry_time_range_high=0,
                    avg_candle_speed=round(atr_pct * 100, 2),
                    volatility_regime=vol_regime,
                    congestion_level=round(congestion, 3),
                    estimated_entry_time=datetime.now(),
                    entry_window_start=datetime.now(),
                    entry_window_end=datetime.now(),
                    reasons=["Price already in entry zone"]
                )
            
            # ===== 3. MULTIPLE PREDICTION MODELS =====
            # Model 1: ATR-based
            candles_model1 = self._atr_based(entry_distance, atr, df, vol_regime)
            
            # Model 2: Velocity-based
            candles_model2 = self._velocity_based(entry_distance, velocity)
            
            # Model 3: Regime-based
            candles_model3 = self._regime_based(entry_distance, vol_regime, congestion)
            
            # Model 4: Volume-weighted
            candles_model4 = self._volume_weighted(entry_distance, volume_profile, vol_regime)
            
            # ===== 4. ENSEMBLE =====
            weights = self._get_weights(vol_regime, congestion)
            
            models = [
                (candles_model1, weights['atr']),
                (candles_model2, weights['velocity']),
                (candles_model3, weights['regime']),
                (candles_model4, weights['volume'])
            ]
            
            entry_candles, confidence_range = self._ensemble(models)
            entry_candles = max(1, int(round(entry_candles)))
            
            # ===== 5. CONFIDENCE & RANGES =====
            range_low, range_high = self._calculate_range(entry_candles, confidence_range, df)
            entry_confidence = self._calculate_confidence(
                entry_candles, velocity, congestion, vol_regime, volume_profile
            )
            
            entry_minutes = self._candles_to_minutes(entry_candles, df)
            now = datetime.now()
            
            reasons = self._build_reasons(
                entry_candles, velocity, vol_regime, congestion,
                entry_confidence, range_low, range_high
            )
            
            return TimingPrediction(
                expected_candles_to_entry=entry_candles,
                expected_minutes_to_entry=entry_minutes,
                entry_confidence=round(entry_confidence, 3),
                entry_confidence_level=self._get_confidence_level(entry_confidence),
                entry_time_range_low=range_low,
                entry_time_range_high=range_high,
                avg_candle_speed=round(atr_pct * 100, 2),
                volatility_regime=vol_regime,
                congestion_level=round(congestion, 3),
                estimated_entry_time=now + timedelta(minutes=entry_minutes),
                entry_window_start=now + timedelta(minutes=range_low),
                entry_window_end=now + timedelta(minutes=range_high),
                reasons=reasons
            )
            
        except Exception as e:
            logging.error(f"Error in predict_entry_timing: {e}")
            return self._fallback_prediction(df, signal)
    
    # ==================== PRIVATE METHODS ====================
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate current ATR"""
        try:
            if 'atr' in df.columns:
                return float(df['atr'].iloc[-1])
            return float(df['close'].iloc[-1]) * 0.02
        except:
            return float(df['close'].iloc[-1]) * 0.02
    
    def _calculate_velocity(self, df: pd.DataFrame, window: int = 10) -> float:
        """Calculate price velocity (% per candle)"""
        try:
            if len(df) < window + 1:
                return 0.01
            
            closes = df['close'].values[-window-1:]
            changes = np.diff(closes) / (closes[:-1] + EPSILON)
            weights = np.exp(np.linspace(0, 1, len(changes)))
            weights /= weights.sum()
            velocity = float(np.average(np.abs(changes), weights=weights))
            return max(0.001, min(0.2, velocity))
        except:
            return 0.01
    
    def _detect_volatility_regime(self, df: pd.DataFrame) -> str:
        """Detect volatility regime"""
        try:
            if len(df) >= 14:
                atr = self._calculate_atr(df)
                atr_pct = atr / df['close'].iloc[-1]
                if atr_pct > 0.04:
                    return 'VERY_HIGH'
                elif atr_pct > 0.025:
                    return 'HIGH'
                elif atr_pct > 0.015:
                    return 'NORMAL'
                elif atr_pct > 0.008:
                    return 'LOW'
                else:
                    return 'VERY_LOW'
            return 'NORMAL'
        except:
            return 'NORMAL'
    
    def _calculate_congestion(self, df: pd.DataFrame) -> float:
        """Calculate market congestion (0-1)"""
        try:
            if len(df) < 20:
                return 0.5
            
            closes = df['close'].values[-30:]
            total_move = abs(closes[-1] - closes[0])
            total_path = sum(abs(np.diff(closes)))
            
            if total_path > EPSILON:
                efficiency = total_move / total_path
                return 1 - min(1, efficiency)
            return 0.5
        except:
            return 0.5
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> float:
        """Analyze volume profile"""
        try:
            if len(df) < 20:
                return 0.5
            recent_vol = df['volume'].iloc[-5:].mean()
            avg_vol = df['volume'].iloc[-20:].mean()
            return min(1.0, recent_vol / avg_vol)
        except:
            return 0.5
    
    def _calculate_distance(self, price: float, direction: str,
                           zone_low: float, zone_high: float) -> float:
        """Calculate distance to entry zone"""
        if zone_low <= price <= zone_high:
            return 0.0
        if direction == 'BUY':
            return max(0, price - zone_high)
        else:
            return max(0, zone_low - price)
    
    def _atr_based(self, distance: float, atr: float, df: pd.DataFrame, regime: str) -> float:
        """ATR-based prediction"""
        if distance <= 0 or atr <= 0:
            return 0
        
        # Recent candle movement
        if len(df) >= 20:
            moves = []
            for i in range(1, 20):
                if len(df) > i:
                    move = abs(df['close'].iloc[-i] - df['close'].iloc[-i-1])
                    if move > 0:
                        moves.append(move / atr)
            if moves:
                avg_move = np.percentile(moves, 60)
                if avg_move > 0:
                    return max(1, (distance / atr) / avg_move)
        
        # Regime-based fallback
        atr_per_candle = {
            'VERY_LOW': 0.2, 'LOW': 0.35, 'NORMAL': 0.5,
            'HIGH': 0.8, 'VERY_HIGH': 1.2
        }.get(regime, 0.5)
        return max(1, (distance / atr) / atr_per_candle)
    
    def _velocity_based(self, distance: float, velocity: float) -> float:
        """Velocity-based prediction"""
        if distance <= 0 or velocity <= EPSILON:
            return 30
        distance_pct = distance / 100
        return max(1, distance_pct / velocity)
    
    def _regime_based(self, distance: float, regime: str, congestion: float) -> float:
        """Regime-based prediction"""
        base_velocity = {
            'VERY_LOW': 0.001, 'LOW': 0.002, 'NORMAL': 0.004,
            'HIGH': 0.008, 'VERY_HIGH': 0.015
        }.get(regime, 0.004)
        
        congestion_factor = 1.0 + (congestion * 0.5)
        distance_pct = distance / 100
        return max(1, distance_pct / base_velocity * congestion_factor)
    
    def _volume_weighted(self, distance: float, volume_profile: float, regime: str) -> float:
        """Volume-weighted prediction"""
        base = self._regime_based(distance, regime, 0.5)
        if volume_profile > 0.7:
            base *= 0.8
        elif volume_profile < 0.3:
            base *= 1.3
        return max(1, base)
    
    def _get_weights(self, regime: str, congestion: float) -> Dict[str, float]:
        """Dynamic weights"""
        weights = {'atr': 0.30, 'velocity': 0.30, 'regime': 0.20, 'volume': 0.20}
        
        if regime in ['HIGH', 'VERY_HIGH']:
            weights['velocity'] += 0.1
            weights['atr'] -= 0.05
        elif regime in ['LOW', 'VERY_LOW']:
            weights['regime'] += 0.1
            weights['velocity'] -= 0.05
        
        if congestion > 0.7:
            weights['regime'] += 0.1
            weights['velocity'] -= 0.1
        
        # Normalize
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
        return weights
    
    def _ensemble(self, models: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Ensemble prediction"""
        valid = [(p, w) for p, w in models if p != float('inf') and p > 0 and w > 0]
        if not valid:
            return 10, 5
        
        weights = np.array([w for _, w in valid])
        weights /= weights.sum()
        preds = np.array([p for p, _ in valid])
        
        weighted_avg = np.average(preds, weights=weights)
        variance = np.average((preds - weighted_avg) ** 2, weights=weights)
        std_dev = np.sqrt(variance)
        
        return weighted_avg, std_dev
    
    def _calculate_range(self, prediction: float, std_dev: float, df: pd.DataFrame) -> Tuple[int, int]:
        """Calculate confidence range"""
        low = max(1, int(prediction - std_dev))
        high = max(low + 1, int(prediction + std_dev))
        return self._candles_to_minutes(low, df), self._candles_to_minutes(high, df)
    
    def _calculate_confidence(self, candles: float, velocity: float,
                             congestion: float, regime: str, volume: float) -> float:
        """Calculate confidence"""
        conf = 0.5
        
        if velocity > 0.02:
            conf += 0.15
        elif velocity > 0.01:
            conf += 0.1
        elif velocity < 0.002:
            conf -= 0.1
        
        if congestion < 0.3:
            conf += 0.15
        elif congestion < 0.5:
            conf += 0.1
        elif congestion > 0.7:
            conf -= 0.15
        
        if regime in ['HIGH', 'VERY_HIGH']:
            conf += 0.1
        elif regime in ['VERY_LOW']:
            conf -= 0.1
        
        if volume > 0.7:
            conf += 0.1
        elif volume < 0.3:
            conf -= 0.1
        
        if 1 <= candles <= 5:
            conf += 0.15
        elif candles > 50:
            conf -= 0.2
        
        if candles == 0:
            conf = 0.95
        
        return float(np.clip(conf, 0.1, 0.95))
    
    def _get_confidence_level(self, confidence: float) -> TimingConfidence:
        if confidence >= 0.8:
            return TimingConfidence.VERY_HIGH
        elif confidence >= 0.6:
            return TimingConfidence.HIGH
        elif confidence >= 0.4:
            return TimingConfidence.MODERATE
        elif confidence >= 0.2:
            return TimingConfidence.LOW
        else:
            return TimingConfidence.VERY_LOW
    
    def _candles_to_minutes(self, candles: int, df: pd.DataFrame) -> int:
        try:
            if len(df) > 1:
                interval = df.index[-1] - df.index[-2]
                minutes = interval.total_seconds() / 60
                self._cached_minutes_per_candle = minutes
                return int(candles * minutes)
        except:
            pass
        return int(candles * self._cached_minutes_per_candle)
    
    def _build_reasons(self, candles: int, velocity: float, regime: str,
                       congestion: float, confidence: float,
                       range_low: int, range_high: int) -> List[str]:
        reasons = []
        
        if candles == 0:
            reasons.append("Price already in entry zone")
        elif candles <= 3:
            reasons.append(f"Very quick entry (~{candles} candles)")
        elif candles <= 8:
            reasons.append(f"Quick entry (~{candles} candles)")
        elif candles <= 15:
            reasons.append(f"Moderate entry (~{candles} candles)")
        else:
            reasons.append(f"Slow entry (~{candles} candles)")
        
        reasons.append(f"Entry window: {range_low}-{range_high} min")
        
        if velocity > 0.02:
            reasons.append(f"High velocity: {velocity:.2%}/candle")
        elif velocity > 0.01:
            reasons.append(f"Moderate velocity: {velocity:.2%}/candle")
        
        regime_names = {'VERY_LOW': 'Very low', 'LOW': 'Low', 'NORMAL': 'Normal',
                        'HIGH': 'High', 'VERY_HIGH': 'Very high'}
        reasons.append(f"{regime_names.get(regime, regime)} volatility")
        
        if congestion > 0.7:
            reasons.append("Highly congested (slow)")
        elif congestion < 0.3:
            reasons.append("Trending (efficient)")
        
        reasons.append(f"Confidence: {self._get_confidence_level(confidence).value}")
        
        return reasons[:5]
    
    def _fallback_prediction(self, df: pd.DataFrame, signal: Dict) -> TimingPrediction:
        """Simple fallback prediction"""
        try:
            current_price = float(df['close'].iloc[-1])
            direction = signal.get('direction', 'BUY')
            entry_zone_low = signal.get('entry_zone_low', current_price * 0.99)
            entry_zone_high = signal.get('entry_zone_high', current_price * 1.01)
            
            atr = self._calculate_atr(df)
            atr_pct = atr / current_price
            
            entry_distance = self._calculate_distance(
                current_price, direction, entry_zone_low, entry_zone_high
            )
            
            if entry_distance <= 0:
                candles = 0
            else:
                candles = max(1, int((entry_distance / current_price) / (atr_pct * 0.5)))
            
            minutes = self._candles_to_minutes(candles, df)
            now = datetime.now()
            
            return TimingPrediction(
                expected_candles_to_entry=candles,
                expected_minutes_to_entry=minutes,
                entry_confidence=0.4,
                entry_confidence_level=TimingConfidence.LOW,
                entry_time_range_low=max(1, minutes // 2),
                entry_time_range_high=minutes * 2,
                avg_candle_speed=round(atr_pct * 100, 2),
                volatility_regime='NORMAL',
                congestion_level=0.5,
                estimated_entry_time=now + timedelta(minutes=minutes),
                entry_window_start=now + timedelta(minutes=max(1, minutes // 2)),
                entry_window_end=now + timedelta(minutes=minutes * 2),
                reasons=["Fallback prediction - using ATR estimate"]
            )
        except:
            return self._default_prediction()
    
    def _default_prediction(self) -> TimingPrediction:
        now = datetime.now()
        return TimingPrediction(
            expected_candles_to_entry=5,
            expected_minutes_to_entry=25,
            entry_confidence=0.3,
            entry_confidence_level=TimingConfidence.LOW,
            entry_time_range_low=10,
            entry_time_range_high=40,
            avg_candle_speed=0.5,
            volatility_regime='NORMAL',
            congestion_level=0.5,
            estimated_entry_time=now + timedelta(minutes=25),
            entry_window_start=now + timedelta(minutes=10),
            entry_window_end=now + timedelta(minutes=40),
            reasons=['Default prediction - limited data']
        )