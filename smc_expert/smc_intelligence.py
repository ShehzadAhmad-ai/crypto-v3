
"""
SMC Expert V3 - Intelligence Layer (COMPLETE REWRITE)
Context analysis, session detection, kill zones, trap detection, AMD integration
FIXED: No .get() on Swing objects, proper attribute access
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time
from collections import defaultdict

from .smc_core import (
    Direction, SessionType, AMDPhase, MitigationState,
    SMCContext, OrderBlock, FVG, LiquiditySweep, Swing, ZoneType,SwingType,
    calculate_atr, normalize, safe_get_swing_price, safe_get_swing_type
)
from .smc_config import CONFIG


class SessionDetector:
    """Detects market sessions and kill zones"""
    
    def __init__(self):
        self.current_session: SessionType = SessionType.OUTSIDE
        self.is_kill_zone_active: bool = False
        self.session_weight: float = 0.5
    
    def detect_session(self, timestamp: datetime) -> SessionType:
        """
        Detect current market session based on UTC time
        """
        hour = timestamp.hour
        
        # Check for overlaps first
        if CONFIG.LONDON_START <= hour < CONFIG.NY_END:
            if hour < CONFIG.LONDON_END:
                return SessionType.LONDON
            elif hour >= CONFIG.NY_START:
                return SessionType.LONDON_NY_OVERLAP
        
        # Check Asia
        if hour >= CONFIG.ASIA_START or hour < CONFIG.ASIA_END:
            return SessionType.ASIA
        
        # Check London only
        if CONFIG.LONDON_START <= hour < CONFIG.LONDON_END:
            return SessionType.LONDON
        
        # Check NY only
        if CONFIG.NY_START <= hour < CONFIG.NY_END:
            return SessionType.NY
        
        return SessionType.OUTSIDE
    
    def is_kill_zone(self, timestamp: datetime) -> bool:
        """
        Check if current time is in a kill zone
        """
        hour = timestamp.hour
        
        for kz in CONFIG.KILL_ZONES:
            start_hour = kz['start']
            end_hour = kz['end']
            
            if start_hour <= hour < end_hour:
                return True
        
        return False
    
    def get_session_weight(self, session: SessionType) -> float:
        """Get weight multiplier for current session"""
        return CONFIG.SESSION_WEIGHTS.get(session.value, 0.5)
    
    def analyze(self, timestamp: datetime) -> Dict:
        """Complete session analysis"""
        self.current_session = self.detect_session(timestamp)
        self.is_kill_zone_active = self.is_kill_zone(timestamp)
        self.session_weight = self.get_session_weight(self.current_session)
        
        return {
            'session': self.current_session,
            'is_kill_zone': self.is_kill_zone_active,
            'weight': self.session_weight,
            'description': self._get_session_description()
        }
    
    def _get_session_description(self) -> str:
        """Get human-readable session description"""
        descriptions = {
            SessionType.ASIA: "Asian session - typically range-bound, lower volatility",
            SessionType.LONDON: "London session - trend initiation, higher volatility",
            SessionType.NY: "New York session - high volume, strong trends",
            SessionType.LONDON_NY_OVERLAP: "London-NY overlap - highest liquidity",
            SessionType.OUTSIDE: "Outside major sessions - lower probability"
        }
        return descriptions.get(self.current_session, "Unknown session")


class TrapDetector:
    """Detects bull traps, bear traps, and stop hunts - FIXED"""
    
    def __init__(self):
        self.bull_traps: List[Dict] = []
        self.bear_traps: List[Dict] = []
        self.stop_hunts: List[Dict] = []
    
    def detect_all(self, df: pd.DataFrame, sweeps: List[LiquiditySweep]) -> Dict:
        """Detect all types of traps"""
        self.bull_traps = self._detect_bull_traps(df)
        self.bear_traps = self._detect_bear_traps(df)
        self.stop_hunts = self._detect_stop_hunts(df, sweeps)
        
        return {
            'bull_traps': self.bull_traps,
            'bear_traps': self.bear_traps,
            'stop_hunts': self.stop_hunts,
            'recent_traps': self._get_recent_traps(df)
        }
    
    def _detect_bull_traps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Bull trap: Break above resistance, then close below
        """
        traps = []
        atr = calculate_atr(df)
        
        if atr <= 0:
            return traps
        
        for i in range(5, len(df) - 3):
            candle = df.iloc[i]
            
            recent_high = df['high'].iloc[max(0, i-10):i].max()
            
            if candle['high'] > recent_high:
                for j in range(i + 1, min(i + 4, len(df))):
                    next_candle = df.iloc[j]
                    
                    if next_candle['close'] < recent_high:
                        trap_size = (candle['high'] - recent_high) / atr
                        trap_strength = min(1.0, trap_size)
                        
                        traps.append({
                            'type': 'BULL_TRAP',
                            'direction': Direction.SELL,
                            'break_price': candle['high'],
                            'trap_level': recent_high,
                            'strength': trap_strength,
                            'index': i,
                            'timestamp': candle.name if isinstance(candle.name, datetime) else datetime.now()
                        })
                        break
        
        return traps
    
    def _detect_bear_traps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Bear trap: Break below support, then close above
        """
        traps = []
        atr = calculate_atr(df)
        
        if atr <= 0:
            return traps
        
        for i in range(5, len(df) - 3):
            candle = df.iloc[i]
            
            recent_low = df['low'].iloc[max(0, i-10):i].min()
            
            if candle['low'] < recent_low:
                for j in range(i + 1, min(i + 4, len(df))):
                    next_candle = df.iloc[j]
                    
                    if next_candle['close'] > recent_low:
                        trap_size = (recent_low - candle['low']) / atr
                        trap_strength = min(1.0, trap_size)
                        
                        traps.append({
                            'type': 'BEAR_TRAP',
                            'direction': Direction.BUY,
                            'break_price': candle['low'],
                            'trap_level': recent_low,
                            'strength': trap_strength,
                            'index': i,
                            'timestamp': candle.name if isinstance(candle.name, datetime) else datetime.now()
                        })
                        break
        
        return traps
    
    def _detect_stop_hunts(self, df: pd.DataFrame, sweeps: List[LiquiditySweep]) -> List[Dict]:
        """Stop hunts: Price sweeps liquidity and reverses"""
        stop_hunts = []
        
        for sweep in sweeps:
            if sweep.reversal_strength > 0.6:
                stop_hunts.append({
                    'type': 'STOP_HUNT',
                    'direction': Direction.BUY if sweep.type == 'SSL_SWEEP' else Direction.SELL,
                    'sweep_price': sweep.price,
                    'target_level': sweep.target_level,
                    'strength': sweep.reversal_strength,
                    'timestamp': sweep.timestamp
                })
        
        return stop_hunts
    
    def _get_recent_traps(self, df: pd.DataFrame) -> List[Dict]:
        """Get traps from last 10 bars"""
        all_traps = self.bull_traps + self.bear_traps + self.stop_hunts
        recent = [t for t in all_traps if t.get('index', len(df)) > len(df) - 10]
        return recent


class AMDIntegrator:
    """Integrates AMD phase analysis with market structure"""
    
    def __init__(self):
        self.phase: AMDPhase = AMDPhase.UNKNOWN
        self.confidence: float = 0.0
    
    def analyze(self, df: pd.DataFrame, swings: List[Swing],
                 bos_points: List[Dict], sweeps: List[LiquiditySweep]) -> Dict:
        """Complete AMD analysis"""
        atr = calculate_atr(df)
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50
        
        # Accumulation score
        acc_score = self._score_accumulation(df, atr)
        
        # Manipulation score
        man_score = self._score_manipulation(df, sweeps, atr)
        
        # Distribution score
        dist_score = self._score_distribution(df, bos_points, atr)
        
        scores = {
            AMDPhase.ACCUMULATION: acc_score,
            AMDPhase.MANIPULATION: man_score,
            AMDPhase.DISTRIBUTION: dist_score
        }
        
        self.phase = max(scores, key=scores.get)
        self.confidence = scores[self.phase]
        
        return {
            'phase': self.phase,
            'confidence': self.confidence,
            'description': self._get_description(),
            'scores': scores
        }
    
    def _score_accumulation(self, df: pd.DataFrame, atr: float) -> float:
        """Score accumulation phase characteristics"""
        lookback = min(20, len(df))
        recent = df.tail(lookback)
        
        price_range = recent['high'].max() - recent['low'].min()
        range_score = 1.0 - min(1.0, price_range / (atr * 3))
        
        volumes = recent['volume'].values
        if len(volumes) > 5 and volumes[0] > 0:
            volume_trend = (volumes[-1] - volumes[0]) / volumes[0]
            volume_score = max(0.0, 1.0 - abs(volume_trend))
        else:
            volume_score = 0.5
        
        if 'body_ratio' in df.columns:
            body_ratios = recent['body_ratio'].values
            candle_score = 1.0 - np.mean(body_ratios)
        else:
            candle_score = 0.5
        
        score = (range_score * 0.4 + volume_score * 0.3 + candle_score * 0.3)
        
        return min(1.0, score)
    
    def _score_manipulation(self, df: pd.DataFrame, sweeps: List[LiquiditySweep], atr: float) -> float:
        """Score manipulation phase characteristics"""
        if not sweeps:
            return 0.3
        
        recent_sweeps = [s for s in sweeps if s.candle_index > len(df) - 15]
        
        if not recent_sweeps:
            return 0.3
        
        sweep_count_score = min(1.0, len(recent_sweeps) / 3)
        avg_strength = np.mean([s.reversal_strength for s in recent_sweeps])
        
        score = (sweep_count_score * 0.5 + avg_strength * 0.5)
        
        return min(1.0, score)
    
    def _score_distribution(self, df: pd.DataFrame, bos_points: List[Dict], atr: float) -> float:
        """Score distribution phase characteristics"""
        if not bos_points:
            return 0.3
        
        recent_bos = [b for b in bos_points if b.get('index', 0) > len(df) - 20]
        
        if not recent_bos:
            return 0.3
        
        avg_bos_strength = np.mean([b.get('strength', 0.5) for b in recent_bos])
        
        price_change = df['close'].iloc[-1] - df['close'].iloc[-20]
        trend_score = min(1.0, abs(price_change) / (atr * 3)) if atr > 0 else 0.5
        
        score = (avg_bos_strength * 0.5 + trend_score * 0.5)
        
        return min(1.0, score)
    
    def _get_description(self) -> str:
        """Get human-readable phase description"""
        descriptions = {
            AMDPhase.ACCUMULATION: "Smart money building positions. Look for range boundaries.",
            AMDPhase.MANIPULATION: "Liquidity sweeps occurring. Wait for reversal confirmation.",
            AMDPhase.DISTRIBUTION: "Trend established. Look for continuation toward targets.",
            AMDPhase.UNKNOWN: "No clear AMD phase detected."
        }
        return descriptions.get(self.phase, "Unknown phase")


class ContextAnalyzer:
    """
    Main intelligence orchestrator - FIXED: Proper Swing object access
    """
    
    def __init__(self):
        self.session_detector = SessionDetector()
        self.trap_detector = TrapDetector()
        self.amd_integrator = AMDIntegrator()
        
        self.context = SMCContext()
        self._context_summary_cache: Dict = {}
    
    def analyze(self, df: pd.DataFrame, swings: List[Swing],
                 bos_points: List[Dict], choch_points: List[Dict],
                 sweeps: List[LiquiditySweep], premium_discount_zone: ZoneType,
                 htf_trend: str = "NEUTRAL") -> SMCContext:
        """
        Build complete market context
        """
        current_price = df['close'].iloc[-1]
        atr = calculate_atr(df)
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50
        
        timestamp = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
        
        # Session analysis
        session_data = self.session_detector.analyze(timestamp)
        
        # Trap analysis
        trap_data = self.trap_detector.detect_all(df, sweeps)
        
        # AMD analysis
        amd_data = self.amd_integrator.analyze(df, swings, bos_points, sweeps)
        
        # Build context
        self.context = SMCContext(
            current_structure=self._determine_structure(swings),
            structure_strength=self._calculate_structure_strength(swings),
            swings=swings,
            bos_points=bos_points,
            choch_points=choch_points,
            amd_phase=amd_data['phase'],
            amd_confidence=amd_data['confidence'],
            session=session_data['session'],
            is_kill_zone=session_data['is_kill_zone'],
            session_weight=session_data['weight'],
            current_price=current_price,
            atr=atr,
            volatility_regime=self._get_volatility_regime(df, atr),
            zone_type=premium_discount_zone,
            zone_percent=self._calculate_zone_percent(premium_discount_zone),
            htf_trend=htf_trend,
            htf_alignment_score=self._calculate_htf_alignment(
                self._determine_structure(swings), htf_trend
            ),
            recent_sweeps=sweeps[-5:] if sweeps else [],
            recent_traps=trap_data['recent_traps'],
            displacement_bars=self._get_displacement_bars(df, atr)
        )
        
        # Cache summary
        self._context_summary_cache = {
            'market_bias': self.context.current_structure,
            'amd_phase': self.context.amd_phase.value,
            'session': self.context.session.value,
            'kill_zone': self.context.is_kill_zone,
            'volatility': self.context.volatility_regime,
            'zone': self.context.zone_type.value,
            'htf_alignment': f"{self.context.htf_alignment_score:.0%}",
            'recent_sweeps': len(self.context.recent_sweeps)
        }
        
        return self.context
    
    def _determine_structure(self, swings: List[Swing]) -> str:
        """Determine market structure from swings - FIXED: Direct attribute access"""
        if len(swings) < 4:
            return "NEUTRAL"
        
        last_4 = swings[-4:]
        
        hh_hl = 0
        lh_ll = 0
        
        for s in last_4:
            if s.type in [SwingType.HH, SwingType.HL]:
                hh_hl += 1
            elif s.type in [SwingType.LH, SwingType.LL]:
                lh_ll += 1
        
        if hh_hl >= 3:
            return "BULLISH"
        elif lh_ll >= 3:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_structure_strength(self, swings: List[Swing]) -> float:
        """Calculate structure strength"""
        if len(swings) < 4:
            return 0.5
        
        last_6 = swings[-6:]
        strengths = [s.strength for s in last_6]
        
        return np.mean(strengths)
    
    def _get_volatility_regime(self, df: pd.DataFrame, atr: float) -> str:
        """Determine volatility regime"""
        if len(df) < 40 or atr <= 0:
            return "NORMAL"
        
        recent_atr = calculate_atr(df.tail(20))
        historical_atr = calculate_atr(df.head(len(df) - 20))
        
        if historical_atr <= 0:
            return "NORMAL"
        
        ratio = recent_atr / historical_atr
        
        if ratio > 1.5:
            return "HIGH"
        elif ratio < 0.7:
            return "LOW"
        else:
            return "NORMAL"
    
    def _calculate_zone_percent(self, zone: ZoneType) -> float:
        """Calculate zone percentage (0-1)"""
        zone_map = {
            ZoneType.DEEP_DISCOUNT: 0.15,
            ZoneType.DISCOUNT: 0.30,
            ZoneType.EQUILIBRIUM: 0.50,
            ZoneType.PREMIUM: 0.70,
            ZoneType.DEEP_PREMIUM: 0.85
        }
        return zone_map.get(zone, 0.50)
    
    def _calculate_htf_alignment(self, ltf_direction: str, htf_trend: str) -> float:
        """Calculate HTF alignment score"""
        if ltf_direction == htf_trend:
            return 0.9
        elif htf_trend == "NEUTRAL":
            return 0.6
        elif ltf_direction == "NEUTRAL":
            return 0.5
        else:
            return 0.3
    
    def _get_displacement_bars(self, df: pd.DataFrame, atr: float) -> List[int]:
        """Get indices of displacement bars"""
        if atr <= 0:
            return []
        
        displacement = []
        
        for i in range(1, len(df)):
            candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]
            
            move = abs(candle['close'] - prev_candle['close'])
            move_atr = move / atr
            
            if move_atr > 0.8:
                avg_volume = df['volume'].iloc[max(0, i-20):i].mean()
                volume_ratio = candle['volume'] / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.2:
                    displacement.append(i)
        
        return displacement
    
    def get_context_summary(self) -> Dict:
        """Get human-readable context summary"""
        return self._context_summary_cache.copy()