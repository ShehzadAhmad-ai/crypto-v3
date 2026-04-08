"""
SMC Expert V3 - Liquidity Analysis (COMPLETE REWRITE)
BSL/SSL detection, sweep detection, liquidity mapping, next target prediction
FIXED: Proper swing object access, no .get() on objects
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from .smc_core import (
    LiquidityLevel, LiquiditySweep, Direction, Swing, SwingType,
    calculate_atr, normalize, safe_get_swing_price, safe_get_swing_type
)
from .smc_config import CONFIG


class LiquidityDetector:
    """Detects Buy Side Liquidity (BSL) and Sell Side Liquidity (SSL)"""
    
    def __init__(self):
        self.bsl_levels: List[LiquidityLevel] = []
        self.ssl_levels: List[LiquidityLevel] = []
    
    def detect_all(self, df: pd.DataFrame, swings: List[Swing]) -> Dict[str, List[LiquidityLevel]]:
        """Detect both BSL and SSL levels"""
        self.bsl_levels = self._detect_bsl(df, swings)
        self.ssl_levels = self._detect_ssl(df, swings)
        
        # Cluster nearby levels
        atr = calculate_atr(df)
        if atr > 0:
            self.bsl_levels = self._cluster_levels(self.bsl_levels, atr)
            self.ssl_levels = self._cluster_levels(self.ssl_levels, atr)
        
        return {
            'bsl': self.bsl_levels,
            'ssl': self.ssl_levels
        }
    
    def _detect_bsl(self, df: pd.DataFrame, swings: List[Swing]) -> List[LiquidityLevel]:
        """
        Buy Side Liquidity - stops above highs
        - Swing highs (HH, LH)
        - Equal highs (multiple touches)
        - Recent highs
        """
        levels = []
        atr = calculate_atr(df)
        current_price = df['close'].iloc[-1]
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50
        
        # 1. Swing highs
        for swing in swings:
            if swing.type in [SwingType.HH, SwingType.LH]:
                distance = abs(swing.price - current_price) / atr
                strength = self._calculate_level_strength(df, swing.index, swing.price, 'high')
                
                level = LiquidityLevel(
                    price=swing.price,
                    type='BSL',
                    touches=1,
                    strength=strength,
                    distance_pct=abs(swing.price - current_price) / current_price if current_price > 0 else 0,
                    distance_atr=distance,
                    timestamp=swing.timestamp
                )
                levels.append(level)
        
        # 2. Equal highs
        equal_highs = self._detect_equal_levels(df, 'high')
        for high in equal_highs:
            distance = abs(high['price'] - current_price) / atr
            strength = min(1.0, high['touches'] / 4)
            
            level = LiquidityLevel(
                price=high['price'],
                type='BSL',
                touches=high['touches'],
                strength=strength,
                distance_pct=abs(high['price'] - current_price) / current_price if current_price > 0 else 0,
                distance_atr=distance,
                timestamp=high['timestamp']
            )
            levels.append(level)
        
        # 3. Recent highs (last 10 bars)
        recent_high = df['high'].iloc[-10:].max()
        if recent_high > current_price:
            distance = abs(recent_high - current_price) / atr
            level = LiquidityLevel(
                price=recent_high,
                type='BSL',
                touches=1,
                strength=0.4,
                distance_pct=abs(recent_high - current_price) / current_price if current_price > 0 else 0,
                distance_atr=distance,
                timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
            )
            levels.append(level)
        
        # Sort by strength and remove duplicates
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        return levels
    
    def _detect_ssl(self, df: pd.DataFrame, swings: List[Swing]) -> List[LiquidityLevel]:
        """
        Sell Side Liquidity - stops below lows
        """
        levels = []
        atr = calculate_atr(df)
        current_price = df['close'].iloc[-1]
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50
        
        # 1. Swing lows
        for swing in swings:
            if swing.type in [SwingType.HL, SwingType.LL]:
                distance = abs(swing.price - current_price) / atr
                strength = self._calculate_level_strength(df, swing.index, swing.price, 'low')
                
                level = LiquidityLevel(
                    price=swing.price,
                    type='SSL',
                    touches=1,
                    strength=strength,
                    distance_pct=abs(swing.price - current_price) / current_price if current_price > 0 else 0,
                    distance_atr=distance,
                    timestamp=swing.timestamp
                )
                levels.append(level)
        
        # 2. Equal lows
        equal_lows = self._detect_equal_levels(df, 'low')
        for low in equal_lows:
            distance = abs(low['price'] - current_price) / atr
            strength = min(1.0, low['touches'] / 4)
            
            level = LiquidityLevel(
                price=low['price'],
                type='SSL',
                touches=low['touches'],
                strength=strength,
                distance_pct=abs(low['price'] - current_price) / current_price if current_price > 0 else 0,
                distance_atr=distance,
                timestamp=low['timestamp']
            )
            levels.append(level)
        
        # 3. Recent lows
        recent_low = df['low'].iloc[-10:].min()
        if recent_low < current_price:
            distance = abs(recent_low - current_price) / atr
            level = LiquidityLevel(
                price=recent_low,
                type='SSL',
                touches=1,
                strength=0.4,
                distance_pct=abs(recent_low - current_price) / current_price if current_price > 0 else 0,
                distance_atr=distance,
                timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
            )
            levels.append(level)
        
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        return levels
    
    def _detect_equal_levels(self, df: pd.DataFrame, level_type: str) -> List[Dict]:
        """Detect equal highs or lows with multiple touches"""
        tolerance_pct = 0.002  # 0.2% tolerance
        levels = defaultdict(list)
        
        if level_type == 'high':
            prices = df['high'].values
        else:
            prices = df['low'].values
        
        for i, price in enumerate(prices):
            matched = False
            for key in list(levels.keys()):
                if abs(price - key) / key < tolerance_pct:
                    levels[key].append({
                        'index': i,
                        'price': price,
                        'timestamp': df.index[i] if isinstance(df.index[i], datetime) else datetime.now()
                    })
                    matched = True
                    break
            
            if not matched:
                levels[price].append({
                    'index': i,
                    'price': price,
                    'timestamp': df.index[i] if isinstance(df.index[i], datetime) else datetime.now()
                })
        
        result = []
        for price, touches in levels.items():
            if len(touches) >= 2:
                result.append({
                    'price': price,
                    'touches': len(touches),
                    'timestamp': touches[-1]['timestamp']
                })
        
        return result
    
    def _calculate_level_strength(self, df: pd.DataFrame, index: int, 
                                   price: float, level_type: str) -> float:
        """Calculate liquidity level strength"""
        # Recency
        recency = 1.0 - (len(df) - index) / len(df) if len(df) > 0 else 0.5
        
        # Volume at that level
        volume_at_level = df['volume'].iloc[index]
        avg_volume = df['volume'].iloc[max(0, index-20):index].mean()
        volume_score = min(1.0, volume_at_level / avg_volume) if avg_volume > 0 else 0.5
        
        strength = (volume_score * 0.5 + recency * 0.5)
        
        return min(1.0, strength)
    
    def _cluster_levels(self, levels: List[LiquidityLevel], atr: float) -> List[LiquidityLevel]:
        """Cluster nearby liquidity levels"""
        if len(levels) < 2 or atr <= 0:
            return levels
        
        tolerance = atr * CONFIG.LIQUIDITY_CLUSTER_TOLERANCE_ATR
        clustered = []
        used = set()
        
        for i, level in enumerate(levels):
            if i in used:
                continue
            
            cluster = [level]
            cluster_indices = [i]
            
            for j in range(i + 1, len(levels)):
                if j in used:
                    continue
                
                if abs(level.price - levels[j].price) < tolerance:
                    cluster.append(levels[j])
                    cluster_indices.append(j)
            
            # Merge cluster
            avg_price = np.mean([l.price for l in cluster])
            total_touches = sum(l.touches for l in cluster)
            avg_strength = np.mean([l.strength for l in cluster])
            avg_distance = np.mean([l.distance_atr for l in cluster])
            
            merged = LiquidityLevel(
                price=avg_price,
                type=level.type,
                touches=total_touches,
                strength=min(1.0, avg_strength * (1 + len(cluster) * 0.1)),
                distance_pct=avg_distance * atr / avg_price if avg_price > 0 else 0,
                distance_atr=avg_distance,
                timestamp=max(l.timestamp for l in cluster),
                cluster_id=len(clustered)
            )
            clustered.append(merged)
            
            for idx in cluster_indices:
                used.add(idx)
        
        return clustered


class SweepDetector:
    """Detects liquidity sweeps (stop hunts)"""
    
    def __init__(self):
        self.sweeps: List[LiquiditySweep] = []
    
    def detect_sweeps(self, df: pd.DataFrame, 
                       bsl_levels: List[LiquidityLevel],
                       ssl_levels: List[LiquidityLevel]) -> List[LiquiditySweep]:
        """Detect when price sweeps through liquidity levels and reverses"""
        self.sweeps = []
        atr = calculate_atr(df)
        
        if atr <= 0:
            return self.sweeps
        
        # Check for BSL sweeps
        for level in bsl_levels:
            sweep = self._check_bsl_sweep(df, level, atr)
            if sweep:
                self.sweeps.append(sweep)
        
        # Check for SSL sweeps
        for level in ssl_levels:
            sweep = self._check_ssl_sweep(df, level, atr)
            if sweep:
                self.sweeps.append(sweep)
        
        return self.sweeps
    
    def _check_bsl_sweep(self, df: pd.DataFrame, level: LiquidityLevel, 
                          atr: float) -> Optional[LiquiditySweep]:
        """Check for BSL sweep (break above level, then close below)"""
        lookback = min(15, len(df))
        
        for i in range(len(df) - lookback, len(df) - 1):
            candle = df.iloc[i]
            
            if candle['high'] > level.price:
                for j in range(i + 1, min(i + 4, len(df))):
                    next_candle = df.iloc[j]
                    
                    if next_candle['close'] < level.price:
                        sweep_size = (candle['high'] - level.price) / atr
                        reversal_size = (level.price - next_candle['close']) / atr
                        
                        reversal_strength = min(1.0, (sweep_size + reversal_size) / 3)
                        
                        return LiquiditySweep(
                            price=candle['high'],
                            type='BSL_SWEEP',
                            target_level=level.price,
                            reversal_strength=reversal_strength,
                            timestamp=candle.name if isinstance(candle.name, datetime) else datetime.now(),
                            candle_index=i
                        )
        
        return None
    
    def _check_ssl_sweep(self, df: pd.DataFrame, level: LiquidityLevel,
                          atr: float) -> Optional[LiquiditySweep]:
        """Check for SSL sweep (break below level, then close above)"""
        lookback = min(15, len(df))
        
        for i in range(len(df) - lookback, len(df) - 1):
            candle = df.iloc[i]
            
            if candle['low'] < level.price:
                for j in range(i + 1, min(i + 4, len(df))):
                    next_candle = df.iloc[j]
                    
                    if next_candle['close'] > level.price:
                        sweep_size = (level.price - candle['low']) / atr
                        reversal_size = (next_candle['close'] - level.price) / atr
                        
                        reversal_strength = min(1.0, (sweep_size + reversal_size) / 3)
                        
                        return LiquiditySweep(
                            price=candle['low'],
                            type='SSL_SWEEP',
                            target_level=level.price,
                            reversal_strength=reversal_strength,
                            timestamp=candle.name if isinstance(candle.name, datetime) else datetime.now(),
                            candle_index=i
                        )
        
        return None


class LiquidityMap:
    """Maps all liquidity and predicts next target"""
    
    def __init__(self):
        self.liquidity_map: Dict = {}
    
    def build_map(self, df: pd.DataFrame, 
                   bsl_levels: List[LiquidityLevel],
                   ssl_levels: List[LiquidityLevel]) -> Dict:
        """Build complete liquidity map"""
        current_price = df['close'].iloc[-1]
        atr = calculate_atr(df)
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50
        
        # Separate above and below
        above = [l for l in bsl_levels if l.price > current_price]
        below = [l for l in ssl_levels if l.price < current_price]
        
        # Sort by distance
        above.sort(key=lambda x: x.distance_atr)
        below.sort(key=lambda x: x.distance_atr)
        
        self.liquidity_map = {
            'above': [
                {
                    'price': l.price,
                    'type': l.type,
                    'touches': l.touches,
                    'strength': l.strength,
                    'distance_pct': l.distance_pct,
                    'distance_atr': l.distance_atr
                }
                for l in above[:10]
            ],
            'below': [
                {
                    'price': l.price,
                    'type': l.type,
                    'touches': l.touches,
                    'strength': l.strength,
                    'distance_pct': l.distance_pct,
                    'distance_atr': l.distance_atr
                }
                for l in below[:10]
            ],
            'total_above': len(above),
            'total_below': len(below),
            'nearest_above': above[0] if above else None,
            'nearest_below': below[0] if below else None
        }
        
        return self.liquidity_map
    
    def get_next_target(self, direction: Direction, current_price: float) -> Optional[Dict]:
        """Predict next liquidity target based on direction"""
        if direction == Direction.BUY:
            targets = self.liquidity_map.get('above', [])
        else:
            targets = self.liquidity_map.get('below', [])
        
        if not targets:
            return None
        
        targets.sort(key=lambda x: (x['distance_atr'], -x['strength']))
        next_target = targets[0]
        
        probability = min(0.9, next_target['strength'] * (1 - min(0.5, next_target['distance_atr'] / 5)))
        
        return {
            'price': next_target['price'],
            'type': next_target['type'],
            'distance_pct': next_target['distance_pct'],
            'distance_atr': next_target['distance_atr'],
            'strength': next_target['strength'],
            'touches': next_target['touches'],
            'probability': probability
        }


class LiquidityManager:
    """Main liquidity manager orchestrating all liquidity analysis"""
    
    def __init__(self):
        self.detector = LiquidityDetector()
        self.sweep_detector = SweepDetector()
        self.map_builder = LiquidityMap()
        
        self.bsl_levels: List[LiquidityLevel] = []
        self.ssl_levels: List[LiquidityLevel] = []
        self.sweeps: List[LiquiditySweep] = []
        self.liquidity_map: Dict = {}
    
    def analyze(self, df: pd.DataFrame, swings: List[Swing]) -> Dict:
        """Complete liquidity analysis"""
        # Detect liquidity levels
        levels = self.detector.detect_all(df, swings)
        self.bsl_levels = levels['bsl']
        self.ssl_levels = levels['ssl']
        
        # Detect sweeps
        self.sweeps = self.sweep_detector.detect_sweeps(df, self.bsl_levels, self.ssl_levels)
        
        # Build liquidity map
        self.liquidity_map = self.map_builder.build_map(df, self.bsl_levels, self.ssl_levels)
        
        return {
            'bsl_levels': self.bsl_levels,
            'ssl_levels': self.ssl_levels,
            'sweeps': self.sweeps,
            'liquidity_map': self.liquidity_map,
            'active_sweeps': [s for s in self.sweeps if s.reversal_strength > 0.5]
        }
    
    def get_next_target(self, direction: Direction, current_price: float) -> Optional[Dict]:
        """Get next liquidity target"""
        return self.map_builder.get_next_target(direction, current_price)