import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from config import Config
from logger import log

EPSILON = 1e-10


@dataclass
class MarketStructure:
    price: float
    index: int
    type: str
    timestamp: pd.Timestamp
    strength: float = 1.0

@dataclass
class FairValueGap:
    high: float
    low: float
    start_idx: int
    end_idx: int
    direction: str
    filled: bool = False
    fill_price: Optional[float] = None
    fill_bars: int = 0

@dataclass
class LiquidityZone:
    price: float
    type: str
    volume: float
    strength: float
    timeframe: str


class MicrostructureEngine:
    def __init__(self):
        self.fvg_threshold = getattr(Config, 'FVG_THRESHOLD', 0.002)
        self.liquidity_merge_threshold = getattr(Config, 'LIQUIDITY_MERGE_THRESHOLD', 0.0015)
        
        # Smart Money thresholds from config
        self.fvg_strength_threshold = getattr(Config, 'SM_FVG_STRENGTH_THRESHOLD', 0.6)
        self.displacement_strength_threshold = getattr(Config, 'SM_DISPLACEMENT_STRENGTH_THRESHOLD', 0.7)
        
        log.info("MicrostructureEngine enhanced with scoring methods")

    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        # defensive
        if df is None or df.empty or len(df) < 20:
            return {
                'structure': [],
                'bos': [],
                'choch': [],
                'fvg': [],
                'displacement': [],
                'liquidity': [],
                'order_block': [],
                'internal_external': {'type': 'UNKNOWN', 'confidence': 0.0},
                'composite_confidence': 0.1,
                'swing_count': 0,
                'last_bos': None,
                'structure_health': 'UNKNOWN'
            }

        results = {}
        try:
            results['structure'] = self._detect_structure_points(df)
            results['bos'] = self._detect_break_of_structure(results['structure'], df)
            results['choch'] = self._detect_change_of_character(results['structure'], df)
            results['fvg'] = self._detect_fair_value_gaps(df)
            results['fvg_filled'] = self._detect_filled_fvg(df, results['fvg'])
            results['displacement'] = self._detect_displacement_moves(df)
            results['liquidity'] = self._detect_liquidity_zones(df)
            results['order_block'] = self._detect_order_blocks(df)
            results['internal_external'] = self._analyze_internal_external(results['structure'])
            results['swing_count'] = len(results['structure'])
            results['last_bos'] = results['bos'][-1] if results['bos'] else None
            results['structure_health'] = self._assess_structure_health(results)
            
            # composite confidence: base + incremental boosts
            base = 0.1
            base += min(0.6, 0.12 * len(results.get('bos', [])))
            base += min(0.2, 0.05 * len(results.get('displacement', [])))
            base += min(0.15, 0.03 * len(results.get('fvg', [])))
            base = min(0.95, base)
            results['composite_confidence'] = float(base)
        except Exception as e:
            # fallback safe structure
            return {
                'structure': [],
                'bos': [],
                'choch': [],
                'fvg': [],
                'displacement': [],
                'liquidity': [],
                'order_block': [],
                'internal_external': {'type': 'UNKNOWN', 'confidence': 0.0},
                'composite_confidence': 0.1,
                'swing_count': 0,
                'last_bos': None,
                'structure_health': 'ERROR',
                'error': str(e)
            }
        return results

    def _detect_structure_points(self, df: pd.DataFrame) -> List[MarketStructure]:
        structs: List[MarketStructure] = []
        window = 3
        n = len(df)
        # iterate and detect local highs/lows
        for i in range(window, n - window):
            try:
                left = df.iloc[i - window:i]
                right = df.iloc[i + 1:i + window + 1]
                if df['high'].iat[i] > left['high'].max() and df['high'].iat[i] > right['high'].max():
                    structs.append(MarketStructure(price=float(df['high'].iat[i]), index=i, type='HH', timestamp=df.index[i], strength=1.0))
                if df['low'].iat[i] < left['low'].min() and df['low'].iat[i] < right['low'].min():
                    structs.append(MarketStructure(price=float(df['low'].iat[i]), index=i, type='LL', timestamp=df.index[i], strength=1.0))
            except Exception:
                continue
        return structs

    def _detect_break_of_structure(self, structures: List[MarketStructure], df: pd.DataFrame) -> List[Dict]:
        bos = []
        if not structures or len(structures) < 2:
            return bos
        for i in range(1, len(structures)):
            prev = structures[i - 1]
            curr = structures[i]
            # bullish BOS: LL then HH and price exceeded prior reference
            if prev.type == 'LL' and curr.type == 'HH':
                if curr.price > prev.price * (1.0 + 0.002):  # small threshold
                    strength = min(1.0, (curr.price - prev.price) / max(prev.price, 1e-8))
                    bos.append({
                        'type': 'BULLISH_BOS',
                        'price': curr.price,
                        'timestamp': curr.timestamp,
                        'break_level': prev.price,
                        'strength': float(strength),
                        'bars_since': 0
                    })
            # bearish BOS: HH then LL
            if prev.type == 'HH' and curr.type == 'LL':
                if curr.price < prev.price * (1.0 - 0.002):
                    strength = min(1.0, (prev.price - curr.price) / max(prev.price, 1e-8))
                    bos.append({
                        'type': 'BEARISH_BOS',
                        'price': curr.price,
                        'timestamp': curr.timestamp,
                        'break_level': prev.price,
                        'strength': float(strength),
                        'bars_since': 0
                    })
        return bos

    def _detect_change_of_character(self, structures: List[MarketStructure], df: pd.DataFrame) -> List[Dict]:
        """
        CHOCH: Change of Character - when the direction of swings reverses
        Bullish CHOCH: after LL, we get HH, then another LL that's HIGHER than prior LL
        Bearish CHOCH: after HH, we get LL, then another HH that's LOWER than prior HH
        """
        choch = []
        if not structures or len(structures) < 3:
            return choch
        
        for i in range(2, len(structures)):
            s2 = structures[i-2]
            s1 = structures[i-1]
            s0 = structures[i]
            
            # Bullish CHOCH: LL -> HH -> LL(higher)
            if s2.type == 'LL' and s1.type == 'HH' and s0.type == 'LL':
                if s0.price > s2.price:
                    strength = (s0.price - s2.price) / max(s2.price, 1e-8)
                    choch.append({
                        'type': 'BULLISH_CHOCH',
                        'price': s0.price,
                        'timestamp': s0.timestamp,
                        'reference_price': s2.price,
                        'strength': min(1.0, strength),
                        'structure': [s2.price, s1.price, s0.price]
                    })
            
            # Bearish CHOCH: HH -> LL -> HH(lower)
            if s2.type == 'HH' and s1.type == 'LL' and s0.type == 'HH':
                if s0.price < s2.price:
                    strength = (s2.price - s0.price) / max(s2.price, 1e-8)
                    choch.append({
                        'type': 'BEARISH_CHOCH',
                        'price': s0.price,
                        'timestamp': s0.timestamp,
                        'reference_price': s2.price,
                        'strength': min(1.0, strength),
                        'structure': [s2.price, s1.price, s0.price]
                    })
        
        return choch

    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        fvgs: List[FairValueGap] = []
        # simple candle-gap detection across recent bars
        for i in range(1, len(df) - 1):
            try:
                c1 = df.iloc[i - 1]
                c2 = df.iloc[i]
                # bullish gap: c2 low > c1 high by threshold
                if c2['low'] > c1['high'] * (1.0 + self.fvg_threshold):
                    gap_height = c2['low'] - c1['high']
                    fvgs.append(FairValueGap(
                        high=float(c2['low']),
                        low=float(c1['high']),
                        start_idx=i - 1,
                        end_idx=i,
                        direction='BULLISH'
                    ))
                # bearish gap
                if c2['high'] < c1['low'] * (1.0 - self.fvg_threshold):
                    fvgs.append(FairValueGap(
                        high=float(c1['low']),
                        low=float(c2['high']),
                        start_idx=i - 1,
                        end_idx=i,
                        direction='BEARISH'
                    ))
            except Exception:
                continue
        return fvgs

    def _detect_filled_fvg(self, df: pd.DataFrame, fvgs: List[FairValueGap]) -> List[Dict]:
        """
        Track FVG fill status - how many bars took to fill
        """
        filled = []
        current_price = float(df['close'].iloc[-1])
        
        for fvg in fvgs:
            if fvg.direction == 'BULLISH':
                # Bullish FVG is filled when price comes back down into it
                if fvg.low <= current_price <= fvg.high:
                    bars_since = len(df) - fvg.end_idx
                    filled.append({
                        'direction': 'BULLISH',
                        'gap_range': fvg.high - fvg.low,
                        'bars_to_fill': bars_since,
                        'status': 'FILLED',
                        'entry_price': fvg.high,
                        'fill_price': current_price
                    })
            else:
                # Bearish FVG is filled when price comes back up into it
                if fvg.low <= current_price <= fvg.high:
                    bars_since = len(df) - fvg.end_idx
                    filled.append({
                        'direction': 'BEARISH',
                        'gap_range': fvg.high - fvg.low,
                        'bars_to_fill': bars_since,
                        'status': 'FILLED',
                        'entry_price': fvg.low,
                        'fill_price': current_price
                    })
        
        return filled

    def _detect_displacement_moves(self, df: pd.DataFrame) -> List[Dict]:
        disps = []
        try:
            atr = float(df['atr'].iat[-1]) if 'atr' in df and df['atr'].iat[-1] > 0 else float(df['close'].iat[-1]) * 0.02
        except Exception:
            atr = float(df['close'].iat[-1]) * 0.02
        
        for i in range(1, len(df)):
            try:
                candle = df.iloc[i]
                prev = df.iloc[i - 1]
                candle_size = abs(float(candle['close']) - float(candle['open']))
                
                if candle_size > atr * 1.5:
                    direction = 'BULLISH' if candle['close'] > candle['open'] else 'BEARISH'
                    vol_mean = df['volume'].rolling(20).mean().iat[i] if len(df) >= 20 else df['volume'].mean()
                    vol_ratio = float(candle['volume']) / max(vol_mean, 1e-8) if vol_mean else 1.0
                    
                    disps.append({
                        'index': i,
                        'timestamp': df.index[i],
                        'direction': direction,
                        'size': float(candle_size),
                        'size_atr_multiple': float(candle_size / atr),
                        'volume_ratio': float(vol_ratio),
                        'strength': float(min(1.0, candle_size / max(atr, 1e-8)))
                    })
            except Exception:
                continue
        return disps

    def _detect_liquidity_zones(self, df: pd.DataFrame) -> List[LiquidityZone]:
        zones: List[LiquidityZone] = []
        try:
            lookback = min(100, max(10, len(df) // 3))
            recent = df.iloc[-lookback:]
            
            # Find most touched prices
            price_touches = {}
            tolerance = float(df['close'].iloc[-1]) * 0.002  # 0.2%
            
            for i in range(len(recent)):
                high = float(recent['high'].iat[i])
                low = float(recent['low'].iat[i])
                
                # Round to tolerance level
                high_level = round(high / tolerance) * tolerance
                low_level = round(low / tolerance) * tolerance
                
                for level in [high_level, low_level]:
                    if level not in price_touches:
                        price_touches[level] = {'count': 0, 'volume': 0.0}
                    price_touches[level]['count'] += 1
                    price_touches[level]['volume'] += float(recent['volume'].iat[i])
            
            # Get top liquidity zones
            sorted_zones = sorted(
                price_touches.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            
            for price_level, data in sorted_zones[:5]:
                if data['count'] >= 2:
                    zones.append(LiquidityZone(
                        price=float(price_level),
                        type='LIQUIDITY_CLUSTER',
                        volume=float(data['volume']),
                        strength=min(1.0, data['count'] / lookback),
                        timeframe='recent'
                    ))
        except Exception:
            pass
        
        return zones

    def _detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        obs = []
        # Order block: impulsive candle followed by pullback that doesn't go back inside
        for i in range(2, len(df) - 2):
            try:
                prev = df.iloc[i - 1]
                curr = df.iloc[i]
                next_candle = df.iloc[i + 1]
                
                # Bullish OB: strong up candle, then pullback down but stays above the low
                if (prev['close'] < prev['open'] and  # bearish
                    curr['close'] > curr['open'] and  # bullish
                    curr['low'] > prev['low'] and  # stays above prior low
                    next_candle['close'] < next_candle['open']):  # pullback begins
                    
                    strength = float(curr['volume'] / max(df['volume'].rolling(20).mean().iat[i] if len(df) >= 20 else df['volume'].mean(), 1))
                    obs.append({
                        'type': 'BULLISH_OB',
                        'price': float(prev['low']),
                        'timestamp': df.index[i],
                        'strength': min(1.0, strength),
                        'block_high': float(curr['high']),
                        'block_low': float(curr['low'])
                    })
                
                # Bearish OB: strong down candle, then pullback up but stays below the high
                if (prev['close'] > prev['open'] and  # bullish
                    curr['close'] < curr['open'] and  # bearish
                    curr['high'] < prev['high'] and  # stays below prior high
                    next_candle['close'] > next_candle['open']):  # pullback begins
                    
                    strength = float(curr['volume'] / max(df['volume'].rolling(20).mean().iat[i] if len(df) >= 20 else df['volume'].mean(), 1))
                    obs.append({
                        'type': 'BEARISH_OB',
                        'price': float(prev['high']),
                        'timestamp': df.index[i],
                        'strength': min(1.0, strength),
                        'block_high': float(curr['high']),
                        'block_low': float(curr['low'])
                    })
            except Exception:
                continue
        
        return obs

    def _analyze_internal_external(self, structures: List[MarketStructure]) -> Dict:
        if not structures:
            return {'type': 'UNKNOWN', 'confidence': 0.0}
        hh = sum(1 for s in structures if s.type == 'HH')
        ll = sum(1 for s in structures if s.type == 'LL')
        if hh > ll:
            return {'type': 'EXTERNAL_BULLISH', 'confidence': 0.8, 'hh_count': hh, 'll_count': ll}
        if ll > hh:
            return {'type': 'EXTERNAL_BEARISH', 'confidence': 0.8, 'hh_count': hh, 'll_count': ll}
        return {'type': 'INTERNAL_CONSOLIDATION', 'confidence': 0.6, 'hh_count': hh, 'll_count': ll}

    def _assess_structure_health(self, results: Dict) -> str:
        """
        Assess overall structure health for trade execution
        CLEAN = clear structure with recent BOS
        FORMING = in process of forming
        BROKEN = structure breaking down
        """
        bos_count = len(results.get('bos', []))
        choch_count = len(results.get('choch', []))
        displacement_count = len(results.get('displacement', []))
        
        if bos_count > 0 and displacement_count > 0:
            return 'CLEAN'
        elif bos_count > 0:
            return 'FORMING'
        elif choch_count > 0:
            return 'BREAKING'
        else:
            return 'UNKNOWN'
        
    def get_microstructure_score(self, df: pd.DataFrame) -> float:
        """
        Get aggregate microstructure score (0-1)
        Higher score = bullish microstructure (unfilled bullish FVGs, bullish displacement)
        Lower score = bearish microstructure (unfilled bearish FVGs, bearish displacement)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Score between 0 and 1
        """
        try:
            structure = self.analyze_structure(df)
            
            score = 0.5
            
            # Factor 1: Fair Value Gaps
            fvgs = structure.get('fvg', [])
            unfilled_bullish = [f for f in fvgs if f.direction == 'BULLISH' and not f.filled]
            unfilled_bearish = [f for f in fvgs if f.direction == 'BEARISH' and not f.filled]
            
            # Strong unfilled gaps = high probability
            if unfilled_bullish:
                score += min(0.2, len(unfilled_bullish) * 0.08)
            if unfilled_bearish:
                score -= min(0.2, len(unfilled_bearish) * 0.08)
            
            # Factor 2: Displacement moves
            displacement = structure.get('displacement', [])
            for disp in displacement:
                if disp.get('direction') == 'BULLISH':
                    score += 0.1 * disp.get('strength', 0.5)
                else:
                    score -= 0.1 * disp.get('strength', 0.5)
            
            # Factor 3: Liquidity zones
            liquidity = structure.get('liquidity', [])
            for zone in liquidity:
                if zone.type == 'LIQUIDITY_CLUSTER':
                    # Liquidity above = resistance, below = support
                    current_price = float(df['close'].iloc[-1])
                    if zone.price > current_price:
                        score -= 0.05 * zone.strength  # Resistance above = bearish
                    else:
                        score += 0.05 * zone.strength  # Support below = bullish
            
            # Factor 4: Order blocks
            order_blocks = structure.get('order_block', [])
            bullish_obs = [ob for ob in order_blocks if ob.get('type') == 'BULLISH_OB']
            bearish_obs = [ob for ob in order_blocks if ob.get('type') == 'BEARISH_OB']
            
            score += len(bullish_obs) * 0.05
            score -= len(bearish_obs) * 0.05
            
            # Factor 5: Structure health
            health = structure.get('structure_health', 'UNKNOWN')
            if health == 'CLEAN':
                score += 0.1
            elif health == 'BREAKING':
                score -= 0.1
            
            # Clamp to 0-1
            return max(0.01, min(0.99, score))
            
        except Exception as e:
            log.debug(f"Error calculating microstructure score: {e}")
            return 0.5
    
    def get_microstructure_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive microstructure summary for scoring layer
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Dictionary with microstructure summary
        """
        try:
            structure = self.analyze_structure(df)
            
            # Get FVG analysis
            fvgs = structure.get('fvg', [])
            unfilled_bullish = [f for f in fvgs if f.direction == 'BULLISH' and not f.filled]
            unfilled_bearish = [f for f in fvgs if f.direction == 'BEARISH' and not f.filled]
            filled_fvgs = structure.get('fvg_filled', [])
            
            # Get displacement analysis
            displacement = structure.get('displacement', [])
            recent_displacement = displacement[-3:] if len(displacement) > 3 else displacement
            
            # Get liquidity zones
            liquidity = structure.get('liquidity', [])
            
            # Get order blocks
            order_blocks = structure.get('order_block', [])
            
            # Calculate net bias
            net_bias = self.get_net_bias(df)
            
            # Determine direction
            if net_bias > 0.2:
                direction = 'BULLISH'
            elif net_bias < -0.2:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            return {
                'score': self.get_microstructure_score(df),
                'net_bias': net_bias,
                'direction': direction,
                'fvg': {
                    'total': len(fvgs),
                    'unfilled_bullish': len(unfilled_bullish),
                    'unfilled_bearish': len(unfilled_bearish),
                    'filled_count': len(filled_fvgs),
                    'has_unfilled_bullish': len(unfilled_bullish) > 0,
                    'has_unfilled_bearish': len(unfilled_bearish) > 0
                },
                'displacement': {
                    'count': len(displacement),
                    'bullish_count': sum(1 for d in displacement if d.get('direction') == 'BULLISH'),
                    'bearish_count': sum(1 for d in displacement if d.get('direction') == 'BEARISH'),
                    'recent_strength': recent_displacement[-1].get('strength', 0) if recent_displacement else 0,
                    'has_recent': len(displacement) > 0
                },
                'liquidity_zones': {
                    'count': len(liquidity),
                    'zones': [{'price': z.price, 'strength': z.strength} for z in liquidity[:5]]
                },
                'order_blocks': {
                    'count': len(order_blocks),
                    'bullish_count': sum(1 for ob in order_blocks if ob.get('type') == 'BULLISH_OB'),
                    'bearish_count': sum(1 for ob in order_blocks if ob.get('type') == 'BEARISH_OB')
                },
                'structure_health': structure.get('structure_health', 'UNKNOWN'),
                'composite_confidence': structure.get('composite_confidence', 0.5),
                'reasons': self._build_reasons(structure, unfilled_bullish, unfilled_bearish, displacement)
            }
            
        except Exception as e:
            log.debug(f"Error getting microstructure summary: {e}")
            return {
                'score': 0.5,
                'net_bias': 0,
                'direction': 'NEUTRAL',
                'reasons': ['Error in microstructure analysis']
            }
    
    def get_net_bias(self, df: pd.DataFrame) -> float:
        """
        Get net microstructure bias (-1 to 1)
        Positive = bullish (unfilled bullish FVGs, bullish displacement)
        Negative = bearish (unfilled bearish FVGs, bearish displacement)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Bias score between -1 and 1
        """
        try:
            structure = self.analyze_structure(df)
            
            bias = 0.0
            
            # FVG contribution
            fvgs = structure.get('fvg', [])
            for fvg in fvgs:
                if not fvg.filled:
                    if fvg.direction == 'BULLISH':
                        bias += 0.15
                    else:
                        bias -= 0.15
            
            # Displacement contribution
            displacement = structure.get('displacement', [])
            for disp in displacement[-3:]:  # Last 3 displacements
                if disp.get('direction') == 'BULLISH':
                    bias += 0.1 * disp.get('strength', 0.5)
                else:
                    bias -= 0.1 * disp.get('strength', 0.5)
            
            # Liquidity zone contribution
            liquidity = structure.get('liquidity', [])
            current_price = float(df['close'].iloc[-1])
            for zone in liquidity[:3]:
                if zone.type == 'LIQUIDITY_CLUSTER':
                    if zone.price > current_price:
                        bias -= 0.05 * zone.strength  # Resistance above
                    else:
                        bias += 0.05 * zone.strength  # Support below
            
            # Order block contribution
            order_blocks = structure.get('order_block', [])
            for ob in order_blocks[-3:]:
                if ob.get('type') == 'BULLISH_OB':
                    bias += 0.08 * ob.get('strength', 0.5)
                elif ob.get('type') == 'BEARISH_OB':
                    bias -= 0.08 * ob.get('strength', 0.5)
            
            # Clamp to -1 to 1
            return max(-1.0, min(1.0, bias))
            
        except Exception as e:
            log.debug(f"Error calculating net bias: {e}")
            return 0.0
    
    def has_unfilled_fvg(self, df: pd.DataFrame, direction: str = 'BULLISH') -> bool:
        """
        Check if there are unfilled Fair Value Gaps in a specific direction
        
        Args:
            df: OHLCV DataFrame
            direction: 'BULLISH' or 'BEARISH'
        
        Returns:
            True if unfilled FVGs exist
        """
        try:
            structure = self.analyze_structure(df)
            fvgs = structure.get('fvg', [])
            
            for fvg in fvgs:
                if fvg.direction == direction and not fvg.filled:
                    return True
            return False
            
        except Exception:
            return False
    
    def has_strong_displacement(self, df: pd.DataFrame, min_strength: float = 0.7) -> bool:
        """
        Check if there are strong displacement moves
        
        Args:
            df: OHLCV DataFrame
            min_strength: Minimum strength threshold
        
        Returns:
            True if strong displacement detected
        """
        try:
            structure = self.analyze_structure(df)
            displacement = structure.get('displacement', [])
            
            for disp in displacement[-5:]:
                if disp.get('strength', 0) >= min_strength:
                    return True
            return False
            
        except Exception:
            return False
    
    def get_fvg_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate overall FVG strength (0-1)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            FVG strength score
        """
        try:
            structure = self.analyze_structure(df)
            fvgs = structure.get('fvg', [])
            
            if not fvgs:
                return 0.0
            
            # Calculate strength based on number of unfilled FVGs and their size
            unfilled = [f for f in fvgs if not f.filled]
            if not unfilled:
                return 0.3
            
            # Count by direction
            bullish_count = sum(1 for f in unfilled if f.direction == 'BULLISH')
            bearish_count = sum(1 for f in unfilled if f.direction == 'BEARISH')
            
            # Calculate size-weighted strength
            total_size = sum(f.high - f.low for f in unfilled)
            avg_size = total_size / len(unfilled) if unfilled else 0
            
            # Normalize size (assume 1% gap is strong)
            size_strength = min(1.0, avg_size / 0.01)
            
            # Direction bias
            if bullish_count > bearish_count:
                direction_strength = bullish_count / len(unfilled)
                return 0.5 + (direction_strength * 0.3 + size_strength * 0.2)
            elif bearish_count > bullish_count:
                direction_strength = bearish_count / len(unfilled)
                return 0.5 - (direction_strength * 0.3 + size_strength * 0.2)
            else:
                return 0.5 + (size_strength * 0.1)
            
        except Exception as e:
            log.debug(f"Error calculating FVG strength: {e}")
            return 0.5
    
    def _build_reasons(self, structure: Dict, unfilled_bullish: List, 
                       unfilled_bearish: List, displacement: List) -> List[str]:
        """Build human-readable reasons for microstructure analysis"""
        reasons = []
        
        # FVG reasons
        if unfilled_bullish:
            reasons.append(f"{len(unfilled_bullish)} unfilled bullish FVGs")
        if unfilled_bearish:
            reasons.append(f"{len(unfilled_bearish)} unfilled bearish FVGs")
        
        # Displacement reasons
        if displacement:
            bullish_disps = [d for d in displacement if d.get('direction') == 'BULLISH']
            bearish_disps = [d for d in displacement if d.get('direction') == 'BEARISH']
            if bullish_disps:
                reasons.append(f"{len(bullish_disps)} bullish displacement moves")
            if bearish_disps:
                reasons.append(f"{len(bearish_disps)} bearish displacement moves")
        
        # Structure health
        health = structure.get('structure_health', 'UNKNOWN')
        if health == 'CLEAN':
            reasons.append("Clean market structure")
        elif health == 'BREAKING':
            reasons.append("Structure breaking down")
        
        # Liquidity zones
        liquidity = structure.get('liquidity', [])
        if liquidity:
            reasons.append(f"{len(liquidity)} liquidity zones detected")
        
        if not reasons:
            reasons.append("No significant microstructure signals")
        
        return reasons[:5]  # Top 5 reasons