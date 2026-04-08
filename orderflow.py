# orderflow.py - PHASE 3: Advanced Order Flow Intelligence with Scoring Methods
"""
Advanced Order Flow Intelligence - Enhanced Version
Detects CVD, volume delta, absorption, exhaustion, and provides scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from config import Config
from logger import log

EPSILON = 1e-10


@dataclass
class OrderFlowSignal:
    type: str  # 'ACCUMULATION', 'DISTRIBUTION', 'EXHAUSTION', 'ABSORPTION'
    strength: float  # 0-1
    direction: str  # 'BUY', 'SELL'
    confidence: float  # 0-1
    timestamp: pd.Timestamp
    details: Dict[str, Any]


class AdvancedOrderFlow:
    def __init__(self):
        self.lookback = 100
        self.min_volume_threshold = 0.8
        
        # Load thresholds from config
        self.absorption_volume_threshold = getattr(Config, 'SM_ABSORPTION_VOLUME_THRESHOLD', 1.5)
        self.exhaustion_volume_threshold = getattr(Config, 'SM_EXHAUSTION_VOLUME_THRESHOLD', 2.0)
        
        log.info("AdvancedOrderFlow enhanced with scoring methods")
    
    # ==================== CUMULATIVE VOLUME DELTA (CVD) ====================
    def detect_cvd_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        True CVD: (uptick volume) - (downtick volume)
        Proxy: use close vs open to estimate direction
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return {'status': 'insufficient_data'}
            
            # Approximate uptick/downtick based on close > open
            uptick_vol = df[df['close'] > df['open']]['volume'].sum()
            downtick_vol = df[df['close'] < df['open']]['volume'].sum()
            
            true_cvd = uptick_vol - downtick_vol
            cvd_sma = pd.Series(
                [uptick_vol - downtick_vol for _ in range(len(df))]
            ).rolling(14).mean().iloc[-1]
            
            # CVD divergence: price up but CVD down (bearish divergence)
            price_trend = df['close'].iloc[-1] - df['close'].iloc[-10]
            cvd_trend = true_cvd - (uptick_vol - downtick_vol)
            
            divergence = None
            if price_trend > 0 and true_cvd < cvd_sma:
                divergence = 'BEARISH'
            elif price_trend < 0 and true_cvd > cvd_sma:
                divergence = 'BULLISH'
            
            return {
                'true_cvd': float(true_cvd),
                'uptick_volume': float(uptick_vol),
                'downtick_volume': float(downtick_vol),
                'cvd_sma_14': float(cvd_sma),
                'divergence': divergence,
                'strength': float(abs(true_cvd) / max(uptick_vol + downtick_vol, 1))
            }
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== VOLUME IMBALANCE (DELTA) ====================
    def detect_volume_delta(self, df: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
        """
        Detect bars with significant buy/sell volume imbalance.
        Imbalance = (close - open) / (high - low) * volume
        """
        deltas = []
        try:
            if df is None or df.empty or len(df) < lookback:
                return deltas
            
            recent = df.iloc[-lookback:]
            vol_mean = recent['volume'].mean()
            
            for i in range(len(recent)):
                row = recent.iloc[i]
                
                # Normalize body position (0 = all down, 1 = all up)
                body_high = max(row['open'], row['close'])
                body_low = min(row['open'], row['close'])
                
                if row['high'] > row['low']:
                    body_position = (body_high - row['low']) / (row['high'] - row['low'])
                else:
                    body_position = 0.5
                
                # Volume-weighted delta
                delta = (body_position - 0.5) * row['volume']
                
                # Detect imbalance
                if row['volume'] > vol_mean * 1.5:
                    direction = 'BUY' if delta > 0 else 'SELL'
                    strength = min(1.0, abs(delta) / (vol_mean * 2))
                    
                    deltas.append({
                        'index': i,
                        'timestamp': recent.index[i],
                        'delta': float(delta),
                        'volume': float(row['volume']),
                        'direction': direction,
                        'strength': float(strength),
                        'body_position': float(body_position)
                    })
        except Exception:
            pass
        
        return deltas
    
    # ==================== ABSORPTION DETECTION ====================
    def detect_absorption(self, df: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
        """
        Absorption: large volume candle with small body (indecision/consolidation)
        Suggests strong accumulation/distribution without immediate directional move
        """
        absorption = []
        try:
            if df is None or df.empty or len(df) < lookback:
                return absorption
            
            recent = df.iloc[-lookback:]
            vol_mean = recent['volume'].mean()
            body_range_mean = (recent['high'] - recent['low']).mean()
            
            for i in range(len(recent)):
                row = recent.iloc[i]
                body = abs(row['close'] - row['open'])
                wick = (row['high'] - row['low']) - body
                
                # Absorption: high volume + small body relative to wicks
                if row['volume'] > vol_mean * self.absorption_volume_threshold and body < body_range_mean * 0.5:
                    absorption_ratio = row['volume'] / vol_mean
                    
                    absorption.append({
                        'index': i,
                        'timestamp': recent.index[i],
                        'volume': float(row['volume']),
                        'body': float(body),
                        'wick': float(wick),
                        'absorption_ratio': float(absorption_ratio),
                        'strength': min(1.0, (absorption_ratio - self.absorption_volume_threshold) / 3.0)
                    })
        except Exception:
            pass
        
        return absorption
    
    # ==================== EXHAUSTION VOLUME ====================
    def detect_exhaustion_volume(self, df: pd.DataFrame, lookback: int = 10) -> List[Dict[str, Any]]:
        """
        Exhaustion: extreme volume spike at end of directional move + reversal candle
        Signs smart money distribution/accumulation
        """
        exhaustion = []
        try:
            if df is None or df.empty or len(df) < lookback + 2:
                return exhaustion
            
            recent = df.iloc[-lookback:]
            vol_mean = recent['volume'].mean()
            vol_std = recent['volume'].std()
            
            for i in range(len(recent) - 1):
                curr = recent.iloc[i]
                next_candle = recent.iloc[i + 1]
                
                # Extreme volume
                if curr['volume'] > vol_mean + (vol_std * 2):
                    # Check if next candle reverses
                    curr_dir = 'UP' if curr['close'] > curr['open'] else 'DOWN'
                    next_dir = 'UP' if next_candle['close'] > next_candle['open'] else 'DOWN'
                    
                    if curr_dir != next_dir:
                        exhaustion.append({
                            'index': i,
                            'timestamp': recent.index[i],
                            'volume': float(curr['volume']),
                            'direction': curr_dir,
                            'reversal': next_dir,
                            'excess_volume_std': float((curr['volume'] - vol_mean) / vol_std),
                            'strength': min(1.0, (curr['volume'] - vol_mean) / (vol_std * 3))
                        })
        except Exception:
            pass
        
        return exhaustion
    
    # ==================== BID-ASK RATIO (PROXY) ====================
    def estimate_bid_ask_ratio(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        """
        Proxy for bid-ask ratio using price position in daily range + volume
        Real exchanges require L2 data; this estimates from OHLCV
        """
        try:
            if df is None or df.empty or len(df) < lookback:
                return {'status': 'insufficient_data'}
            
            recent = df.iloc[-lookback:]
            
            # Estimate buy/sell pressure
            buy_pressure = recent[recent['close'] > recent['open']]['volume'].sum()
            sell_pressure = recent[recent['close'] < recent['open']]['volume'].sum()
            
            total_vol = buy_pressure + sell_pressure
            if total_vol <= 0:
                return {'bid_ask_ratio': 1.0, 'bias': 'NEUTRAL'}
            
            ratio = buy_pressure / total_vol
            
            if ratio > 0.6:
                bias = 'BUY'
            elif ratio < 0.4:
                bias = 'SELL'
            else:
                bias = 'NEUTRAL'
            
            return {
                'bid_ask_ratio': float(ratio),
                'buy_pressure': float(buy_pressure),
                'sell_pressure': float(sell_pressure),
                'bias': bias,
                'strength': float(abs(ratio - 0.5) * 2)
            }
        except Exception:
            return {'status': 'error'}
    
    # ==================== ICEBERG DETECTOR (PROXY) ====================
    def detect_iceberg_orders(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict[str, Any]]:
        """
        Proxy iceberg detection: repeated small volume clusters at support/resistance
        Real detection needs L2 orderbook
        """
        icebergs = []
        try:
            if df is None or df.empty or len(df) < lookback:
                return icebergs
            
            recent = df.iloc[-lookback:]
            vol_mean = recent['volume'].mean()
            vol_std = recent['volume'].std()
            
            # Find price levels that revisit multiple times with small volume
            price_visits = {}
            vol_threshold = vol_mean * 0.7  # smaller than average
            
            for i in range(len(recent)):
                row = recent.iloc[i]
                
                if row['volume'] < vol_threshold:
                    # Round price to find clusters
                    price_level = round(row['high'], -int(np.floor(np.log10(row['high']))) + 3)
                    
                    if price_level not in price_visits:
                        price_visits[price_level] = []
                    price_visits[price_level].append({
                        'index': i,
                        'volume': row['volume'],
                        'timestamp': recent.index[i]
                    })
            
            # Flag levels with 3+ visits
            for price_level, visits in price_visits.items():
                if len(visits) >= 3:
                    total_vol = sum(v['volume'] for v in visits)
                    icebergs.append({
                        'price_level': float(price_level),
                        'visits': len(visits),
                        'total_volume': float(total_vol),
                        'avg_volume': float(total_vol / len(visits)),
                        'timestamps': [v['timestamp'] for v in visits],
                        'strength': min(1.0, len(visits) / 10)
                    })
        except Exception:
            pass
        
        return icebergs
    
    # ==================== CONSOLIDATED SIGNAL ====================
    def generate_orderflow_signal(self, df: pd.DataFrame) -> OrderFlowSignal:
        """
        Aggregate all orderflow signals into single strong/weak signal
        """
        try:
            cvd = self.detect_cvd_signal(df)
            deltas = self.detect_volume_delta(df)
            absorption = self.detect_absorption(df)
            exhaustion = self.detect_exhaustion_volume(df)
            bid_ask = self.estimate_bid_ask_ratio(df)
            
            # Score composite
            score = 0.5
            
            if cvd.get('divergence') == 'BEARISH':
                score -= 0.1
            elif cvd.get('divergence') == 'BULLISH':
                score += 0.1
            
            if deltas and deltas[-1]['direction'] == 'BUY':
                score += 0.1 * deltas[-1]['strength']
            elif deltas and deltas[-1]['direction'] == 'SELL':
                score -= 0.1 * deltas[-1]['strength']
            
            if bid_ask.get('bias') == 'BUY':
                score += 0.05 * bid_ask.get('strength', 0)
            elif bid_ask.get('bias') == 'SELL':
                score -= 0.05 * bid_ask.get('strength', 0)
            
            score = max(0.0, min(1.0, score))
            
            direction = 'BUY' if score > 0.55 else ('SELL' if score < 0.45 else 'NEUTRAL')
            
            return OrderFlowSignal(
                type='COMPOSITE',
                strength=abs(score - 0.5) * 2,
                direction=direction,
                confidence=abs(score - 0.5),
                timestamp=df.index[-1],
                details={
                    'cvd': cvd,
                    'latest_delta': deltas[-1] if deltas else None,
                    'absorption_count': len(absorption),
                    'exhaustion_count': len(exhaustion),
                    'bid_ask': bid_ask
                }
            )
        except Exception as e:
            return OrderFlowSignal(
                type='ERROR',
                strength=0.0,
                direction='NEUTRAL',
                confidence=0.0,
                timestamp=df.index[-1] if not df.empty else pd.Timestamp.now(),
                details={'error': str(e)}
            )
    
    # ==================== NEW SCORING METHODS ====================
    
    def get_orderflow_score(self, df: pd.DataFrame) -> float:
        """
        Get aggregate order flow score (0-1)
        Higher score = bullish order flow (buying pressure)
        Lower score = bearish order flow (selling pressure)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Score between 0 and 1
        """
        try:
            score = 0.5
            
            # Factor 1: CVD divergence
            cvd = self.detect_cvd_signal(df)
            if cvd.get('divergence') == 'BULLISH':
                score += 0.15
            elif cvd.get('divergence') == 'BEARISH':
                score -= 0.15
            
            # Factor 2: Latest volume delta
            deltas = self.detect_volume_delta(df)
            if deltas:
                latest_delta = deltas[-1]
                if latest_delta.get('direction') == 'BUY':
                    score += 0.1 * latest_delta.get('strength', 0.5)
                else:
                    score -= 0.1 * latest_delta.get('strength', 0.5)
            
            # Factor 3: Absorption (bullish if detected)
            absorption = self.detect_absorption(df)
            for a in absorption:
                # Absorption on down candle = bullish accumulation
                # Absorption on up candle = bearish distribution
                if a.get('absorption_ratio', 0) > self.absorption_volume_threshold:
                    # Check direction context
                    price = df['close'].iloc[a.get('index', -1)] if a.get('index', -1) >= 0 else None
                    if price and a.get('index', 0) > 0:
                        prev_price = df['close'].iloc[a['index'] - 1] if a['index'] > 0 else price
                        if price < prev_price:
                            score += 0.1 * a.get('strength', 0.5)  # Downside absorption = bullish
                        else:
                            score -= 0.1 * a.get('strength', 0.5)  # Upside absorption = bearish
            
            # Factor 4: Exhaustion (bearish if detected at top, bullish if at bottom)
            exhaustion = self.detect_exhaustion_volume(df)
            for e in exhaustion:
                if e.get('direction') == 'UP' and e.get('reversal') == 'DOWN':
                    score -= 0.15 * e.get('strength', 0.5)  # Up exhaustion = bearish reversal
                elif e.get('direction') == 'DOWN' and e.get('reversal') == 'UP':
                    score += 0.15 * e.get('strength', 0.5)  # Down exhaustion = bullish reversal
            
            # Factor 5: Bid/Ask ratio
            bid_ask = self.estimate_bid_ask_ratio(df)
            if bid_ask.get('bias') == 'BUY':
                score += 0.05 * bid_ask.get('strength', 0)
            elif bid_ask.get('bias') == 'SELL':
                score -= 0.05 * bid_ask.get('strength', 0)
            
            # Clamp to 0-1
            return max(0.01, min(0.99, score))
            
        except Exception as e:
            log.debug(f"Error calculating order flow score: {e}")
            return 0.5
    
    def get_orderflow_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive order flow summary for scoring layer
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Dictionary with order flow summary
        """
        try:
            # Get all detection results
            cvd = self.detect_cvd_signal(df)
            deltas = self.detect_volume_delta(df)
            absorption = self.detect_absorption(df)
            exhaustion = self.detect_exhaustion_volume(df)
            bid_ask = self.estimate_bid_ask_ratio(df)
            icebergs = self.detect_iceberg_orders(df)
            
            # Calculate net bias
            net_bias = self.get_net_bias(df)
            
            # Determine direction
            if net_bias > 0.2:
                direction = 'BULLISH'
            elif net_bias < -0.2:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            # Get latest delta
            latest_delta = deltas[-1] if deltas else None
            
            return {
                'score': self.get_orderflow_score(df),
                'net_bias': net_bias,
                'direction': direction,
                'cvd': {
                    'value': cvd.get('true_cvd', 0),
                    'divergence': cvd.get('divergence'),
                    'strength': cvd.get('strength', 0)
                },
                'latest_delta': {
                    'direction': latest_delta.get('direction') if latest_delta else None,
                    'strength': latest_delta.get('strength', 0) if latest_delta else 0,
                    'delta': latest_delta.get('delta', 0) if latest_delta else 0
                } if latest_delta else None,
                'absorption_count': len(absorption),
                'absorption_detected': len(absorption) > 0,
                'exhaustion_count': len(exhaustion),
                'exhaustion_detected': len(exhaustion) > 0,
                'bid_ask_ratio': bid_ask.get('bid_ask_ratio', 1.0),
                'bid_ask_bias': bid_ask.get('bias', 'NEUTRAL'),
                'iceberg_count': len(icebergs),
                'reasons': self._build_reasons(cvd, deltas, absorption, exhaustion, bid_ask)
            }
            
        except Exception as e:
            log.debug(f"Error getting order flow summary: {e}")
            return {
                'score': 0.5,
                'net_bias': 0,
                'direction': 'NEUTRAL',
                'reasons': ['Error in order flow analysis']
            }
    
    def get_net_bias(self, df: pd.DataFrame) -> float:
        """
        Get net order flow bias (-1 to 1)
        Positive = bullish (buying pressure)
        Negative = bearish (selling pressure)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Bias score between -1 and 1
        """
        try:
            bias = 0.0
            
            # CVD contribution
            cvd = self.detect_cvd_signal(df)
            if cvd.get('divergence') == 'BULLISH':
                bias += 0.2
            elif cvd.get('divergence') == 'BEARISH':
                bias -= 0.2
            
            # Delta contribution
            deltas = self.detect_volume_delta(df)
            if deltas:
                latest_delta = deltas[-1]
                if latest_delta.get('direction') == 'BUY':
                    bias += 0.15 * latest_delta.get('strength', 0.5)
                else:
                    bias -= 0.15 * latest_delta.get('strength', 0.5)
            
            # Absorption contribution
            absorption = self.detect_absorption(df)
            for a in absorption:
                price = df['close'].iloc[a.get('index', -1)] if a.get('index', -1) >= 0 else None
                if price and a.get('index', 0) > 0:
                    prev_price = df['close'].iloc[a['index'] - 1] if a['index'] > 0 else price
                    if price < prev_price:
                        bias += 0.1 * a.get('strength', 0.5)  # Downside absorption = bullish
                    else:
                        bias -= 0.1 * a.get('strength', 0.5)  # Upside absorption = bearish
            
            # Exhaustion contribution
            exhaustion = self.detect_exhaustion_volume(df)
            for e in exhaustion:
                if e.get('direction') == 'UP' and e.get('reversal') == 'DOWN':
                    bias -= 0.15 * e.get('strength', 0.5)  # Up exhaustion = bearish reversal
                elif e.get('direction') == 'DOWN' and e.get('reversal') == 'UP':
                    bias += 0.15 * e.get('strength', 0.5)  # Down exhaustion = bullish reversal
            
            # Bid/Ask contribution
            bid_ask = self.estimate_bid_ask_ratio(df)
            if bid_ask.get('bias') == 'BUY':
                bias += 0.1 * bid_ask.get('strength', 0)
            elif bid_ask.get('bias') == 'SELL':
                bias -= 0.1 * bid_ask.get('strength', 0)
            
            # Clamp to -1 to 1
            return max(-1.0, min(1.0, bias))
            
        except Exception as e:
            log.debug(f"Error calculating net bias: {e}")
            return 0.0
    
    def has_absorption(self, df: pd.DataFrame, min_strength: float = 0.5) -> bool:
        """
        Check if there is absorption pattern (accumulation)
        
        Args:
            df: OHLCV DataFrame
            min_strength: Minimum strength threshold
        
        Returns:
            True if absorption detected
        """
        try:
            absorption = self.detect_absorption(df)
            for a in absorption:
                if a.get('strength', 0) >= min_strength:
                    return True
            return False
        except Exception:
            return False
    
    def has_exhaustion(self, df: pd.DataFrame, min_strength: float = 0.5) -> bool:
        """
        Check if there is exhaustion pattern (reversal)
        
        Args:
            df: OHLCV DataFrame
            min_strength: Minimum strength threshold
        
        Returns:
            True if exhaustion detected
        """
        try:
            exhaustion = self.detect_exhaustion_volume(df)
            for e in exhaustion:
                if e.get('strength', 0) >= min_strength:
                    return True
            return False
        except Exception:
            return False
    
    def _build_reasons(self, cvd: Dict, deltas: List, absorption: List, 
                       exhaustion: List, bid_ask: Dict) -> List[str]:
        """Build human-readable reasons for order flow analysis"""
        reasons = []
        
        # CVD divergence
        divergence = cvd.get('divergence')
        if divergence:
            reasons.append(f"CVD {divergence} divergence detected")
        
        # Latest delta
        if deltas:
            latest = deltas[-1]
            reasons.append(f"Latest delta: {latest.get('direction', 'NEUTRAL')} ({latest.get('strength', 0):.0%})")
        
        # Absorption
        if absorption:
            reasons.append(f"{len(absorption)} absorption pattern(s) detected")
        
        # Exhaustion
        if exhaustion:
            reasons.append(f"{len(exhaustion)} exhaustion pattern(s) detected")
        
        # Bid/Ask
        bias = bid_ask.get('bias', 'NEUTRAL')
        if bias != 'NEUTRAL':
            ratio = bid_ask.get('bid_ask_ratio', 0.5)
            reasons.append(f"Bid/Ask ratio: {ratio:.2f} ({bias} bias)")
        
        if not reasons:
            reasons.append("No significant order flow signals detected")
        
        return reasons[:5]  # Top 5 reasons


# ==================== LEGACY FUNCTIONS ====================
def detect_order_blocks(df: pd.DataFrame, lookback: int = 100) -> List[Dict[str,Any]]:
    """Legacy function"""
    obs = []
    try:
        if df is None or df.empty or len(df) < 10:
            return obs
        for i in range(2, min(len(df)-2, lookback)):
            prev = df.iloc[-i-1]
            curr = df.iloc[-i]
            if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['low'] > prev['low']:
                strength = float(curr['volume'] / max(df['volume'].rolling(20).mean().iloc[-i] if len(df)>=20 else df['volume'].mean(), 1))
                obs.append({'type':'BULLISH_OB','price': float(prev['low']), 'timestamp': df.index[-i].isoformat(), 'strength': min(1.0, strength)})
            if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['high'] < prev['high']:
                strength = float(curr['volume'] / max(df['volume'].rolling(20).mean().iloc[-i] if len(df)>=20 else df['volume'].mean(), 1))
                obs.append({'type':'BEARISH_OB','price': float(prev['high']), 'timestamp': df.index[-i].isoformat(), 'strength': min(1.0, strength)})
        return obs
    except Exception:
        return obs

def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 60, threshold: float = 0.002) -> List[Dict[str,Any]]:
    """Legacy function"""
    fvgs = []
    try:
        if df is None or df.empty or len(df) < 3:
            return fvgs
        recent = df.iloc[-lookback:]
        for i in range(1, len(recent)):
            c1 = recent.iloc[i-1]; c2 = recent.iloc[i]
            if c2['low'] > c1['high'] * (1+threshold):
                fvgs.append({'type':'FVG_BULL','start': recent.index[i-1].isoformat(), 'end': recent.index[i].isoformat(), 'high': float(c2['low']), 'low': float(c1['high'])})
            if c2['high'] < c1['low'] * (1-threshold):
                fvgs.append({'type':'FVG_BEAR','start': recent.index[i-1].isoformat(), 'end': recent.index[i].isoformat(), 'high': float(c1['low']), 'low': float(c2['high'])})
        return fvgs
    except Exception:
        return fvgs

def detect_volume_imbalance(df: pd.DataFrame, window: int = 20, threshold: float = 1.6) -> List[Dict[str,Any]]:
    """Legacy function"""
    results = []
    try:
        if df is None or df.empty or len(df) < window+1:
            return results
        vol_sma = df['volume'].rolling(window).mean()
        for i in range(len(df)-window, len(df)):
            v = df['volume'].iat[i]
            if v > vol_sma.iat[i] * threshold:
                direction = 'BULL' if df['close'].iat[i] > df['open'].iat[i] else 'BEAR'
                results.append({'index': df.index[i].isoformat(), 'volume': float(v), 'direction': direction})
        return results
    except Exception:
        return results