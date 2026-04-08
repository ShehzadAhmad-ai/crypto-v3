# liquidity.py - ENHANCED with Scoring Methods
"""
Liquidity Intelligence - Enhanced Version
Detects liquidity sweeps, inducements, sweep failures, and provides scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from config import Config
from logger import log

EPSILON = 1e-10


@dataclass
class LiquidityEvent:
    type: str  # 'SWEEP', 'POOL', 'VOID', 'HUNT', 'INDUCEMENT', 'SWEEP_FAILURE'
    price: float
    strength: float
    timestamp: pd.Timestamp
    direction: str
    details: Dict[str, Any]


class LiquidityIntelligence:
    def __init__(self):
        self.tolerance = 0.0015  # 0.15%
        self.atr_mult_threshold = 2.0
        
        # Load thresholds from config
        self.sweep_strength_threshold = getattr(Config, 'SM_SWEEP_STRENGTH_THRESHOLD', 0.6)
        self.enable_inducement = getattr(Config, 'SM_ENABLE_INDUCEMENT_DETECTION', True)
        self.enable_sweep_failure = getattr(Config, 'SM_ENABLE_SWEEP_FAILURE_DETECTION', True)
        
        log.info("LiquidityIntelligence enhanced with scoring methods")
    
    # ==================== EQUAL HIGH/LOW DETECTION ====================
    def detect_equal_highs_lows(self, df: pd.DataFrame, lookback: int = 100, precision: float = 0.002) -> Dict[str, List[Dict]]:
        """
        Detect price levels that have been touched multiple times (equal highs/lows).
        These are institutional resting liquidity levels.
        """
        try:
            if df is None or df.empty or len(df) < lookback:
                return {'equal_highs': [], 'equal_lows': [], 'strength_map': {}}
            
            recent = df.iloc[-lookback:]
            current_price = float(df['close'].iloc[-1])
            
            # Cluster highs and lows
            highs = recent['high'].values
            lows = recent['low'].values
            
            high_levels = {}
            low_levels = {}
            
            tolerance_pct = current_price * precision
            
            for high in highs:
                high = float(high)
                found = False
                for level in high_levels:
                    if abs(high - level) <= tolerance_pct:
                        high_levels[level]['count'] += 1
                        high_levels[level]['prices'].append(high)
                        found = True
                        break
                if not found:
                    high_levels[high] = {'count': 1, 'prices': [high], 'touches': []}
            
            for low in lows:
                low = float(low)
                found = False
                for level in low_levels:
                    if abs(low - level) <= tolerance_pct:
                        low_levels[level]['count'] += 1
                        low_levels[level]['prices'].append(low)
                        found = True
                        break
                if not found:
                    low_levels[low] = {'count': 1, 'prices': [low], 'touches': []}
            
            # Filter to significant levels (3+ touches)
            equal_highs = []
            for level, data in sorted(high_levels.items(), key=lambda x: x[1]['count'], reverse=True):
                if data['count'] >= 3:
                    equal_highs.append({
                        'price': float(level),
                        'touches': data['count'],
                        'strength': min(1.0, data['count'] / 10),
                        'distance_from_current': abs(level - current_price) / current_price
                    })
            
            equal_lows = []
            for level, data in sorted(low_levels.items(), key=lambda x: x[1]['count'], reverse=True):
                if data['count'] >= 3:
                    equal_lows.append({
                        'price': float(level),
                        'touches': data['count'],
                        'strength': min(1.0, data['count'] / 10),
                        'distance_from_current': abs(level - current_price) / current_price
                    })
            
            return {
                'equal_highs': equal_highs,
                'equal_lows': equal_lows,
                'strength_map': {'high_count': len(equal_highs), 'low_count': len(equal_lows)}
            }
        except Exception as e:
            return {'equal_highs': [], 'equal_lows': [], 'error': str(e)}
    
    # ==================== STOP HUNT PROBABILITY ====================
    def detect_stop_hunt_probability(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """
        Stop hunts: extreme wicks beyond ATR + volume spike + quick reversal
        """
        try:
            if df is None or df.empty or len(df) < lookback:
                return {'probability': 0.0, 'signals': []}
            
            recent = df.iloc[-lookback:]
            atr = float(df['atr'].iloc[-1]) if 'atr' in df and df['atr'].iloc[-1] > 0 else float(df['close'].iloc[-1]) * 0.02
            vol_mean = recent['volume'].mean()
            
            stop_hunt_signals = []
            
            for i in range(len(recent) - 1):
                row = recent.iloc[i]
                next_row = recent.iloc[i + 1]
                
                upper_wick = row['high'] - max(row['open'], row['close'])
                lower_wick = min(row['open'], row['close']) - row['low']
                
                # Extreme wick
                if upper_wick > atr * self.atr_mult_threshold:
                    # Check volume spike
                    if row['volume'] > vol_mean * 1.5:
                        # Check reversal
                        if next_row['close'] < row['open']:
                            stop_hunt_signals.append({
                                'type': 'UPSIDE_HUNT',
                                'index': i,
                                'wick_size': float(upper_wick),
                                'wick_atr_multiple': float(upper_wick / atr),
                                'volume_ratio': float(row['volume'] / vol_mean),
                                'reversal_close': float(next_row['close']),
                                'strength': min(1.0, (upper_wick / atr - 2.0) / 3.0)
                            })
                
                if lower_wick > atr * self.atr_mult_threshold:
                    # Check volume spike
                    if row['volume'] > vol_mean * 1.5:
                        # Check reversal
                        if next_row['close'] > row['open']:
                            stop_hunt_signals.append({
                                'type': 'DOWNSIDE_HUNT',
                                'index': i,
                                'wick_size': float(lower_wick),
                                'wick_atr_multiple': float(lower_wick / atr),
                                'volume_ratio': float(row['volume'] / vol_mean),
                                'reversal_close': float(next_row['close']),
                                'strength': min(1.0, (lower_wick / atr - 2.0) / 3.0)
                            })
            
            probability = min(1.0, len(stop_hunt_signals) / 10)
            
            return {
                'probability': float(probability),
                'signals': stop_hunt_signals,
                'signal_count': len(stop_hunt_signals),
                'assessment': 'HIGH' if probability > 0.6 else ('MEDIUM' if probability > 0.3 else 'LOW')
            }
        except Exception as e:
            return {'probability': 0.0, 'error': str(e)}
    
    # ==================== LIQUIDITY VOID ====================
    def detect_liquidity_voids(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict[str, Any]]:
        """
        Liquidity voids: gaps in price action where no volume traded
        Often setup for impulsive moves
        """
        voids = []
        try:
            if df is None or df.empty or len(df) < lookback:
                return voids
            
            recent = df.iloc[-lookback:]
            vol_mean = recent['volume'].mean()
            
            for i in range(len(recent)):
                row = recent.iloc[i]
                
                # Low volume candle in trend
                if row['volume'] < vol_mean * 0.5:
                    # Check if in trend (EMA ordered)
                    if i >= 10:
                        trend = 'BULLISH' if row['close'] > recent.iloc[i-10]['close'] else 'BEARISH'
                        
                        voids.append({
                            'index': i,
                            'timestamp': recent.index[i],
                            'volume': float(row['volume']),
                            'volume_ratio': float(row['volume'] / vol_mean),
                            'trend': trend,
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'void_range': float(row['high'] - row['low']),
                            'strength': 1.0 - (row['volume'] / vol_mean)
                        })
        except Exception:
            pass
        
        return voids
    
    # ==================== RESTING LIQUIDITY ESTIMATOR ====================
    def estimate_resting_liquidity(self, df: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
        """
        Estimate where institutional orders are resting.
        Uses historical price clustering + volume concentration.
        """
        try:
            if df is None or df.empty or len(df) < lookback:
                return {'status': 'insufficient_data'}
            
            recent = df.iloc[-lookback:]
            current_price = float(df['close'].iloc[-1])
            
            # Price level clustering
            prices = recent['close'].values
            price_bins = np.histogram(prices, bins=50)[0]
            bin_edges = np.histogram(prices, bins=50)[1]
            
            # Volume concentration at levels
            liquidity_levels = []
            for i in range(len(price_bins)):
                if price_bins[i] > np.mean(price_bins) * 1.5:  # above average clustering
                    level_price = (bin_edges[i] + bin_edges[i+1]) / 2
                    
                    # Get volume at that level
                    level_volume = recent[
                        (recent['close'] >= bin_edges[i]) &
                        (recent['close'] <= bin_edges[i+1])
                    ]['volume'].sum()
                    
                    liquidity_levels.append({
                        'price': float(level_price),
                        'price_touches': int(price_bins[i]),
                        'volume_resting': float(level_volume),
                        'distance_pct': abs(level_price - current_price) / current_price * 100,
                        'direction': 'ABOVE' if level_price > current_price else ('BELOW' if level_price < current_price else 'AT')
                    })
            
            # Sort by distance
            liquidity_levels.sort(key=lambda x: x['distance_pct'])
            
            return {
                'next_above': liquidity_levels[0] if liquidity_levels and liquidity_levels[0]['direction'] == 'ABOVE' else None,
                'next_below': liquidity_levels[0] if liquidity_levels and liquidity_levels[0]['direction'] == 'BELOW' else None,
                'all_levels': liquidity_levels[:10]
            }
        except Exception as e:
            return {'error': str(e)}
    
    # ==================== INDUCEMENT DETECTOR ====================
    def detect_inducement(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Inducement: false break/breakout that draws in retail traders before reversal
        Pattern: break of structure + volume + quick rejection + reversal
        """
        inducements = []
        try:
            if df is None or df.empty or len(df) < 20:
                return inducements
            
            recent = df.iloc[-20:]
            
            for i in range(2, len(recent) - 2):
                prev_high = recent['high'].iloc[:i].max()
                prev_low = recent['low'].iloc[:i].min()
                
                curr = recent.iloc[i]
                next1 = recent.iloc[i+1]
                next2 = recent.iloc[i+2]
                
                # Upside inducement: break above previous high but reverse down
                if curr['high'] > prev_high and next1['close'] < curr['close'] and next2['close'] < next1['close']:
                    if curr['volume'] > recent['volume'].mean() * 1.5:
                        inducements.append({
                            'type': 'UPSIDE_INDUCEMENT',
                            'index': i,
                            'break_level': float(prev_high),
                            'fake_high': float(curr['high']),
                            'rejection_close': float(next1['close']),
                            'volume': float(curr['volume']),
                            'strength': min(1.0, (curr['high'] - prev_high) / prev_high)
                        })
                
                # Downside inducement: break below previous low but reverse up
                if curr['low'] < prev_low and next1['close'] > curr['close'] and next2['close'] > next1['close']:
                    if curr['volume'] > recent['volume'].mean() * 1.5:
                        inducements.append({
                            'type': 'DOWNSIDE_INDUCEMENT',
                            'index': i,
                            'break_level': float(prev_low),
                            'fake_low': float(curr['low']),
                            'rejection_close': float(next1['close']),
                            'volume': float(curr['volume']),
                            'strength': min(1.0, (prev_low - curr['low']) / prev_low)
                        })
        except Exception:
            pass
        
        return inducements
    
    # ==================== SWEEP FAILURE DETECTOR ====================
    def detect_sweep_failure(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Sweep failure: price sweeps a level but fails to sustain direction
        Often leads to strong reversal
        """
        failures = []
        try:
            if df is None or df.empty or len(df) < 15:
                return failures
            
            recent = df.iloc[-15:]
            atr = float(df['atr'].iloc[-1]) if 'atr' in df else float(df['close'].iloc[-1]) * 0.02
            
            for i in range(2, len(recent) - 2):
                prev_low = recent['low'].iloc[:i].min()
                prev_high = recent['high'].iloc[:i].max()
                
                curr = recent.iloc[i]
                next1 = recent.iloc[i+1]
                
                # Bullish sweep failure: price dips below previous low but closes above, then reverses down
                if curr['low'] < prev_low and curr['close'] > prev_low and next1['close'] < next1['open']:
                    failures.append({
                        'type': 'BULLISH_SWEEP_FAILURE',
                        'index': i,
                        'sweep_low': float(curr['low']),
                        'previous_low': float(prev_low),
                        'sweep_range': float(prev_low - curr['low']),
                        'reversal_close': float(next1['close']),
                        'strength': min(1.0, (prev_low - curr['low']) / atr)
                    })
                
                # Bearish sweep failure: price spikes above previous high but closes below, then reverses up
                if curr['high'] > prev_high and curr['close'] < prev_high and next1['close'] > next1['open']:
                    failures.append({
                        'type': 'BEARISH_SWEEP_FAILURE',
                        'index': i,
                        'sweep_high': float(curr['high']),
                        'previous_high': float(prev_high),
                        'sweep_range': float(curr['high'] - prev_high),
                        'reversal_close': float(next1['close']),
                        'strength': min(1.0, (curr['high'] - prev_high) / atr)
                    })
        except Exception:
            pass
        
        return failures

    # ==================== GET LIQUIDITY SWEEPS FOR STRATEGIES ====================
    def get_liquidity_sweeps_for_strategies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Returns liquidity sweeps in the format expected by smart money strategies
        Used by smart_money_pipeline.py to pass to strategies
        """
        sweeps = []
        try:
            if df is None or df.empty or len(df) < 30:
                return sweeps
            
            # Get stop hunt signals which contain sweep information
            stop_hunt = self.detect_stop_hunt_probability(df)
            if 'signals' in stop_hunt:
                for signal in stop_hunt['signals']:
                    if signal['type'] == 'UPSIDE_HUNT':
                        sweeps.append({
                            'type': 'UP_SWEEP',
                            'strength': signal.get('strength', 0.5),
                            'price': signal.get('wick_size', 0),
                            'timestamp': df.index[-len(stop_hunt['signals']) + signal.get('index', 0)] if 'index' in signal else None
                        })
                    elif signal['type'] == 'DOWNSIDE_HUNT':
                        sweeps.append({
                            'type': 'DOWN_SWEEP',
                            'strength': signal.get('strength', 0.5),
                            'price': signal.get('wick_size', 0),
                            'timestamp': df.index[-len(stop_hunt['signals']) + signal.get('index', 0)] if 'index' in signal else None
                        })
            
            # Also get sweep failures
            failures = self.detect_sweep_failure(df)
            for failure in failures:
                if failure['type'] == 'BULLISH_SWEEP_FAILURE':
                    sweeps.append({
                        'type': 'DOWN_SWEEP_FAILURE',
                        'strength': failure.get('strength', 0.5),
                        'price': failure.get('sweep_low', 0),
                        'swept_level': failure.get('previous_low', 0)
                    })
                elif failure['type'] == 'BEARISH_SWEEP_FAILURE':
                    sweeps.append({
                        'type': 'UP_SWEEP_FAILURE',
                        'strength': failure.get('strength', 0.5),
                        'price': failure.get('sweep_high', 0),
                        'swept_level': failure.get('previous_high', 0)
                    })
            
            return sweeps
        except Exception as e:
            return []

    # ==================== LIQUIDITY SWEEP BOOLEAN FLAGS ====================
    def has_recent_down_sweep(self, df: pd.DataFrame, lookback_bars: int = 10) -> bool:
        """Check if there was a downside liquidity sweep in recent bars"""
        try:
            sweeps = self.get_liquidity_sweeps_for_strategies(df)
            for sweep in sweeps[-lookback_bars:]:
                if sweep['type'] in ['DOWN_SWEEP', 'DOWN_SWEEP_FAILURE']:
                    return True
            return False
        except Exception:
            return False
    
    def has_recent_up_sweep(self, df: pd.DataFrame, lookback_bars: int = 10) -> bool:
        """Check if there was an upside liquidity sweep in recent bars"""
        try:
            sweeps = self.get_liquidity_sweeps_for_strategies(df)
            for sweep in sweeps[-lookback_bars:]:
                if sweep['type'] in ['UP_SWEEP', 'UP_SWEEP_FAILURE']:
                    return True
            return False
        except Exception:
            return False
    
    # ==================== NEW SCORING METHODS ====================
    
    def get_liquidity_score(self, df: pd.DataFrame) -> float:
        """
        Get aggregate liquidity score (0-1)
        Higher score = bullish liquidity (sweeps below support, etc.)
        Lower score = bearish liquidity (sweeps above resistance, etc.)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Score between 0 and 1
        """
        try:
            score = 0.5
            
            # Factor 1: Stop hunt probability
            stop_hunt = self.detect_stop_hunt_probability(df)
            if stop_hunt.get('probability', 0) > 0.6:
                # Stop hunts often lead to reversals
                # Check direction of the hunt
                signals = stop_hunt.get('signals', [])
                if signals:
                    last_signal = signals[-1]
                    if last_signal.get('type') == 'DOWNSIDE_HUNT':
                        score += 0.15  # Downside hunt = bullish reversal
                    elif last_signal.get('type') == 'UPSIDE_HUNT':
                        score -= 0.15  # Upside hunt = bearish reversal
            
            # Factor 2: Inducements (false breakouts)
            if self.enable_inducement:
                inducements = self.detect_inducement(df)
                for inducement in inducements:
                    if inducement.get('type') == 'DOWNSIDE_INDUCEMENT':
                        score += 0.1 * inducement.get('strength', 0.5)
                    elif inducement.get('type') == 'UPSIDE_INDUCEMENT':
                        score -= 0.1 * inducement.get('strength', 0.5)
            
            # Factor 3: Sweep failures
            if self.enable_sweep_failure:
                failures = self.detect_sweep_failure(df)
                for failure in failures:
                    if failure.get('type') == 'BULLISH_SWEEP_FAILURE':
                        score += 0.15 * failure.get('strength', 0.5)
                    elif failure.get('type') == 'BEARISH_SWEEP_FAILURE':
                        score -= 0.15 * failure.get('strength', 0.5)
            
            # Factor 4: Resting liquidity levels
            resting = self.estimate_resting_liquidity(df)
            current_price = float(df['close'].iloc[-1])
            
            next_above = resting.get('next_above')
            next_below = resting.get('next_below')
            
            if next_below:
                distance = next_below.get('distance_pct', 100) / 100
                if distance < 0.01:  # Very close to liquidity below
                    score += 0.1
            if next_above:
                distance = next_above.get('distance_pct', 100) / 100
                if distance < 0.01:  # Very close to liquidity above
                    score -= 0.1
            
            # Clamp to 0-1
            return max(0.01, min(0.99, score))
            
        except Exception as e:
            log.debug(f"Error calculating liquidity score: {e}")
            return 0.5
    
    def get_liquidity_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive liquidity summary for scoring layer
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Dictionary with liquidity summary
        """
        try:
            # Get all detection results
            stop_hunt = self.detect_stop_hunt_probability(df)
            inducements = self.detect_inducement(df) if self.enable_inducement else []
            failures = self.detect_sweep_failure(df) if self.enable_sweep_failure else []
            equal_levels = self.detect_equal_highs_lows(df)
            resting = self.estimate_resting_liquidity(df)
            sweeps = self.get_liquidity_sweeps_for_strategies(df)
            
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
                'score': self.get_liquidity_score(df),
                'net_bias': net_bias,
                'direction': direction,
                'stop_hunt_probability': stop_hunt.get('probability', 0),
                'stop_hunt_assessment': stop_hunt.get('assessment', 'LOW'),
                'inducement_count': len(inducements),
                'sweep_failure_count': len(failures),
                'equal_highs_count': len(equal_levels.get('equal_highs', [])),
                'equal_lows_count': len(equal_levels.get('equal_lows', [])),
                'has_down_sweep': self.has_recent_down_sweep(df),
                'has_up_sweep': self.has_recent_up_sweep(df),
                'next_liquidity_above': resting.get('next_above'),
                'next_liquidity_below': resting.get('next_below'),
                'sweeps': sweeps[-5:],  # Last 5 sweeps
                'reasons': self._build_reasons(stop_hunt, inducements, failures, sweeps)
            }
            
        except Exception as e:
            log.debug(f"Error getting liquidity summary: {e}")
            return {
                'score': 0.5,
                'net_bias': 0,
                'direction': 'NEUTRAL',
                'reasons': ['Error in liquidity analysis']
            }
    
    def get_net_bias(self, df: pd.DataFrame) -> float:
        """
        Get net liquidity bias (-1 to 1)
        Positive = bullish liquidity (sweeps below, inducements down)
        Negative = bearish liquidity (sweeps above, inducements up)
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Bias score between -1 and 1
        """
        try:
            bias = 0.0
            
            # Stop hunt contribution
            stop_hunt = self.detect_stop_hunt_probability(df)
            signals = stop_hunt.get('signals', [])
            for signal in signals:
                if signal.get('type') == 'DOWNSIDE_HUNT':
                    bias += 0.15 * signal.get('strength', 0.5)
                elif signal.get('type') == 'UPSIDE_HUNT':
                    bias -= 0.15 * signal.get('strength', 0.5)
            
            # Inducement contribution
            if self.enable_inducement:
                inducements = self.detect_inducement(df)
                for inducement in inducements:
                    if inducement.get('type') == 'DOWNSIDE_INDUCEMENT':
                        bias += 0.1 * inducement.get('strength', 0.5)
                    elif inducement.get('type') == 'UPSIDE_INDUCEMENT':
                        bias -= 0.1 * inducement.get('strength', 0.5)
            
            # Sweep failure contribution
            if self.enable_sweep_failure:
                failures = self.detect_sweep_failure(df)
                for failure in failures:
                    if failure.get('type') == 'BULLISH_SWEEP_FAILURE':
                        bias += 0.15 * failure.get('strength', 0.5)
                    elif failure.get('type') == 'BEARISH_SWEEP_FAILURE':
                        bias -= 0.15 * failure.get('strength', 0.5)
            
            # Recent sweeps contribution
            sweeps = self.get_liquidity_sweeps_for_strategies(df)
            for sweep in sweeps[-3:]:
                if 'DOWN' in sweep.get('type', ''):
                    bias += 0.1 * sweep.get('strength', 0.5)
                elif 'UP' in sweep.get('type', ''):
                    bias -= 0.1 * sweep.get('strength', 0.5)
            
            # Clamp to -1 to 1
            return max(-1.0, min(1.0, bias))
            
        except Exception as e:
            log.debug(f"Error calculating net bias: {e}")
            return 0.0
    
    def has_strong_sweep(self, df: pd.DataFrame, min_strength: float = 0.7) -> bool:
        """
        Check if there is a strong liquidity sweep
        
        Args:
            df: OHLCV DataFrame
            min_strength: Minimum strength threshold
        
        Returns:
            True if strong sweep detected
        """
        try:
            sweeps = self.get_liquidity_sweeps_for_strategies(df)
            for sweep in sweeps[-5:]:
                if sweep.get('strength', 0) >= min_strength:
                    return True
            return False
        except Exception:
            return False
    
    def _build_reasons(self, stop_hunt: Dict, inducements: List, 
                       failures: List, sweeps: List) -> List[str]:
        """Build human-readable reasons for liquidity analysis"""
        reasons = []
        
        # Stop hunt
        prob = stop_hunt.get('probability', 0)
        if prob > 0.6:
            reasons.append(f"High stop hunt probability ({prob:.0%})")
        elif prob > 0.3:
            reasons.append(f"Moderate stop hunt probability ({prob:.0%})")
        
        # Sweeps
        down_sweeps = [s for s in sweeps if 'DOWN' in s.get('type', '')]
        up_sweeps = [s for s in sweeps if 'UP' in s.get('type', '')]
        
        if down_sweeps:
            reasons.append(f"{len(down_sweeps)} downside sweep(s) detected")
        if up_sweeps:
            reasons.append(f"{len(up_sweeps)} upside sweep(s) detected")
        
        # Inducements
        if inducements:
            reasons.append(f"{len(inducements)} inducement pattern(s) detected")
        
        # Sweep failures
        if failures:
            reasons.append(f"{len(failures)} sweep failure(s) detected")
        
        return reasons[:5]  # Top 5 reasons


# ==================== LEGACY FUNCTIONS (Keep as is) ====================
def detect_liquidity_pools(df: pd.DataFrame, lookback: int = 100, tolerance: float = 0.0015) -> List[Dict[str,Any]]:
    """Legacy function"""
    pools = []
    try:
        if df is None or df.empty or len(df) < 20:
            return pools
        highs = df['high'].iloc[-lookback:]
        lows = df['low'].iloc[-lookback:]
        price = df['close'].iloc[-1]
        bins = np.round(highs / (price * tolerance)) * (price * tolerance)
        uniq = bins.value_counts().index.tolist()
        for u in uniq:
            count = (abs(highs - u) <= price * tolerance).sum()
            if count >= 2:
                pools.append({'price': float(u), 'type': 'HIGH_POOL', 'count': int(count)})
        binsl = np.round(lows / (price * tolerance)) * (price * tolerance)
        uniql = binsl.value_counts().index.tolist()
        for u in uniql:
            count = (abs(lows - u) <= price * tolerance).sum()
            if count >= 2:
                pools.append({'price': float(u), 'type': 'LOW_POOL', 'count': int(count)})
        return pools
    except Exception:
        return pools


def detect_liquidity_sweep(df: pd.DataFrame, threshold_atr_mult: float = 1.5) -> List[Dict[str,Any]]:
    """Legacy function"""
    sweeps = []
    try:
        if df is None or df.empty or len(df) < 30:
            return sweeps
        atr = float(df['atr'].iloc[-1]) if 'atr' in df and df['atr'].iloc[-1] > 0 else df['close'].iloc[-1]*0.02
        recent = df.iloc[-10:]
        vol_avg = recent['volume'].mean() if not recent['volume'].empty else 1.0
        for i in range(len(recent)):
            row = recent.iloc[i]
            wick_up = row['high'] - max(row['open'], row['close'])
            wick_down = min(row['open'], row['close']) - row['low']
            if wick_up > atr * threshold_atr_mult and row['volume'] > vol_avg * 1.5:
                sweeps.append({'type': 'UP_SWEEP', 'index': recent.index[i].isoformat(), 'wick': float(wick_up)})
            if wick_down > atr * threshold_atr_mult and row['volume'] > vol_avg * 1.5:
                sweeps.append({'type': 'DOWN_SWEEP', 'index': recent.index[i].isoformat(), 'wick': float(wick_down)})
        return sweeps
    except Exception:
        return sweeps


def detect_stop_hunt(df: pd.DataFrame, lookback: int = 30) -> bool:
    """Legacy function"""
    try:
        if df is None or df.empty or len(df) < lookback:
            return False
        recent = df.iloc[-lookback:]
        atr = float(recent['atr'].iloc[-1]) if 'atr' in recent and recent['atr'].iloc[-1] > 0 else recent['close'].iloc[-1]*0.02
        bars = recent.iloc[-5:]
        for i in range(len(bars)):
            row = bars.iloc[i]
            wick_up = row['high'] - max(row['open'], row['close'])
            wick_down = min(row['open'], row['close']) - row['low']
            if wick_up > atr * 2.0 or wick_down > atr * 2.0:
                if i < len(bars)-1:
                    nextc = bars.iloc[i+1]
                    if wick_up > atr*2 and nextc['close'] < nextc['open'] and nextc['volume'] > bars['volume'].mean()*1.2:
                        return True
                    if wick_down > atr*2 and nextc['close'] > nextc['open'] and nextc['volume'] > bars['volume'].mean()*1.2:
                        return True
        return False
    except Exception:
        return False