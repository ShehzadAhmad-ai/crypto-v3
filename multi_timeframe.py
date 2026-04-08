# multi_timeframe.py - Enhanced with Elite Multi-Timeframe Intelligence
from config import Config
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats

class TimeframeBias(Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"

class TimeframeAlignment(Enum):
    STRONG_CONFIRMATION = "STRONG_CONFIRMATION"
    CONFIRMATION = "CONFIRMATION"
    NEUTRAL = "NEUTRAL"
    CONFLICT = "CONFLICT"
    STRONG_CONFLICT = "STRONG_CONFLICT"

@dataclass
class MTFMetrics:
    bias: TimeframeBias
    alignment: TimeframeAlignment
    divergence_score: float  # -1 to 1 (negative = bearish divergence)
    momentum_alignment: float  # 0 to 1
    temporal_pattern: str
    multi_res_sr: Dict[str, List[float]]
    stacking_score: float  # 0 to 1
    hierarchy_score: float  # 0 to 1
    fractal_alignment: float  # 0 to 1
    top_down_confidence: float  # 0 to 1
    htf_liquidity_map: Dict[str, Any]

class MultiTimeframeAligner:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        # Define timeframe hierarchy with weights
        self.tf_hierarchy = {
            '1m': {'weight': 0.3, 'group': 'ultra_low'},
            '5m': {'weight': 0.5, 'group': 'low'},
            '15m': {'weight': 0.7, 'group': 'medium_low'},
            '30m': {'weight': 0.8, 'group': 'medium'},
            '1h': {'weight': 1.0, 'group': 'medium_high'},
            '4h': {'weight': 1.2, 'group': 'high'},
            '1d': {'weight': 1.5, 'group': 'ultra_high'},
            '1w': {'weight': 2.0, 'group': 'extreme'}
        }
        
        # Define which timeframes confirm each other
        self.confirmation_matrix = {
            '1m': {'confirm_with': ['5m'], 'conflict_with': ['1h']},
            '5m': {'confirm_with': ['15m', '30m'], 'conflict_with': ['4h']},
            '15m': {'confirm_with': ['5m', '30m', '1h'], 'conflict_with': ['1d']},
            '30m': {'confirm_with': ['15m', '1h'], 'conflict_with': ['1d']},
            '1h': {'confirm_with': ['30m', '4h'], 'conflict_with': ['1w']},
            '4h': {'confirm_with': ['1h', '1d'], 'conflict_with': ['1m']},
            '1d': {'confirm_with': ['4h', '1w'], 'conflict_with': ['5m']},
            '1w': {'confirm_with': ['1d'], 'conflict_with': ['15m']}
        }
    
    def get_higher_tf_bias(self, symbol: str, scalping_tf: str) -> Dict:
        """Legacy method - kept for compatibility"""
        directions = []
        for tf in Config.HIGHER_TIMEFRAMES:
            df = self.data_fetcher.fetch_ohlcv(symbol, tf, limit=120)
            if df is None or df.empty or len(df) < 20:
                continue
            ema_short = df['close'].ewm(span=8).mean().iloc[-1]
            ema_long = df['close'].ewm(span=21).mean().iloc[-1]
            if ema_short > ema_long:
                directions.append('BULL')
            elif ema_short < ema_long:
                directions.append('BEAR')
            else:
                directions.append('NEUTRAL')
        if not directions:
            return {'bias':'NEUTRAL','aligned_count':0,'confidence_boost':0.0}
        bull = directions.count('BULL'); bear = directions.count('BEAR')
        aligned = max(bull,bear)
        bias = 'BULL' if bull>bear else ('BEAR' if bear>bull else 'NEUTRAL')
        boost = 0.0
        if aligned >= Config.MTF_MIN_CONFIRM:
            boost = Config.MTF_CONFIDENCE_BOOST * (aligned / len(Config.HIGHER_TIMEFRAMES))
        return {'bias':bias,'aligned_count':aligned,'confidence_boost':boost}
    
    # ==================== ENHANCED MTF METHODS ====================
    
    def get_comprehensive_mtf_analysis(self, symbol: str, primary_tf: str = '5m') -> MTFMetrics:
        """
        Get comprehensive multi-timeframe analysis with all elite features
        """
        try:
            # Fetch data for all timeframes
            tf_data = {}
            for tf in self.tf_hierarchy.keys():
                df = self.data_fetcher.fetch_ohlcv(symbol, tf, limit=200)
                if df is not None and not df.empty and len(df) >= 50:
                    tf_data[tf] = df
            
            if not tf_data:
                return self._default_mtf_metrics()
            
            # Calculate all MTF features
            bias = self._calculate_mtf_bias(tf_data)
            alignment = self._calculate_alignment(tf_data, primary_tf)
            divergence = self._calculate_timeframe_divergence(tf_data, primary_tf)
            momentum_alignment = self._calculate_momentum_alignment(tf_data)
            temporal_pattern = self._detect_temporal_pattern(tf_data)
            multi_res_sr = self._find_multi_resolution_sr(tf_data)
            stacking_score = self._calculate_stacking_score(tf_data, primary_tf)
            hierarchy_score = self._calculate_hierarchy_score(tf_data)
            fractal_alignment = self._calculate_fractal_alignment(tf_data)
            top_down_conf = self._calculate_top_down_confidence(tf_data, primary_tf)
            htf_liquidity = self._map_htf_liquidity(tf_data)
            
            return MTFMetrics(
                bias=bias,
                alignment=alignment,
                divergence_score=divergence,
                momentum_alignment=momentum_alignment,
                temporal_pattern=temporal_pattern,
                multi_res_sr=multi_res_sr,
                stacking_score=stacking_score,
                hierarchy_score=hierarchy_score,
                fractal_alignment=fractal_alignment,
                top_down_confidence=top_down_conf,
                htf_liquidity_map=htf_liquidity
            )
            
        except Exception as e:
            print(f"Error in MTF analysis: {e}")
            return self._default_mtf_metrics()
    
    def _default_mtf_metrics(self) -> MTFMetrics:
        """Return default MTF metrics"""
        return MTFMetrics(
            bias=TimeframeBias.NEUTRAL,
            alignment=TimeframeAlignment.NEUTRAL,
            divergence_score=0.0,
            momentum_alignment=0.5,
            temporal_pattern="UNKNOWN",
            multi_res_sr={'supports': [], 'resistances': [], 'pivots': []},
            stacking_score=0.5,
            hierarchy_score=0.5,
            fractal_alignment=0.5,
            top_down_confidence=0.0,
            htf_liquidity_map={}
        )
    
    # ==================== TIMEFRAME DIVERGENCE ====================
    
    def _calculate_timeframe_divergence(self, tf_data: Dict[str, pd.DataFrame], primary_tf: str) -> float:
        """
        Calculate divergence between timeframes
        Returns score from -1 (bearish divergence) to 1 (bullish divergence)
        """
        try:
            if primary_tf not in tf_data:
                return 0.0
            
            primary_df = tf_data[primary_tf]
            primary_price = primary_df['close'].iloc[-1]
            primary_roc = primary_df['close'].pct_change(10).iloc[-1] if len(primary_df) > 10 else 0
            
            divergence_scores = []
            
            for tf, df in tf_data.items():
                if tf == primary_tf or len(df) < 20:
                    continue
                
                tf_price = df['close'].iloc[-1]
                tf_roc = df['close'].pct_change(10).iloc[-1] if len(df) > 10 else 0
                
                # Check for hidden divergence
                if primary_roc > 0 and tf_roc > 0:
                    # Both up - no divergence
                    div_score = 0
                elif primary_roc > 0 and tf_roc < 0:
                    # Primary up, higher tf down - bearish divergence
                    div_score = -abs(primary_roc) * 2
                elif primary_roc < 0 and tf_roc > 0:
                    # Primary down, higher tf up - bullish divergence
                    div_score = abs(primary_roc) * 2
                else:
                    # Both down - no divergence
                    div_score = 0
                
                # Weight by timeframe
                weight = self.tf_hierarchy.get(tf, {}).get('weight', 1.0)
                divergence_scores.append(div_score * weight)
            
            if divergence_scores:
                return max(-1, min(1, np.mean(divergence_scores)))
            return 0.0
            
        except Exception:
            return 0.0
    
    # ==================== MOMENTUM ALIGNMENT ====================
    
    def _calculate_momentum_alignment(self, tf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate how well momentum aligns across timeframes
        Returns score from 0 (no alignment) to 1 (perfect alignment)
        """
        try:
            momentum_signs = []
            
            for tf, df in tf_data.items():
                if len(df) < 20:
                    continue
                
                # Calculate momentum (ROC)
                roc = df['close'].pct_change(10).iloc[-1] if len(df) > 10 else 0
                momentum_sign = 1 if roc > 0 else (-1 if roc < 0 else 0)
                
                if momentum_sign != 0:
                    momentum_signs.append(momentum_sign)
            
            if len(momentum_signs) < 2:
                return 0.5
            
            # Calculate alignment (all same sign = 1.0)
            unique_signs = set(momentum_signs)
            if len(unique_signs) == 1:
                return 1.0
            elif len(unique_signs) == 2 and 0 in unique_signs:
                # Some neutral, some directional
                return 0.5
            else:
                # Conflicting signs
                return 0.0
                
        except Exception:
            return 0.5
    
    # ==================== TEMPORAL PATTERN RECOGNITION ====================
    
    def _detect_temporal_pattern(self, tf_data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect temporal patterns across timeframes
        Returns: 'EXPANSION', 'CONTRACTION', 'ACCELERATION', 'DECELERATION', 'SYNCHRONIZED', 'UNKNOWN'
        """
        try:
            # Calculate volatility across timeframes
            vol_ratios = []
            
            for tf, df in tf_data.items():
                if len(df) < 50:
                    continue
                
                recent_vol = df['close'].pct_change().std() * np.sqrt(len(df))
                older_vol = df['close'].iloc[:len(df)//2].pct_change().std() * np.sqrt(len(df)//2)
                
                if older_vol > 0:
                    vol_ratio = recent_vol / older_vol
                    vol_ratios.append(vol_ratio)
            
            if len(vol_ratios) < 3:
                return 'UNKNOWN'
            
            avg_ratio = np.mean(vol_ratios)
            
            if avg_ratio > 1.3:
                return 'EXPANSION'
            elif avg_ratio < 0.7:
                return 'CONTRACTION'
            elif avg_ratio > 1.1:
                return 'ACCELERATION'
            elif avg_ratio < 0.9:
                return 'DECELERATION'
            else:
                return 'SYNCHRONIZED'
                
        except Exception:
            return 'UNKNOWN'
    
    # ==================== MULTI-RESOLUTION S/R ====================
    
    def _find_multi_resolution_sr(self, tf_data: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]:
        """
        Find support/resistance levels that appear across multiple timeframes
        """
        try:
            all_levels = {'supports': [], 'resistances': [], 'pivots': []}
            
            for tf, df in tf_data.items():
                if len(df) < 50:
                    continue
                
                # Find swing points
                highs = df['high'].values
                lows = df['low'].values
                
                # Local maxima (resistance)
                for i in range(2, len(highs) - 2):
                    if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                       highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                        all_levels['resistances'].append(highs[i])
                
                # Local minima (support)
                for i in range(2, len(lows) - 2):
                    if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                       lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                        all_levels['supports'].append(lows[i])
                
                # Pivot points (previous day high/low)
                all_levels['pivots'].append(df['high'].iloc[-1])
                all_levels['pivots'].append(df['low'].iloc[-1])
            
            # Cluster nearby levels (within 0.5%)
            for key in all_levels:
                if all_levels[key]:
                    # Sort and cluster
                    all_levels[key].sort()
                    clustered = []
                    current_cluster = [all_levels[key][0]]
                    
                    for level in all_levels[key][1:]:
                        if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.005:
                            current_cluster.append(level)
                        else:
                            clustered.append(np.mean(current_cluster))
                            current_cluster = [level]
                    
                    if current_cluster:
                        clustered.append(np.mean(current_cluster))
                    
                    all_levels[key] = clustered
            
            return all_levels
            
        except Exception:
            return {'supports': [], 'resistances': [], 'pivots': []}
    
    # ==================== TIMEFRAME STACKING ====================
    
    def _calculate_stacking_score(self, tf_data: Dict[str, pd.DataFrame], primary_tf: str) -> float:
        """
        Calculate how well timeframes are stacked (ordered alignment)
        Higher score means all timeframes aligned in same direction
        """
        try:
            if primary_tf not in tf_data:
                return 0.5
            
            # Get trend for each timeframe
            trends = []
            
            for tf, df in tf_data.items():
                if len(df) < 20:
                    continue
                
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                price = df['close'].iloc[-1]
                
                if price > ema_8 > ema_21:
                    trends.append(1)  # Bull
                elif price < ema_8 < ema_21:
                    trends.append(-1)  # Bear
                else:
                    trends.append(0)  # Neutral
            
            if len(trends) < 3:
                return 0.5
            
            # Check stacking (should be consistent)
            non_zero = [t for t in trends if t != 0]
            if not non_zero:
                return 0.5
            
            # All same direction?
            if all(t == non_zero[0] for t in non_zero):
                # Weight by number of timeframes aligned
                stacking_score = len(non_zero) / len(trends)
                return stacking_score
            else:
                # Conflicting directions
                return 0.0
                
        except Exception:
            return 0.5
    
    # ==================== TIMEFRAME BIAS HIERARCHY ====================
    
    def _calculate_hierarchy_score(self, tf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate hierarchy alignment score
        Higher timeframes should have more weight in determining overall bias
        """
        try:
            weighted_scores = []
            total_weight = 0
            
            for tf, df in tf_data.items():
                if len(df) < 20 or tf not in self.tf_hierarchy:
                    continue
                
                weight = self.tf_hierarchy[tf]['weight']
                
                # Calculate bias strength on this timeframe
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                
                if ema_8 > ema_21:
                    # Bullish - strength based on separation
                    strength = min(1.0, (ema_8 - ema_21) / ema_21 * 10)
                    weighted_scores.append(strength * weight)
                elif ema_8 < ema_21:
                    # Bearish - strength based on separation
                    strength = min(1.0, (ema_21 - ema_8) / ema_21 * 10)
                    weighted_scores.append(-strength * weight)
                
                total_weight += weight
            
            if total_weight == 0:
                return 0.5
            
            # Normalize to 0-1
            hierarchy_score = (np.sum(weighted_scores) / total_weight + 1) / 2
            return max(0, min(1, hierarchy_score))
            
        except Exception:
            return 0.5
    
    # ==================== HTF CONFLICT RESOLVER ====================
    
    def _resolve_htf_conflict(self, tf_data: Dict[str, pd.DataFrame], primary_tf: str) -> Dict[str, Any]:
        """
        Resolve conflicts between higher and lower timeframes
        Returns resolution strategy
        """
        try:
            if primary_tf not in tf_data:
                return {'resolution': 'USE_PRIMARY', 'confidence': 0.5}
            
            # Group timeframes
            lower_tfs = []
            higher_tfs = []
            
            for tf in tf_data.keys():
                if tf not in self.tf_hierarchy:
                    continue
                
                if self.tf_hierarchy[tf]['weight'] <= self.tf_hierarchy[primary_tf]['weight']:
                    lower_tfs.append(tf)
                else:
                    higher_tfs.append(tf)
            
            # Get bias for each group
            lower_bias = self._get_group_bias(tf_data, lower_tfs)
            higher_bias = self._get_group_bias(tf_data, higher_tfs)
            
            # Check for conflict
            if lower_bias * higher_bias < 0:  # Opposite signs
                # Higher timeframe wins but with reduced confidence
                resolution = 'FOLLOW_HIGHER'
                confidence = 0.6
            elif lower_bias == 0 or higher_bias == 0:
                # One side neutral
                resolution = 'USE_NON_NEUTRAL'
                confidence = 0.7
            else:
                # Aligned
                resolution = 'ALIGNED'
                confidence = 0.9
            
            return {
                'resolution': resolution,
                'confidence': confidence,
                'lower_bias': lower_bias,
                'higher_bias': higher_bias
            }
            
        except Exception:
            return {'resolution': 'USE_PRIMARY', 'confidence': 0.5}
    
    def _get_group_bias(self, tf_data: Dict[str, pd.DataFrame], tfs: List[str]) -> float:
        """Get average bias for a group of timeframes"""
        biases = []
        
        for tf in tfs:
            if tf not in tf_data or len(tf_data[tf]) < 20:
                continue
            
            df = tf_data[tf]
            ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
            ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
            
            if ema_8 > ema_21:
                biases.append(1)
            elif ema_8 < ema_21:
                biases.append(-1)
            else:
                biases.append(0)
        
        return np.mean(biases) if biases else 0
    
    # ==================== FRACTAL ALIGNMENT SCORE ====================
    
    def _calculate_fractal_alignment(self, tf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate fractal alignment across timeframes
        Higher score means patterns are self-similar across timeframes
        """
        try:
            # Calculate pattern similarity across timeframes
            pattern_correlations = []
            
            tfs = list(tf_data.keys())
            for i in range(len(tfs)):
                for j in range(i+1, len(tfs)):
                    tf1 = tfs[i]
                    tf2 = tfs[j]
                    
                    if tf1 not in tf_data or tf2 not in tf_data:
                        continue
                    
                    df1 = tf_data[tf1]
                    df2 = tf_data[tf2]
                    
                    if len(df1) < 30 or len(df2) < 30:
                        continue
                    
                    # Normalize to same scale
                    price1 = (df1['close'] - df1['close'].min()) / (df1['close'].max() - df1['close'].min() + 1e-8)  # ADD EPSILON
                    price2 = (df2['close'] - df2['close'].min()) / (df2['close'].max() - df2['close'].min() + 1e-8)  # ADD EPSILON
                    # Align lengths
                    min_len = min(len(price1), len(price2))
                    if min_len < 10:
                        continue
                    price1 = price1.iloc[-min_len:]
                    price2 = price2.iloc[-min_len:]
                    
                    # Calculate correlation
                    corr = price1.corr(price2 + 1e-8) 
                    if not pd.isna(corr):
                        pattern_correlations.append(abs(corr))
            
            if pattern_correlations:
                return np.mean(pattern_correlations)
            return 0.5
            
        except Exception:
            return 0.5
    
    # ==================== TOP-DOWN CONFIDENCE ENGINE ====================
    
    def _calculate_top_down_confidence(self, tf_data: Dict[str, pd.DataFrame], primary_tf: str) -> float:
        """
        Calculate confidence using top-down analysis
        Start from highest timeframe and work down
        """
        try:
            # Sort timeframes by weight (highest first)
            sorted_tfs = sorted(
                [tf for tf in tf_data.keys() if tf in self.tf_hierarchy],
                key=lambda x: self.tf_hierarchy[x]['weight'],
                reverse=True
            )
            
            if not sorted_tfs:
                return 0.0
            
            confidence = 1.0
            prev_bias = None
            
            for tf in sorted_tfs:
                if tf not in tf_data or len(tf_data[tf]) < 20:
                    continue
                
                df = tf_data[tf]
                
                # Calculate bias on this timeframe
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                
                if ema_8 > ema_21:
                    current_bias = 1
                elif ema_8 < ema_21:
                    current_bias = -1
                else:
                    current_bias = 0
                
                # Check alignment with previous timeframe
                if prev_bias is not None:
                    if current_bias == 0:
                        # Neutral doesn't reduce confidence
                        pass
                    elif current_bias == prev_bias:
                        # Aligned - maintain confidence
                        pass
                    else:
                        # Conflicting - reduce confidence
                        confidence *= 0.7
                
                prev_bias = current_bias if current_bias != 0 else prev_bias
            
            return confidence
            
        except Exception:
            return 0.5
    
    # ==================== HTF LIQUIDITY MAPPING ====================
    
    def _map_htf_liquidity(self, tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Map liquidity across higher timeframes
        Identifies major liquidity pools
        """
        try:
            liquidity_map = {
                'major_supports': [],
                'major_resistances': [],
                'liquidity_clusters': []
            }
            
            # Find major levels across timeframes
            all_highs = []
            all_lows = []
            
            for tf, df in tf_data.items():
                if len(df) < 50:
                    continue
                
                # Add recent highs/lows
                all_highs.extend(df['high'].iloc[-20:].tolist())
                all_lows.extend(df['low'].iloc[-20:].tolist())
            
            if not all_highs or not all_lows:
                return liquidity_map
            
            # Cluster highs (resistance clusters)
            all_highs.sort()
            clusters = []
            current_cluster = [all_highs[0]]
            
            for level in all_highs[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.01:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= 3:
                        clusters.append({
                            'price': np.mean(current_cluster),
                            'strength': len(current_cluster) / 5,
                            'touches': len(current_cluster)
                        })
                    current_cluster = [level]
            
            liquidity_map['major_resistances'] = clusters
            
            # Cluster lows (support clusters)
            all_lows.sort()
            clusters = []
            current_cluster = [all_lows[0]]
            
            for level in all_lows[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.01:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= 3:
                        clusters.append({
                            'price': np.mean(current_cluster),
                            'strength': len(current_cluster) / 5,
                            'touches': len(current_cluster)
                        })
                    current_cluster = [level]
            
            liquidity_map['major_supports'] = clusters
            
            return liquidity_map
            
        except Exception:
            return {'major_supports': [], 'major_resistances': [], 'liquidity_clusters': []}
    
    # ==================== HELPER METHODS ====================
    
    def _calculate_mtf_bias(self, tf_data: Dict[str, pd.DataFrame]) -> TimeframeBias:
        """Calculate overall MTF bias"""
        try:
            weighted_score = 0
            total_weight = 0
            
            for tf, df in tf_data.items():
                if tf not in self.tf_hierarchy or len(df) < 20:
                    continue
                
                weight = self.tf_hierarchy[tf]['weight']
                
                # Calculate trend score on this timeframe
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                
                if ema_8 > ema_21:
                    score = 1
                elif ema_8 < ema_21:
                    score = -1
                else:
                    score = 0
                
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight == 0:
                return TimeframeBias.NEUTRAL
            
            avg_score = weighted_score / total_weight
            
            if avg_score > 0.5:
                return TimeframeBias.STRONG_BULL
            elif avg_score > 0.2:
                return TimeframeBias.BULL
            elif avg_score > -0.2:
                return TimeframeBias.NEUTRAL
            elif avg_score > -0.5:
                return TimeframeBias.BEAR
            else:
                return TimeframeBias.STRONG_BEAR
                
        except Exception:
            return TimeframeBias.NEUTRAL
    
    def _calculate_alignment(self, tf_data: Dict[str, pd.DataFrame], primary_tf: str) -> TimeframeAlignment:
        """Calculate alignment between timeframes"""
        try:
            if primary_tf not in tf_data:
                return TimeframeAlignment.NEUTRAL
            
            primary_df = tf_data[primary_tf]
            primary_ema_8 = primary_df['close'].ewm(span=8).mean().iloc[-1]
            primary_ema_21 = primary_df['close'].ewm(span=21).mean().iloc[-1]
            
            if primary_ema_8 > primary_ema_21:
                primary_bias = 1
            elif primary_ema_8 < primary_ema_21:
                primary_bias = -1
            else:
                primary_bias = 0
            
            alignment_scores = []
            
            for tf, df in tf_data.items():
                if tf == primary_tf or tf not in self.tf_hierarchy or len(df) < 20:
                    continue
                
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                
                if ema_8 > ema_21:
                    tf_bias = 1
                elif ema_8 < ema_21:
                    tf_bias = -1
                else:
                    tf_bias = 0
                
                if primary_bias == 0 or tf_bias == 0:
                    alignment_scores.append(0)
                elif primary_bias == tf_bias:
                    alignment_scores.append(1)
                else:
                    alignment_scores.append(-1)
            
            if not alignment_scores:
                return TimeframeAlignment.NEUTRAL
            
            avg_alignment = np.mean(alignment_scores)
            
            if avg_alignment > 0.7:
                return TimeframeAlignment.STRONG_CONFIRMATION
            elif avg_alignment > 0.3:
                return TimeframeAlignment.CONFIRMATION
            elif avg_alignment > -0.3:
                return TimeframeAlignment.NEUTRAL
            elif avg_alignment > -0.7:
                return TimeframeAlignment.CONFLICT
            else:
                return TimeframeAlignment.STRONG_CONFLICT
                
        except Exception:
            return TimeframeAlignment.NEUTRAL
    
    # ==================== LEGACY METHODS (ENHANCED) ====================
    
    def check_timeframe_alignment(self, symbol: str, primary_tf: str, signal_direction: str) -> dict:
        """
        Enhanced alignment check using MTF metrics
        """
        try:
            # Get comprehensive MTF analysis
            mtf_metrics = self.get_comprehensive_mtf_analysis(symbol, primary_tf)
            
            # Determine if aligned
            is_aligned = False
            if signal_direction == 'BUY':
                is_aligned = mtf_metrics.bias in [TimeframeBias.BULL, TimeframeBias.STRONG_BULL]
            elif signal_direction == 'SELL':
                is_aligned = mtf_metrics.bias in [TimeframeBias.BEAR, TimeframeBias.STRONG_BEAR]
            
            # Get confirming/conflicting timeframes
            confirming = []
            conflicting = []
            
            tf_data = {}
            for tf in self.tf_hierarchy.keys():
                df = self.data_fetcher.fetch_ohlcv(symbol, tf, limit=50)
                if df is not None and not df.empty and len(df) >= 20:
                    tf_data[tf] = df
            
            for tf, df in tf_data.items():
                if tf == primary_tf:
                    continue
                
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                
                if ema_8 > ema_21:
                    tf_bias = 'BULL'
                elif ema_8 < ema_21:
                    tf_bias = 'BEAR'
                else:
                    tf_bias = 'NEUTRAL'
                
                if (signal_direction == 'BUY' and tf_bias == 'BULL') or \
                   (signal_direction == 'SELL' and tf_bias == 'BEAR'):
                    confirming.append(tf)
                elif tf_bias != 'NEUTRAL':
                    conflicting.append(tf)
            
            return {
                'aligned': is_aligned,
                'confirming_tfs': confirming,
                'conflicting_tfs': conflicting,
                'score': mtf_metrics.stacking_score,
                'required_alignment': f"Stacking score: {mtf_metrics.stacking_score:.2f}",
                'mtf_bias': mtf_metrics.bias.value,
                'alignment': mtf_metrics.alignment.value,
                'divergence': mtf_metrics.divergence_score,
                'top_down_confidence': mtf_metrics.top_down_confidence
            }
            
        except Exception as e:
            # Fallback to simple method
            return self._simple_alignment_check(symbol, primary_tf, signal_direction)
    
    def _simple_alignment_check(self, symbol: str, primary_tf: str, signal_direction: str) -> dict:
        """Simple alignment check as fallback"""
        if primary_tf not in self.tf_hierarchy:
            return {'aligned': False, 'confirming_tfs': [], 'conflicting_tfs': [], 'score': 0.0}
        
        confirm_with = self.tf_hierarchy[primary_tf].get('confirm_with', [])
        confirming = []
        conflicting = []
        
        for tf in confirm_with:
            try:
                df = self.data_fetcher.fetch_ohlcv(symbol, tf, limit=50)
                if df is None or df.empty or len(df) < 20:
                    continue
                
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                price = df['close'].iloc[-1]
                
                if price > ema_8 > ema_21:
                    tf_trend = 'BULL'
                elif price < ema_8 < ema_21:
                    tf_trend = 'BEAR'
                else:
                    tf_trend = 'NEUTRAL'
                
                if (signal_direction == 'BUY' and tf_trend == 'BULL') or \
                   (signal_direction == 'SELL' and tf_trend == 'BEAR'):
                    confirming.append(tf)
                elif tf_trend != 'NEUTRAL':
                    conflicting.append(tf)
                    
            except Exception:
                continue
        
        total_needed = len(confirm_with)
        score = len(confirming) / total_needed if total_needed > 0 else 0.0
        
        # Determine if aligned
        if primary_tf == '5m':
            aligned = ('15m' in confirming and '30m' in confirming)
        elif primary_tf == '15m':
            aligned = ('5m' in confirming and '30m' in confirming)
        elif primary_tf == '30m':
            aligned = len(confirming) >= 1
        else:
            aligned = len(confirming) >= len(confirm_with) // 2
        
        return {
            'aligned': aligned,
            'confirming_tfs': confirming,
            'conflicting_tfs': conflicting,
            'score': round(score, 2)
        }
    
    def get_mtf_bias_strength(self, symbol: str) -> dict:
        """Get overall multi-timeframe bias strength using enhanced metrics"""
        try:
            mtf_metrics = self.get_comprehensive_mtf_analysis(symbol)
            
            return {
                'bias': mtf_metrics.bias.value,
                'strength': mtf_metrics.stacking_score,
                'bull_tfs': len([k for k, v in mtf_metrics.multi_res_sr.items() if 'resistances' in k]),
                'bear_tfs': len([k for k, v in mtf_metrics.multi_res_sr.items() if 'supports' in k]),
                'neutral_tfs': 0,
                'tf_data': {},
                'alignment': mtf_metrics.alignment.value,
                'divergence': mtf_metrics.divergence_score,
                'fractal_alignment': mtf_metrics.fractal_alignment,
                'top_down_confidence': mtf_metrics.top_down_confidence
            }
            
        except Exception:
            # Fallback to original method
            return self._simple_mtf_bias(symbol)
    
    def _simple_mtf_bias(self, symbol: str) -> dict:
        """Simple MTF bias as fallback"""
        tfs = ['5m', '15m', '30m', '1h']
        bull_count = 0
        bear_count = 0
        neutral_count = 0
        tf_data = {}
        
        for tf in tfs:
            try:
                df = self.data_fetcher.fetch_ohlcv(symbol, tf, limit=50)
                if df is None or df.empty or len(df) < 20:
                    continue
                
                ema_8 = df['close'].ewm(span=8).mean().iloc[-1]
                ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
                price = df['close'].iloc[-1]
                
                if price > ema_8 > ema_21:
                    trend = 'BULL'
                    bull_count += 1
                elif price < ema_8 < ema_21:
                    trend = 'BEAR'
                    bear_count += 1
                else:
                    trend = 'NEUTRAL'
                    neutral_count += 1
                
                tf_data[tf] = trend
            except Exception:
                continue
        
        total = bull_count + bear_count + neutral_count
        if total == 0:
            return {'bias': 'NEUTRAL', 'strength': 0.0, 'tf_data': tf_data}
        
        if bull_count > bear_count:
            bias = 'BULL'
            strength = bull_count / total
        elif bear_count > bull_count:
            bias = 'BEAR'
            strength = bear_count / total
        else:
            bias = 'NEUTRAL'
            strength = neutral_count / total if neutral_count > 0 else 0.5
        
        return {
            'bias': bias,
            'strength': round(strength, 2),
            'bull_tfs': bull_count,
            'bear_tfs': bear_count,
            'neutral_tfs': neutral_count,
            'tf_data': tf_data
        }