# high_timeframe_confirmer.py - Enhanced with Elite Higher Timeframe Intelligence
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
from config import Config
from scipy import stats

EPSILON = 1e-10  # Small value to prevent division by zero

class HTFTrendStrength(Enum):
    EXTREME_BULL = "EXTREME_BULL"
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    EXTREME_BEAR = "EXTREME_BEAR"

class HTFMarketPhase(Enum):
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    UNKNOWN = "UNKNOWN"

class HTFLiquidityState(Enum):
    HIGH_LIQUIDITY = "HIGH_LIQUIDITY"
    NORMAL_LIQUIDITY = "NORMAL_LIQUIDITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    UNKNOWN = "UNKNOWN"

@dataclass
class HTFStructure:
    trend: str
    strength: float
    strength_label: HTFTrendStrength
    aligned_tfs: int
    total_tfs: int
    market_phase: str
    liquidity_state: str
    key_levels: Dict[str, List[float]]
    divergence_score: float
    momentum_alignment: float
    temporal_pattern: str
    multi_res_sr: Dict[str, List[float]]
    stacking_score: float
    hierarchy_score: float
    conflict_resolution: Dict[str, Any]
    fractal_alignment: float
    top_down_confidence: float
    sync_index: float
    htf_liquidity_map: Dict[str, Any]
    confidence_boost: float
    detailed_metrics: Dict[str, Any]

class HighTimeframeConfirmer:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.htf_timeframes = ['1h', '4h', '1d', '1w']  # Expanded HTF list
        
        # HTF weights for hierarchy
        self.htf_weights = {
            '1h': 1.0,
            '4h': 1.5,
            '1d': 2.0,
            '1w': 2.5
        }
        
        # Confirmation matrix for HTFs
        self.htf_confirmation = {
            '1h': {'confirms': ['4h'], 'conflicts_with': ['1w']},
            '4h': {'confirms': ['1h', '1d'], 'conflicts_with': []},
            '1d': {'confirms': ['4h', '1w'], 'conflicts_with': ['1h']},
            '1w': {'confirms': ['1d'], 'conflicts_with': ['1h']}
        }

    def get_htf_trend_structure(self, symbol: str, ltf: str) -> Dict:
        """
        Enhanced HTF trend structure analysis with all elite features
        """
        # Initialize result with defaults
        result = {
            'trend': 'NEUTRAL',
            'strength': 0.0,
            'strength_label': 'NEUTRAL',
            'aligned': 0,
            'total_tfs': 0,
            'market_phase': 'UNKNOWN',
            'liquidity_state': 'UNKNOWN',
            'key_levels': {'supports': [], 'resistances': []},
            'divergence_score': 0.0,
            'momentum_alignment': 0.0,
            'temporal_pattern': 'UNKNOWN',
            'multi_res_sr': {'supports': [], 'resistances': [], 'pivots': []},
            'stacking_score': 0.0,
            'hierarchy_score': 0.0,
            'conflict_resolution': {'resolution': 'NEUTRAL', 'confidence': 0.0},
            'fractal_alignment': 0.0,
            'top_down_confidence': 0.0,
            'sync_index': 0.0,
            'htf_liquidity_map': {},
            'confidence_boost': 0.0,
            'detailed_metrics': {}
        }

        try:
            # Fetch data for all HTFs
            htf_data = {}
            for tf in self.htf_timeframes:
                if self._is_higher_tf(tf, ltf):
                    df = self.data_fetcher.fetch_ohlcv(symbol, tf, limit=200)
                    if df is not None and not df.empty and len(df) >= 50:
                        # Calculate indicators
                        df = df.copy()
                        df['ema_8'] = df['close'].ewm(span=8).mean()
                        df['ema_21'] = df['close'].ewm(span=21).mean()
                        df['ema_50'] = df['close'].ewm(span=50).mean()
                        df['ema_200'] = df['close'].ewm(span=200).mean()
                        
                        # Calculate additional indicators
                        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
                        df['volume_sma'] = df['volume'].rolling(20).mean()
                        df['volume_ratio'] = df['volume'] / df['volume_sma']
                        
                        htf_data[tf] = df

            if not htf_data:
                return result

            # ==================== BASIC TREND ANALYSIS ====================
            trends = []
            strengths = []
            aligned = 0
            
            for tf, df in htf_data.items():
                price = df['close'].iloc[-1]
                ema8 = df['ema_8'].iloc[-1]
                ema21 = df['ema_21'].iloc[-1]
                ema50 = df['ema_50'].iloc[-1]
                
                # Determine trend with strength
                if price > ema8 > ema21 > ema50:
                    trends.append('BULL')
                    # Calculate strength based on separation
                    sep_score = min(1.0, (ema8 - ema21) / ema21 * 20)
                    vol_score = min(1.0, df['volume_ratio'].iloc[-1] / 2)
                    strength = (sep_score * 0.7 + vol_score * 0.3)
                    strengths.append(strength)
                    aligned += 1
                elif price < ema8 < ema21 < ema50:
                    trends.append('BEAR')
                    sep_score = min(1.0, (ema21 - ema8) / ema21 * 20)
                    vol_score = min(1.0, df['volume_ratio'].iloc[-1] / 2)
                    strength = (sep_score * 0.7 + vol_score * 0.3)
                    strengths.append(strength)
                    aligned += 1
                else:
                    trends.append('NEUTRAL')
                    strengths.append(0.3)

            # ==================== ADVANCED HTF FEATURES ====================
            
            # 1. HTF Divergence Detection
            divergence_score = self._detect_htf_divergence(htf_data)
            
            # 2. Momentum Alignment
            momentum_alignment = self._calculate_htf_momentum_alignment(htf_data)
            
            # 3. Temporal Pattern Recognition
            temporal_pattern = self._detect_htf_temporal_pattern(htf_data)
            
            # 4. Multi-resolution S/R
            multi_res_sr = self._find_htf_multi_res_sr(htf_data)
            
            # 5. Timeframe Stacking
            stacking_score = self._calculate_htf_stacking(htf_data)
            
            # 6. HTF Bias Hierarchy
            hierarchy_score = self._calculate_htf_hierarchy(htf_data)
            
            # 7. HTF Conflict Resolution
            conflict_resolution = self._resolve_htf_conflicts(htf_data)
            
            # 8. Fractal Alignment
            fractal_alignment = self._calculate_htf_fractal_alignment(htf_data)
            
            # 9. Top-Down Confidence
            top_down_confidence = self._calculate_htf_top_down_confidence(htf_data)
            
            # 10. Timeframe Synchronization Index
            sync_index = self._calculate_htf_sync_index(htf_data)
            
            # 11. HTF Liquidity Mapping
            htf_liquidity_map = self._map_htf_liquidity_zones(htf_data)
            
            # 12. Market Phase Detection
            market_phase = self._detect_htf_market_phase(htf_data)
            
            # 13. Liquidity State
            liquidity_state = self._detect_htf_liquidity_state(htf_data)
            
            # 14. Key Levels
            key_levels = self._extract_htf_key_levels(htf_data)

            # ==================== AGGREGATE RESULTS ====================
            
            # Determine overall trend
            bull_count = trends.count('BULL')
            bear_count = trends.count('BEAR')
            
            if bull_count > bear_count:
                trend = 'BULL'
                avg_strength = np.mean([s for t, s in zip(trends, strengths) if t == 'BULL']) if bull_count > 0 else 0
            elif bear_count > bull_count:
                trend = 'BEAR'
                avg_strength = np.mean([s for t, s in zip(trends, strengths) if t == 'BEAR']) if bear_count > 0 else 0
            else:
                trend = 'NEUTRAL'
                avg_strength = 0.3

            # Determine strength label
            strength_label = self._get_strength_label(trend, avg_strength)

            # Calculate confidence boost
            confidence_boost = self._calculate_confidence_boost(
                avg_strength, stacking_score, hierarchy_score, sync_index
            )

            # Compile detailed metrics
            detailed_metrics = {
                'trend_distribution': {'bull': bull_count, 'bear': bear_count, 'neutral': trends.count('NEUTRAL')},
                'avg_strength': avg_strength,
                'divergence': divergence_score,
                'momentum_alignment': momentum_alignment,
                'temporal_pattern': temporal_pattern,
                'stacking': stacking_score,
                'hierarchy': hierarchy_score,
                'fractal': fractal_alignment,
                'sync_index': sync_index
            }

            # Update result
            result.update({
                'trend': trend,
                'strength': round(avg_strength, 3),
                'strength_label': strength_label.value,
                'aligned': aligned,
                'total_tfs': len(htf_data),
                'market_phase': market_phase,
                'liquidity_state': liquidity_state,
                'key_levels': key_levels,
                'divergence_score': round(divergence_score, 3),
                'momentum_alignment': round(momentum_alignment, 3),
                'temporal_pattern': temporal_pattern,
                'multi_res_sr': multi_res_sr,
                'stacking_score': round(stacking_score, 3),
                'hierarchy_score': round(hierarchy_score, 3),
                'conflict_resolution': conflict_resolution,
                'fractal_alignment': round(fractal_alignment, 3),
                'top_down_confidence': round(top_down_confidence, 3),
                'sync_index': round(sync_index, 3),
                'htf_liquidity_map': htf_liquidity_map,
                'confidence_boost': round(confidence_boost, 3),
                'detailed_metrics': detailed_metrics
            })

            return result

        except Exception as e:
            print(f"Error in get_htf_trend_structure: {e}")
            return result

    # ==================== HTF DIVERGENCE ====================

    def _detect_htf_divergence(self, htf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Detect divergence between higher timeframes
        Returns score from -1 (bearish divergence) to 1 (bullish divergence)
        """
        try:
            if len(htf_data) < 2:
                return 0.0

            divergence_scores = []
            timeframes = list(htf_data.keys())

            for i in range(len(timeframes)):
                for j in range(i + 1, len(timeframes)):
                    tf1 = timeframes[i]
                    tf2 = timeframes[j]

                    df1 = htf_data[tf1]
                    df2 = htf_data[tf2]

                    # Calculate price and momentum
                    price1 = df1['close'].iloc[-1]
                    price2 = df2['close'].iloc[-1]
                    
                    # Momentum (ROC)
                    roc1 = df1['close'].pct_change(5).iloc[-1] if len(df1) > 5 else 0
                    roc2 = df2['close'].pct_change(5).iloc[-1] if len(df2) > 5 else 0

                    # Check for divergence
                    if price1 > price2 and roc1 < roc2:
                        # Bearish divergence
                        div_score = -min(1.0, abs(roc1 - roc2) * 5)
                    elif price1 < price2 and roc1 > roc2:
                        # Bullish divergence
                        div_score = min(1.0, abs(roc1 - roc2) * 5)
                    else:
                        div_score = 0

                    # Weight by timeframe importance
                    weight = (self.htf_weights.get(tf1, 1) + self.htf_weights.get(tf2, 1)) / 2
                    divergence_scores.append(div_score * weight)

            return np.mean(divergence_scores) if divergence_scores else 0.0

        except Exception:
            return 0.0

    # ==================== HTF MOMENTUM ALIGNMENT ====================

    def _calculate_htf_momentum_alignment(self, htf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate momentum alignment across HTFs
        Returns score from 0 (no alignment) to 1 (perfect alignment)
        """
        try:
            momentums = []
            
            for tf, df in htf_data.items():
                if len(df) > 10:
                    # Calculate multiple momentum indicators
                    roc_5 = df['close'].pct_change(5).iloc[-1]
                    roc_10 = df['close'].pct_change(10).iloc[-1] if len(df) > 10 else 0
                    macd = (df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()).iloc[-1] if len(df) > 26 else 0
                    
                    # Composite momentum
                    momentum = (roc_5 * 0.5 + roc_10 * 0.3 + (1 if macd > 0 else -1) * 0.2)
                    momentums.append(momentum)

            if len(momentums) < 2:
                return 0.5

            # Check if all momentums have same sign
            signs = [1 if m > 0 else (-1 if m < 0 else 0) for m in momentums]
            non_zero_signs = [s for s in signs if s != 0]

            if not non_zero_signs:
                return 0.5

            if all(s == non_zero_signs[0] for s in non_zero_signs):
                # All aligned - strength based on magnitude
                avg_magnitude = np.mean([abs(m) for m in momentums])
                return min(1.0, avg_magnitude * 5)
            else:
                # Conflicting
                return 0.2

        except Exception:
            return 0.5

    # ==================== HTF TEMPORAL PATTERN ====================

    def _detect_htf_temporal_pattern(self, htf_data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect temporal patterns across HTFs
        Returns: 'EXPANDING', 'CONTRACTING', 'ACCELERATING', 'DECELERATING', 'STABLE', 'UNKNOWN'
        """
        try:
            if len(htf_data) < 2:
                return 'UNKNOWN'

            # Calculate volatility trends
            vol_trends = []
            
            for tf, df in htf_data.items():
                if len(df) > 50:
                    recent_vol = df['close'].pct_change().std() * np.sqrt(252)
                    old_vol = df['close'].iloc[:len(df)//2].pct_change().std() * np.sqrt(252)
                    
                    if old_vol > 0:
                        vol_ratio = recent_vol / old_vol
                        vol_trends.append(vol_ratio)

            if len(vol_trends) < 2:
                return 'UNKNOWN'

            avg_vol_trend = np.mean(vol_trends)

            if avg_vol_trend > 1.3:
                return 'EXPANDING'
            elif avg_vol_trend < 0.7:
                return 'CONTRACTING'
            elif avg_vol_trend > 1.1:
                return 'ACCELERATING'
            elif avg_vol_trend < 0.9:
                return 'DECELERATING'
            else:
                return 'STABLE'

        except Exception:
            return 'UNKNOWN'

    # ==================== HTF MULTI-RESOLUTION S/R ====================

    def _find_htf_multi_res_sr(self, htf_data: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]:
        """
        Find support/resistance levels that appear across multiple HTFs
        """
        try:
            all_levels = {'supports': [], 'resistances': [], 'pivots': []}
            
            for tf, df in htf_data.items():
                if len(df) < 50:
                    continue

                # Find swing points
                highs = df['high'].values
                lows = df['low'].values

                # Local maxima (resistance)
                for i in range(2, len(highs) - 2):
                    if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                        highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                        all_levels['resistances'].append(highs[i])

                # Local minima (support)
                for i in range(2, len(lows) - 2):
                    if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                        lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                        all_levels['supports'].append(lows[i])

                # Pivot points (recent highs/lows)
                all_levels['pivots'].append(df['high'].iloc[-1])
                all_levels['pivots'].append(df['low'].iloc[-1])

            # Cluster nearby levels
            for key in all_levels:
                if all_levels[key]:
                    all_levels[key].sort()
                    clustered = []
                    current_cluster = [all_levels[key][0]]

                    for level in all_levels[key][1:]:
                        if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.005:
                            current_cluster.append(level)
                        else:
                            if len(current_cluster) >= 2:
                                clustered.append(np.mean(current_cluster))
                            current_cluster = [level]

                    if len(current_cluster) >= 2:
                        clustered.append(np.mean(current_cluster))

                    all_levels[key] = clustered

            return all_levels

        except Exception:
            return {'supports': [], 'resistances': [], 'pivots': []}

    # ==================== HTF STACKING ====================

    def _calculate_htf_stacking(self, htf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate how well HTFs are stacked (ordered alignment)
        """
        try:
            positions = []
            
            for tf, df in htf_data.items():
                if len(df) < 20:
                    continue

                price = df['close'].iloc[-1]
                ema8 = df['ema_8'].iloc[-1]
                ema21 = df['ema_21'].iloc[-1]
                ema50 = df['ema_50'].iloc[-1]

                # Calculate position score (-1 to 1)
                if price > ema8 > ema21 > ema50:
                    positions.append(1)  # Strong bull
                elif price > ema8 > ema21:
                    positions.append(0.7)  # Bull
                elif price < ema8 < ema21 < ema50:
                    positions.append(-1)  # Strong bear
                elif price < ema8 < ema21:
                    positions.append(-0.7)  # Bear
                else:
                    positions.append(0)  # Mixed

            if len(positions) < 2:
                return 0.5

            # Check stacking consistency
            non_zero = [p for p in positions if abs(p) > 0.3]
            if not non_zero:
                return 0.5

            # All same direction?
            all_same = all(p * non_zero[0] > 0 for p in non_zero)
            if all_same:
                # Calculate strength based on position magnitudes
                avg_magnitude = np.mean([abs(p) for p in non_zero])
                coverage = len(non_zero) / len(positions)
                return avg_magnitude * coverage
            else:
                return 0.2

        except Exception:
            return 0.5

    # ==================== HTF BIAS HIERARCHY ====================

    def _calculate_htf_hierarchy(self, htf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate hierarchy alignment score
        Higher timeframes have more weight
        """
        try:
            weighted_scores = []
            total_weight = 0

            for tf, df in htf_data.items():
                if tf not in self.htf_weights or len(df) < 20:
                    continue

                weight = self.htf_weights[tf]

                # Calculate bias strength
                ema8 = df['ema_8'].iloc[-1]
                ema21 = df['ema_21'].iloc[-1]
                
                if ema8 > ema21:
                    strength = min(1.0, (ema8 - ema21) / ema21 * 20)
                    weighted_scores.append(strength * weight)
                elif ema8 < ema21:
                    strength = min(1.0, (ema21 - ema8) / ema21 * 20)
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

    def _resolve_htf_conflicts(self, htf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Resolve conflicts between HTFs
        """
        try:
            # Group by timeframe hierarchy
            low_htf = ['1h']
            mid_htf = ['4h']
            high_htf = ['1d', '1w']

            # Get bias for each group
            low_bias = self._get_htf_group_bias(htf_data, low_htf)
            mid_bias = self._get_htf_group_bias(htf_data, mid_htf)
            high_bias = self._get_htf_group_bias(htf_data, high_htf)

            # Check for conflicts
            biases = [b for b in [low_bias, mid_bias, high_bias] if b != 0]
            
            if len(biases) < 2:
                return {'resolution': 'INCONCLUSIVE', 'confidence': 0.5, 'primary_bias': biases[0] if biases else 0}

            # All aligned?
            if all(b * biases[0] > 0 for b in biases):
                return {
                    'resolution': 'ALIGNED',
                    'confidence': 0.9,
                    'primary_bias': biases[0],
                    'low_bias': low_bias,
                    'mid_bias': mid_bias,
                    'high_bias': high_bias
                }

            # Higher timeframe wins if conflict
            if high_bias != 0 and (low_bias * high_bias < 0 or mid_bias * high_bias < 0):
                return {
                    'resolution': 'FOLLOW_HIGHEST',
                    'confidence': 0.7,
                    'primary_bias': high_bias,
                    'low_bias': low_bias,
                    'mid_bias': mid_bias,
                    'high_bias': high_bias
                }

            # Mid timeframe wins if higher is neutral
            if mid_bias != 0 and low_bias * mid_bias < 0 and high_bias == 0:
                return {
                    'resolution': 'FOLLOW_MID',
                    'confidence': 0.6,
                    'primary_bias': mid_bias,
                    'low_bias': low_bias,
                    'mid_bias': mid_bias,
                    'high_bias': high_bias
                }

            return {
                'resolution': 'MIXED',
                'confidence': 0.4,
                'primary_bias': 0,
                'low_bias': low_bias,
                'mid_bias': mid_bias,
                'high_bias': high_bias
            }

        except Exception:
            return {'resolution': 'ERROR', 'confidence': 0.0}

    def _get_htf_group_bias(self, htf_data: Dict[str, pd.DataFrame], tfs: List[str]) -> float:
        """Get average bias for a group of HTFs"""
        biases = []

        for tf in tfs:
            if tf in htf_data and len(htf_data[tf]) >= 20:
                df = htf_data[tf]
                ema8 = df['ema_8'].iloc[-1]
                ema21 = df['ema_21'].iloc[-1]

                if ema8 > ema21:
                    biases.append(1)
                elif ema8 < ema21:
                    biases.append(-1)
                else:
                    biases.append(0)

        return np.mean(biases) if biases else 0

    # ==================== HTF FRACTAL ALIGNMENT ====================

    def _calculate_htf_fractal_alignment(self, htf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate fractal alignment across HTFs
        """
        try:
            correlations = []
            timeframes = list(htf_data.keys())

            for i in range(len(timeframes)):
                for j in range(i + 1, len(timeframes)):
                    tf1 = timeframes[i]
                    tf2 = timeframes[j]

                    df1 = htf_data[tf1]
                    df2 = htf_data[tf2]

                    if len(df1) < 30 or len(df2) < 30:
                        continue

                    # Normalize prices
                    
                    range1 = (df1['close'].max() - df1['close'].min()) + EPSILON
                    range2 = (df2['close'].max() - df2['close'].min()) + EPSILON
                    price1 = (df1['close'] - df1['close'].min()) / range1
                    price2 = (df2['close'] - df2['close'].min()) / range2

                    # Align lengths
                    min_len = min(len(price1), len(price2))
                    price1 = price1.iloc[-min_len:]
                    price2 = price2.iloc[-min_len:]

                    # Calculate correlation
                    corr = price1.corr(price2)
                    if not pd.isna(corr):
                        correlations.append(abs(corr))

            return np.mean(correlations) if correlations else 0.5

        except Exception:
            return 0.5

    # ==================== HTF TOP-DOWN CONFIDENCE ====================

    def _calculate_htf_top_down_confidence(self, htf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate confidence using top-down analysis
        Start from highest timeframe and work down
        """
        try:
            # Sort HTFs by weight (highest first)
            sorted_tfs = sorted(
                [tf for tf in htf_data.keys() if tf in self.htf_weights],
                key=lambda x: self.htf_weights[x],
                reverse=True
            )

            if not sorted_tfs:
                return 0.0

            confidence = 1.0
            prev_bias = None

            for tf in sorted_tfs:
                if tf not in htf_data or len(htf_data[tf]) < 20:
                    continue

                df = htf_data[tf]

                # Calculate bias
                ema8 = df['ema_8'].iloc[-1]
                ema21 = df['ema_21'].iloc[-1]

                if ema8 > ema21:
                    current_bias = 1
                elif ema8 < ema21:
                    current_bias = -1
                else:
                    current_bias = 0

                # Check alignment with previous timeframe
                if prev_bias is not None:
                    if current_bias == 0:
                        # Neutral - maintain confidence
                        pass
                    elif current_bias == prev_bias:
                        # Aligned - maintain confidence
                        pass
                    else:
                        # Conflicting - reduce confidence
                        confidence *= 0.6

                prev_bias = current_bias if current_bias != 0 else prev_bias

            return confidence

        except Exception:
            return 0.5

    # ==================== HTF SYNCHRONIZATION INDEX ====================

    def _calculate_htf_sync_index(self, htf_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate synchronization index across HTFs
        Measures how well HTFs move together
        """
        try:
            if len(htf_data) < 2:
                return 0.5

            # Calculate returns for each HTF
            returns_dict = {}
            for tf, df in htf_data.items():
                if len(df) > 20:
                    returns_dict[tf] = df['close'].pct_change().dropna()

            if len(returns_dict) < 2:
                return 0.5

            # Calculate pairwise correlations
            correlations = []
            tfs = list(returns_dict.keys())

            for i in range(len(tfs)):
                for j in range(i + 1, len(tfs)):
                    # Align returns
                    common_idx = returns_dict[tfs[i]].index.intersection(returns_dict[tfs[j]].index)
                    if len(common_idx) > 10:
                        r1 = returns_dict[tfs[i]].loc[common_idx]
                        r2 = returns_dict[tfs[j]].loc[common_idx]
                        corr = r1.corr(r2)
                        if not pd.isna(corr):
                            correlations.append(abs(corr))

            if correlations:
                # Average correlation is sync index
                sync_index = np.mean(correlations)
                return max(0, min(1, sync_index))

            return 0.5

        except Exception:
            return 0.5

    # ==================== HTF LIQUIDITY MAPPING ====================

    def _map_htf_liquidity_zones(self, htf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Map liquidity zones across HTFs
        """
        try:
            liquidity_map = {
                'major_supports': [],
                'major_resistances': [],
                'high_volume_nodes': [],
                'low_volume_nodes': []
            }

            all_highs = []
            all_lows = []
            all_volumes = []

            for tf, df in htf_data.items():
                if len(df) < 50:
                    continue

                # Add recent highs/lows
                all_highs.extend(df['high'].iloc[-20:].tolist())
                all_lows.extend(df['low'].iloc[-20:].tolist())
                
                # Add volume data
                for i in range(min(20, len(df))):
                    all_volumes.append({
                        'price': df['close'].iloc[-i-1],
                        'volume': df['volume'].iloc[-i-1]
                    })

            # Cluster highs (resistance clusters)
            if all_highs:
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
            if all_lows:
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

            # Find high volume nodes
            if all_volumes:
                # Sort by volume
                all_volumes.sort(key=lambda x: x['volume'], reverse=True)
                for i in range(min(5, len(all_volumes))):
                    liquidity_map['high_volume_nodes'].append({
                        'price': all_volumes[i]['price'],
                        'volume': all_volumes[i]['volume']
                    })

                # Find low volume nodes
                all_volumes.sort(key=lambda x: x['volume'])
                for i in range(min(5, len(all_volumes))):
                    liquidity_map['low_volume_nodes'].append({
                        'price': all_volumes[i]['price'],
                        'volume': all_volumes[i]['volume']
                    })

            return liquidity_map

        except Exception:
            return {'major_supports': [], 'major_resistances': [], 'high_volume_nodes': [], 'low_volume_nodes': []}

    # ==================== HTF MARKET PHASE ====================

    def _detect_htf_market_phase(self, htf_data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect Wyckoff market phase on higher timeframes
        """
        try:
            if not htf_data:
                return 'UNKNOWN'

            # Use daily as primary for phase detection
            if '1d' in htf_data and len(htf_data['1d']) >= 100:
                df = htf_data['1d']
            elif '4h' in htf_data and len(htf_data['4h']) >= 200:
                df = htf_data['4h']
            else:
                return 'UNKNOWN'

            # Find swing points
            from scipy.signal import argrelextrema
            price = df['close'].values
            volume = df['volume'].values

            local_max = argrelextrema(price, np.greater, order=10)[0]
            local_min = argrelextrema(price, np.less, order=10)[0]

            if len(local_max) < 2 or len(local_min) < 2:
                return 'UNKNOWN'

            # Get recent swings
            recent_max = price[local_max[-2:]] if len(local_max) >= 2 else []
            recent_min = price[local_min[-2:]] if len(local_min) >= 2 else []

            # Volume trend
            vol_trend = volume[-20:].mean() / volume[-40:-20].mean() if len(volume) >= 40 else 1

            # Detect phase
            if len(recent_max) >= 2 and len(recent_min) >= 2:
                if recent_max[-1] > recent_max[0] and recent_min[-1] > recent_min[0] and vol_trend > 1.1:
                    return 'MARKUP'
                elif recent_max[-1] < recent_max[0] and recent_min[-1] < recent_min[0] and vol_trend > 1.1:
                    return 'MARKDOWN'
                elif abs(recent_max[-1] - recent_max[0]) / recent_max[0] < 0.05 and vol_trend < 0.9:
                    return 'ACCUMULATION'
                elif abs(recent_max[-1] - recent_max[0]) / recent_max[0] < 0.05 and vol_trend > 1.2:
                    return 'DISTRIBUTION'

            return 'UNKNOWN'

        except Exception:
            return 'UNKNOWN'

    # ==================== HTF LIQUIDITY STATE ====================

    def _detect_htf_liquidity_state(self, htf_data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect liquidity state on higher timeframes
        """
        try:
            liquidity_scores = []

            for tf, df in htf_data.items():
                if len(df) < 50:
                    continue

                # Volume metrics
                avg_volume = df['volume'].mean()
                current_volume = df['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Spread proxy (high-low range)
                avg_range = (df['high'] - df['low']).mean()
                current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
                range_ratio = current_range / avg_range if avg_range > 0 else 1

                # Combined liquidity score
                # Higher volume + lower range = higher liquidity
                liquidity_score = volume_ratio / range_ratio if range_ratio > 0 else 1
                liquidity_scores.append(liquidity_score)

            if not liquidity_scores:
                return 'UNKNOWN'

            avg_liquidity = np.mean(liquidity_scores)

            if avg_liquidity > 1.5:
                return 'HIGH_LIQUIDITY'
            elif avg_liquidity > 0.7:
                return 'NORMAL_LIQUIDITY'
            else:
                return 'LOW_LIQUIDITY'

        except Exception:
            return 'UNKNOWN'

    # ==================== HTF KEY LEVELS ====================

    def _extract_htf_key_levels(self, htf_data: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]:
        """
        Extract key support/resistance levels from HTFs
        """
        try:
            key_levels = {'supports': [], 'resistances': []}

            for tf, df in htf_data.items():
                if len(df) < 50:
                    continue

                # Add recent swing points
                highs = df['high'].iloc[-20:]
                lows = df['low'].iloc[-20:]

                # Local maxima in recent data
                for i in range(2, len(highs) - 2):
                    if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
                        highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                        key_levels['resistances'].append(highs.iloc[i])

                # Local minima in recent data
                for i in range(2, len(lows) - 2):
                    if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and
                        lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                        key_levels['supports'].append(lows.iloc[i])

                # Add EMAs as dynamic levels
                key_levels['supports'].append(df['ema_50'].iloc[-1])
                key_levels['supports'].append(df['ema_200'].iloc[-1])
                key_levels['resistances'].append(df['ema_50'].iloc[-1])
                key_levels['resistances'].append(df['ema_200'].iloc[-1])

            # Remove duplicates and sort
            for key in key_levels:
                if key_levels[key]:
                    # Round to reasonable precision
                    key_levels[key] = [round(x, 6) for x in key_levels[key]]
                    # Remove duplicates (within 0.1%)
                    unique_levels = []
                    for level in sorted(key_levels[key]):
                        if not unique_levels or abs(level - unique_levels[-1]) / level > 0.001:
                            unique_levels.append(level)
                    key_levels[key] = unique_levels[-5:]  # Keep last 5

            return key_levels

        except Exception:
            return {'supports': [], 'resistances': []}

    # ==================== HELPER METHODS ====================

    def _get_strength_label(self, trend: str, strength: float) -> HTFTrendStrength:
        """Get strength label based on trend and strength value"""
        if trend == 'BULL':
            if strength >= 0.8:
                return HTFTrendStrength.EXTREME_BULL
            elif strength >= 0.6:
                return HTFTrendStrength.STRONG_BULL
            elif strength >= 0.4:
                return HTFTrendStrength.BULL
            else:
                return HTFTrendStrength.NEUTRAL
        elif trend == 'BEAR':
            if strength >= 0.8:
                return HTFTrendStrength.EXTREME_BEAR
            elif strength >= 0.6:
                return HTFTrendStrength.STRONG_BEAR
            elif strength >= 0.4:
                return HTFTrendStrength.BEAR
            else:
                return HTFTrendStrength.NEUTRAL
        else:
            return HTFTrendStrength.NEUTRAL

    def _calculate_confidence_boost(self, strength: float, stacking: float, 
                                   hierarchy: float, sync: float) -> float:
        """
        Calculate confidence boost based on multiple factors
        """
        try:
            # Weighted combination
            boost = (strength * 0.3 + stacking * 0.3 + hierarchy * 0.2 + sync * 0.2)
            
            # Scale to reasonable range
            boost = boost * 0.25  # Max 25% boost
            
            return boost

        except Exception:
            return 0.0

    def _is_higher_tf(self, tf1: str, tf2: str) -> bool:
        """Check if tf1 is higher timeframe than tf2"""
        order = {'1m': 1, '5m': 2, '15m': 3, '30m': 4, '1h': 5, '4h': 6, '1d': 7, '1w': 8}
        return order.get(tf1, 0) > order.get(tf2, 0)