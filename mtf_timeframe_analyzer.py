# mtf_timeframe_analyzer.py
"""
Multi-Timeframe Timeframe Analyzer
Runs COMPLETE 8-layer technical analysis on a single timeframe
Used by MTF Pipeline to analyze each higher timeframe individually
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from config import Config
from logger import log

# Import analysis modules
from market_regime import MarketRegimeDetector
from technical_analyzer import TechnicalAnalyzer
from dynamic_indicators import DynamicIndicatorConfig
from price_action import AdvancedPriceAction
from enhanced_patterns import EnhancedPatternDetector
from market_structure import (
    detect_swings, get_structure_score, get_structure_bias, 
    get_structure_summary, has_strong_structure
)
from price_location import (
    detect_support_resistance, vwap_position, get_sr_score, 
    get_sr_bias, get_sr_summary, has_support_nearby, has_resistance_nearby
)
from volume_analyzer import VolumeAnalyzer
from layer_scoring import LayerScoring, create_layer_data_from_analysis


class MTFTimeframeAnalyzer:
    """
    Runs COMPLETE technical analysis on a single timeframe
    Used by MTF Pipeline to analyze each higher timeframe
    
    Output includes:
    - Market regime (bull/bear/range with confidence)
    - Indicator scores (RSI, MACD, EMA stack)
    - Price action patterns (candlestick patterns)
    - Chart patterns (triangles, H&S, etc.)
    - Market structure (BOS, CHOCH, order blocks)
    - Support/Resistance levels
    - Volume analysis
    - Overall score (0-1)
    - Direction (BULLISH/BEARISH/NEUTRAL)
    - Story (human-readable summary)
    """
    
    def __init__(self):
        # Initialize all analysis modules
        self.regime_detector = MarketRegimeDetector()
        self.tech_analyzer = TechnicalAnalyzer()
        self.dynamic_indicators = DynamicIndicatorConfig()
        self.price_action_detector = AdvancedPriceAction()
        self.chart_pattern_detector = EnhancedPatternDetector()
        self.volume_analyzer = VolumeAnalyzer()
        self.layer_scoring = LayerScoring()
        
        # Cache for asset profiles
        self.asset_profiles = {}
        
        log.debug("MTFTimeframeAnalyzer initialized")
    
    def analyze_timeframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        primary_timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete technical analysis on a single timeframe
        
        Args:
            df: OHLCV DataFrame for this timeframe
            symbol: Trading pair symbol
            timeframe: Current timeframe being analyzed (e.g., '1h', '4h')
            primary_timeframe: Primary trading timeframe (for context)
        
        Returns:
            Dictionary with complete analysis results
        """
        if df is None or df.empty or len(df) < 50:
            return self._empty_result(timeframe)
        
        try:
            # ===== LAYER 1: MARKET REGIME DETECTION =====
            regime_result = self.regime_detector.detect_regime_complete(df)
            
            regime_dict = {
                'regime': regime_result.regime,
                'type': regime_result.regime_type,
                'bias': regime_result.bias,
                'bias_score': regime_result.bias_score,
                'confidence': regime_result.regime_confidence,
                'trending_score': regime_result.trending_score,
                'volatility_state': regime_result.volatility_state,
                'wyckoff_phase': regime_result.wyckoff_phase,
                'indicator_settings': regime_result.indicator_settings
            }
            
            # ===== LAYER 2: DYNAMIC INDICATOR SETTINGS =====
            self.tech_analyzer.set_regime_settings(regime_dict.get('indicator_settings', {}))
            
            # ===== LAYER 3: TECHNICAL INDICATORS =====
            df = self.tech_analyzer.calculate_all_indicators(df, symbol)
            indicator_analysis = self.tech_analyzer.get_latest_indicator_analysis(df)
            indicator_score = indicator_analysis['combined'].score

            
            # ===== LAYER 4: PRICE ACTION ANALYSIS =====
            price_action_summary = self.price_action_detector.get_pattern_summary(df, regime_dict)
            price_action_score = self.price_action_detector.get_pattern_score(df, 'BULL')
            price_action_bias = self.price_action_detector.get_net_bias(df, regime_dict)
            
            # ===== LAYER 5: CHART PATTERN ANALYSIS =====
            chart_patterns = self.chart_pattern_detector.detect_all_patterns(df)
            chart_summary = self.chart_pattern_detector.get_pattern_summary(chart_patterns)
            chart_score = self.chart_pattern_detector.get_pattern_score(df, 'BUY')
            chart_bias = self.chart_pattern_detector.get_net_bias(df, regime_dict)
            
            # ===== LAYER 6: MARKET STRUCTURE ANALYSIS =====
            structure_data = detect_swings(df)
            structure_summary = get_structure_summary(df)
            structure_score = get_structure_score(df)
            structure_bias = get_structure_bias(df)
            
            # ===== LAYER 7: SUPPORT/RESISTANCE =====
            sr_data = detect_support_resistance(df)
            vwap_data = vwap_position(df)
            sr_summary = get_sr_summary(df)
            sr_score = get_sr_score(df)
            sr_bias = get_sr_bias(df)
            
            # ===== LAYER 8: VOLUME ANALYSIS =====
            volume_summary = self.volume_analyzer.get_volume_summary(df)
            volume_score = self.volume_analyzer.get_volume_score(df)
            volume_bias = self.volume_analyzer.get_volume_bias(df)
            
            # ===== LAYER 9: SCORE AGGREGATION =====
            layer_data = create_layer_data_from_analysis(
                regime_result=regime_dict,
                indicator_result=indicator_analysis,
                price_action_result=price_action_summary,
                chart_patterns_result={
                    'net_bias': chart_bias,
                    'buy_signals': chart_summary.get('buy_signals', 0),
                    'sell_signals': chart_summary.get('sell_signals', 0),
                    'total_patterns': chart_summary.get('total_patterns', 0),
                    'pattern_score': chart_score
                },
                structure_result=structure_summary,
                sr_result=sr_summary,
                volume_result=volume_summary
            )
            
            aggregated = self.layer_scoring.aggregate_scores(layer_data)
            
            # ===== DETERMINE DIRECTION =====
            direction = self._get_direction_from_score(aggregated.net_bias)
            
            # ===== BUILD TIMEFRAME SUMMARY =====
            result = {
                'timeframe': timeframe,
                'weight': self._get_timeframe_weight(timeframe, primary_timeframe),
                
                # Direction and confidence
                'direction': direction,
                'bias_score': aggregated.net_bias,
                'confidence': aggregated.confidence,
                'overall_score': aggregated.total_score,
                
                # Layer scores
                'layer_scores': {
                    'regime': regime_dict.get('bias_score', 0),
                        
                    'price_action': price_action_bias,
                    'indicators': indicator_score,
                    'chart_patterns': chart_bias,
                    'structure': structure_bias,
                    'sr': sr_bias,
                    'volume': volume_bias,
                    'aggregated': aggregated.total_score
                },
                
                # Market regime
                'regime': {
                    'type': regime_dict.get('regime', 'UNKNOWN'),
                    'bias': regime_dict.get('bias', 'NEUTRAL'),
                    'strength': regime_dict.get('trending_score', 0.5),
                    'wyckoff_phase': regime_dict.get('wyckoff_phase', 'UNKNOWN')
                },
                
                # Key levels on this timeframe
                'support': sr_summary.get('closest_support'),
                'resistance': sr_summary.get('closest_resistance'),
                'supports': sr_summary.get('supports', [])[:3],
                'resistances': sr_summary.get('resistances', [])[:3],
                
                # VWAP
                'vwap': vwap_data.get('vwap'),
                'vwap_position': vwap_data.get('position', 'NEUTRAL'),
                
                # Volume
                'volume': {
                    'trend': volume_summary.get('volume_trend', 'NEUTRAL'),
                    'ratio': volume_summary.get('volume_ratio', 1.0),
                    'has_spike': volume_summary.get('volume_ratio', 1) > 1.5,
                    'divergence': volume_summary.get('divergence')
                },
                
                # Structure
                'structure': {
                    'trend': structure_summary.get('trend', 'NEUTRAL'),
                    'has_bos': structure_summary.get('has_recent_bos', False),
                    'has_choch': structure_summary.get('has_recent_choch', False),
                    'bos_count': structure_summary.get('bos_count', 0),
                    'choch_count': structure_summary.get('choch_count', 0)
                },
                
                # Patterns detected
                'patterns': {
                    'price_action': price_action_summary.get('pattern_list', [])[:5],
                    'chart_patterns': chart_summary.get('pattern_types', {})
                },
                
                # Story for this timeframe
                'story': self._build_timeframe_story(
                    direction, aggregated.total_score, regime_dict,
                    price_action_summary, structure_summary, volume_summary
                ),
                
                # Raw data for debugging
                'raw_data': {
                    'price': float(df['close'].iloc[-1]),
                    'volume': float(df['volume'].iloc[-1]),
                    'timestamp': df.index[-1] if hasattr(df.index, 'iloc') else datetime.now()
                }
            }
            
            return result
            
        except Exception as e:
            log.error(f"Error analyzing {timeframe} for {symbol}: {e}")
            return self._empty_result(timeframe, error=str(e))
    
    def _get_direction_from_score(self, bias_score: float) -> str:
        """Convert bias score to direction"""
        if bias_score > 0.2:
            return 'BULLISH'
        elif bias_score < -0.2:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _get_timeframe_weight(self, timeframe: str, primary_timeframe: Optional[str]) -> float:
        """Get weight for this timeframe based on config"""
        # Get weights from config
        weights = getattr(Config, 'MTF_WEIGHTS', {
            '15m': 0.5,
            '30m': 0.7,
            '1h': 1.0,
            '4h': 1.5,
            '1d': 2.0,
            '1w': 2.5
        })
        
        # If this is the primary timeframe, weight is 0 (not used)
        if primary_timeframe and timeframe == primary_timeframe:
            return 0.0
        
        return weights.get(timeframe, 1.0)
    
    def _build_timeframe_story(
        self,
        direction: str,
        score: float,
        regime: Dict,
        price_action: Dict,
        structure: Dict,
        volume: Dict
    ) -> str:
        """Build human-readable story for this timeframe"""
        parts = []
        
        # Direction and confidence
        if direction == 'BULLISH':
            parts.append(f"Bullish with {score:.0%} confidence")
        elif direction == 'BEARISH':
            parts.append(f"Bearish with {score:.0%} confidence")
        else:
            parts.append("Neutral bias")
        
        # Regime context
        regime_type = regime.get('type', 'UNKNOWN')
        if 'BULL' in regime_type:
            parts.append(f"in {regime_type.lower()} regime")
        elif 'BEAR' in regime_type:
            parts.append(f"in {regime_type.lower()} regime")
        
        # Price action patterns
        patterns = price_action.get('pattern_list', [])
        if patterns:
            top_patterns = [p['name'] for p in patterns[:2]]
            parts.append(f"with {', '.join(top_patterns)} patterns")
        
        # Structure
        if structure.get('has_bos'):
            parts.append("recent BOS confirmed")
        
        # Volume
        if volume.get('has_spike'):
            parts.append(f"volume spike ({volume.get('ratio', 1):.1f}x)")
        elif volume.get('trend') == 'INCREASING':
            parts.append("volume increasing")
        
        return " ".join(parts) if parts else f"{direction} bias"
    
    def _empty_result(self, timeframe: str, error: str = "") -> Dict[str, Any]:
        """Return empty result when analysis fails"""
        return {
            'timeframe': timeframe,
            'weight': 0,
            'direction': 'NEUTRAL',
            'bias_score': 0,
            'confidence': 0,
            'overall_score': 0.5,
            'layer_scores': {},
            'regime': {'type': 'UNKNOWN', 'bias': 'NEUTRAL'},
            'support': None,
            'resistance': None,
            'supports': [],
            'resistances': [],
            'vwap': None,
            'vwap_position': 'NEUTRAL',
            'volume': {'trend': 'NEUTRAL', 'ratio': 1.0},
            'structure': {'trend': 'NEUTRAL'},
            'patterns': {},
            'story': f"Analysis failed: {error}" if error else "No data available",
            'raw_data': {}
        }


# ==================== HELPER FUNCTION ====================

def analyze_timeframe(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    primary_timeframe: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze a single timeframe
    
    Args:
        df: OHLCV DataFrame for this timeframe
        symbol: Trading pair symbol
        timeframe: Timeframe name (e.g., '1h', '4h')
        primary_timeframe: Primary trading timeframe
    
    Returns:
        Complete analysis results
    """
    analyzer = MTFTimeframeAnalyzer()
    return analyzer.analyze_timeframe(df, symbol, timeframe, primary_timeframe)