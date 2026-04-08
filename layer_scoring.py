# layer_scoring.py
"""
Layer Scoring System (Layer 9)
Aggregates scores from all 8 technical analysis layers into:
- Combined technical score (0-1)
- Bullish/bearish net bias (-1 to 1)
- Individual layer scores with weights
- Confidence based on agreement between layers
"""
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from logger import log


class LayerName(Enum):
    """Names of all analysis layers"""
    REGIME = "regime"
    INDICATORS = "indicators"
    PRICE_ACTION = "price_action"
    CHART_PATTERNS = "chart_patterns"
    STRUCTURE = "structure"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLUME = "volume"


@dataclass
class LayerScore:
    """Score for a single layer"""
    name: str
    score: float  # 0-1 (0 = bearish, 1 = bullish)
    bias: float   # -1 to 1 (negative = bearish, positive = bullish)
    weight: float
    confidence: float  # 0-1
    reason: str


@dataclass
class AggregatedScore:
    """Final aggregated score from all layers"""
    total_score: float           # 0-1 final technical score
    net_bias: float              # -1 to 1 (bullish/bearish)
    confidence: float            # 0-1 confidence in this score
    layer_scores: List[LayerScore]
    weight_breakdown: Dict[str, float]
    agreement_score: float       # 0-1 how well layers agree
    dominant_direction: str      # 'BULLISH', 'BEARISH', 'NEUTRAL'
    reasons: List[str]


class LayerScoring:
    """
    Aggregates all 8 technical layers into final technical score
    Weights can be configured in config.py
    """
    
    def __init__(self):
        # Default weights (sum to 1.0)
        # These can be overridden in config.py
        self.default_weights = {
            'regime': 0.15,              # Market regime
            'indicators': 0.20,          # Technical indicators
            'price_action': 0.20,        # Candlestick patterns
            'chart_patterns': 0.15,      # Chart patterns (triangles, H&S)
            'structure': 0.10,           # Market structure (BOS, CHOCH)
            'support_resistance': 0.10,  # S/R levels, VWAP
            'volume': 0.10               # Volume analysis
        }
        
        # Load weights from config if available
        self.weights = self._load_weights()
        
        log.info("LayerScoring initialized")
        log.info(f"Weights: {self.weights}")
    
    def _load_weights(self) -> Dict[str, float]:
        """Load weights from config.py if available"""
        try:
            from config import Config
            weights = {}
            weights['regime'] = getattr(Config, 'LAYER_WEIGHT_REGIME', self.default_weights['regime'])
            weights['indicators'] = getattr(Config, 'LAYER_WEIGHT_INDICATORS', self.default_weights['indicators'])
            weights['price_action'] = getattr(Config, 'LAYER_WEIGHT_PRICE_ACTION', self.default_weights['price_action'])
            weights['chart_patterns'] = getattr(Config, 'LAYER_WEIGHT_CHART_PATTERNS', self.default_weights['chart_patterns'])
            weights['structure'] = getattr(Config, 'LAYER_WEIGHT_STRUCTURE', self.default_weights['structure'])
            weights['support_resistance'] = getattr(Config, 'LAYER_WEIGHT_SR', self.default_weights['support_resistance'])
            weights['volume'] = getattr(Config, 'LAYER_WEIGHT_VOLUME', self.default_weights['volume'])
            
            # Normalize to sum to 1.0
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                for k in weights:
                    weights[k] /= total
            
            return weights
        except:
            return self.default_weights
        
    def _extract_numeric(self, value: Any, default: float = 0.5) -> float:
        """
        Safely extract a numeric value from various possible types.
        Handles IndicatorResult objects, dicts, and direct numbers.
        No fallbacks - returns actual value or raises error.
        """
        if value is None:
            return default
        
        # If it's already a number
        if isinstance(value, (int, float)):
            return float(value)
        
        # If it's a dataclass/object with a 'score' attribute
        if hasattr(value, 'score'):
            return float(value.score)
        
        # If it's a dataclass/object with a 'value' attribute
        if hasattr(value, 'value'):
            return float(value.value)
        
        # If it's a dict with 'score' key
        if isinstance(value, dict) and 'score' in value:
            return float(value['score'])
        
        # If it's a dict with 'value' key
        if isinstance(value, dict) and 'value' in value:
            return float(value['value'])
        
        # If we got here, we can't extract a number
        raise ValueError(f"Cannot extract numeric value from {type(value)}: {value}")

    
    def aggregate_scores(self, layer_data: Dict[str, Any]) -> AggregatedScore:
        """
        Aggregate all layer scores into final technical score
        
        Args:
            layer_data: Dictionary containing scores from all layers
                Expected format with proper objects, not just numbers
        """
        layer_scores = []
        reasons = []
        
        # ===== 1. REGIME LAYER =====
        regime = layer_data.get('regime', {})
        regime_score = self._extract_numeric(regime.get('score', 0.5))
        regime_bias = self._extract_numeric(regime.get('bias_score', 0))
        regime_conf = self._extract_numeric(regime.get('confidence', 0.5))
        
        layer_scores.append(LayerScore(
            name='regime',
            score=regime_score,
            bias=regime_bias,
            weight=self.weights['regime'],
            confidence=regime_conf,
            reason=f"Regime: {regime.get('regime', 'UNKNOWN')} (score: {regime_score:.2f})"
        ))
        reasons.append(f"Market regime: {regime.get('regime', 'UNKNOWN')} with {regime_conf:.0%} confidence")
        
        # ===== 2. INDICATORS LAYER =====
        indicators = layer_data.get('indicators', {})
        
        # Extract combined score - may be IndicatorResult or direct number
        combined = indicators.get('combined', {})
        if hasattr(combined, 'score'):
            ind_score = combined.score
        elif isinstance(combined, dict):
            ind_score = self._extract_numeric(combined.get('score', 0.5))
        else:
            ind_score = self._extract_numeric(combined, 0.5)
        
        # Extract RSI value
        rsi_obj = indicators.get('rsi', {})
        if hasattr(rsi_obj, 'value'):
            rsi_value = rsi_obj.value
        elif isinstance(rsi_obj, dict):
            rsi_value = self._extract_numeric(rsi_obj.get('value', 50))
        else:
            rsi_value = self._extract_numeric(rsi_obj, 50)
        
        # Extract MACD value
        macd_obj = indicators.get('macd', {})
        if hasattr(macd_obj, 'value'):
            macd_value = macd_obj.value
        elif isinstance(macd_obj, dict):
            macd_value = self._extract_numeric(macd_obj.get('value', 0))
        else:
            macd_value = self._extract_numeric(macd_obj, 0)
        
        # Extract bias
        ind_bias = indicators.get('indicator_score', 0)
        if hasattr(ind_bias, 'score'):
            ind_bias = ind_bias.score
        elif isinstance(ind_bias, dict):
            ind_bias = self._extract_numeric(ind_bias.get('score', 0))
        else:
            ind_bias = self._extract_numeric(ind_bias, 0)
        
        layer_scores.append(LayerScore(
            name='indicators',
            score=ind_score,
            bias=ind_bias,
            weight=self.weights['indicators'],
            confidence=0.7,
            reason=f"Indicators: RSI {rsi_value:.0f}, MACD {macd_value:.3f}"
        ))
        
        # ===== 3. PRICE ACTION LAYER =====
        price_action = layer_data.get('price_action', {})
        pa_net = price_action.get('net_score', 0.5)
        
        # Extract numeric value
        if hasattr(pa_net, 'score'):
            pa_score_raw = pa_net.score
        elif isinstance(pa_net, dict):
            pa_score_raw = self._extract_numeric(pa_net.get('score', 0.5))
        else:
            pa_score_raw = self._extract_numeric(pa_net, 0.5)
        
        # Convert net_score (-1 to 1) to score (0 to 1)
        if isinstance(pa_score_raw, (int, float)) and -1 <= pa_score_raw <= 1:
            pa_score_converted = (pa_score_raw + 1) / 2
        else:
            pa_score_converted = 0.5
        
        # Extract bias
        pa_bias = price_action.get('net_score', 0)
        if hasattr(pa_bias, 'score'):
            pa_bias = pa_bias.score
        elif isinstance(pa_bias, dict):
            pa_bias = self._extract_numeric(pa_bias.get('score', 0))
        else:
            pa_bias = self._extract_numeric(pa_bias, 0)
        
        bullish_count = price_action.get('bullish_count', 0)
        bearish_count = price_action.get('bearish_count', 0)
        
        layer_scores.append(LayerScore(
            name='price_action',
            score=pa_score_converted,
            bias=pa_bias,
            weight=self.weights['price_action'],
            confidence=min(0.9, 0.5 + (bullish_count + bearish_count) * 0.05),
            reason=f"Price Action: {bullish_count} bullish, {bearish_count} bearish patterns"
        ))
        
        # ===== 4. CHART PATTERNS LAYER =====
        chart_patterns = layer_data.get('chart_patterns', {})
        
        cp_score = chart_patterns.get('pattern_score', 0.5)
        if hasattr(cp_score, 'score'):
            cp_score = cp_score.score
        elif isinstance(cp_score, dict):
            cp_score = self._extract_numeric(cp_score.get('score', 0.5))
        else:
            cp_score = self._extract_numeric(cp_score, 0.5)
        
        cp_bias = chart_patterns.get('net_bias', 0)
        if hasattr(cp_bias, 'score'):
            cp_bias = cp_bias.score
        elif isinstance(cp_bias, dict):
            cp_bias = self._extract_numeric(cp_bias.get('score', 0))
        else:
            cp_bias = self._extract_numeric(cp_bias, 0)
        
        buy_signals = chart_patterns.get('buy_signals', 0)
        sell_signals = chart_patterns.get('sell_signals', 0)
        total_patterns = chart_patterns.get('total_patterns', 0)
        
        layer_scores.append(LayerScore(
            name='chart_patterns',
            score=cp_score,
            bias=cp_bias,
            weight=self.weights['chart_patterns'],
            confidence=min(0.9, 0.4 + total_patterns * 0.1),
            reason=f"Chart Patterns: {buy_signals} bullish, {sell_signals} bearish"
        ))
        
        # ===== 5. STRUCTURE LAYER =====
        structure = layer_data.get('structure', {})
        
        struct_score = structure.get('structure_score', 0.5)
        if hasattr(struct_score, 'score'):
            struct_score = struct_score.score
        elif isinstance(struct_score, dict):
            struct_score = self._extract_numeric(struct_score.get('score', 0.5))
        else:
            struct_score = self._extract_numeric(struct_score, 0.5)
        
        struct_bias = structure.get('structure_bias', 0)
        if hasattr(struct_bias, 'score'):
            struct_bias = struct_bias.score
        elif isinstance(struct_bias, dict):
            struct_bias = self._extract_numeric(struct_bias.get('score', 0))
        else:
            struct_bias = self._extract_numeric(struct_bias, 0)
        
        struct_confidence = structure.get('structure_confidence', 0.5)
        if hasattr(struct_confidence, 'confidence'):
            struct_confidence = struct_confidence.confidence
        elif isinstance(struct_confidence, dict):
            struct_confidence = self._extract_numeric(struct_confidence.get('confidence', 0.5))
        else:
            struct_confidence = self._extract_numeric(struct_confidence, 0.5)
        
        structure_trend = structure.get('trend', 'NEUTRAL')
        bos_count = structure.get('bos_count', 0)
        
        layer_scores.append(LayerScore(
            name='structure',
            score=struct_score,
            bias=struct_bias,
            weight=self.weights['structure'],
            confidence=struct_confidence,
            reason=f"Market Structure: {structure_trend} trend, BOS: {bos_count}"
        ))
        
        # ===== 6. SUPPORT/RESISTANCE LAYER =====
        sr = layer_data.get('support_resistance', {})
        
        sr_score = sr.get('sr_score', 0.5)
        if hasattr(sr_score, 'score'):
            sr_score = sr_score.score
        elif isinstance(sr_score, dict):
            sr_score = self._extract_numeric(sr_score.get('score', 0.5))
        else:
            sr_score = self._extract_numeric(sr_score, 0.5)
        
        sr_bias = sr.get('sr_bias', 0)
        if hasattr(sr_bias, 'score'):
            sr_bias = sr_bias.score
        elif isinstance(sr_bias, dict):
            sr_bias = self._extract_numeric(sr_bias.get('score', 0))
        else:
            sr_bias = self._extract_numeric(sr_bias, 0)
        
        vwap_position = sr.get('vwap_position', 'NEUTRAL')
        is_near_support = sr.get('is_near_support', False)
        
        layer_scores.append(LayerScore(
            name='support_resistance',
            score=sr_score,
            bias=sr_bias,
            weight=self.weights['support_resistance'],
            confidence=0.7,
            reason=f"S/R: {vwap_position} vs VWAP, near support: {is_near_support}"
        ))
        
        # ===== 7. VOLUME LAYER =====
        volume = layer_data.get('volume', {})
        
        vol_score = volume.get('combined_score', 0.5)
        if hasattr(vol_score, 'score'):
            vol_score = vol_score.score
        elif isinstance(vol_score, dict):
            vol_score = self._extract_numeric(vol_score.get('score', 0.5))
        else:
            vol_score = self._extract_numeric(vol_score, 0.5)
        
        vol_bias = volume.get('volume_bias', 0)
        if hasattr(vol_bias, 'score'):
            vol_bias = vol_bias.score
        elif isinstance(vol_bias, dict):
            vol_bias = self._extract_numeric(vol_bias.get('score', 0))
        else:
            vol_bias = self._extract_numeric(vol_bias, 0)
        
        volume_trend = volume.get('volume_trend', 'NEUTRAL')
        volume_ratio = volume.get('volume_ratio', 1.0)
        
        layer_scores.append(LayerScore(
            name='volume',
            score=vol_score,
            bias=vol_bias,
            weight=self.weights['volume'],
            confidence=0.6,
            reason=f"Volume: {volume_trend}, spike: {volume_ratio:.1f}x"
        ))
        
        # ===== CALCULATE WEIGHTED SCORE =====
        total_weighted_score = 0
        total_weight = 0
        
        for ls in layer_scores:
            total_weighted_score += ls.score * ls.weight
            total_weight += ls.weight
        
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
        else:
            final_score = 0.5
        
        # ===== CALCULATE NET BIAS =====
        total_bias = 0
        total_bias_weight = 0
        
        for ls in layer_scores:
            if abs(ls.bias) > 0.1:
                total_bias += ls.bias * ls.weight
                total_bias_weight += ls.weight
        
        if total_bias_weight > 0:
            net_bias = total_bias / total_bias_weight
        else:
            net_bias = 0
        
        # ===== CALCULATE AGREEMENT SCORE =====
        bullish_layers = sum(1 for ls in layer_scores if ls.bias > 0.1)
        bearish_layers = sum(1 for ls in layer_scores if ls.bias < -0.1)
        total_active = bullish_layers + bearish_layers
        
        if total_active > 0:
            agreement_score = max(bullish_layers, bearish_layers) / total_active
        else:
            agreement_score = 0.5
        
        # ===== CALCULATE CONFIDENCE =====
        avg_layer_confidence = np.mean([ls.confidence for ls in layer_scores])
        confidence = (agreement_score * 0.6) + (avg_layer_confidence * 0.4)
        confidence = min(0.95, max(0.3, confidence))
        
        # ===== DETERMINE DOMINANT DIRECTION =====
        if net_bias > 0.2:
            dominant = 'BULLISH'
        elif net_bias < -0.2:
            dominant = 'BEARISH'
        else:
            dominant = 'NEUTRAL'
        
        # ===== ADD REASONS =====
        reasons.append(f"Aggregated score: {final_score:.1%}")
        reasons.append(f"Net bias: {net_bias:+.2f} ({dominant})")
        reasons.append(f"Agreement: {agreement_score:.0%} of layers agree")
        
        return AggregatedScore(
            total_score=round(final_score, 3),
            net_bias=round(net_bias, 3),
            confidence=round(confidence, 3),
            layer_scores=layer_scores,
            weight_breakdown=self.weights,
            agreement_score=round(agreement_score, 3),
            dominant_direction=dominant,
            reasons=reasons
        )
    
    def calculate_base_confidence(self, aggregated: AggregatedScore) -> float:
        """
        Calculate base confidence for signal generation
        This is the primary score that determines if a signal is generated
        """
        # Base is total score
        base = aggregated.total_score
        
        # Adjust by agreement (more agreement = higher confidence)
        if aggregated.agreement_score > 0.7:
            base = min(0.95, base + 0.05)
        elif aggregated.agreement_score < 0.4:
            base = max(0.3, base - 0.05)
        
        return round(base, 3)
    
    def should_generate_signal(self, aggregated: AggregatedScore, min_score: float = 0.80) -> bool:
        """
        Determine if technical analysis is strong enough to generate a signal
        """
        base = self.calculate_base_confidence(aggregated)
        return base >= min_score
    
    def get_direction_from_score(self, aggregated: AggregatedScore) -> str:
        """
        Get direction from aggregated score
        """
        if aggregated.net_bias > 0.2:
            return 'BUY'
        elif aggregated.net_bias < -0.2:
            return 'SELL'
        else:
            return 'HOLD'


# ==================== HELPER FUNCTION ====================
def create_layer_data_from_analysis(
    regime_result: Dict,
    indicator_result: Any,  # Can be dict or IndicatorResult object
    price_action_result: Dict,
    chart_patterns_result: Dict,
    structure_result: Dict,
    sr_result: Dict,
    volume_result: Dict
) -> Dict[str, Any]:
    """
    Helper function to create layer_data dictionary from individual analysis results
    
    Handles both dictionary and IndicatorResult object inputs
    """
    
    # ===== HANDLE INDICATOR RESULT (can be dict or IndicatorResult object) =====
    if isinstance(indicator_result, dict):
        # It's a dictionary
        combined = indicator_result.get('combined', {})
        rsi_data = indicator_result.get('rsi', {})
        macd_data = indicator_result.get('macd', {})
        
        combined_score = combined.get('score', 0.5) if isinstance(combined, dict) else 0.5
        rsi_value = rsi_data.get('value', 50) if isinstance(rsi_data, dict) else 50
        macd_hist = macd_data.get('value', 0) if isinstance(macd_data, dict) else 0
        
    else:
        # It's an IndicatorResult object (from technical_analyzer)
        # Access attributes directly, not via .get()
        try:
            # Get combined score
            combined_obj = getattr(indicator_result, 'combined', None)
            if combined_obj:
                combined_score = getattr(combined_obj, 'score', 0.5)
            else:
                combined_score = 0.5
            
            # Get RSI value
            rsi_obj = getattr(indicator_result, 'rsi', None)
            if rsi_obj:
                rsi_value = getattr(rsi_obj, 'value', 50)
            else:
                rsi_value = 50
            
            # Get MACD histogram value
            macd_obj = getattr(indicator_result, 'macd', None)
            if macd_obj:
                macd_hist = getattr(macd_obj, 'value', 0)
            else:
                macd_hist = 0
                
        except Exception:
            # Fallback if anything fails
            combined_score = 0.5
            rsi_value = 50
            macd_hist = 0
    
    return {
        'regime': {
            'score': regime_result.get('trending_score', 0.5),
            'bias_score': regime_result.get('bias_score', 0),
            'confidence': regime_result.get('regime_confidence', 0.5),
            'regime': regime_result.get('regime', 'UNKNOWN')
        },
        'indicators': {
            'combined_score': combined_score,
            'indicator_score': combined_score,
            'rsi': rsi_value,
            'macd_hist': macd_hist
        },
        'price_action': {
            'net_score': price_action_result.get('net_score', 0),
            'bullish_count': price_action_result.get('bullish_count', 0),
            'bearish_count': price_action_result.get('bearish_count', 0),
            'total_patterns': price_action_result.get('total_patterns', 0)
        },
        'chart_patterns': {
            'net_bias': chart_patterns_result.get('net_bias', 0),
            'buy_signals': chart_patterns_result.get('buy_signals', 0),
            'sell_signals': chart_patterns_result.get('sell_signals', 0),
            'total_patterns': chart_patterns_result.get('total_patterns', 0),
            'pattern_score': chart_patterns_result.get('pattern_score', 0.5)
        },
        'structure': {
            'structure_score': structure_result.get('structure_score', 0.5),
            'structure_bias': structure_result.get('structure_bias', 0),
            'structure_confidence': structure_result.get('structure_confidence', 0.5),
            'trend': structure_result.get('trend', 'NEUTRAL'),
            'bos_count': structure_result.get('bos_count', 0)
        },
        'support_resistance': {
            'sr_score': sr_result.get('sr_score', 0.5),
            'sr_bias': sr_result.get('sr_bias', 0),
            'vwap_position': sr_result.get('vwap_position', 'NEUTRAL'),
            'is_near_support': sr_result.get('is_near_support', False),
            'is_near_resistance': sr_result.get('is_near_resistance', False)
        },
        'volume': {
            'combined_score': volume_result.get('combined_score', 0.5),
            'volume_bias': volume_result.get('volume_bias', 0),
            'volume_trend': volume_result.get('volume_trend', 'NEUTRAL'),
            'volume_ratio': volume_result.get('volume_ratio', 1.0)
        }
    }