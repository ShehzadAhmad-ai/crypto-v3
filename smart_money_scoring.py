# smart_money_scoring.py
"""
Smart Money Scoring Module
Aggregates all Smart Money component scores into final score
Combines:
- Liquidity Intelligence
- Order Flow Intelligence
- Market Structure (order blocks, mitigation)
- Microstructure (FVGs, displacement)
- Liquidation Intelligence
- Market Maker Intelligence
- Regime Alignment
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from config import Config
from logger import log


class SmartMoneyBias(Enum):
    """Smart Money directional bias"""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class ComponentScore:
    """Score for a single Smart Money component"""
    name: str
    score: float          # 0-1
    bias: float           # -1 to 1
    weight: float
    confidence: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmartMoneyResult:
    """Complete Smart Money analysis result"""
    total_score: float                    # 0-1
    net_bias: float                       # -1 to 1
    bias_label: SmartMoneyBias
    confidence: float                     # 0-1
    component_scores: List[ComponentScore]
    weight_breakdown: Dict[str, float]
    agreement_score: float                # How well components agree
    dominant_direction: str               # 'BULLISH', 'BEARISH', 'NEUTRAL'
    reasons: List[str]
    confidence_boost: float               # Amount to add to signal confidence


class SmartMoneyScoring:
    """
    Aggregates all Smart Money components into final score
    Weights are configurable from config.py
    """
    
    def __init__(self):
        # Default weights (sum to 1.0)
        self.default_weights = {
            'liquidity': 0.20,
            'orderflow': 0.20,
            'market_structure': 0.20,
            'microstructure': 0.15,
            'liquidations': 0.10,
            'market_maker': 0.10,
            'regime_alignment': 0.05
        }
        
        # Load weights from config
        self.weights = self._load_weights()
        
        # Minimum score to confirm Smart Money signal
        self.min_score = getattr(Config, 'SMART_MONEY_MIN_SCORE', 0.60)
        
        # Confidence boost factor
        self.confidence_boost_factor = getattr(Config, 'SM_CONFIDENCE_BOOST_FACTOR', 0.15)
        
        log.info("SmartMoneyScoring initialized")
        log.info(f"Weights: {self.weights}")
        log.info(f"Min Score: {self.min_score}")
    
    def _load_weights(self) -> Dict[str, float]:
        """Load weights from config.py"""
        try:
            weights = {}
            weights['liquidity'] = getattr(Config, 'SM_WEIGHT_LIQUIDITY', self.default_weights['liquidity'])
            weights['orderflow'] = getattr(Config, 'SM_WEIGHT_ORDERFLOW', self.default_weights['orderflow'])
            weights['market_structure'] = getattr(Config, 'SM_WEIGHT_MICROSTRUCTURE', self.default_weights['market_structure'])
            weights['microstructure'] = getattr(Config, 'SM_WEIGHT_MICROSTRUCTURE', self.default_weights['microstructure'])
            weights['liquidations'] = getattr(Config, 'SM_WEIGHT_LIQUIDATIONS', self.default_weights['liquidations'])
            weights['market_maker'] = getattr(Config, 'SM_WEIGHT_MARKET_MAKER', self.default_weights['market_maker'])
            weights['regime_alignment'] = getattr(Config, 'SM_WEIGHT_REGIME_ALIGNMENT', self.default_weights['regime_alignment'])
            
            # Normalize to sum to 1.0
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                for k in weights:
                    weights[k] /= total
            
            return weights
            
        except Exception:
            return self.default_weights
    
    def aggregate_scores(self, component_data: Dict[str, Any]) -> SmartMoneyResult:
        """
        Aggregate all Smart Money component scores
        
        Args:
            component_data: Dictionary with scores from all components
                Expected format:
                {
                    'liquidity': {
                        'score': 0.75,
                        'bias': 0.6,
                        'confidence': 0.8,
                        'reasons': [...]
                    },
                    'orderflow': {
                        'score': 0.82,
                        'bias': 0.7,
                        'confidence': 0.75,
                        'reasons': [...]
                    },
                    'market_structure': {
                        'score': 0.70,
                        'bias': 0.4,
                        'confidence': 0.7,
                        'reasons': [...]
                    },
                    'microstructure': {
                        'score': 0.65,
                        'bias': 0.3,
                        'confidence': 0.65,
                        'reasons': [...]
                    },
                    'liquidations': {
                        'score': 0.85,
                        'bias': 0.8,
                        'confidence': 0.7,
                        'reasons': [...]
                    },
                    'market_maker': {
                        'score': 0.60,
                        'bias': 0.2,
                        'confidence': 0.6,
                        'reasons': [...]
                    },
                    'regime_alignment': {
                        'score': 0.90,
                        'bias': 0.9,
                        'confidence': 0.85,
                        'reasons': [...]
                    }
                }
        
        Returns:
            SmartMoneyResult with final score and details
        """
        component_scores = []
        all_reasons = []
        
        # ===== 1. LIQUIDITY COMPONENT =====
        liquidity = component_data.get('liquidity', {})
        liquidity_score = liquidity.get('score', 0.5)
        liquidity_bias = liquidity.get('net_bias', liquidity.get('bias', 0))
        liquidity_conf = liquidity.get('confidence', 0.5)
        
        component_scores.append(ComponentScore(
            name='liquidity',
            score=liquidity_score,
            bias=liquidity_bias,
            weight=self.weights['liquidity'],
            confidence=liquidity_conf,
            reason=f"Liquidity: {liquidity.get('direction', 'NEUTRAL')} | Stop hunt: {liquidity.get('stop_hunt_probability', 0):.0%}",
            details=liquidity
        ))
        all_reasons.extend(liquidity.get('reasons', [])[:2])
        
        # ===== 2. ORDER FLOW COMPONENT =====
        orderflow = component_data.get('orderflow', {})
        of_score = orderflow.get('score', 0.5)
        of_bias = orderflow.get('net_bias', orderflow.get('bias', 0))
        of_conf = orderflow.get('confidence', 0.5)
        
        component_scores.append(ComponentScore(
            name='orderflow',
            score=of_score,
            bias=of_bias,
            weight=self.weights['orderflow'],
            confidence=of_conf,
            reason=f"Order Flow: {orderflow.get('direction', 'NEUTRAL')} | Delta: {orderflow.get('latest_delta', {}).get('direction', 'NONE')}",
            details=orderflow
        ))
        all_reasons.extend(orderflow.get('reasons', [])[:2])
        
        # ===== 3. MARKET STRUCTURE COMPONENT =====
        market_structure = component_data.get('market_structure', {})
        ms_score = market_structure.get('structure_score', 0.5)
        ms_bias = market_structure.get('structure_bias', 0)
        ms_conf = market_structure.get('structure_confidence', 0.5)
        
        component_scores.append(ComponentScore(
            name='market_structure',
            score=ms_score,
            bias=ms_bias,
            weight=self.weights['market_structure'],
            confidence=ms_conf,
            reason=f"Structure: {market_structure.get('trend', 'NEUTRAL')} | OBs: {market_structure.get('order_blocks_count', 0)}",
            details=market_structure
        ))
        all_reasons.extend(market_structure.get('reasons', [])[:2])
        
        # ===== 4. MICROSTRUCTURE COMPONENT =====
        microstructure = component_data.get('microstructure', {})
        micro_score = microstructure.get('score', 0.5)
        micro_bias = microstructure.get('net_bias', microstructure.get('bias', 0))
        micro_conf = microstructure.get('composite_confidence', 0.5)
        
        component_scores.append(ComponentScore(
            name='microstructure',
            score=micro_score,
            bias=micro_bias,
            weight=self.weights['microstructure'],
            confidence=micro_conf,
            reason=f"Microstructure: {microstructure.get('direction', 'NEUTRAL')} | FVGs: {microstructure.get('fvg', {}).get('unfilled_bullish', 0)}B/{microstructure.get('fvg', {}).get('unfilled_bearish', 0)}S",
            details=microstructure
        ))
        all_reasons.extend(microstructure.get('reasons', [])[:2])
        
        # ===== 5. LIQUIDATIONS COMPONENT =====
        liquidations = component_data.get('liquidations', {})
        liq_score = liquidations.get('score', 0.5)
        liq_bias = 0
        if liquidations.get('bias') == 'BULLISH':
            liq_bias = 0.6
        elif liquidations.get('bias') == 'BEARISH':
            liq_bias = -0.6
        liq_conf = liquidations.get('confidence', 0.5)
        
        component_scores.append(ComponentScore(
            name='liquidations',
            score=liq_score,
            bias=liq_bias,
            weight=self.weights['liquidations'],
            confidence=liq_conf,
            reason=f"Liquidations: {liquidations.get('dominant', 'NONE')} | Total: ${liquidations.get('total_liq_value', 0):,.0f}",
            details=liquidations
        ))
        all_reasons.extend(liquidations.get('reasons', [])[:2])
        
        # ===== 6. MARKET MAKER COMPONENT =====
        market_maker = component_data.get('market_maker', {})
        mm_score = market_maker.get('score', 0.5)
        mm_bias = 0
        if market_maker.get('bias') == 'BULLISH':
            mm_bias = 0.5
        elif market_maker.get('bias') == 'BEARISH':
            mm_bias = -0.5
        mm_conf = market_maker.get('confidence', 0.5)
        
        component_scores.append(ComponentScore(
            name='market_maker',
            score=mm_score,
            bias=mm_bias,
            weight=self.weights['market_maker'],
            confidence=mm_conf,
            reason=f"Market Maker: {market_maker.get('activity', 'NEUTRAL')} | Imbalance: {market_maker.get('inventory_imbalance', 0):.2f}",
            details=market_maker
        ))
        all_reasons.extend(market_maker.get('reasons', [])[:2])
        
        # ===== 7. REGIME ALIGNMENT COMPONENT =====
        regime_alignment = component_data.get('regime_alignment', {})
        ra_score = regime_alignment.get('score', 0.5)
        ra_bias = regime_alignment.get('bias', 0)
        ra_conf = regime_alignment.get('confidence', 0.5)
        
        component_scores.append(ComponentScore(
            name='regime_alignment',
            score=ra_score,
            bias=ra_bias,
            weight=self.weights['regime_alignment'],
            confidence=ra_conf,
            reason=f"Regime Alignment: {regime_alignment.get('alignment', 0):.0%} aligned",
            details=regime_alignment
        ))
        all_reasons.extend(regime_alignment.get('reasons', [])[:2])
        
        # ===== CALCULATE WEIGHTED SCORE =====
        total_weighted_score = 0
        total_bias_weighted = 0
        total_weight = 0
        
        for cs in component_scores:
            total_weighted_score += cs.score * cs.weight
            total_bias_weighted += cs.bias * cs.weight
            total_weight += cs.weight
        
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
            net_bias = total_bias_weighted / total_weight
        else:
            final_score = 0.5
            net_bias = 0
        
        # ===== CALCULATE AGREEMENT SCORE =====
        # How many components agree on direction?
        bullish_components = sum(1 for cs in component_scores if cs.bias > 0.1)
        bearish_components = sum(1 for cs in component_scores if cs.bias < -0.1)
        total_active = bullish_components + bearish_components
        
        if total_active > 0:
            agreement_score = max(bullish_components, bearish_components) / total_active
        else:
            agreement_score = 0.5
        
        # ===== CALCULATE CONFIDENCE =====
        avg_component_confidence = sum(cs.confidence for cs in component_scores) / len(component_scores)
        confidence = (agreement_score * 0.6) + (avg_component_confidence * 0.4)
        confidence = min(0.95, max(0.3, confidence))
        
        # ===== DETERMINE BIAS LABEL =====
        if net_bias > 0.6:
            bias_label = SmartMoneyBias.STRONG_BULLISH
            dominant = 'STRONG_BULLISH'
        elif net_bias > 0.2:
            bias_label = SmartMoneyBias.BULLISH
            dominant = 'BULLISH'
        elif net_bias < -0.6:
            bias_label = SmartMoneyBias.STRONG_BEARISH
            dominant = 'STRONG_BEARISH'
        elif net_bias < -0.2:
            bias_label = SmartMoneyBias.BEARISH
            dominant = 'BEARISH'
        else:
            bias_label = SmartMoneyBias.NEUTRAL
            dominant = 'NEUTRAL'
        
        # ===== CALCULATE CONFIDENCE BOOST =====
        if final_score >= self.min_score and abs(net_bias) > 0.2:
            # Boost based on score and agreement
            boost = final_score * self.confidence_boost_factor
            boost = min(0.15, boost)  # Max 15% boost
        else:
            boost = 0
        
        # ===== DETERMINE IF CONFIRMED =====
        confirmed = final_score >= self.min_score
        
        # ===== BUILD REASONS =====
        reasons = [
            f"Smart Money Score: {final_score:.1%}",
            f"Net Bias: {net_bias:+.2f} ({dominant})",
            f"Agreement: {agreement_score:.0%} of components agree",
            f"Confidence: {confidence:.1%}"
        ]
        reasons.extend(all_reasons[:5])
        
        return SmartMoneyResult(
            total_score=round(final_score, 3),
            net_bias=round(net_bias, 3),
            bias_label=bias_label,
            confidence=round(confidence, 3),
            component_scores=component_scores,
            weight_breakdown=self.weights,
            agreement_score=round(agreement_score, 3),
            dominant_direction=dominant,
            reasons=reasons,
            confidence_boost=round(boost, 3)
        )
    
    def get_component_summary(self, component_scores: List[ComponentScore]) -> Dict[str, Any]:
        """Get summary of component scores for display"""
        summary = {}
        for cs in component_scores:
            summary[cs.name] = {
                'score': cs.score,
                'bias': cs.bias,
                'weight': cs.weight,
                'confidence': cs.confidence,
                'reason': cs.reason
            }
        return summary
    
    def should_confirm(self, result: SmartMoneyResult) -> bool:
        """Determine if Smart Money confirms the signal"""
        return result.total_score >= self.min_score and abs(result.net_bias) > 0.2
    
    def get_direction(self, result: SmartMoneyResult) -> str:
        """Get direction from Smart Money analysis"""
        if result.net_bias > 0.2:
            return 'BUY'
        elif result.net_bias < -0.2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def print_summary(self, result: SmartMoneyResult):
        """Print formatted summary of Smart Money analysis"""
        print("\n" + "=" * 70)
        print(" SMART MONEY ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total Score: {result.total_score:.1%}")
        print(f"Net Bias: {result.net_bias:+.2f} ({result.dominant_direction})")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Agreement: {result.agreement_score:.0%}")
        print("-" * 70)
        print("Component Scores:")
        for cs in result.component_scores:
            bias_icon = "🟢" if cs.bias > 0 else ("🔴" if cs.bias < 0 else "⚪")
            print(f"  {bias_icon} {cs.name:18s}: {cs.score:.1%} (bias: {cs.bias:+.2f})")
        print("-" * 70)
        print("Key Reasons:")
        for reason in result.reasons[:5]:
            print(f"  • {reason}")
        print("=" * 70)