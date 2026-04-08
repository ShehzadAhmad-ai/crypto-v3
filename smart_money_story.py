# smart_money_story.py
"""
Smart Money Story Builder
Generates human-readable narrative about institutional activity
Combines all Smart Money components into a cohesive story
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SmartMoneyStory:
    """Complete story for Smart Money analysis"""
    summary: str                           # One-line summary
    full_story: str                        # Full narrative
    key_points: List[str]                  # Bullet points
    confidence_statement: str              # Confidence level statement
    risk_warnings: List[str]               # Any warnings
    timestamp: datetime = field(default_factory=datetime.now)


class SmartMoneyStoryBuilder:
    """
    Builds human-readable narrative from Smart Money analysis
    Explains institutional footprints and what they mean
    """
    
    def __init__(self):
        self.templates = {
            'liquidity_sweep_bullish': "Liquidity sweep below support detected with reversal. Smart money stopped out weak hands.",
            'liquidity_sweep_bearish': "Liquidity sweep above resistance detected with reversal. Smart money trapped buyers.",
            'absorption_bullish': "Absorption detected at lows. Institutions are accumulating quietly.",
            'absorption_bearish': "Absorption detected at highs. Institutions are distributing quietly.",
            'exhaustion_bullish': "Exhaustion volume at bottom. Selling pressure exhausted, reversal likely.",
            'exhaustion_bearish': "Exhaustion volume at top. Buying pressure exhausted, reversal likely.",
            'order_block_bullish': "Bullish order block at {price} acting as support.",
            'order_block_bearish': "Bearish order block at {price} acting as resistance.",
            'fvg_bullish': "Unfilled bullish fair value gap below price - likely to be filled.",
            'fvg_bearish': "Unfilled bearish fair value gap above price - likely to be filled.",
            'liquidations_bullish': "Short liquidations of ${value} driving price up.",
            'liquidations_bearish': "Long liquidations of ${value} driving price down.",
            'cascade_risk': "Liquidation clusters nearby - cascade risk {risk:.0%}.",
            'market_maker_bullish': "Market makers are passively buying, absorbing sell orders.",
            'market_maker_bearish': "Market makers are passively selling, absorbing buy orders.",
            'regime_aligned': "Smart money bias aligns with {regime} market regime.",
            'regime_misaligned': "Smart money bias conflicts with {regime} market regime - caution advised.",
            'confidence': "Overall confidence is {confidence:.0%} with {agreement:.0%} of components agreeing."
        }
        
        self.bias_map = {
            'STRONG_BULLISH': "strongly bullish",
            'BULLISH': "bullish",
            'NEUTRAL': "neutral",
            'BEARISH': "bearish",
            'STRONG_BEARISH': "strongly bearish"
        }
    
    def build_story(self, component_data: Dict[str, Any], 
                    scoring_result: Any) -> SmartMoneyStory:
        """
        Build complete story from Smart Money analysis
        
        Args:
            component_data: Dictionary with all component results
            scoring_result: SmartMoneyResult from scoring module
        
        Returns:
            SmartMoneyStory object with human-readable narrative
        """
        key_points = []
        story_parts = []
        warnings = []
        
        # ===== 1. LIQUIDITY STORY =====
        liquidity = component_data.get('liquidity', {})
        liquidity_score = liquidity.get('score', 0.5)
        liquidity_bias = liquidity.get('net_bias', 0)
        sweeps = liquidity.get('sweeps', [])
        stop_hunt_prob = liquidity.get('stop_hunt_probability', 0)
        
        if sweeps:
            # Check most recent sweep
            last_sweep = sweeps[-1]
            sweep_type = last_sweep.get('type', '')
            sweep_strength = last_sweep.get('strength', 0.5)
            
            if 'DOWN' in sweep_type and sweep_strength > 0.6:
                story_parts.append(self.templates['liquidity_sweep_bullish'])
                key_points.append("Liquidity sweep below support (bullish)")
            elif 'UP' in sweep_type and sweep_strength > 0.6:
                story_parts.append(self.templates['liquidity_sweep_bearish'])
                key_points.append("Liquidity sweep above resistance (bearish)")
        
        if stop_hunt_prob > 0.6:
            key_points.append(f"High stop hunt probability ({stop_hunt_prob:.0%})")
        
        # ===== 2. ORDER FLOW STORY =====
        orderflow = component_data.get('orderflow', {})
        absorption_count = orderflow.get('absorption_count', 0)
        exhaustion_count = orderflow.get('exhaustion_count', 0)
        latest_delta = orderflow.get('latest_delta', {})
        
        if absorption_count > 0:
            if liquidity_bias > 0:
                story_parts.append(self.templates['absorption_bullish'])
                key_points.append(f"{absorption_count} absorption pattern(s) detected (accumulation)")
            else:
                story_parts.append(self.templates['absorption_bearish'])
                key_points.append(f"{absorption_count} absorption pattern(s) detected (distribution)")
        
        if exhaustion_count > 0:
            if liquidity_bias > 0:
                story_parts.append(self.templates['exhaustion_bullish'])
                key_points.append(f"{exhaustion_count} exhaustion pattern(s) detected (reversal)")
            else:
                story_parts.append(self.templates['exhaustion_bearish'])
                key_points.append(f"{exhaustion_count} exhaustion pattern(s) detected (reversal)")
        
        if latest_delta:
            delta_dir = latest_delta.get('direction', 'NEUTRAL')
            delta_strength = latest_delta.get('strength', 0)
            if delta_dir == 'BUY' and delta_strength > 0.6:
                key_points.append(f"Strong buying pressure in recent candles")
            elif delta_dir == 'SELL' and delta_strength > 0.6:
                key_points.append(f"Strong selling pressure in recent candles")
        
        # ===== 3. MARKET STRUCTURE STORY =====
        market_structure = component_data.get('market_structure', {})
        order_blocks = market_structure.get('order_blocks', [])
        structure_trend = market_structure.get('trend', 'NEUTRAL')
        unmitigated_bullish = market_structure.get('unmitigated_bullish_obs', 0)
        unmitigated_bearish = market_structure.get('unmitigated_bearish_obs', 0)
        
        # Order blocks
        if order_blocks:
            # Find strongest order block
            strong_obs = [ob for ob in order_blocks if ob.get('strength_score', 0) > 0.6]
            if strong_obs:
                strongest = strong_obs[0]
                ob_price = strongest.get('price', 0)
                ob_type = strongest.get('type', '')
                if ob_type == 'BULLISH_OB':
                    story_parts.append(self.templates['order_block_bullish'].format(price=ob_price))
                    key_points.append(f"Bullish order block at {ob_price:.4f}")
                else:
                    story_parts.append(self.templates['order_block_bearish'].format(price=ob_price))
                    key_points.append(f"Bearish order block at {ob_price:.4f}")
        
        # Unmitigated order blocks
        if unmitigated_bullish > 0:
            key_points.append(f"{unmitigated_bullish} unmitigated bullish order blocks")
        if unmitigated_bearish > 0:
            key_points.append(f"{unmitigated_bearish} unmitigated bearish order blocks")
        
        # ===== 4. MICROSTRUCTURE STORY =====
        microstructure = component_data.get('microstructure', {})
        fvg = microstructure.get('fvg', {})
        unfilled_bullish = fvg.get('unfilled_bullish', 0)
        unfilled_bearish = fvg.get('unfilled_bearish', 0)
        displacement = microstructure.get('displacement', {})
        displacement_count = displacement.get('count', 0)
        structure_health = microstructure.get('structure_health', 'UNKNOWN')
        
        if unfilled_bullish > 0:
            story_parts.append(self.templates['fvg_bullish'])
            key_points.append(f"{unfilled_bullish} unfilled bullish FVGs")
        if unfilled_bearish > 0:
            story_parts.append(self.templates['fvg_bearish'])
            key_points.append(f"{unfilled_bearish} unfilled bearish FVGs")
        
        if displacement_count > 0:
            key_points.append(f"{displacement_count} displacement move(s) detected")
        
        if structure_health == 'CLEAN':
            key_points.append("Clean market structure")
        elif structure_health == 'BREAKING':
            key_points.append("Structure breaking down")
            warnings.append("Market structure is breaking down - wait for confirmation")
        
        # ===== 5. LIQUIDATIONS STORY =====
        liquidations = component_data.get('liquidations', {})
        short_liq_value = liquidations.get('short_liq_value', 0)
        long_liq_value = liquidations.get('long_liq_value', 0)
        total_liq = liquidations.get('total_liq_value', 0)
        cascade_risk = liquidations.get('cascade_risk', 0)
        dominant = liquidations.get('dominant', 'NONE')
        
        if total_liq > 100000:
            if dominant == 'SHORT_LIQUIDATIONS':
                story_parts.append(self.templates['liquidations_bullish'].format(value=short_liq_value))
                key_points.append(f"Short liquidations: ${short_liq_value:,.0f}")
            elif dominant == 'LONG_LIQUIDATIONS':
                story_parts.append(self.templates['liquidations_bearish'].format(value=long_liq_value))
                key_points.append(f"Long liquidations: ${long_liq_value:,.0f}")
        
        if cascade_risk > 0.5:
            story_parts.append(self.templates['cascade_risk'].format(risk=cascade_risk))
            warnings.append(f"High liquidation cascade risk ({cascade_risk:.0%})")
        
        # ===== 6. MARKET MAKER STORY =====
        market_maker = component_data.get('market_maker', {})
        mm_activity = market_maker.get('activity', 'NEUTRAL')
        mm_imbalance = market_maker.get('inventory_imbalance', 0)
        
        if 'BUYING' in mm_activity or mm_imbalance > 0.3:
            story_parts.append(self.templates['market_maker_bullish'])
            key_points.append("Market makers showing bullish positioning")
        elif 'SELLING' in mm_activity or mm_imbalance < -0.3:
            story_parts.append(self.templates['market_maker_bearish'])
            key_points.append("Market makers showing bearish positioning")
        
        # ===== 7. REGIME ALIGNMENT STORY =====
        regime_alignment = component_data.get('regime_alignment', {})
        alignment_score = regime_alignment.get('score', 0.5)
        regime_bias = regime_alignment.get('regime_bias', 'NEUTRAL')
        
        if alignment_score > 0.7:
            story_parts.append(self.templates['regime_aligned'].format(regime=regime_bias))
            key_points.append(f"Aligned with {regime_bias} market regime")
        elif alignment_score < 0.3:
            story_parts.append(self.templates['regime_misaligned'].format(regime=regime_bias))
            warnings.append(f"Smart money conflicts with {regime_bias} regime")
        
        # ===== 8. CONFIDENCE STORY =====
        total_score = scoring_result.total_score if hasattr(scoring_result, 'total_score') else 0.5
        net_bias = scoring_result.net_bias if hasattr(scoring_result, 'net_bias') else 0
        agreement = scoring_result.agreement_score if hasattr(scoring_result, 'agreement_score') else 0.5
        bias_label = scoring_result.bias_label.value if hasattr(scoring_result, 'bias_label') else 'NEUTRAL'
        
        bias_text = self.bias_map.get(bias_label, 'neutral')
        
        story_parts.append(self.templates['confidence'].format(confidence=total_score, agreement=agreement))
        key_points.append(f"Confidence: {total_score:.0%} ({bias_text})")
        
        # ===== BUILD SUMMARY =====
        if net_bias > 0.2:
            summary = f"Smart Money is {bias_text}. {len(key_points)} institutional signals detected."
        elif net_bias < -0.2:
            summary = f"Smart Money is {bias_text}. {len(key_points)} institutional signals detected."
        else:
            summary = f"Smart Money is neutral. Mixed signals detected."
        
        # ===== BUILD FULL STORY =====
        full_story = " ".join(story_parts) if story_parts else "No significant Smart Money signals detected."
        
        # ===== ADD WARNINGS FROM SCORING =====
        if total_score < 0.4:
            warnings.append("Low Smart Money confidence - institutional signals weak")
        
        return SmartMoneyStory(
            summary=summary,
            full_story=full_story,
            key_points=key_points,
            confidence_statement=f"Smart Money Confidence: {total_score:.0%} ({bias_text})",
            risk_warnings=warnings
        )
    
    def get_summary_only(self, component_data: Dict[str, Any], 
                         scoring_result: Any) -> str:
        """Get one-line summary only"""
        story = self.build_story(component_data, scoring_result)
        return story.summary
    
    def print_story(self, story: SmartMoneyStory):
        """Print story in readable format"""
        print("\n" + "=" * 70)
        print(f"Smart Money Analysis: {story.summary}")
        print("=" * 70)
        print()
        print("Key Points:")
        print("-" * 50)
        for point in story.key_points[:8]:
            print(f"  - {point}")
        print()
        if story.full_story:
            print("Full Analysis:")
            print("-" * 50)
            print(f"  {story.full_story}")
            print()
        if story.risk_warnings:
            print("Warnings:")
            print("-" * 50)
            for warning in story.risk_warnings:
                print(f"  - {warning}")
        print()
        print("=" * 70)