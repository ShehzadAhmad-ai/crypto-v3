# story_builder.py
"""
Story Building System (Layer 10)
Generates human-readable narrative explaining why a signal was generated
Combines all analysis layers into a cohesive story like a human trader
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Story:
    """Complete story for a trading signal"""
    summary: str                           # One-line summary
    full_story: str                        # Full narrative
    key_points: List[str]                  # Bullet points
    confidence_statement: str              # Confidence level statement
    risk_warnings: List[str]               # Any warnings
    timestamp: datetime = field(default_factory=datetime.now)


class StoryBuilder:
    """
    Builds human-readable narrative from technical analysis
    Like a human trader explaining their analysis
    """
    
    def __init__(self):
        self.templates = {
            'bullish_trend': "Price is in a {strength} uptrend, confirmed by {confirmations}.",
            'bearish_trend': "Price is in a {strength} downtrend, confirmed by {confirmations}.",
            'pullback': "Price has pulled back to {level} support, offering a discount entry.",
            'breakout': "Price is breaking out above {level} resistance with {volume} volume.",
            'pattern_bullish': "A {pattern} pattern has formed, suggesting upside continuation.",
            'pattern_bearish': "A {pattern} pattern has formed, suggesting downside continuation.",
            'volume_confirmation': "Volume is {trend} at {ratio}x average, confirming the move.",
            'divergence_bullish': "Bullish divergence detected: price made lower lows but RSI made higher lows.",
            'divergence_bearish': "Bearish divergence detected: price made higher highs but RSI made lower highs.",
            'structure_bullish': "Market structure is bullish with higher highs and higher lows.",
            'structure_bearish': "Market structure is bearish with lower highs and lower lows.",
            'support': "Price is holding at {level} support, a key level.",
            'resistance': "Price is approaching {level} resistance, a key level.",
            'vwap_discount': "Price is trading at a {level} discount to VWAP, offering value.",
            'vwap_premium': "Price is trading at a {level} premium to VWAP, indicating overextension.",
            'entry_zone': "Entry zone is between {low} and {high}.",
            'stop_loss': "Stop loss is placed at {level} below/above the structure.",
            'take_profit': "Take profit target is at {level} (next resistance/support).",
            'risk_reward': "Risk/Reward ratio is {rr}:1, which is {quality}.",
            'confidence': "Overall confidence is {confidence:.0%} based on {agreement} layers agreeing."
        }
        
        self.strength_map = {
            0.8: "strong",
            0.6: "moderate",
            0.4: "weak",
            0.2: "very weak"
        }
        
        self.quality_map = {
            3.0: "excellent",
            2.5: "very good",
            2.0: "good",
            1.5: "acceptable",
            1.0: "poor"
        }
    
    def build_story(self, 
                    direction: str,
                    aggregated_score: Any,
                    layer_data: Dict[str, Any],
                    signal_data: Dict[str, Any]) -> Story:
        """
        Build complete story from all analysis
        
        Args:
            direction: 'BUY' or 'SELL'
            aggregated_score: Output from LayerScoring.aggregate_scores()
            layer_data: Raw layer data from all analysis
            signal_data: Signal data with entry, stop, target, RR
        
        Returns:
            Story object with human-readable narrative
        """
        key_points = []
        story_parts = []
        
        # ===== 1. REGIME STORY =====
        regime = layer_data.get('regime', {})
        regime_name = regime.get('regime', 'UNKNOWN')
        regime_score = regime.get('score', 0.5)
        regime_strength = self._get_strength_text(regime_score)
        
        if 'BULL' in regime_name:
            story_parts.append(f"Market is in a {regime_strength} uptrend.")
            key_points.append(f"📈 {regime_strength.title()} uptrend confirmed")
        elif 'BEAR' in regime_name:
            story_parts.append(f"Market is in a {regime_strength} downtrend.")
            key_points.append(f"📉 {regime_strength.title()} downtrend confirmed")
        else:
            story_parts.append("Market is ranging with no clear trend.")
            key_points.append("🔄 Market ranging, mean reversion approach")
        
        # ===== 2. INDICATOR STORY =====
        indicators = layer_data.get('indicators', {})
        ind_score = indicators.get('combined_score', 0.5)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd_hist', 0)
        
        indicator_parts = []
        if direction == 'BUY':
            if rsi > 50:
                indicator_parts.append(f"RSI at {rsi:.0f} (bullish zone)")
            if macd > 0:
                indicator_parts.append("MACD positive")
        else:
            if rsi < 50:
                indicator_parts.append(f"RSI at {rsi:.0f} (bearish zone)")
            if macd < 0:
                indicator_parts.append("MACD negative")
        
        if indicator_parts:
            story_parts.append(f"Indicators: {', '.join(indicator_parts)}.")
            key_points.append(f"📊 {', '.join(indicator_parts)}")
        
        # ===== 3. PRICE ACTION STORY =====
        pa = layer_data.get('price_action', {})
        bullish_patterns = pa.get('bullish_patterns', [])
        bearish_patterns = pa.get('bearish_patterns', [])
        best_pattern = pa.get('best_pattern')
        
        if direction == 'BUY' and bullish_patterns:
            patterns_text = ', '.join(bullish_patterns[:3])
            story_parts.append(f"Bullish patterns detected: {patterns_text}.")
            key_points.append(f"🕯️ Bullish patterns: {patterns_text}")
        elif direction == 'SELL' and bearish_patterns:
            patterns_text = ', '.join(bearish_patterns[:3])
            story_parts.append(f"Bearish patterns detected: {patterns_text}.")
            key_points.append(f"🕯️ Bearish patterns: {patterns_text}")
        
        # ===== 4. CHART PATTERN STORY =====
        chart = layer_data.get('chart_patterns', {})
        chart_patterns = chart.get('pattern_list', [])
        
        if chart_patterns:
            top_pattern = chart_patterns[0] if chart_patterns else None
            if top_pattern:
                pattern_name = top_pattern.get('name', 'Unknown')
                pattern_dir = top_pattern.get('direction', 'NEUTRAL')
                if (direction == 'BUY' and pattern_dir == 'BUY') or (direction == 'SELL' and pattern_dir == 'SELL'):
                    story_parts.append(f"A {pattern_name} pattern is forming, supporting the directional bias.")
                    key_points.append(f"📐 {pattern_name} pattern detected")
        
        # ===== 5. STRUCTURE STORY =====
        structure = layer_data.get('structure', {})
        structure_trend = structure.get('trend', 'NEUTRAL')
        has_bos = structure.get('has_recent_bos', False)
        has_choch = structure.get('has_recent_choch', False)
        
        if direction == 'BUY' and structure_trend == 'BULLISH':
            story_parts.append("Market structure is bullish with higher highs and higher lows.")
            key_points.append("🏗️ Bullish market structure")
            if has_bos:
                story_parts.append("Recent Break of Structure (BOS) confirms bullish momentum.")
                key_points.append("✅ Break of Structure (BOS) confirmed")
        elif direction == 'SELL' and structure_trend == 'BEARISH':
            story_parts.append("Market structure is bearish with lower highs and lower lows.")
            key_points.append("🏗️ Bearish market structure")
            if has_bos:
                story_parts.append("Recent Break of Structure (BOS) confirms bearish momentum.")
                key_points.append("✅ Break of Structure (BOS) confirmed")
        
        # ===== 6. SUPPORT/RESISTANCE STORY =====
        sr = layer_data.get('support_resistance', {})
        is_near_support = sr.get('is_near_support', False)
        is_near_resistance = sr.get('is_near_resistance', False)
        vwap_pos = sr.get('vwap_position', 'NEUTRAL')
        
        if direction == 'BUY':
            if is_near_support:
                story_parts.append("Price is trading near key support level.")
                key_points.append("🛡️ Near support level")
            if 'DISCOUNT' in vwap_pos:
                story_parts.append(f"Price is at a {vwap_pos.lower()} to VWAP, offering value entry.")
                key_points.append(f"💰 {vwap_pos} to VWAP")
        else:
            if is_near_resistance:
                story_parts.append("Price is trading near key resistance level.")
                key_points.append("⚠️ Near resistance level")
            if 'PREMIUM' in vwap_pos:
                story_parts.append(f"Price is at a {vwap_pos.lower()} to VWAP, indicating overextension.")
                key_points.append(f"💸 {vwap_pos} to VWAP")
        
        # ===== 7. VOLUME STORY =====
        volume = layer_data.get('volume', {})
        vol_trend = volume.get('volume_trend', 'NEUTRAL')
        vol_ratio = volume.get('volume_ratio', 1.0)
        divergence = volume.get('divergence', None)
        
        if vol_trend == 'INCREASING':
            story_parts.append(f"Volume is increasing at {vol_ratio:.1f}x average, confirming the move.")
            key_points.append(f"📊 Volume increasing ({vol_ratio:.1f}x)")
        elif vol_trend == 'DECREASING':
            story_parts.append(f"Volume is decreasing ({vol_ratio:.1f}x average) - caution advised.")
            key_points.append(f"⚠️ Volume decreasing")
        
        if divergence == 'BULLISH' and direction == 'BUY':
            story_parts.append("Bullish volume divergence detected - price down but volume up.")
            key_points.append("🔄 Bullish volume divergence")
        elif divergence == 'BEARISH' and direction == 'SELL':
            story_parts.append("Bearish volume divergence detected - price up but volume down.")
            key_points.append("🔄 Bearish volume divergence")
        
        # ===== 8. ENTRY, STOP, TARGET STORY =====
        entry_low = signal_data.get('entry_zone_low', 0)
        entry_high = signal_data.get('entry_zone_high', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', 0)
        rr = signal_data.get('risk_reward_ratio', 0)
        
        story_parts.append(f"Entry zone: {entry_low:.4f} - {entry_high:.4f}.")
        story_parts.append(f"Stop loss at {stop_loss:.4f}, take profit at {take_profit:.4f}.")
        
        rr_quality = self._get_quality_text(rr)
        story_parts.append(f"Risk/Reward ratio is {rr:.1f}:1, which is {rr_quality}.")
        
        key_points.append(f"🎯 Entry: {entry_low:.4f} - {entry_high:.4f}")
        key_points.append(f"🛑 Stop: {stop_loss:.4f}")
        key_points.append(f"🏁 Target: {take_profit:.4f} (RR: {rr:.1f})")
        
        # ===== 9. CONFIDENCE STORY =====
        total_score = aggregated_score.total_score if hasattr(aggregated_score, 'total_score') else 0.7
        agreement = aggregated_score.agreement_score if hasattr(aggregated_score, 'agreement_score') else 0.6
        
        confidence_text = self._get_confidence_text(total_score)
        story_parts.append(f"Overall confidence is {total_score:.0%} with {agreement:.0%} of analysis layers agreeing.")
        
        key_points.append(f"🎯 Confidence: {total_score:.0%} ({confidence_text})")
        
        # ===== 10. WARNINGS =====
        warnings = []
        if vol_trend == 'DECREASING' and vol_ratio < 0.8:
            warnings.append("⚠️ Volume is decreasing - may lack follow-through")
        if is_near_resistance and direction == 'BUY':
            warnings.append("⚠️ Price approaching resistance - may face selling pressure")
        if is_near_support and direction == 'SELL':
            warnings.append("⚠️ Price approaching support - may find buying interest")
        if rr < 1.5:
            warnings.append("⚠️ Risk/Reward ratio below 1.5 - consider tighter stop or higher target")
        
        # ===== BUILD SUMMARY =====
        if direction == 'BUY':
            summary = f"BUY {signal_data.get('symbol', 'Unknown')} - {total_score:.0%} confidence. {regime_strength.title()} uptrend with bullish patterns and strong volume."
        else:
            summary = f"SELL {signal_data.get('symbol', 'Unknown')} - {total_score:.0%} confidence. {regime_strength.title()} downtrend with bearish patterns and increasing volume."
        
        # ===== BUILD FULL STORY =====
        full_story = " ".join(story_parts)
        
        return Story(
            summary=summary,
            full_story=full_story,
            key_points=key_points,
            confidence_statement=f"Confidence: {total_score:.0%} ({confidence_text})",
            risk_warnings=warnings
        )
    
    def _get_strength_text(self, score: float) -> str:
        """Convert score to strength text"""
        if score >= 0.8:
            return "strong"
        elif score >= 0.6:
            return "moderate"
        elif score >= 0.4:
            return "weak"
        else:
            return "very weak"
    
    def _get_quality_text(self, rr: float) -> str:
        """Convert RR to quality text"""
        if rr >= 3.0:
            return "excellent"
        elif rr >= 2.5:
            return "very good"
        elif rr >= 2.0:
            return "good"
        elif rr >= 1.5:
            return "acceptable"
        else:
            return "poor"
    
    def _get_confidence_text(self, confidence: float) -> str:
        """Convert confidence to text"""
        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.65:
            return "Moderate"
        elif confidence >= 0.55:
            return "Fair"
        else:
            return "Low"
    
    def get_market_summary(self, layer_data: Dict[str, Any]) -> str:
        """
        Get quick market summary (1-2 sentences)
        """
        regime = layer_data.get('regime', {})
        regime_name = regime.get('regime', 'UNKNOWN')
        pa = layer_data.get('price_action', {})
        bullish_patterns = pa.get('bullish_patterns', [])
        bearish_patterns = pa.get('bearish_patterns', [])
        
        if 'BULL' in regime_name:
            if bullish_patterns:
                return f"Bullish market with {len(bullish_patterns)} bullish patterns. Look for buy setups."
            else:
                return "Bullish market but no clear patterns. Wait for confirmation."
        elif 'BEAR' in regime_name:
            if bearish_patterns:
                return f"Bearish market with {len(bearish_patterns)} bearish patterns. Look for sell setups."
            else:
                return "Bearish market but no clear patterns. Wait for confirmation."
        else:
            return "Market ranging. Look for breakouts or mean reversion setups."
    
    def print_story(self, story: Story):
        """Print story in readable format"""
        print("\n" + "="*70)
        print(f"📊 {story.summary}")
        print("="*70)
        print()
        print("📝 ANALYSIS:")
        print("-"*50)
        for point in story.key_points:
            print(f"  {point}")
        print()
        print("📖 FULL STORY:")
        print("-"*50)
        print(f"  {story.full_story}")
        print()
        if story.risk_warnings:
            print("⚠️ WARNINGS:")
            print("-"*50)
            for warning in story.risk_warnings:
                print(f"  {warning}")
        print()
        print("="*70)