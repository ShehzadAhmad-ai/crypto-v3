"""
SMC Expert V3 - Scoring, Grading & Narrative
Meta scoring engine, confidence calibration, grade assignment, narrative builder
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .smc_core import (
    Direction, SMCContext, OrderBlock, FVG, POI,
    AMDPhase, ZoneType, MitigationState
)
from .smc_config import CONFIG


class MetaScoringEngine:
    """
    Meta scoring engine that weights all SMC concepts
    Continuous scoring 0-1 (not boolean)
    """
    
    def __init__(self):
        self.weights = CONFIG.SCORING_WEIGHTS
        self.component_scores: Dict[str, float] = {}
    
    def calculate_confluence_score(self, context: SMCContext, 
                                    entry: Dict, pois: List[POI]) -> float:
        """
        Calculate weighted confluence score from all components
        """
        # Market structure score
        self.component_scores['market_structure'] = self._score_market_structure(context)
        
        # Order block score
        self.component_scores['order_block'] = self._score_order_block(entry, context)
        
        # FVG score
        self.component_scores['fvg'] = self._score_fvg(entry, context)
        
        # Liquidity score
        self.component_scores['liquidity'] = self._score_liquidity(context)
        
        # Premium/Discount score
        self.component_scores['premium_discount'] = self._score_premium_discount(context)
        
        # Session score
        self.component_scores['session'] = self._score_session(context)
        
        # Displacement score
        self.component_scores['displacement'] = self._score_displacement(context)
        
        # Confluence score (POI count and strength)
        self.component_scores['confluence'] = self._score_confluence(pois, entry)
        
        # AMD phase score
        self.component_scores['amd_phase'] = self._score_amd_phase(context)
        
        # HTF alignment score
        self.component_scores['htf_alignment'] = self._score_htf_alignment(context)
        
        # Calculate weighted total
        total = 0
        for component, score in self.component_scores.items():
            weight = self.weights.get(component, 0.05)
            total += score * weight
        
        return min(1.0, total)
    
    def _score_market_structure(self, context: SMCContext) -> float:
        """Score market structure strength"""
        if context.current_structure == "BULLISH":
            base_score = 0.8
        elif context.current_structure == "BEARISH":
            base_score = 0.8
        else:
            base_score = 0.5
        
        # Adjust by structure strength
        return base_score * context.structure_strength
    
    def _score_order_block(self, entry: Dict, context: SMCContext) -> float:
        """Score order block quality"""
        if entry.get('type') != 'OB_RETEST':
            return 0.5
        
        ob_strength = entry.get('strength', 0.5)
        
        # Adjust for age (newer is better)
        age_score = 1.0 - min(0.5, entry.get('age_bars', 0) / CONFIG.OB_MAX_AGE_BARS)
        
        # Adjust for mitigation state
        ob_data = entry.get('poi', {}).components[0].get('data') if entry.get('poi') else None
        if ob_data:
            mitigation_score = {
                MitigationState.UNMITIGATED: 1.0,
                MitigationState.PARTIAL: 0.6,
                MitigationState.FULL: 0.3,
                MitigationState.INVALIDATED: 0.0
            }.get(ob_data.mitigation_state, 0.5)
        else:
            mitigation_score = 0.5
        
        return (ob_strength * 0.4 + age_score * 0.3 + mitigation_score * 0.3)
    
    def _score_fvg(self, entry: Dict, context: SMCContext) -> float:
        """Score FVG quality"""
        if entry.get('type') != 'FVG_RETEST':
            return 0.5
        
        fvg_strength = entry.get('strength', 0.5)
        
        # Adjust for age
        age_score = 1.0 - min(0.5, entry.get('age_bars', 0) / CONFIG.FVG_MAX_AGE_BARS)
        
        # Adjust for mitigation
        fvg_data = entry.get('poi', {}).components[0].get('data') if entry.get('poi') else None
        if fvg_data:
            mitigation_score = {
                MitigationState.UNMITIGATED: 1.0,
                MitigationState.PARTIAL: 0.5,
                MitigationState.FULL: 0.2
            }.get(fvg_data.mitigation_state, 0.5)
        else:
            mitigation_score = 0.5
        
        return (fvg_strength * 0.4 + age_score * 0.3 + mitigation_score * 0.3)
    
    def _score_liquidity(self, context: SMCContext) -> float:
        """Score liquidity conditions"""
        if not context.recent_sweeps:
            return 0.5
        
        # Recent sweep strength
        avg_sweep_strength = np.mean([s.reversal_strength for s in context.recent_sweeps[-3:]])
        
        # Sweep count in last 10 bars
        sweep_count = len([s for s in context.recent_sweeps if s.candle_index > len(context.recent_sweeps) - 10])
        count_score = min(1.0, sweep_count / 3)
        
        return (avg_sweep_strength * 0.6 + count_score * 0.4)
    
    def _score_premium_discount(self, context: SMCContext) -> float:
        """Score premium/discount zone"""
        zone_scores = {
            ZoneType.DEEP_DISCOUNT: 1.0,
            ZoneType.DISCOUNT: 0.9,
            ZoneType.EQUILIBRIUM: 0.5,
            ZoneType.PREMIUM: 0.3,
            ZoneType.DEEP_PREMIUM: 0.1
        }
        
        return zone_scores.get(context.zone_type, 0.5)
    
    def _score_session(self, context: SMCContext) -> float:
        """Score session quality"""
        return context.session_weight
    
    def _score_displacement(self, context: SMCContext) -> float:
        """Score displacement strength"""
        if not context.displacement_bars:
            return 0.5
        
        # Count recent displacement bars
        recent_displacement = [i for i in context.displacement_bars if i > len(context.displacement_bars) - 10]
        count_score = min(1.0, len(recent_displacement) / 3)
        
        return 0.5 + count_score * 0.5
    
    def _score_confluence(self, pois: List[POI], entry: Dict) -> float:
        """Score confluence from multiple POIs"""
        # Count POIs within reasonable distance
        relevant_pois = [p for p in pois if p.is_active]
        
        # Higher score for more confluences
        count_score = min(1.0, len(relevant_pois) / 5)
        
        # Strength of the main POI
        strength_score = entry.get('strength', 0.5)
        
        return (count_score * 0.5 + strength_score * 0.5)
    
    def _score_amd_phase(self, context: SMCContext) -> float:
        """Score AMD phase"""
        phase_scores = {
            AMDPhase.DISTRIBUTION: 0.9,   # Trending
            AMDPhase.MANIPULATION: 0.6,   # Wait for confirmation
            AMDPhase.ACCUMULATION: 0.4,   # Avoid
            AMDPhase.UNKNOWN: 0.5
        }
        
        base_score = phase_scores.get(context.amd_phase, 0.5)
        
        # Adjust by confidence
        return base_score * context.amd_confidence
    
    def _score_htf_alignment(self, context: SMCContext) -> float:
        """Score HTF alignment"""
        return context.htf_alignment_score
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """Get detailed component scores for debugging"""
        return self.component_scores.copy()


class ConfidenceCalibrator:
    """
    Calibrates confidence based on historical similarity and market conditions
    """
    
    def __init__(self):
        self.calibration_factor: float = 1.0
    
    def calibrate(self, raw_confidence: float, context: SMCContext,
                   replay_analysis: Dict) -> float:
        """
        Apply calibration to raw confidence score
        """
        calibrated = raw_confidence
        
        # Adjust based on volatility regime
        if context.volatility_regime == "HIGH":
            calibrated *= 0.9
        elif context.volatility_regime == "LOW":
            calibrated *= 1.05
        
        # Adjust based on kill zone
        if not context.is_kill_zone:
            calibrated *= CONFIG.KILL_ZONE_PENALTY
        
        # Adjust based on replay analysis
        if not replay_analysis.get('insufficient_data', True):
            win_rate = replay_analysis.get('win_rate', 0.5)
            similarity = replay_analysis.get('setup_similarity', 0.5)
            replay_factor = (win_rate * 0.6 + similarity * 0.4)
            calibrated *= replay_factor
        
        # Adjust based on AMD phase confidence
        calibrated *= (0.5 + context.amd_confidence * 0.5)
        
        # Cap at 0.95 (never 100% confident)
        return min(0.95, calibrated)
    
    def get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level"""
        if confidence >= 0.85:
            return "VERY_HIGH"
        elif confidence >= 0.75:
            return "HIGH"
        elif confidence >= 0.65:
            return "MODERATE"
        elif confidence >= 0.55:
            return "LOW"
        else:
            return "VERY_LOW"


class GradeAssigner:
    """
    Assigns letter grades A+ through F based on confidence
    """
    
    def __init__(self):
        self.grade: str = "F"
        self.multiplier: float = 0.0
    
    def assign(self, confidence: float) -> Tuple[str, float]:
        """
        Assign grade and position multiplier
        """
        for grade, threshold in CONFIG.GRADE_THRESHOLDS.items():
            if confidence >= threshold:
                self.grade = grade
                self.multiplier = CONFIG.POSITION_MULTIPLIERS.get(grade, 0.0)
                return grade, self.multiplier
        
        return "F", 0.0
    
    def get_grade_description(self, grade: str) -> str:
        """Get description of grade"""
        descriptions = {
            'A+': "Exceptional setup - all conditions optimal",
            'A': "Excellent setup - strong confluence",
            'B+': "Very good setup - above average confidence",
            'B': "Good setup - solid confluence",
            'B-': "Decent setup - acceptable risk",
            'C+': "Moderate setup - some weaknesses",
            'C': "Weak setup - proceed with caution",
            'D': "Poor setup - avoid",
            'F': "Invalid setup - skip"
        }
        return descriptions.get(grade, "Unknown grade")


class NarrativeBuilder:
    """
    Builds human-readable market story from detected concepts
    """
    
    def __init__(self):
        self.story_parts: List[str] = []
    
    def build_story(self, context: SMCContext, entry: Dict,
                     confidence: float, grade: str) -> Dict:
        """
        Build complete narrative for the setup
        """
        self.story_parts = []
        
        # 1. Market context
        self._add_market_context(context)
        
        # 2. AMD phase
        self._add_amd_phase(context)
        
        # 3. Entry rationale
        self._add_entry_rationale(entry, context)
        
        # 4. Liquidity context
        self._add_liquidity_context(context)
        
        # 5. Session context
        self._add_session_context(context)
        
        # 6. Risk assessment
        self._add_risk_assessment(confidence, grade, context)
        
        # 7. Expected outcome
        self._add_expected_outcome(context)
        
        full_story = " ".join(self.story_parts)
        
        # Extract key points
        key_points = self._extract_key_points(context, entry)
        
        # Identify risk warnings
        risk_warnings = self._identify_risks(context)
        
        return {
            'full_story': full_story,
            'summary': self._create_summary(context, entry),
            'key_points': key_points,
            'risk_warnings': risk_warnings,
            'market_phase': self._get_market_phase_description(context),
            'trading_plan': self._create_trading_plan(entry, context)
        }
    
    def _add_market_context(self, context: SMCContext):
        """Add market structure context"""
        structure = context.current_structure
        strength = context.structure_strength
        
        if structure == "BULLISH":
            self.story_parts.append(
                f"Market is in a bullish structure with {strength:.0%} strength. "
                f"Higher highs and higher lows are forming."
            )
        elif structure == "BEARISH":
            self.story_parts.append(
                f"Market is in a bearish structure with {strength:.0%} strength. "
                f"Lower highs and lower lows are forming."
            )
        else:
            self.story_parts.append(
                "Market structure is neutral with no clear directional bias."
            )
    
    def _add_amd_phase(self, context: SMCContext):
        """Add AMD phase context"""
        phase = context.amd_phase
        confidence = context.amd_confidence
        
        phase_descriptions = {
            AMDPhase.ACCUMULATION: "smart money is accumulating positions",
            AMDPhase.MANIPULATION: "liquidity sweeps and false breaks are occurring",
            AMDPhase.DISTRIBUTION: "smart money is distributing positions in a trending move",
            AMDPhase.UNKNOWN: "no clear AMD phase is identified"
        }
        
        self.story_parts.append(
            f"AMD analysis shows {phase_descriptions.get(phase, 'unknown phase')} "
            f"with {confidence:.0%} confidence."
        )
    
    def _add_entry_rationale(self, entry: Dict, context: SMCContext):
        """Add entry rationale based on entry type"""
        entry_type = entry.get('type', 'UNKNOWN')
        direction = entry['direction']
        
        type_descriptions = {
            'OB_RETEST': f"Price is retesting a {direction.value} order block",
            'FVG_RETEST': f"Price is filling a {direction.value} fair value gap",
            'SWEEP_ENTRY': f"Entry after a liquidity sweep with strong reversal",
            'OTE_ENTRY': f"Entry at optimal trade entry (OTE) level",
            'SD_ENTRY': f"Entry at {direction.value} supply/demand zone"
        }
        
        description = type_descriptions.get(entry_type, f"{direction.value} setup")
        strength = entry.get('strength', 0.5)
        
        self.story_parts.append(
            f"{description} with {strength:.0%} strength. "
        )
        
        # Add price location context
        zone = context.zone_type
        zone_desc = {
            ZoneType.DEEP_DISCOUNT: "deep discount zone (attractive for buys)",
            ZoneType.DISCOUNT: "discount zone (good for buys)",
            ZoneType.EQUILIBRIUM: "equilibrium (neutral)",
            ZoneType.PREMIUM: "premium zone (good for sells)",
            ZoneType.DEEP_PREMIUM: "deep premium zone (attractive for sells)"
        }
        
        self.story_parts.append(
            f"Price is in {zone_desc.get(zone, 'equilibrium')}. "
        )
    
    def _add_liquidity_context(self, context: SMCContext):
        """Add liquidity context"""
        if context.recent_sweeps:
            recent_sweep = context.recent_sweeps[-1]
            sweep_type = "sell-side" if recent_sweep.type == 'SSL_SWEEP' else "buy-side"
            strength = recent_sweep.reversal_strength
            
            self.story_parts.append(
                f"Recent {sweep_type} liquidity sweep with {strength:.0%} reversal strength. "
            )
        else:
            self.story_parts.append("No recent liquidity sweeps detected. ")
    
    def _add_session_context(self, context: SMCContext):
        """Add session context"""
        session_name = context.session.value
        kill_zone = context.is_kill_zone
        
        if kill_zone:
            self.story_parts.append(
                f"Currently in {session_name} kill zone - high probability window. "
            )
        else:
            self.story_parts.append(
                f"Currently in {session_name} session - moderate probability. "
            )
    
    def _add_risk_assessment(self, confidence: float, grade: str, context: SMCContext):
        """Add risk assessment"""
        confidence_level = self._get_confidence_level_text(confidence)
        
        self.story_parts.append(
            f"Setup confidence is {confidence_level} ({confidence:.0%}) with grade {grade}. "
        )
        
        if context.volatility_regime == "HIGH":
            self.story_parts.append("High volatility regime - wider stops recommended. ")
        elif context.volatility_regime == "LOW":
            self.story_parts.append("Low volatility regime - tighter stops possible. ")
    
    def _add_expected_outcome(self, context: SMCContext):
        """Add expected outcome based on AMD phase"""
        if context.amd_phase == AMDPhase.DISTRIBUTION:
            self.story_parts.append(
                "Expect continuation toward next liquidity targets."
            )
        elif context.amd_phase == AMDPhase.MANIPULATION:
            self.story_parts.append(
                "Expect further manipulation before trend continuation."
            )
        else:
            self.story_parts.append(
                "Wait for confirmation before aggressive entries."
            )
    
    def _get_confidence_level_text(self, confidence: float) -> str:
        """Get text description of confidence level"""
        if confidence >= 0.85:
            return "very high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.65:
            return "moderate"
        else:
            return "low"
    
    def _create_summary(self, context: SMCContext, entry: Dict) -> str:
        """Create short summary"""
        direction = entry['direction'].value
        entry_type = entry.get('type', 'setup')
        phase = context.amd_phase.value
        kill_zone = "kill zone" if context.is_kill_zone else "regular session"
        
        return f"{direction} {entry_type} during {phase} phase in {kill_zone}"
    
    def _extract_key_points(self, context: SMCContext, entry: Dict) -> List[str]:
        """Extract key points from analysis"""
        points = []
        
        # Structure
        points.append(f"Market structure: {context.current_structure}")
        
        # AMD phase
        points.append(f"AMD phase: {context.amd_phase.value}")
        
        # Liquidity
        if context.recent_sweeps:
            points.append(f"Recent sweep: {context.recent_sweeps[-1].type}")
        
        # Session
        if context.is_kill_zone:
            points.append(f"Kill zone active: {context.session.value}")
        
        # Zone
        points.append(f"Price zone: {context.zone_type.value}")
        
        # Entry type
        points.append(f"Entry model: {entry.get('type', 'Unknown')}")
        
        return points[:5]  # Limit to 5 key points
    
    def _identify_risks(self, context: SMCContext) -> List[str]:
        """Identify potential risks"""
        risks = []
        
        # Structure risk
        if context.current_structure == "NEUTRAL":
            risks.append("Neutral structure - no strong directional bias")
        
        # AMD phase risk
        if context.amd_phase == AMDPhase.ACCUMULATION:
            risks.append("Accumulation phase - potential for further range trading")
        elif context.amd_phase == AMDPhase.MANIPULATION:
            risks.append("Manipulation phase - false breaks possible")
        
        # Kill zone risk
        if not context.is_kill_zone:
            risks.append("Outside kill zone - lower probability")
        
        # HTF alignment risk
        if context.htf_alignment_score < 0.5:
            risks.append("HTF misalignment - conflicting timeframes")
        
        # Volatility risk
        if context.volatility_regime == "HIGH":
            risks.append("High volatility - increased slippage risk")
        
        return risks[:3]  # Limit to 3 risks
    
    def _get_market_phase_description(self, context: SMCContext) -> str:
        """Get overall market phase description"""
        if context.amd_phase == AMDPhase.DISTRIBUTION:
            return "Trending phase - follow the trend"
        elif context.amd_phase == AMDPhase.MANIPULATION:
            return "Transition phase - wait for confirmation"
        elif context.amd_phase == AMDPhase.ACCUMULATION:
            return "Range phase - trade boundaries"
        else:
            return "Undefined phase - exercise caution"
    
    def _create_trading_plan(self, entry: Dict, context: SMCContext) -> str:
        """Create simple trading plan"""
        direction = entry['direction'].value
        entry_price = entry['entry']
        
        if direction == "BUY":
            return f"Buy at {entry_price:.2f}. Stop below recent swing low. Target next BSL above."
        else:
            return f"Sell at {entry_price:.2f}. Stop above recent swing high. Target next SSL below."