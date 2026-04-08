# """
# pattern_scoring.py - Meta Scoring & Confidence Calibration for Pattern V4

# Combines all pattern scores with:
# - Weighted average formula (similarity + context)
# - HTF confluence boost
# - Regime validity multiplier
# - Learning-based weight adjustment
# - False breakout penalty
# - Grade assignment (A+, A, B+, B, B-, C+, C, D, F)

# Version: 4.0
# Author: Pattern Intelligence System
# """

# import numpy as np
# import pandas as pd
# from typing import Dict, List, Optional, Tuple, Any
# from dataclasses import dataclass, field
# from datetime import datetime

# from .pattern_core import PatternV4, Grade
# from .pattern_config import CONFIG, get_grade, get_position_multiplier, get_action


# # ============================================================================
# # RISK PENALTY CALCULATOR
# # ============================================================================

# class RiskPenaltyCalculatorV4:
#     """
#     Calculates risk penalties that reduce final confidence.
#     Higher penalty = more risk = lower confidence.
#     """
    
#     def __init__(self):
#         self.penalties = CONFIG.risk_penalties if hasattr(CONFIG, 'risk_penalties') else {
#             'high_volatility': 0.15,
#             'low_liquidity': 0.20,
#             'active_trap': 0.25,
#             'conflicting_patterns': 0.15,
#             'poor_volume': 0.10,
#             'late_entry': 0.20,
#             'overextended': 0.15,
#         }
    
#     def calculate(self, pattern: Dict, df: pd.DataFrame) -> float:
#         """
#         Calculate total risk penalty (0-0.5 range).
#         Returns penalty to subtract from raw score.
#         """
#         total_penalty = 0.0
        
#         # 1. High volatility penalty
#         context_components = pattern.get('context_components', {})
#         volatility = context_components.get('volatility_condition', 0.5)
#         if volatility < 0.3:
#             total_penalty += self.penalties.get('high_volatility', 0.15)
        
#         # 2. Low liquidity penalty
#         liquidity = pattern.get('liquidity', {})
#         liquidity_score = liquidity.get('score', 0.5)
#         if liquidity_score < 0.3:
#             total_penalty += self.penalties.get('low_liquidity', 0.20)
        
#         # 3. Active trap penalty
#         trap = pattern.get('trap', {})
#         if trap.get('detected', False) and not trap.get('converted', False):
#             total_penalty += self.penalties.get('active_trap', 0.25)
        
#         # 4. Poor volume penalty
#         similarity_components = pattern.get('components', {})
#         volume_pattern = similarity_components.get('volume_pattern', 0.5)
#         if volume_pattern < 0.4:
#             total_penalty += self.penalties.get('poor_volume', 0.10)
        
#         # 5. Overextended penalty
#         current_price = float(df['close'].iloc[-1])
#         neckline = pattern.get('neckline', current_price)
#         distance = abs(current_price - neckline) / neckline if neckline > 0 else 0
#         if distance > 0.02:
#             total_penalty += self.penalties.get('overextended', 0.15)
        
#         return min(0.5, total_penalty)


# # ============================================================================
# # FALSE BREAKOUT PENALTY CALCULATOR
# # ============================================================================

# class FalseBreakoutPenaltyV4:
#     """
#     Calculates penalty for potential false breakouts.
#     """
    
#     def __init__(self):
#         self.config = CONFIG.false_breakout_config
    
#     def calculate(self, pattern: Dict, df: pd.DataFrame) -> float:
#         """
#         Calculate false breakout penalty (0-0.3 range).
#         """
#         if not self.config.get('enabled', True):
#             return 0.0
        
#         end_idx = pattern.get('end_idx', len(df) - 1)
        
#         if end_idx + 2 >= len(df):
#             return 0.0
        
#         penalty = 0.0
#         direction = pattern.get('direction', 'NEUTRAL')
        
#         # Check breakout candle
#         breakout_candle = df.iloc[end_idx + 1] if end_idx + 1 < len(df) else None
#         if breakout_candle is None:
#             return 0.0
        
#         # Volume check
#         avg_volume = df['volume'].iloc[max(0, end_idx-20):end_idx].mean()
#         if breakout_candle['volume'] < avg_volume * self.config.get('volume_threshold', 1.2):
#             penalty += 0.20
        
#         # Wick check
#         candle_range = breakout_candle['high'] - breakout_candle['low']
#         if candle_range > 0:
#             if direction == 'BUY':
#                 wick = breakout_candle['high'] - max(breakout_candle['open'], breakout_candle['close'])
#             else:
#                 wick = min(breakout_candle['open'], breakout_candle['close']) - breakout_candle['low']
            
#             if wick / candle_range > self.config.get('wick_ratio_threshold', 0.6):
#                 penalty += 0.25
        
#         # Immediate reversal check
#         if end_idx + 2 < len(df):
#             next_candle = df.iloc[end_idx + 2]
#             if direction == 'BUY' and next_candle['close'] < next_candle['open']:
#                 penalty += 0.30
#             elif direction == 'SELL' and next_candle['close'] > next_candle['open']:
#                 penalty += 0.30
        
#         return min(0.3, penalty)


# # ============================================================================
# # CONFIDENCE CALCULATOR - Main Formula
# # ============================================================================

# class ConfidenceCalculatorV4:
#     """
#     Calculates final confidence using weighted average formula:
#     Final = (similarity_weight * similarity + context_weight * context) * htf_boost * regime_validity * learning_multiplier - penalties
#     """
    
#     def __init__(self):
#         self.weights = CONFIG.confidence_weights
#         self.similarity_weight = self.weights.get('similarity_weight', 0.50)
#         self.context_weight = self.weights.get('context_weight', 0.30)
#         self.mtf_weight = self.weights.get('mtf_weight', 0.10)
#         self.regime_weight = self.weights.get('regime_weight', 0.05)
#         self.learning_weight = self.weights.get('learning_weight', 0.05)
        
#         self.max_confidence = 0.95
#         self.min_confidence = 0.25
    
#     def calculate(self, pattern: Dict, context_score: float,
#                   htf_boost: float = 1.0, regime_validity: float = 1.0,
#                   learning_multiplier: float = 1.0) -> float:
#         """
#         Calculate final confidence.
        
#         Formula: 
#         base = (sim_weight * similarity) + (context_weight * context)
#         adjusted = base * htf_boost * regime_validity * learning_multiplier
#         final = adjusted - penalties
#         """
#         similarity = pattern.get('similarity', 0.5)
        
#         # Base confidence (weighted average of similarity and context)
#         # Normalize weights to sum to 1.0
#         sim_weight = self.similarity_weight / (self.similarity_weight + self.context_weight)
#         ctx_weight = self.context_weight / (self.similarity_weight + self.context_weight)
#         base_confidence = (sim_weight * similarity) + (ctx_weight * context_score)
        
#         # Apply multipliers
#         adjusted_confidence = base_confidence * htf_boost * regime_validity * learning_multiplier
        
#         # Apply penalties
#         risk_penalty = pattern.get('risk_penalty', 0.0)
#         false_breakout_penalty = pattern.get('false_breakout_penalty', 0.0)
        
#         final_confidence = adjusted_confidence - risk_penalty - false_breakout_penalty
        
#         # Clamp to realistic range
#         final_confidence = max(self.min_confidence, min(self.max_confidence, final_confidence))
        
#         return round(final_confidence, 3)
    
#     def calculate_with_mtf(self, pattern: Dict, context_score: float,
#                            mtf_confluence: Dict, regime_validity: float = 1.0,
#                            learning_multiplier: float = 1.0) -> float:
#         """
#         Calculate final confidence including MTF confluence.
#         """
#         similarity = pattern.get('similarity', 0.5)
#         mtf_boost = mtf_confluence.get('boost_factor', 1.0)
        
#         # Extended formula with MTF
#         base_confidence = (
#             self.similarity_weight * similarity +
#             self.context_weight * context_score +
#             self.mtf_weight * mtf_confluence.get('weighted_score', 0.5)
#         )
        
#         adjusted_confidence = base_confidence * mtf_boost * regime_validity * learning_multiplier
        
#         risk_penalty = pattern.get('risk_penalty', 0.0)
#         false_breakout_penalty = pattern.get('false_breakout_penalty', 0.0)
        
#         final_confidence = adjusted_confidence - risk_penalty - false_breakout_penalty
#         final_confidence = max(self.min_confidence, min(self.max_confidence, final_confidence))
        
#         return round(final_confidence, 3)


# # ============================================================================
# # GRADE ASSIGNER
# # ============================================================================

# class GradeAssignerV4:
#     """
#     Assigns letter grades based on final confidence.
#     A+ = Strongest, F = Skip.
#     """
    
#     def __init__(self):
#         self.thresholds = CONFIG.grade_thresholds
#         self.position_multipliers = CONFIG.position_multipliers
    
#     def assign_grade(self, confidence: float) -> Grade:
#         """Assign grade based on confidence"""
#         for grade, threshold in sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True):
#             if confidence >= threshold:
#                 return Grade(grade)
#         return Grade.F
    
#     def get_position_multiplier(self, grade: Grade) -> float:
#         """Get position multiplier based on grade"""
#         return self.position_multipliers.get(grade.value, 1.0)
    
#     def get_grade_description(self, grade: Grade) -> str:
#         """Get human-readable description for grade"""
#         descriptions = {
#             Grade.A_PLUS: "Exceptional - Strong confluence, ideal setup",
#             Grade.A: "Excellent - High probability, clear pattern",
#             Grade.B_PLUS: "Very Good - Solid setup with good confluence",
#             Grade.B: "Good - Solid setup, acceptable risk",
#             Grade.B_MINUS: "Above Average - Decent setup, some concerns",
#             Grade.C_PLUS: "Fair - Acceptable but needs confirmation",
#             Grade.C: "Mediocre - Weak signals, high risk",
#             Grade.D: "Poor - Significant concerns, avoid",
#             Grade.F: "Skip - Not tradeable"
#         }
#         return descriptions.get(grade, "Unknown")


# # ============================================================================
# # ACTION DETERMINER
# # ============================================================================

# class ActionDeterminerV4:
#     """
#     Determines action based on confidence and completion percentage.
#     """
    
#     def __init__(self):
#         self.thresholds = CONFIG.action_thresholds
    
#     def determine_action(self, confidence: float, completion_pct: float = 1.0) -> Tuple[str, str]:
#         """
#         Determine action and action detail.
#         Returns (action, action_detail)
#         """
#         if confidence >= self.thresholds['strong_entry']:
#             action = "STRONG_ENTRY"
#             detail = f"Exceptional setup with {confidence:.0%} confidence"
#         elif confidence >= self.thresholds['enter_now']:
#             if completion_pct >= 0.95:
#                 action = "ENTER_NOW"
#                 detail = f"Pattern confirmed with {confidence:.0%} confidence"
#             else:
#                 action = "WAIT_FOR_RETEST"
#                 detail = f"Pattern {completion_pct:.0%} complete - waiting for retest"
#         elif confidence >= self.thresholds['wait_retest']:
#             action = "WAIT_FOR_RETEST"
#             detail = f"Pattern forming - {completion_pct:.0%} complete, {confidence:.0%} confidence"
#         else:
#             action = "SKIP"
#             detail = f"Insufficient confidence ({confidence:.0%})"
        
#         return action, detail


# # ============================================================================
# # MAIN SCORING ENGINE
# # ============================================================================

# class PatternScoringEngineV4:
#     """
#     Main scoring engine that orchestrates all scoring components.
#     """
    
#     def __init__(self):
#         self.risk_calculator = RiskPenaltyCalculatorV4()
#         self.false_breakout_calculator = FalseBreakoutPenaltyV4()
#         self.confidence_calculator = ConfidenceCalculatorV4()
#         self.grade_assigner = GradeAssignerV4()
#         self.action_determiner = ActionDeterminerV4()
    
#     def score_pattern(self, pattern: Dict, df: pd.DataFrame,
#                       context_score: float, htf_confluence: Dict = None,
#                       regime_validity: float = 1.0,
#                       learning_multiplier: float = 1.0) -> Dict:
#         """
#         Complete scoring pipeline for a single pattern.
#         Returns pattern with all scores added.
#         """
        
#         # 1. Calculate risk penalty
#         risk_penalty = self.risk_calculator.calculate(pattern, df)
#         pattern['risk_penalty'] = risk_penalty
        
#         # 2. Calculate false breakout penalty
#         false_breakout_penalty = self.false_breakout_calculator.calculate(pattern, df)
#         pattern['false_breakout_penalty'] = false_breakout_penalty
        
#         # 3. Calculate final confidence
#         if htf_confluence:
#             final_confidence = self.confidence_calculator.calculate_with_mtf(
#                 pattern, context_score, htf_confluence, regime_validity, learning_multiplier
#             )
#         else:
#             final_confidence = self.confidence_calculator.calculate(
#                 pattern, context_score, 1.0, regime_validity, learning_multiplier
#             )
        
#         pattern['final_confidence'] = final_confidence
        
#         # 4. Assign grade
#         grade = self.grade_assigner.assign_grade(final_confidence)
#         pattern['grade'] = grade.value
#         pattern['grade_description'] = self.grade_assigner.get_grade_description(grade)
        
#         # 5. Determine action
#         completion_pct = pattern.get('completion_pct', 1.0)
#         action, action_detail = self.action_determiner.determine_action(final_confidence, completion_pct)
#         pattern['action'] = action
#         pattern['action_detail'] = action_detail
        
#         # 6. Calculate position multiplier
#         pattern['position_multiplier'] = self.grade_assigner.get_position_multiplier(grade)
        
#         # 7. Build reasons
#         pattern['reasons'] = self._build_reasons(pattern, final_confidence, grade)
        
#         return pattern
    
#     def score_batch(self, patterns: List[Dict], df: pd.DataFrame,
#                     context_scores: List[float], htf_confluence_list: List[Dict] = None,
#                     regime_validity: float = 1.0,
#                     learning_multiplier: float = 1.0) -> List[Dict]:
#         """
#         Score multiple patterns.
#         """
#         scored = []
        
#         for i, pattern in enumerate(patterns):
#             try:
#                 context_score = context_scores[i] if i < len(context_scores) else 0.5
#                 htf_confluence = htf_confluence_list[i] if htf_confluence_list and i < len(htf_confluence_list) else None
                
#                 scored_pattern = self.score_pattern(
#                     pattern, df, context_score, htf_confluence,
#                     regime_validity, learning_multiplier
#                 )
#                 scored.append(scored_pattern)
#             except Exception as e:
#                 continue
        
#         # Sort by confidence (highest first)
#         scored.sort(key=lambda x: x.get('final_confidence', 0), reverse=True)
        
#         return scored
    
#     def _build_reasons(self, pattern: Dict, confidence: float, grade: Grade) -> List[str]:
#         """Build human-readable reasons for the decision"""
#         reasons = []
        
#         # Pattern quality
#         similarity = pattern.get('similarity', 0)
#         if similarity >= 0.8:
#             reasons.append(f"Excellent pattern quality ({similarity:.0%} similarity)")
#         elif similarity >= 0.6:
#             reasons.append(f"Good pattern quality ({similarity:.0%} similarity)")
        
#         # Context
#         context_score = pattern.get('context_score', 0)
#         if context_score >= 0.7:
#             reasons.append(f"Favorable market context ({context_score:.0%})")
        
#         # Grade
#         grade_descriptions = {
#             Grade.A_PLUS: "Exceptional setup - strong confluence",
#             Grade.A: "Excellent setup - high probability",
#             Grade.B_PLUS: "Very good setup - solid confluence",
#             Grade.B: "Good setup - meets all criteria",
#             Grade.B_MINUS: "Above average setup - acceptable",
#         }
#         if grade in grade_descriptions:
#             reasons.append(grade_descriptions[grade])
        
#         # HTF confluence
#         if pattern.get('htf_confluence', {}).get('boost_factor', 1.0) > 1.0:
#             reasons.append("Higher timeframe confluence confirms setup")
        
#         # Trap conversion
#         if pattern.get('trap', {}).get('converted', False):
#             reasons.append("Trap detected and converted - favorable reversal")
        
#         # Liquidity confirmation
#         liquidity = pattern.get('liquidity', {})
#         if liquidity.get('has_down_sweep') and pattern.get('direction') == 'BUY':
#             reasons.append("Liquidity sweep below support - bullish confirmation")
#         elif liquidity.get('has_up_sweep') and pattern.get('direction') == 'SELL':
#             reasons.append("Liquidity sweep above resistance - bearish confirmation")
        
#         # Risk warnings
#         risk_penalty = pattern.get('risk_penalty', 0)
#         if risk_penalty > 0.15:
#             reasons.append(f"⚠️ Risk penalty applied ({risk_penalty:.0%}) - exercise caution")
        
#         return reasons[:7]  # Top 7 reasons
    

#     def is_tradeable(self, pattern: Dict) -> Tuple[bool, str]:
#         """
#         Final check if pattern is tradeable.
#         ONLY checks confidence and RR - NO action check, NO grade check.
#         All thresholds controlled from config.
        
#         Returns (is_tradeable, reason)
#         """
#         confidence = pattern.get('final_confidence', 0)
#         rr = pattern.get('risk_reward', 0)
        
#         # Get thresholds from config (you can change these in pattern_config.py)
#         min_confidence = getattr(CONFIG, 'min_trade_confidence', 0.35)
#         min_rr = getattr(CONFIG, 'min_trade_rr', 1.0)
        
#         # Check 1: Confidence must be >= threshold
#         if confidence < min_confidence:
#             return False, f"Confidence {confidence:.0%} < {min_confidence:.0%}"
        
#         # Check 2: RR must be >= threshold
#         if rr < min_rr:
#             return False, f"RR {rr:.2f} < {min_rr:.1f}"
        
#         # REMOVED: Grade check
#         # REMOVED: Action check  
#         # REMOVED: Trap check
#         # REMOVED: Risk penalty check
        
#         return True, "Approved for trade"
    
#     # def is_tradeable(self, pattern: Dict) -> Tuple[bool, str]:
#     #     """
#     #     Final check if pattern is tradeable.
#     #     Returns (is_tradeable, reason)
#     #     """
#     #     confidence = pattern.get('final_confidence', 0)
#     #     action = pattern.get('action', 'SKIP')
#     #     grade = pattern.get('grade', 'F')
        
#     #     # Check 1: Must have A, B, or C+ grade
#     #     if grade not in ['A+', 'A', 'B+', 'B', 'B-','C+', 'C', 'D']:
#     #         return False, f"Grade {grade} too low"
        
#     #     # Check 2: Confidence must be >= 0.55
#     #     if confidence < 0.30:
#     #         return False, f"Confidence {confidence:.0%} < 55%"
        
#     #     # Check 3: Action must be ENTER_NOW or STRONG_ENTRY or WAIT_FOR_RETEST
#     #     if action not in ['STRONG_ENTRY', 'ENTER_NOW', 'WAIT_FOR_RETEST']:
#     #         return False, f"Action {action} not tradeable"
        
#     #     # Check 4: Must not have active trap
#     #     trap = pattern.get('trap', {})
#     #     if trap.get('detected', False) and not trap.get('converted', False):
#     #         return False, f"Active trap: {trap.get('trap_type', 'unknown')}"
        
#     #     # Check 5: Risk penalty too high
#     #     risk_penalty = pattern.get('risk_penalty', 0)
#     #     if risk_penalty > 0.35:
#     #         return False, f"Risk penalty too high ({risk_penalty:.0%})"
        
#     #     return True, "Approved for trade"


# # ============================================================================
# # SCORE SUMMARY UTILITY
# # ============================================================================

# def get_score_summary(pattern: Dict) -> Dict[str, Any]:
#     """
#     Get detailed score summary for debugging.
#     """
#     return {
#         'pattern_name': pattern.get('pattern_name', 'Unknown'),
#         'direction': pattern.get('direction', 'NEUTRAL'),
        
#         # Similarity components
#         'similarity': pattern.get('similarity', 0),
#         'similarity_components': pattern.get('components', {}),
        
#         # Context components
#         'context_score': pattern.get('context_score', 0),
#         'context_components': pattern.get('context_components', {}),
        
#         # Penalties
#         'risk_penalty': pattern.get('risk_penalty', 0),
#         'false_breakout_penalty': pattern.get('false_breakout_penalty', 0),
        
#         # Final scores
#         'final_confidence': pattern.get('final_confidence', 0),
#         'grade': pattern.get('grade', 'F'),
#         'action': pattern.get('action', 'SKIP'),
        
#         # Position sizing
#         'position_multiplier': pattern.get('position_multiplier', 1.0),
        
#         # Trade setup
#         'entry': pattern.get('entry'),
#         'stop_loss': pattern.get('stop_loss'),
#         'take_profit': pattern.get('take_profit'),
#         'risk_reward': pattern.get('risk_reward'),
#     }


# # ============================================================================
# # EXPORTS
# # ============================================================================

# __all__ = [
#     'RiskPenaltyCalculatorV4',
#     'FalseBreakoutPenaltyV4',
#     'ConfidenceCalculatorV4',
#     'GradeAssignerV4',
#     'ActionDeterminerV4',
#     'PatternScoringEngineV4',
#     'get_score_summary',
# ]






















































"""
pattern_scoring.py - Meta Scoring & Confidence Calibration for Pattern V4

Combines all pattern scores with:
- Weighted average formula (similarity + context)
- HTF confluence boost
- Regime validity multiplier
- Learning-based weight adjustment
- False breakout penalty
- Grade assignment (A+, A, B+, B, B-, C+, C, D, F)

Version: 4.0
Author: Pattern Intelligence System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from .pattern_core import PatternV4, Grade
from .pattern_config import CONFIG, get_grade, get_position_multiplier, get_action


# ============================================================================
# RISK PENALTY CALCULATOR
# ============================================================================

class RiskPenaltyCalculatorV4:
    """
    Calculates risk penalties that reduce final confidence.
    Higher penalty = more risk = lower confidence.
    """
    
    def __init__(self):
        self.penalties = CONFIG.risk_penalties if hasattr(CONFIG, 'risk_penalties') else {
            'high_volatility': 0.15,
            'low_liquidity': 0.20,
            'active_trap': 0.25,
            'conflicting_patterns': 0.15,
            'poor_volume': 0.10,
            'late_entry': 0.20,
            'overextended': 0.15,
        }
    
    def calculate(self, pattern: Dict, df: pd.DataFrame) -> float:
        """
        Calculate total risk penalty (0-0.5 range).
        Returns penalty to subtract from raw score.
        """
        total_penalty = 0.0
        
        # 1. High volatility penalty
        context_components = pattern.get('context_components', {})
        volatility = context_components.get('volatility_condition', 0.5)
        if volatility < 0.3:
            total_penalty += self.penalties.get('high_volatility', 0.15)
        
        # 2. Low liquidity penalty
        liquidity = pattern.get('liquidity', {})
        liquidity_score = liquidity.get('score', 0.5)
        if liquidity_score < 0.3:
            total_penalty += self.penalties.get('low_liquidity', 0.20)
        
        # 3. Active trap penalty
        trap = pattern.get('trap', {})
        if trap.get('detected', False) and not trap.get('converted', False):
            total_penalty += self.penalties.get('active_trap', 0.25)
        
        # 4. Poor volume penalty
        similarity_components = pattern.get('components', {})
        volume_pattern = similarity_components.get('volume_pattern', 0.5)
        if volume_pattern < 0.4:
            total_penalty += self.penalties.get('poor_volume', 0.10)
        
        # 5. Overextended penalty
        current_price = float(df['close'].iloc[-1])
        neckline = pattern.get('neckline', current_price)
        distance = abs(current_price - neckline) / neckline if neckline > 0 else 0
        if distance > 0.02:
            total_penalty += self.penalties.get('overextended', 0.15)
        
        return min(0.5, total_penalty)


# ============================================================================
# FALSE BREAKOUT PENALTY CALCULATOR
# ============================================================================

class FalseBreakoutPenaltyV4:
    """
    Calculates penalty for potential false breakouts.
    """
    
    def __init__(self):
        self.config = CONFIG.false_breakout_config
    
    def calculate(self, pattern: Dict, df: pd.DataFrame) -> float:
        """
        Calculate false breakout penalty (0-0.3 range).
        """
        if not self.config.get('enabled', True):
            return 0.0
        
        end_idx = pattern.get('end_idx', len(df) - 1)
        
        if end_idx + 2 >= len(df):
            return 0.0
        
        penalty = 0.0
        direction = pattern.get('direction', 'NEUTRAL')
        
        # Check breakout candle
        breakout_candle = df.iloc[end_idx + 1] if end_idx + 1 < len(df) else None
        if breakout_candle is None:
            return 0.0
        
        # Volume check
        avg_volume = df['volume'].iloc[max(0, end_idx-20):end_idx].mean()
        if breakout_candle['volume'] < avg_volume * self.config.get('volume_threshold', 1.2):
            penalty += 0.20
        
        # Wick check
        candle_range = breakout_candle['high'] - breakout_candle['low']
        if candle_range > 0:
            if direction == 'BUY':
                wick = breakout_candle['high'] - max(breakout_candle['open'], breakout_candle['close'])
            else:
                wick = min(breakout_candle['open'], breakout_candle['close']) - breakout_candle['low']
            
            if wick / candle_range > self.config.get('wick_ratio_threshold', 0.6):
                penalty += 0.25
        
        # Immediate reversal check
        if end_idx + 2 < len(df):
            next_candle = df.iloc[end_idx + 2]
            if direction == 'BUY' and next_candle['close'] < next_candle['open']:
                penalty += 0.30
            elif direction == 'SELL' and next_candle['close'] > next_candle['open']:
                penalty += 0.30
        
        return min(0.3, penalty)


# ============================================================================
# CONFIDENCE CALCULATOR - Main Formula
# ============================================================================

class ConfidenceCalculatorV4:
    """
    Calculates final confidence using weighted average formula:
    Final = (similarity_weight * similarity + context_weight * context) * htf_boost * regime_validity * learning_multiplier - penalties
    """
    
    def __init__(self):
        self.weights = CONFIG.confidence_weights
        self.similarity_weight = self.weights.get('similarity_weight', 0.50)
        self.context_weight = self.weights.get('context_weight', 0.30)
        self.mtf_weight = self.weights.get('mtf_weight', 0.10)
        self.regime_weight = self.weights.get('regime_weight', 0.05)
        self.learning_weight = self.weights.get('learning_weight', 0.05)
        
        self.max_confidence = 0.95
        self.min_confidence = 0.25
    
    def calculate(self, pattern: Dict, context_score: float,
                  htf_boost: float = 1.0, regime_validity: float = 1.0,
                  learning_multiplier: float = 1.0) -> float:
        """
        Calculate final confidence.
        
        Formula: 
        base = (sim_weight * similarity) + (context_weight * context)
        adjusted = base * htf_boost * regime_validity * learning_multiplier
        final = adjusted - penalties
        """
        similarity = pattern.get('similarity', 0.5)
        
        # Base confidence (weighted average of similarity and context)
        # Normalize weights to sum to 1.0
        sim_weight = self.similarity_weight / (self.similarity_weight + self.context_weight)
        ctx_weight = self.context_weight / (self.similarity_weight + self.context_weight)
        base_confidence = (sim_weight * similarity) + (ctx_weight * context_score)
        
        # Apply multipliers
        adjusted_confidence = base_confidence * htf_boost * regime_validity * learning_multiplier
        
        # Apply penalties
        risk_penalty = pattern.get('risk_penalty', 0.0)
        false_breakout_penalty = pattern.get('false_breakout_penalty', 0.0)
        
        final_confidence = adjusted_confidence - risk_penalty - false_breakout_penalty
        
        # Clamp to realistic range
        final_confidence = max(self.min_confidence, min(self.max_confidence, final_confidence))
        
        return round(final_confidence, 3)
    
    def calculate_with_mtf(self, pattern: Dict, context_score: float,
                           mtf_confluence: Dict, regime_validity: float = 1.0,
                           learning_multiplier: float = 1.0) -> float:
        """
        Calculate final confidence including MTF confluence.
        """
        similarity = pattern.get('similarity', 0.5)
        mtf_boost = mtf_confluence.get('boost_factor', 1.0)
        
        # Extended formula with MTF
        base_confidence = (
            self.similarity_weight * similarity +
            self.context_weight * context_score +
            self.mtf_weight * mtf_confluence.get('weighted_score', 0.5)
        )
        
        adjusted_confidence = base_confidence * mtf_boost * regime_validity * learning_multiplier
        
        risk_penalty = pattern.get('risk_penalty', 0.0)
        false_breakout_penalty = pattern.get('false_breakout_penalty', 0.0)
        
        final_confidence = adjusted_confidence - risk_penalty - false_breakout_penalty
        final_confidence = max(self.min_confidence, min(self.max_confidence, final_confidence))
        
        return round(final_confidence, 3)


# ============================================================================
# GRADE ASSIGNER
# ============================================================================

class GradeAssignerV4:
    """
    Assigns letter grades based on final confidence.
    A+ = Strongest, F = Skip.
    """
    
    def __init__(self):
        self.thresholds = CONFIG.grade_thresholds
        self.position_multipliers = CONFIG.position_multipliers
    
    def assign_grade(self, confidence: float) -> Grade:
        """Assign grade based on confidence"""
        for grade, threshold in sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True):
            if confidence >= threshold:
                return Grade(grade)
        return Grade.F
    
    def get_position_multiplier(self, grade: Grade) -> float:
        """Get position multiplier based on grade"""
        return self.position_multipliers.get(grade.value, 1.0)
    
    def get_grade_description(self, grade: Grade) -> str:
        """Get human-readable description for grade"""
        descriptions = {
            Grade.A_PLUS: "Exceptional - Strong confluence, ideal setup",
            Grade.A: "Excellent - High probability, clear pattern",
            Grade.B_PLUS: "Very Good - Solid setup with good confluence",
            Grade.B: "Good - Solid setup, acceptable risk",
            Grade.B_MINUS: "Above Average - Decent setup, some concerns",
            Grade.C_PLUS: "Fair - Acceptable but needs confirmation",
            Grade.C: "Mediocre - Weak signals, high risk",
            Grade.D: "Poor - Significant concerns, avoid",
            Grade.F: "Skip - Not tradeable"
        }
        return descriptions.get(grade, "Unknown")


# ============================================================================
# ACTION DETERMINER
# ============================================================================

class ActionDeterminerV4:
    """
    Determines action based on confidence and completion percentage.
    """
    
    def __init__(self):
        self.thresholds = CONFIG.action_thresholds
    
    def determine_action(self, confidence: float, completion_pct: float = 1.0) -> Tuple[str, str]:
        """
        Determine action and action detail.
        Returns (action, action_detail)
        """
        if confidence >= self.thresholds['strong_entry']:
            action = "STRONG_ENTRY"
            detail = f"Exceptional setup with {confidence:.0%} confidence"
        elif confidence >= self.thresholds['enter_now']:
            if completion_pct >= 0.95:
                action = "ENTER_NOW"
                detail = f"Pattern confirmed with {confidence:.0%} confidence"
            else:
                action = "WAIT_FOR_RETEST"
                detail = f"Pattern {completion_pct:.0%} complete - waiting for retest"
        elif confidence >= self.thresholds['wait_retest']:
            action = "WAIT_FOR_RETEST"
            detail = f"Pattern forming - {completion_pct:.0%} complete, {confidence:.0%} confidence"
        else:
            action = "SKIP"
            detail = f"Insufficient confidence ({confidence:.0%})"
        
        return action, detail


# ============================================================================
# MAIN SCORING ENGINE
# ============================================================================

class PatternScoringEngineV4:
    """
    Main scoring engine that orchestrates all scoring components.
    """
    
    def __init__(self):
        self.risk_calculator = RiskPenaltyCalculatorV4()
        self.false_breakout_calculator = FalseBreakoutPenaltyV4()
        self.confidence_calculator = ConfidenceCalculatorV4()
        self.grade_assigner = GradeAssignerV4()
        self.action_determiner = ActionDeterminerV4()
    
    def score_pattern(self, pattern: Dict, df: pd.DataFrame,
                      context_score: float, htf_confluence: Dict = None,
                      regime_validity: float = 1.0,
                      learning_multiplier: float = 1.0) -> Dict:
        """
        Complete scoring pipeline for a single pattern.
        Returns pattern with all scores added.
        """
        
        # 1. Calculate risk penalty
        risk_penalty = self.risk_calculator.calculate(pattern, df)
        pattern['risk_penalty'] = risk_penalty
        
        # 2. Calculate false breakout penalty
        false_breakout_penalty = self.false_breakout_calculator.calculate(pattern, df)
        pattern['false_breakout_penalty'] = false_breakout_penalty
        
        # 3. Calculate final confidence
        if htf_confluence:
            final_confidence = self.confidence_calculator.calculate_with_mtf(
                pattern, context_score, htf_confluence, regime_validity, learning_multiplier
            )
        else:
            final_confidence = self.confidence_calculator.calculate(
                pattern, context_score, 1.0, regime_validity, learning_multiplier
            )
        
        pattern['final_confidence'] = final_confidence
        
        # 4. Assign grade
        grade = self.grade_assigner.assign_grade(final_confidence)
        pattern['grade'] = grade.value
        pattern['grade_description'] = self.grade_assigner.get_grade_description(grade)
        
        # 5. Determine action
        completion_pct = pattern.get('completion_pct', 1.0)
        action, action_detail = self.action_determiner.determine_action(final_confidence, completion_pct)
        pattern['action'] = action
        pattern['action_detail'] = action_detail
        
        # 6. Calculate position multiplier
        pattern['position_multiplier'] = self.grade_assigner.get_position_multiplier(grade)
        
        # 7. Build reasons
        pattern['reasons'] = self._build_reasons(pattern, final_confidence, grade)
        
        return pattern
    
    def score_batch(self, patterns: List[Dict], df: pd.DataFrame,
                    context_scores: List[float], htf_confluence_list: List[Dict] = None,
                    regime_validity: float = 1.0,
                    learning_multiplier: float = 1.0) -> List[Dict]:
        """
        Score multiple patterns.
        """
        scored = []
        
        for i, pattern in enumerate(patterns):
            try:
                context_score = context_scores[i] if i < len(context_scores) else 0.5
                htf_confluence = htf_confluence_list[i] if htf_confluence_list and i < len(htf_confluence_list) else None
                
                scored_pattern = self.score_pattern(
                    pattern, df, context_score, htf_confluence,
                    regime_validity, learning_multiplier
                )
                scored.append(scored_pattern)
            except Exception as e:
                continue
        
        # Sort by confidence (highest first)
        scored.sort(key=lambda x: x.get('final_confidence', 0), reverse=True)
        
        return scored
    
    def _build_reasons(self, pattern: Dict, confidence: float, grade: Grade) -> List[str]:
        """Build human-readable reasons for the decision"""
        reasons = []
        
        # Pattern quality
        similarity = pattern.get('similarity', 0)
        if similarity >= 0.8:
            reasons.append(f"Excellent pattern quality ({similarity:.0%} similarity)")
        elif similarity >= 0.6:
            reasons.append(f"Good pattern quality ({similarity:.0%} similarity)")
        
        # Context
        context_score = pattern.get('context_score', 0)
        if context_score >= 0.7:
            reasons.append(f"Favorable market context ({context_score:.0%})")
        
        # Grade
        grade_descriptions = {
            Grade.A_PLUS: "Exceptional setup - strong confluence",
            Grade.A: "Excellent setup - high probability",
            Grade.B_PLUS: "Very good setup - solid confluence",
            Grade.B: "Good setup - meets all criteria",
            Grade.B_MINUS: "Above average setup - acceptable",
        }
        if grade in grade_descriptions:
            reasons.append(grade_descriptions[grade])
        
        # HTF confluence
        if pattern.get('htf_confluence', {}).get('boost_factor', 1.0) > 1.0:
            reasons.append("Higher timeframe confluence confirms setup")
        
        # Trap conversion
        if pattern.get('trap', {}).get('converted', False):
            reasons.append("Trap detected and converted - favorable reversal")
        
        # Liquidity confirmation
        liquidity = pattern.get('liquidity', {})
        if liquidity.get('has_down_sweep') and pattern.get('direction') == 'BUY':
            reasons.append("Liquidity sweep below support - bullish confirmation")
        elif liquidity.get('has_up_sweep') and pattern.get('direction') == 'SELL':
            reasons.append("Liquidity sweep above resistance - bearish confirmation")
        
        # Risk warnings
        risk_penalty = pattern.get('risk_penalty', 0)
        if risk_penalty > 0.15:
            reasons.append(f"⚠️ Risk penalty applied ({risk_penalty:.0%}) - exercise caution")
        
        return reasons[:7]  # Top 7 reasons
    

    def is_tradeable(self, pattern: Dict) -> Tuple[bool, str]:
        """
        Final check if pattern is tradeable.
        ONLY checks confidence >= CONFIG.min_trade_confidence
        AND risk/reward >= CONFIG.min_trade_rr.
        
        No grade check. No action check. No trap check.
        Change CONFIG.min_trade_confidence and CONFIG.min_trade_rr to tune.
        
        Returns (is_tradeable, reason)
        """
        confidence = pattern.get('final_confidence', 0)
        rr = pattern.get('risk_reward', 0)
        
        min_confidence = CONFIG.min_trade_confidence
        min_rr = CONFIG.min_trade_rr
        
        if confidence < min_confidence:
            return False, f"Confidence {confidence:.0%} < {min_confidence:.0%}"
        
        if rr < min_rr:
            return False, f"RR {rr:.2f} < {min_rr:.1f}"
        
        return True, "Approved for trade"
    
    # def is_tradeable(self, pattern: Dict) -> Tuple[bool, str]:
    #     """
    #     Final check if pattern is tradeable.
    #     Returns (is_tradeable, reason)
    #     """
    #     confidence = pattern.get('final_confidence', 0)
    #     action = pattern.get('action', 'SKIP')
    #     grade = pattern.get('grade', 'F')
        
    #     # Check 1: Must have A, B, or C+ grade
    #     if grade not in ['A+', 'A', 'B+', 'B', 'B-','C+', 'C', 'D']:
    #         return False, f"Grade {grade} too low"
        
    #     # Check 2: Confidence must be >= 0.55
    #     if confidence < 0.30:
    #         return False, f"Confidence {confidence:.0%} < 55%"
        
    #     # Check 3: Action must be ENTER_NOW or STRONG_ENTRY or WAIT_FOR_RETEST
    #     if action not in ['STRONG_ENTRY', 'ENTER_NOW', 'WAIT_FOR_RETEST']:
    #         return False, f"Action {action} not tradeable"
        
    #     # Check 4: Must not have active trap
    #     trap = pattern.get('trap', {})
    #     if trap.get('detected', False) and not trap.get('converted', False):
    #         return False, f"Active trap: {trap.get('trap_type', 'unknown')}"
        
    #     # Check 5: Risk penalty too high
    #     risk_penalty = pattern.get('risk_penalty', 0)
    #     if risk_penalty > 0.35:
    #         return False, f"Risk penalty too high ({risk_penalty:.0%})"
        
    #     return True, "Approved for trade"


# ============================================================================
# SCORE SUMMARY UTILITY
# ============================================================================

def get_score_summary(pattern: Dict) -> Dict[str, Any]:
    """
    Get detailed score summary for debugging.
    """
    return {
        'pattern_name': pattern.get('pattern_name', 'Unknown'),
        'direction': pattern.get('direction', 'NEUTRAL'),
        
        # Similarity components
        'similarity': pattern.get('similarity', 0),
        'similarity_components': pattern.get('components', {}),
        
        # Context components
        'context_score': pattern.get('context_score', 0),
        'context_components': pattern.get('context_components', {}),
        
        # Penalties
        'risk_penalty': pattern.get('risk_penalty', 0),
        'false_breakout_penalty': pattern.get('false_breakout_penalty', 0),
        
        # Final scores
        'final_confidence': pattern.get('final_confidence', 0),
        'grade': pattern.get('grade', 'F'),
        'action': pattern.get('action', 'SKIP'),
        
        # Position sizing
        'position_multiplier': pattern.get('position_multiplier', 1.0),
        
        # Trade setup
        'entry': pattern.get('entry'),
        'stop_loss': pattern.get('stop_loss'),
        'take_profit': pattern.get('take_profit'),
        'risk_reward': pattern.get('risk_reward'),
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'RiskPenaltyCalculatorV4',
    'FalseBreakoutPenaltyV4',
    'ConfidenceCalculatorV4',
    'GradeAssignerV4',
    'ActionDeterminerV4',
    'PatternScoringEngineV4',
    'get_score_summary',
]


