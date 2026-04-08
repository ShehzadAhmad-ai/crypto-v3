# final_scoring_pipeline.py - Phase 9: Final Scoring & Grading
"""
Final Scoring Pipeline - Phase 9

Combines all phase scores into final probability and grade:
- Phase 4: Expert Consensus (agreement_ratio, consensus_confidence)
- Phase 5: MTF Confirmation (alignment_score, mtf_score)
- Phase 6: Smart Money Filter (smart_money_score, net_bias)
- Phase 7: Light Confirmations (cross_asset, funding_oi, sentiment)
- Phase 8: Risk Management (risk_reward_ratio, position_multiplier)

Features:
- Weighted ensemble from config
- Bayesian adjustment based on historical performance
- Grade assignment (A+ through F)
- Position multiplier based on grade
- Edge persistence tracking
"""

from typing import Dict, Optional, List, Any, Tuple
import numpy as np
from datetime import datetime

from config import Config
from logger import log
from signal_model import Signal, SignalStatus


class FinalScoringPipeline:
    """
    Final Scoring Pipeline - Phase 9
    
    Combines all phase scores into final probability and grade.
    """
    
    def __init__(self):
        # ===== LOAD WEIGHTS FROM CONFIG =====
        self.weights = self._load_weights()
        
        # ===== LOAD GRADE THRESHOLDS =====
        self.grade_thresholds = self._load_grade_thresholds()
        
        # ===== POSITION MULTIPLIERS BY GRADE =====
        self.position_multipliers = self._load_position_multipliers()
        
        # ===== MINIMUM THRESHOLDS =====
        self.min_probability = getattr(Config, 'FINAL_MIN_PROBABILITY', 0.65)
        self.min_agreement = getattr(Config, 'FINAL_MIN_AGREEMENT', 0.55)
        
        # ===== HISTORICAL PERFORMANCE FOR BAYESIAN =====
        self.performance_history: List[Dict] = []
        self.max_history = getattr(Config, 'FINAL_MAX_HISTORY', 200)
        
        # ===== BAYESIAN SETTINGS =====
        self.enable_bayesian = getattr(Config, 'FINAL_ENABLE_BAYESIAN', True)
        self.bayesian_weight = getattr(Config, 'FINAL_BAYESIAN_WEIGHT', 0.3)
        
        log.info("=" * 60)
        log.info("Final Scoring Pipeline initialized (Phase 9)")
        log.info("=" * 60)
        log.info(f"  Weights:")
        for key, weight in self.weights.items():
            log.info(f"    {key:20s}: {weight:.2f}")
        log.info(f"  Min Probability: {self.min_probability:.1%}")
        log.info(f"  Min Agreement: {self.min_agreement:.1%}")
        log.info(f"  Bayesian Enabled: {self.enable_bayesian}")
        log.info("=" * 60)
    
    # ========================================================================
    # LOAD CONFIGURATION
    # ========================================================================
    
    def _load_weights(self) -> Dict[str, float]:
        """Load weights from config (sum must be 1.0)"""
        default_weights = {
            'expert_consensus': 0.25,      # Phase 4: All 5 experts agree
            'mtf_confirmation': 0.15,      # Phase 5: Higher timeframe alignment
            'smart_money': 0.15,           # Phase 6: Smart money confirmation
            'light_confirm': 0.10,         # Phase 7: Cross-asset, funding, sentiment
            'risk_reward': 0.10,           # Phase 8: Risk/Reward ratio
            'position_sizing': 0.05,       # Phase 8: Position size factor
            'strategy_agreement': 0.10,    # Phase 4: How many experts agreed
            'timing': 0.10                 # Phase 10: Expected entry timing
        }
        
        weights = {}
        weights['expert_consensus'] = getattr(Config, 'FINAL_WEIGHT_CONSENSUS', default_weights['expert_consensus'])
        weights['mtf_confirmation'] = getattr(Config, 'FINAL_WEIGHT_MTF', default_weights['mtf_confirmation'])
        weights['smart_money'] = getattr(Config, 'FINAL_WEIGHT_SMART_MONEY', default_weights['smart_money'])
        weights['light_confirm'] = getattr(Config, 'FINAL_WEIGHT_LIGHT_CONFIRM', default_weights['light_confirm'])
        weights['risk_reward'] = getattr(Config, 'FINAL_WEIGHT_RISK_REWARD', default_weights['risk_reward'])
        weights['position_sizing'] = getattr(Config, 'FINAL_WEIGHT_POSITION_SIZING', default_weights['position_sizing'])
        weights['strategy_agreement'] = getattr(Config, 'FINAL_WEIGHT_STRATEGY_AGREEMENT', default_weights['strategy_agreement'])
        weights['timing'] = getattr(Config, 'FINAL_WEIGHT_TIMING', default_weights['timing'])
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            log.warning(f"Weights sum to {total:.2f}, normalizing to 1.0")
            for key in weights:
                weights[key] /= total
        
        return weights
    
    def _load_grade_thresholds(self) -> Dict[str, float]:
        """Load grade thresholds from config"""
        return {
            'A+': getattr(Config, 'FINAL_GRADE_A_PLUS', 0.92),
            'A': getattr(Config, 'FINAL_GRADE_A', 0.85),
            'B+': getattr(Config, 'FINAL_GRADE_B_PLUS', 0.78),
            'B': getattr(Config, 'FINAL_GRADE_B', 0.72),
            'B-': getattr(Config, 'FINAL_GRADE_B_MINUS', 0.65),
            'C+': getattr(Config, 'FINAL_GRADE_C_PLUS', 0.60),
            'C': getattr(Config, 'FINAL_GRADE_C', 0.55),
            'D': getattr(Config, 'FINAL_GRADE_D', 0.50),
            'F': getattr(Config, 'FINAL_GRADE_F', 0.00)
        }
    
    def _load_position_multipliers(self) -> Dict[str, float]:
        """Load position multipliers from config"""
        return {
            'A+': getattr(Config, 'POSITION_MULTIPLIER_A_PLUS', 1.5),
            'A': getattr(Config, 'POSITION_MULTIPLIER_A', 1.3),
            'B+': getattr(Config, 'POSITION_MULTIPLIER_B_PLUS', 1.1),
            'B': getattr(Config, 'POSITION_MULTIPLIER_B', 1.0),
            'B-': getattr(Config, 'POSITION_MULTIPLIER_B_MINUS', 0.9),
            'C+': getattr(Config, 'POSITION_MULTIPLIER_C_PLUS', 0.75),
            'C': getattr(Config, 'POSITION_MULTIPLIER_C', 0.5),
            'D': getattr(Config, 'POSITION_MULTIPLIER_D', 0.25),
            'F': getattr(Config, 'POSITION_MULTIPLIER_F', 0.0)
        }
    
    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================
    
    def process_signal(self, signal: Signal) -> Optional[Signal]:
        """
        Process signal through final scoring (Phase 9)
        
        Args:
            signal: Signal that passed Phase 8 (Risk Management)
        
        Returns:
            Updated signal with final probability and grade
        """
        # Only run on signals that passed Risk Management
        if signal.status not in [SignalStatus.RISK_PASSED, SignalStatus.TIMING_PASSED]:
            log.debug(f"Signal status {signal.status} not eligible for final scoring")
            return None
        
        try:
            # ===== STEP 1: COLLECT ALL PHASE SCORES =====
            scores = self._collect_scores(signal)
            
            # ===== STEP 2: CALCULATE AGREEMENT METRICS =====
            agreement_metrics = self._calculate_agreement_metrics(signal)
            
            # ===== STEP 3: CALCULATE WEIGHTED ENSEMBLE =====
            raw_probability = self._calculate_weighted_ensemble(scores, agreement_metrics)
            
            # ===== STEP 4: APPLY BAYESIAN ADJUSTMENT =====
            if self.enable_bayesian:
                bayesian_prob = self._bayesian_adjustment(raw_probability, signal)
                final_probability = (1 - self.bayesian_weight) * raw_probability + self.bayesian_weight * bayesian_prob
            else:
                final_probability = raw_probability
            
            # ===== STEP 5: APPLY MINIMUM AGREEMENT FILTER =====
            if agreement_metrics['overall_agreement'] < self.min_agreement:
                log.debug(f"Agreement {agreement_metrics['overall_agreement']:.1%} < {self.min_agreement:.1%}")
                signal.warning_flags.append(f"Low agreement: {agreement_metrics['overall_agreement']:.1%}")
                # Don't reject, just warn
            
            # ===== STEP 6: ASSIGN GRADE =====
            grade = self._assign_grade(final_probability)
            
            # ===== STEP 7: GET POSITION MULTIPLIER =====
            position_multiplier = self.position_multipliers.get(grade, 1.0)
            
            # ===== STEP 8: CALCULATE CONFIDENCE =====
            confidence = self._calculate_confidence(final_probability, scores, agreement_metrics)
            
            # ===== STEP 9: CALCULATE EDGE PERSISTENCE =====
            edge_persistence = self._calculate_edge_persistence(final_probability)
            
            # ===== STEP 10: UPDATE SIGNAL =====
            signal.probability = round(final_probability, 3)
            signal.confidence_level = round(confidence, 3)
            signal.final_score = round(final_probability, 3)
            signal.signal_grade = grade
            signal.edge_persistence = round(edge_persistence, 3)
            
            # Update position multiplier from grade
            if hasattr(signal, 'metadata') and signal.metadata:
                signal.metadata['position_multiplier'] = position_multiplier
                signal.metadata['grade'] = grade
                signal.metadata['grade_description'] = self._get_grade_description(grade)
            
            # Add confirmation reasons
            signal.confirmation_reasons.extend([
                f"Final probability: {final_probability:.1%}",
                f"Grade: {grade}",
                f"Confidence: {confidence:.1%}",
                f"Expert agreement: {agreement_metrics['expert_agreement']:.0%}",
                f"MTF alignment: {scores['mtf_alignment']:.1%}",
                f"Smart money score: {scores['smart_money']:.1%}"
            ])
            
            # ===== STEP 11: CHECK IF TRADEABLE =====
            if not self.is_tradeable(signal):
                log.debug(f"Signal {signal.symbol} not tradeable: probability {final_probability:.1%} < {self.min_probability:.1%}")
                return None
            
            signal.status = SignalStatus.FINAL
            
            # ===== STEP 12: PRINT OUTPUT =====
            self._print_final_output(signal, scores, agreement_metrics, final_probability, grade, confidence)
            
            return signal
            
        except Exception as e:
            log.error(f"Error in final scoring for {signal.symbol}: {e}", exc_info=True)
            return signal
    
    # ========================================================================
    # SCORE COLLECTION
    # ========================================================================
    
    def _collect_scores(self, signal: Signal) -> Dict[str, float]:
        """Collect scores from all phases"""
        scores = {}
        
        # Phase 4: Expert Consensus
        if hasattr(signal, 'expert_consensus_reached'):
            scores['expert_consensus'] = 1.0 if signal.expert_consensus_reached else 0.5
        else:
            scores['expert_consensus'] = 0.5
        
        # Phase 4: Consensus confidence (if available)
        if hasattr(signal, 'consensus_confidence') and signal.consensus_confidence > 0:
            scores['consensus_confidence'] = signal.consensus_confidence
        else:
            scores['consensus_confidence'] = 0.5
        
        # Phase 5: MTF Confirmation
        scores['mtf_alignment'] = signal.mtf_score if hasattr(signal, 'mtf_score') else 0.5
        
        # Phase 6: Smart Money
        scores['smart_money'] = signal.smart_money_score if hasattr(signal, 'smart_money_score') else 0.5
        
        # Phase 7: Light Confirmations
        light_scores = []
        if hasattr(signal, 'cross_asset_score') and signal.cross_asset_score > 0:
            light_scores.append(signal.cross_asset_score)
        if hasattr(signal, 'funding_oi_score') and signal.funding_oi_score > 0:
            light_scores.append(signal.funding_oi_score)
        if hasattr(signal, 'sentiment_score') and signal.sentiment_score > 0:
            light_scores.append(signal.sentiment_score)
        scores['light_confirm'] = np.mean(light_scores) if light_scores else 0.5
        
        # Phase 8: Risk/Reward
        rr = signal.risk_reward_ratio if hasattr(signal, 'risk_reward_ratio') else 1.5
        scores['risk_reward'] = self._rr_to_score(rr)
        
        # Phase 8: Position Sizing (risk amount as percentage)
        if hasattr(signal, 'risk_amount') and hasattr(signal, 'position_value') and signal.position_value > 0:
            risk_pct = signal.risk_amount / signal.position_value if signal.position_value > 0 else 0.02
            scores['position_sizing'] = min(1.0, risk_pct / 0.02)  # Normalize to 2% base risk
        else:
            scores['position_sizing'] = 0.5
        
        # Phase 10: Timing (if available)
        if hasattr(signal, 'expected_minutes_to_entry') and signal.expected_minutes_to_entry > 0:
            minutes = signal.expected_minutes_to_entry
            if minutes <= 15:
                scores['timing'] = 0.9
            elif minutes <= 30:
                scores['timing'] = 0.7
            elif minutes <= 60:
                scores['timing'] = 0.5
            elif minutes <= 120:
                scores['timing'] = 0.3
            else:
                scores['timing'] = 0.1
        else:
            scores['timing'] = 0.5
        
        return scores
    
    def _rr_to_score(self, risk_reward: float) -> float:
        """Convert risk/reward ratio to a score (0-1)"""
        if risk_reward >= 3.0:
            return 1.0
        elif risk_reward >= 2.0:
            return 0.8 + (risk_reward - 2.0) * 0.2
        elif risk_reward >= 1.5:
            return 0.6 + (risk_reward - 1.5) * 0.4
        elif risk_reward >= 1.0:
            return 0.3 + (risk_reward - 1.0) * 0.6
        else:
            return max(0.1, risk_reward / 3.0)
    
    def _calculate_agreement_metrics(self, signal: Signal) -> Dict[str, float]:
        """Calculate agreement metrics from Phase 4 (Expert Consensus)"""
        metrics = {
            'expert_agreement': 0.0,
            'overall_agreement': 0.0,
            'agreeing_experts': 0,
            'total_experts': 5
        }
        
        # Get expert agreement from metadata
        if hasattr(signal, 'metadata') and signal.metadata:
            expert_details = signal.metadata.get('expert_details', {})
            if expert_details:
                agreeing = sum(1 for e in expert_details.values() if e.get('agreed', False))
                metrics['expert_agreement'] = agreeing / 5.0
                metrics['agreeing_experts'] = agreeing
        
        # Also check signal attributes
        if hasattr(signal, 'expert_consensus_reached'):
            if signal.expert_consensus_reached:
                metrics['expert_agreement'] = 1.0
        
        # Calculate overall agreement (average of all phase agreements)
        phase_agreements = [
            metrics['expert_agreement'],
            signal.mtf_score if hasattr(signal, 'mtf_score') else 0.5,
            signal.smart_money_score if hasattr(signal, 'smart_money_score') else 0.5
        ]
        
        metrics['overall_agreement'] = np.mean(phase_agreements)
        
        return metrics
    
    # ========================================================================
    # SCORE CALCULATION
    # ========================================================================
    
    def _calculate_weighted_ensemble(self, scores: Dict[str, float], 
                                     agreement: Dict[str, float]) -> float:
        """Calculate weighted ensemble score"""
        total = 0.0
        weight_sum = 0.0
        
        # Expert consensus (using agreement as weight)
        consensus_value = scores['expert_consensus'] * (0.5 + agreement['expert_agreement'] * 0.5)
        total += consensus_value * self.weights['expert_consensus']
        weight_sum += self.weights['expert_consensus']
        
        # Strategy agreement (from agreement metrics)
        total += agreement['expert_agreement'] * self.weights['strategy_agreement']
        weight_sum += self.weights['strategy_agreement']
        
        # MTF Confirmation
        total += scores['mtf_alignment'] * self.weights['mtf_confirmation']
        weight_sum += self.weights['mtf_confirmation']
        
        # Smart Money
        total += scores['smart_money'] * self.weights['smart_money']
        weight_sum += self.weights['smart_money']
        
        # Light Confirmations
        total += scores['light_confirm'] * self.weights['light_confirm']
        weight_sum += self.weights['light_confirm']
        
        # Risk/Reward
        total += scores['risk_reward'] * self.weights['risk_reward']
        weight_sum += self.weights['risk_reward']
        
        # Position Sizing
        total += scores['position_sizing'] * self.weights['position_sizing']
        weight_sum += self.weights['position_sizing']
        
        # Timing
        total += scores['timing'] * self.weights['timing']
        weight_sum += self.weights['timing']
        
        if weight_sum > 0:
            return max(0.01, min(0.99, total / weight_sum))
        return 0.5
    
    # ========================================================================
    # BAYESIAN ADJUSTMENT
    # ========================================================================
    
    def _bayesian_adjustment(self, current_prob: float, signal: Signal) -> float:
        """Apply Bayesian adjustment based on historical performance"""
        if len(self.performance_history) < 10:
            return current_prob
        
        # Calculate prior (overall win rate)
        wins = sum(1 for p in self.performance_history if p['won'])
        total = len(self.performance_history)
        prior = wins / total if total > 0 else 0.5
        
        # Find similar signals
        similar = [
            p for p in self.performance_history
            if abs(p['probability'] - current_prob) < 0.1
            or p.get('grade') == signal.signal_grade
        ]
        
        if len(similar) < 5:
            likelihood = prior
        else:
            similar_wins = sum(1 for p in similar if p['won'])
            likelihood = similar_wins / len(similar)
        
        # Bayes theorem
        evidence = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        if evidence > 0:
            posterior = (likelihood * prior) / evidence
        else:
            posterior = prior
        
        return max(0.01, min(0.99, posterior))
    
    # ========================================================================
    # GRADE ASSIGNMENT
    # ========================================================================
    
    def _assign_grade(self, probability: float) -> str:
        """Assign letter grade based on probability"""
        for grade, threshold in sorted(self.grade_thresholds.items(), 
                                       key=lambda x: x[1], reverse=True):
            if probability >= threshold:
                return grade
        return 'F'
    
    def _get_grade_description(self, grade: str) -> str:
        """Get human-readable grade description"""
        descriptions = {
            'A+': "Exceptional - High probability setup with strong conviction",
            'A': "Excellent - Very high probability setup",
            'B+': "Very Good - Above average probability",
            'B': "Good - Solid setup with clear edge",
            'B-': "Decent - Moderate probability, acceptable risk",
            'C+': "Fair - Average setup, careful position sizing",
            'C': "Weak - Below average probability",
            'D': "Poor - High risk, consider skipping",
            'F': "Skip - No trade"
        }
        return descriptions.get(grade, "Unknown grade")
    
    # ========================================================================
    # CONFIDENCE CALCULATION
    # ========================================================================
    
    def _calculate_confidence(self, probability: float, scores: Dict,
                              agreement: Dict) -> float:
        """Calculate confidence level based on score agreement"""
        # Variance of component scores
        score_values = [v for v in scores.values() if v > 0]
        if len(score_values) > 1:
            variance = np.var(score_values)
            agreement_score = 1.0 - min(0.5, variance)
        else:
            agreement_score = 0.5
        
        # Expert agreement factor
        expert_factor = 0.5 + agreement['expert_agreement'] * 0.5
        
        # Probability factor (higher probability = higher confidence)
        prob_factor = 0.5 + probability * 0.5
        
        # Combine
        confidence = (agreement_score * 0.4 + expert_factor * 0.3 + prob_factor * 0.3)
        
        return max(0.1, min(0.95, confidence))
    
    # ========================================================================
    # EDGE PERSISTENCE
    # ========================================================================
    
    def _calculate_edge_persistence(self, current_prob: float) -> float:
        """Calculate edge persistence based on recent performance"""
        if len(self.performance_history) < 10:
            return 0.5
        
        recent = self.performance_history[-20:]
        recent_probs = [p['probability'] for p in recent]
        
        # Consistency (low variance)
        if len(recent_probs) > 1:
            variance = np.var(recent_probs)
            consistency = 1.0 - min(0.5, variance)
        else:
            consistency = 0.5
        
        # Trend (recent vs older)
        if len(recent_probs) >= 10:
            recent_avg = np.mean(recent_probs[-5:])
            older_avg = np.mean(recent_probs[-10:-5])
            if older_avg > 0:
                trend = max(0, (recent_avg - older_avg) / older_avg)
                consistency += min(0.3, trend)
        
        return max(0.1, min(1.0, consistency))
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def is_tradeable(self, signal: Signal) -> bool:
        """Check if signal meets minimum thresholds"""
        return signal.probability >= self.min_probability
    
    def record_outcome(self, signal: Signal, won: bool):
        """Record trade outcome for Bayesian learning"""
        self.performance_history.append({
            'symbol': signal.symbol,
            'probability': signal.probability,
            'grade': signal.signal_grade,
            'direction': signal.direction,
            'market_regime': signal.market_regime,
            'won': won,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_history:
            return {'total': 0, 'win_rate': 0}
        
        wins = sum(1 for p in self.performance_history if p['won'])
        total = len(self.performance_history)
        
        # Win rate by grade
        grade_stats = {}
        for grade in ['A+', 'A', 'B+', 'B', 'B-', 'C+', 'C', 'D', 'F']:
            grade_trades = [p for p in self.performance_history if p.get('grade') == grade]
            if grade_trades:
                grade_wins = sum(1 for p in grade_trades if p['won'])
                grade_stats[grade] = {
                    'trades': len(grade_trades),
                    'wins': grade_wins,
                    'win_rate': grade_wins / len(grade_trades)
                }
        
        return {
            'total': total,
            'wins': wins,
            'win_rate': wins / total if total > 0 else 0,
            'by_grade': grade_stats
        }
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def _print_final_output(self, signal: Signal, scores: Dict,
                           agreement: Dict, probability: float,
                           grade: str, confidence: float):
        """Print formatted final output"""
        log.info("\n" + "=" * 70)
        log.info(f"[FINAL SCORE] {signal.symbol} {signal.timeframe}")
        log.info("=" * 70)
        
        # Phase Scores
        log.info("Phase Scores:")
        log.info(f"  Phase 4 (Expert Consensus):    {scores.get('expert_consensus', 0):.1%} (agreement: {agreement['expert_agreement']:.0%})")
        log.info(f"  Phase 5 (MTF Confirmation):    {scores.get('mtf_alignment', 0):.1%}")
        log.info(f"  Phase 6 (Smart Money):         {scores.get('smart_money', 0):.1%}")
        log.info(f"  Phase 7 (Light Confirm):       {scores.get('light_confirm', 0):.1%}")
        log.info(f"  Phase 8 (Risk/Reward):         {scores.get('risk_reward', 0):.1%}")
        log.info(f"  Phase 8 (Position Sizing):     {scores.get('position_sizing', 0):.1%}")
        log.info(f"  Phase 10 (Timing):             {scores.get('timing', 0):.1%}")
        
        log.info("-" * 70)
        log.info(f"Final Probability: {probability:.1%}")
        log.info(f"Grade: {grade}")
        log.info(f"Confidence: {confidence:.1%}")
        log.info(f"Edge Persistence: {signal.edge_persistence:.1%}")
        
        # Position Multiplier
        multiplier = self.position_multipliers.get(grade, 1.0)
        log.info(f"Position Multiplier: {multiplier:.1f}x")
        
        log.info("-" * 70)
        log.info(f"Decision: {'TRADE' if self.is_tradeable(signal) else 'SKIP'}")
        log.info(f"Action: {self._get_action_from_grade(grade)}")
        
        log.info("=" * 70)
    
    def _get_action_from_grade(self, grade: str) -> str:
        """Get action recommendation from grade"""
        if grade in ['A+', 'A']:
            return "STRONG_ENTRY"
        elif grade in ['B+', 'B', 'B-']:
            return "ENTER_NOW"
        elif grade in ['C+', 'C']:
            return "CAUTIOUS_ENTRY"
        else:
            return "SKIP"