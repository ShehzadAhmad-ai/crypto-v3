"""
Strategy Factory - Main Orchestrator for Strategy Expert
Runs the complete 8-step pipeline to generate trading signals

Pipeline Steps:
1. Run Strategies - Execute all enabled strategies to get raw outputs
2. Quality Filter - Drop strategies with confidence < min_confidence
3. Limit to Top N - Sort by weight, take top N strategies (default 5)
4. Weighted Voting - buy_score = Σ(conf × weight), sell_score = Σ(conf × weight)
5. Agreement Check - agreement_ratio >= min_agreement (default 0.55)
6. Weighted Average - entry/SL/TP = Σ(value × weight) / Σ(weight)
7. Risk Filter - risk_reward >= min_risk_reward (default 1.5)
8. Late Entry Filter - |current_price - entry| <= ATR × multiplier
9. Generate Output - direction, entry, SL, TP, confidence, grade, reasons

Features:
- Auto-loads all strategies from strategies/ folder
- Config-driven strategy enable/disable
- Dynamic weight management based on performance
- Complete 8-step pipeline with full logging
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import uuid

from strategy_expert.base_strategy import BaseStrategy, StrategyOutput
from strategy_expert.strategy_config import StrategyExpertConfigLoader, get_pipeline_config, PipelineConfig
from strategy_expert.strategy_loader import StrategyLoader
from strategy_expert.weight_manager import WeightManager
from strategy_expert.voting_engine import VotingEngine, VoteResult, VotingStats
from strategy_expert.trade_combiner import TradeCombiner, CombinedTrade
from strategy_expert.risk_filters import RiskFilter, FilterResult
from strategy_expert.scoring_engine import ScoringEngine, SignalGrade, Grade


class StrategyFactory:
    """
    Main orchestrator for Strategy Expert
    Runs the complete 8-step pipeline to generate trading signals
    
    Usage:
        factory = StrategyFactory()
        signal = factory.run(df, indicators, market_regime)
        
        if signal:
            print(f"Signal: {signal['direction']} | Grade: {signal['grade']}")
            print(f"Entry: {signal['entry']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}")
    """
    
    def __init__(self, config_loader: StrategyExpertConfigLoader = None,
                 auto_load: bool = True,
                 strategies_path: Optional[str] = None):
        """
        Initialize Strategy Factory
        
        Args:
            config_loader: Configuration loader instance
            auto_load: Automatically load all strategies from strategies folder
            strategies_path: Custom path to strategies folder (optional)
        """
        # Configuration
        self.config_loader = config_loader or StrategyExpertConfigLoader()
        self.pipeline_config: PipelineConfig = get_pipeline_config()
        
        # Initialize core components
        self.weight_manager = WeightManager(self.config_loader)
        self.voting_engine = VotingEngine(self.pipeline_config)
        self.trade_combiner = TradeCombiner()
        self.risk_filter = RiskFilter(self.pipeline_config)
        self.scoring_engine = ScoringEngine(self.pipeline_config)
        
        # Strategy registry
        self.strategies: Dict[str, BaseStrategy] = {}
        
        # Stats tracking
        self.last_run_stats: Dict[str, Any] = {}
        self.run_history: List[Dict[str, Any]] = []
        
        # Auto-load strategies if requested
        if auto_load:
            self._auto_load_strategies(strategies_path)
    
    def _auto_load_strategies(self, strategies_path: Optional[str] = None):
        """
        Auto-load all strategies from strategies folder
        
        Args:
            strategies_path: Custom path to strategies folder
        """
        print("\n" + "="*70)
        print("STRATEGY EXPERT - INITIALIZATION")
        print("="*70)
        
        # Load strategies using StrategyLoader
        loader = StrategyLoader(strategies_path)
        all_strategies = loader.load_all_strategies()
        
        # Register each strategy with config
        registered_count = 0
        disabled_count = 0
        error_count = 0
        
        for name, strategy in all_strategies.items():
            try:
                if self.config_loader.is_strategy_enabled(name):
                    self._register_strategy(strategy)
                    registered_count += 1
                    print(f"  ✅ Registered: {name}")
                else:
                    disabled_count += 1
                    print(f"  ⏸️  Disabled: {name}")
            except Exception as e:
                error_count += 1
                print(f"  ❌ Error registering {name}: {e}")
        
        print("-" * 70)
        print(f"📊 Summary:")
        print(f"   ✅ Registered: {registered_count} strategies")
        print(f"   ⏸️  Disabled: {disabled_count} strategies")
        print(f"   ❌ Errors: {error_count} strategies")
        print(f"   📦 Total available: {len(all_strategies)} strategies")
        print("="*70 + "\n")
        
        # Log any loader errors
        if loader.load_errors:
            print("⚠️  Strategy Loader Errors:")
            for name, error in loader.load_errors.items():
                print(f"   • {name}: {error}")
            print()
    
    def _register_strategy(self, strategy: BaseStrategy):
        """
        Register a single strategy with configuration
        
        Args:
            strategy: Strategy instance
        """
        # Get config for this strategy
        strat_config = self.config_loader.get_strategy_config(strategy.name)
        
        # Apply configuration
        strategy.enabled = strat_config.enabled
        strategy.base_weight = strat_config.base_weight
        strategy.weight = self.weight_manager.get_weight(strategy.name)
        strategy.min_confidence = strat_config.min_confidence
        strategy.min_rr = strat_config.min_rr
        
        # Store strategy
        self.strategies[strategy.name] = strategy
    
    def register_strategies(self, strategies: List[BaseStrategy]):
        """
        Register multiple strategies manually
        
        Args:
            strategies: List of strategy instances
        """
        for strategy in strategies:
            self._register_strategy(strategy)
        print(f"✅ Manually registered {len(strategies)} strategies")
    
    def reload_strategies(self):
        """
        Reload all strategies (useful for development)
        Updates weights and configs without restarting
        """
        print("🔄 Reloading strategies...")
        
        for name, strategy in self.strategies.items():
            # Get fresh config
            strat_config = self.config_loader.get_strategy_config(name)
            
            # Update strategy
            strategy.enabled = strat_config.enabled
            strategy.base_weight = strat_config.base_weight
            strategy.weight = self.weight_manager.get_weight(name)
            strategy.min_confidence = strat_config.min_confidence
            strategy.min_rr = strat_config.min_rr
        
        print(f"✅ Reloaded {len(self.strategies)} strategies")
    
    def run(self, df: pd.DataFrame, indicators: Dict[str, Any],
            market_regime: Dict[str, Any], module_signals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Run the complete 8-step pipeline to generate a trading signal
        
        Args:
            df: Raw OHLCV DataFrame
            indicators: Pre-calculated indicators (RSI, MACD, ADX, ATR, etc.)
            market_regime: Market regime data (regime, bias, trend_strength)
            module_signals: Optional signals from other modules (for reference only)
        
        Returns:
            Final signal dictionary or None if no trade
        """
        # Initialize stats for this run
        self.last_run_stats = {
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'steps': {}
        }
        
        try:
            # =================================================================
            # STEP 1: Run all enabled strategies to get raw outputs
            # =================================================================
            raw_outputs = self._step1_run_strategies(df, indicators, market_regime, module_signals)
            self.last_run_stats['steps']['step1_raw_count'] = len(raw_outputs)
            
            if not raw_outputs:
                self.last_run_stats['status'] = 'no_signals'
                self.last_run_stats['reason'] = 'No strategies generated signals'
                return None
            
            # =================================================================
            # STEP 2: Quality Filter (confidence >= min_confidence)
            # =================================================================
            quality_outputs = self._step2_quality_filter(raw_outputs)
            self.last_run_stats['steps']['step2_quality_count'] = len(quality_outputs)
            
            if not quality_outputs:
                self.last_run_stats['status'] = 'quality_filter'
                self.last_run_stats['reason'] = 'All strategies failed quality filter'
                return None
            
            # =================================================================
            # STEP 3: Limit to Top N strategies by weight
            # =================================================================
            top_outputs = self._step3_limit_top_strategies(quality_outputs)
            self.last_run_stats['steps']['step3_top_count'] = len(top_outputs)
            self.last_run_stats['steps']['top_strategies'] = [s.strategy_name for s in top_outputs]
            
            if not top_outputs:
                self.last_run_stats['status'] = 'top_filter'
                self.last_run_stats['reason'] = 'No strategies after top N filter'
                return None
            
            # =================================================================
            # STEP 4 & 5: Weighted Voting + Agreement Check
            # =================================================================
            vote_result, vote_stats = self._step4_voting(top_outputs)
            self.last_run_stats['steps']['vote_result'] = vote_result.value
            self.last_run_stats['steps']['vote_stats'] = vote_stats.to_dict()
            
            if vote_result not in [VoteResult.BUY, VoteResult.SELL]:
                self.last_run_stats['status'] = 'voting'
                self.last_run_stats['reason'] = self.voting_engine.generate_vote_reason(vote_result, vote_stats)
                return None
            
            # =================================================================
            # STEP 6: Weighted Average for Entry/SL/TP
            # =================================================================
            combined_trade = self._step6_weighted_average(top_outputs, vote_result, vote_stats)
            
            if not combined_trade:
                self.last_run_stats['status'] = 'combine'
                self.last_run_stats['reason'] = 'Failed to combine trades'
                return None
            
            self.last_run_stats['steps']['combined_trade'] = combined_trade.to_dict()
            
            # =================================================================
            # STEP 7: Risk Filter (RR >= min_risk_reward)
            # =================================================================
            current_price = indicators.get('price', df['close'].iloc[-1])
            
            risk_result, risk_stats = self._step7_risk_filter(
                combined_trade, current_price, indicators, df
            )
            self.last_run_stats['steps']['risk_filter'] = risk_stats.to_dict()
            
            if risk_result != FilterResult.PASS:
                self.last_run_stats['status'] = 'risk_filter'
                self.last_run_stats['reason'] = risk_stats.reason
                return None
            
            # =================================================================
            # STEP 8: Generate Output with Scoring
            # =================================================================
            final_signal = self._step8_generate_output(
                combined_trade, vote_result, vote_stats, top_outputs, market_regime
            )
            
            self.last_run_stats['status'] = 'success'
            self.last_run_stats['reason'] = 'Signal generated successfully'
            self.last_run_stats['signal'] = final_signal
            
            # Store in history
            self.run_history.append(self.last_run_stats.copy())
            if len(self.run_history) > 100:
                self.run_history = self.run_history[-100:]
            
            return final_signal
            
        except Exception as e:
            self.last_run_stats['status'] = 'error'
            self.last_run_stats['reason'] = str(e)
            print(f"❌ Error in Strategy Factory: {e}")
            return None
        

    def analyze_with_all_data(self, df: pd.DataFrame, symbol: str, timeframe: str,
                              market_regime: Dict, indicators: Dict,
                              consolidated_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Analyze with consolidated data from all other experts
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading pair symbol
            timeframe: Timeframe string
            market_regime: Market regime data
            indicators: Technical indicators
            consolidated_data: Consolidated data from all other experts
                Contains:
                - expert_signals: Signals from Pattern, Price Action, SMC, Technical
                - expert_agreement: Buy/sell/hold counts
                - pattern_data: Pattern V3 specific data
                - price_action_data: Price Action specific data
                - smc_data: SMC specific data
                - technical_data: Technical specific data
                - structure: Market structure data
                - support_resistance: S/R data
                - liquidity: Liquidity data
        """
        try:
            from logger import log
            
            # Extract consolidated data
            expert_signals = consolidated_data.get('expert_signals', {})
            expert_agreement = consolidated_data.get('expert_agreement', {})
            pattern_data = consolidated_data.get('pattern_data', {})
            price_action_data = consolidated_data.get('price_action_data', {})
            smc_data = consolidated_data.get('smc_data', {})
            technical_data = consolidated_data.get('technical_data', {})
            structure_data = consolidated_data.get('structure', {})
            sr_data = consolidated_data.get('support_resistance', {})
            liquidity_data = consolidated_data.get('liquidity', {})
            
            # ===== EXPERT CONSENSUS WEIGHTING =====
            buy_count = expert_agreement.get('buy_count', 0)
            sell_count = expert_agreement.get('sell_count', 0)
            total_count = buy_count + sell_count
            
            if total_count > 0:
                agreement_ratio = max(buy_count, sell_count) / total_count
            else:
                agreement_ratio = 0.5
            
            # Determine direction from other experts
            if buy_count > sell_count:
                expert_direction = 'BUY'
                base_confidence = expert_agreement.get('avg_confidence', {}).get('buy', 0.6)
            elif sell_count > buy_count:
                expert_direction = 'SELL'
                base_confidence = expert_agreement.get('avg_confidence', {}).get('sell', 0.6)
            else:
                expert_direction = None
                base_confidence = 0.5
            
            # ===== PATTERN V3 CONFLUENCE =====
            pattern_confidence = pattern_data.get('pattern_confidence', 0.5)
            best_pattern = pattern_data.get('best_pattern', '')
            
            # ===== PRICE ACTION CONFLUENCE =====
            pa_confidence = price_action_data.get('pattern_confidence', 0.5)
            pa_pattern = price_action_data.get('pattern_name', '')
            trap_detected = price_action_data.get('trap_detected', False)
            
            # ===== SMC CONFLUENCE =====
            smc_confidence = smc_data.get('confidence', 0.5) if isinstance(smc_data, dict) else 0.5
            amd_phase = smc_data.get('amd_phase', 'UNKNOWN')
            has_liquidity_sweep = bool(smc_data.get('liquidity_sweeps', []))
            
            # ===== TECHNICAL CONFLUENCE =====
            tech_confidence = technical_data.get('confidence', 0.5) if isinstance(technical_data, dict) else 0.5
            regime = technical_data.get('regime', 'UNKNOWN')
            
            # ===== MARKET STRUCTURE =====
            structure_trend = structure_data.get('trend', 'NEUTRAL')
            
            # ===== CALCULATE FINAL CONFIDENCE =====
            confidence_components = [
                (base_confidence, 0.25),      # Expert consensus
                (pattern_confidence, 0.15),   # Pattern V3
                (pa_confidence, 0.15),        # Price Action
                (smc_confidence, 0.20),       # SMC
                (tech_confidence, 0.15),      # Technical
                (agreement_ratio, 0.10)       # Agreement ratio
            ]
            
            total_confidence = sum(comp * weight for comp, weight in confidence_components)
            total_confidence = min(0.95, max(0.05, total_confidence))
            
            # ===== DETERMINE DIRECTION =====
            if expert_direction:
                direction = expert_direction
            else:
                if regime == 'BULL_TREND' or structure_trend == 'BULL':
                    direction = 'BUY'
                elif regime == 'BEAR_TREND' or structure_trend == 'BEAR':
                    direction = 'SELL'
                else:
                    direction = 'HOLD'
            
            # ===== ADJUST FOR TRAPS =====
            if trap_detected:
                direction = 'SELL' if direction == 'BUY' else 'BUY'
                total_confidence = min(0.85, total_confidence + 0.10)
            
            # ===== ADJUST FOR LIQUIDITY SWEEPS =====
            if has_liquidity_sweep:
                total_confidence = min(0.90, total_confidence + 0.05)
            
            # ===== CALCULATE ENTRY, SL, TP =====
            current_price = float(df['close'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
            
            if direction == 'BUY':
                entry = current_price * 0.998
                stop_loss = current_price - atr * 1.5
                take_profit = current_price + atr * 2.5
            elif direction == 'SELL':
                entry = current_price * 1.002
                stop_loss = current_price + atr * 1.5
                take_profit = current_price - atr * 2.5
            else:
                return None
            
            # Calculate risk/reward
            if direction == 'BUY':
                risk = entry - stop_loss
                reward = take_profit - entry
            else:
                risk = stop_loss - entry
                reward = entry - take_profit
            
            risk_reward = reward / risk if risk > 0 else 0
            
            # Determine grade
            if total_confidence >= 0.85:
                grade = 'A'
            elif total_confidence >= 0.75:
                grade = 'B'
            elif total_confidence >= 0.65:
                grade = 'C'
            else:
                grade = 'D'
            
            # Build reasons
            reasons = [
                f"Expert consensus: {buy_count} BUY / {sell_count} SELL",
                f"Pattern V3: {best_pattern} ({pattern_confidence:.0%})",
                f"Price Action: {pa_pattern} ({pa_confidence:.0%})",
                f"SMC: {amd_phase} phase",
                f"Technical regime: {regime}"
            ]
            
            return {
                "module": "strategy_expert",
                "signal_id": f"strat_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "direction": direction,
                "confidence": round(total_confidence, 3),
                "grade": grade,
                "position_multiplier": 1.0,
                "entry": round(entry, 4),
                "stop_loss": round(stop_loss, 4),
                "take_profit": round(take_profit, 4),
                "risk_reward": round(risk_reward, 2),
                "action": "STRONG_ENTRY" if total_confidence >= 0.80 else "ENTRY",
                "decision_reason": f"{direction} | Confidence {total_confidence:.0%} | Grade {grade}",
                "reasons": reasons[:5],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in analyze_with_all_data: {e}")
            return None
    
    def _step1_run_strategies(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, module_signals: Dict = None) -> List[StrategyOutput]:
        """
        STEP 1: Run all enabled strategies and collect outputs
        """
        outputs = []
        
        for name, strategy in self.strategies.items():
            # Check if strategy is enabled
            if not strategy.enabled:
                continue
            
            try:
                # Generate signal
                signal = strategy.generate_signal(df, indicators, market_regime, module_signals)
                
                if signal and signal.action != 'HOLD':
                    outputs.append(signal)
                    
            except Exception as e:
                # Log error but continue with other strategies
                print(f"⚠️  Error in strategy {name}: {e}")
                continue
        
        return outputs
    
    def _step2_quality_filter(self, outputs: List[StrategyOutput]) -> List[StrategyOutput]:
        """
        STEP 2: Apply quality filter
        Keep only strategies with confidence >= min_confidence
        """
        filtered = []
        
        for output in outputs:
            # Get strategy config
            strat_config = self.config_loader.get_strategy_config(output.strategy_name)
            min_conf = strat_config.min_confidence
            
            if output.confidence >= min_conf:
                filtered.append(output)
        
        return filtered
    
    def _step3_limit_top_strategies(self, outputs: List[StrategyOutput]) -> List[StrategyOutput]:
        """
        STEP 3: Limit to top N strategies by weight
        """
        max_strategies = self.pipeline_config.max_strategies_to_use
        
        if len(outputs) <= max_strategies:
            return outputs
        
        # Sort by weight (descending) and take top N
        sorted_outputs = sorted(outputs, key=lambda x: x.weight, reverse=True)
        return sorted_outputs[:max_strategies]
    
    def _step4_voting(self, outputs: List[StrategyOutput]) -> Tuple[VoteResult, VotingStats]:
        """
        STEP 4 & 5: Weighted Voting and Agreement Check
        """
        return self.voting_engine.vote(outputs)
    
    def _step6_weighted_average(self, outputs: List[StrategyOutput],
                                 vote_result: VoteResult,
                                 vote_stats: VotingStats) -> Optional[CombinedTrade]:
        """
        STEP 6: Weighted average for Entry/SL/TP
        """
        return self.trade_combiner.combine(outputs, vote_result, vote_stats)
    
    def _step7_risk_filter(self, combined_trade: CombinedTrade,
                           current_price: float,
                           indicators: Dict,
                           df: pd.DataFrame) -> Tuple[FilterResult, Any]:
        """
        STEP 7: Risk Filter (RR + Late Entry)
        """
        return self.risk_filter.apply_all_filters(
            combined_trade, current_price, indicators, df
        )
    
    def _step8_generate_output(self, combined_trade: CombinedTrade,
                               vote_result: VoteResult,
                               vote_stats: VotingStats,
                               strategy_outputs: List[StrategyOutput],
                               market_regime: Dict) -> Dict[str, Any]:
        """
        STEP 8: Generate final output with scoring and grading
        """
        # Get winning strategies
        if vote_result == VoteResult.BUY:
            winning_strategies = vote_stats.buy_strategies
        else:
            winning_strategies = vote_stats.sell_strategies
        
        # Get current strategy weights for scoring
        strategy_weights = {
            name: self.weight_manager.get_weight(name) 
            for name in self.strategies.keys()
        }
        
        # Calculate regime bias from market regime
        regime_bias = market_regime.get('bias_score', 0)
        
        # Grade the signal
        signal_grade = self.scoring_engine.grade_signal(
            combined_trade, vote_result, vote_stats,
            regime_bias, strategy_weights
        )
        
        # Generate decision reason
        decision_reason = self._generate_decision_reason(
            vote_result, winning_strategies, signal_grade
        )
        
        # Generate signal ID
        signal_id = f"strat_{combined_trade.direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Determine action based on grade
        action = self._determine_action(signal_grade.grade)
        
        # Get best reasons from top strategies
        best_reasons = self._collect_best_reasons(strategy_outputs, winning_strategies)
        
        return {
            "module": "strategy_expert",
            "signal_id": signal_id,
            "direction": combined_trade.direction,
            "confidence": round(signal_grade.components.confidence, 3),
            "grade": signal_grade.grade.value,
            "position_multiplier": round(signal_grade.position_multiplier, 2),
            "entry": round(combined_trade.entry, 4),
            "stop_loss": round(combined_trade.stop_loss, 4),
            "take_profit": round(combined_trade.take_profit, 4),
            "risk_reward": round(combined_trade.risk_reward, 2),
            "action": action,
            "decision_reason": decision_reason,
            "reasons": best_reasons,
            "strategies_used": winning_strategies,
            "strategies_total": vote_stats.total_strategies,
            "strategies_agreed": len(winning_strategies),
            "agreement_ratio": round(vote_stats.agreement_ratio, 3),
            "buy_score": round(vote_stats.buy_score, 3),
            "sell_score": round(vote_stats.sell_score, 3),
            "timestamp": datetime.now().isoformat(),
            "score": round(signal_grade.score, 3),
            "strengths": signal_grade.strengths,
            "weaknesses": signal_grade.weaknesses
        }
    
    def _generate_decision_reason(self, vote_result: VoteResult,
                                  winning_strategies: List[str],
                                  signal_grade: SignalGrade) -> str:
        """
        Generate human-readable decision reason
        """
        direction = vote_result.value
        
        # Format strategy names nicely
        if len(winning_strategies) <= 3:
            strategies_str = " + ".join(winning_strategies)
        else:
            strategies_str = f"{len(winning_strategies)} strategies"
        
        base_reason = f"{direction}: {strategies_str} aligned"
        
        # Add grade info
        if signal_grade.grade == Grade.A:
            base_reason += f" [Grade A - {signal_grade.recommendation}]"
        elif signal_grade.grade == Grade.B:
            base_reason += f" [Grade B - {signal_grade.recommendation}]"
        elif signal_grade.grade == Grade.C:
            base_reason += f" [Grade C - {signal_grade.recommendation}]"
        elif signal_grade.grade == Grade.D:
            base_reason += f" [Grade D - {signal_grade.recommendation}]"
        
        return base_reason
    
    def _determine_action(self, grade: Grade) -> str:
        """
        Determine action based on grade
        """
        if grade == Grade.A:
            return "STRONG_ENTRY"
        elif grade == Grade.B:
            return "ENTRY"
        elif grade == Grade.C:
            return "CAUTIOUS_ENTRY"
        elif grade == Grade.D:
            return "REDUCED_ENTRY"
        else:
            return "SKIP"
    
    def _collect_best_reasons(self, strategy_outputs: List[StrategyOutput],
                              winning_strategies: List[str]) -> List[str]:
        """
        Collect best reasons from winning strategies
        """
        all_reasons = []
        
        for output in strategy_outputs:
            if output.strategy_name in winning_strategies:
                all_reasons.extend(output.reasons)
        
        # Remove duplicates and limit to 5
        seen = set()
        unique_reasons = []
        for reason in all_reasons:
            if reason not in seen and len(unique_reasons) < 5:
                seen.add(reason)
                unique_reasons.append(reason)
        
        return unique_reasons
    
    # ========================================================================
    # Public Methods for Management
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from the last run"""
        return self.last_run_stats
    
    def get_run_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent run history"""
        return self.run_history[-limit:]
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current weights for all strategies"""
        return {
            name: self.weight_manager.get_weight(name)
            for name in self.strategies.keys()
        }
    
    def get_strategy_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered strategies"""
        info = []
        for name, strategy in self.strategies.items():
            info.append({
                'name': name,
                'enabled': strategy.enabled,
                'weight': strategy.weight,
                'base_weight': strategy.base_weight,
                'min_confidence': strategy.min_confidence,
                'min_rr': strategy.min_rr,
                'description': strategy.description
            })
        return sorted(info, key=lambda x: x['weight'], reverse=True)
    
    def update_strategy_weight(self, strategy_name: str, won: bool, 
                               risk_reward: float):
        """
        Update strategy weight based on trade outcome
        
        Args:
            strategy_name: Name of the strategy
            won: Whether the trade was profitable
            risk_reward: Achieved risk/reward ratio
        """
        # Record trade in weight manager
        self.weight_manager.record_trade(strategy_name, won, risk_reward)
        
        # Update strategy instance weight
        if strategy_name in self.strategies:
            new_weight = self.weight_manager.get_weight(strategy_name)
            self.strategies[strategy_name].weight = new_weight
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            print(f"✅ Enabled: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            print(f"⏸️  Disabled: {strategy_name}")
    
    def print_summary(self):
        """Print detailed summary of current state"""
        print("\n" + "="*70)
        print("STRATEGY EXPERT - SYSTEM SUMMARY")
        print("="*70)
        
        # Strategies
        print(f"\n📊 STRATEGIES:")
        print(f"   Total Registered: {len(self.strategies)}")
        enabled_count = sum(1 for s in self.strategies.values() if s.enabled)
        print(f"   ✅ Enabled: {enabled_count}")
        print(f"   ⏸️  Disabled: {len(self.strategies) - enabled_count}")
        
        # Weights (Top 10)
        print(f"\n⚖️  STRATEGY WEIGHTS (Top 10):")
        weights = self.get_strategy_weights()
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]:
            status = "✅" if self.strategies[name].enabled else "⏸️"
            print(f"   {status} {name[:40]:<40}: {weight:.3f}")
        
        # Performance
        perf_summary = self.weight_manager.get_performance_summary()
        print(f"\n📈 PERFORMANCE:")
        print(f"   Total Trades: {perf_summary['total_trades']}")
        print(f"   Active Strategies: {perf_summary['total_strategies']}")
        
        # Last Run
        if self.last_run_stats:
            print(f"\n🔄 LAST RUN:")
            print(f"   Status: {self.last_run_stats.get('status', 'N/A')}")
            print(f"   Reason: {self.last_run_stats.get('reason', 'N/A')}")
            if self.last_run_stats.get('steps'):
                steps = self.last_run_stats['steps']
                print(f"   Raw Strategies: {steps.get('step1_raw_count', 0)}")
                print(f"   After Quality: {steps.get('step2_quality_count', 0)}")
                print(f"   Top Strategies: {steps.get('step3_top_count', 0)}")
                print(f"   Vote Result: {steps.get('vote_result', 'N/A')}")
        
        # Pipeline Config
        print(f"\n⚙️  PIPELINE CONFIG:")
        print(f"   Min Agreement: {self.pipeline_config.min_agreement_ratio:.0%}")
        print(f"   Min Risk/Reward: {self.pipeline_config.min_risk_reward}")
        print(f"   Max Strategies: {self.pipeline_config.max_strategies_to_use}")
        print(f"   Late Entry Multiplier: {self.pipeline_config.late_entry_atr_multiplier}")
        
        print("\n" + "="*70)
    
    def export_signal(self, signal: Dict[str, Any], format: str = "json") -> str:
        """
        Export signal in different formats
        
        Args:
            signal: Signal dictionary
            format: Output format ('json', 'csv', or 'text')
        
        Returns:
            Formatted string representation
        """
        import json
        
        if format == "json":
            return json.dumps(signal, indent=2)
        
        elif format == "csv":
            # Simple CSV format for signal
            fields = ['direction', 'entry', 'stop_loss', 'take_profit', 'risk_reward', 
                     'grade', 'confidence', 'position_multiplier', 'action']
            values = [str(signal.get(f, '')) for f in fields]
            return ",".join(fields) + "\n" + ",".join(values)
        
        else:  # text format
            lines = [
                "="*50,
                "STRATEGY EXPERT SIGNAL",
                "="*50,
                f"Direction: {signal.get('direction', 'N/A')}",
                f"Grade: {signal.get('grade', 'N/A')}",
                f"Confidence: {signal.get('confidence', 0):.1%}",
                f"Position Multiplier: {signal.get('position_multiplier', 1.0)}x",
                f"",
                f"Entry: {signal.get('entry', 'N/A')}",
                f"Stop Loss: {signal.get('stop_loss', 'N/A')}",
                f"Take Profit: {signal.get('take_profit', 'N/A')}",
                f"Risk/Reward: {signal.get('risk_reward', 0):.2f}",
                f"",
                f"Action: {signal.get('action', 'N/A')}",
                f"Decision: {signal.get('decision_reason', 'N/A')}",
                f"",
                f"Strategies: {', '.join(signal.get('strategies_used', []))}",
                f"Agreement: {signal.get('agreement_ratio', 0):.0%} ({signal.get('strategies_agreed', 0)}/{signal.get('strategies_total', 0)})",
                f"",
                f"Signal ID: {signal.get('signal_id', 'N/A')}",
                f"Timestamp: {signal.get('timestamp', 'N/A')}",
                "="*50
            ]
            return "\n".join(lines)


# ============================================================================
# SIMPLIFIED INTERFACE
# ============================================================================

class StrategyExpert:
    """
    Simplified interface for Strategy Expert
    Easier to use with minimal configuration
    
    Usage:
        expert = StrategyExpert()
        signal = expert.analyze(df, indicators, market_regime)
        
        if signal:
            print(expert.export_signal(signal))
    """
    
    def __init__(self, config_overrides: Dict[str, Any] = None,
                 auto_load: bool = True,
                 strategies_path: Optional[str] = None):
        """
        Initialize Strategy Expert
        
        Args:
            config_overrides: Optional configuration overrides
            auto_load: Automatically load all strategies
            strategies_path: Custom path to strategies folder
        """
        # Create config loader with overrides
        config_loader = StrategyExpertConfigLoader(config_overrides)
        
        # Create factory
        self.factory = StrategyFactory(
            config_loader=config_loader,
            auto_load=auto_load,
            strategies_path=strategies_path
        )
        
        self.last_signal = None
    
    def analyze(self, df: pd.DataFrame, indicators: Dict[str, Any],
                market_regime: Dict[str, Any], 
                module_signals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze market and generate signal
        
        Args:
            df: OHLCV DataFrame
            indicators: Technical indicators
            market_regime: Market regime data
            module_signals: Signals from other modules
        
        Returns:
            Signal dictionary or None
        """
        self.last_signal = self.factory.run(
            df, indicators, market_regime, module_signals
        )
        return self.last_signal
    
    def get_signal(self) -> Optional[Dict[str, Any]]:
        """Get the last generated signal"""
        return self.last_signal
    
    def record_trade_result(self, strategy_name: str, won: bool, risk_reward: float):
        """Record trade result for weight updates"""
        self.factory.update_strategy_weight(strategy_name, won, risk_reward)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.factory.get_stats()
    
    def print_summary(self):
        """Print summary"""
        self.factory.print_summary()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.factory.get_strategy_weights()
    
    def get_strategies(self) -> List[str]:
        """Get list of all strategy names"""
        return list(self.factory.strategies.keys())
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy"""
        self.factory.enable_strategy(strategy_name)
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy"""
        self.factory.disable_strategy(strategy_name)
    
    def export_signal(self, signal: Dict[str, Any] = None, 
                     format: str = "text") -> str:
        """Export signal in readable format"""
        if signal is None:
            signal = self.last_signal
        if signal is None:
            return "No signal available"
        return self.factory.export_signal(signal, format)


# ============================================================================
# DIRECT USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # This is a test/example block
    print("Strategy Factory Module Loaded")
    print("Use StrategyExpert() to initialize the system")
    
    # Example:
    # expert = StrategyExpert()
    # signal = expert.analyze(df, indicators, market_regime)
    # print(expert.export_signal(signal))