"""
expert_executor.py - Runs All 5 Experts on a Single Coin

Features:
- Runs all 5 experts (Pattern, Price Action, SMC, Technical, Strategy)
- Handles experts that don't generate signals (re-runs for direction)
- Saves raw signals to signals/raw/ for debugging
- Returns list of ExpertSignal objects to aggregator

Version: 3.0 (Complete Rewrite)
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
import traceback

from expert_interface import ExpertSignal, create_skip_signal, create_direction_only_signal


class ExpertExecutor:
    """
    Executes all 5 experts and collects their signals
    Handles "no signal" cases by re-running for direction preference
    """
    
    def __init__(self, config: Dict = None, debug: bool = False):
        """
        Initialize Expert Executor
        
        Args:
            config: Configuration dictionary (loads from config.py if not provided)
            debug: Enable debug logging
        """
        self.debug = debug
        self.config = config or self._load_config()
        
        # Expert instances
        self.experts = {}
        self.available_experts = {}
        
        # Latest data storage (for Strategy Expert)
        self.latest_pattern_data = {}
        self.latest_price_action_data = {}
        self.latest_smc_data = {}
        self.latest_technical_data = {}
        
        # Initialize all experts
        self._init_experts()
        
        # Statistics
        self.run_count = 0
        self.total_signals_generated = 0
        
        if self.debug:
            print("[ExpertExecutor] Initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration from config.py"""
        try:
            from config import Config
            return {
                'min_confidence': getattr(Config, 'EXPERT_MIN_CONFIDENCE', 0.60),
                'min_experts_with_signal': getattr(Config, 'MIN_EXPERTS_WITH_SIGNAL', 3),
                'min_experts_to_agree': getattr(Config, 'MIN_EXPERTS_TO_AGREE', 3),
                'expert_weights': getattr(Config, 'EXPERT_INITIAL_WEIGHTS', {
                    'pattern_v3': 1.25,
                    'price_action': 1.20,
                    'smc': 1.30,
                    'technical': 1.15,
                    'strategy': 1.10
                })
            }
        except ImportError:
            return {
                'min_confidence': 0.60,
                'min_experts_with_signal': 3,
                'min_experts_to_agree': 3,
                'expert_weights': {
                    'pattern_v3': 1.25,
                    'price_action': 1.20,
                    'smc': 1.30,
                    'technical': 1.15,
                    'strategy': 1.10
                }
            }
    
    def _init_experts(self):
        """Initialize all 5 expert modules"""
        
        # 1. Pattern V4
        try:
            from pattern_v3.pattern_factory import PatternFactoryV4
            self.experts['pattern_v3'] = PatternFactoryV4(debug_mode=self.debug)
            self.available_experts['pattern_v3'] = True
            if self.debug:
                print("[ExpertExecutor] ✅ Pattern V4 initialized")
        except ImportError as e:
            print(f"[ExpertExecutor] ❌ Pattern V4 not available: {e}")
            self.available_experts['pattern_v3'] = False
        
        # 2. Price Action Expert
        try:
            from price_action_expert.expert_price_action import ExpertPriceAction
            self.experts['price_action'] = ExpertPriceAction()
            self.available_experts['price_action'] = True
            if self.debug:
                print("[ExpertExecutor] ✅ Price Action V3.5 initialized")
        except ImportError as e:
            print(f"[ExpertExecutor] ❌ Price Action not available: {e}")
            self.available_experts['price_action'] = False
        
        # 3. SMC Expert
        try:
            from smc_expert.smc_factory import SMCFactory
            self.experts['smc'] = SMCFactory()
            self.available_experts['smc'] = True
            if self.debug:
                print("[ExpertExecutor] ✅ SMC V3 initialized")
        except ImportError as e:
            print(f"[ExpertExecutor] ❌ SMC not available: {e}")
            self.available_experts['smc'] = False
        
        # 4. Technical Analyzer
        try:
            from technical_analyzer_expert.ta_factory import TechnicalAnalyzerFactory
            self.experts['technical'] = TechnicalAnalyzerFactory()
            self.available_experts['technical'] = True
            if self.debug:
                print("[ExpertExecutor] ✅ Technical Analyzer V2.0 initialized")
        except ImportError as e:
            print(f"[ExpertExecutor] ❌ Technical Analyzer not available: {e}")
            self.available_experts['technical'] = False
        
        # 5. Strategy Expert
        try:
            from strategy_expert.strategy_factory import StrategyExpert
            self.experts['strategy'] = StrategyExpert()
            self.available_experts['strategy'] = True
            if self.debug:
                print("[ExpertExecutor] ✅ Strategy Expert initialized")
        except ImportError as e:
            print(f"[ExpertExecutor] ❌ Strategy Expert not available: {e}")
            self.available_experts['strategy'] = False
    
    # ============================================================================
    # MAIN EXECUTION METHOD
    # ============================================================================
    
    def execute_for_coin(self, symbol: str, df: pd.DataFrame, timeframe: str = "5m",
                         htf_data: Dict[str, pd.DataFrame] = None,
                         market_regime: Dict = None,
                         structure_data: Dict = None,
                         sr_data: Dict = None,
                         liquidity_data: Dict = None,
                         indicators: Dict = None) -> List[ExpertSignal]:
        """
        Execute all 5 experts for a single coin
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            df: OHLCV DataFrame for primary timeframe
            timeframe: Timeframe string (e.g., "5m", "15m", "1h")
            htf_data: Dict of higher timeframe DataFrames
            market_regime: Market regime data
            structure_data: Market structure data
            sr_data: Support/Resistance data
            liquidity_data: Liquidity data
            indicators: Pre-calculated indicators
        
        Returns:
            List of ExpertSignal objects (one per expert, or skip signals)
        """
        self.run_count += 1
        signals = []
        
        if self.debug:
            print(f"\n[ExpertExecutor] {'='*60}")
            print(f"[ExpertExecutor] Analyzing {symbol} | {timeframe}")
            print(f"[ExpertExecutor] {'='*60}")
        
        # ===== STEP 1: RUN ALL 5 EXPERTS =====
        
        # Pattern V4
        pattern_signal = self._run_pattern_v4(df, symbol, timeframe, market_regime)
        signals.append(pattern_signal)
        if self.debug and pattern_signal.is_tradeable():
            print(f"[ExpertExecutor] Pattern: {pattern_signal.direction} ({pattern_signal.confidence:.1%})")
        
        # Price Action
        price_action_signal = self._run_price_action(df, symbol, timeframe, htf_data, market_regime)
        signals.append(price_action_signal)
        if self.debug and price_action_signal.is_tradeable():
            print(f"[ExpertExecutor] Price Action: {price_action_signal.direction} ({price_action_signal.confidence:.1%})")
        
        # SMC
        smc_signal = self._run_smc(df, symbol, timeframe, htf_data, market_regime)
        signals.append(smc_signal)
        if self.debug and smc_signal.is_tradeable():
            print(f"[ExpertExecutor] SMC: {smc_signal.direction} ({smc_signal.confidence:.1%})")
        
        # Technical Analyzer
        technical_signal = self._run_technical(df, symbol, timeframe, htf_data, market_regime, indicators)
        signals.append(technical_signal)
        if self.debug and technical_signal.is_tradeable():
            print(f"[ExpertExecutor] Technical: {technical_signal.direction} ({technical_signal.confidence:.1%})")
        
        # ===== STEP 2: CHECK WHICH EXPERTS NEED RE-RUN =====
        # Identify experts that didn't generate tradeable signals
        no_signal_experts = []
        for i, sig in enumerate(signals):
            if not sig.is_tradeable():
                expert_name = ['pattern_v3', 'price_action', 'smc', 'technical'][i]
                no_signal_experts.append(expert_name)
        
        # ===== STEP 3: RE-RUN EXPERTS WITH CONSENSUS DIRECTION =====
        if no_signal_experts:
            # Calculate consensus direction from experts that DID generate signals
            consensus_direction = self._calculate_consensus_direction(signals)
            
            if consensus_direction != "NEUTRAL" and consensus_direction != "HOLD":
                if self.debug:
                    print(f"[ExpertExecutor] Consensus direction: {consensus_direction}")
                    print(f"[ExpertExecutor] Re-running: {no_signal_experts}")
                
                # Re-run each expert that didn't generate a signal
                for expert_name in no_signal_experts:
                    direction_signal = self._run_expert_for_direction(
                        expert_name, df, symbol, timeframe, htf_data,
                        market_regime, structure_data, sr_data, 
                        liquidity_data, indicators, consensus_direction
                    )
                    if direction_signal:
                        # Replace the skip signal with direction-only signal
                        for i, s in enumerate(signals):
                            if s.expert_name == expert_name:
                                signals[i] = direction_signal
                                break
        
        # ===== STEP 4: RUN STRATEGY EXPERT (receives all data) =====
        # Build consolidated data from all other experts
        consolidated_data = self._build_consolidated_data(
            signals, market_regime, structure_data, sr_data, liquidity_data, indicators
        )
        
        strategy_signal = self._run_strategy_with_all_data(
            df, symbol, timeframe, market_regime, indicators, consolidated_data
        )
        signals.append(strategy_signal)
        
        if self.debug and strategy_signal.is_tradeable():
            print(f"[ExpertExecutor] Strategy: {strategy_signal.direction} ({strategy_signal.confidence:.1%})")
        
        # ===== STEP 5: SAVE RAW SIGNALS =====
        self._save_raw_signals(symbol, signals)
        
        # ===== STEP 6: UPDATE STATISTICS =====
        tradeable_count = sum(1 for s in signals if s.is_tradeable())
        self.total_signals_generated += tradeable_count
        
        if self.debug:
            print(f"[ExpertExecutor] Generated {tradeable_count}/5 tradeable signals")
            print(f"[ExpertExecutor] {'='*60}\n")
        
        return signals
    
    # ============================================================================
    # EXPERT RUNNERS (With Adapters)
    # ============================================================================
    
    def _run_pattern_v4(self, df: pd.DataFrame, symbol: str, timeframe: str,
                        market_regime: Dict) -> ExpertSignal:
        """Run Pattern V4 and convert to ExpertSignal using its adapter"""
        if not self.available_experts.get('pattern_v3', False):
            return create_skip_signal('pattern_v3', 'Module not available')
        
        try:
            # Run Pattern V4
            decisions = self.experts['pattern_v3'].process_symbol(
                df=df, symbol=symbol, timeframe=timeframe, 
                regime_data=market_regime
            )
            
            # Use Pattern's adapter
            from pattern_v3.adapter_pattern_v3 import pattern_v4_to_expert_signal
            
            if decisions and len(decisions) > 0:
                best = decisions[0]
                signal = pattern_v4_to_expert_signal(best, symbol)
                
                # Store pattern data for Strategy Expert
                self.latest_pattern_data = {
                    'patterns_detected': [d.get('pattern_name', '') for d in decisions[:3]],
                    'best_pattern': best.get('pattern_name', ''),
                    'pattern_confidence': best.get('confidence', 0.5),
                    'pattern_grade': best.get('grade', 'C'),
                    'pattern_reasons': best.get('reasons', [])
                }
                signal.metadata = self.latest_pattern_data
                signal.weight = self.config['expert_weights'].get('pattern_v3', 1.25)
                
                return signal
            else:
                return create_skip_signal('pattern_v3', 'No pattern detected')
                
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            return create_skip_signal('pattern_v3', f'Error: {str(e)[:50]}')
    
    def _run_price_action(self, df: pd.DataFrame, symbol: str, timeframe: str,
                          htf_data: Dict, market_regime: Dict) -> ExpertSignal:
        """Run Price Action Expert and convert using its adapter"""
        if not self.available_experts.get('price_action', False):
            return create_skip_signal('price_action', 'Module not available')
        
        try:
            # Run Price Action
            result = self.experts['price_action'].analyze(
                df=df, symbol=symbol, timeframe=timeframe,
                htf_data=htf_data, regime_data=market_regime
            )
            
            # Use Price Action's adapter
            from price_action_expert.adapter import price_action_to_expert_signal
            
            if result and result.action in ['STRONG_ENTRY', 'ENTER_NOW', 'WAIT_FOR_RETEST']:
                signal = price_action_to_expert_signal(result, symbol)
                
                # Store price action data for Strategy Expert
                self.latest_price_action_data = {
                    'pattern_name': getattr(result, 'pattern_name', ''),
                    'pattern_confidence': result.confidence,
                    'pattern_grade': result.grade,
                    'sequence_type': getattr(result, 'sequence_type', None),
                    'trap_detected': getattr(result, 'is_trap', False),
                    'key_moments': getattr(result, 'key_moments', [])
                }
                signal.metadata = self.latest_price_action_data
                signal.weight = self.config['expert_weights'].get('price_action', 1.20)
                
                return signal
            else:
                return create_skip_signal('price_action', 'No tradeable pattern')
                
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            return create_skip_signal('price_action', f'Error: {str(e)[:50]}')
    
    def _run_smc(self, df: pd.DataFrame, symbol: str, timeframe: str,
                 htf_data: Dict, market_regime: Dict) -> ExpertSignal:
        """Run SMC Expert and convert using its adapter"""
        if not self.available_experts.get('smc', False):
            return create_skip_signal('smc', 'Module not available')
        
        try:
            # Run SMC
            htf_df = htf_data.get('1h') if htf_data else None
            result = self.experts['smc'].analyze(
                df=df, symbol=symbol, timeframe=timeframe, htf_data=htf_data
            )
            
            # Use SMC's adapter
            from smc_expert.adapter import smc_to_expert_signal
            
            if result and result.get('action') in ['STRONG_ENTRY', 'ENTER_NOW', 'WAIT_RETEST']:
                signal = smc_to_expert_signal(result, symbol)
                
                # Store SMC data for Strategy Expert
                self.latest_smc_data = {
                    'order_blocks': result.get('order_blocks', [])[:3],
                    'fvgs': result.get('fvgs', [])[:3],
                    'liquidity_sweeps': result.get('liquidity_sweeps', [])[:3],
                    'market_structure': result.get('market_structure', 'NEUTRAL'),
                    'amd_phase': result.get('amd_phase', 'UNKNOWN')
                }
                signal.metadata = self.latest_smc_data
                signal.weight = self.config['expert_weights'].get('smc', 1.30)
                
                return signal
            else:
                return create_skip_signal('smc', 'No SMC setup')
                
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            return create_skip_signal('smc', f'Error: {str(e)[:50]}')
    
    def _run_technical(self, df: pd.DataFrame, symbol: str, timeframe: str,
                       htf_data: Dict, market_regime: Dict, indicators: Dict) -> ExpertSignal:
        """Run Technical Analyzer and convert using its adapter"""
        if not self.available_experts.get('technical', False):
            return create_skip_signal('technical', 'Module not available')
        
        try:
            # Run Technical Analyzer
            result = self.experts['technical'].analyze(
                df=df, symbol=symbol, htf_data=htf_data
            )
            
            # Use Technical's adapter
            from technical_analyzer_expert.adapter import technical_to_expert_signal
            
            if result and result.action in ['STRONG_ENTRY', 'ENTER_NOW']:
                signal = technical_to_expert_signal(result)
                
                # Store technical data for Strategy Expert
                self.latest_technical_data = {
                    'regime': getattr(result, 'regime', 'UNKNOWN'),
                    'trend_strength': getattr(result, 'trend_strength', 0.5),
                    'divergences': getattr(result, 'divergence_count', 0),
                    'indicator_signals': getattr(result, 'indicator_count', 0)
                }
                signal.metadata = self.latest_technical_data
                signal.weight = self.config['expert_weights'].get('technical', 1.15)
                
                return signal
            else:
                return create_skip_signal('technical', 'No technical signal')
                
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            return create_skip_signal('technical', f'Error: {str(e)[:50]}')
    
    def _run_strategy_with_all_data(self, df: pd.DataFrame, symbol: str, timeframe: str,
                                     market_regime: Dict, indicators: Dict,
                                     consolidated_data: Dict) -> ExpertSignal:
        """Run Strategy Expert with all consolidated data"""
        if not self.available_experts.get('strategy', False):
            return create_skip_signal('strategy', 'Module not available')
        
        try:
            # Try analyze_with_all_data method first
            if hasattr(self.experts['strategy'], 'analyze_with_all_data'):
                result = self.experts['strategy'].analyze_with_all_data(
                    df=df, symbol=symbol, timeframe=timeframe,
                    market_regime=market_regime, indicators=indicators,
                    consolidated_data=consolidated_data
                )
            else:
                # Fallback to standard analyze
                result = self.experts['strategy'].analyze(
                    df=df, indicators=indicators or {},
                    market_regime=market_regime or {},
                    module_signals=consolidated_data.get('expert_signals', {})
                )
            
            # Use Strategy's adapter
            from strategy_expert.adapter import strategy_to_expert_signal
            
            if result and result.get('action') in ['STRONG_ENTRY', 'ENTRY', 'CAUTIOUS_ENTRY']:
                signal = strategy_to_expert_signal(result, symbol)
                signal.weight = self.config['expert_weights'].get('strategy', 1.10)
                return signal
            else:
                return create_skip_signal('strategy', 'No strategy consensus')
                
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            return create_skip_signal('strategy', f'Error: {str(e)[:50]}')
    
    def _run_expert_for_direction(self, expert_name: str, df: pd.DataFrame, 
                                   symbol: str, timeframe: str, htf_data: Dict,
                                   market_regime: Dict, structure_data: Dict,
                                   sr_data: Dict, liquidity_data: Dict,
                                   indicators: Dict, consensus_direction: str) -> Optional[ExpertSignal]:
        """
        Re-run an expert that didn't generate a signal
        Ask: "Given the consensus direction, what is your direction preference?"
        """
        try:
            # Try to call expert's direction preference method
            expert = self.experts.get(expert_name)
            if not expert:
                return None
            
            if hasattr(expert, 'get_direction_preference'):
                direction, confidence = expert.get_direction_preference(
                    df=df, symbol=symbol, consensus_direction=consensus_direction
                )
                return create_direction_only_signal(
                    expert_name=expert_name,
                    direction=direction if direction != 'NEUTRAL' else consensus_direction,
                    confidence=confidence,
                    reason=f"Direction only (agrees with consensus: {consensus_direction})"
                )
            
            # If no direction preference method, use consensus direction with low confidence
            return create_direction_only_signal(
                expert_name=expert_name,
                direction=consensus_direction,
                confidence=0.50,
                reason=f"No signal generated, following consensus: {consensus_direction}"
            )
            
        except Exception as e:
            if self.debug:
                print(f"[ExpertExecutor] Direction re-run failed for {expert_name}: {e}")
            return None
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _calculate_consensus_direction(self, signals: List[ExpertSignal]) -> str:
        """Calculate consensus direction from experts that generated signals"""
        buy_count = 0
        sell_count = 0
        
        for s in signals:
            if s.is_tradeable():
                if s.direction == 'BUY':
                    buy_count += 1
                elif s.direction == 'SELL':
                    sell_count += 1
        
        if buy_count > sell_count:
            return 'BUY'
        elif sell_count > buy_count:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _build_consolidated_data(self, signals: List[ExpertSignal],
                                  market_regime: Dict, structure_data: Dict,
                                  sr_data: Dict, liquidity_data: Dict,
                                  indicators: Dict) -> Dict[str, Any]:
        """Build consolidated data for Strategy Expert"""
        
        # Helper to count directions
        def count_direction(direction):
            return sum(1 for s in signals if s.is_tradeable() and s.direction == direction)
        
        # Helper to get average confidence
        def avg_confidence(direction):
            confs = [s.confidence for s in signals if s.is_tradeable() and s.direction == direction]
            return sum(confs) / len(confs) if confs else 0.0
        
        return {
            'expert_signals': {s.expert_name: s.to_dict() for s in signals},
            'expert_agreement': {
                'buy_count': count_direction('BUY'),
                'sell_count': count_direction('SELL'),
                'hold_count': sum(1 for s in signals if not s.is_tradeable())
            },
            'avg_confidence': {
                'buy': avg_confidence('BUY'),
                'sell': avg_confidence('SELL')
            },
            'pattern_data': self.latest_pattern_data,
            'price_action_data': self.latest_price_action_data,
            'smc_data': self.latest_smc_data,
            'technical_data': self.latest_technical_data,
            'market_regime': market_regime,
            'structure': structure_data,
            'support_resistance': sr_data,
            'liquidity': liquidity_data,
            'indicators': indicators,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_raw_signals(self, symbol: str, signals: List[ExpertSignal]):
        """Save raw signals to signals/raw/ directory"""
        try:
            # Create directory if needed
            raw_dir = os.path.join('signals', 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            
            # Save to file
            date_str = datetime.now().strftime("%Y-%m-%d")
            filepath = os.path.join(raw_dir, f"{date_str}.json")
            
            # Load existing data
            existing_data = {}
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            
            # Add new signals
            if symbol not in existing_data:
                existing_data[symbol] = []
            
            existing_data[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'signals': [s.to_dict() for s in signals]
            })
            
            # Keep only last 100 entries per symbol
            if len(existing_data[symbol]) > 100:
                existing_data[symbol] = existing_data[symbol][-100:]
            
            # Save
            with open(filepath, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
        except Exception as e:
            if self.debug:
                print(f"[ExpertExecutor] Error saving raw signals: {e}")
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def is_expert_available(self, expert_name: str) -> bool:
        """Check if a specific expert is available"""
        return self.available_experts.get(expert_name, False)
    
    def get_available_experts(self) -> List[str]:
        """Get list of available expert names"""
        return [name for name, avail in self.available_experts.items() if avail]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'run_count': self.run_count,
            'total_signals_generated': self.total_signals_generated,
            'available_experts': self.get_available_experts(),
            'avg_signals_per_run': self.total_signals_generated / max(1, self.run_count)
        }
    
    def reset_statistics(self):
        """Reset statistics"""
        self.run_count = 0
        self.total_signals_generated = 0