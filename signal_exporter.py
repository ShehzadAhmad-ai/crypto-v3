# signal_exporter.py - Phase 12: Clean Signal Exporter
"""
Signal Exporter - Phase 12
Exports validated signals with clean, readable data:
- Keeps story, key points, summary for manual reading
- Includes expert consensus from Phase 4
- Includes all 5 TP levels
- Includes entry zone and timing data
- Removes large DataFrames and arrays
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from logger import log
from signal_model import Signal, SignalStatus, SignalStage, TakeProfitLevel


class SignalExporter:
    """
    Exports signals with clean, readable data - Phase 12
    - Keeps story, key points, summary for manual reading
    - Includes expert consensus (Phase 4)
    - Includes all 5 TP levels
    - Removes large DataFrames and arrays
    """
    
    def __init__(self):
        self.max_warnings = 5
        self.max_key_points = 8
        self.max_story_length = 500
        
        # Expert names for display
        self.expert_names = {
            'pattern_v3': 'Pattern V3',
            'price_action': 'Price Action',
            'smc': 'SMC',
            'technical': 'Technical',
            'strategy': 'Strategy'
        }
    
    def export_signal(self, signal: Signal, stage: SignalStage) -> Dict[str, Any]:
        """
        Export signal with clean data - Enhanced with Phase 4 expert consensus
        
        Args:
            signal: Signal object
            stage: RAW, CONFIRMED, or FINAL
        
        Returns:
            Clean dict ready for JSON
        """
        # Get metadata safely
        metadata = getattr(signal, 'metadata', {}) or {}
        
        # ===== EXPERT CONSENSUS (Phase 4) =====
        expert_details = metadata.get('expert_details', {})
        agreeing_experts = []
        disagreeing_experts = []
        
        for name, details in expert_details.items():
            display_name = self.expert_names.get(name, name)
            if details.get('agreed', False):
                agreeing_experts.append({
                    'name': display_name,
                    'confidence': details.get('confidence', 0),
                    'grade': details.get('grade', 'C'),
                    'weight': details.get('weight', 1.0)
                })
            else:
                disagreeing_experts.append({
                    'name': display_name,
                    'confidence': details.get('confidence', 0),
                    'grade': details.get('grade', 'C')
                })
        
        # ===== TAKE PROFIT LEVELS (Phase 4) =====
        tp_levels = []
        if hasattr(signal, 'take_profit_levels') and signal.take_profit_levels:
            for i, tp in enumerate(signal.take_profit_levels[:5], 1):
                tp_levels.append({
                    'level': i,
                    'price': tp.price,
                    'percentage': tp.percentage,
                    'description': tp.description
                })
        elif signal.take_profit:
            tp_levels.append({
                'level': 1,
                'price': signal.take_profit,
                'percentage': 1.0,
                'description': 'Primary target'
            })
        
        clean_data = {
            # ===== IDENTIFIERS =====
            'signal_id': metadata.get('signal_id', f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"),
            'symbol': signal.symbol,
            'timeframe': signal.timeframe,
            'direction': signal.direction,
            'timestamp': signal.timestamp.isoformat(),
            'stage': stage.value if stage else 'unknown',
            
            # ===== PHASE 4: EXPERT CONSENSUS =====
            'expert_consensus': {
                'reached': getattr(signal, 'expert_consensus_reached', False),
                'confidence': getattr(signal, 'consensus_confidence', 0),
                'agreeing_experts': agreeing_experts,
                'disagreeing_experts': disagreeing_experts,
                'agreement_ratio': len(agreeing_experts) / 5 if expert_details else 0
            },
            
            # ===== PHASE 5: MTF CONFIRMATION =====
            'mtf': {
                'score': signal.mtf_score,
                'alignment': getattr(signal, 'mtf_alignment', 0.5),
                'bullish_htf': metadata.get('mtf', {}).get('bullish_htf_count', 0),
                'bearish_htf': metadata.get('mtf', {}).get('bearish_htf_count', 0)
            },
            
            # ===== PHASE 6: SMART MONEY =====
            'smart_money': {
                'score': signal.smart_money_score,
                'bias': getattr(signal, 'smart_money_bias', 0),
                'has_liquidity_sweep': metadata.get('has_down_sweep', False) or metadata.get('has_up_sweep', False),
                'stop_hunt_prob': metadata.get('stop_hunt_prob', 0)
            },
            
            # ===== PHASE 7: LIGHT CONFIRMATIONS =====
            'light_confirm': {
                'score': getattr(signal, 'light_confirm_score', 0),
                'cross_asset': getattr(signal, 'cross_asset_score', 0),
                'funding_oi': getattr(signal, 'funding_oi_score', 0),
                'sentiment': getattr(signal, 'sentiment_score', 0)
            },
            
            # ===== PHASE 8: RISK MANAGEMENT =====
            'risk': {
                'entry_zone_low': signal.entry_zone_low,
                'entry_zone_high': signal.entry_zone_high,
                'entry_type': signal.entry_type.value if signal.entry_type else 'LIMIT',
                'stop_loss': signal.stop_loss,
                'take_profit_levels': tp_levels,
                'risk_reward': signal.risk_reward_ratio,
                'position_size': getattr(signal, 'position_size', 0),
                'position_value': getattr(signal, 'position_value', 0),
                'risk_amount': getattr(signal, 'risk_amount', 0)
            },
            
            # ===== PHASE 9: FINAL SCORING =====
            'final_scoring': {
                'probability': signal.probability,
                'confidence_level': getattr(signal, 'confidence_level', 0),
                'grade': signal.signal_grade,
                'edge_persistence': getattr(signal, 'edge_persistence', 0.5),
                'final_score': getattr(signal, 'final_score', 0)
            },
            
            # ===== PHASE 10: SIGNAL VALIDATION =====
            'validation': {
                'status': signal.status.value if signal.status else 'UNKNOWN',
                'warning_flags': signal.warning_flags[:self.max_warnings] if signal.warning_flags else []
            },
            
            # ===== PHASE 11: TIMING =====
            'timing': {
                'expected_minutes_to_entry': getattr(signal, 'expected_minutes_to_entry', 0),
                'expected_candles': getattr(signal, 'expected_candles_to_entry', 0)
            },
            
            # ===== MARKET CONTEXT =====
            'market_context': {
                'regime': signal.market_regime,
                'volatility_state': signal.volatility_state,
                'htf_aligned': getattr(signal, 'htf_aligned', False),
                'trend_alignment': getattr(signal, 'trend_alignment', 'NEUTRAL')
            },
            
            # ===== STRATEGY INFO =====
            'strategy': {
                'confirmed': signal.strategy_confirmed,
                'agreed_count': metadata.get('strategies_agreed', 0),
                'total_count': metadata.get('strategies_total', 0)
            },
            
            # ===== HUMAN-READABLE CONTENT =====
            'story': self._get_story(signal, metadata),
            'summary': self._get_summary(signal, metadata),
            'key_points': self._get_key_points(metadata),
            'reasons': signal.confirmation_reasons[:5] if signal.confirmation_reasons else [],
            
            # ===== QUICK METRICS =====
            'metrics': {
                'has_entry_zone': signal.entry_zone_low is not None and signal.entry_zone_low > 0,
                'has_stop_loss': signal.stop_loss is not None and signal.stop_loss > 0,
                'has_take_profit': bool(tp_levels),
                'rr_valid': signal.risk_reward_ratio >= 1.5 if signal.risk_reward_ratio else False,
                'expert_consensus_reached': getattr(signal, 'expert_consensus_reached', False),
                'mtf_aligned': signal.mtf_score >= 0.7 if signal.mtf_score else False,
                'smart_money_aligned': signal.smart_money_score >= 0.6 if signal.smart_money_score else False
            }
        }
        
        return clean_data
    
    def _get_story(self, signal: Signal, metadata: Dict) -> str:
        """Get human-readable story with expert consensus"""
        # Try to get from metadata first
        story = metadata.get('story', '')
        if story:
            return story[:self.max_story_length]
        
        # Build story from expert consensus
        expert_details = metadata.get('expert_details', {})
        if expert_details:
            agreeing = [name for name, d in expert_details.items() if d.get('agreed', False)]
            disagreeing = [name for name, d in expert_details.items() if not d.get('agreed', False)]
            
            if len(agreeing) == 5:
                story = f"Strong consensus: All 5 experts agree on {signal.direction}. "
            elif len(agreeing) >= 3:
                story = f"Majority consensus: {len(agreeing)}/5 experts agree on {signal.direction}. "
            else:
                story = f"Weak consensus: Only {len(agreeing)}/5 experts agree. "
            
            story += f"MTF confirms with {signal.mtf_score:.0%} alignment. "
            story += f"Smart money {signal.smart_money_score:.0%} confidence. "
            story += f"Final probability {signal.probability:.0%} (Grade {signal.signal_grade}). "
            
            if signal.risk_reward_ratio > 0:
                story += f"Risk/Reward {signal.risk_reward_ratio:.1f}. "
            
            return story[:self.max_story_length]
        
        # Fallback: basic story
        story = f"{signal.direction} {signal.symbol} - "
        story += f"Technical: {signal.technical_score:.0%}, "
        story += f"MTF: {signal.mtf_score:.0%}, "
        story += f"Smart Money: {signal.smart_money_score:.0%}. "
        story += f"RR: {signal.risk_reward_ratio:.1f}"
        
        return story[:self.max_story_length]
    
    def _get_summary(self, signal: Signal, metadata: Dict) -> str:
        """Get one-line summary with grade and consensus"""
        # Try from metadata
        summary = metadata.get('story_summary', '')
        if summary:
            return summary[:150]
        
        # Build summary
        grade_icon = self._get_grade_icon(signal.signal_grade)
        
        expert_details = metadata.get('expert_details', {})
        if expert_details:
            agreeing = sum(1 for d in expert_details.values() if d.get('agreed', False))
            return f"{grade_icon} {signal.symbol} - {signal.direction} | Grade: {signal.signal_grade} | Prob: {signal.probability:.0%} | Experts: {agreeing}/5 | RR: {signal.risk_reward_ratio:.1f}"
        
        return f"{grade_icon} {signal.symbol} - {signal.direction} | Grade: {signal.signal_grade} | Prob: {signal.probability:.0%} | RR: {signal.risk_reward_ratio:.1f}"
    
    def _get_key_points(self, metadata: Dict) -> List[str]:
        """Get key points from metadata"""
        key_points = metadata.get('key_points', [])
        if key_points:
            return key_points[:self.max_key_points]
        
        # Return empty list if no key points
        return []
    
    def _get_grade_icon(self, grade: str) -> str:
        """Get emoji/icon for grade"""
        icons = {
            'A+': '🟢', 'A': '🟢', 'B+': '🟢', 'B': '🟢', 'B-': '🟡',
            'C+': '🟡', 'C': '🟠', 'C-': '🟠', 'D': '🔴', 'F': '⚫'
        }
        return icons.get(grade, '⚪')
    
    # ==================== SAVE METHODS ====================
    
    def save_signal(self, signal: Signal, stage: SignalStage, filepath: Path):
        """
        Save signal with clean data to file
        
        Args:
            signal: Signal object
            stage: RAW, CONFIRMED, or FINAL
            filepath: Path to save (e.g., signals/raw/2024-01-15.json)
        """
        clean_data = self.export_signal(signal, stage)
        
        # Load existing signals
        existing = []
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing = []
        
        # Check for duplicate (same symbol + direction + date)
        signal_key = f"{clean_data['symbol']}_{clean_data['direction']}_{clean_data['timestamp'][:10]}"
        existing = [s for s in existing if f"{s['symbol']}_{s['direction']}_{s['timestamp'][:10]}" != signal_key]
        
        # Add new signal
        existing.append(clean_data)
        
        # Sort by timestamp (newest first)
        existing.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, default=str)
        
        log.debug(f"Saved {stage.value} signal for {signal.symbol} to {filepath}")
    
    def save_signals_batch(self, signals: List[Signal], stage: SignalStage, filepath: Path):
        """
        Save multiple signals to file
        
        Args:
            signals: List of Signal objects
            stage: RAW, CONFIRMED, or FINAL
            filepath: Path to save
        """
        if not signals:
            return
        
        clean_data_list = []
        for signal in signals:
            clean_data_list.append(self.export_signal(signal, stage))
        
        # Load existing
        existing = []
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing = []
        
        # Merge and deduplicate
        all_signals = existing + clean_data_list
        seen_keys = set()
        unique_signals = []
        
        for s in all_signals:
            key = f"{s['symbol']}_{s['direction']}_{s['timestamp'][:10]}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_signals.append(s)
        
        # Sort by timestamp (newest first)
        unique_signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(unique_signals, f, indent=2, default=str)
        
        log.info(f"Saved {len(clean_data_list)} {stage.value} signals to {filepath}")
    
    # ==================== READ METHODS ====================
    
    def load_signals(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Load signals from file
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            List of signal dicts
        """
        if not filepath.exists():
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            log.error(f"Error loading signals from {filepath}: {e}")
            return []
    
    def load_signals_by_date(self, date_str: str, category: str = 'final') -> List[Dict[str, Any]]:
        """
        Load signals for a specific date
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            category: 'raw', 'confirmed', or 'final'
        
        Returns:
            List of signal dicts
        """
        filepath = Path(f"signals/{category}/{date_str}.json")
        return self.load_signals(filepath)
    
    def get_latest_signals(self, category: str = 'final', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest signals from category
        
        Args:
            category: 'raw', 'confirmed', or 'final'
            limit: Maximum number of signals to return
        
        Returns:
            List of latest signal dicts
        """
        import glob
        
        pattern = f"signals/{category}/*.json"
        files = sorted(glob.glob(pattern), reverse=True)
        
        if not files:
            return []
        
        all_signals = []
        for filepath in files[:5]:  # Check last 5 days
            signals = self.load_signals(Path(filepath))
            all_signals.extend(signals)
        
        # Sort by timestamp and limit
        all_signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return all_signals[:limit]
    
    def print_signal(self, signal_dict: Dict[str, Any]):
        """Print a signal in readable format - Enhanced"""
        print("\n" + "=" * 90)
        print(f"📊 {signal_dict.get('symbol')} - {signal_dict.get('direction')} [{signal_dict.get('final_scoring', {}).get('grade', 'N/A')}]")
        print("=" * 90)
        
        # Summary
        print(f"\n📝 {signal_dict.get('summary', 'No summary')}")
        
        # Expert consensus
        expert = signal_dict.get('expert_consensus', {})
        if expert.get('agreeing_experts'):
            agreeing_names = [e['name'] for e in expert['agreeing_experts']]
            print(f"\n🤝 Expert Consensus: {len(agreeing_names)}/5 experts agree ({', '.join(agreeing_names)})")
        
        # Key points
        key_points = signal_dict.get('key_points', [])
        if key_points:
            print("\n🔑 Key Points:")
            for point in key_points[:5]:
                print(f"   • {point}")
        
        # Trade parameters
        risk = signal_dict.get('risk', {})
        print(f"\n🎯 Entry Zone: {risk.get('entry_zone_low', 0):.6f} - {risk.get('entry_zone_high', 0):.6f}")
        
        # TP Levels
        tp_levels = risk.get('take_profit_levels', [])
        if tp_levels:
            print(f"\n🎯 Take Profit Levels:")
            for tp in tp_levels:
                print(f"   TP{tp.get('level')}: {tp.get('price', 0):.6f} ({tp.get('percentage', 0):.0%}) - {tp.get('description', '')}")
        
        print(f"🛑 Stop Loss: {risk.get('stop_loss', 0):.6f}")
        print(f"📊 Risk/Reward: {risk.get('risk_reward', 0):.2f}")
        
        # Scores
        final = signal_dict.get('final_scoring', {})
        print(f"\n📈 Probability: {final.get('probability', 0):.1%} (Grade {final.get('grade', 'N/A')})")
        print(f"   MTF: {signal_dict.get('mtf', {}).get('score', 0):.1%}")
        print(f"   Smart Money: {signal_dict.get('smart_money', {}).get('score', 0):.1%}")
        
        # Warnings
        validation = signal_dict.get('validation', {})
        warnings = validation.get('warning_flags', [])
        if warnings:
            print(f"\n⚠️ Warnings:")
            for w in warnings[:3]:
                print(f"   • {w}")
        
        print("=" * 90)