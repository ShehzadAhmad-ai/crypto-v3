# signal_summary.py - Phase 12: Daily Signal Summary Generator
"""
Generates daily summaries of validated signals (Phase 10)
Creates files in /signals/summary/ folder with:
- Entry zones (low/high)
- All 5 TP levels with percentages
- Expert consensus (which experts agreed)
- Story summary with expert narrative
- Key points from all phases
- Strategy agreement
- Timing accuracy
- Confidence breakdown
"""

import os
import csv
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path

from logger import log
from signal_model import Signal, SignalStatus, SignalStage, TakeProfitLevel


class SignalSummary:
    """
    Generates daily summaries of validated signals - Phase 12
    - Creates .txt file with formatted summary (full details)
    - Creates .csv file for easy import into spreadsheets (key metrics)
    - Creates .md file for markdown format (for documentation)
    - Shows entry zones, all TP levels, expert consensus, story
    """
    
    def __init__(self, base_path: str = 'signals'):
        self.base_path = Path(base_path)
        self.summary_path = self.base_path / 'summary'
        self._create_directory()
        
        # Grade color mapping for console
        self.grade_colors = {
            'A+': '🟢', 'A': '🟢', 'B+': '🟢', 'B': '🟢', 'B-': '🟡',
            'C+': '🟡', 'C': '🟠', 'C-': '🟠', 'D': '🔴', 'F': '⚫'
        }
        
        # Expert names for display
        self.expert_names = {
            'pattern_v3': '📐 Pattern',
            'price_action': '🕯️ Price Action',
            'smc': '💰 SMC',
            'technical': '📊 Technical',
            'strategy': '🎯 Strategy'
        }
    
    def _create_directory(self):
        """Create summary directory if it doesn't exist"""
        self.summary_path.mkdir(parents=True, exist_ok=True)
        log.debug(f"Summary directory: {self.summary_path}")
    
    def generate_daily_summary(self, signals: List[Signal], summary_date: Optional[date] = None) -> Dict[str, str]:
        """
        Generate daily summary files for validated signals (Phase 10)
        
        Args:
            signals: List of validated signals (status = FINAL)
            summary_date: Date for summary (defaults to today)
        
        Returns:
            Dict with paths to generated files
        """
        if summary_date is None:
            summary_date = date.today()
        
        # Filter only validated signals (Phase 10)
        validated_signals = [s for s in signals if getattr(s, 'status', None) == SignalStatus.FINAL]
        if not validated_signals:
            log.info(f"No validated signals to summarize for {summary_date}")
            return {}
        
        # Generate files
        txt_path = self._generate_txt_summary(validated_signals, summary_date)
        csv_path = self._generate_csv_summary(validated_signals, summary_date)
        md_path = self._generate_markdown_summary(validated_signals, summary_date)
        
        log.info(f"Generated signal summaries for {summary_date}:")
        log.info(f"   TXT: {txt_path}")
        log.info(f"   CSV: {csv_path}")
        log.info(f"   MD: {md_path}")
        
        return {
            'txt': str(txt_path),
            'csv': str(csv_path),
            'md': str(md_path),
            'count': len(validated_signals)
        }
    
    def _generate_txt_summary(self, signals: List[Signal], summary_date: date) -> Path:
        """Generate human-readable text summary - Enhanced with expert consensus"""
        filepath = self.summary_path / f"{summary_date.isoformat()}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 120 + "\n")
            f.write(f"                    SIGNAL SUMMARY - {summary_date.isoformat()}\n")
            f.write("=" * 120 + "\n\n")
            f.write(f"Total Validated Signals: {len(signals)}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 120 + "\n\n")
            
            # Group by direction
            buys = [s for s in signals if s.direction == 'BUY']
            sells = [s for s in signals if s.direction == 'SELL']
            
            if buys:
                f.write(f"🟢 BUY SIGNALS ({len(buys)}):\n")
                f.write("=" * 120 + "\n")
                for i, s in enumerate(buys, 1):
                    self._write_signal_txt(f, s, i)
                f.write("\n")
            
            if sells:
                f.write(f"🔴 SELL SIGNALS ({len(sells)}):\n")
                f.write("=" * 120 + "\n")
                for i, s in enumerate(sells, 1):
                    self._write_signal_txt(f, s, i)
                f.write("\n")
            
            # Summary statistics
            self._write_summary_stats(f, signals)
        
        return filepath
    
    def _write_signal_txt(self, f, signal: Signal, index: int):
        """Write a single signal to text file - Enhanced with expert consensus"""
        # Header line
        grade_icon = self.grade_colors.get(signal.signal_grade, '⚪')
        f.write(f"\n{'─' * 120}\n")
        f.write(f"{index:2d}. {grade_icon} {signal.symbol:12s} | {signal.direction} | Grade: {signal.signal_grade}\n")
        f.write(f"{'─' * 120}\n")
        
        # ===== EXPERT CONSENSUS (Phase 4) =====
        if hasattr(signal, 'metadata') and signal.metadata:
            expert_details = signal.metadata.get('expert_details', {})
            if expert_details:
                f.write(f"\n  🤝 EXPERT CONSENSUS (Phase 4):\n")
                agreeing = []
                disagreeing = []
                for name, details in expert_details.items():
                    display_name = self.expert_names.get(name, name)
                    if details.get('agreed', False):
                        agreeing.append(f"{display_name}({details.get('confidence', 0):.0%})")
                    else:
                        disagreeing.append(display_name)
                
                if agreeing:
                    f.write(f"     ✅ Agreeing: {', '.join(agreeing)}\n")
                if disagreeing:
                    f.write(f"     ❌ Disagreeing: {', '.join(disagreeing)}\n")
                f.write(f"     📊 Consensus Confidence: {signal.consensus_confidence:.1%}\n")
        
        # ===== ENTRY ZONE (Phase 8) =====
        f.write(f"\n  📍 ENTRY ZONE (Phase 8):\n")
        f.write(f"     Zone: {signal.entry_zone_low:.8f} - {signal.entry_zone_high:.8f}\n")
        f.write(f"     Type: {signal.entry_type.value.upper()}\n")
        if hasattr(signal, 'expected_minutes_to_entry') and signal.expected_minutes_to_entry > 0:
            f.write(f"     Expected: ~{signal.expected_minutes_to_entry} minutes\n")
        
        # ===== TAKE PROFIT LEVELS (Phase 4) =====
        if hasattr(signal, 'take_profit_levels') and signal.take_profit_levels:
            f.write(f"\n  🎯 TAKE PROFIT LEVELS:\n")
            for i, tp in enumerate(signal.take_profit_levels[:5], 1):
                f.write(f"     TP{i}: {tp.price:.8f} ({tp.percentage:.0%}) - {tp.description}\n")
        elif signal.take_profit:
            f.write(f"\n  🎯 TAKE PROFIT: {signal.take_profit:.8f}\n")
        
        # ===== STOP LOSS =====
        f.write(f"\n  🛑 STOP LOSS: {signal.stop_loss:.8f}\n")
        f.write(f"  📊 RISK/REWARD: {signal.risk_reward_ratio:.2f}\n")
        
        # ===== CONFIDENCE (Phase 9) =====
        f.write(f"\n  📈 CONFIDENCE (Phase 9):\n")
        f.write(f"     Final Probability: {signal.probability:.1%}\n")
        if hasattr(signal, 'confidence_level') and signal.confidence_level:
            f.write(f"     Confidence Level: {signal.confidence_level:.1%}\n")
        if hasattr(signal, 'edge_persistence') and signal.edge_persistence:
            f.write(f"     Edge Persistence: {signal.edge_persistence:.1%}\n")
        
        # ===== PHASE SCORES =====
        f.write(f"\n  📊 PHASE SCORES:\n")
        f.write(f"     MTF (Phase 5):      {signal.mtf_score:.1%}\n")
        f.write(f"     Smart Money (Phase 6): {signal.smart_money_score:.1%}\n")
        if hasattr(signal, 'light_confirm_score') and signal.light_confirm_score:
            f.write(f"     Light Confirm (Phase 7): {signal.light_confirm_score:.1%}\n")
        
        # ===== MARKET CONTEXT =====
        f.write(f"\n  🌍 MARKET CONTEXT:\n")
        f.write(f"     Regime: {signal.market_regime}\n")
        f.write(f"     Volatility: {signal.volatility_state}\n")
        f.write(f"     HTF Aligned: {'✅' if getattr(signal, 'htf_aligned', False) else '❌'}\n")
        
        # ===== STORY AND KEY POINTS =====
        if hasattr(signal, 'metadata') and signal.metadata:
            story = signal.metadata.get('story', '')
            if story:
                f.write(f"\n  📖 STORY:\n")
                f.write(f"     {story[:200]}...\n")
            
            key_points = signal.metadata.get('key_points', [])
            if key_points:
                f.write(f"\n  🔑 KEY POINTS:\n")
                for point in key_points[:5]:
                    f.write(f"     • {point}\n")
        
        # ===== WARNINGS =====
        if signal.warning_flags:
            f.write(f"\n  ⚠️ WARNINGS:\n")
            for warning in signal.warning_flags[:3]:
                f.write(f"     • {warning}\n")
        
        # ===== POSITION SIZING (Phase 8) =====
        if hasattr(signal, 'position_size') and signal.position_size > 0:
            f.write(f"\n  💰 POSITION SIZING:\n")
            f.write(f"     Size: {signal.position_size:.4f} units\n")
            f.write(f"     Value: ${signal.position_value:.2f}\n")
            f.write(f"     Risk: ${signal.risk_amount:.2f}\n")
        
        f.write("\n")
    
    def _write_summary_stats(self, f, signals: List[Signal]):
        """Write summary statistics - Enhanced"""
        f.write("\n" + "=" * 120 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 120 + "\n\n")
        
        # Basic stats
        buys = [s for s in signals if s.direction == 'BUY']
        sells = [s for s in signals if s.direction == 'SELL']
        
        f.write(f"Total Signals: {len(signals)}\n")
        f.write(f"  BUY:  {len(buys)}\n")
        f.write(f"  SELL: {len(sells)}\n\n")
        
        # Grade distribution
        f.write("Grade Distribution:\n")
        grades = {}
        for s in signals:
            grades[s.signal_grade] = grades.get(s.signal_grade, 0) + 1
        for grade in ['A+', 'A', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']:
            if grade in grades:
                f.write(f"  {grade}: {grades[grade]} signals\n")
        
        f.write("\n")
        
        # Probability stats
        avg_prob = sum(s.probability for s in signals) / len(signals)
        max_prob = max(s.probability for s in signals)
        min_prob = min(s.probability for s in signals)
        f.write(f"Probability:\n")
        f.write(f"  Average: {avg_prob:.1%}\n")
        f.write(f"  Highest: {max_prob:.1%}\n")
        f.write(f"  Lowest:  {min_prob:.1%}\n\n")
        
        # Risk/Reward stats
        avg_rr = sum(s.risk_reward_ratio for s in signals) / len(signals)
        max_rr = max(s.risk_reward_ratio for s in signals)
        f.write(f"Risk/Reward:\n")
        f.write(f"  Average: {avg_rr:.2f}\n")
        f.write(f"  Highest: {max_rr:.2f}\n\n")
        
        # Entry time stats
        entry_times = [s.expected_minutes_to_entry for s in signals if hasattr(s, 'expected_minutes_to_entry') and s.expected_minutes_to_entry > 0]
        if entry_times:
            avg_entry = sum(entry_times) / len(entry_times)
            f.write(f"Entry Time:\n")
            f.write(f"  Average: {avg_entry:.0f} minutes\n")
            f.write(f"  Range: {min(entry_times)} - {max(entry_times)} minutes\n\n")
        
        # Expert consensus stats
        expert_agreement = []
        for s in signals:
            if hasattr(s, 'metadata') and s.metadata:
                expert_details = s.metadata.get('expert_details', {})
                if expert_details:
                    agreeing = sum(1 for d in expert_details.values() if d.get('agreed', False))
                    expert_agreement.append(agreeing / len(expert_details))
        
        if expert_agreement:
            avg_agreement = sum(expert_agreement) / len(expert_agreement)
            f.write(f"Expert Consensus:\n")
            f.write(f"  Average Agreement: {avg_agreement:.1%}\n\n")
        
        # Market regime breakdown
        f.write("Market Regime Breakdown:\n")
        regimes = {}
        for s in signals:
            regimes[s.market_regime] = regimes.get(s.market_regime, 0) + 1
        for regime, count in sorted(regimes.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {regime}: {count} signals\n")
        
        f.write("=" * 120 + "\n")
    
    def _generate_csv_summary(self, signals: List[Signal], summary_date: date) -> Path:
        """Generate CSV summary - Enhanced with expert consensus"""
        filepath = self.summary_path / f"{summary_date.isoformat()}.csv"
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header - Enhanced with expert consensus
            writer.writerow([
                'Symbol', 'Direction', 'Grade', 'Probability', 'Confidence',
                'Entry Zone Low', 'Entry Zone High', 'Entry Type', 'Entry Time (min)',
                'Stop Loss', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'Risk/Reward',
                'Expert Consensus', 'Agreeing Experts', 'Disagreeing Experts',
                'MTF Score', 'Smart Money Score', 'Light Confirm Score',
                'Market Regime', 'Volatility', 'HTF Aligned',
                'Story Summary', 'Warnings'
            ])
            
            # Data rows
            for s in signals:
                # Get expert consensus
                agreeing_experts = ''
                disagreeing_experts = ''
                expert_consensus = 'NO'
                
                if hasattr(s, 'metadata') and s.metadata:
                    expert_details = s.metadata.get('expert_details', {})
                    if expert_details:
                        agreeing = [name for name, d in expert_details.items() if d.get('agreed', False)]
                        disagreeing = [name for name, d in expert_details.items() if not d.get('agreed', False)]
                        agreeing_experts = ', '.join(agreeing)
                        disagreeing_experts = ', '.join(disagreeing)
                        expert_consensus = 'YES' if len(agreeing) == 5 else 'PARTIAL'
                
                # Get TP levels
                tp1 = tp2 = tp3 = tp4 = tp5 = ''
                if hasattr(s, 'take_profit_levels') and s.take_profit_levels:
                    for i, tp in enumerate(s.take_profit_levels[:5]):
                        if i == 0:
                            tp1 = f"{tp.price:.8f} ({tp.percentage:.0%})"
                        elif i == 1:
                            tp2 = f"{tp.price:.8f} ({tp.percentage:.0%})"
                        elif i == 2:
                            tp3 = f"{tp.price:.8f} ({tp.percentage:.0%})"
                        elif i == 3:
                            tp4 = f"{tp.price:.8f} ({tp.percentage:.0%})"
                        elif i == 4:
                            tp5 = f"{tp.price:.8f} ({tp.percentage:.0%})"
                elif s.take_profit:
                    tp1 = f"{s.take_profit:.8f} (100%)"
                
                story_summary = ''
                if hasattr(s, 'metadata') and s.metadata:
                    story_summary = s.metadata.get('story_summary', '')[:100]
                
                warnings_str = '; '.join(s.warning_flags[:2]) if s.warning_flags else ''
                
                writer.writerow([
                    s.symbol,
                    s.direction,
                    s.signal_grade,
                    f"{s.probability:.1%}",
                    f"{getattr(s, 'confidence_level', 0):.1%}",
                    f"{s.entry_zone_low:.8f}",
                    f"{s.entry_zone_high:.8f}",
                    s.entry_type.value.upper() if s.entry_type else 'LIMIT',
                    getattr(s, 'expected_minutes_to_entry', 0),
                    f"{s.stop_loss:.8f}",
                    tp1, tp2, tp3, tp4, tp5,
                    f"{s.risk_reward_ratio:.2f}",
                    expert_consensus,
                    agreeing_experts,
                    disagreeing_experts,
                    f"{s.mtf_score:.1%}",
                    f"{s.smart_money_score:.1%}",
                    f"{getattr(s, 'light_confirm_score', 0):.1%}",
                    s.market_regime,
                    s.volatility_state,
                    'Yes' if getattr(s, 'htf_aligned', False) else 'No',
                    story_summary,
                    warnings_str
                ])
        
        return filepath
    
    def _generate_markdown_summary(self, signals: List[Signal], summary_date: date) -> Path:
        """Generate markdown summary - Enhanced with expert consensus"""
        filepath = self.summary_path / f"{summary_date.isoformat()}.md"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Signal Summary - {summary_date.isoformat()}\n\n")
            f.write(f"**Total Signals:** {len(signals)}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by direction
            buys = [s for s in signals if s.direction == 'BUY']
            sells = [s for s in signals if s.direction == 'SELL']
            
            if buys:
                f.write("## 🟢 BUY Signals\n\n")
                f.write("| # | Symbol | Grade | Prob | Expert Agree | Entry Zone | TP1 | SL | RR |\n")
                f.write("|---|--------|-------|------|--------------|------------|-----|----|----|\n")
                for i, s in enumerate(buys, 1):
                    # Get expert agreement count
                    agree_count = 0
                    if hasattr(s, 'metadata') and s.metadata:
                        expert_details = s.metadata.get('expert_details', {})
                        agree_count = sum(1 for d in expert_details.values() if d.get('agreed', False))
                    
                    tp1 = s.take_profit_levels[0].price if hasattr(s, 'take_profit_levels') and s.take_profit_levels else s.take_profit
                    f.write(f"| {i} | {s.symbol} | {s.signal_grade} | {s.probability:.1%} | {agree_count}/5 | {s.entry_zone_low:.6f}-{s.entry_zone_high:.6f} | {tp1:.6f} | {s.stop_loss:.6f} | {s.risk_reward_ratio:.2f} |\n")
                f.write("\n")
            
            if sells:
                f.write("## 🔴 SELL Signals\n\n")
                f.write("| # | Symbol | Grade | Prob | Expert Agree | Entry Zone | TP1 | SL | RR |\n")
                f.write("|---|--------|-------|------|--------------|------------|-----|----|----|\n")
                for i, s in enumerate(sells, 1):
                    agree_count = 0
                    if hasattr(s, 'metadata') and s.metadata:
                        expert_details = s.metadata.get('expert_details', {})
                        agree_count = sum(1 for d in expert_details.values() if d.get('agreed', False))
                    
                    tp1 = s.take_profit_levels[0].price if hasattr(s, 'take_profit_levels') and s.take_profit_levels else s.take_profit
                    f.write(f"| {i} | {s.symbol} | {s.signal_grade} | {s.probability:.1%} | {agree_count}/5 | {s.entry_zone_low:.6f}-{s.entry_zone_high:.6f} | {tp1:.6f} | {s.stop_loss:.6f} | {s.risk_reward_ratio:.2f} |\n")
                f.write("\n")
            
            # Detailed signals
            f.write("## Detailed Signal Analysis\n\n")
            for i, s in enumerate(signals, 1):
                grade_icon = self.grade_colors.get(s.signal_grade, '⚪')
                f.write(f"### {i}. {grade_icon} {s.symbol} - {s.direction} ({s.signal_grade})\n\n")
                
                # Expert consensus
                if hasattr(s, 'metadata') and s.metadata:
                    expert_details = s.metadata.get('expert_details', {})
                    if expert_details:
                        agreeing = [name for name, d in expert_details.items() if d.get('agreed', False)]
                        disagreeing = [name for name, d in expert_details.items() if not d.get('agreed', False)]
                        
                        f.write(f"**Expert Consensus:**\n")
                        if agreeing:
                            f.write(f"- ✅ Agreeing: {', '.join(agreeing)}\n")
                        if disagreeing:
                            f.write(f"- ❌ Disagreeing: {', '.join(disagreeing)}\n")
                        f.write(f"- 📊 Consensus Confidence: {s.consensus_confidence:.1%}\n\n")
                
                # Entry and Risk
                f.write(f"**Entry Zone:** {s.entry_zone_low:.6f} - {s.entry_zone_high:.6f} ({s.entry_type.value.upper()})\n")
                f.write(f"**Stop Loss:** {s.stop_loss:.6f}\n")
                
                # TP Levels
                if hasattr(s, 'take_profit_levels') and s.take_profit_levels:
                    f.write(f"**Take Profit Levels:**\n")
                    for j, tp in enumerate(s.take_profit_levels[:5], 1):
                        f.write(f"- TP{j}: {tp.price:.6f} ({tp.percentage:.0%}) - {tp.description}\n")
                elif s.take_profit:
                    f.write(f"**Take Profit:** {s.take_profit:.6f}\n")
                
                f.write(f"**Risk/Reward:** {s.risk_reward_ratio:.2f}\n\n")
                
                # Confidence
                f.write(f"**Confidence:** {s.probability:.1%} (Grade {s.signal_grade})\n")
                f.write(f"**MTF Score:** {s.mtf_score:.1%}\n")
                f.write(f"**Smart Money Score:** {s.smart_money_score:.1%}\n\n")
                
                # Story
                if hasattr(s, 'metadata') and s.metadata:
                    story = s.metadata.get('story_summary', '')
                    if story:
                        f.write(f"**Story:** {story}\n\n")
                    
                    key_points = s.metadata.get('key_points', [])
                    if key_points:
                        f.write(f"**Key Points:**\n")
                        for point in key_points[:5]:
                            f.write(f"- {point}\n")
                        f.write("\n")
                
                # Warnings
                if s.warning_flags:
                    f.write(f"**Warnings:**\n")
                    for warning in s.warning_flags[:3]:
                        f.write(f"- ⚠️ {warning}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        return filepath
    
    def _grade_value(self, grade: str) -> float:
        """Convert grade to numeric value for sorting"""
        grade_map = {
            'A+': 6.0, 'A': 5.0, 'B+': 4.5, 'B': 4.0, 'B-': 3.5,
            'C+': 3.0, 'C': 2.5, 'C-': 2.0, 'D': 1.0, 'F': 0.0
        }
        return grade_map.get(grade, 0)
    
    def get_latest_summary(self) -> Optional[Dict]:
        """Get the most recent summary file"""
        txt_files = sorted(self.summary_path.glob("*.txt"), reverse=True)
        if not txt_files:
            return None
        
        latest = txt_files[0]
        date_str = latest.stem
        
        return {
            'date': date_str,
            'txt': str(latest),
            'csv': str(self.summary_path / f"{date_str}.csv"),
            'md': str(self.summary_path / f"{date_str}.md")
        }
    
    def print_daily_summary(self, signals: List[Signal]):
        """Print a quick summary to console - Enhanced with expert consensus"""
        if not signals:
            return
        
        print("\n" + "=" * 120)
        print(f"                    DAILY SIGNAL SUMMARY - {date.today().isoformat()}")
        print("=" * 120)
        
        # Group by direction
        buys = [s for s in signals if s.direction == 'BUY']
        sells = [s for s in signals if s.direction == 'SELL']
        
        if buys:
            print(f"\n 🟢 BUY SIGNALS ({len(buys)}):")
            print("-" * 120)
            for s in buys:
                grade_icon = self.grade_colors.get(s.signal_grade, '⚪')
                entry_zone = f"{s.entry_zone_low:.6f}-{s.entry_zone_high:.6f}"
                
                # Get expert agreement
                agree_count = 0
                if hasattr(s, 'metadata') and s.metadata:
                    expert_details = s.metadata.get('expert_details', {})
                    agree_count = sum(1 for d in expert_details.values() if d.get('agreed', False))
                
                tp1 = s.take_profit_levels[0].price if hasattr(s, 'take_profit_levels') and s.take_profit_levels else s.take_profit
                print(f"  {grade_icon} {s.symbol:10s} [{s.signal_grade}] {s.probability:.1%} Exp:{agree_count}/5  Zone:{entry_zone:18s}  TP1:{tp1:.6f}  SL:{s.stop_loss:.6f}  RR:{s.risk_reward_ratio:.2f}")
        
        if sells:
            print(f"\n 🔴 SELL SIGNALS ({len(sells)}):")
            print("-" * 120)
            for s in sells:
                grade_icon = self.grade_colors.get(s.signal_grade, '⚪')
                entry_zone = f"{s.entry_zone_low:.6f}-{s.entry_zone_high:.6f}"
                
                agree_count = 0
                if hasattr(s, 'metadata') and s.metadata:
                    expert_details = s.metadata.get('expert_details', {})
                    agree_count = sum(1 for d in expert_details.values() if d.get('agreed', False))
                
                tp1 = s.take_profit_levels[0].price if hasattr(s, 'take_profit_levels') and s.take_profit_levels else s.take_profit
                print(f"  {grade_icon} {s.symbol:10s} [{s.signal_grade}] {s.probability:.1%} Exp:{agree_count}/5  Zone:{entry_zone:18s}  TP1:{tp1:.6f}  SL:{s.stop_loss:.6f}  RR:{s.risk_reward_ratio:.2f}")
        
        # Summary line
        print("\n" + "-" * 120)
        print(f"  Total: {len(signals)} signals | BUY: {len(buys)} | SELL: {len(sells)}")
        print(f"  Avg Probability: {sum(s.probability for s in signals)/len(signals):.1%}")
        print(f"  Avg RR: {sum(s.risk_reward_ratio for s in signals)/len(signals):.2f}")
        print("=" * 120)
        print(f"  Summary saved to: /signals/summary/{date.today().isoformat()}.txt")
        print("=" * 120)