# performance_tracker.py - Phase 13: Complete Performance Tracking
"""
Performance Tracker - Phase 13

Records and tracks all trade outcomes from TradeOutcomeAnalyzer:
- Stores trade records with full details
- Updates expert weights based on performance
- Generates daily/weekly/monthly reports
- Manages active trade tracking
- Provides statistics for the entire system

This module works hand-in-hand with TradeOutcomeAnalyzer:
1. TradeOutcomeAnalyzer determines outcomes
2. PerformanceTracker records and updates weights
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

from config import Config
from logger import log


class TradeStatus(Enum):
    """Status of a trade"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"
    NEVER_ENTERED = "NEVER_ENTERED"


@dataclass
class TradeRecord:
    """
    Complete trade record stored in performance system
    """
    # ===== IDENTIFICATION =====
    signal_id: str
    symbol: str
    direction: str
    timeframe: str
    signal_time: datetime
    
    # ===== TRADE SETUP =====
    entry_zone_low: float
    entry_zone_high: float
    entry_price: float
    stop_loss: float
    tp_levels: List[Dict]
    position_size: float
    position_value: float
    risk_amount: float
    risk_per_unit: float
    risk_reward: float
    
    # ===== TIMING =====
    expected_minutes_to_entry: int = 0
    actual_minutes_to_entry: Optional[int] = None
    entry_zone_hit_time: Optional[datetime] = None
    entry_zone_hit_price: Optional[float] = None
    
    # ===== OUTCOME (from TradeOutcomeAnalyzer) =====
    outcome: str = "PENDING"
    # WIN, PARTIAL_WIN, LOSS, NEVER_ENTERED_TP_HIT, NEVER_ENTERED_SL_HIT, NEVER_ENTERED_NO_HIT, EXPIRED
    
    first_hit_type: str = ""           # "TP", "SL", "NONE"
    first_hit_time: Optional[datetime] = None
    first_hit_price: Optional[float] = None
    first_tp_level: int = 0
    
    tp_hits: List[Dict] = field(default_factory=list)
    sl_hit: bool = False
    sl_hit_time: Optional[datetime] = None
    sl_hit_price: Optional[float] = None
    
    total_pnl: float = 0.0
    total_r: float = 0.0
    exit_reason: str = ""
    exit_time: Optional[datetime] = None
    
    # ===== PHASE SCORES =====
    final_probability: float = 0.0
    signal_grade: str = "UNKNOWN"
    mtf_score: float = 0.0
    smart_money_score: float = 0.0
    light_confirm_score: float = 0.0
    
    # ===== EXPERT DETAILS (for weight updates) =====
    expert_details: Dict[str, Dict] = field(default_factory=dict)
    agreeing_experts: List[str] = field(default_factory=list)
    disagreeing_experts: List[str] = field(default_factory=list)
    
    # ===== MARKET CONTEXT =====
    market_regime: str = "UNKNOWN"
    volatility_state: str = "NORMAL"
    
    # ===== STATUS =====
    status: str = "PENDING"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list):
                result[key] = value
            elif isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary"""
        # Parse datetime fields
        datetime_fields = ['signal_time', 'entry_zone_hit_time', 'first_hit_time',
                          'sl_hit_time', 'exit_time', 'created_at', 'updated_at']
        for field in datetime_fields:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    @property
    def is_win(self) -> bool:
        """Check if trade is a win"""
        return self.outcome in ['WIN', 'NEVER_ENTERED_TP_HIT']
    
    @property
    def is_partial_win(self) -> bool:
        """Check if trade is a partial win"""
        return self.outcome == 'PARTIAL_WIN'
    
    @property
    def is_loss(self) -> bool:
        """Check if trade is a loss"""
        return self.outcome in ['LOSS', 'NEVER_ENTERED_SL_HIT']
    
    @property
    def win_contribution(self) -> float:
        """Contribution to win rate (0, 0.5, or 1)"""
        if self.is_win:
            return 1.0
        elif self.is_partial_win:
            return 0.5
        else:
            return 0.0
    
    @property
    def achieved_r(self) -> float:
        """Get achieved R-multiple for weight updates"""
        if self.is_win:
            return self.total_r if self.total_r > 0 else 1.0
        elif self.is_partial_win:
            return self.total_r if self.total_r > 0 else 0.5
        else:
            return -1.0


@dataclass
class DailyPerformance:
    """Daily performance summary"""
    date: str
    total_trades: int = 0
    wins: int = 0
    partial_wins: int = 0
    losses: int = 0
    never_entered_tp: int = 0
    never_entered_sl: int = 0
    never_entered_none: int = 0
    pending: int = 0
    expired: int = 0
    
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    
    expert_performance: Dict[str, Dict] = field(default_factory=dict)
    grade_performance: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceTracker:
    """
    Complete Performance Tracking System - Phase 13
    
    Features:
    - Stores all trade records
    - Updates expert weights based on performance
    - Generates daily/weekly/monthly reports
    - Provides statistics for the entire system
    """
    
    def __init__(self):
        self.base_path = Path('signals/performance')
        self.trades_path = self.base_path / 'trades'
        self.daily_path = self.base_path / 'daily'
        self.monthly_path = self.base_path / 'monthly'
        self.expert_path = self.base_path / 'experts'
        
        # Create directories
        self._create_directories()
        
        # Active trades (not yet completed)
        self.active_trades: Dict[str, TradeRecord] = {}
        
        # All trades history
        self.trades: List[TradeRecord] = []
        
        # Expert performance cache
        self.expert_performance: Dict[str, Dict] = {}
        
        # Load existing trades
        self._load_all_trades()
        
        log.info("=" * 60)
        log.info("Performance Tracker initialized (Phase 13)")
        log.info("=" * 60)
        log.info(f"  Loaded {len(self.trades)} historical trades")
        log.info(f"  Active trades: {len(self.active_trades)}")
        log.info("=" * 60)
    
    # ========================================================================
    # DIRECTORY MANAGEMENT
    # ========================================================================
    
    def _create_directories(self):
        """Create all required directories"""
        for path in [self.trades_path, self.daily_path, self.monthly_path, self.expert_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # TRADE RECORDING
    # ========================================================================
    
    def record_signal(self, signal: Dict) -> Optional[str]:
        """
        Record a new validated signal as a pending trade
        
        Args:
            signal: Signal dictionary from Phase 10 (final validated)
        
        Returns:
            signal_id for reference
        """
        try:
            signal_id = signal.get('signal_id', '')
            if not signal_id:
                signal_id = f"{signal.get('symbol', 'UNKNOWN')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Extract trade setup
            direction = signal.get('direction', '')
            entry_zone_low = signal.get('entry_zone_low', 0)
            entry_zone_high = signal.get('entry_zone_high', 0)
            stop_loss = signal.get('stop_loss', 0)
            
            # Entry price based on direction
            if direction == 'BUY':
                entry_price = entry_zone_high
            else:
                entry_price = entry_zone_low
            
            # Get TP levels
            tp_levels = signal.get('take_profit_levels', [])
            if not tp_levels and signal.get('take_profit'):
                tp_levels = [{
                    'level': 1,
                    'price': signal['take_profit'],
                    'percentage': 1.0,
                    'description': 'Primary target'
                }]
            
            # Get position sizing
            position_size = signal.get('position_size', 1000)
            position_value = signal.get('position_value', 0)
            risk_amount = signal.get('risk_amount', 0)
            risk_per_unit = abs(entry_price - stop_loss)
            risk_reward = signal.get('risk_reward_ratio', 0)
            
            # Get expert details
            expert_details = signal.get('expert_details', {})
            agreeing_experts = []
            disagreeing_experts = []
            for name, details in expert_details.items():
                if details.get('agreed', False):
                    agreeing_experts.append(name)
                else:
                    disagreeing_experts.append(name)
            
            # Create trade record
            trade = TradeRecord(
                signal_id=signal_id,
                symbol=signal.get('symbol', ''),
                direction=direction,
                timeframe=signal.get('timeframe', '5m'),
                signal_time=self._parse_timestamp(signal.get('timestamp')),
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                entry_price=entry_price,
                stop_loss=stop_loss,
                tp_levels=tp_levels,
                position_size=position_size,
                position_value=position_value,
                risk_amount=risk_amount,
                risk_per_unit=risk_per_unit,
                risk_reward=risk_reward,
                expected_minutes_to_entry=signal.get('expected_minutes_to_entry', 0),
                final_probability=signal.get('probability', 0.5),
                signal_grade=signal.get('grade', 'UNKNOWN'),
                mtf_score=signal.get('mtf_score', 0.5),
                smart_money_score=signal.get('smart_money_score', 0.5),
                light_confirm_score=signal.get('light_confirm_score', 0.5),
                expert_details=expert_details,
                agreeing_experts=agreeing_experts,
                disagreeing_experts=disagreeing_experts,
                market_regime=signal.get('market_regime', 'UNKNOWN'),
                volatility_state=signal.get('volatility_state', 'NORMAL'),
                status=TradeStatus.PENDING.value,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store in active trades
            self.active_trades[signal_id] = trade
            self.trades.append(trade)
            
            # Save to file
            self._save_trades_for_date(trade.signal_time.date())
            
            log.debug(f"📝 Recorded signal: {signal_id} (Grade: {trade.signal_grade})")
            return signal_id
            
        except Exception as e:
            log.error(f"Error recording signal: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_trade_outcome(self, outcome: Dict) -> bool:
        """
        Update a trade with its outcome from TradeOutcomeAnalyzer
        
        Args:
            outcome: Outcome dictionary from TradeOutcomeAnalyzer
        
        Returns:
            True if update successful
        """
        try:
            signal_id = outcome.get('signal_id', '')
            if signal_id not in self.active_trades:
                # Check if already completed
                for trade in self.trades:
                    if trade.signal_id == signal_id and trade.status != TradeStatus.PENDING.value:
                        log.debug(f"Trade {signal_id} already completed")
                        return True
                log.warning(f"Trade {signal_id} not found in active trades")
                return False
            
            trade = self.active_trades[signal_id]
            
            # Update with outcome data
            trade.outcome = outcome.get('outcome', 'PENDING')
            trade.first_hit_type = outcome.get('first_hit_type', '')
            trade.first_hit_time = outcome.get('first_hit_time')
            trade.first_hit_price = outcome.get('first_hit_price')
            trade.first_tp_level = outcome.get('first_tp_level', 0)
            trade.tp_hits = outcome.get('tp_hits', [])
            trade.sl_hit = outcome.get('sl_hit', False)
            trade.sl_hit_time = outcome.get('sl_hit_time')
            trade.sl_hit_price = outcome.get('sl_hit_price')
            trade.total_pnl = outcome.get('total_pnl', 0)
            trade.total_r = outcome.get('total_r', 0)
            trade.exit_reason = outcome.get('exit_reason', '')
            trade.exit_time = outcome.get('exit_time')
            
            # Update entry zone hit info
            entry_zone = outcome.get('entry_zone_hit', {})
            if entry_zone.get('entered'):
                trade.entry_zone_hit_time = entry_zone.get('hit_time')
                trade.entry_zone_hit_price = entry_zone.get('hit_price')
                trade.actual_minutes_to_entry = entry_zone.get('minutes_to_entry')
            
            # Update status
            if trade.outcome in ['WIN', 'PARTIAL_WIN', 'LOSS', 'NEVER_ENTERED_TP_HIT', 
                                  'NEVER_ENTERED_SL_HIT', 'NEVER_ENTERED_NO_HIT', 'EXPIRED']:
                trade.status = TradeStatus.COMPLETED.value
                del self.active_trades[signal_id]
            else:
                trade.status = TradeStatus.ACTIVE.value
            
            trade.updated_at = datetime.now()
            
            # Save to file
            self._save_trades_for_date(trade.signal_time.date())
            
            log.debug(f"📊 Updated trade {signal_id}: {trade.outcome} (R: {trade.total_r:.2f})")
            return True
            
        except Exception as e:
            log.error(f"Error updating trade outcome: {e}")
            return False
    
    def record_entry_zone_hit(self, signal_id: str, hit_time: datetime, hit_price: float):
        """Record when entry zone was hit (for active trades)"""
        if signal_id in self.active_trades:
            trade = self.active_trades[signal_id]
            trade.entry_zone_hit_time = hit_time
            trade.entry_zone_hit_price = hit_price
            
            if trade.expected_minutes_to_entry > 0:
                minutes = (hit_time - trade.signal_time).total_seconds() / 60
                trade.actual_minutes_to_entry = int(minutes)
            
            trade.updated_at = datetime.now()
            self._save_trades_for_date(trade.signal_time.date())
    
    # ========================================================================
    # EXPERT WEIGHT UPDATES
    # ========================================================================
    
    def update_expert_weights(self, outcome: Dict, weight_manager=None) -> Dict[str, float]:
        """
        Update expert weights based on trade outcome
        
        Args:
            outcome: Trade outcome dictionary
            weight_manager: ExpertWeightManager instance (optional)
        
        Returns:
            Updated weights dictionary
        """
        signal_id = outcome.get('signal_id', '')
        
        # Find the trade record
        trade = None
        for t in self.trades:
            if t.signal_id == signal_id:
                trade = t
                break
        
        if not trade:
            log.warning(f"Trade {signal_id} not found for weight update")
            return {}
        
        # Determine if trade was profitable
        is_win = trade.is_win or trade.is_partial_win
        achieved_r = trade.achieved_r
        
        updated_weights = {}
        
        # Update each expert's weight
        for expert_name, details in trade.expert_details.items():
            # Check if expert agreed with the trade
            agreed = details.get('agreed', False)
            
            # Calculate weight adjustment
            if is_win:
                # Winning trade: increase weight if expert agreed, decrease if disagreed
                if agreed:
                    adjustment = +0.05
                else:
                    adjustment = -0.03
            else:
                # Losing trade: decrease weight if expert agreed, increase if disagreed
                if agreed:
                    adjustment = -0.05
                else:
                    adjustment = +0.03
            
            # Get current weight
            current_weight = details.get('weight', 1.0)
            new_weight = max(0.5, min(2.0, current_weight + adjustment))
            
            updated_weights[expert_name] = round(new_weight, 3)
            
            # Update in trade record
            trade.expert_details[expert_name]['weight'] = new_weight
        
        # Update weight manager if provided
        if weight_manager:
            for expert_name, new_weight in updated_weights.items():
                weight_manager.update_weight(expert_name, new_weight)
        
        # Save updated trade
        self._save_trades_for_date(trade.signal_time.date())
        
        log.debug(f"⚖️ Updated expert weights for {signal_id}: {updated_weights}")
        
        return updated_weights
    
    # ========================================================================
    # PERFORMANCE REPORTS
    # ========================================================================
    
    def get_expert_performance(self, min_trades: int = 5) -> Dict[str, Dict]:
        """
        Get performance metrics per expert for weight updates
        
        Returns:
            Dictionary with expert_name -> {win_rate, avg_r, performance_score, total_trades}
        """
        expert_stats = {}
        
        for trade in self.trades:
            if trade.status != TradeStatus.COMPLETED.value:
                continue
            
            for expert_name, details in trade.expert_details.items():
                if expert_name not in expert_stats:
                    expert_stats[expert_name] = {
                        'trades': 0,
                        'wins': 0,
                        'total_r': 0.0,
                        'weighted_r': 0.0
                    }
                
                expert_stats[expert_name]['trades'] += 1
                
                # Check if expert agreed with the trade
                if details.get('agreed', False):
                    if trade.is_win or trade.is_partial_win:
                        expert_stats[expert_name]['wins'] += 1
                        expert_stats[expert_name]['total_r'] += trade.achieved_r
                        expert_stats[expert_name]['weighted_r'] += trade.achieved_r
                    else:
                        expert_stats[expert_name]['total_r'] += trade.achieved_r
                        expert_stats[expert_name]['weighted_r'] += trade.achieved_r
                else:
                    # Expert disagreed
                    if trade.is_win or trade.is_partial_win:
                        # Expert disagreed but trade won - penalize
                        expert_stats[expert_name]['weighted_r'] -= trade.achieved_r * 0.5
                    else:
                        # Expert disagreed and trade lost - reward
                        expert_stats[expert_name]['weighted_r'] += abs(trade.achieved_r) * 0.3
        
        # Calculate performance scores
        result = {}
        for expert, stats in expert_stats.items():
            if stats['trades'] >= min_trades:
                win_rate = stats['wins'] / stats['trades']
                avg_r = stats['total_r'] / stats['trades']
                weighted_score = stats['weighted_r'] / stats['trades']
                
                # Performance score = win_rate × avg_r (weighted by agreement)
                performance_score = win_rate * avg_r * (1 + max(0, weighted_score))
                performance_score = min(1.5, max(0.3, performance_score))
                
                result[expert] = {
                    'total_trades': stats['trades'],
                    'wins': stats['wins'],
                    'win_rate': round(win_rate, 3),
                    'avg_r': round(avg_r, 3),
                    'weighted_score': round(weighted_score, 3),
                    'performance_score': round(performance_score, 3),
                    'recommended_weight': round(1.0 * performance_score / 0.75, 2)
                }
            else:
                result[expert] = {
                    'total_trades': stats['trades'],
                    'wins': stats['wins'],
                    'win_rate': 0.5,
                    'avg_r': 1.5,
                    'performance_score': 0.75,
                    'recommended_weight': 1.0
                }
        
        # Save expert performance
        self._save_expert_performance(result)
        
        return result
    
    def get_grade_win_rates(self, min_trades: int = 5) -> Dict[str, float]:
        """Get win rates by signal grade"""
        grade_stats = {}
        
        for trade in self.trades:
            if trade.status != TradeStatus.COMPLETED.value:
                continue
            
            grade = trade.signal_grade
            if grade not in grade_stats:
                grade_stats[grade] = {'wins': 0, 'total': 0}
            
            grade_stats[grade]['total'] += 1
            if trade.is_win or trade.is_partial_win:
                grade_stats[grade]['wins'] += 1
        
        win_rates = {}
        for grade, stats in grade_stats.items():
            if stats['total'] >= min_trades:
                win_rates[grade] = stats['wins'] / stats['total']
        
        return win_rates
    
    def generate_daily_summary(self, target_date: date) -> DailyPerformance:
        """Generate daily performance summary"""
        # Get all completed trades for this date
        day_trades = [
            t for t in self.trades 
            if t.status == TradeStatus.COMPLETED.value 
            and t.exit_time and t.exit_time.date() == target_date
        ]
        
        if not day_trades:
            return DailyPerformance(date=target_date.isoformat())
        
        # Count outcomes
        wins = sum(1 for t in day_trades if t.is_win)
        partial_wins = sum(1 for t in day_trades if t.is_partial_win)
        losses = sum(1 for t in day_trades if t.is_loss)
        never_entered_tp = sum(1 for t in day_trades if t.outcome == 'NEVER_ENTERED_TP_HIT')
        never_entered_sl = sum(1 for t in day_trades if t.outcome == 'NEVER_ENTERED_SL_HIT')
        never_entered_none = sum(1 for t in day_trades if t.outcome == 'NEVER_ENTERED_NO_HIT')
        expired = sum(1 for t in day_trades if t.outcome == 'EXPIRED')
        total = len(day_trades)
        
        # Calculate win rate (weighted)
        weighted_wins = sum(t.win_contribution for t in day_trades)
        win_rate = weighted_wins / total if total > 0 else 0
        
        # Calculate P&L
        total_pnl = sum(t.total_pnl for t in day_trades)
        
        # Calculate R values
        r_values = [t.total_r for t in day_trades if t.total_r != 0]
        avg_r = sum(r_values) / len(r_values) if r_values else 0
        
        # Calculate profit factor
        gross_profit = sum(t.total_pnl for t in day_trades if t.total_pnl > 0)
        gross_loss = abs(sum(t.total_pnl for t in day_trades if t.total_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expert performance for this day
        expert_perf = {}
        for trade in day_trades:
            for expert_name, details in trade.expert_details.items():
                if expert_name not in expert_perf:
                    expert_perf[expert_name] = {'trades': 0, 'wins': 0, 'total_r': 0}
                
                expert_perf[expert_name]['trades'] += 1
                if trade.is_win or trade.is_partial_win:
                    expert_perf[expert_name]['wins'] += 1
                expert_perf[expert_name]['total_r'] += trade.achieved_r
        
        # Grade performance
        grade_perf = {}
        for trade in day_trades:
            grade = trade.signal_grade
            if grade not in grade_perf:
                grade_perf[grade] = {'trades': 0, 'wins': 0, 'total_r': 0}
            
            grade_perf[grade]['trades'] += 1
            if trade.is_win or trade.is_partial_win:
                grade_perf[grade]['wins'] += 1
            grade_perf[grade]['total_r'] += trade.achieved_r
        
        summary = DailyPerformance(
            date=target_date.isoformat(),
            total_trades=total,
            wins=wins,
            partial_wins=partial_wins,
            losses=losses,
            never_entered_tp=never_entered_tp,
            never_entered_sl=never_entered_sl,
            never_entered_none=never_entered_none,
            pending=0,
            expired=expired,
            win_rate=round(win_rate, 3),
            total_pnl=round(total_pnl, 2),
            avg_r=round(avg_r, 3),
            profit_factor=round(profit_factor, 2),
            expert_performance=expert_perf,
            grade_performance=grade_perf
        )
        
        # Save summary
        file_path = self.daily_path / f"{target_date.isoformat()}.json"
        with open(file_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        
        return summary
    
    def generate_weekly_report(self, end_date: date, weeks: int = 1) -> Dict[str, Any]:
        """Generate weekly performance report"""
        start_date = end_date - timedelta(days=weeks * 7)
        
        week_trades = [
            t for t in self.trades 
            if t.status == TradeStatus.COMPLETED.value 
            and t.exit_time and start_date <= t.exit_time.date() <= end_date
        ]
        
        if not week_trades:
            return {'status': 'No trades in period', 'total_trades': 0}
        
        # Calculate metrics
        wins = sum(1 for t in week_trades if t.is_win)
        partial_wins = sum(1 for t in week_trades if t.is_partial_win)
        losses = sum(1 for t in week_trades if t.is_loss)
        total = len(week_trades)
        
        weighted_wins = sum(t.win_contribution for t in week_trades)
        win_rate = weighted_wins / total if total > 0 else 0
        
        total_pnl = sum(t.total_pnl for t in week_trades)
        
        r_values = [t.total_r for t in week_trades if t.total_r != 0]
        avg_r = sum(r_values) / len(r_values) if r_values else 0
        
        gross_profit = sum(t.total_pnl for t in week_trades if t.total_pnl > 0)
        gross_loss = abs(sum(t.total_pnl for t in week_trades if t.total_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Daily breakdown
        daily_pnl = {}
        for t in week_trades:
            if t.exit_time:
                day = t.exit_time.date()
                daily_pnl[day] = daily_pnl.get(day, 0) + t.total_pnl
        
        best_day = max(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)
        worst_day = min(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)
        
        return {
            'period': f"{start_date} to {end_date}",
            'total_trades': total,
            'wins': wins,
            'partial_wins': partial_wins,
            'losses': losses,
            'win_rate': round(win_rate, 3),
            'total_pnl': round(total_pnl, 2),
            'avg_r': round(avg_r, 3),
            'profit_factor': round(profit_factor, 2),
            'best_day': f"{best_day[0]} (${best_day[1]:.2f})" if best_day[0] else "N/A",
            'worst_day': f"{worst_day[0]} (${worst_day[1]:.2f})" if worst_day[0] else "N/A"
        }
    
    def generate_monthly_report(self, year: int, month: int) -> Dict[str, Any]:
        """Generate monthly performance report"""
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)
        
        month_trades = [
            t for t in self.trades 
            if t.status == TradeStatus.COMPLETED.value 
            and t.exit_time and start_date <= t.exit_time.date() <= end_date
        ]
        
        if not month_trades:
            return {'status': 'No trades in period', 'total_trades': 0}
        
        # Calculate metrics
        wins = sum(1 for t in month_trades if t.is_win)
        partial_wins = sum(1 for t in month_trades if t.is_partial_win)
        losses = sum(1 for t in month_trades if t.is_loss)
        total = len(month_trades)
        
        weighted_wins = sum(t.win_contribution for t in month_trades)
        win_rate = weighted_wins / total if total > 0 else 0
        
        total_pnl = sum(t.total_pnl for t in month_trades)
        
        r_values = [t.total_r for t in month_trades if t.total_r != 0]
        avg_r = sum(r_values) / len(r_values) if r_values else 0
        
        gross_profit = sum(t.total_pnl for t in month_trades if t.total_pnl > 0)
        gross_loss = abs(sum(t.total_pnl for t in month_trades if t.total_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Daily breakdown
        daily_pnl = {}
        for t in month_trades:
            if t.exit_time:
                day = t.exit_time.date()
                daily_pnl[day] = daily_pnl.get(day, 0) + t.total_pnl
        
        # Weekly breakdown
        weekly_pnl = {}
        for t in month_trades:
            if t.exit_time:
                week_num = t.exit_time.isocalendar()[1]
                weekly_pnl[week_num] = weekly_pnl.get(week_num, 0) + t.total_pnl
        
        return {
            'month': f"{year}-{month:02d}",
            'total_trades': total,
            'wins': wins,
            'partial_wins': partial_wins,
            'losses': losses,
            'win_rate': round(win_rate, 3),
            'total_pnl': round(total_pnl, 2),
            'avg_r': round(avg_r, 3),
            'profit_factor': round(profit_factor, 2),
            'best_day': max(daily_pnl.values()) if daily_pnl else 0,
            'worst_day': min(daily_pnl.values()) if daily_pnl else 0,
            'best_week': max(weekly_pnl.values()) if weekly_pnl else 0,
            'worst_week': min(weekly_pnl.values()) if weekly_pnl else 0
        }
    
    # ========================================================================
    # DATA PERSISTENCE
    # ========================================================================
    
    def _load_all_trades(self):
        """Load all historical trades from files"""
        try:
            for file_path in sorted(self.trades_path.glob("*.json")):
                try:
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for trade_data in file_data:
                                trade = TradeRecord.from_dict(trade_data)
                                if trade.status == TradeStatus.COMPLETED.value:
                                    self.trades.append(trade)
                                else:
                                    self.active_trades[trade.signal_id] = trade
                                    self.trades.append(trade)
                        elif isinstance(file_data, dict):
                            trade = TradeRecord.from_dict(file_data)
                            if trade.status == TradeStatus.COMPLETED.value:
                                self.trades.append(trade)
                            else:
                                self.active_trades[trade.signal_id] = trade
                                self.trades.append(trade)
                except Exception as e:
                    log.debug(f"Error loading {file_path}: {e}")
            
            # Load expert performance cache
            expert_file = self.expert_path / 'expert_performance.json'
            if expert_file.exists():
                with open(expert_file, 'r') as f:
                    self.expert_performance = json.load(f)
                    
        except Exception as e:
            log.error(f"Error loading trades: {e}")
    
    def _save_trades_for_date(self, trade_date: date):
        """Save all trades for a specific date"""
        try:
            # Get all trades for this date
            date_trades = []
            for t in self.trades:
                if t.signal_time.date() == trade_date:
                    date_trades.append(t.to_dict())
            
            # Add active trades for this date
            for t in self.active_trades.values():
                if t.signal_time.date() == trade_date:
                    date_trades.append(t.to_dict())
            
            if not date_trades:
                return
            
            # Save to file
            file_path = self.trades_path / f"{trade_date.isoformat()}.json"
            with open(file_path, 'w') as f:
                json.dump(date_trades, f, indent=2, default=str)
            
        except Exception as e:
            log.error(f"Error saving trades for {trade_date}: {e}")
    
    def _save_expert_performance(self, expert_performance: Dict[str, Dict]):
        """Save expert performance data"""
        try:
            file_path = self.expert_path / 'expert_performance.json'
            with open(file_path, 'w') as f:
                json.dump(expert_performance, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Error saving expert performance: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp to datetime"""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        return datetime.now()
    
    def get_active_trades(self) -> List[TradeRecord]:
        """Get all active (open) trades"""
        return list(self.active_trades.values())
    
    def get_completed_trades(self, days: int = 30) -> List[TradeRecord]:
        """Get completed trades from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [t for t in self.trades if t.exit_time and t.exit_time > cutoff]
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        recent_trades = self.get_completed_trades(days)
        
        if not recent_trades:
            return {'status': 'No trades in period', 'total_trades': 0}
        
        wins = sum(1 for t in recent_trades if t.is_win)
        partial_wins = sum(1 for t in recent_trades if t.is_partial_win)
        losses = sum(1 for t in recent_trades if t.is_loss)
        total = len(recent_trades)
        
        weighted_wins = sum(t.win_contribution for t in recent_trades)
        win_rate = weighted_wins / total if total > 0 else 0
        
        r_values = [t.total_r for t in recent_trades if t.total_r != 0]
        avg_r = sum(r_values) / len(r_values) if r_values else 0
        
        gross_profit = sum(t.total_pnl for t in recent_trades if t.total_pnl > 0)
        gross_loss = abs(sum(t.total_pnl for t in recent_trades if t.total_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_pnl = sum(t.total_pnl for t in recent_trades)
        
        return {
            'period_days': days,
            'total_trades': total,
            'wins': wins,
            'partial_wins': partial_wins,
            'losses': losses,
            'win_rate': round(win_rate, 3),
            'avg_r': round(avg_r, 3),
            'total_pnl': round(total_pnl, 2),
            'profit_factor': round(profit_factor, 2),
            'by_grade': self.get_grade_win_rates(),
            'by_expert': self.get_expert_performance()
        }
    
    def get_trade_by_id(self, signal_id: str) -> Optional[TradeRecord]:
        """Get trade record by signal ID"""
        for trade in self.trades:
            if trade.signal_id == signal_id:
                return trade
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        completed = [t for t in self.trades if t.status == TradeStatus.COMPLETED.value]
        
        if not completed:
            return {'total_trades': 0, 'active_trades': len(self.active_trades)}
        
        wins = sum(1 for t in completed if t.is_win)
        partial_wins = sum(1 for t in completed if t.is_partial_win)
        losses = sum(1 for t in completed if t.is_loss)
        total = len(completed)
        
        weighted_wins = sum(t.win_contribution for t in completed)
        win_rate = weighted_wins / total if total > 0 else 0
        
        total_pnl = sum(t.total_pnl for t in completed)
        
        r_values = [t.total_r for t in completed if t.total_r != 0]
        avg_r = sum(r_values) / len(r_values) if r_values else 0
        
        return {
            'total_trades': len(self.trades),
            'completed_trades': total,
            'active_trades': len(self.active_trades),
            'wins': wins,
            'partial_wins': partial_wins,
            'losses': losses,
            'win_rate': round(win_rate, 3),
            'total_pnl': round(total_pnl, 2),
            'avg_r': round(avg_r, 3),
            'best_trade': max((t.total_pnl for t in completed), default=0),
            'worst_trade': min((t.total_pnl for t in completed), default=0)
        }
    
    def cleanup_old_files(self, days: int = 90):
        """Clean up old trade files"""
        cutoff = datetime.now() - timedelta(days=days)
        
        for file_path in self.trades_path.glob("*.json"):
            try:
                date_str = file_path.stem
                file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if datetime.combine(file_date, datetime.min.time()) < cutoff:
                    file_path.unlink()
                    log.debug(f"Deleted old trade file: {file_path}")
            except Exception:
                pass


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_performance_tracker() -> PerformanceTracker:
    """Create and return a PerformanceTracker instance"""
    return PerformanceTracker()