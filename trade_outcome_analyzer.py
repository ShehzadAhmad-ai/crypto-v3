# trade_outcome_analyzer.py - Phase 13: Complete Trade Outcome Analysis
"""
Trade Outcome Analyzer - Phase 13

3-PART ANALYSIS LOGIC:
1. PART 1: From signal time, which hits first? TP1 or SL?
   - Track price action from signal time (not waiting for entry)
   - Determine if TP or SL was hit first
   - Record first hit type, time, price, and TP level if applicable

2. PART 2: Check if price ever enters the entry zone
   - For BUY: price must drop INTO the zone from above
   - For SELL: price must rise INTO the zone from below
   - If never entered → outcome based on PART 1:
     * If TP hit first → NEVER_ENTERED_TP_HIT (WIN)
     * If SL hit first → NEVER_ENTERED_SL_HIT (LOSS)
     * If nothing hit → NEVER_ENTERED_NO_HIT (Neutral)

3. PART 3: If entered zone, track after entry
   - Track chronological order of TP hits and SL
   - If all TPs hit → WIN
   - If some TPs hit then SL → PARTIAL_WIN
   - If SL hits before any TP → LOSS

Outputs:
- Daily summary report (JSON)
- Terminal summary with key metrics
- Individual trade outcomes for weight updates
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

from config import Config
from logger import log
from data_fetcher import DataFetcher


@dataclass
class TradeOutcome:
    """Complete trade outcome result"""
    # ===== IDENTIFICATION =====
    signal_id: str
    symbol: str
    direction: str
    signal_time: datetime
    
    # ===== TRADE SETUP =====
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    tp_levels: List[Dict]
    position_size: float
    risk_per_unit: float = 0.0
    
    # ===== PART 1: FIRST HIT FROM SIGNAL TIME =====
    first_hit_type: str = ""           # "TP", "SL", or "NONE"
    first_hit_time: Optional[datetime] = None
    first_hit_price: Optional[float] = None
    first_tp_level: int = 0             # Which TP level hit first
    
    # ===== PART 2: ENTRY ZONE CHECK =====
    entered_zone: bool = False
    entry_zone_hit_time: Optional[datetime] = None
    entry_zone_hit_price: Optional[float] = None
    minutes_to_entry: Optional[int] = None
    
    # ===== PART 3: AFTER ENTRY TRACKING =====
    tp_hits_after_entry: List[Dict] = field(default_factory=list)   # Chronological order
    sl_hit_after_entry: bool = False
    sl_hit_time_after_entry: Optional[datetime] = None
    sl_hit_price_after_entry: Optional[float] = None
    
    # ===== FINAL OUTCOME =====
    outcome: str = "PENDING"
    # Possible outcomes:
    # - WIN: All TPs hit after entry
    # - PARTIAL_WIN: Some TPs hit, then SL
    # - LOSS: SL hit first (from signal time or after entry)
    # - NEVER_ENTERED_TP_HIT: Never entered zone, but TP hit from signal price (WIN)
    # - NEVER_ENTERED_SL_HIT: Never entered zone, but SL hit from signal price (LOSS)
    # - NEVER_ENTERED_NO_HIT: Never entered zone, no TP/SL hit
    # - EXPIRED: Time expired
    # - ERROR: Analysis error
    
    total_pnl: float = 0.0
    total_r: float = 0.0
    exit_reason: str = ""
    exit_time: Optional[datetime] = None
    
    # Expert details for weight updates (from signal metadata)
    expert_details: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'signal_time': self.signal_time.isoformat(),
            'entry_zone_low': self.entry_zone_low,
            'entry_zone_high': self.entry_zone_high,
            'stop_loss': self.stop_loss,
            'tp_levels': self.tp_levels,
            'position_size': self.position_size,
            'part1_first_hit': {
                'type': self.first_hit_type,
                'time': self.first_hit_time.isoformat() if self.first_hit_time else None,
                'price': self.first_hit_price,
                'tp_level': self.first_tp_level
            },
            'part2_entry_zone': {
                'entered': self.entered_zone,
                'hit_time': self.entry_zone_hit_time.isoformat() if self.entry_zone_hit_time else None,
                'hit_price': self.entry_zone_hit_price,
                'minutes_to_entry': self.minutes_to_entry
            },
            'part3_after_entry': {
                'tp_hits': self.tp_hits_after_entry,
                'sl_hit': self.sl_hit_after_entry,
                'sl_hit_time': self.sl_hit_time_after_entry.isoformat() if self.sl_hit_time_after_entry else None,
                'sl_hit_price': self.sl_hit_price_after_entry
            },
            'outcome': self.outcome,
            'total_pnl': round(self.total_pnl, 2),
            'total_r': round(self.total_r, 3),
            'exit_reason': self.exit_reason,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'expert_details': self.expert_details
        }
    
    @property
    def is_win(self) -> bool:
        """Check if trade is a win (for weight updates)"""
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


class TradeOutcomeAnalyzer:
    """
    Phase 13: Analyzes trade outcomes with 3-part logic
    
    This is the CORE analysis module that determines:
    1. First hit from signal time (TP or SL)
    2. Entry zone hit status
    3. After entry tracking
    4. Final outcome classification
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.max_check_hours = 168  # 7 days
        self.signals_folder = Path('signals/final')
        self.reports_folder = Path('signals/performance/daily')
        self.reports_folder.mkdir(parents=True, exist_ok=True)
        
        # TP percentages from config
        self.tp_percentages = getattr(Config, 'TP_PERCENTAGES', [0.25, 0.35, 0.25, 0.10, 0.05])
        
        log.info("=" * 60)
        log.info("Trade Outcome Analyzer initialized (Phase 13)")
        log.info("=" * 60)
    
    # ========================================================================
    # MAIN ANALYSIS METHODS
    # ========================================================================
    
    def analyze_previous_day(self) -> Dict[str, Any]:
        """Analyze all signals from previous day"""
        target_date = date.today() - timedelta(days=1)
        return self.analyze_date(target_date)
    
    def analyze_date(self, target_date: date) -> Dict[str, Any]:
        """
        Analyze all signals from a specific date
        
        Args:
            target_date: Date to analyze (e.g., 2026-03-30)
        
        Returns:
            Daily summary dictionary
        """
        log.info(f"\n{'='*70}")
        log.info(f"[ANALYSIS] Analyzing trade outcomes for {target_date}")
        log.info(f"{'='*70}")
        
        # Load signals from that date
        signals = self._load_signals(target_date)
        
        if not signals:
            log.info(f"[WARNING] No validated signals found for {target_date}")
            return self._empty_summary(target_date)
        
        log.info(f"📈 Found {len(signals)} signals to analyze")
        
        # Analyze each signal
        outcomes = []
        for signal in signals:
            outcome = self._analyze_signal(signal)
            outcomes.append(outcome)
        
        # Generate summary
        summary = self._generate_summary(target_date, outcomes)
        
        # Save report
        self._save_report(target_date, summary, outcomes)
        
        # Print terminal summary
        self._print_summary(summary)
        
        return summary
    
    # ========================================================================
    # SINGLE SIGNAL ANALYSIS - 3-PART LOGIC
    # ========================================================================
    
    def _analyze_signal(self, signal: Dict) -> TradeOutcome:
        """
        Analyze a single signal using 3-PART LOGIC
        
        PART 1: From signal time, which hits first? TP1 or SL?
        PART 2: Check if price ever enters entry zone
        PART 3: If entered, track after entry for TP/SL hits
        """
        
        # Extract signal data
        signal_id = signal.get('signal_id', '')
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', '')
        signal_time = self._parse_timestamp(signal.get('timestamp'))
        entry_zone_low = signal.get('entry_zone_low', 0)
        entry_zone_high = signal.get('entry_zone_high', 0)
        stop_loss = signal.get('stop_loss', 0)
        position_size = signal.get('position_size', 1000)
        
        # Get TP levels (sorted by distance from entry)
        tp_levels = self._get_tp_levels(signal, direction)
        
        # Get expert details for weight updates
        expert_details = signal.get('expert_details', {})
        
        # Calculate risk per unit
        if direction == 'BUY':
            entry_price = entry_zone_high
        else:
            entry_price = entry_zone_low
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Create outcome object
        outcome = TradeOutcome(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            signal_time=signal_time,
            entry_zone_low=entry_zone_low,
            entry_zone_high=entry_zone_high,
            stop_loss=stop_loss,
            tp_levels=tp_levels,
            position_size=position_size,
            risk_per_unit=risk_per_unit,
            expert_details=expert_details
        )
        
        # Fetch price data from signal time onward
        df = self._fetch_price_data(symbol, signal_time)
        
        if df is None or df.empty:
            log.debug(f"  {symbol}: No price data")
            outcome.outcome = "ERROR"
            outcome.exit_reason = "No price data"
            return outcome
        
        # ====================================================================
        # PART 1: FROM SIGNAL TIME, WHICH HITS FIRST? TP1 OR SL?
        # ====================================================================
        first_hit = self._find_first_hit(df, direction, stop_loss, tp_levels, signal_time)
        
        outcome.first_hit_type = first_hit['type']
        outcome.first_hit_time = first_hit['time']
        outcome.first_hit_price = first_hit['price']
        outcome.first_tp_level = first_hit.get('tp_level', 0)
        
        # If SL hit first, outcome is LOSS (even if later enters zone)
        if first_hit['type'] == 'SL':
            outcome.outcome = "LOSS"
            outcome.exit_reason = f"SL hit first at {first_hit['price']:.6f}"
            outcome.exit_time = first_hit['time']
            outcome.total_pnl = self._calculate_pnl(outcome, first_hit['price'])
            outcome.total_r = -1.0
            log.debug(f"  {symbol}: SL hit first at {first_hit['price']:.6f} → LOSS")
            return outcome
        
        # ====================================================================
        # PART 2: CHECK IF PRICE EVER ENTERS ENTRY ZONE
        # ====================================================================
        entered, hit_time, hit_price = self._check_entry_zone_hit(
            df=df,
            entry_low=entry_zone_low,
            entry_high=entry_zone_high,
            direction=direction,
            signal_time=signal_time
        )
        
        outcome.entered_zone = entered
        outcome.entry_zone_hit_time = hit_time
        outcome.entry_zone_hit_price = hit_price
        
        if entered:
            minutes = (hit_time - signal_time).total_seconds() / 60
            outcome.minutes_to_entry = int(minutes)
            log.debug(f"  {symbol}: Entered zone at {hit_price:.6f} after {minutes:.0f} min")
        
        # If never entered zone, determine outcome based on PART 1
        if not entered:
            if first_hit['type'] == 'TP':
                # TP hit first from signal price (never entered zone)
                outcome.outcome = "NEVER_ENTERED_TP_HIT"
                outcome.exit_reason = f"TP{first_hit['tp_level']} hit first from signal price (never entered zone)"
                outcome.exit_time = first_hit['time']
                outcome.total_pnl = self._calculate_pnl(outcome, first_hit['price'])
                outcome.total_r = self._calculate_r_from_tp_hit(outcome, first_hit)
                log.debug(f"  {symbol}: TP hit first from signal price (never entered) → WIN")
            elif first_hit['type'] == 'NONE':
                # Nothing hit and never entered
                outcome.outcome = "NEVER_ENTERED_NO_HIT"
                outcome.exit_reason = "Never entered zone, no TP/SL hit"
                outcome.total_pnl = 0
                outcome.total_r = 0
                log.debug(f"  {symbol}: Never entered, nothing hit → NEUTRAL")
            return outcome
        
        # ====================================================================
        # PART 3: ENTERED ZONE - TRACK AFTER ENTRY
        # ====================================================================
        self._track_after_entry(outcome, df, hit_time, direction, tp_levels, stop_loss)
        
        # Determine final outcome based on after-entry tracking
        if outcome.sl_hit_after_entry and outcome.tp_hits_after_entry:
            # Some TPs hit, then SL
            total_percentage = sum(h['percentage'] for h in outcome.tp_hits_after_entry)
            total_weighted_r = sum(h['r_multiple'] * h['percentage'] for h in outcome.tp_hits_after_entry)
            
            outcome.outcome = "PARTIAL_WIN"
            outcome.exit_reason = f"SL hit after {len(outcome.tp_hits_after_entry)} TP(s)"
            outcome.total_r = total_weighted_r / total_percentage if total_percentage > 0 else 0
            outcome.exit_time = outcome.sl_hit_time_after_entry
            log.debug(f"  {symbol}: Partial win - {len(outcome.tp_hits_after_entry)} TPs hit before SL")
            
        elif outcome.sl_hit_after_entry and not outcome.tp_hits_after_entry:
            # SL hit first after entry (no TPs hit)
            outcome.outcome = "LOSS"
            outcome.exit_reason = f"SL hit first after entry at {outcome.sl_hit_price_after_entry:.6f}"
            outcome.total_r = -1.0
            outcome.exit_time = outcome.sl_hit_time_after_entry
            log.debug(f"  {symbol}: SL hit first after entry → LOSS")
            
        elif not outcome.sl_hit_after_entry and outcome.tp_hits_after_entry:
            # Check if all TPs hit
            all_tp_levels = set(t['level'] for t in tp_levels)
            hit_levels = set(h['level'] for h in outcome.tp_hits_after_entry)
            
            if all_tp_levels.issubset(hit_levels):
                # All TPs hit
                total_percentage = sum(h['percentage'] for h in outcome.tp_hits_after_entry)
                total_weighted_r = sum(h['r_multiple'] * h['percentage'] for h in outcome.tp_hits_after_entry)
                
                outcome.outcome = "WIN"
                outcome.exit_reason = "All TPs hit"
                outcome.total_r = total_weighted_r / total_percentage if total_percentage > 0 else 0
                outcome.exit_time = outcome.tp_hits_after_entry[-1]['time']
                log.debug(f"  {symbol}: All TPs hit → WIN")
            else:
                # Some TPs hit, still active
                outcome.outcome = "PENDING"
                outcome.exit_reason = f"{len(outcome.tp_hits_after_entry)} TPs hit, still active"
        
        elif not outcome.sl_hit_after_entry and not outcome.tp_hits_after_entry:
            # Entered zone but nothing hit yet
            outcome.outcome = "PENDING"
            outcome.exit_reason = "Entered zone, waiting for TP/SL"
        
        # Calculate total PnL
        if outcome.exit_time:
            outcome.total_pnl = self._calculate_pnl(outcome, outcome.exit_price if hasattr(outcome, 'exit_price') else 0)
        
        return outcome
    
    # ========================================================================
    # PART 1 HELPER: FIND FIRST HIT FROM SIGNAL TIME
    # ========================================================================
    
    def _find_first_hit(self, df: pd.DataFrame, direction: str, stop_loss: float,
                        tp_levels: List[Dict], signal_time: datetime) -> Dict:
        """
        Find which hits first: TP or SL, from signal time
        Returns: {'type': 'TP' or 'SL' or 'NONE', 'time': datetime, 'price': float, 'tp_level': int}
        """
        # Filter data after signal time
        df_after = df[df.index > signal_time]
        
        if df_after.empty:
            return {'type': 'NONE', 'time': None, 'price': None, 'tp_level': 0}
        
        # Track events in order
        for idx, row in df_after.iterrows():
            candle_high = float(row['high'])
            candle_low = float(row['low'])
            
            # Check SL first
            if direction == 'BUY' and candle_low <= stop_loss:
                return {'type': 'SL', 'time': idx, 'price': stop_loss, 'tp_level': 0}
            elif direction == 'SELL' and candle_high >= stop_loss:
                return {'type': 'SL', 'time': idx, 'price': stop_loss, 'tp_level': 0}
            
            # Check TPs in order (closest first)
            for tp in tp_levels:
                if direction == 'BUY' and candle_high >= tp['price']:
                    return {'type': 'TP', 'time': idx, 'price': tp['price'], 'tp_level': tp['level']}
                elif direction == 'SELL' and candle_low <= tp['price']:
                    return {'type': 'TP', 'time': idx, 'price': tp['price'], 'tp_level': tp['level']}
        
        return {'type': 'NONE', 'time': None, 'price': None, 'tp_level': 0}
    
    # ========================================================================
    # PART 2 HELPER: CHECK ENTRY ZONE HIT
    # ========================================================================
    
    def _check_entry_zone_hit(self, df: pd.DataFrame, entry_low: float, entry_high: float,
                              direction: str, signal_time: datetime) -> Tuple[bool, Optional[datetime], Optional[float]]:
        """
        Check if price entered the entry zone AFTER signal time
        
        For BUY: Price must drop INTO the zone (from above)
        For SELL: Price must rise INTO the zone (from below)
        """
        df_after = df[df.index > signal_time]
        
        if df_after.empty:
            return False, None, None
        
        for idx, row in df_after.iterrows():
            candle_low = float(row['low'])
            candle_high = float(row['high'])
            
            # Check if candle touches the entry zone
            if candle_low <= entry_high and candle_high >= entry_low:
                # Price entered the zone
                if direction == 'BUY':
                    entry_price = max(entry_low, candle_low)
                else:
                    entry_price = min(entry_high, candle_high)
                return True, idx, entry_price
        
        return False, None, None
    
    # ========================================================================
    # PART 3 HELPER: TRACK AFTER ENTRY
    # ========================================================================
    
    def _track_after_entry(self, outcome: TradeOutcome, df: pd.DataFrame,
                           entry_time: datetime, direction: str,
                           tp_levels: List[Dict], stop_loss: float):
        """
        Track price action after entry to determine TP/SL hits in chronological order
        """
        # Get data after entry
        df_after = df[df.index >= entry_time]
        
        if df_after.empty:
            return
        
        # Track pending TP levels (by level number)
        pending_levels = [tp['level'] for tp in tp_levels]
        pending_levels.sort()  # Closest first
        
        # Track hits
        tp_hits = []
        sl_hit = False
        sl_time = None
        sl_price = None
        
        # Iterate through candles in chronological order
        for idx, row in df_after.iterrows():
            candle_high = float(row['high'])
            candle_low = float(row['low'])
            
            # Check SL
            if not sl_hit:
                if direction == 'BUY' and candle_low <= stop_loss:
                    sl_hit = True
                    sl_time = idx
                    sl_price = stop_loss
                elif direction == 'SELL' and candle_high >= stop_loss:
                    sl_hit = True
                    sl_time = idx
                    sl_price = stop_loss
            
            # Check TPs in order (closest first)
            for level in pending_levels[:]:
                tp = next((t for t in tp_levels if t['level'] == level), None)
                if tp:
                    if direction == 'BUY' and candle_high >= tp['price']:
                        # TP hit!
                        risk = outcome.risk_per_unit
                        reward = tp['price'] - outcome.entry_zone_high if direction == 'BUY' else outcome.entry_zone_low - tp['price']
                        r_multiple = reward / risk if risk > 0 else 0
                        
                        tp_hits.append({
                            'level': level,
                            'price': tp['price'],
                            'percentage': tp['percentage'],
                            'time': idx,
                            'r_multiple': r_multiple
                        })
                        pending_levels.remove(level)
                        
                    elif direction == 'SELL' and candle_low <= tp['price']:
                        risk = outcome.risk_per_unit
                        reward = outcome.entry_zone_low - tp['price'] if direction == 'SELL' else tp['price'] - outcome.entry_zone_high
                        r_multiple = reward / risk if risk > 0 else 0
                        
                        tp_hits.append({
                            'level': level,
                            'price': tp['price'],
                            'percentage': tp['percentage'],
                            'time': idx,
                            'r_multiple': r_multiple
                        })
                        pending_levels.remove(level)
            
            # Stop if we have outcome
            if sl_hit and not pending_levels:
                break
        
        # Store results
        outcome.tp_hits_after_entry = tp_hits
        outcome.sl_hit_after_entry = sl_hit
        outcome.sl_hit_time_after_entry = sl_time
        outcome.sl_hit_price_after_entry = sl_price
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _load_signals(self, target_date: date) -> List[Dict]:
        """Load validated signals from Phase 10 for a specific date"""
        file_path = self.signals_folder / f"{target_date.isoformat()}.json"
        
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                return []
        except Exception as e:
            log.error(f"Error loading {file_path}: {e}")
            return []
    
    def _fetch_price_data(self, symbol: str, start_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from signal time onward"""
        try:
            df = self.data_fetcher.fetch_ohlcv(symbol, '5m', limit=500)
            if df is None or df.empty:
                return None
            
            # Filter after signal time
            df = df[df.index >= start_time]
            return df
            
        except Exception as e:
            log.debug(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp to datetime"""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        return datetime.now()
    
    def _get_tp_levels(self, signal: Dict, direction: str) -> List[Dict]:
        """Get TP levels sorted by distance from entry"""
        tp_levels = []
        
        # Try to get from take_profit_levels
        if signal.get('take_profit_levels'):
            for i, tp in enumerate(signal['take_profit_levels']):
                tp_levels.append({
                    'level': i + 1,
                    'price': tp.get('price', 0),
                    'percentage': tp.get('percentage', self.tp_percentages[i] if i < len(self.tp_percentages) else 0.1),
                    'description': tp.get('description', f'TP{i+1}')
                })
        elif signal.get('take_profit'):
            tp_levels.append({
                'level': 1,
                'price': signal['take_profit'],
                'percentage': 1.0,
                'description': 'Primary target'
            })
        
        # Sort by distance from entry
        entry = signal.get('entry_zone_high') if direction == 'BUY' else signal.get('entry_zone_low')
        if entry:
            if direction == 'BUY':
                tp_levels.sort(key=lambda x: x['price'])
            else:
                tp_levels.sort(key=lambda x: x['price'], reverse=True)
        
        return tp_levels
    
    def _calculate_pnl(self, outcome: TradeOutcome, exit_price: float) -> float:
        """Calculate P&L for the trade"""
        if outcome.direction == 'BUY':
            entry = outcome.entry_zone_high
            pnl = (exit_price - entry) * outcome.position_size
        else:
            entry = outcome.entry_zone_low
            pnl = (entry - exit_price) * outcome.position_size
        return pnl
    
    def _calculate_r_from_tp_hit(self, outcome: TradeOutcome, first_hit: Dict) -> float:
        """Calculate R-multiple from first TP hit (never entered case)"""
        if outcome.direction == 'BUY':
            entry = outcome.entry_zone_high
            reward = first_hit['price'] - entry
        else:
            entry = outcome.entry_zone_low
            reward = entry - first_hit['price']
        
        risk = outcome.risk_per_unit
        return reward / risk if risk > 0 else 0
    
    # ========================================================================
    # SUMMARY GENERATION
    # ========================================================================
    
    def _generate_summary(self, target_date: date, outcomes: List[TradeOutcome]) -> Dict[str, Any]:
        """Generate daily summary from outcomes"""
        
        # Count outcomes
        total = len(outcomes)
        wins = sum(1 for o in outcomes if o.is_win)
        partial_wins = sum(1 for o in outcomes if o.is_partial_win)
        losses = sum(1 for o in outcomes if o.is_loss)
        never_entered_tp = sum(1 for o in outcomes if o.outcome == 'NEVER_ENTERED_TP_HIT')
        never_entered_sl = sum(1 for o in outcomes if o.outcome == 'NEVER_ENTERED_SL_HIT')
        never_entered_none = sum(1 for o in outcomes if o.outcome == 'NEVER_ENTERED_NO_HIT')
        pending = sum(1 for o in outcomes if o.outcome == 'PENDING')
        expired = sum(1 for o in outcomes if o.outcome == 'EXPIRED')
        errors = sum(1 for o in outcomes if o.outcome == 'ERROR')
        
        # Calculate P&L
        total_pnl = sum(o.total_pnl for o in outcomes)
        
        # Calculate R values
        r_values = [o.total_r for o in outcomes if o.total_r != 0]
        avg_r = sum(r_values) / len(r_values) if r_values else 0
        
        # Calculate profit factor
        gross_profit = sum(o.total_pnl for o in outcomes if o.total_pnl > 0)
        gross_loss = abs(sum(o.total_pnl for o in outcomes if o.total_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate win rate (weighted: win=1, partial=0.5)
        weighted_wins = sum(o.win_contribution for o in outcomes)
        win_rate = weighted_wins / total if total > 0 else 0
        
        # Expert performance
        expert_perf = {}
        for outcome in outcomes:
            for expert_name, details in outcome.expert_details.items():
                if expert_name not in expert_perf:
                    expert_perf[expert_name] = {'trades': 0, 'wins': 0, 'total_r': 0}
                
                expert_perf[expert_name]['trades'] += 1
                if outcome.is_win or outcome.is_partial_win:
                    expert_perf[expert_name]['wins'] += 1
                expert_perf[expert_name]['total_r'] += outcome.achieved_r
        
        # Calculate expert win rates
        for expert in expert_perf:
            trades = expert_perf[expert]['trades']
            wins = expert_perf[expert]['wins']
            expert_perf[expert]['win_rate'] = wins / trades if trades > 0 else 0
            expert_perf[expert]['avg_r'] = expert_perf[expert]['total_r'] / trades if trades > 0 else 0
            expert_perf[expert]['performance_score'] = expert_perf[expert]['win_rate'] * expert_perf[expert]['avg_r']
        
        # Performance by grade
        grade_perf = {}
        for outcome in outcomes:
            grade = outcome.expert_details.get('grade', 'UNKNOWN')
            if grade not in grade_perf:
                grade_perf[grade] = {'trades': 0, 'wins': 0, 'total_r': 0}
            
            grade_perf[grade]['trades'] += 1
            if outcome.is_win or outcome.is_partial_win:
                grade_perf[grade]['wins'] += 1
            grade_perf[grade]['total_r'] += outcome.achieved_r
        
        return {
            'date': target_date.isoformat(),
            'total_signals': total,
            'wins': wins,
            'partial_wins': partial_wins,
            'losses': losses,
            'never_entered_tp_hit': never_entered_tp,
            'never_entered_sl_hit': never_entered_sl,
            'never_entered_no_hit': never_entered_none,
            'pending': pending,
            'expired': expired,
            'errors': errors,
            'win_rate': round(win_rate, 3),
            'total_pnl': round(total_pnl, 2),
            'avg_r': round(avg_r, 3),
            'profit_factor': round(profit_factor, 2),
            'expert_performance': expert_perf,
            'grade_performance': grade_perf
        }
    
    def _save_report(self, target_date: date, summary: Dict, outcomes: List[TradeOutcome]):
        """Save analysis report to file"""
        report = {
            'summary': summary,
            'outcomes': [o.to_dict() for o in outcomes]
        }
        
        file_path = self.reports_folder / f"{target_date.isoformat()}.json"
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        log.info(f"📁 Report saved: {file_path}")
    
    def _print_summary(self, summary: Dict):
        """Print formatted terminal summary"""
        print("\n" + "=" * 80)
        print(f"📊 PERFORMANCE SUMMARY - {summary['date']}")
        print("=" * 80)
        
        # Basic stats
        total = summary['total_signals']
        wins = summary['wins']
        partial = summary['partial_wins']
        losses = summary['losses']
        
        print(f"\n📈 SIGNAL OUTCOMES:")
        print(f"   Total Signals:     {total}")
        print(f"   ✅ Wins:           {wins} ({wins/total*100:.1f}%)" if total > 0 else "   ✅ Wins:           0")
        print(f"   ⚡ Partial Wins:   {partial} ({partial/total*100:.1f}%)" if total > 0 else "   ⚡ Partial Wins:   0")
        print(f"   ❌ Losses:         {losses} ({losses/total*100:.1f}%)" if total > 0 else "   ❌ Losses:         0")
        print(f"   🚫 Never Entered:  {summary['never_entered_tp_hit'] + summary['never_entered_sl_hit'] + summary['never_entered_no_hit']}")
        print(f"   ⏳ Pending:        {summary['pending']}")
        
        # P&L
        print(f"\n💰 P&L SUMMARY:")
        print(f"   Total P&L:        ${summary['total_pnl']:,.2f}")
        print(f"   Avg R-Multiple:   {summary['avg_r']:.2f}")
        print(f"   Profit Factor:    {summary['profit_factor']:.2f}")
        
        # Expert performance
        print(f"\n🎯 EXPERT PERFORMANCE:")
        print(f"   {'Expert':<18} {'Trades':<8} {'Win%':<8} {'Avg R':<8} {'Score':<8}")
        print(f"   {'-'*50}")
        
        expert_perf = summary.get('expert_performance', {})
        for expert, perf in sorted(expert_perf.items(), key=lambda x: x[1]['performance_score'], reverse=True):
            print(f"   {expert:<18} {perf['trades']:<8} {perf['win_rate']*100:.0f}%{'':<4} "
                  f"{perf['avg_r']:<8.2f} {perf['performance_score']:<8.2f}")
        
        # Grade performance
        print(f"\n📊 GRADE PERFORMANCE:")
        print(f"   {'Grade':<8} {'Trades':<8} {'Win%':<8} {'Avg R':<8}")
        print(f"   {'-'*35}")
        
        grade_perf = summary.get('grade_performance', {})
        for grade, perf in sorted(grade_perf.items()):
            win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            avg_r = perf['total_r'] / perf['trades'] if perf['trades'] > 0 else 0
            print(f"   {grade:<8} {perf['trades']:<8} {win_rate*100:.0f}%{'':<4} {avg_r:<8.2f}")
        
        print("\n" + "=" * 80)
    
    def _empty_summary(self, target_date: date) -> Dict[str, Any]:
        """Return empty summary when no signals found"""
        return {
            'date': target_date.isoformat(),
            'total_signals': 0,
            'status': 'no_signals'
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_trade_outcomes(data_fetcher: DataFetcher, target_date: date = None) -> Dict[str, Any]:
    """
    Convenience function to analyze trade outcomes
    
    Args:
        data_fetcher: DataFetcher instance
        target_date: Date to analyze (default: yesterday)
    
    Returns:
        Daily summary dictionary
    """
    analyzer = TradeOutcomeAnalyzer(data_fetcher)
    if target_date:
        return analyzer.analyze_date(target_date)
    return analyzer.analyze_previous_day()