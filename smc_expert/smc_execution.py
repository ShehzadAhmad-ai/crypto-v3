"""
SMC Expert V3 - Execution Engine
Entry models, SL/TP calculation, risk management, invalidation, action determination
FIXED: Entry rejection logic, all classes preserved
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .smc_core import (
    Direction, ActionType, MitigationState, OBType, FVGType,
    OrderBlock, FVG, POI, SMCContext, LiquiditySweep, Candle,
    calculate_atr, normalize
)
from .smc_config import CONFIG, get_min_rr


class EntryModelRouter:
    """
    Routes to appropriate entry model based on POI type
    FIXED: Less strict conditions, proper validation
    """
    
    def __init__(self):
        self.entry: Optional[Dict] = None
    
    def get_entry(self, poi: POI, df: pd.DataFrame, context: SMCContext) -> Optional[Dict]:
        """
        Route to appropriate entry model based on POI type
        """
        if poi.type == 'OB':
            return self._ob_retest_entry(poi, df, context)
        elif poi.type == 'FVG':
            return self._fvg_retest_entry(poi, df, context)
        elif poi.type == 'SWEEP':
            return self._sweep_entry(poi, df, context)
        elif poi.type == 'OTE':
            return self._ote_entry(poi, df, context)
        elif poi.type == 'SILVER_BULLET':
            return self._silver_bullet_entry(poi, df, context)
        elif poi.type == 'TURTLE_SOUP':
            return self._turtle_soup_entry(poi, df, context)
        elif poi.type == 'SD':
            return self._sd_entry(poi, df, context)
        else:
            return None
    
    def _ob_retest_entry(self, poi: POI, df: pd.DataFrame, 
                          context: SMCContext) -> Optional[Dict]:
        """Order Block retest entry - FIXED: No strict confirmation required"""
        ob_data = poi.components[0]['data'] if poi.components else None
        
        if not ob_data or not isinstance(ob_data, OrderBlock):
            return None
        
        current_price = context.current_price
        atr = context.atr
        
        # Calculate distance to OB
        distance = abs(current_price - ob_data.price) / atr if atr > 0 else 0
        
        # FIXED: Allow entry if within 2.0 ATR (previously 1.5)
        if distance > CONFIG.POI_DISTANCE_MAX_ATR:
            return None
        
        entry = ob_data.price
        stop = ob_data.stop
        
        return {
            'type': 'OB_RETEST',
            'sub_type': ob_data.type.value,
            'direction': ob_data.direction,
            'entry': entry,
            'stop': stop,
            'strength': ob_data.strength,
            'poi': poi,
            'age_bars': ob_data.age_bars,
            'distance_atr': distance
        }
    
    def _fvg_retest_entry(self, poi: POI, df: pd.DataFrame, 
                           context: SMCContext) -> Optional[Dict]:
        """FVG retest entry - FIXED: Less strict conditions"""
        fvg_data = poi.components[0]['data'] if poi.components else None
        
        if not fvg_data or not isinstance(fvg_data, FVG):
            return None
        
        current_price = context.current_price
        atr = context.atr
        
        if fvg_data.direction == Direction.BUY:
            entry = fvg_data.mid
            stop = fvg_data.lower - (atr * 0.3)
        else:
            entry = fvg_data.mid
            stop = fvg_data.upper + (atr * 0.3)
        
        distance = abs(current_price - entry) / atr if atr > 0 else 0
        
        # FIXED: Allow entry within 2.5 ATR
        if distance <= CONFIG.POI_DISTANCE_MAX_ATR * 1.25:
            return {
                'type': 'FVG_RETEST',
                'sub_type': fvg_data.type.value,
                'direction': fvg_data.direction,
                'entry': entry,
                'stop': stop,
                'strength': fvg_data.strength,
                'poi': poi,
                'age_bars': fvg_data.age_bars,
                'distance_atr': distance
            }
        
        return None
    
    def _sweep_entry(self, poi: POI, df: pd.DataFrame, 
                      context: SMCContext) -> Optional[Dict]:
        """Liquidity sweep entry - FIXED: Allow older sweeps"""
        sweep_data = poi.components[0]['data'] if poi.components else None
        
        if not sweep_data or not isinstance(sweep_data, LiquiditySweep):
            return None
        
        current_price = context.current_price
        atr = context.atr
        
        if sweep_data.type == 'SSL_SWEEP':
            direction = Direction.BUY
            entry = current_price
            stop = sweep_data.target_level - (atr * 0.3)
        else:
            direction = Direction.SELL
            entry = current_price
            stop = sweep_data.target_level + (atr * 0.3)
        
        return {
            'type': 'SWEEP_ENTRY',
            'sub_type': sweep_data.type,
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'strength': sweep_data.reversal_strength,
            'poi': poi,
            'age_bars': 0,
            'distance_atr': 0
        }
    
    def _ote_entry(self, poi: POI, df: pd.DataFrame, 
                    context: SMCContext) -> Optional[Dict]:
        """OTE entry - FIXED: Less strict"""
        ote_data = poi.components[0]['data'] if poi.components else None
        
        if not ote_data:
            return None
        
        direction = poi.direction
        current_price = context.current_price
        atr = context.atr
        
        if direction == Direction.BUY:
            entry = ote_data.get('ote_2', ote_data.get('ote_1', current_price))
            stop = entry - (atr * 0.8)
            
            if current_price <= entry + (atr * 0.5):
                return {
                    'type': 'OTE_ENTRY',
                    'sub_type': f"OTE_{direction.value}",
                    'direction': direction,
                    'entry': entry,
                    'stop': stop,
                    'strength': 0.65,
                    'poi': poi,
                    'age_bars': 0,
                    'distance_atr': abs(current_price - entry) / atr if atr > 0 else 0
                }
        else:
            entry = ote_data.get('ote_2', ote_data.get('ote_1', current_price))
            stop = entry + (atr * 0.8)
            
            if current_price >= entry - (atr * 0.5):
                return {
                    'type': 'OTE_ENTRY',
                    'sub_type': f"OTE_{direction.value}",
                    'direction': direction,
                    'entry': entry,
                    'stop': stop,
                    'strength': 0.65,
                    'poi': poi,
                    'age_bars': 0,
                    'distance_atr': abs(current_price - entry) / atr if atr > 0 else 0
                }
        
        return None
    
    def _silver_bullet_entry(self, poi: POI, df: pd.DataFrame,
                              context: SMCContext) -> Optional[Dict]:
        """Silver Bullet pattern entry"""
        sb_data = poi.components[0]['data'] if poi.components else None
        
        if not sb_data:
            return None
        
        return {
            'type': 'SILVER_BULLET',
            'sub_type': sb_data.get('type', 'UNKNOWN'),
            'direction': sb_data.get('direction', Direction.NEUTRAL),
            'entry': sb_data.get('entry_price', context.current_price),
            'stop': sb_data.get('stop_loss', context.current_price - context.atr),
            'strength': sb_data.get('strength', 0.7),
            'poi': poi,
            'age_bars': 0,
            'distance_atr': 0
        }
    
    def _turtle_soup_entry(self, poi: POI, df: pd.DataFrame,
                            context: SMCContext) -> Optional[Dict]:
        """Turtle Soup pattern entry"""
        ts_data = poi.components[0]['data'] if poi.components else None
        
        if not ts_data:
            return None
        
        return {
            'type': 'TURTLE_SOUP',
            'sub_type': ts_data.get('type', 'UNKNOWN'),
            'direction': ts_data.get('direction', Direction.NEUTRAL),
            'entry': ts_data.get('entry_price', context.current_price),
            'stop': ts_data.get('stop_loss', context.current_price - context.atr),
            'strength': ts_data.get('strength', 0.6),
            'poi': poi,
            'age_bars': 0,
            'distance_atr': 0
        }
    
    def _sd_entry(self, poi: POI, df: pd.DataFrame, 
                   context: SMCContext) -> Optional[Dict]:
        """Supply/Demand zone entry"""
        sd_data = poi.components[0]['data'] if poi.components else None
        
        if not sd_data:
            return None
        
        direction = poi.direction
        current_price = context.current_price
        atr = context.atr
        
        if direction == Direction.BUY:
            entry = sd_data.get('price', sd_data.get('high', current_price))
            stop = sd_data.get('stop', sd_data.get('low', current_price - atr))
            
            if current_price <= entry + (atr * 0.5):
                return {
                    'type': 'SD_ENTRY',
                    'sub_type': sd_data.get('type', 'UNKNOWN'),
                    'direction': direction,
                    'entry': entry,
                    'stop': stop,
                    'strength': sd_data.get('strength', 0.5),
                    'poi': poi,
                    'age_bars': 0,
                    'distance_atr': abs(current_price - entry) / atr if atr > 0 else 0
                }
        else:
            entry = sd_data.get('price', sd_data.get('low', current_price))
            stop = sd_data.get('stop', sd_data.get('high', current_price + atr))
            
            if current_price >= entry - (atr * 0.5):
                return {
                    'type': 'SD_ENTRY',
                    'sub_type': sd_data.get('type', 'UNKNOWN'),
                    'direction': direction,
                    'entry': entry,
                    'stop': stop,
                    'strength': sd_data.get('strength', 0.5),
                    'poi': poi,
                    'age_bars': 0,
                    'distance_atr': abs(current_price - entry) / atr if atr > 0 else 0
                }
        
        return None


class SLTPEngine:
    """Dynamic Stop Loss and Take Profit calculation"""
    
    def __init__(self):
        self.min_rr: float = 1.2
    
    def calculate_sl(self, entry: float, direction: Direction, 
                      entry_type: str, context: SMCContext) -> float:
        """Calculate stop loss based on structure and ATR"""
        atr = context.atr
        
        # Base stop distance (0.8 ATR)
        base_distance = atr * 0.8
        
        # Adjust based on entry type
        if entry_type in ['OB_RETEST', 'SILVER_BULLET', 'TURTLE_SOUP']:
            return entry - base_distance if direction == Direction.BUY else entry + base_distance
        
        elif entry_type == 'FVG_RETEST':
            tight_distance = atr * 0.6
            return entry - tight_distance if direction == Direction.BUY else entry + tight_distance
        
        elif entry_type == 'SWEEP_ENTRY':
            tight_distance = atr * 0.5
            return entry - tight_distance if direction == Direction.BUY else entry + tight_distance
        
        else:
            structure_sl = self._get_structure_stop(entry, direction, context)
            
            if structure_sl:
                return structure_sl
            else:
                return entry - base_distance if direction == Direction.BUY else entry + base_distance
    
    def _get_structure_stop(self, entry: float, direction: Direction,
                             context: SMCContext) -> Optional[float]:
        """Get stop loss based on nearest swing"""
        if not context.swings:
            return None
        
        if direction == Direction.BUY:
            swings_below = [s for s in context.swings if s.price < entry]
            if swings_below:
                nearest = max(swings_below, key=lambda x: x.price)
                return nearest.price
        else:
            swings_above = [s for s in context.swings if s.price > entry]
            if swings_above:
                nearest = min(swings_above, key=lambda x: x.price)
                return nearest.price
        
        return None
    
    def calculate_tp(self, entry: float, direction: Direction,
                      risk: float, context: SMCContext, 
                      next_target: Optional[Dict] = None) -> float:
        """Calculate take profit"""
        if next_target and next_target.get('price'):
            return next_target['price']
        
        fib_target = self._get_fib_target(entry, direction, context)
        if fib_target:
            return fib_target
        
        return entry + (risk * 2.0) if direction == Direction.BUY else entry - (risk * 2.0)
    
    def _get_fib_target(self, entry: float, direction: Direction,
                         context: SMCContext) -> Optional[float]:
        """Get nearest Fibonacci extension target"""
        if not context.ote_levels:
            return None
        
        if direction == Direction.BUY:
            targets = [context.ote_levels.get('ote_1'), context.ote_levels.get('ote_2')]
            targets = [t for t in targets if t and t > entry]
            return min(targets) if targets else None
        else:
            targets = [context.ote_levels.get('ote_1'), context.ote_levels.get('ote_2')]
            targets = [t for t in targets if t and t < entry]
            return max(targets) if targets else None
    
    def calculate_rr(self, entry: float, sl: float, tp: float, 
                      direction: Direction) -> float:
        """Calculate Risk/Reward ratio"""
        if direction == Direction.BUY:
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        
        if risk <= 0:
            return 0
        
        return reward / risk


class RiskManager:
    """Manages risk based on confidence, regime, and POI strength"""
    
    def __init__(self):
        self.min_rr: float = 1.2
        self.position_size_multiplier: float = 1.0
    
    def get_min_rr(self, confidence: float, context: SMCContext, 
                    poi_strength: str) -> float:
        """Get dynamic minimum RR - LOWERED for more signals"""
        min_rr = 1.0
        
        if confidence > 0.80:
            min_rr = max(min_rr, 1.2)
        elif confidence > 0.65:
            min_rr = max(min_rr, 1.5)
        else:
            min_rr = max(min_rr, 1.8)
        
        if context.volatility_regime == "TRENDING":
            min_rr = max(min_rr, 1.2)
        elif context.volatility_regime == "RANGING":
            min_rr = max(min_rr, 1.5)
        else:
            min_rr = max(min_rr, 1.8)
        
        if not context.is_kill_zone:
            min_rr *= 1.1
        
        return round(min_rr, 2)
    
    def calculate_position_size(self, account_balance: float, 
                                 risk_amount: float, stop_distance: float) -> float:
        """Calculate position size based on risk"""
        if stop_distance <= 0:
            return 0
        
        max_risk = account_balance * CONFIG.MAX_RISK_PER_TRADE
        position_size = max_risk / stop_distance
        
        return position_size * self.position_size_multiplier
    
    def validate_rr(self, rr: float, min_rr: float) -> Tuple[bool, str]:
        """Validate if RR meets minimum requirement"""
        if rr >= min_rr:
            return True, f"RR {rr:.2f} meets minimum {min_rr:.2f}"
        else:
            return False, f"RR {rr:.2f} below minimum {min_rr:.2f}"


class InvalidationEngine:
    """Post-entry trade management and invalidation checks"""
    
    def __init__(self):
        self.invalidation_reason: Optional[str] = None
    
    def check_invalidation(self, entry: Dict, df: pd.DataFrame,
                            context: SMCContext) -> Dict:
        """Check if trade should be invalidated after entry"""
        current_price = context.current_price
        direction = entry['direction']
        
        checks = {
            'structure_broken': self._check_structure_broken(context, direction),
            'ob_invalidated': self._check_ob_invalidated(entry, context),
            'fvg_filled': self._check_fvg_filled(entry, context),
            'stop_hunt': self._check_stop_hunt(current_price, entry, context),
            'time_invalidation': self._check_time_invalidation(entry, context)
        }
        
        if checks['structure_broken']:
            return {'invalidated': True, 'reason': 'Structure broken', 'action': 'EXIT', 'severity': 'HIGH'}
        
        if checks['ob_invalidated']:
            return {'invalidated': True, 'reason': 'Order block invalidated', 'action': 'EXIT', 'severity': 'HIGH'}
        
        if checks['fvg_filled']:
            return {'invalidated': 'partial', 'reason': 'FVG filled', 'action': 'MOVE_STOP_TO_BREAKEVEN', 'severity': 'MEDIUM'}
        
        if checks['stop_hunt']:
            return {'invalidated': True, 'reason': 'Stop hunt detected', 'action': 'FLIP', 'severity': 'HIGH'}
        
        if checks['time_invalidation']:
            return {'invalidated': True, 'reason': 'Time invalidation', 'action': 'EXIT', 'severity': 'MEDIUM'}
        
        return {'invalidated': False, 'reason': None, 'action': 'HOLD', 'severity': 'NONE'}
    
    def _check_structure_broken(self, context: SMCContext, direction: Direction) -> bool:
        """Check if market structure has broken against position"""
        if not context.swings:
            return False
        
        current_price = context.current_price
        
        if direction == Direction.BUY:
            last_hl = [s for s in context.swings if s.type.value == 'HL']
            if last_hl and current_price < last_hl[-1].price:
                return True
        else:
            last_lh = [s for s in context.swings if s.type.value == 'LH']
            if last_lh and current_price > last_lh[-1].price:
                return True
        
        return False
    
    def _check_ob_invalidated(self, entry: Dict, context: SMCContext) -> bool:
        """Check if order block has been invalidated"""
        if entry['type'] != 'OB_RETEST':
            return False
        
        ob_data = entry.get('poi', {}).components[0].get('data') if entry.get('poi') else None
        
        if not ob_data:
            return False
        
        current_price = context.current_price
        atr = context.atr
        tolerance = atr * CONFIG.OB_MITIGATION_TOLERANCE_ATR
        
        if entry['direction'] == Direction.BUY:
            if current_price < ob_data.stop - tolerance:
                return True
        else:
            if current_price > ob_data.stop + tolerance:
                return True
        
        return False
    
    def _check_fvg_filled(self, entry: Dict, context: SMCContext) -> bool:
        """Check if FVG has been fully filled"""
        if entry['type'] != 'FVG_RETEST':
            return False
        
        fvg_data = entry.get('poi', {}).components[0].get('data') if entry.get('poi') else None
        
        if not fvg_data:
            return False
        
        current_price = context.current_price
        
        if entry['direction'] == Direction.BUY:
            if current_price >= fvg_data.upper:
                return True
        else:
            if current_price <= fvg_data.lower:
                return True
        
        return False
    
    def _check_stop_hunt(self, current_price: float, entry: Dict, context: SMCContext) -> bool:
        """Check if current move might be a stop hunt"""
        stop = entry.get('stop')
        direction = entry['direction']
        
        if not stop:
            return False
        
        atr = context.atr
        
        if direction == Direction.BUY:
            distance_to_stop = current_price - stop
            if distance_to_stop < atr * 0.2:
                return True
        else:
            distance_to_stop = stop - current_price
            if distance_to_stop < atr * 0.2:
                return True
        
        return False
    
    def _check_time_invalidation(self, entry: Dict, context: SMCContext) -> bool:
        """Check if setup is taking too long to trigger"""
        if entry.get('age_bars', 0) > 8:
            return True
        return False
    
    def get_stop_adjustment(self, entry: Dict, current_price: float,
                             context: SMCContext) -> Optional[float]:
        """Get adjusted stop loss level"""
        direction = entry['direction']
        stop = entry['stop']
        atr = context.atr
        
        if direction == Direction.BUY:
            profit = current_price - entry['entry']
            if profit > atr * 1.5:
                return entry['entry']
            
            highest_price = max(entry['entry'], current_price)
            new_stop = highest_price - atr
            if new_stop > stop:
                return new_stop
        else:
            profit = entry['entry'] - current_price
            if profit > atr * 1.5:
                return entry['entry']
            
            lowest_price = min(entry['entry'], current_price)
            new_stop = lowest_price + atr
            if new_stop < stop:
                return new_stop
        
        return None


class ActionDeterminer:
    """Determines final trading action based on all factors - RELAXED"""
    
    def __init__(self):
        self.action: ActionType = ActionType.SKIP
        self.reason: str = ""
    
    def determine_action(self, confidence: float, entry: Dict,
                          rr: float, min_rr: float,
                          context: SMCContext) -> Tuple[ActionType, str]:
        """Determine final action - FIXED: Less strict"""
        reasons = []
        
        if rr < min_rr:
            reasons.append(f"RR {rr:.2f} below minimum {min_rr:.2f}")
            return ActionType.SKIP, " | ".join(reasons)
        
        if not context.is_kill_zone and confidence < CONFIG.KILL_ZONE_REQUIRED_SCORE:
            reasons.append(f"Not in kill zone")
            confidence *= 0.9
        
        if confidence >= CONFIG.STRONG_ENTRY_THRESHOLD:
            action = ActionType.STRONG_ENTRY
            reasons.append(f"Strong entry ({confidence:.0%})")
        elif confidence >= CONFIG.ENTER_NOW_THRESHOLD:
            action = ActionType.ENTER_NOW
            reasons.append(f"Entry now ({confidence:.0%})")
        elif confidence >= CONFIG.WAIT_RETEST_THRESHOLD:
            action = ActionType.WAIT_RETEST
            reasons.append(f"Wait for retest ({confidence:.0%})")
        else:
            action = ActionType.SKIP
            reasons.append(f"Low confidence ({confidence:.0%})")
        
        self.action = action
        self.reason = " | ".join(reasons)
        
        return action, self.reason


class ExecutionManager:
    """Main execution orchestrator - FIXED FULL VERSION"""
    
    def __init__(self):
        self.entry_router = EntryModelRouter()
        self.sl_tp_engine = SLTPEngine()
        self.risk_manager = RiskManager()
        self.invalidation_engine = InvalidationEngine()
        self.action_determiner = ActionDeterminer()
        
        self.active_entry: Optional[Dict] = None
        self.signal: Optional[Dict] = None
    
    def process_entry(self, poi: POI, df: pd.DataFrame, 
                       context: SMCContext) -> Optional[Dict]:
        """Process a POI into a tradable entry"""
        entry = self.entry_router.get_entry(poi, df, context)
        
        if not entry:
            return None
        
        self.active_entry = entry
        return entry
    
    def calculate_risk_parameters(self, entry: Dict, context: SMCContext,
                                   next_target: Optional[Dict] = None) -> Dict:
        """Calculate SL, TP, RR based on entry and context"""
        direction = entry['direction']
        entry_price = entry['entry']
        
        sl = self.sl_tp_engine.calculate_sl(
            entry_price, direction, entry['type'], context
        )
        
        if entry.get('stop'):
            sl = entry['stop']
        
        if direction == Direction.BUY:
            risk = entry_price - sl
        else:
            risk = sl - entry_price
        
        tp = self.sl_tp_engine.calculate_tp(
            entry_price, direction, risk, context, next_target
        )
        
        rr = self.sl_tp_engine.calculate_rr(entry_price, sl, tp, direction)
        
        return {'sl': sl, 'tp': tp, 'risk': risk, 'rr': rr}
    
    def generate_signal(self, entry: Dict, risk_params: Dict,
                         confidence: float, grade: str, 
                         context: SMCContext, decision_reason: str) -> Optional[Dict]:
        """Generate final trading signal"""
        direction = entry['direction']
        
        action, action_reason = self.action_determiner.determine_action(
            confidence, entry, risk_params['rr'], 
            self.risk_manager.min_rr, context
        )
        
        if action == ActionType.SKIP:
            return None
        
        multiplier = CONFIG.POSITION_MULTIPLIERS.get(grade, 1.0)
        
        self.signal = {
            'pattern_name': f"SMC_{entry['type']}_{direction.value}",
            'direction': direction.value,
            'entry': round(entry['entry'], 4),
            'stop_loss': round(risk_params['sl'], 4),
            'take_profit': round(risk_params['tp'], 4),
            'risk_reward': round(risk_params['rr'], 2),
            'confidence': round(confidence, 2),
            'grade': grade,
            'action': action.value,
            'decision_reason': f"{decision_reason} | {action_reason}",
            'entry_details': {
                'type': entry['type'],
                'sub_type': entry.get('sub_type', 'UNKNOWN'),
                'strength': entry.get('strength', 0.5)
            },
            'risk_details': {
                'min_rr_required': self.risk_manager.min_rr,
                'position_multiplier': multiplier
            }
        }
        
        return self.signal
    
    def check_invalidation(self, df: pd.DataFrame, context: SMCContext) -> Dict:
        """Check if active trade should be invalidated"""
        if not self.active_entry:
            return {'invalidated': False}
        
        return self.invalidation_engine.check_invalidation(
            self.active_entry, df, context
        )
    
    def update_stop(self, current_price: float, context: SMCContext) -> Optional[float]:
        """Get updated stop loss for active trade"""
        if not self.active_entry:
            return None
        
        return self.invalidation_engine.get_stop_adjustment(
            self.active_entry, current_price, context
        )