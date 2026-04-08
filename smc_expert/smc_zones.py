"""
SMC Expert V3 - Zone Analysis (COMPLETE REWRITE)
Premium/Discount zones, OTE levels, Supply/Demand zones, Fibonacci levels
FIXED: Index bounds, proper array access, no out-of-bounds errors
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .smc_core import (
    ZoneType, Direction, Candle, calculate_atr, normalize
)
from .smc_config import CONFIG


class PremiumDiscountAnalyzer:
    """Analyzes premium/discount zones and OTE levels"""
    
    def __init__(self):
        self.range_high: float = 0
        self.range_low: float = 0
        self.current_zone: ZoneType = ZoneType.EQUILIBRIUM
        self.zone_percent: float = 0.5
    
    def analyze(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """Analyze premium/discount based on recent range"""
        if len(df) < lookback:
            lookback = len(df)
        
        recent_df = df.tail(lookback)
        self.range_high = recent_df['high'].max()
        self.range_low = recent_df['low'].min()
        
        current_price = df['close'].iloc[-1]
        range_size = self.range_high - self.range_low
        
        if range_size <= 0:
            return {
                'zone': ZoneType.EQUILIBRIUM,
                'zone_percent': 0.5,
                'range_high': self.range_high,
                'range_low': self.range_low
            }
        
        # Calculate position in range (0 = low, 1 = high)
        self.zone_percent = (current_price - self.range_low) / range_size
        self.zone_percent = max(0.0, min(1.0, self.zone_percent))
        
        # Determine zone type
        if self.zone_percent <= 0.236:
            self.current_zone = ZoneType.DEEP_DISCOUNT
        elif self.zone_percent <= 0.382:
            self.current_zone = ZoneType.DISCOUNT
        elif self.zone_percent <= 0.618:
            self.current_zone = ZoneType.EQUILIBRIUM
        elif self.zone_percent <= 0.786:
            self.current_zone = ZoneType.PREMIUM
        else:
            self.current_zone = ZoneType.DEEP_PREMIUM
        
        return {
            'zone': self.current_zone,
            'zone_percent': self.zone_percent,
            'range_high': self.range_high,
            'range_low': self.range_low,
            'is_discount': self.zone_percent <= 0.382,
            'is_premium': self.zone_percent >= 0.618
        }
    
    def get_ote_levels(self, high: float, low: float, direction: Direction) -> Dict:
        """
        Calculate OTE (Optimal Trade Entry) levels
        OTE: 70.5% and 79% retracement
        """
        range_size = high - low
        
        if range_size <= 0:
            return {
                'ote_1': (high + low) / 2,
                'ote_2': (high + low) / 2,
                'level_1_pct': CONFIG.OTE_LEVEL_1,
                'level_2_pct': CONFIG.OTE_LEVEL_2,
                'range_high': high,
                'range_low': low
            }
        
        if direction == Direction.BUY:
            # For BUY: retrace from high to low
            ote_1 = high - (range_size * CONFIG.OTE_LEVEL_1)
            ote_2 = high - (range_size * CONFIG.OTE_LEVEL_2)
        else:
            # For SELL: retrace from low to high
            ote_1 = low + (range_size * CONFIG.OTE_LEVEL_1)
            ote_2 = low + (range_size * CONFIG.OTE_LEVEL_2)
        
        return {
            'ote_1': ote_1,
            'ote_2': ote_2,
            'level_1_pct': CONFIG.OTE_LEVEL_1,
            'level_2_pct': CONFIG.OTE_LEVEL_2,
            'range_high': high,
            'range_low': low
        }
    
    def is_in_ote(self, price: float, high: float, low: float, direction: Direction) -> Tuple[bool, float]:
        """Check if price is in OTE zone"""
        ote_levels = self.get_ote_levels(high, low, direction)
        
        if direction == Direction.BUY:
            in_zone = price >= ote_levels['ote_2'] and price <= ote_levels['ote_1']
            if in_zone and ote_levels['ote_1'] != ote_levels['ote_2']:
                depth = (price - ote_levels['ote_2']) / (ote_levels['ote_1'] - ote_levels['ote_2'])
                return True, min(1.0, depth)
        else:
            in_zone = price <= ote_levels['ote_2'] and price >= ote_levels['ote_1']
            if in_zone and ote_levels['ote_2'] != ote_levels['ote_1']:
                depth = (ote_levels['ote_2'] - price) / (ote_levels['ote_2'] - ote_levels['ote_1'])
                return True, min(1.0, depth)
        
        return False, 0.0


class SupplyDemandDetector:
    """Detects supply and demand zones - FIXED: Index bounds"""
    
    def __init__(self):
        self.supply_zones: List[Dict] = []
        self.demand_zones: List[Dict] = []
    
    def detect_all(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect supply and demand zones"""
        self.supply_zones = self._detect_zones(df, 'supply')
        self.demand_zones = self._detect_zones(df, 'demand')
        
        return {
            'supply': self.supply_zones,
            'demand': self.demand_zones
        }
    
    def _detect_zones(self, df: pd.DataFrame, zone_type: str) -> List[Dict]:
        """
        Detect supply (resistance) or demand (support) zones
        """
        zones = []
        atr = calculate_atr(df)
        
        if atr <= 0:
            atr = (df['high'].max() - df['low'].min()) / 50
        
        # SAFE: Proper bounds checking
        for i in range(5, len(df) - 6):
            if zone_type == 'supply':
                if self._is_supply_candidate(df, i):
                    zone = self._create_supply_zone(df, i, atr)
                    if zone:
                        zones.append(zone)
            else:
                if self._is_demand_candidate(df, i):
                    zone = self._create_demand_zone(df, i, atr)
                    if zone:
                        zones.append(zone)
        
        # Deduplicate and sort by strength
        zones = self._deduplicate_zones(zones, atr)
        zones.sort(key=lambda x: x['strength'], reverse=True)
        
        return zones
    
    def _is_supply_candidate(self, df: pd.DataFrame, i: int) -> bool:
        """Check if candle i is a potential supply zone"""
        # SAFE: Check bounds first
        if i + 2 >= len(df):
            return False
        
        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]
        
        # Check next_next safely
        if i + 2 < len(df):
            next_next = df.iloc[i + 2]
        else:
            return False
        
        # Price reaches high, then reverses down
        recent_high = df['high'].iloc[max(0, i-5):i].max()
        
        if candle['high'] > recent_high:
            if next_candle['close'] < next_candle['open']:
                if next_next['close'] < next_next['open']:
                    return True
        
        return False
    
    def _is_demand_candidate(self, df: pd.DataFrame, i: int) -> bool:
        """Check if candle i is a potential demand zone"""
        if i + 2 >= len(df):
            return False
        
        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]
        
        if i + 2 < len(df):
            next_next = df.iloc[i + 2]
        else:
            return False
        
        recent_low = df['low'].iloc[max(0, i-5):i].min()
        
        if candle['low'] < recent_low:
            if next_candle['close'] > next_candle['open']:
                if next_next['close'] > next_next['open']:
                    return True
        
        return False
    
    def _create_supply_zone(self, df: pd.DataFrame, i: int, atr: float) -> Optional[Dict]:
        """Create supply zone from reversal point"""
        candle = df.iloc[i]
        
        high = candle['high']
        low = candle['low']
        
        # Calculate strength
        avg_volume = df['volume'].iloc[max(0, i-20):i].mean()
        volume_ratio = candle['volume'] / avg_volume if avg_volume > 0 else 1.0
        volume_score = min(1.0, volume_ratio / 1.5)
        
        # Reversal strength
        if i + 1 < len(df):
            next_candle = df.iloc[i + 1]
            reversal_size = abs(next_candle['close'] - candle['close']) / atr if atr > 0 else 0
            reversal_score = min(1.0, reversal_size)
        else:
            reversal_score = 0.5
        
        strength = (volume_score * 0.4 + reversal_score * 0.6)
        
        return {
            'type': 'SUPPLY',
            'direction': Direction.SELL,
            'high': high,
            'low': low,
            'price': low,
            'stop': high,
            'strength': strength,
            'age_bars': len(df) - i,
            'timestamp': candle.name if isinstance(candle.name, datetime) else datetime.now(),
            'volume_ratio': volume_ratio,
            'reversal_strength': reversal_score
        }
    
    def _create_demand_zone(self, df: pd.DataFrame, i: int, atr: float) -> Optional[Dict]:
        """Create demand zone from reversal point"""
        candle = df.iloc[i]
        
        high = candle['high']
        low = candle['low']
        
        avg_volume = df['volume'].iloc[max(0, i-20):i].mean()
        volume_ratio = candle['volume'] / avg_volume if avg_volume > 0 else 1.0
        volume_score = min(1.0, volume_ratio / 1.5)
        
        if i + 1 < len(df):
            next_candle = df.iloc[i + 1]
            reversal_size = abs(next_candle['close'] - candle['close']) / atr if atr > 0 else 0
            reversal_score = min(1.0, reversal_size)
        else:
            reversal_score = 0.5
        
        strength = (volume_score * 0.4 + reversal_score * 0.6)
        
        return {
            'type': 'DEMAND',
            'direction': Direction.BUY,
            'high': high,
            'low': low,
            'price': high,
            'stop': low,
            'strength': strength,
            'age_bars': len(df) - i,
            'timestamp': candle.name if isinstance(candle.name, datetime) else datetime.now(),
            'volume_ratio': volume_ratio,
            'reversal_strength': reversal_score
        }
    
    def _deduplicate_zones(self, zones: List[Dict], atr: float) -> List[Dict]:
        """Remove overlapping zones"""
        if len(zones) < 2 or atr <= 0:
            return zones
        
        deduped = []
        tolerance = atr * 0.5
        
        for zone in zones:
            overlap = False
            for existing in deduped:
                if abs(zone['price'] - existing['price']) < tolerance:
                    overlap = True
                    break
            
            if not overlap:
                deduped.append(zone)
        
        return deduped


class FibonacciAnalyzer:
    """Calculates Fibonacci retracement and extension levels"""
    
    def __init__(self):
        self.retracements: Dict[str, float] = {}
        self.extensions: Dict[str, float] = {}
    
    def calculate_retracements(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        range_size = high - low
        
        if range_size <= 0:
            return {'0.0': high, '0.5': (high + low) / 2, '1.0': low}
        
        self.retracements = {
            '0.0': high,
            '0.236': high - (range_size * 0.236),
            '0.382': high - (range_size * 0.382),
            '0.5': high - (range_size * 0.5),
            '0.618': high - (range_size * 0.618),
            '0.705': high - (range_size * 0.705),
            '0.786': high - (range_size * 0.786),
            '1.0': low
        }
        
        return self.retracements
    
    def calculate_extensions(self, high: float, low: float, direction: Direction) -> Dict[str, float]:
        """Calculate Fibonacci extension levels"""
        range_size = high - low
        
        if range_size <= 0:
            return {'1.272': high, '1.618': high, '2.0': high}
        
        if direction == Direction.BUY:
            self.extensions = {
                '1.272': high + (range_size * 0.272),
                '1.414': high + (range_size * 0.414),
                '1.618': high + (range_size * 0.618),
                '2.0': high + range_size,
                '2.618': high + (range_size * 1.618)
            }
        else:
            self.extensions = {
                '1.272': low - (range_size * 0.272),
                '1.414': low - (range_size * 0.414),
                '1.618': low - (range_size * 0.618),
                '2.0': low - range_size,
                '2.618': low - (range_size * 1.618)
            }
        
        return self.extensions


class ZoneManager:
    """Main zone manager orchestrating all zone analysis"""
    
    def __init__(self):
        self.premium_discount = PremiumDiscountAnalyzer()
        self.supply_demand = SupplyDemandDetector()
        self.fib = FibonacciAnalyzer()
        
        self.current_zone: ZoneType = ZoneType.EQUILIBRIUM
        self.ote_levels: Dict = {}
        self.supply_zones: List[Dict] = []
        self.demand_zones: List[Dict] = []
    
    def analyze(self, df: pd.DataFrame, direction: Direction, 
                 range_high: float, range_low: float) -> Dict:
        """Complete zone analysis"""
        # Premium/Discount analysis
        pd_result = self.premium_discount.analyze(df)
        self.current_zone = pd_result['zone']
        
        # OTE levels
        self.ote_levels = self.premium_discount.get_ote_levels(range_high, range_low, direction)
        
        # Supply/Demand zones
        sd_zones = self.supply_demand.detect_all(df)
        self.supply_zones = sd_zones['supply']
        self.demand_zones = sd_zones['demand']
        
        # Fibonacci levels
        retracements = self.fib.calculate_retracements(range_high, range_low)
        extensions = self.fib.calculate_extensions(range_high, range_low, direction)
        
        # Check if in OTE
        current_price = df['close'].iloc[-1]
        in_ote, ote_depth = self.premium_discount.is_in_ote(current_price, range_high, range_low, direction)
        
        return {
            'premium_discount': pd_result,
            'ote_levels': self.ote_levels,
            'supply_zones': self.supply_zones[:10],
            'demand_zones': self.demand_zones[:10],
            'fib_retracements': retracements,
            'fib_extensions': extensions,
            'is_in_ote': in_ote,
            'ote_depth': ote_depth,
            'direction': direction
        }
    
    def get_best_zone(self, direction: Direction, current_price: float, atr: float) -> Optional[Dict]:
        """Get best zone for entry based on direction"""
        if direction == Direction.BUY:
            zones = self.demand_zones
            target_zone = 'demand'
        else:
            zones = self.supply_zones
            target_zone = 'supply'
        
        if not zones:
            return None
        
        # Find zones within reasonable distance
        valid_zones = []
        for zone in zones:
            if direction == Direction.BUY and zone['price'] < current_price:
                distance_pct = (current_price - zone['price']) / current_price if current_price > 0 else 0
                if distance_pct < 0.05:
                    valid_zones.append(zone)
            elif direction == Direction.SELL and zone['price'] > current_price:
                distance_pct = (zone['price'] - current_price) / current_price if current_price > 0 else 0
                if distance_pct < 0.05:
                    valid_zones.append(zone)
        
        if not valid_zones:
            return None
        
        valid_zones.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'type': target_zone,
            'zone': valid_zones[0],
            'has_confluence': len(valid_zones) > 1
        }