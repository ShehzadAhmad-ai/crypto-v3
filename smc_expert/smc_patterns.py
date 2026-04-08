"""
SMC Expert V3 - ICT Patterns (COMPLETE REWRITE)
Silver Bullet, Power of Three (PO3), Turtle Soup patterns
FIXED: Proper object access, working detection logic
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .smc_core import (
    Direction, Candle, OrderBlock, FVG, LiquiditySweep,
    calculate_atr, df_row_to_candle, safe_get_swing_price
)
from .smc_config import CONFIG
from .smc_intelligence import SessionDetector


class SilverBulletDetector:
    """
    ICT Silver Bullet Pattern - COMPLETE REWRITE
    - London Silver Bullet: 3-4 AM EST (or 8-9 AM UTC)
    - NY Silver Bullet: 10-11 AM EST (or 2-3 PM UTC)
    - Requires: Liquidity sweep + FVG + Order Block confluence
    """
    
    def __init__(self):
        self.session_detector = SessionDetector()
    
    def detect(self, df: pd.DataFrame, sweeps: List[LiquiditySweep],
                order_blocks: List[OrderBlock], fvgs: List[FVG]) -> List[Dict]:
        """
        Detect Silver Bullet patterns
        """
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        timestamp = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
        
        # Check session for Silver Bullet timing
        session = self.session_detector.detect_session(timestamp)
        hour = timestamp.hour
        
        # Convert to UTC hour mapping
        # London Silver Bullet: 8-9 AM UTC
        # NY Silver Bullet: 2-3 PM UTC
        
        if session.value == "LONDON" and 8 <= hour < 9:
            patterns.extend(self._detect_silver_bullet(df, sweeps, order_blocks, fvgs, 'LONDON'))
        
        elif session.value == "NY" and 14 <= hour < 15:
            patterns.extend(self._detect_silver_bullet(df, sweeps, order_blocks, fvgs, 'NY'))
        
        # Also check London-NY overlap (13-15 UTC)
        elif session.value == "LONDON_NY_OVERLAP" and 13 <= hour < 15:
            patterns.extend(self._detect_silver_bullet(df, sweeps, order_blocks, fvgs, 'LONDON_NY'))
        
        return patterns
    
    def _detect_silver_bullet(self, df: pd.DataFrame, sweeps: List[LiquiditySweep],
                               order_blocks: List[OrderBlock], fvgs: List[FVG],
                               session_name: str) -> List[Dict]:
        """Core Silver Bullet detection logic"""
        patterns = []
        
        # Need recent sweeps (last 10 candles)
        current_idx = len(df)
        recent_sweeps = [s for s in sweeps if s.candle_index > current_idx - 15]
        
        if not recent_sweeps:
            return patterns
        
        # For each sweep, look for matching OB and FVG
        for sweep in recent_sweeps:
            if sweep.reversal_strength < 0.5:
                continue
            
            if sweep.type == 'SSL_SWEEP':
                # Bullish Silver Bullet: SSL sweep → look for bullish OB and FVG
                matching_obs = [ob for ob in order_blocks if ob.direction == Direction.BUY and ob.strength > 0.4]
                matching_fvgs = [fvg for fvg in fvgs if fvg.direction == Direction.BUY and fvg.strength > 0.3]
                
                if matching_obs and matching_fvgs:
                    # Take strongest OB and FVG
                    best_ob = max(matching_obs, key=lambda x: x.strength)
                    best_fvg = max(matching_fvgs, key=lambda x: x.strength)
                    
                    # Check if they are near the sweep
                    ob_distance = abs(best_ob.price - sweep.price) / (df['close'].iloc[-1] * 0.01) if df['close'].iloc[-1] > 0 else 1
                    fvg_distance = abs(best_fvg.mid - sweep.price) / (df['close'].iloc[-1] * 0.01) if df['close'].iloc[-1] > 0 else 1
                    
                    if ob_distance < 2 and fvg_distance < 2:  # Within 2%
                        strength = (sweep.reversal_strength + best_ob.strength + best_fvg.strength) / 3
                        
                        patterns.append({
                            'type': 'SILVER_BULLET_BULLISH',
                            'direction': Direction.BUY,
                            'sweep': sweep,
                            'order_block': best_ob,
                            'fvg': best_fvg,
                            'strength': strength,
                            'entry_price': best_ob.price,
                            'stop_loss': best_ob.stop,
                            'timestamp': sweep.timestamp,
                            'session': session_name
                        })
            
            elif sweep.type == 'BSL_SWEEP':
                # Bearish Silver Bullet: BSL sweep → look for bearish OB and FVG
                matching_obs = [ob for ob in order_blocks if ob.direction == Direction.SELL and ob.strength > 0.4]
                matching_fvgs = [fvg for fvg in fvgs if fvg.direction == Direction.SELL and fvg.strength > 0.3]
                
                if matching_obs and matching_fvgs:
                    best_ob = max(matching_obs, key=lambda x: x.strength)
                    best_fvg = max(matching_fvgs, key=lambda x: x.strength)
                    
                    ob_distance = abs(best_ob.price - sweep.price) / (df['close'].iloc[-1] * 0.01) if df['close'].iloc[-1] > 0 else 1
                    fvg_distance = abs(best_fvg.mid - sweep.price) / (df['close'].iloc[-1] * 0.01) if df['close'].iloc[-1] > 0 else 1
                    
                    if ob_distance < 2 and fvg_distance < 2:
                        strength = (sweep.reversal_strength + best_ob.strength + best_fvg.strength) / 3
                        
                        patterns.append({
                            'type': 'SILVER_BULLET_BEARISH',
                            'direction': Direction.SELL,
                            'sweep': sweep,
                            'order_block': best_ob,
                            'fvg': best_fvg,
                            'strength': strength,
                            'entry_price': best_ob.price,
                            'stop_loss': best_ob.stop,
                            'timestamp': sweep.timestamp,
                            'session': session_name
                        })
        
        return patterns


class PowerOfThreeDetector:
    """
    ICT Power of Three (PO3) - COMPLETE REWRITE
    Accumulation → Manipulation → Distribution
    """
    
    def __init__(self):
        self.phase = "UNKNOWN"
        self.accumulation_start = None
        self.manipulation_point = None
        self.distribution_start = None
    
    def detect(self, df: pd.DataFrame, sweeps: List[LiquiditySweep],
               bos_points: List[Dict]) -> List[Dict]:
        """
        Detect PO3 patterns and current phase
        """
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Detect accumulation phase (range-bound)
        accumulation = self._detect_accumulation(df)
        
        # Detect manipulation phase (liquidity sweep)
        manipulation = self._detect_manipulation(df, sweeps)
        
        # Detect distribution phase (trend move)
        distribution = self._detect_distribution(df, bos_points)
        
        # Determine current phase
        if accumulation and not manipulation:
            self.phase = "ACCUMULATION"
            self.accumulation_start = df.index[-20] if len(df) > 20 else df.index[0]
        elif manipulation and not distribution:
            self.phase = "MANIPULATION"
            self.manipulation_point = sweeps[-1] if sweeps else None
        elif distribution:
            self.phase = "DISTRIBUTION"
            self.distribution_start = df.index[-10] if len(df) > 10 else df.index[0]
        else:
            self.phase = "UNKNOWN"
        
        # Detect completed PO3 pattern (all 3 phases detected in sequence)
        if self._is_completed_pattern(df):
            patterns.append({
                'type': 'POWER_OF_THREE',
                'direction': self._get_direction(bos_points),
                'accumulation': True,
                'manipulation': True,
                'distribution': True,
                'strength': 0.8,
                'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
            })
        
        return patterns
    
    def _detect_accumulation(self, df: pd.DataFrame) -> bool:
        """Detect accumulation phase: range-bound with decreasing volume"""
        recent_df = df.tail(20)
        
        # Range width (should be narrow)
        range_high = recent_df['high'].max()
        range_low = recent_df['low'].min()
        range_width = (range_high - range_low) / recent_df['close'].mean()
        
        # Volume trend (should be decreasing)
        volumes = recent_df['volume'].values
        if len(volumes) > 5:
            volume_trend = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
            volume_decreasing = volume_trend < -0.2
        else:
            volume_decreasing = False
        
        return range_width < 0.02 and volume_decreasing
    
    def _detect_manipulation(self, df: pd.DataFrame, sweeps: List[LiquiditySweep]) -> bool:
        """Detect manipulation phase: liquidity sweep in last 10 bars"""
        if not sweeps:
            return False
        
        current_idx = len(df)
        recent_sweeps = [s for s in sweeps if s.candle_index > current_idx - 15]
        
        return len(recent_sweeps) > 0
    
    def _detect_distribution(self, df: pd.DataFrame, bos_points: List[Dict]) -> bool:
        """Detect distribution phase: trending with BOS"""
        if not bos_points:
            return False
        
        current_idx = len(df)
        recent_bos = [b for b in bos_points if b.get('index', 0) > current_idx - 20]
        
        return len(recent_bos) > 0
    
    def _is_completed_pattern(self, df: pd.DataFrame) -> bool:
        """Check if we have a completed PO3 pattern"""
        # Need at least 50 bars of history
        if len(df) < 50:
            return False
        
        # Check if we had accumulation (range) 20-40 bars ago
        acc_range = df.iloc[-40:-20]
        acc_high = acc_range['high'].max()
        acc_low = acc_range['low'].min()
        acc_width = (acc_high - acc_low) / acc_range['close'].mean()
        
        # Check if we had sweep in middle
        mid_range = df.iloc[-20:-10]
        mid_low = mid_range['low'].min()
        mid_high = mid_range['high'].max()
        
        # Check if we are now distributing (trending)
        recent = df.tail(10)
        trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
        
        return (acc_width < 0.02 and 
                abs(mid_low - acc_low) < acc_width * 0.5 and
                abs(trend) > 0)
    
    def _get_direction(self, bos_points: List[Dict]) -> str:
        """Get direction from recent BOS"""
        if not bos_points:
            return "NEUTRAL"
        
        recent_bos = bos_points[-5:] if len(bos_points) > 5 else bos_points
        
        bullish = sum(1 for b in recent_bos if b.get('type') == 'BULLISH')
        bearish = sum(1 for b in recent_bos if b.get('type') == 'BEARISH')
        
        if bullish > bearish:
            return "BUY"
        elif bearish > bullish:
            return "SELL"
        
        return "NEUTRAL"
    
    def get_phase(self) -> str:
        """Get current PO3 phase"""
        return self.phase


class TurtleSoupDetector:
    """
    ICT Turtle Soup Pattern - COMPLETE REWRITE
    - Bullish: Break below previous low → reversal up
    - Bearish: Break above previous high → reversal down
    """
    
    def __init__(self):
        pass
    
    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Turtle Soup patterns
        """
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        # Bullish Turtle Soup: break below previous low → close above
        bullish = self._detect_bullish_turtle_soup(df)
        if bullish:
            patterns.append(bullish)
        
        # Bearish Turtle Soup: break above previous high → close below
        bearish = self._detect_bearish_turtle_soup(df)
        if bearish:
            patterns.append(bearish)
        
        return patterns
    
    def _detect_bullish_turtle_soup(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Break below previous low, then reverse up
        """
        # Get previous low (5-10 bars ago)
        prev_low = df['low'].iloc[-10:-5].min()
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        
        # Break below previous low
        if current_low < prev_low:
            # Check for reversal (close above break level)
            if current_close > prev_low:
                # Calculate reversal strength
                atr = calculate_atr(df)
                reversal_size = (current_close - current_low) / atr if atr > 0 else 0
                strength = min(1.0, reversal_size / 1.5)
                
                return {
                    'type': 'TURTLE_SOUP_BULLISH',
                    'direction': Direction.BUY,
                    'break_level': prev_low,
                    'break_price': current_low,
                    'reversal_price': current_close,
                    'strength': strength,
                    'entry_price': current_close,
                    'stop_loss': current_low - atr * 0.3 if atr > 0 else current_low * 0.99,
                    'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
                }
        
        return None
    
    def _detect_bearish_turtle_soup(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Break above previous high, then reverse down
        """
        # Get previous high (5-10 bars ago)
        prev_high = df['high'].iloc[-10:-5].max()
        current_high = df['high'].iloc[-1]
        current_close = df['close'].iloc[-1]
        
        # Break above previous high
        if current_high > prev_high:
            # Check for reversal (close below break level)
            if current_close < prev_high:
                # Calculate reversal strength
                atr = calculate_atr(df)
                reversal_size = (current_high - current_close) / atr if atr > 0 else 0
                strength = min(1.0, reversal_size / 1.5)
                
                return {
                    'type': 'TURTLE_SOUP_BEARISH',
                    'direction': Direction.SELL,
                    'break_level': prev_high,
                    'break_price': current_high,
                    'reversal_price': current_close,
                    'strength': strength,
                    'entry_price': current_close,
                    'stop_loss': current_high + atr * 0.3 if atr > 0 else current_high * 1.01,
                    'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
                }
        
        return None


class ICTPatternManager:
    """Main ICT Pattern Manager - COMPLETE REWRITE"""
    
    def __init__(self):
        self.silver_bullet = SilverBulletDetector()
        self.power_of_three = PowerOfThreeDetector()
        self.turtle_soup = TurtleSoupDetector()
    
    def detect_all(self, df: pd.DataFrame, sweeps: List[LiquiditySweep],
                    order_blocks: List[OrderBlock], fvgs: List[FVG],
                    bos_points: List[Dict]) -> Dict:
        """
        Detect all ICT patterns
        """
        return {
            'silver_bullet': self.silver_bullet.detect(df, sweeps, order_blocks, fvgs),
            'power_of_three': self.power_of_three.detect(df, sweeps, bos_points),
            'turtle_soup': self.turtle_soup.detect(df),
            'current_po3_phase': self.power_of_three.get_phase()
        }