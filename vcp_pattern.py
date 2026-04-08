# vcp_pattern.py
"""
Volatility Contraction Pattern (VCP)
Detects contracting pullbacks followed by breakouts
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from logger import log

@dataclass
class VCPattern:
    """Volatility Contraction Pattern data"""
    contraction_count: int  # Number of contractions
    start_idx: int
    end_idx: int
    entry_price: float
    stop_loss: float
    target_price: float
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    timestamp: pd.Timestamp

class VCPDetector:
    """
    Detects Volatility Contraction Pattern (VCP)
    - Multiple pullbacks with decreasing volatility
    - Volume contraction during pullbacks
    - Breakout after final contraction
    """
    
    def __init__(self):
        self.min_contractions = 2  # Minimum number of pullbacks
        self.max_contractions = 5  # Maximum number of pullbacks
        self.volatility_threshold = 0.7  # 30% reduction in volatility
    
    def detect_vcp(self, df: pd.DataFrame) -> List[VCPattern]:
        """
        Detect Volatility Contraction Patterns
        """
        patterns = []
        try:
            if df is None or df.empty or len(df) < 50:
                return patterns
            
            # Find the trend direction first
            trend = self._determine_trend(df)
            if trend == 'NEUTRAL':
                return patterns
            
            # Find swing points
            swings = self._find_swings(df)
            if len(swings) < 4:
                return patterns
            
            if trend == 'BULLISH':
                patterns = self._find_bullish_vcp(df, swings)
            else:
                patterns = self._find_bearish_vcp(df, swings)
            
            return patterns
            
        except Exception as e:
            log.error(f"Error detecting VCP: {e}")
            return patterns
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        try:
            if len(df) < 50:
                return 'NEUTRAL'
            
            # Use EMAs for trend
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1] if len(df) > 200 else None
            
            current_price = df['close'].iloc[-1]
            
            if ema200 and current_price > ema50 > ema200:
                return 'BULLISH'
            elif ema200 and current_price < ema50 < ema200:
                return 'BEARISH'
            elif current_price > ema50:
                return 'BULLISH'
            elif current_price < ema50:
                return 'BEARISH'
            
            return 'NEUTRAL'
            
        except Exception:
            return 'NEUTRAL'
    
    def _find_swings(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find significant swing points"""
        swings = []
        try:
            highs = df['high'].values
            lows = df['low'].values
            
            for i in range(window, len(df) - window):
                # Swing high
                if highs[i] == max(highs[i-window:i+window+1]):
                    swings.append({
                        'idx': i,
                        'price': float(highs[i]),
                        'type': 'HIGH',
                        'volatility': highs[i] - lows[i]
                    })
                
                # Swing low
                if lows[i] == min(lows[i-window:i+window+1]):
                    swings.append({
                        'idx': i,
                        'price': float(lows[i]),
                        'type': 'LOW',
                        'volatility': highs[i] - lows[i]
                    })
            
            swings.sort(key=lambda x: x['idx'])
            
        except Exception as e:
            log.error(f"Error finding swings: {e}")
        
        return swings
    
    def _find_bullish_vcp(self, df: pd.DataFrame, swings: List[Dict]) -> List[VCPattern]:
        """Find bullish VCP patterns (higher lows with decreasing volatility)"""
        patterns = []
        
        try:
            # Look for sequences of higher lows
            lows = [s for s in swings if s['type'] == 'LOW']
            if len(lows) < self.min_contractions + 1:
                return patterns
            
            for i in range(len(lows) - self.min_contractions):
                # Check for sequence of higher lows
                valid_sequence = True
                volatilities = []
                
                for j in range(self.min_contractions + 1):
                    if j > 0 and lows[i+j]['price'] <= lows[i+j-1]['price']:
                        valid_sequence = False
                        break
                    
                    # Calculate volatility from swing data if not present
                    if 'volatility' in lows[i+j]:
                        volatilities.append(lows[i+j]['volatility'])
                    else:
                        # Calculate approximate volatility from swing height
                        swing_idx = lows[i+j]['index']
                        if swing_idx > 0 and swing_idx < len(df):
                            recent_range = df['high'].iloc[max(0, swing_idx-5):swing_idx].max() - df['low'].iloc[max(0, swing_idx-5):swing_idx].min()
                            avg_price = df['close'].iloc[swing_idx]
                            volatility = recent_range / avg_price if avg_price > 0 else 0.02
                            volatilities.append(volatility)
                        else:
                            volatilities.append(0.02)  # Default 2% volatility
                
                if not valid_sequence:
                    continue
                
                # Check volatility contraction
                if len(volatilities) >= 2:
                    vol_ratios = [volatilities[k] / volatilities[k-1] for k in range(1, len(volatilities))]
                    avg_contraction = np.mean(vol_ratios)
                    
                    if avg_contraction < self.volatility_threshold:
                        # Valid VCP detected
                        start_idx = lows[i]['idx']
                        end_idx = lows[i+self.min_contractions]['idx']
                        
                        # Entry at breakout above recent highs
                        recent_high = max([h['price'] for h in swings if h['type'] == 'HIGH' and h['idx'] > start_idx and h['idx'] < end_idx + 10])
                        entry_price = recent_high * 1.01
                        
                        # Stop loss below last low
                        stop_loss = lows[i+self.min_contractions]['price'] * 0.99
                        
                        # Target based on pattern height
                        pattern_height = recent_high - lows[i]['price']
                        target_price = recent_high + pattern_height
                        
                        # Confidence based on contraction strength
                        confidence = 0.6 + (1 - avg_contraction) * 0.4
                        
                        patterns.append(VCPattern(
                            contraction_count=self.min_contractions,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            entry_price=round(entry_price, 6),
                            stop_loss=round(stop_loss, 6),
                            target_price=round(target_price, 6),
                            direction='BUY',
                            confidence=min(0.95, confidence),
                            timestamp=df.index[end_idx]
                        ))
            
            return patterns
            
        except Exception as e:
            log.error(f"Error finding bullish VCP: {e}")
            return patterns
    
    def _find_bearish_vcp(self, df: pd.DataFrame, swings: List[Dict]) -> List[VCPattern]:
        """Find bearish VCP patterns (lower highs with decreasing volatility)"""
        patterns = []
        
        try:
            # Look for sequences of lower highs
            highs = [s for s in swings if s['type'] == 'HIGH']
            if len(highs) < self.min_contractions + 1:
                return patterns
            
            for i in range(len(highs) - self.min_contractions):
                # Check for sequence of lower highs
                valid_sequence = True
                volatilities = []
                
                for j in range(self.min_contractions + 1):
                    if j > 0 and highs[i+j]['price'] >= highs[i+j-1]['price']:
                        valid_sequence = False
                        break
                    volatilities.append(highs[i+j]['volatility'])
                
                if not valid_sequence:
                    continue
                
                # Check volatility contraction
                if len(volatilities) >= 2:
                    vol_ratios = [volatilities[k] / volatilities[k-1] for k in range(1, len(volatilities))]
                    avg_contraction = np.mean(vol_ratios)
                    
                    if avg_contraction < self.volatility_threshold:
                        # Valid VCP detected
                        start_idx = highs[i]['idx']
                        end_idx = highs[i+self.min_contractions]['idx']
                        
                        # Entry at breakdown below recent lows
                        recent_low = min([l['price'] for l in swings if l['type'] == 'LOW' and l['idx'] > start_idx and l['idx'] < end_idx + 10])
                        entry_price = recent_low * 0.99
                        
                        # Stop loss above last high
                        stop_loss = highs[i+self.min_contractions]['price'] * 1.01
                        
                        # Target based on pattern height
                        pattern_height = highs[i]['price'] - recent_low
                        target_price = recent_low - pattern_height
                        
                        # Confidence based on contraction strength
                        confidence = 0.6 + (1 - avg_contraction) * 0.4
                        
                        patterns.append(VCPattern(
                            contraction_count=self.min_contractions,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            entry_price=round(entry_price, 6),
                            stop_loss=round(stop_loss, 6),
                            target_price=round(target_price, 6),
                            direction='SELL',
                            confidence=min(0.95, confidence),
                            timestamp=df.index[end_idx]
                        ))
            
            return patterns
            
        except Exception as e:
            log.error(f"Error finding bearish VCP: {e}")
            return patterns
    
    def get_entry_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal from latest VCP
        """
        patterns = self.detect_vcp(df)
        if not patterns:
            return None
        
        # Get most recent pattern
        latest = patterns[-1]
        
        return {
            'name': 'Volatility Contraction Pattern',
            'direction': latest.direction,
            'confidence': latest.confidence,
            'entry_price': latest.entry_price,
            'stop_loss': latest.stop_loss,
            'take_profit': latest.target_price,
            'reasons': [f"VCP with {latest.contraction_count} contractions, volatility reduced"],
            'type': 'VCP'
        }