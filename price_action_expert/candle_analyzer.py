"""
candle_analyzer.py
Layer 1: Candle Analyzer for Price Action Expert V3.5

Converts raw OHLCV data into classified candles with:
- Body/wick measurements (raw and ATR-normalized)
- Strength scoring (0-1) based on body ratio
- Close position (0-1 within candle range)
- Candle type classification (marubozu, doji, etc.)
- Vectorized operations for speed
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Import configuration
from .price_action_config import (
    ATR_PERIOD,
    ATR_MULTIPLIER_WEAK,
    ATR_MULTIPLIER_STRONG,
    CANDLE_STRENGTH,
    DOJI_THRESHOLD
)


@dataclass
class CandleData:
    """
    Complete candle data with all derived metrics
    Used throughout the Price Action Expert for analysis
    """
    # Raw data
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Any
    index: int
    
    # Derived metrics
    body: float
    body_abs: float                    # Absolute body size (|close - open|)
    upper_wick: float
    lower_wick: float
    range: float
    close_position: float              # 0 = low, 1 = high
    
    # Normalized metrics (ATR-based)
    body_atr: float                    # Body / ATR
    upper_wick_atr: float
    lower_wick_atr: float
    range_atr: float
    
    # Classifications
    is_bullish: bool
    is_bearish: bool
    is_doji: bool
    candle_type: str                   # 'marubozu', 'strong', 'moderate', 'weak', 'doji'
    strength: float                    # 0-1 strength score
    conviction: str                    # 'weak', 'moderate', 'strong', 'extreme'
    
    # Additional metrics
    body_ratio: float                  # body / range (0-1)
    wick_top_ratio: float              # upper_wick / range (0-1)
    wick_bottom_ratio: float           # lower_wick / range (0-1)
    volume_ratio: float                # volume / average volume (20-period)
    
    # Optional: For reversal detection
    is_rejection_top: bool             # Long upper wick
    is_rejection_bottom: bool          # Long lower wick
    rejection_strength: float          # 0-1 strength of rejection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'body': self.body,
            'upper_wick': self.upper_wick,
            'lower_wick': self.lower_wick,
            'range': self.range,
            'close_position': self.close_position,
            'body_atr': self.body_atr,
            'is_bullish': self.is_bullish,
            'is_bearish': self.is_bearish,
            'is_doji': self.is_doji,
            'candle_type': self.candle_type,
            'strength': self.strength,
            'conviction': self.conviction,
            'body_ratio': self.body_ratio,
            'is_rejection_top': self.is_rejection_top,
            'is_rejection_bottom': self.is_rejection_bottom,
            'rejection_strength': self.rejection_strength,
        }


class CandleAnalyzer:
    """
    Advanced candle analyzer with vectorized operations
    Converts OHLCV data into classified CandleData objects
    
    Features:
    - ATR calculation for normalization
    - Strength scoring based on body ratio
    - Wick analysis for rejection detection
    - Volume ratio calculation
    - Vectorized for performance on large datasets
    """
    
    def __init__(self):
        """Initialize the candle analyzer"""
        self.atr_cache = {}
        self.volume_ma_cache = {}
        self.last_df_hash = None
        self.cached_atr = None
        self.cached_volume_ma = None
    
    def calculate_atr(self, df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
        """
        Calculate Average True Range (ATR) vectorized
        Cached for performance on repeated calls
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period (default 14)
        
        Returns:
            numpy array of ATR values
        """
        # Cache key
        df_hash = hash(len(df)) + hash(df.index[-1] if len(df) > 0 else 0)
        
        if self.cached_atr is not None and self.last_df_hash == df_hash:
            return self.cached_atr
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range components (vectorized)
            hl = high - low
            hc = np.abs(high - np.roll(close, 1))
            lc = np.abs(low - np.roll(close, 1))
            
            # True Range is max of the three
            tr = np.maximum(hl, np.maximum(hc, lc))
            
            # Set first value (no previous close)
            tr[0] = hl[0]
            
            # Calculate ATR with rolling mean
            atr = pd.Series(tr).rolling(window=period).mean().values
            
            # Fill NaN at the beginning with expanding mean
            for i in range(period):
                if np.isnan(atr[i]):
                    atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
            
            self.cached_atr = atr
            self.last_df_hash = df_hash
            
            return atr
            
        except Exception as e:
            # Fallback: simple ATR approximation
            print(f"Error calculating ATR: {e}")
            close_prices = df['close'].values
            atr = np.full(len(df), np.std(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1] * 0.02)
            return atr
    
    def calculate_volume_ma(self, df: pd.DataFrame, period: int = 20) -> np.ndarray:
        """
        Calculate volume moving average vectorized
        
        Args:
            df: DataFrame with OHLCV data
            period: MA period (default 20)
        
        Returns:
            numpy array of volume MA values
        """
        df_hash = hash(len(df))
        
        if self.cached_volume_ma is not None and self.last_df_hash == df_hash:
            return self.cached_volume_ma
        
        try:
            volume = df['volume'].values
            vol_ma = pd.Series(volume).rolling(window=period).mean().values
            
            # Fill NaN with expanding mean
            for i in range(period):
                if np.isnan(vol_ma[i]):
                    vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[0]
            
            self.cached_volume_ma = vol_ma
            return vol_ma
            
        except Exception as e:
            print(f"Error calculating volume MA: {e}")
            return np.ones(len(df)) * df['volume'].mean()
    
    def analyze_candle(self, df: pd.DataFrame, idx: int, atr_array: np.ndarray, vol_ma_array: np.ndarray) -> CandleData:
        """
        Analyze a single candle and return CandleData object
        
        Args:
            df: DataFrame with OHLCV data
            idx: Index of candle to analyze
            atr_array: Pre-calculated ATR array
            vol_ma_array: Pre-calculated volume MA array
        
        Returns:
            CandleData object with all derived metrics
        """
        row = df.iloc[idx]
        
        open_price = float(row['open'])
        high_price = float(row['high'])
        low_price = float(row['low'])
        close_price = float(row['close'])
        volume = float(row['volume'])
        timestamp = row.name if hasattr(row, 'name') else idx
        atr = float(atr_array[idx])
        vol_ma = float(vol_ma_array[idx])
        
        # Basic calculations
        body_abs = abs(close_price - open_price)
        range_candle = high_price - low_price
        
        # Wick calculations
        if close_price > open_price:  # Bullish
            upper_wick = high_price - close_price
            lower_wick = open_price - low_price
        else:  # Bearish
            upper_wick = high_price - open_price
            lower_wick = close_price - low_price
        
        # Close position (0 = low, 1 = high)
        if range_candle > 0:
            close_position = (close_price - low_price) / range_candle
        else:
            close_position = 0.5
        
        # Body ratio (body / range)
        if range_candle > 0:
            body_ratio = body_abs / range_candle
        else:
            body_ratio = 1.0
        
        # Normalized metrics (ATR-based)
        body_atr = body_abs / (atr + 1e-8)
        upper_wick_atr = upper_wick / (atr + 1e-8)
        lower_wick_atr = lower_wick / (atr + 1e-8)
        range_atr = range_candle / (atr + 1e-8)
        
        # Volume ratio
        volume_ratio = volume / (vol_ma + 1e-8)
        
        # Candle classification based on body ratio
        is_bullish = close_price > open_price
        is_bearish = close_price < open_price
        is_doji = body_ratio <= DOJI_THRESHOLD
        
        # Determine candle type and strength
        candle_type = 'normal'
        strength = 0.5
        
        if is_doji:
            candle_type = 'doji'
            strength = CANDLE_STRENGTH['doji']['strength']
        else:
            # Find strength based on body ratio
            for type_name, config in CANDLE_STRENGTH.items():
                if type_name == 'doji':
                    continue
                if body_ratio >= config['min_ratio']:
                    candle_type = type_name
                    strength = config['strength']
                    break
            
            # If no match, use moderate as default
            if candle_type == 'normal':
                if body_ratio >= 0.5:
                    candle_type = 'moderate'
                    strength = 0.65
                else:
                    candle_type = 'weak'
                    strength = 0.50
        
        # Determine conviction based on strength and volume
        if strength >= 0.85 and volume_ratio > 1.5:
            conviction = 'extreme'
        elif strength >= 0.80:
            conviction = 'strong'
        elif strength >= 0.60:
            conviction = 'moderate'
        else:
            conviction = 'weak'
        
        # Wick ratio for rejection detection
        wick_top_ratio = upper_wick / (range_candle + 1e-8)
        wick_bottom_ratio = lower_wick / (range_candle + 1e-8)
        
        # Rejection detection (wick > body × 2)
        is_rejection_top = upper_wick > body_abs * 2 and upper_wick > lower_wick * 1.5
        is_rejection_bottom = lower_wick > body_abs * 2 and lower_wick > upper_wick * 1.5
        
        # Rejection strength based on wick size relative to ATR
        rejection_strength = 0.0
        if is_rejection_top:
            rejection_strength = min(1.0, upper_wick_atr / 2.0)
        elif is_rejection_bottom:
            rejection_strength = min(1.0, lower_wick_atr / 2.0)
        
        # Create CandleData object
        candle = CandleData(
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timestamp=timestamp,
            index=idx,
            body=close_price - open_price,
            body_abs=body_abs,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            range=range_candle,
            close_position=close_position,
            body_atr=body_atr,
            upper_wick_atr=upper_wick_atr,
            lower_wick_atr=lower_wick_atr,
            range_atr=range_atr,
            is_bullish=is_bullish,
            is_bearish=is_bearish,
            is_doji=is_doji,
            candle_type=candle_type,
            strength=strength,
            conviction=conviction,
            body_ratio=body_ratio,
            wick_top_ratio=wick_top_ratio,
            wick_bottom_ratio=wick_bottom_ratio,
            volume_ratio=volume_ratio,
            is_rejection_top=is_rejection_top,
            is_rejection_bottom=is_rejection_bottom,
            rejection_strength=rejection_strength,
        )
        
        return candle
    
    def analyze_all_candles(self, df: pd.DataFrame) -> List[CandleData]:
        """
        Analyze all candles in DataFrame (vectorized)
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            List of CandleData objects for all candles
        """
        if df is None or df.empty or len(df) < 5:
            return []
        
        try:
            # Pre-calculate ATR and volume MA
            atr_array = self.calculate_atr(df)
            vol_ma_array = self.calculate_volume_ma(df)
            
            # Analyze each candle
            candles = []
            for i in range(len(df)):
                candle = self.analyze_candle(df, i, atr_array, vol_ma_array)
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            print(f"Error analyzing candles: {e}")
            return []
    
    def get_latest_candle(self, df: pd.DataFrame) -> Optional[CandleData]:
        """
        Analyze only the latest candle (for real-time analysis)
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            CandleData for the latest candle, or None if error
        """
        if df is None or df.empty or len(df) < 5:
            return None
        
        try:
            atr_array = self.calculate_atr(df)
            vol_ma_array = self.calculate_volume_ma(df)
            return self.analyze_candle(df, len(df) - 1, atr_array, vol_ma_array)
            
        except Exception as e:
            print(f"Error getting latest candle: {e}")
            return None
    
    def get_candle_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of recent candles
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with summary statistics
        """
        candles = self.analyze_all_candles(df)
        
        if not candles:
            return {}
        
        recent_candles = candles[-20:] if len(candles) >= 20 else candles
        
        # Count by type
        type_counts = {}
        for c in recent_candles:
            type_counts[c.candle_type] = type_counts.get(c.candle_type, 0) + 1
        
        # Average strength
        avg_strength = np.mean([c.strength for c in recent_candles])
        
        # Bullish/Bearish count
        bullish_count = sum(1 for c in recent_candles if c.is_bullish)
        bearish_count = sum(1 for c in recent_candles if c.is_bearish)
        
        # Rejection detection
        rejection_top_count = sum(1 for c in recent_candles if c.is_rejection_top)
        rejection_bottom_count = sum(1 for c in recent_candles if c.is_rejection_bottom)
        
        # Volume analysis
        avg_volume_ratio = np.mean([c.volume_ratio for c in recent_candles])
        
        return {
            'total_candles': len(candles),
            'recent_candles': len(recent_candles),
            'type_counts': type_counts,
            'avg_strength': round(avg_strength, 3),
            'bullish_ratio': round(bullish_count / len(recent_candles), 3) if recent_candles else 0,
            'bearish_ratio': round(bearish_count / len(recent_candles), 3) if recent_candles else 0,
            'rejection_top_count': rejection_top_count,
            'rejection_bottom_count': rejection_bottom_count,
            'avg_volume_ratio': round(avg_volume_ratio, 2),
            'latest': candles[-1].to_dict() if candles else None,
        }
    
    def is_strong_candle(self, candle: CandleData) -> bool:
        """
        Check if a candle is strong (high conviction)
        
        Args:
            candle: CandleData object
        
        Returns:
            True if strong or extreme conviction
        """
        return candle.conviction in ['strong', 'extreme']
    
    def is_weak_candle(self, candle: CandleData) -> bool:
        """
        Check if a candle is weak (low conviction)
        
        Args:
            candle: CandleData object
        
        Returns:
            True if weak conviction
        """
        return candle.conviction == 'weak'
    
    def get_candle_direction(self, candle: CandleData) -> str:
        """
        Get directional bias of a candle
        
        Args:
            candle: CandleData object
        
        Returns:
            'BULL', 'BEAR', or 'NEUTRAL'
        """
        if candle.is_bullish:
            return 'BULL'
        elif candle.is_bearish:
            return 'BEAR'
        else:
            return 'NEUTRAL'


# ==================== CONVENIENCE FUNCTIONS ====================

def analyze_candles(df: pd.DataFrame) -> List[CandleData]:
    """
    Convenience function to analyze all candles
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        List of CandleData objects
    """
    analyzer = CandleAnalyzer()
    return analyzer.analyze_all_candles(df)


def get_latest_candle(df: pd.DataFrame) -> Optional[CandleData]:
    """
    Convenience function to get latest candle
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        CandleData for latest candle
    """
    analyzer = CandleAnalyzer()
    return analyzer.get_latest_candle(df)


def get_candle_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to get candle summary
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Summary dictionary
    """
    analyzer = CandleAnalyzer()
    return analyzer.get_candle_summary(df)


# ==================== TEST EXAMPLE ====================

if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    # Generate realistic price movement
    np.random.seed(42)
    base_price = 50000
    prices = [base_price]
    for i in range(99):
        change = np.random.randn() * 50
        prices.append(prices[-1] + change)
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        open_price = close - np.random.randn() * 20
        high = max(open_price, close) + abs(np.random.randn() * 30)
        low = min(open_price, close) - abs(np.random.randn() * 30)
        volume = abs(np.random.randn() * 10000) + 5000
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Analyze
    analyzer = CandleAnalyzer()
    candles = analyzer.analyze_all_candles(df)
    
    print(f"Analyzed {len(candles)} candles")
    print(f"Latest candle: {candles[-1].to_dict()}")
    print(f"Summary: {analyzer.get_candle_summary(df)}")