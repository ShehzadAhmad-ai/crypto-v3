"""
Calculator Module for Strategy Expert
Calculates missing indicators when they are not provided in the input

Features:
- Calculates all common technical indicators
- Caches results for performance
- Handles missing data gracefully
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import OrderedDict


class IndicatorCalculator:
    """
    Calculates technical indicators when missing from input
    
    Usage:
        calculator = IndicatorCalculator()
        indicators = calculator.calculate_missing(df, provided_indicators, required_list)
    """
    
    def __init__(self, cache_size: int = 100):
        """
        Initialize calculator with cache
        
        Args:
            cache_size: Number of indicator sets to cache
        """
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
    def calculate_missing(self, df: pd.DataFrame, 
                         provided: Dict[str, float],
                         required: List[str]) -> Dict[str, float]:
        """
        Calculate missing indicators that are required
        
        Args:
            df: OHLCV DataFrame
            provided: Already calculated indicators
            required: List of indicator names needed
        
        Returns:
            Complete indicators dict with missing ones calculated
        """
        result = provided.copy()
        
        # Check which indicators are missing
        missing = [r for r in required if r not in provided]
        
        if not missing:
            return result
        
        # Calculate each missing indicator
        for indicator in missing:
            try:
                value = self._calculate_indicator(df, indicator)
                if value is not None:
                    result[indicator] = value
            except Exception as e:
                # Log error but continue with default
                print(f"⚠️  Failed to calculate {indicator}: {e}")
                result[indicator] = self._get_default(indicator, df)
        
        return result
    
    def _calculate_indicator(self, df: pd.DataFrame, indicator: str) -> Optional[float]:
        """
        Calculate a single indicator
        
        Args:
            df: OHLCV DataFrame
            indicator: Indicator name
        
        Returns:
            Calculated value or None
        """
        indicator = indicator.lower()
        
        # Basic price
        if indicator == 'price' or indicator == 'close':
            return df['close'].iloc[-1]
        
        if indicator == 'open':
            return df['open'].iloc[-1]
        
        if indicator == 'high':
            return df['high'].iloc[-1]
        
        if indicator == 'low':
            return df['low'].iloc[-1]
        
        # RSI
        if indicator == 'rsi':
            return self._calculate_rsi(df)
        
        if indicator == 'rsi_htf':
            return self._calculate_rsi(df, period=14)  # Can use longer period for HTF
        
        # MACD
        if indicator == 'macd':
            macd, signal, hist = self._calculate_macd(df)
            return macd.iloc[-1] if not isinstance(macd, float) else macd
        
        if indicator == 'macd_signal':
            macd, signal, hist = self._calculate_macd(df)
            return signal.iloc[-1] if not isinstance(signal, float) else signal
        
        if indicator == 'macd_histogram' or indicator == 'macd_hist':
            macd, signal, hist = self._calculate_macd(df)
            return hist.iloc[-1] if not isinstance(hist, float) else hist
        
        # Bollinger Bands
        if indicator == 'bb_upper':
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            return bb_upper.iloc[-1] if not isinstance(bb_upper, float) else bb_upper
        
        if indicator == 'bb_middle':
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            return bb_middle.iloc[-1] if not isinstance(bb_middle, float) else bb_middle
        
        if indicator == 'bb_lower':
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            return bb_lower.iloc[-1] if not isinstance(bb_lower, float) else bb_lower
        
        if indicator == 'bb_width':
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            width = (bb_upper - bb_lower) / bb_middle
            return width.iloc[-1] if not isinstance(width, float) else width
        
        if indicator == 'bb_position':
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            price = df['close'].iloc[-1]
            position = (price - bb_lower) / (bb_upper - bb_lower)
            return position.iloc[-1] if not isinstance(position, float) else position
        
        # ATR
        if indicator == 'atr':
            return self._calculate_atr(df)
        
        if indicator == 'atr_avg':
            atr = self._calculate_atr(df)
            return atr  # ATR is already an average
        
        # ADX
        if indicator == 'adx':
            adx, plus_di, minus_di = self._calculate_adx(df)
            return adx.iloc[-1] if not isinstance(adx, float) else adx
        
        if indicator == 'adx_htf':
            adx, plus_di, minus_di = self._calculate_adx(df, period=20)  # Longer period for HTF
            return adx.iloc[-1] if not isinstance(adx, float) else adx
        
        if indicator == 'dmi_plus':
            adx, plus_di, minus_di = self._calculate_adx(df)
            return plus_di.iloc[-1] if not isinstance(plus_di, float) else plus_di
        
        if indicator == 'dmi_minus':
            adx, plus_di, minus_di = self._calculate_adx(df)
            return minus_di.iloc[-1] if not isinstance(minus_di, float) else minus_di
        
        # EMA
        if indicator == 'ema_8':
            return self._calculate_ema(df, 8)
        
        if indicator == 'ema_21':
            return self._calculate_ema(df, 21)
        
        if indicator == 'ema_50':
            return self._calculate_ema(df, 50)
        
        if indicator == 'ema_200':
            return self._calculate_ema(df, 200)
        
        # Volume
        if indicator == 'volume':
            return df['volume'].iloc[-1] if 'volume' in df.columns else 0
        
        if indicator == 'volume_avg':
            return df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else 0
        
        if indicator == 'volume_ratio':
            volume = df['volume'].iloc[-1] if 'volume' in df.columns else 0
            volume_avg = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else 1
            return volume / volume_avg if volume_avg > 0 else 1.0
        
        # OBV
        if indicator == 'obv':
            return self._calculate_obv(df)
        
        if indicator == 'obv_ma':
            obv = self._calculate_obv(df)
            if isinstance(obv, pd.Series):
                return obv.rolling(20).mean().iloc[-1]
            return obv
        
        # Support/Resistance
        if indicator == 'previous_high':
            return df['high'].iloc[-2] if len(df) >= 2 else df['high'].iloc[-1]
        
        if indicator == 'previous_low':
            return df['low'].iloc[-2] if len(df) >= 2 else df['low'].iloc[-1]
        
        if indicator == 'resistance':
            return self._find_resistance(df)
        
        if indicator == 'support':
            return self._find_support(df)
        
        # Range
        if indicator == 'range_high':
            return df['high'].rolling(20).max().iloc[-1]
        
        if indicator == 'range_low':
            return df['low'].rolling(20).min().iloc[-1]
        
        # Smart Money
        if indicator == 'bos':
            return self._detect_bos(df)
        
        if indicator == 'choch':
            return self._detect_choch(df)
        
        # Cumulative Delta
        if indicator == 'cum_buy_delta':
            return self._calculate_cumulative_buy_delta(df)
        
        if indicator == 'cum_sell_delta':
            return self._calculate_cumulative_sell_delta(df)
        
        # Default
        return self._get_default(indicator, df)
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, 
                        slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                                    std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX and DMI"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Exponential Moving Average"""
        ema = df['close'].ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (df['volume'] * ((df['close'] > df['close'].shift()).astype(int) * 2 - 1)).cumsum()
        return obv
    
    def _find_resistance(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Find nearest resistance level"""
        recent_highs = df['high'].tail(lookback).values
        current_price = df['close'].iloc[-1]
        
        # Find peaks
        peaks = []
        for i in range(1, len(recent_highs) - 1):
            if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                peaks.append(recent_highs[i])
        
        # Find nearest resistance above current price
        resistances = [p for p in peaks if p > current_price]
        return min(resistances) if resistances else current_price * 1.02
    
    def _find_support(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Find nearest support level"""
        recent_lows = df['low'].tail(lookback).values
        current_price = df['close'].iloc[-1]
        
        # Find troughs
        troughs = []
        for i in range(1, len(recent_lows) - 1):
            if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                troughs.append(recent_lows[i])
        
        # Find nearest support below current price
        supports = [t for t in troughs if t < current_price]
        return max(supports) if supports else current_price * 0.98
    
    def _detect_bos(self, df: pd.DataFrame, lookback: int = 10) -> bool:
        """Detect Break of Structure"""
        if len(df) < lookback:
            return False
        
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_close = df['close'].iloc[-1]
        
        # BOS occurs when price breaks recent high or low
        if current_close > recent_high:
            return True
        if current_close < recent_low:
            return True
        
        return False
    
    def _detect_choch(self, df: pd.DataFrame, lookback: int = 10) -> bool:
        """Detect Change of Character"""
        if len(df) < lookback * 2:
            return False
        
        # Check for higher highs and higher lows pattern reversal
        highs = df['high'].tail(lookback).values
        lows = df['low'].tail(lookback).values
        
        # Simple detection: if last low is lower than previous low after a series of higher lows
        if len(lows) >= 3:
            if lows[-1] < lows[-2] and lows[-2] > lows[-3]:
                return True
        
        return False
    
    def _calculate_cumulative_buy_delta(self, df: pd.DataFrame) -> float:
        """Calculate cumulative buy delta (simplified)"""
        if 'volume' not in df.columns:
            return 0
        
        # Simplified: volume * (close - open) / (high - low) approximation
        price_range = df['high'] - df['low']
        buy_pressure = df['volume'] * ((df['close'] - df['low']) / price_range)
        
        return buy_pressure.tail(20).sum()
    
    def _calculate_cumulative_sell_delta(self, df: pd.DataFrame) -> float:
        """Calculate cumulative sell delta (simplified)"""
        if 'volume' not in df.columns:
            return 0
        
        price_range = df['high'] - df['low']
        sell_pressure = df['volume'] * ((df['high'] - df['close']) / price_range)
        
        return sell_pressure.tail(20).sum()
    
    def _get_default(self, indicator: str, df: pd.DataFrame) -> float:
        """Get default value for missing indicator"""
        defaults = {
            'rsi': 50,
            'macd': 0,
            'macd_histogram': 0,
            'adx': 20,
            'atr': df['close'].iloc[-1] * 0.01 if len(df) > 0 else 1,
            'volume_ratio': 1,
            'obv': 0,
            'bos': 0,
            'choch': 0,
        }
        
        return defaults.get(indicator.lower(), 0)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def quick_calculate(df: pd.DataFrame, indicator: str) -> float:
    """Quick one-off indicator calculation"""
    calc = IndicatorCalculator()
    return calc._calculate_indicator(df, indicator)


def calculate_all_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate all common indicators at once"""
    calc = IndicatorCalculator()
    
    indicators = {}
    
    # Price
    indicators['price'] = df['close'].iloc[-1]
    indicators['open'] = df['open'].iloc[-1]
    indicators['high'] = df['high'].iloc[-1]
    indicators['low'] = df['low'].iloc[-1]
    
    # Trend
    indicators['rsi'] = calc._calculate_rsi(df)
    indicators['adx'] = calc._calculate_adx(df)[0].iloc[-1]
    indicators['ema_8'] = calc._calculate_ema(df, 8)
    indicators['ema_21'] = calc._calculate_ema(df, 21)
    indicators['ema_50'] = calc._calculate_ema(df, 50)
    indicators['ema_200'] = calc._calculate_ema(df, 200)
    
    # Volatility
    indicators['atr'] = calc._calculate_atr(df)
    bb_upper, bb_middle, bb_lower = calc._calculate_bollinger_bands(df)
    indicators['bb_upper'] = bb_upper.iloc[-1]
    indicators['bb_middle'] = bb_middle.iloc[-1]
    indicators['bb_lower'] = bb_lower.iloc[-1]
    indicators['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
    
    # Volume
    if 'volume' in df.columns:
        indicators['volume'] = df['volume'].iloc[-1]
        indicators['volume_avg'] = df['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = indicators['volume'] / indicators['volume_avg'] if indicators['volume_avg'] > 0 else 1
    
    # Support/Resistance
    indicators['resistance'] = calc._find_resistance(df)
    indicators['support'] = calc._find_support(df)
    
    return indicators


if __name__ == "__main__":
    # Test the calculator
    print("Testing Indicator Calculator...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Calculate all indicators
    indicators = calculate_all_indicators(df)
    
    print(f"✅ Calculated {len(indicators)} indicators:")
    for name, value in list(indicators.items())[:10]:
        print(f"   {name}: {value:.4f}")
    print(f"   ... and {len(indicators) - 10} more")