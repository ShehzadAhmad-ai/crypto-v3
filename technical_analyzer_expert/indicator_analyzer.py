# indicator_analyzer.py - Complete Indicator Analysis Engine
"""
Indicator Analysis Expert - Wrapper for 50+ Technical Indicators
- Calculates and analyzes all indicators from technical_analyzer.py
- Detects direction (rising/falling) for each indicator
- Generates IndicatorSignal objects for each indicator
- Aggregates scores by category (momentum, trend, volatility, volume)
- All thresholds from ta_config.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import configuration and core classes
from .ta_config import *
from .ta_core import IndicatorSignal, IndicatorSignalType, CategoryScore

# Import existing technical_analyzer for indicator calculations
try:
    from .technical_analyzer import TechnicalAnalyzer
    TECHNICAL_ANALYZER_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYZER_AVAILABLE = False
    import logging
    log = logging.getLogger(__name__)

# Try to import logger
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class IndicatorAnalyzer:
    """
    Complete Indicator Analysis Engine
    Wraps TechnicalAnalyzer and adds direction detection, scoring, and aggregation
    """
    
    def __init__(self):
        """Initialize the indicator analyzer"""
        self.technical_analyzer = None
        if TECHNICAL_ANALYZER_AVAILABLE:
            self.technical_analyzer = TechnicalAnalyzer()
            log.info("IndicatorAnalyzer initialized with TechnicalAnalyzer")
        else:
            log.warning("TechnicalAnalyzer not available, using fallback calculations")
        
        # Cache for indicator values
        self.last_calculated_df = None
        self.last_results = {}
        
        # Category definitions
        self.momentum_indicators = [
            'rsi', 'stoch_k', 'stoch_d', 'stoch_rsi_k', 'stoch_rsi_d',
            'cci', 'williams_r', 'ultimate_oscillator', 'awesome_oscillator',
            'roc_5', 'roc_10', 'roc_20', 'tsi', 'kst', 'trix', 'mass_index'
        ]
        
        self.trend_indicators = [
            'ema_8', 'ema_21', 'ema_50', 'ema_200', 'macd', 'macd_signal', 'macd_hist',
            'adx', 'adx_pos', 'adx_neg', 'ichimoku_conv', 'ichimoku_base',
            'kst_signal', 'trix_signal', 'vhf', 'rvi', 'coppock'
        ]
        
        self.volatility_indicators = [
            'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'kc_high', 'kc_mid', 'kc_low', 'donchian_high', 'donchian_low',
            'atr_trailing_stop', 'psar'
        ]
        
        self.volume_indicators = [
            'volume_ratio', 'volume_trend', 'obv', 'obv_ma', 'cmf', 'mfi',
            'adl', 'klinger_oscillator', 'klinger_signal', 'chaikin_osc',
            'eom', 'vpt', 'nvi', 'pvi', 'volume_oscillator'
        ]
        
        log.info(f"IndicatorAnalyzer configured with {len(self.momentum_indicators)} momentum, "
                f"{len(self.trend_indicators)} trend, {len(self.volatility_indicators)} volatility, "
                f"{len(self.volume_indicators)} volume indicators")
    
    # ============================================================================
    # MAIN PUBLIC METHODS
    # ============================================================================
    
    def analyze_indicators(self, df: pd.DataFrame, regime_settings: Dict[str, Any] = None,
                           symbol: str = "") -> Tuple[List[IndicatorSignal], Dict[str, CategoryScore]]:
        """
        Analyze all indicators and return signals with category scores
        
        Args:
            df: DataFrame with OHLCV data (will calculate indicators if missing)
            regime_settings: Dynamic settings from market regime
            symbol: Symbol name for logging
        
        Returns:
            Tuple of (list of indicator signals, dict of category scores)
        """
        if df is None or df.empty or len(df) < MIN_DATA_CANDLES:
            log.warning(f"Insufficient data for indicator analysis: {len(df) if df is not None else 0} candles")
            return [], self._empty_category_scores()
        
        try:
            # Step 1: Calculate all indicators using TechnicalAnalyzer
            df_indicators = self._calculate_indicators(df, regime_settings, symbol)
            
            # Step 2: Cache for potential reuse
            self.last_calculated_df = df_indicators
            
            # Step 3: Analyze each indicator category
            all_signals = []
            category_signals = {
                'momentum': [],
                'trend': [],
                'volatility': [],
                'volume': []
            }
            
            # Momentum Indicators
            momentum_signals = self._analyze_momentum_indicators(df_indicators)
            all_signals.extend(momentum_signals)
            category_signals['momentum'] = momentum_signals
            
            # Trend Indicators
            trend_signals = self._analyze_trend_indicators(df_indicators)
            all_signals.extend(trend_signals)
            category_signals['trend'] = trend_signals
            
            # Volatility Indicators
            volatility_signals = self._analyze_volatility_indicators(df_indicators)
            all_signals.extend(volatility_signals)
            category_signals['volatility'] = volatility_signals
            
            # Volume Indicators
            volume_signals = self._analyze_volume_indicators(df_indicators)
            all_signals.extend(volume_signals)
            category_signals['volume'] = volume_signals
            
            # Step 4: Calculate category scores
            category_scores = self._calculate_category_scores(category_signals)
            
            log.debug(f"Indicator analysis complete: {len(all_signals)} signals generated")
            
            return all_signals, category_scores
            
        except Exception as e:
            log.error(f"Error in indicator analysis: {e}")
            return [], self._empty_category_scores()
    
    def get_indicator_value(self, indicator_name: str, df: pd.DataFrame = None) -> Optional[float]:
        """
        Get the latest value of a specific indicator
        
        Args:
            indicator_name: Name of the indicator
            df: Optional DataFrame, uses cached if not provided
        
        Returns:
            Indicator value or None if not found
        """
        try:
            if df is None:
                df = self.last_calculated_df
            
            if df is not None and indicator_name in df.columns:
                return float(df[indicator_name].iloc[-1])
            
            return None
        except Exception:
            return None
    
    # ============================================================================
    # INDICATOR CALCULATION METHODS
    # ============================================================================
    
    def _calculate_indicators(self, df: pd.DataFrame, regime_settings: Dict[str, Any],
                               symbol: str) -> pd.DataFrame:
        """
        Calculate all indicators using TechnicalAnalyzer
        """
        if self.technical_analyzer:
            try:
                # Apply regime settings if provided
                if regime_settings:
                    self.technical_analyzer.set_regime_settings(regime_settings)
                
                # Calculate all indicators
                result_df = self.technical_analyzer.calculate_all_indicators(df, symbol)
                
                if result_df is not None and not result_df.empty:
                    return result_df
                    
            except Exception as e:
                log.error(f"Error in TechnicalAnalyzer: {e}")
        
        # Fallback: calculate basic indicators
        return self._calculate_basic_indicators(df)
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback: Calculate basic indicators from scratch
        """
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMAs
        df['ema_8'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=EMA_MEDIUM, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=EMA_VERY_SLOW, adjust=False).mean()
        
        # MACD
        df['macd'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean() - df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=VOLUME_MA_PERIOD).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # OBV
        df['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
        df['obv_ma'] = df['obv'].rolling(window=OBV_MA_PERIOD).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = df['atr']
        plus_di = 100 * plus_dm.rolling(window=ADX_PERIOD).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=ADX_PERIOD).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=ADX_PERIOD).mean()
        df['adx_pos'] = plus_di
        df['adx_neg'] = minus_di
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=BB_PERIOD).mean()
        bb_std = df['close'].rolling(window=BB_PERIOD).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * BB_STD)
        df['bb_lower'] = df['bb_middle'] - (bb_std * BB_STD)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        return df
    
    # ============================================================================
    # INDICATOR ANALYSIS METHODS
    # ============================================================================
    
    def _analyze_momentum_indicators(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """Analyze all momentum indicators"""
        signals = []
        
        # RSI
        if 'rsi' in df.columns:
            signals.append(self._analyze_rsi(df))
        
        # Stochastic
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            signals.append(self._analyze_stochastic(df))
        
        # CCI
        if 'cci' in df.columns:
            signals.append(self._analyze_cci(df))
        
        # Williams %R
        if 'williams_r' in df.columns:
            signals.append(self._analyze_williams_r(df))
        
        # Ultimate Oscillator
        if 'ultimate_oscillator' in df.columns:
            signals.append(self._analyze_ultimate_oscillator(df))
        
        # Awesome Oscillator
        if 'awesome_oscillator' in df.columns:
            signals.append(self._analyze_awesome_oscillator(df))
        
        # ROC
        if 'roc_5' in df.columns:
            signals.append(self._analyze_roc(df, period=5))
        
        # TSI
        if 'tsi' in df.columns:
            signals.append(self._analyze_tsi(df))
        
        # KST
        if 'kst' in df.columns:
            signals.append(self._analyze_kst(df))
        
        # TRIX
        if 'trix' in df.columns:
            signals.append(self._analyze_trix(df))
        
        return signals
    
    def _analyze_trend_indicators(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """Analyze all trend indicators"""
        signals = []
        
        # EMA Stack
        if all(col in df.columns for col in ['ema_8', 'ema_21', 'ema_50', 'ema_200']):
            signals.append(self._analyze_ema_stack(df))
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            signals.append(self._analyze_macd(df))
        
        # ADX
        if 'adx' in df.columns:
            signals.append(self._analyze_adx(df))
        
        # Ichimoku
        if 'ichimoku_conv' in df.columns and 'ichimoku_base' in df.columns:
            signals.append(self._analyze_ichimoku(df))
        
        # VHF (Vertical Horizontal Filter)
        if 'vhf' in df.columns:
            signals.append(self._analyze_vhf(df))
        
        # RVI (Relative Vigor Index)
        if 'rvi' in df.columns:
            signals.append(self._analyze_rvi(df))
        
        # Coppock Curve
        if 'coppock' in df.columns:
            signals.append(self._analyze_coppock(df))
        
        return signals
    
    def _analyze_volatility_indicators(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """Analyze all volatility indicators"""
        signals = []
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_position', 'bb_width']):
            signals.append(self._analyze_bollinger(df))
        
        # ATR
        if 'atr' in df.columns:
            signals.append(self._analyze_atr(df))
        
        # Keltner Channels
        if all(col in df.columns for col in ['kc_high', 'kc_mid', 'kc_low']):
            signals.append(self._analyze_keltner(df))
        
        # Donchian Channels
        if all(col in df.columns for col in ['donchian_high', 'donchian_low']):
            signals.append(self._analyze_donchian(df))
        
        # Parabolic SAR
        if 'psar' in df.columns:
            signals.append(self._analyze_psar(df))
        
        # ATR Trailing Stop
        if 'atr_trailing_stop' in df.columns:
            signals.append(self._analyze_atr_trailing_stop(df))
        
        return signals
    
    def _analyze_volume_indicators(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """Analyze all volume indicators"""
        signals = []
        
        # Volume Ratio
        if 'volume_ratio' in df.columns:
            signals.append(self._analyze_volume_ratio(df))
        
        # OBV
        if 'obv' in df.columns:
            signals.append(self._analyze_obv(df))
        
        # CMF (Chaikin Money Flow)
        if 'cmf' in df.columns:
            signals.append(self._analyze_cmf(df))
        
        # MFI (Money Flow Index)
        if 'mfi' in df.columns:
            signals.append(self._analyze_mfi(df))
        
        # Klinger Oscillator
        if 'klinger_oscillator' in df.columns:
            signals.append(self._analyze_klinger(df))
        
        # Ease of Movement
        if 'eom' in df.columns:
            signals.append(self._analyze_eom(df))
        
        return signals
    
    # ============================================================================
    # INDIVIDUAL INDICATOR ANALYSIS
    # ============================================================================
    
    def _analyze_rsi(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze RSI with direction detection"""
        rsi = float(df['rsi'].iloc[-1])
        rsi_prev = float(df['rsi'].iloc[-2]) if len(df) >= 2 else rsi
        
        # Direction
        direction = "RISING" if rsi > rsi_prev else "FALLING" if rsi < rsi_prev else "FLAT"
        
        # Signal determination
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        # Check for turning from oversold
        if rsi < RSI_OVERSOLD and direction == "RISING":
            signal = IndicatorSignalType.BULLISH
            strength = 0.8
            reason = f"RSI turning up from oversold: {rsi:.1f}"
        elif rsi < RSI_EXTREME_OVERSOLD:
            signal = IndicatorSignalType.OVERSOLD
            strength = 0.7
            reason = f"RSI extreme oversold: {rsi:.1f}"
        elif rsi < RSI_OVERSOLD:
            signal = IndicatorSignalType.BULLISH
            strength = 0.5
            reason = f"RSI oversold: {rsi:.1f}"
        
        # Check for turning from overbought
        elif rsi > RSI_OVERBOUGHT and direction == "FALLING":
            signal = IndicatorSignalType.BEARISH
            strength = 0.8
            reason = f"RSI turning down from overbought: {rsi:.1f}"
        elif rsi > RSI_EXTREME_OVERBOUGHT:
            signal = IndicatorSignalType.OVERBOUGHT
            strength = 0.7
            reason = f"RSI extreme overbought: {rsi:.1f}"
        elif rsi > RSI_OVERBOUGHT:
            signal = IndicatorSignalType.BEARISH
            strength = 0.5
            reason = f"RSI overbought: {rsi:.1f}"
        
        # Bullish zone
        elif rsi > RSI_BULLISH_MIN and direction == "RISING":
            signal = IndicatorSignalType.BULLISH
            strength = 0.4
            reason = f"RSI bullish and rising: {rsi:.1f}"
        elif rsi > RSI_BULLISH_MIN:
            signal = IndicatorSignalType.BULLISH
            strength = 0.3
            reason = f"RSI bullish zone: {rsi:.1f}"
        
        # Bearish zone
        elif rsi < RSI_BEARISH_MAX and direction == "FALLING":
            signal = IndicatorSignalType.BEARISH
            strength = 0.4
            reason = f"RSI bearish and falling: {rsi:.1f}"
        elif rsi < RSI_BEARISH_MAX:
            signal = IndicatorSignalType.BEARISH
            strength = 0.3
            reason = f"RSI bearish zone: {rsi:.1f}"
        
        else:
            reason = f"RSI neutral: {rsi:.1f}"
        
        return IndicatorSignal(
            name="RSI",
            value=rsi,
            signal=signal,
            strength=strength,
            weight=0.20,
            direction=direction,
            reason=reason
        )
    
    def _analyze_macd(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze MACD with histogram acceleration"""
        hist = float(df['macd_hist'].iloc[-1])
        hist_prev = float(df['macd_hist'].iloc[-2]) if len(df) >= 2 else hist
        hist_prev2 = float(df['macd_hist'].iloc[-3]) if len(df) >= 3 else hist
        
        # Direction
        direction = "RISING" if hist > hist_prev else "FALLING" if hist < hist_prev else "FLAT"
        
        # Acceleration (increasing momentum)
        acceleration = (hist - hist_prev) - (hist_prev - hist_prev2) if len(df) >= 3 else 0
        
        # Signal determination
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if hist > 0:
            if direction == "RISING" and acceleration > 0:
                signal = IndicatorSignalType.BULLISH
                strength = 0.9
                reason = "MACD bullish with accelerating momentum"
            elif direction == "RISING":
                signal = IndicatorSignalType.BULLISH
                strength = 0.7
                reason = "MACD bullish histogram rising"
            else:
                signal = IndicatorSignalType.BULLISH
                strength = 0.4
                reason = "MACD positive but losing momentum"
        elif hist < 0:
            if direction == "FALLING" and acceleration < 0:
                signal = IndicatorSignalType.BEARISH
                strength = 0.9
                reason = "MACD bearish with accelerating momentum"
            elif direction == "FALLING":
                signal = IndicatorSignalType.BEARISH
                strength = 0.7
                reason = "MACD bearish histogram falling"
            else:
                signal = IndicatorSignalType.BEARISH
                strength = 0.4
                reason = "MACD negative but recovering"
        else:
            # Check for crossovers
            if hist_prev < 0 and hist > 0:
                signal = IndicatorSignalType.CROSS_ABOVE
                strength = 0.8
                reason = "MACD bullish crossover"
            elif hist_prev > 0 and hist < 0:
                signal = IndicatorSignalType.CROSS_BELOW
                strength = 0.8
                reason = "MACD bearish crossover"
            else:
                reason = "MACD neutral"
        
        return IndicatorSignal(
            name="MACD",
            value=hist,
            signal=signal,
            strength=strength,
            weight=0.25,
            direction=direction,
            reason=reason
        )
    
    def _analyze_ema_stack(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze EMA alignment"""
        price = df['close'].iloc[-1]
        e8 = df['ema_8'].iloc[-1]
        e21 = df['ema_21'].iloc[-1]
        e50 = df['ema_50'].iloc[-1]
        e200 = df['ema_200'].iloc[-1]
        
        # Check alignment
        bullish_aligned = price > e8 > e21 > e50 > e200
        bearish_aligned = price < e8 < e21 < e50 < e200
        weak_bullish = price > e8 and e8 > e21
        weak_bearish = price < e8 and e8 < e21
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if bullish_aligned:
            signal = IndicatorSignalType.BULLISH
            strength = 1.0
            reason = "Strong bullish EMA alignment (8>21>50>200)"
        elif bearish_aligned:
            signal = IndicatorSignalType.BEARISH
            strength = 1.0
            reason = "Strong bearish EMA alignment (8<21<50<200)"
        elif weak_bullish:
            signal = IndicatorSignalType.BULLISH
            strength = 0.6
            reason = "Weak bullish EMA alignment"
        elif weak_bearish:
            signal = IndicatorSignalType.BEARISH
            strength = 0.6
            reason = "Weak bearish EMA alignment"
        else:
            reason = "No clear EMA alignment"
        
        return IndicatorSignal(
            name="EMA Stack",
            value=0,
            signal=signal,
            strength=strength,
            weight=0.30,
            direction="ALIGNED" if strength > 0.5 else "MIXED",
            reason=reason
        )
    
    def _analyze_adx(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze ADX trend strength"""
        adx = float(df['adx'].iloc[-1])
        plus_di = float(df['adx_pos'].iloc[-1]) if 'adx_pos' in df.columns else 0
        minus_di = float(df['adx_neg'].iloc[-1]) if 'adx_neg' in df.columns else 0
        
        signal = IndicatorSignalType.NEUTRAL
        strength = min(1.0, adx / 50)
        reason = ""
        
        if adx > ADX_STRONG_TREND:
            if plus_di > minus_di:
                signal = IndicatorSignalType.BULLISH
                reason = f"Strong bullish trend (ADX: {adx:.1f})"
            elif minus_di > plus_di:
                signal = IndicatorSignalType.BEARISH
                reason = f"Strong bearish trend (ADX: {adx:.1f})"
            else:
                reason = f"Strong trend, direction unclear (ADX: {adx:.1f})"
        elif adx > ADX_TRENDING:
            if plus_di > minus_di:
                signal = IndicatorSignalType.BULLISH
                reason = f"Bullish trend developing (ADX: {adx:.1f})"
            elif minus_di > plus_di:
                signal = IndicatorSignalType.BEARISH
                reason = f"Bearish trend developing (ADX: {adx:.1f})"
            else:
                reason = f"Trending but direction unclear (ADX: {adx:.1f})"
        elif adx < ADX_WEAK_TREND:
            signal = IndicatorSignalType.NEUTRAL
            reason = f"No clear trend, ranging market (ADX: {adx:.1f})"
        else:
            reason = f"Weak trend (ADX: {adx:.1f})"
        
        return IndicatorSignal(
            name="ADX",
            value=adx,
            signal=signal,
            strength=strength,
            weight=0.25,
            direction="TRENDING" if adx > ADX_TRENDING else "RANGING",
            reason=reason
        )
    
    def _analyze_bollinger(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Bollinger Bands position and squeeze"""
        position = float(df['bb_position'].iloc[-1])
        width = float(df['bb_width'].iloc[-1])
        
        # Squeeze detection
        is_squeeze = width < BB_SQUEEZE_THRESHOLD
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if is_squeeze:
            signal = IndicatorSignalType.SQUEEZE
            strength = 0.7
            reason = f"Bollinger squeeze detected (width: {width:.3f})"
        elif position < BB_DISCOUNT_ZONE:
            signal = IndicatorSignalType.BULLISH
            strength = 0.8 - (position / BB_DISCOUNT_ZONE) * 0.3
            reason = f"Price at discount zone (position: {position:.2f})"
        elif position > BB_PREMIUM_ZONE:
            signal = IndicatorSignalType.BEARISH
            strength = 0.8 - ((1 - position) / (1 - BB_PREMIUM_ZONE)) * 0.3
            reason = f"Price at premium zone (position: {position:.2f})"
        elif position < BB_EXTREME_DISCOUNT:
            signal = IndicatorSignalType.BULLISH
            strength = 1.0
            reason = "Price at extreme discount - strong buy signal"
        elif position > BB_EXTREME_PREMIUM:
            signal = IndicatorSignalType.BEARISH
            strength = 1.0
            reason = "Price at extreme premium - strong sell signal"
        else:
            reason = f"Price in neutral zone (position: {position:.2f})"
        
        return IndicatorSignal(
            name="Bollinger Bands",
            value=position,
            signal=signal,
            strength=strength,
            weight=0.20,
            direction="SQUEEZE" if is_squeeze else "EXPANDING",
            reason=reason
        )
    
    def _analyze_volume_ratio(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze volume ratio"""
        vol_ratio = float(df['volume_ratio'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if vol_ratio > VOLUME_EXTREME_SPIKE:
            signal = IndicatorSignalType.BULLISH if df['close'].iloc[-1] > df['open'].iloc[-1] else IndicatorSignalType.BEARISH
            strength = 0.9
            reason = f"Extreme volume spike: {vol_ratio:.1f}x average"
        elif vol_ratio > VOLUME_SPIKE_THRESHOLD:
            signal = IndicatorSignalType.BULLISH if df['close'].iloc[-1] > df['open'].iloc[-1] else IndicatorSignalType.BEARISH
            strength = 0.7
            reason = f"Volume spike: {vol_ratio:.1f}x average"
        elif vol_ratio > VOLUME_CONFIRMATION:
            signal = IndicatorSignalType.BULLISH if df['close'].iloc[-1] > df['open'].iloc[-1] else IndicatorSignalType.BEARISH
            strength = 0.4
            reason = f"Above average volume: {vol_ratio:.1f}x"
        elif vol_ratio < VOLUME_DRYING_UP:
            signal = IndicatorSignalType.NEUTRAL
            strength = 0.2
            reason = f"Volume drying up: {vol_ratio:.1f}x"
        else:
            reason = f"Normal volume: {vol_ratio:.1f}x"
        
        return IndicatorSignal(
            name="Volume",
            value=vol_ratio,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="HIGH" if vol_ratio > 1 else "LOW",
            reason=reason
        )
    
    # ============================================================================
    # SIMPLIFIED ANALYZERS FOR OTHER INDICATORS
    # ============================================================================
    
    def _analyze_stochastic(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Stochastic Oscillator"""
        k = float(df['stoch_k'].iloc[-1])
        d = float(df['stoch_d'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if k < STOCH_OVERSOLD and k > d:
            signal = IndicatorSignalType.BULLISH
            strength = 0.7
            reason = f"Stochastic oversold crossover (K={k:.1f})"
        elif k > STOCH_OVERBOUGHT and k < d:
            signal = IndicatorSignalType.BEARISH
            strength = 0.7
            reason = f"Stochastic overbought crossover (K={k:.1f})"
        elif k < STOCH_OVERSOLD:
            signal = IndicatorSignalType.BULLISH
            strength = 0.5
            reason = f"Stochastic oversold: {k:.1f}"
        elif k > STOCH_OVERBOUGHT:
            signal = IndicatorSignalType.BEARISH
            strength = 0.5
            reason = f"Stochastic overbought: {k:.1f}"
        else:
            reason = f"Stochastic neutral: K={k:.1f}, D={d:.1f}"
        
        return IndicatorSignal(
            name="Stochastic",
            value=k,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="RISING" if k > d else "FALLING",
            reason=reason
        )
    
    def _analyze_cci(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze CCI"""
        cci = float(df['cci'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if cci < -CCI_OVERSOLD:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, abs(cci) / 200)
            reason = f"CCI oversold: {cci:.1f}"
        elif cci > CCI_OVERBOUGHT:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, cci / 200)
            reason = f"CCI overbought: {cci:.1f}"
        else:
            reason = f"CCI neutral: {cci:.1f}"
        
        return IndicatorSignal(
            name="CCI",
            value=cci,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="RISING" if cci > df['cci'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_williams_r(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Williams %R"""
        wr = float(df['williams_r'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if wr < WILLIAMS_R_OVERSOLD:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, abs(wr + 80) / 20)
            reason = f"Williams %R oversold: {wr:.1f}"
        elif wr > WILLIAMS_R_OVERBOUGHT:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, abs(wr + 20) / 20)
            reason = f"Williams %R overbought: {wr:.1f}"
        else:
            reason = f"Williams %R neutral: {wr:.1f}"
        
        return IndicatorSignal(
            name="Williams %R",
            value=wr,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="RISING" if wr > df['williams_r'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_ultimate_oscillator(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Ultimate Oscillator"""
        uo = float(df['ultimate_oscillator'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if uo < ULTIMATE_OSCILLATOR_OVERSOLD:
            signal = IndicatorSignalType.BULLISH
            strength = 0.7
            reason = f"Ultimate Oscillator oversold: {uo:.1f}"
        elif uo > ULTIMATE_OSCILLATOR_OVERBOUGHT:
            signal = IndicatorSignalType.BEARISH
            strength = 0.7
            reason = f"Ultimate Oscillator overbought: {uo:.1f}"
        else:
            reason = f"Ultimate Oscillator neutral: {uo:.1f}"
        
        return IndicatorSignal(
            name="Ultimate Oscillator",
            value=uo,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="RISING" if uo > df['ultimate_oscillator'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_awesome_oscillator(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Awesome Oscillator"""
        ao = float(df['awesome_oscillator'].iloc[-1])
        ao_prev = float(df['awesome_oscillator'].iloc[-2]) if len(df) >= 2 else ao
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        # Saucer pattern detection
        if ao > 0 and ao > ao_prev:
            signal = IndicatorSignalType.BULLISH
            strength = 0.6
            reason = "Awesome Oscillator bullish and rising"
        elif ao < 0 and ao < ao_prev:
            signal = IndicatorSignalType.BEARISH
            strength = 0.6
            reason = "Awesome Oscillator bearish and falling"
        elif ao > 0:
            signal = IndicatorSignalType.BULLISH
            strength = 0.4
            reason = "Awesome Oscillator positive"
        elif ao < 0:
            signal = IndicatorSignalType.BEARISH
            strength = 0.4
            reason = "Awesome Oscillator negative"
        else:
            reason = "Awesome Oscillator neutral"
        
        return IndicatorSignal(
            name="Awesome Oscillator",
            value=ao,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="RISING" if ao > ao_prev else "FALLING",
            reason=reason
        )
    
    def _analyze_roc(self, df: pd.DataFrame, period: int) -> IndicatorSignal:
        """Analyze Rate of Change"""
        roc_col = f'roc_{period}'
        if roc_col not in df.columns:
            return None
        
        roc = float(df[roc_col].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if roc > ROC_BULLISH_THRESHOLD:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, roc / 0.10)
            reason = f"ROC bullish: {roc:.2%}"
        elif roc < ROC_BEARISH_THRESHOLD:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, abs(roc) / 0.10)
            reason = f"ROC bearish: {roc:.2%}"
        else:
            reason = f"ROC neutral: {roc:.2%}"
        
        return IndicatorSignal(
            name=f"ROC({period})",
            value=roc,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="RISING" if roc > df[roc_col].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_tsi(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze TSI"""
        tsi = float(df['tsi'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if tsi > TSI_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, tsi / 50)
            reason = f"TSI bullish: {tsi:.2f}"
        elif tsi < TSI_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, abs(tsi) / 50)
            reason = f"TSI bearish: {tsi:.2f}"
        else:
            reason = f"TSI neutral: {tsi:.2f}"
        
        return IndicatorSignal(
            name="TSI",
            value=tsi,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="RISING" if tsi > df['tsi'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_kst(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze KST"""
        kst = float(df['kst'].iloc[-1])
        kst_signal = float(df['kst_signal'].iloc[-1]) if 'kst_signal' in df.columns else kst
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if kst > kst_signal and kst > 0:
            signal = IndicatorSignalType.BULLISH
            strength = 0.7
            reason = "KST bullish crossover"
        elif kst < kst_signal and kst < 0:
            signal = IndicatorSignalType.BEARISH
            strength = 0.7
            reason = "KST bearish crossover"
        elif kst > 0:
            signal = IndicatorSignalType.BULLISH
            strength = 0.4
            reason = "KST positive"
        elif kst < 0:
            signal = IndicatorSignalType.BEARISH
            strength = 0.4
            reason = "KST negative"
        else:
            reason = "KST neutral"
        
        return IndicatorSignal(
            name="KST",
            value=kst,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="RISING" if kst > kst_signal else "FALLING",
            reason=reason
        )
    
    def _analyze_trix(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze TRIX"""
        trix = float(df['trix'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if trix > TRIX_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, trix / 5)
            reason = f"TRIX bullish: {trix:.2f}"
        elif trix < TRIX_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, abs(trix) / 5)
            reason = f"TRIX bearish: {trix:.2f}"
        else:
            reason = f"TRIX neutral: {trix:.2f}"
        
        return IndicatorSignal(
            name="TRIX",
            value=trix,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="RISING" if trix > df['trix'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_ichimoku(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Ichimoku Cloud"""
        conv = float(df['ichimoku_conv'].iloc[-1])
        base = float(df['ichimoku_base'].iloc[-1])
        price = df['close'].iloc[-1]
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if price > conv > base:
            signal = IndicatorSignalType.BULLISH
            strength = 0.8
            reason = "Ichimoku bullish: price above conversion and base"
        elif price < conv < base:
            signal = IndicatorSignalType.BEARISH
            strength = 0.8
            reason = "Ichimoku bearish: price below conversion and base"
        elif conv > base:
            signal = IndicatorSignalType.BULLISH
            strength = 0.5
            reason = "Ichimoku bullish twist (conversion above base)"
        elif conv < base:
            signal = IndicatorSignalType.BEARISH
            strength = 0.5
            reason = "Ichimoku bearish twist (conversion below base)"
        else:
            reason = "Ichimoku neutral"
        
        return IndicatorSignal(
            name="Ichimoku",
            value=0,
            signal=signal,
            strength=strength,
            weight=0.20,
            direction="BULLISH" if conv > base else "BEARISH",
            reason=reason
        )
    
    def _analyze_vhf(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Vertical Horizontal Filter"""
        vhf = float(df['vhf'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if vhf > VHF_TRENDING:
            signal = IndicatorSignalType.BULLISH if df['close'].iloc[-1] > df['close'].iloc[-20] else IndicatorSignalType.BEARISH
            strength = min(1.0, vhf)
            reason = f"Trending market detected (VHF: {vhf:.2f})"
        elif vhf < VHF_RANGING:
            signal = IndicatorSignalType.NEUTRAL
            strength = 0.3
            reason = f"Ranging market detected (VHF: {vhf:.2f})"
        else:
            reason = f"Mixed conditions (VHF: {vhf:.2f})"
        
        return IndicatorSignal(
            name="VHF",
            value=vhf,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="TRENDING" if vhf > VHF_TRENDING else "RANGING",
            reason=reason
        )
    
    def _analyze_rvi(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Relative Vigor Index"""
        rvi = float(df['rvi'].iloc[-1])
        rvi_signal = float(df['rvi_signal'].iloc[-1]) if 'rvi_signal' in df.columns else rvi
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if rvi > rvi_signal and rvi > RVI_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = 0.7
            reason = "RVI bullish crossover"
        elif rvi < rvi_signal and rvi < RVI_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = 0.7
            reason = "RVI bearish crossover"
        elif rvi > RVI_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = 0.4
            reason = "RVI bullish zone"
        elif rvi < RVI_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = 0.4
            reason = "RVI bearish zone"
        else:
            reason = "RVI neutral"
        
        return IndicatorSignal(
            name="RVI",
            value=rvi,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="RISING" if rvi > rvi_signal else "FALLING",
            reason=reason
        )
    
    def _analyze_coppock(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Coppock Curve"""
        coppock = float(df['coppock'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if coppock > COPPOCK_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, coppock / 50)
            reason = f"Coppock Curve bullish: {coppock:.2f}"
        elif coppock < COPPOCK_BULLISH:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, abs(coppock) / 50)
            reason = f"Coppock Curve bearish: {coppock:.2f}"
        else:
            reason = f"Coppock Curve neutral: {coppock:.2f}"
        
        return IndicatorSignal(
            name="Coppock",
            value=coppock,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="RISING" if coppock > df['coppock'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_keltner(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Keltner Channels"""
        price = df['close'].iloc[-1]
        kc_high = float(df['kc_high'].iloc[-1])
        kc_low = float(df['kc_low'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if price > kc_high:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, (price - kc_high) / (kc_high + 1e-10) * 10)
            reason = "Price above Keltner Channel - bullish breakout"
        elif price < kc_low:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, (kc_low - price) / (kc_low + 1e-10) * 10)
            reason = "Price below Keltner Channel - bearish breakdown"
        else:
            reason = "Price within Keltner Channel"
        
        return IndicatorSignal(
            name="Keltner Channels",
            value=(price - kc_low) / (kc_high - kc_low + 1e-10),
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="ABOVE" if price > kc_high else "BELOW" if price < kc_low else "INSIDE",
            reason=reason
        )
    
    def _analyze_donchian(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Donchian Channels"""
        price = df['close'].iloc[-1]
        donchian_high = float(df['donchian_high'].iloc[-1])
        donchian_low = float(df['donchian_low'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if price > donchian_high:
            signal = IndicatorSignalType.BULLISH
            strength = 0.9
            reason = "20-period high breakout - bullish signal"
        elif price < donchian_low:
            signal = IndicatorSignalType.BEARISH
            strength = 0.9
            reason = "20-period low breakdown - bearish signal"
        else:
            reason = "Price within Donchian Channel"
        
        return IndicatorSignal(
            name="Donchian Channels",
            value=(price - donchian_low) / (donchian_high - donchian_low + 1e-10),
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="BREAKOUT" if price > donchian_high or price < donchian_low else "INSIDE",
            reason=reason
        )
    
    def _analyze_psar(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Parabolic SAR"""
        psar = float(df['psar'].iloc[-1])
        price = df['close'].iloc[-1]
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if price > psar:
            signal = IndicatorSignalType.BULLISH
            strength = 0.6
            reason = "Price above Parabolic SAR - bullish trend"
        elif price < psar:
            signal = IndicatorSignalType.BEARISH
            strength = 0.6
            reason = "Price below Parabolic SAR - bearish trend"
        else:
            reason = "Price at Parabolic SAR - potential reversal"
        
        return IndicatorSignal(
            name="Parabolic SAR",
            value=psar,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="ABOVE" if price > psar else "BELOW",
            reason=reason
        )
    
    def _analyze_atr_trailing_stop(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze ATR Trailing Stop"""
        atr_stop = float(df['atr_trailing_stop'].iloc[-1])
        price = df['close'].iloc[-1]
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if price > atr_stop:
            signal = IndicatorSignalType.BULLISH
            strength = 0.5
            reason = "Price above ATR trailing stop - bullish momentum"
        else:
            signal = IndicatorSignalType.BEARISH
            strength = 0.5
            reason = "Price below ATR trailing stop - bearish momentum"
        
        return IndicatorSignal(
            name="ATR Trailing Stop",
            value=atr_stop,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="ABOVE" if price > atr_stop else "BELOW",
            reason=reason
        )
    
    def _analyze_obv(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze On-Balance Volume"""
        obv = float(df['obv'].iloc[-1])
        obv_ma = float(df['obv_ma'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if obv > obv_ma and df['close'].iloc[-1] > df['close'].iloc[-2]:
            signal = IndicatorSignalType.BULLISH
            strength = 0.7
            reason = "OBV above MA with price rising - bullish confirmation"
        elif obv < obv_ma and df['close'].iloc[-1] < df['close'].iloc[-2]:
            signal = IndicatorSignalType.BEARISH
            strength = 0.7
            reason = "OBV below MA with price falling - bearish confirmation"
        elif obv > obv_ma:
            signal = IndicatorSignalType.BULLISH
            strength = 0.4
            reason = "OBV above MA - accumulation"
        elif obv < obv_ma:
            signal = IndicatorSignalType.BEARISH
            strength = 0.4
            reason = "OBV below MA - distribution"
        else:
            reason = "OBV neutral"
        
        return IndicatorSignal(
            name="OBV",
            value=obv,
            signal=signal,
            strength=strength,
            weight=0.20,
            direction="RISING" if obv > obv_ma else "FALLING",
            reason=reason
        )
    
    def _analyze_cmf(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Chaikin Money Flow"""
        cmf = float(df['cmf'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if cmf > CMF_STRONG_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = 1.0
            reason = f"Strong buying pressure (CMF: {cmf:.3f})"
        elif cmf > CMF_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = 0.6
            reason = f"Buying pressure (CMF: {cmf:.3f})"
        elif cmf < CMF_STRONG_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = 1.0
            reason = f"Strong selling pressure (CMF: {cmf:.3f})"
        elif cmf < CMF_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = 0.6
            reason = f"Selling pressure (CMF: {cmf:.3f})"
        else:
            reason = f"Money flow neutral (CMF: {cmf:.3f})"
        
        return IndicatorSignal(
            name="CMF",
            value=cmf,
            signal=signal,
            strength=strength,
            weight=0.20,
            direction="RISING" if cmf > df['cmf'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_mfi(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Money Flow Index"""
        mfi = float(df['mfi'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if mfi < MFI_OVERSOLD:
            signal = IndicatorSignalType.BULLISH
            strength = 0.7
            reason = f"MFI oversold: {mfi:.1f}"
        elif mfi > MFI_OVERBOUGHT:
            signal = IndicatorSignalType.BEARISH
            strength = 0.7
            reason = f"MFI overbought: {mfi:.1f}"
        elif mfi < MFI_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = 0.4
            reason = f"MFI bullish zone: {mfi:.1f}"
        elif mfi > MFI_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = 0.4
            reason = f"MFI bearish zone: {mfi:.1f}"
        else:
            reason = f"MFI neutral: {mfi:.1f}"
        
        return IndicatorSignal(
            name="MFI",
            value=mfi,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="RISING" if mfi > df['mfi'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    def _analyze_klinger(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Klinger Oscillator"""
        klinger = float(df['klinger_oscillator'].iloc[-1])
        klinger_signal = float(df['klinger_signal'].iloc[-1]) if 'klinger_signal' in df.columns else klinger
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if klinger > klinger_signal and klinger > 0:
            signal = IndicatorSignalType.BULLISH
            strength = 0.7
            reason = "Klinger bullish crossover"
        elif klinger < klinger_signal and klinger < 0:
            signal = IndicatorSignalType.BEARISH
            strength = 0.7
            reason = "Klinger bearish crossover"
        elif klinger > 0:
            signal = IndicatorSignalType.BULLISH
            strength = 0.4
            reason = "Klinger positive"
        elif klinger < 0:
            signal = IndicatorSignalType.BEARISH
            strength = 0.4
            reason = "Klinger negative"
        else:
            reason = "Klinger neutral"
        
        return IndicatorSignal(
            name="Klinger",
            value=klinger,
            signal=signal,
            strength=strength,
            weight=0.15,
            direction="RISING" if klinger > klinger_signal else "FALLING",
            reason=reason
        )
    
    def _analyze_eom(self, df: pd.DataFrame) -> IndicatorSignal:
        """Analyze Ease of Movement"""
        eom = float(df['eom'].iloc[-1])
        
        signal = IndicatorSignalType.NEUTRAL
        strength = 0.0
        reason = ""
        
        if eom > EOM_BULLISH:
            signal = IndicatorSignalType.BULLISH
            strength = min(1.0, eom / 0.5)
            reason = f"Ease of Movement bullish: {eom:.4f}"
        elif eom < EOM_BEARISH:
            signal = IndicatorSignalType.BEARISH
            strength = min(1.0, abs(eom) / 0.5)
            reason = f"Ease of Movement bearish: {eom:.4f}"
        else:
            reason = f"Ease of Movement neutral: {eom:.4f}"
        
        return IndicatorSignal(
            name="EOM",
            value=eom,
            signal=signal,
            strength=strength,
            weight=0.10,
            direction="RISING" if eom > df['eom'].iloc[-2] else "FALLING",
            reason=reason
        )
    
    # ============================================================================
    # CATEGORY SCORING METHODS
    # ============================================================================
    
    def _calculate_category_scores(self, category_signals: Dict[str, List[IndicatorSignal]]) -> Dict[str, CategoryScore]:
        """Calculate scores for each category"""
        category_scores = {}
        
        for category, signals in category_signals.items():
            if not signals:
                category_scores[category] = CategoryScore(
                    name=category,
                    bullish_score=0.0,
                    bearish_score=0.0,
                    net_score=0.0,
                    agreement=0.0,
                    signals=[]
                )
                continue
            
            # Calculate weighted scores
            total_weight = 0.0
            bullish_weighted = 0.0
            bearish_weighted = 0.0
            
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for sig in signals:
                weight = sig.weight
                total_weight += weight
                
                if sig.signal in [IndicatorSignalType.BULLISH, IndicatorSignalType.CROSS_ABOVE,
                                  IndicatorSignalType.DIVERGENCE_BULLISH]:
                    bullish_weighted += weight * sig.strength
                    bullish_count += 1
                elif sig.signal in [IndicatorSignalType.BEARISH, IndicatorSignalType.CROSS_BELOW,
                                    IndicatorSignalType.DIVERGENCE_BEARISH]:
                    bearish_weighted += weight * sig.strength
                    bearish_count += 1
                else:
                    neutral_count += 1
            
            # Normalize scores
            bullish_score = bullish_weighted / total_weight if total_weight > 0 else 0
            bearish_score = bearish_weighted / total_weight if total_weight > 0 else 0
            net_score = bullish_score - bearish_score
            
            # Calculate agreement (how many indicators agree on direction)
            total_signals = bullish_count + bearish_count
            if total_signals > 0:
                agreement = max(bullish_count, bearish_count) / total_signals
            else:
                agreement = 0.0
            
            category_scores[category] = CategoryScore(
                name=category,
                bullish_score=round(bullish_score, 3),
                bearish_score=round(bearish_score, 3),
                net_score=round(net_score, 3),
                agreement=round(agreement, 3),
                signals=signals
            )
        
        return category_scores
    
    def _empty_category_scores(self) -> Dict[str, CategoryScore]:
        """Return empty category scores"""
        categories = ['momentum', 'trend', 'volatility', 'volume']
        return {cat: CategoryScore(name=cat, bullish_score=0, bearish_score=0, net_score=0, agreement=0, signals=[])
                for cat in categories}


# ============================================================================
# SIMPLE WRAPPER FUNCTION
# ============================================================================

def analyze_all_indicators(df: pd.DataFrame, regime_settings: Dict[str, Any] = None,
                           symbol: str = "") -> Tuple[List[IndicatorSignal], Dict[str, CategoryScore]]:
    """
    Simple wrapper function for indicator analysis
    """
    analyzer = IndicatorAnalyzer()
    return analyzer.analyze_indicators(df, regime_settings, symbol)