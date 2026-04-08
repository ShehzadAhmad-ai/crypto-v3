# technical_analyzer.py - ENHANCED FOR HUMAN-LIKE ANALYSIS
"""
Technical Analyzer - Enhanced Version
- Calculates 50+ technical indicators
- Dynamic indicator settings based on market regime
- Indicator divergence detection
- Indicator state tracking (rising/falling, crossing)
- Scoring functions for each indicator
- All periods configurable from config.py
"""
import pandas as pd
import numpy as np
import ta
from scipy.stats import binned_statistic
from scipy.signal import argrelextrema
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from config import Config
from logger import log

EPSILON = 1e-10


class IndicatorState(Enum):
    """Indicator state classification"""
    RISING = "RISING"
    FALLING = "FALLING"
    CROSSED_ABOVE = "CROSSED_ABOVE"
    CROSSED_BELOW = "CROSSED_BELOW"
    OVERBOUGHT = "OVERBOUGHT"
    OVERSOLD = "OVERSOLD"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    DIVERGENCE_BULLISH = "DIVERGENCE_BULLISH"
    DIVERGENCE_BEARISH = "DIVERGENCE_BEARISH"


@dataclass
class IndicatorResult:
    """Complete indicator analysis result"""
    name: str
    value: float
    state: IndicatorState
    score: float  # -1 to 1, bullish = positive, bearish = negative
    strength: float  # 0-1, how strong the signal is
    reason: str
    details: Dict[str, Any]


class TechnicalAnalyzer:
    """
    Enhanced Technical Analyzer with dynamic settings and indicator intelligence
    """
    
    def __init__(self):
        # Base periods from config
        self.base_periods = {
            'rsi': getattr(Config, 'RSI_PERIOD', 14),
            'macd_fast': getattr(Config, 'MACD_FAST', 12),
            'macd_slow': getattr(Config, 'MACD_SLOW', 26),
            'macd_signal': getattr(Config, 'MACD_SIGNAL', 9),
            'ema_fast': getattr(Config, 'EMA_FAST', 8),
            'ema_medium': getattr(Config, 'EMA_MEDIUM', 21),
            'ema_slow': getattr(Config, 'EMA_SLOW', 50),
            'ema_very_slow': getattr(Config, 'EMA_VERY_SLOW', 200),
            'bb_period': getattr(Config, 'BB_PERIOD', 20),
            'bb_std': getattr(Config, 'BB_STD', 2.0),
            'atr_period': getattr(Config, 'ATR_WINDOW', 14),
            'adx_period': getattr(Config, 'ADX_PERIOD', 14),
            'volume_ma': getattr(Config, 'VOLUME_MA_PERIOD', 20),
            'atr_multiplier_stop': getattr(Config, 'ATR_STOP_MULTIPLIER', 1.5),
            'atr_tp_multiplier': getattr(Config, 'ATR_TP_MULTIPLIER', 2.0),
        }
        
        # Current active settings (updated by set_regime_settings)
        self.active_periods = self.base_periods.copy()
        
        log.info("TechnicalAnalyzer initialized with configurable periods")
    
    def set_regime_settings(self, regime_settings: Dict[str, Any]):
        """
        Set dynamic indicator settings based on market regime
        Called by TechnicalPipeline after regime detection
        """
        if regime_settings:
            self.active_periods.update(regime_settings)
            log.debug(f"Updated indicator settings: {self.active_periods}")
    
    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str = '') -> pd.DataFrame:
        """
        Calculate all indicators with current active settings
        """
        if df is None or df.empty or len(df) < 10:
            return pd.DataFrame() if df is None else df

        df = df.copy()
        
        # ===== CORE INDICATORS WITH DYNAMIC SETTINGS =====
        
        # ATR
        try:
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], 
                window=self.active_periods['atr_period']
            ).average_true_range()
        except Exception:
            df['atr'] = df['close'] * 0.02
        
        # ATR Average
        try:
            df['atr_avg'] = df['atr'].rolling(20).mean()
        except Exception:
            df['atr_avg'] = df['atr']

        # RSI - with dynamic period
        try:
            df['rsi'] = ta.momentum.RSIIndicator(
                df['close'], 
                window=self.active_periods['rsi']
            ).rsi()
        except Exception:
            df['rsi'] = 50.0

        # MACD - with dynamic settings
        if len(df) >= self.active_periods['macd_slow'] + 5:
            try:
                macd = ta.trend.MACD(
                    df['close'],
                    window_fast=self.active_periods['macd_fast'],
                    window_slow=self.active_periods['macd_slow'],
                    window_sign=self.active_periods['macd_signal']
                )
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_hist'] = macd.macd_diff()
            except Exception:
                df['macd'] = 0.0
                df['macd_signal'] = 0.0
                df['macd_hist'] = 0.0
        else:
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_hist'] = 0.0

        # EMAs - with dynamic settings
        for span, name in [
            (self.active_periods['ema_fast'], 'ema_8'),
            (self.active_periods['ema_medium'], 'ema_21'),
            (self.active_periods['ema_slow'], 'ema_50'),
            (self.active_periods['ema_very_slow'], 'ema_200')
        ]:
            try:
                df[name] = df['close'].ewm(span=span, adjust=False).mean()
            except Exception:
                df[name] = df['close']

        # Bollinger Bands - with dynamic settings
        if len(df) >= self.active_periods['bb_period'] + 10:
            try:
                bb = ta.volatility.BollingerBands(
                    df['close'],
                    window=self.active_periods['bb_period'],
                    window_dev=self.active_periods['bb_std']
                )
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_middle'] = bb.bollinger_mavg()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'].replace(0, np.nan) + EPSILON)
                # BB Position (0-1 where 0 is lower band, 1 is upper band)
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + EPSILON)
            except Exception:
                df['bb_upper'] = np.nan
                df['bb_middle'] = np.nan
                df['bb_lower'] = np.nan
                df['bb_width'] = np.nan
                df['bb_position'] = np.nan
        else:
            df['bb_upper'] = np.nan
            df['bb_middle'] = np.nan
            df['bb_lower'] = np.nan
            df['bb_width'] = np.nan
            df['bb_position'] = np.nan

        # Volume SMA and ratio
        df['volume_sma'] = df['volume'].rolling(self.active_periods['volume_ma']).mean().ffill().fillna(1.0)
        df['volume_ratio'] = (df['volume'] / df['volume_sma']).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        df['volume_avg'] = df['volume_sma']  # Alias for compatibility

        # VWAP
        try:
            typical = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (typical * df['volume']).cumsum() / df['volume'].cumsum().replace(0, np.nan)
        except Exception:
            df['vwap'] = df['close']

        # Range High/Low
        try:
            df['range_high'] = df['high'].rolling(20).max()
            df['range_low'] = df['low'].rolling(20).min()
        except Exception:
            df['range_high'] = df['high']
            df['range_low'] = df['low']
        
        # Support/Resistance levels
        try:
            recent_highs = df['high'].rolling(20).max()
            recent_lows = df['low'].rolling(20).min()
            df['resistance'] = recent_highs
            df['support'] = recent_lows
        except Exception:
            df['resistance'] = df['high']
            df['support'] = df['low']
        
        # ROC
        df['roc_5'] = df['close'].pct_change(5).fillna(0)
        df['roc_10'] = df['close'].pct_change(10).fillna(0)
        df['roc_20'] = df['close'].pct_change(20).fillna(0)

        # ADX - with dynamic period
        if len(df) >= self.active_periods['adx_period'] + 5:
            try:
                adx = ta.trend.ADXIndicator(
                    df['high'], df['low'], df['close'],
                    window=self.active_periods['adx_period']
                )
                df['adx'] = adx.adx()
                df['adx_pos'] = adx.adx_pos()
                df['adx_neg'] = adx.adx_neg()
            except Exception:
                df['adx'] = 20.0
                df['adx_pos'] = 20.0
                df['adx_neg'] = 20.0
        else:
            df['adx'] = 20.0
            df['adx_pos'] = 20.0
            df['adx_neg'] = 20.0

        # ===== TREND & VOLATILITY INDICATORS =====

        # Ichimoku Cloud - requires 52 candles
        if len(df) >= 72:
            try:
                ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
                df['ichimoku_base'] = ichimoku.ichimoku_base_line()
                df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
                df['ichimoku_a'] = ichimoku.ichimoku_a()
                df['ichimoku_b'] = ichimoku.ichimoku_b()
            except Exception:
                df['ichimoku_base'] = np.nan
                df['ichimoku_conv'] = np.nan
                df['ichimoku_a'] = np.nan
                df['ichimoku_b'] = np.nan
        else:
            df['ichimoku_base'] = np.nan
            df['ichimoku_conv'] = np.nan
            df['ichimoku_a'] = np.nan
            df['ichimoku_b'] = np.nan

        # Keltner Channels - requires 20 candles
        if len(df) >= 50:
            try:
                kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20, window_atr=10)
                df['kc_high'] = kc.keltner_channel_hband()
                df['kc_mid'] = kc.keltner_channel_mband()
                df['kc_low'] = kc.keltner_channel_lband()
            except Exception:
                df['kc_high'] = np.nan
                df['kc_mid'] = np.nan
                df['kc_low'] = np.nan
        else:
            df['kc_high'] = np.nan
            df['kc_mid'] = np.nan
            df['kc_low'] = np.nan

        # Donchian Channels - requires 20 candles
        if len(df) >= 60:
            try:
                donchian_high = df['high'].rolling(window=20).max()
                donchian_low = df['low'].rolling(window=20).min()
                df['donchian_high'] = donchian_high
                df['donchian_low'] = donchian_low
                df['donchian_mid'] = (donchian_high + donchian_low) / 2.0
            except Exception:
                df['donchian_high'] = np.nan
                df['donchian_low'] = np.nan
                df['donchian_mid'] = np.nan
        else:
            df['donchian_high'] = np.nan
            df['donchian_low'] = np.nan
            df['donchian_mid'] = np.nan

        # Parabolic SAR
        try:
            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'], iaf=0.02, maxaf=0.2)
            df['psar'] = psar.psar()
            df['psar_ep'] = psar.psar_ep()
            df['psar_af'] = psar.psar_af()
        except Exception:
            df['psar'] = np.nan
            df['psar_ep'] = np.nan
            df['psar_af'] = np.nan

        # ATR Trailing Stop
        try:
            atr = df['atr'].fillna(method='ffill').fillna(df['close'] * 0.02)
            multiplier = 3.0
            hl2 = (df['high'] + df['low']) / 2.0
            df['atr_trailing_stop'] = hl2 - (atr * multiplier)
        except Exception:
            df['atr_trailing_stop'] = np.nan

        # Elder Ray Index
        try:
            ema_13 = df['close'].ewm(span=13, adjust=False).mean()
            df['elder_bull_power'] = df['high'] - ema_13
            df['elder_bear_power'] = df['low'] - ema_13
        except Exception:
            df['elder_bull_power'] = np.nan
            df['elder_bear_power'] = np.nan

        # Vortex Indicator - requires 14 candles
        if len(df) >= 74:
            try:
                vi = ta.trend.VortexIndicator(df['high'], df['low'], df['close'], window=14)
                df['vortex_vi_plus'] = vi.vortex_indicator_pos()
                df['vortex_vi_minus'] = vi.vortex_indicator_neg()
            except Exception:
                df['vortex_vi_plus'] = np.nan
                df['vortex_vi_minus'] = np.nan
        else:
            df['vortex_vi_plus'] = np.nan
            df['vortex_vi_minus'] = np.nan

        # Chande Momentum Oscillator (CMO) - requires 9 candles
        if len(df) >= 97:
            try:
                cmo = ta.momentum.CMOIndicator(df['close'], window=9)
                df['cmo'] = cmo.cmo()
            except Exception:
                df['cmo'] = np.nan
        else:
            df['cmo'] = np.nan

        # ===== VOLUME & MONEY FLOW INDICATORS =====

        # Chaikin Money Flow (CMF) - requires 20 candles
        if len(df) >= 70:
            try:
                cmf = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20)
                df['cmf'] = cmf.chaikin_money_flow()
            except Exception:
                df['cmf'] = 0.0
        else:
            df['cmf'] = 0.0

        # On-Balance Volume (OBV)
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            df['obv'] = obv.on_balance_volume()
            df['obv_ma'] = df['obv'].rolling(20).mean()
        except Exception:
            df['obv'] = df['volume'].cumsum()
            df['obv_ma'] = df['obv']

        # Money Flow Index (MFI) - requires 14 candles
        if len(df) >= 84:
            try:
                mfi = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14)
                df['mfi'] = mfi.money_flow_index()
            except Exception:
                df['mfi'] = 50.0
        else:
            df['mfi'] = 50.0

        # Accumulation/Distribution Line (ADL)
        try:
            ad = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
            df['adl'] = ad.acc_dist_index()
        except Exception:
            df['adl'] = 0.0

        # Klinger Oscillator - requires 55 candles
        if len(df) >= 75:
            try:
                ko = ta.volume.KlingerVolumeOscillator(df['high'], df['low'], df['close'], df['volume'], window1=34, window2=55, window_sign=13)
                df['klinger_oscillator'] = ko.kvo()
                df['klinger_signal'] = ko.kvo_signal()
            except Exception:
                df['klinger_oscillator'] = 0.0
                df['klinger_signal'] = 0.0
        else:
            df['klinger_oscillator'] = 0.0
            df['klinger_signal'] = 0.0

        # Market Facilitation Index
        try:
            mfi_custom = df['high'] - df['low']
            mfi_custom = mfi_custom / (df['volume'].replace(0, 1) + EPSILON)
            df['mfi_custom'] = mfi_custom
        except Exception:
            df['mfi_custom'] = np.nan

        # ===== MOMENTUM INDICATORS =====

        # Commodity Channel Index (CCI) - requires 20 candles
        if len(df) >= 80:
            try:
                cci = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20)
                df['cci'] = cci.cci()
            except Exception:
                df['cci'] = 0.0
        else:
            df['cci'] = 0.0

        # Williams %R - requires 14 candles
        if len(df) >= 84:
            try:
                williams_r = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14)
                df['williams_r'] = williams_r.williams_r()
            except Exception:
                df['williams_r'] = -50.0
        else:
            df['williams_r'] = -50.0

        # Stochastic RSI - requires 14 candles
        if len(df) >= 84:
            try:
                stoch_rsi = ta.momentum.StochasticOscillator(df['close'], window=14, smooth_window=3)
                df['stoch_rsi_k'] = stoch_rsi.stoch()
                df['stoch_rsi_d'] = stoch_rsi.stoch_signal()
            except Exception:
                df['stoch_rsi_k'] = 50.0
                df['stoch_rsi_d'] = 50.0
        else:
            df['stoch_rsi_k'] = 50.0
            df['stoch_rsi_d'] = 50.0

        # Ultimate Oscillator (UO) - requires 28 candles
        if len(df) >= 88:
            try:
                uo = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close'], window1=7, window2=14, window3=28)
                df['ultimate_oscillator'] = uo.ultimate_oscillator()
            except Exception:
                df['ultimate_oscillator'] = 50.0
        else:
            df['ultimate_oscillator'] = 50.0

        # Awesome Oscillator
        try:
            median_price = (df['high'] + df['low']) / 2
            df['awesome_oscillator'] = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        except Exception:
            df['awesome_oscillator'] = 0.0

        # ===== ADVANCED VOLUME =====

        # Volume Profile (POC)
        try:
            window = min(200, len(df))
            prices = df['close'].iloc[-window:]
            volumes = df['volume'].iloc[-window:]
            counts, bin_edges, _ = binned_statistic(prices, volumes, statistic='sum', bins=40)
            max_idx = int(np.nanargmax(counts))
            poc = float((bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0)
            df['vprofile_poc'] = poc
        except Exception:
            df['vprofile_poc'] = np.nan

        # Volume Profile (VAH/VAL)
        try:
            window = min(200, len(df))
            prices = df['close'].iloc[-window:]
            volumes = df['volume'].iloc[-window:]
            
            # Value Area High/Low (68% of volume)
            cumsum_vol = volumes.cumsum()
            total_vol = cumsum_vol.iloc[-1]
            threshold = total_vol * 0.68
            
            sorted_idx = np.argsort(volumes.values)[::-1]
            top_68_idx = sorted_idx[:max(1, len(sorted_idx) // 3)]
            top_prices = prices.iloc[top_68_idx]
            
            df['vprofile_val_high'] = top_prices.max()
            df['vprofile_val_low'] = top_prices.min()
        except Exception:
            df['vprofile_val_high'] = np.nan
            df['vprofile_val_low'] = np.nan

        # ===== ADDITIONAL MOMENTUM =====

        # True Strength Index (TSI) - requires 25 candles
        if len(df) >= 65:
            try:
                tsi = ta.momentum.TSIIndicator(df['close'], window_fast=25, window_slow=13)
                df['tsi'] = tsi.tsi()
            except Exception:
                df['tsi'] = 0.0
        else:
            df['tsi'] = 0.0

        # Know Sure Thing (KST) - requires 30 candles
        if len(df) >= 50:
            try:
                kst = ta.momentum.KSTIndicator(df['close'], roc1=10, roc2=15, roc3=20, roc4=30,
                                                window1=10, window2=10, window3=10, window4=15)
                df['kst'] = kst.kst()
                df['kst_signal'] = kst.kst_sig()
            except Exception:
                df['kst'] = 0.0
                df['kst_signal'] = 0.0
        else:
            df['kst'] = 0.0
            df['kst_signal'] = 0.0

        # ===== NEW INDICATORS ADDED =====

        # Ease of Movement (EOM)
        try:
            distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
            box_ratio = (df['volume'] / 1000000) / (df['high'] - df['low'] + EPSILON)
            df['eom'] = distance / (box_ratio + EPSILON)
            df['eom'] = df['eom'].rolling(14).mean()
        except Exception:
            df['eom'] = 0.0

        # Volume Price Trend (VPT)
        try:
            vpt = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
            df['vpt'] = vpt.cumsum()
        except Exception:
            df['vpt'] = 0.0

        # Negative Volume Index (NVI)
        try:
            nvi = [1000]
            for i in range(1, len(df)):
                if df['volume'].iloc[i] < df['volume'].iloc[i-1]:
                    nvi.append(nvi[-1] + ((df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]) * nvi[-1])
                else:
                    nvi.append(nvi[-1])
            df['nvi'] = nvi
        except Exception:
            df['nvi'] = 1000.0

        # Positive Volume Index (PVI)
        try:
            pvi = [1000]
            for i in range(1, len(df)):
                if df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                    pvi.append(pvi[-1] + ((df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]) * pvi[-1])
                else:
                    pvi.append(pvi[-1])
            df['pvi'] = pvi
        except Exception:
            df['pvi'] = 1000.0

        # Volume Oscillator
        try:
            volume_fast = df['volume'].rolling(5).mean()
            volume_slow = df['volume'].rolling(20).mean()
            df['volume_oscillator'] = ((volume_fast - volume_slow) / (volume_slow + EPSILON)) * 100
        except Exception:
            df['volume_oscillator'] = 0.0

        # Chaikin Oscillator
        try:
            adl = df['adl'] if 'adl' in df else 0
            df['chaikin_osc'] = adl.rolling(3).mean() - adl.rolling(10).mean()
        except Exception:
            df['chaikin_osc'] = 0.0

        # Coppock Curve
        try:
            roc_14 = df['close'].pct_change(14) * 100
            roc_11 = df['close'].pct_change(11) * 100
            wma = (roc_14 + roc_11).rolling(10).mean()
            df['coppock'] = wma
        except Exception:
            df['coppock'] = 0.0

        # TRIX
        try:
            triple_ema = df['close'].ewm(span=15).mean().ewm(span=15).mean().ewm(span=15).mean()
            df['trix'] = triple_ema.pct_change() * 100
        except Exception:
            df['trix'] = 0.0

        # Mass Index
        try:
            high_low_range = df['high'] - df['low']
            ema9 = high_low_range.ewm(span=9).mean()
            ema9_ema9 = ema9.ewm(span=9).mean()
            mass_index = (ema9 / (ema9_ema9 + EPSILON)).rolling(25).sum()
            df['mass_index'] = mass_index
        except Exception:
            df['mass_index'] = 0.0

        # Vertical Horizontal Filter (VHF)
        try:
            period = 28
            close = df['close']
            hcp = close.rolling(period).max()
            lcp = close.rolling(period).min()
            numerator = hcp - lcp
            denominator = (close - close.shift(1)).abs().rolling(period).sum()
            df['vhf'] = numerator / (denominator + EPSILON)
        except Exception:
            df['vhf'] = 0.0

        # Relative Vigor Index (RVI)
        try:
            close_open = df['close'] - df['open']
            high_low = df['high'] - df['low']
            numerator = close_open.rolling(10).mean()
            denominator = high_low.rolling(10).mean()
            df['rvi'] = numerator / (denominator + EPSILON)
            df['rvi_signal'] = df['rvi'].rolling(4).mean()
        except Exception:
            df['rvi'] = 0.0
            df['rvi_signal'] = 0.0

        # ===== SWING POINTS FOR S/R =====
        
        # Previous High/Low (swing points)
        try:
            window = 5
            df['prev_high'] = df['high'].rolling(window, center=True).apply(
                lambda x: x[len(x)//2] if len(x) > 0 and x[len(x)//2] == max(x) else np.nan, raw=True)
            df['prev_low'] = df['low'].rolling(window, center=True).apply(
                lambda x: x[len(x)//2] if len(x) > 0 and x[len(x)//2] == min(x) else np.nan, raw=True)
            df['previous_high'] = df['prev_high'].ffill()
            df['previous_low'] = df['prev_low'].ffill()
        except Exception:
            df['previous_high'] = df['high'].shift(1).rolling(10).max()
            df['previous_low'] = df['low'].shift(1).rolling(10).min()

        # ===== CUMULATIVE DELTA =====
        try:
            df['delta'] = (df['close'] - df['open']) * df['volume'] / (df['close'] + EPSILON)
            df['cum_buy_delta'] = df[df['delta'] > 0]['delta'].cumsum()
            df['cum_sell_delta'] = df[df['delta'] < 0]['delta'].cumsum()
            df['cum_buy_delta'] = df['cum_buy_delta'].fillna(method='ffill').fillna(0)
            df['cum_sell_delta'] = df['cum_sell_delta'].fillna(method='ffill').fillna(0)
            df['cum_delta'] = df['cum_buy_delta'] - df['cum_sell_delta']
            df['cum_delta_ma'] = df['cum_delta'].rolling(20).mean()
        except Exception:
            df['cum_buy_delta'] = 0
            df['cum_sell_delta'] = 0
            df['cum_delta'] = 0
            df['cum_delta_ma'] = 0

        # ===== HIDDEN SUPPORT/RESISTANCE =====
        try:
            if len(df) > 50:
                price_bins = 20
                price_min = df['low'].min()
                price_max = df['high'].max()
                bin_edges = np.linspace(price_min, price_max, price_bins + 1)
                
                volume_by_price = []
                for i in range(price_bins):
                    mask = (df['close'] >= bin_edges[i]) & (df['close'] < bin_edges[i+1])
                    if mask.any():
                        vol_sum = df.loc[mask, 'volume'].sum()
                        volume_by_price.append((bin_edges[i], vol_sum))
                
                if volume_by_price:
                    volume_by_price.sort(key=lambda x: x[1], reverse=True)
                    if len(volume_by_price) >= 2:
                        df['hidden_support'] = min(volume_by_price[0][0], volume_by_price[1][0])
                        df['hidden_resistance'] = max(volume_by_price[0][0], volume_by_price[1][0])
                    else:
                        df['hidden_support'] = df['low'].rolling(50).min()
                        df['hidden_resistance'] = df['high'].rolling(50).max()
            else:
                df['hidden_support'] = df['low'].rolling(20).min()
                df['hidden_resistance'] = df['high'].rolling(20).max()
        except Exception:
            df['hidden_support'] = df['low'].rolling(20).min()
            df['hidden_resistance'] = df['high'].rolling(20).max()

        # ===== HIGHER TIMEFRAME INDICATORS =====
        try:
            df['rsi_htf'] = ta.momentum.RSIIndicator(df['close'], window=28).rsi()
            if len(df) >= 28:
                adx_htf = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=28)
                df['adx_htf'] = adx_htf.adx()
            else:
                df['adx_htf'] = df.get('adx', 20)
        except Exception:
            df['rsi_htf'] = df.get('rsi', 50)
            df['adx_htf'] = df.get('adx', 20)

        # ===== ADD INDICATOR STATE DETECTION =====
        df = self._add_indicator_states(df)
        
        # ===== ADD DIVERGENCE DETECTION =====
        df = self._add_divergence_detection(df)
        
        # ===== ADD INDICATOR SCORES =====
        df = self._add_indicator_scores(df)
        
        # ==================== FILL NaNs SAFELY ====================
        df = df.ffill().bfill().fillna(0)
        
        return df
    
    # ==================== INDICATOR STATE DETECTION ====================
    
    def _add_indicator_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator state columns (rising/falling, crossing, etc.)
        """
        try:
            # RSI State
            df['rsi_state'] = 'NEUTRAL'
            df['rsi_rising'] = False
            df['rsi_falling'] = False
            
            if len(df) >= 3:
                rsi_rising = df['rsi'] > df['rsi'].shift(1)
                rsi_falling = df['rsi'] < df['rsi'].shift(1)
                df['rsi_rising'] = rsi_rising
                df['rsi_falling'] = rsi_falling
                
                # State classification
                df.loc[df['rsi'] > 70, 'rsi_state'] = 'OVERBOUGHT'
                df.loc[df['rsi'] < 30, 'rsi_state'] = 'OVERSOLD'
                df.loc[(df['rsi'] > 50) & (df['rsi_rising']), 'rsi_state'] = 'BULLISH'
                df.loc[(df['rsi'] < 50) & (df['rsi_falling']), 'rsi_state'] = 'BEARISH'
            
            # MACD State
            df['macd_state'] = 'NEUTRAL'
            df['macd_cross'] = 'NONE'
            df['macd_hist_state'] = 'NEUTRAL'
            df['macd_hist_trend'] = 'NEUTRAL'
            
            if len(df) >= 2:
                # MACD crossing above signal
                macd_cross_above = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
                # MACD crossing below signal
                macd_cross_below = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
                
                df.loc[macd_cross_above, 'macd_cross'] = 'CROSS_ABOVE'
                df.loc[macd_cross_below, 'macd_cross'] = 'CROSS_BELOW'
                
                # State classification
                df.loc[df['macd'] > df['macd_signal'], 'macd_state'] = 'BULLISH'
                df.loc[df['macd'] < df['macd_signal'], 'macd_state'] = 'BEARISH'
                df.loc[df['macd_hist'] > 0, 'macd_hist_state'] = 'POSITIVE'
                df.loc[df['macd_hist'] < 0, 'macd_hist_state'] = 'NEGATIVE'
                df.loc[df['macd_hist'] > df['macd_hist'].shift(1), 'macd_hist_trend'] = 'RISING'
                df.loc[df['macd_hist'] < df['macd_hist'].shift(1), 'macd_hist_trend'] = 'FALLING'
            
            # EMA Stack State
            if all(col in df.columns for col in ['ema_8', 'ema_21', 'ema_50']):
                df['ema_stack'] = 'NEUTRAL'
                
                bull_stack = (df['close'] > df['ema_8']) & (df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
                bear_stack = (df['close'] < df['ema_8']) & (df['ema_8'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
                weak_bull = (df['close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])
                weak_bear = (df['close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])
                
                df.loc[bull_stack, 'ema_stack'] = 'STRONG_BULL'
                df.loc[bear_stack, 'ema_stack'] = 'STRONG_BEAR'
                df.loc[weak_bull & ~bull_stack, 'ema_stack'] = 'WEAK_BULL'
                df.loc[weak_bear & ~bear_stack, 'ema_stack'] = 'WEAK_BEAR'
            
            # ADX State
            if 'adx' in df.columns:
                df['adx_state'] = 'WEAK_TREND'
                df.loc[df['adx'] > 25, 'adx_state'] = 'TRENDING'
                df.loc[df['adx'] > 40, 'adx_state'] = 'STRONG_TREND'
                df.loc[df['adx'] < 20, 'adx_state'] = 'RANGING'
            
            # Volume State
            if 'volume_ratio' in df.columns:
                df['volume_state'] = 'NORMAL'
                df.loc[df['volume_ratio'] > 1.5, 'volume_state'] = 'SPIKE'
                df.loc[df['volume_ratio'] > 2.0, 'volume_state'] = 'EXTREME_SPIKE'
                df.loc[df['volume_ratio'] < 0.5, 'volume_state'] = 'DRYING_UP'
            
            return df
            
        except Exception as e:
            log.debug(f"Error adding indicator states: {e}")
            return df
    
    # ==================== DIVERGENCE DETECTION ====================
    
    def _add_divergence_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect RSI and MACD divergences with price
        """
        try:
            # Initialize divergence columns
            df['rsi_divergence'] = 'NONE'
            df['macd_divergence'] = 'NONE'
            
            if len(df) < 20:
                return df
            
            price = df['close'].values
            rsi = df['rsi'].values
            macd_hist = df['macd_hist'].values
            
            # Find local minima and maxima
            local_min = argrelextrema(price, np.less, order=5)[0]
            local_max = argrelextrema(price, np.greater, order=5)[0]
            
            # RSI Divergence Detection
            # Bullish Divergence: Price lower low, RSI higher low
            for i in range(1, len(local_min)):
                idx1 = local_min[i-1]
                idx2 = local_min[i]
                
                if price[idx2] < price[idx1] and rsi[idx2] > rsi[idx1]:
                    df.loc[df.index[idx2], 'rsi_divergence'] = 'BULLISH'
            
            # Bearish Divergence: Price higher high, RSI lower high
            for i in range(1, len(local_max)):
                idx1 = local_max[i-1]
                idx2 = local_max[i]
                
                if price[idx2] > price[idx1] and rsi[idx2] < rsi[idx1]:
                    df.loc[df.index[idx2], 'rsi_divergence'] = 'BEARISH'
            
            # MACD Divergence Detection
            # Bullish Divergence: Price lower low, MACD higher low
            for i in range(1, len(local_min)):
                idx1 = local_min[i-1]
                idx2 = local_min[i]
                
                if price[idx2] < price[idx1] and macd_hist[idx2] > macd_hist[idx1]:
                    df.loc[df.index[idx2], 'macd_divergence'] = 'BULLISH'
            
            # Bearish Divergence: Price higher high, MACD lower high
            for i in range(1, len(local_max)):
                idx1 = local_max[i-1]
                idx2 = local_max[i]
                
                if price[idx2] > price[idx1] and macd_hist[idx2] < macd_hist[idx1]:
                    df.loc[df.index[idx2], 'macd_divergence'] = 'BEARISH'
            
            return df
            
        except Exception as e:
            log.debug(f"Error detecting divergences: {e}")
            return df
    
    # ==================== INDICATOR SCORING ====================
    
    def _add_indicator_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator score columns for each indicator
        Score: -1 (bearish) to +1 (bullish)
        """
        try:
            # RSI Score
            df['rsi_score'] = 0.0
            df.loc[df['rsi'] < 30, 'rsi_score'] = 0.8
            df.loc[(df['rsi'] >= 30) & (df['rsi'] < 40), 'rsi_score'] = 0.4
            df.loc[(df['rsi'] >= 40) & (df['rsi'] < 50), 'rsi_score'] = 0.2
            df.loc[(df['rsi'] >= 50) & (df['rsi'] < 60), 'rsi_score'] = -0.2
            df.loc[(df['rsi'] >= 60) & (df['rsi'] < 70), 'rsi_score'] = -0.4
            df.loc[df['rsi'] > 70, 'rsi_score'] = -0.8
            
            # Adjust for RSI divergence
            df.loc[df['rsi_divergence'] == 'BULLISH', 'rsi_score'] += 0.3
            df.loc[df['rsi_divergence'] == 'BEARISH', 'rsi_score'] -= 0.3
            
            # MACD Score
            df['macd_score'] = 0.0
            df.loc[df['macd_hist'] > 0, 'macd_score'] += 0.3
            df.loc[df['macd_hist'] < 0, 'macd_score'] -= 0.3
            df.loc[df['macd_cross'] == 'CROSS_ABOVE', 'macd_score'] += 0.4
            df.loc[df['macd_cross'] == 'CROSS_BELOW', 'macd_score'] -= 0.4
            
            # MACD divergence adjustment
            df.loc[df['macd_divergence'] == 'BULLISH', 'macd_score'] += 0.3
            df.loc[df['macd_divergence'] == 'BEARISH', 'macd_score'] -= 0.3
            
            # EMA Stack Score
            df['ema_score'] = 0.0
            df.loc[df['ema_stack'] == 'STRONG_BULL', 'ema_score'] = 0.5
            df.loc[df['ema_stack'] == 'WEAK_BULL', 'ema_score'] = 0.25
            df.loc[df['ema_stack'] == 'WEAK_BEAR', 'ema_score'] = -0.25
            df.loc[df['ema_stack'] == 'STRONG_BEAR', 'ema_score'] = -0.5
            
            # ADX Score (trend strength)
            df['adx_score'] = 0.0
            df.loc[df['adx'] > 25, 'adx_score'] += 0.1
            df.loc[df['adx'] > 40, 'adx_score'] += 0.15
            
            # Volume Score
            df['volume_score'] = 0.0
            df.loc[df['volume_state'] == 'SPIKE', 'volume_score'] += 0.2
            df.loc[df['volume_state'] == 'EXTREME_SPIKE', 'volume_score'] += 0.3
            df.loc[df['volume_state'] == 'DRYING_UP', 'volume_score'] -= 0.15
            
            # Bollinger Band Score
            df['bb_score'] = 0.0
            if 'bb_position' in df.columns:
                df.loc[df['bb_position'] < 0.2, 'bb_score'] = 0.3
                df.loc[df['bb_position'] > 0.8, 'bb_score'] = -0.3
            
            # Combined Score (weighted average)
            df['indicator_score'] = (
                df['rsi_score'] * 0.20 +
                df['macd_score'] * 0.20 +
                df['ema_score'] * 0.25 +
                df['adx_score'] * 0.10 +
                df['volume_score'] * 0.15 +
                df['bb_score'] * 0.10
            )
            
            # Clamp to -1 to 1
            df['indicator_score'] = df['indicator_score'].clip(-1, 1)
            
            return df
            
        except Exception as e:
            log.debug(f"Error adding indicator scores: {e}")
            return df
    
    # ==================== GET LATEST INDICATOR ANALYSIS ====================
    
    def get_latest_indicator_analysis(self, df: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """
        Get comprehensive analysis of all indicators for the latest candle
        """
        if df is None or df.empty:
            return {}
        
        results = {}
        latest = df.iloc[-1]
        
        # RSI Analysis
        results['rsi'] = IndicatorResult(
            name='RSI',
            value=float(latest['rsi']) if 'rsi' in latest else 50,
            state=self._get_rsi_state(latest),
            score=float(latest['rsi_score']) if 'rsi_score' in latest else 0,
            strength=abs(float(latest['rsi_score'])) if 'rsi_score' in latest else 0,
            reason=self._get_rsi_reason(latest),
            details={'rsi': float(latest['rsi']), 'divergence': latest.get('rsi_divergence', 'NONE')}
        )
        
        # MACD Analysis
        results['macd'] = IndicatorResult(
            name='MACD',
            value=float(latest['macd_hist']) if 'macd_hist' in latest else 0,
            state=self._get_macd_state(latest),
            score=float(latest['macd_score']) if 'macd_score' in latest else 0,
            strength=abs(float(latest['macd_score'])) if 'macd_score' in latest else 0,
            reason=self._get_macd_reason(latest),
            details={'histogram': float(latest['macd_hist']), 'divergence': latest.get('macd_divergence', 'NONE')}
        )
        
        # EMA Stack Analysis
        results['ema'] = IndicatorResult(
            name='EMA Stack',
            value=0,  # No single value
            state=self._get_ema_state(latest),
            score=float(latest['ema_score']) if 'ema_score' in latest else 0,
            strength=abs(float(latest['ema_score'])) if 'ema_score' in latest else 0,
            reason=latest.get('ema_stack', 'NEUTRAL'),
            details={'stack': latest.get('ema_stack', 'NEUTRAL')}
        )
        
        # Volume Analysis
        results['volume'] = IndicatorResult(
            name='Volume',
            value=float(latest['volume_ratio']) if 'volume_ratio' in latest else 1,
            state=self._get_volume_state(latest),
            score=float(latest['volume_score']) if 'volume_score' in latest else 0,
            strength=abs(float(latest['volume_score'])) if 'volume_score' in latest else 0,
            reason=latest.get('volume_state', 'NORMAL'),
            details={'ratio': float(latest['volume_ratio'])}
        )
        
        # ADX Analysis
        results['adx'] = IndicatorResult(
            name='ADX',
            value=float(latest['adx']) if 'adx' in latest else 20,
            state=self._get_adx_state(latest),
            score=float(latest['adx_score']) if 'adx_score' in latest else 0,
            strength=abs(float(latest['adx_score'])) if 'adx_score' in latest else 0,
            reason=latest.get('adx_state', 'WEAK_TREND'),
            details={'adx': float(latest['adx'])}
        )
        
        # Combined Score
        results['combined'] = IndicatorResult(
            name='Combined',
            value=float(latest['indicator_score']) if 'indicator_score' in latest else 0,
            state=IndicatorState.BULLISH if float(latest.get('indicator_score', 0)) > 0.2 else
                  IndicatorState.BEARISH if float(latest.get('indicator_score', 0)) < -0.2 else
                  IndicatorState.NEUTRAL,
            score=float(latest['indicator_score']) if 'indicator_score' in latest else 0,
            strength=abs(float(latest['indicator_score'])) if 'indicator_score' in latest else 0,
            reason=f"Combined indicator score: {latest.get('indicator_score', 0):.2f}",
            details={}
        )
        
        return results
    
    # ==================== STATE HELPER METHODS ====================
    
    def _get_rsi_state(self, row: pd.Series) -> IndicatorState:
        rsi = row.get('rsi', 50)
        divergence = row.get('rsi_divergence', 'NONE')
        
        if rsi > 70:
            return IndicatorState.OVERBOUGHT
        if rsi < 30:
            return IndicatorState.OVERSOLD
        if divergence == 'BULLISH':
            return IndicatorState.DIVERGENCE_BULLISH
        if divergence == 'BEARISH':
            return IndicatorState.DIVERGENCE_BEARISH
        if row.get('rsi_rising', False):
            return IndicatorState.RISING
        if row.get('rsi_falling', False):
            return IndicatorState.FALLING
        return IndicatorState.NEUTRAL
    
    def _get_rsi_reason(self, row: pd.Series) -> str:
        rsi = row.get('rsi', 50)
        divergence = row.get('rsi_divergence', 'NONE')
        
        if divergence == 'BULLISH':
            return f"Bullish RSI divergence (price lower low, RSI higher low)"
        if divergence == 'BEARISH':
            return f"Bearish RSI divergence (price higher high, RSI lower high)"
        if rsi > 70:
            return f"RSI overbought: {rsi:.1f}"
        if rsi < 30:
            return f"RSI oversold: {rsi:.1f}"
        if rsi > 55:
            return f"RSI bullish: {rsi:.1f}"
        if rsi < 45:
            return f"RSI bearish: {rsi:.1f}"
        return f"RSI neutral: {rsi:.1f}"
    
    def _get_macd_state(self, row: pd.Series) -> IndicatorState:
        hist = row.get('macd_hist', 0)
        cross = row.get('macd_cross', 'NONE')
        divergence = row.get('macd_divergence', 'NONE')
        
        if cross == 'CROSS_ABOVE':
            return IndicatorState.CROSSED_ABOVE
        if cross == 'CROSS_BELOW':
            return IndicatorState.CROSSED_BELOW
        if divergence == 'BULLISH':
            return IndicatorState.DIVERGENCE_BULLISH
        if divergence == 'BEARISH':
            return IndicatorState.DIVERGENCE_BEARISH
        if hist > 0:
            return IndicatorState.BULLISH
        if hist < 0:
            return IndicatorState.BEARISH
        return IndicatorState.NEUTRAL
    
    def _get_macd_reason(self, row: pd.Series) -> str:
        hist = row.get('macd_hist', 0)
        cross = row.get('macd_cross', 'NONE')
        divergence = row.get('macd_divergence', 'NONE')
        
        if divergence == 'BULLISH':
            return "Bullish MACD divergence"
        if divergence == 'BEARISH':
            return "Bearish MACD divergence"
        if cross == 'CROSS_ABOVE':
            return "MACD bullish crossover"
        if cross == 'CROSS_BELOW':
            return "MACD bearish crossover"
        if hist > 0:
            return f"MACD bullish: {hist:.4f}"
        if hist < 0:
            return f"MACD bearish: {hist:.4f}"
        return "MACD neutral"
    
    def _get_ema_state(self, row: pd.Series) -> IndicatorState:
        stack = row.get('ema_stack', 'NEUTRAL')
        if 'STRONG_BULL' in stack:
            return IndicatorState.BULLISH
        if 'STRONG_BEAR' in stack:
            return IndicatorState.BEARISH
        if 'WEAK_BULL' in stack:
            return IndicatorState.BULLISH
        if 'WEAK_BEAR' in stack:
            return IndicatorState.BEARISH
        return IndicatorState.NEUTRAL
    
    def _get_volume_state(self, row: pd.Series) -> IndicatorState:
        state = row.get('volume_state', 'NORMAL')
        if state == 'SPIKE' or state == 'EXTREME_SPIKE':
            return IndicatorState.BULLISH if row.get('close', 0) > row.get('open', 0) else IndicatorState.BEARISH
        if state == 'DRYING_UP':
            return IndicatorState.NEUTRAL
        return IndicatorState.NEUTRAL
    
    def _get_adx_state(self, row: pd.Series) -> IndicatorState:
        state = row.get('adx_state', 'WEAK_TREND')
        if state == 'STRONG_TREND':
            return IndicatorState.BULLISH if row.get('adx_pos', 0) > row.get('adx_neg', 0) else IndicatorState.BEARISH
        if state == 'TRENDING':
            return IndicatorState.BULLISH if row.get('adx_pos', 0) > row.get('adx_neg', 0) else IndicatorState.BEARISH
        return IndicatorState.NEUTRAL
    
    # ==================== LEGACY METHODS (Keep for compatibility) ====================
    
    def calculate_volatility_stops(self, df: pd.DataFrame, entry_price: float, action: str, 
                                    atr_multiplier: float = None, use_pivot_levels: bool = True):
        """Calculate volatility-based stops using new indicators"""
        if df is None or df.empty:
            return {'stop_loss': None, 'take_profit': None, 'risk_reward': 0}
        
        atr = float(df['atr'].iloc[-1]) if 'atr' in df and df['atr'].iloc[-1] > 0 else entry_price * 0.02
        if atr_multiplier is None:
            atr_multiplier = self.active_periods.get('atr_multiplier_stop', 1.5)
        tp_multiplier = self.active_periods.get('atr_tp_multiplier', 2.0)
        
        # Try Keltner Channel first
        if 'kc_high' in df and 'kc_low' in df and pd.notna(df['kc_high'].iloc[-1]):
            if action == 'BUY':
                stop_loss = float(df['kc_low'].iloc[-1]) * 0.995
                take_profit = entry_price + atr * tp_multiplier
            else:
                stop_loss = float(df['kc_high'].iloc[-1]) * 1.005
                take_profit = entry_price - atr * tp_multiplier
        else:
            # Fallback to ATR
            if action == 'BUY':
                stop_loss = entry_price - atr * atr_multiplier
                take_profit = entry_price + atr * tp_multiplier
            elif action == 'SELL':
                stop_loss = entry_price + atr * atr_multiplier
                take_profit = entry_price - atr * tp_multiplier
            else:
                return {'stop_loss': None, 'take_profit': None, 'risk_reward': 0}
        
        # ensure stop is on the other side
        if action == 'BUY' and stop_loss >= entry_price:
            stop_loss = entry_price * 0.99
        if action == 'SELL' and stop_loss <= entry_price:
            stop_loss = entry_price * 1.01
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        return {
            'stop_loss': round(stop_loss, 6),
            'take_profit': round(take_profit, 6),
            'atr': round(atr, 6),
            'atr_percent': round(atr / entry_price, 4),
            'risk_reward': round(rr, 2)
        }
    
    def find_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find S/R using Donchian channels and swing points"""
        try:
            if df is None or df.empty or len(df) < 20:
                return [], []
            
            supports = []
            resistances = []
            
            # Use Donchian lows as supports, highs as resistance
            if 'donchian_low' in df:
                supports.append(float(df['donchian_low'].iloc[-1]))
            if 'donchian_high' in df:
                resistances.append(float(df['donchian_high'].iloc[-1]))
            
            # Also add recent swing points
            if 'previous_high' in df and not pd.isna(df['previous_high'].iloc[-1]):
                resistances.append(float(df['previous_high'].iloc[-1]))
            if 'previous_low' in df and not pd.isna(df['previous_low'].iloc[-1]):
                supports.append(float(df['previous_low'].iloc[-1]))
            
            # Add hidden levels if available
            if 'hidden_support' in df and not pd.isna(df['hidden_support'].iloc[-1]):
                supports.append(float(df['hidden_support'].iloc[-1]))
            if 'hidden_resistance' in df and not pd.isna(df['hidden_resistance'].iloc[-1]):
                resistances.append(float(df['hidden_resistance'].iloc[-1]))
            
            # Add VAH/VAL if available
            if 'vprofile_val_high' in df and not pd.isna(df['vprofile_val_high'].iloc[-1]):
                resistances.append(float(df['vprofile_val_high'].iloc[-1]))
            if 'vprofile_val_low' in df and not pd.isna(df['vprofile_val_low'].iloc[-1]):
                supports.append(float(df['vprofile_val_low'].iloc[-1]))
            
            # Add POC if available
            if 'vprofile_poc' in df and not pd.isna(df['vprofile_poc'].iloc[-1]):
                poc = float(df['vprofile_poc'].iloc[-1])
                supports.append(poc)
                resistances.append(poc)
            
            # Remove duplicates (within 0.1%)
            unique_supports = []
            for s in sorted(supports):
                if not unique_supports or abs(s - unique_supports[-1]) / s > 0.001:
                    unique_supports.append(s)
            
            unique_resistances = []
            for r in sorted(resistances):
                if not unique_resistances or abs(r - unique_resistances[-1]) / r > 0.001:
                    unique_resistances.append(r)
            
            return unique_supports[-5:], unique_resistances[:5]
        
        except Exception:
            return [], []