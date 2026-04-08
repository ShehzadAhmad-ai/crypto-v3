# market_regime_advanced.py - Advanced Market Regime Detection
"""
Advanced Market Regime Detection Expert
- Detects market regime with confidence scoring
- Wyckoff phase detection (Accumulation, Markup, Distribution, Markdown)
- Volatility and liquidity regimes
- DECIDES direction and bias for the entire system
- Dynamic indicator settings based on regime
- All thresholds from ta_config.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from scipy import stats
from scipy.signal import argrelextrema
from datetime import datetime

# Import configuration and core classes
from .ta_config import *
from .ta_core import RegimeResult, RegimeType

# Try to import logger, fallback to print if not available
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class MarketRegimeAdvanced:
    """
    Advanced Market Regime Detection Expert
    Determines market state, direction bias, and provides dynamic settings
    """
    
    def __init__(self):
        """Initialize the regime detector with thresholds from config"""
        self.regime_history = []
        self.last_regime: Optional[RegimeResult] = None
        
        # Load thresholds from config
        self.trend_adx_threshold = REGIME_ADX_TRENDING
        self.strong_trend_adx = REGIME_ADX_STRONG_TREND
        self.bb_volatile_threshold = REGIME_BB_WIDTH_VOLATILE
        self.bb_squeeze_threshold = REGIME_BB_WIDTH_SQUEEZE
        
        log.info("MarketRegimeAdvanced initialized")
    
    # ============================================================================
    # MAIN PUBLIC METHODS
    # ============================================================================
    
    def detect_regime(self, df: pd.DataFrame, symbol: str = "") -> RegimeResult:
        """
        Complete regime detection with all metrics
        This is the main method that DECIDES direction for the system
        
        Args:
            df: DataFrame with OHLCV data (must have at least 100 candles)
            symbol: Symbol name for logging
        
        Returns:
            RegimeResult with full analysis and bias decision
        """
        if df is None or df.empty or len(df) < MIN_DATA_CANDLES:
            log.warning(f"Insufficient data for regime detection: {len(df) if df is not None else 0} candles")
            return self._default_regime_result()
        
        try:
            # ===== STEP 1: CALCULATE BASE METRICS =====
            adx = self._calculate_adx(df)
            atr_pct = self._calculate_atr_pct(df)
            slope = self._calculate_slope(df)
            hurst = self._calculate_hurst_exponent(df['close'].values)
            
            # ===== STEP 2: REGIME METRICS =====
            vol_regime = self._detect_volatility_regime(df)
            liq_regime = self._detect_liquidity_regime(df)
            bb_squeeze, bb_intensity = self._detect_bollinger_squeeze(df)
            trending_score = self._calculate_trending_score(df, hurst, adx)
            wyckoff_phase, wyckoff_conf = self._detect_wyckoff_phase(df)
            
            # ===== STEP 3: DETERMINE REGIME TYPE =====
            regime, regime_type, regime_confidence = self._classify_regime(
                adx, slope, atr_pct, trending_score
            )
            
            # ===== STEP 4: DETERMINE DIRECTIONAL BIAS (CRITICAL) =====
            bias, bias_score = self._calculate_directional_bias(df, adx, slope, trending_score, wyckoff_phase)
            
            # ===== STEP 5: DYNAMIC INDICATOR SETTINGS =====
            indicator_settings = self._get_dynamic_indicator_settings(regime, bias_score, atr_pct)
            
            # ===== STEP 6: STRATEGY RECOMMENDATIONS =====
            recommended, avoid = self._get_strategy_recommendations(regime, bias_score, trending_score)
            
            # ===== STEP 7: HUMAN-READABLE SUMMARY =====
            summary = self._generate_regime_summary(
                regime, bias, bias_score, trending_score, wyckoff_phase, atr_pct
            )
            
            # ===== CREATE RESULT OBJECT =====
            result = RegimeResult(
                regime=regime,
                regime_type=regime_type,
                confidence=regime_confidence,
                bias=bias,
                bias_score=bias_score,
                trend_strength=trending_score,
                volatility_state=vol_regime,
                liquidity_state=liq_regime,
                adx=round(adx, 1),
                atr_pct=round(atr_pct, 4),
                slope=round(slope, 4),
                is_squeeze=bb_squeeze,
                squeeze_intensity=round(bb_intensity, 3),
                wyckoff_phase=wyckoff_phase,
                wyckoff_confidence=wyckoff_conf,
                recommended_strategies=recommended,
                avoid_strategies=avoid,
                indicator_settings=indicator_settings,
                summary=summary,
                details={
                    'hurst': round(hurst, 3),
                    'trending_score': round(trending_score, 3),
                    'vol_regime': vol_regime,
                    'liq_regime': liq_regime,
                    'adx': adx,
                    'atr_pct': atr_pct,
                    'slope': slope,
                }
            )
            
            # Store in history
            self.last_regime = result
            self.regime_history.append({
                'regime': regime,
                'bias': bias,
                'bias_score': bias_score,
                'timestamp': datetime.now(),
                'trending_score': trending_score,
            })
            
            # Keep history size manageable
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            log.info(f"Regime detected: {regime} | Bias: {bias} ({bias_score:.2f}) | Confidence: {regime_confidence:.2f}")
            
            return result
            
        except Exception as e:
            log.error(f"Error in regime detection: {e}")
            return self._default_regime_result()
    
    def get_current_bias(self) -> Tuple[str, float]:
        """
        Get current market bias (direction decision)
        
        Returns:
            Tuple of (bias_string, bias_score) where bias_score is -1 to 1
        """
        if self.last_regime:
            return self.last_regime.bias, self.last_regime.bias_score
        return "NEUTRAL", 0.0
    
    def is_trending(self) -> bool:
        """Check if current market is trending"""
        if self.last_regime:
            return self.last_regime.regime_type == "TREND"
        return False
    
    def is_ranging(self) -> bool:
        """Check if current market is ranging"""
        if self.last_regime:
            return self.last_regime.regime_type == "RANGE"
        return False
    
    def is_volatile(self) -> bool:
        """Check if current market is volatile"""
        if self.last_regime:
            return self.last_regime.regime_type == "VOLATILE"
        return False
    
    def get_regime_summary(self) -> str:
        """Get current regime as human-readable summary"""
        if self.last_regime:
            return self.last_regime.summary
        return "No regime data available"
    
    # ============================================================================
    # PRIVATE CALCULATION METHODS
    # ============================================================================
    
    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calculate ADX - use existing or compute from scratch"""
        try:
            if 'adx' in df and not pd.isna(df['adx'].iloc[-1]):
                return float(df['adx'].iloc[-1])
            
            # Calculate ADX from scratch
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate Directional Movement
            up_move = high - np.roll(high, 1)
            down_move = np.roll(low, 1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth with Wilder's smoothing (14 periods)
            period = ADX_PERIOD
            atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = dx.ewm(alpha=1/period, adjust=False).mean()
            
            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 20.0
            
        except Exception as e:
            log.debug(f"Error calculating ADX: {e}")
            return 20.0
    
    def _calculate_atr_pct(self, df: pd.DataFrame) -> float:
        """Calculate ATR as percentage of price"""
        try:
            if 'atr' in df and not pd.isna(df['atr'].iloc[-1]) and df['close'].iloc[-1] > 0:
                return float(df['atr'].iloc[-1] / df['close'].iloc[-1])
            
            # Calculate ATR from scratch
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            atr = pd.Series(tr).rolling(window=ATR_PERIOD).mean()
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
            current_price = df['close'].iloc[-1]
            
            if current_price > 0:
                return float(current_atr / current_price)
            return 0.02
            
        except Exception as e:
            log.debug(f"Error calculating ATR%: {e}")
            return 0.02
    
    def _calculate_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate price slope over specified period"""
        try:
            if len(df) >= period:
                return (df['close'].iloc[-1] - df['close'].iloc[-period]) / (df['close'].iloc[-period] + 1e-10)
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_hurst_exponent(self, price_series: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst Exponent for trend persistence"""
        try:
            if len(price_series) < 100:
                return 0.5
            
            returns = np.diff(np.log(price_series + 1e-10))
            lags = range(2, min(max_lag, len(returns) // 4))
            tau = []
            
            for lag in lags:
                diff = np.array([returns[i] - returns[i - lag] for i in range(lag, len(returns))])
                if len(diff) > 0:
                    tau.append(np.std(diff))
            
            if len(tau) > 0 and len(lags) == len(tau):
                poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                hurst = poly[0] / 2.0
                return max(0.0, min(1.0, hurst))
            
            return 0.5
        except Exception:
            return 0.5
    
    def _detect_volatility_regime(self, df: pd.DataFrame) -> str:
        """Detect volatility regime based on percentiles"""
        try:
            if len(df) < 100:
                return 'NORMAL'
            
            returns = df['close'].pct_change().dropna()
            hist_vol = returns.rolling(20).std() * np.sqrt(252)
            current_vol = hist_vol.iloc[-1] if not pd.isna(hist_vol.iloc[-1]) else 0
            
            vol_percentile = stats.percentileofscore(hist_vol.dropna(), current_vol) / 100
            
            if vol_percentile < 0.1:
                return 'EXTREME_LOW'
            elif vol_percentile < 0.25:
                return 'LOW'
            elif vol_percentile < 0.75:
                return 'NORMAL'
            elif vol_percentile < 0.9:
                return 'HIGH'
            else:
                return 'EXTREME_HIGH'
        except Exception:
            return 'NORMAL'
    
    def _detect_liquidity_regime(self, df: pd.DataFrame) -> str:
        """Detect liquidity regime based on volume"""
        try:
            if len(df) < 50:
                return 'NORMAL'
            
            volume = df['volume']
            volume_sma = volume.rolling(VOLUME_MA_PERIOD).mean()
            current_vol_ratio = volume.iloc[-1] / (volume_sma.iloc[-1] + 1e-10)
            volume_trend = volume.iloc[-10:].mean() / (volume.iloc[-20:-10].mean() + 1e-10) if len(volume) >= 20 else 1
            
            liquidity_score = current_vol_ratio * volume_trend
            
            if liquidity_score < 0.3:
                return 'VERY_LOW'
            elif liquidity_score < 0.7:
                return 'LOW'
            elif liquidity_score < 1.3:
                return 'NORMAL'
            elif liquidity_score < 2.0:
                return 'HIGH'
            else:
                return 'VERY_HIGH'
        except Exception:
            return 'NORMAL'
    
    def _detect_bollinger_squeeze(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Detect Bollinger Band squeeze"""
        try:
            if len(df) < BB_PERIOD + 10:
                return False, 0.0
            
            sma = df['close'].rolling(BB_PERIOD).mean()
            std = df['close'].rolling(BB_PERIOD).std()
            
            upper = sma + (std * BB_STD)
            lower = sma - (std * BB_STD)
            
            bandwidth = (upper - lower) / (sma + 1e-10)
            hist_bandwidth = bandwidth.rolling(BB_PERIOD * 3).mean()
            current_bandwidth = bandwidth.iloc[-1]
            hist_bandwidth_val = hist_bandwidth.iloc[-1] if not pd.isna(hist_bandwidth.iloc[-1]) else current_bandwidth
            
            if hist_bandwidth_val > 0:
                squeeze_ratio = current_bandwidth / hist_bandwidth_val
                is_squeeze = squeeze_ratio < self.bb_squeeze_threshold
                intensity = max(0, min(1, (1 - squeeze_ratio) / 0.5))
                return is_squeeze, intensity
            
            return False, 0.0
        except Exception:
            return False, 0.0
    
    def _calculate_trending_score(self, df: pd.DataFrame, hurst: float, adx: float) -> float:
        """Calculate trending vs ranging score (0-1)"""
        try:
            scores = []
            
            # Hurst contribution (0.5 = random, 1 = trending)
            hurst_score = max(0, min(1, (hurst - 0.5) * 2))
            scores.append(hurst_score)
            
            # ADX contribution
            adx_score = min(1, adx / 50) if adx > 0 else 0
            scores.append(adx_score)
            
            # EMA alignment
            if all(col in df for col in ['ema_8', 'ema_21', 'ema_50']):
                price = df['close'].iloc[-1]
                e8 = df['ema_8'].iloc[-1]
                e21 = df['ema_21'].iloc[-1]
                e50 = df['ema_50'].iloc[-1]
                
                if (price > e8 > e21 > e50) or (price < e8 < e21 < e50):
                    scores.append(1.0)
                elif (price > e8 and price > e21 and price > e50) or (price < e8 and price < e21 and price < e50):
                    scores.append(0.7)
                else:
                    scores.append(0.3)
            
            return np.mean(scores) if scores else 0.5
        except Exception:
            return 0.5
    
    def _classify_regime(self, adx: float, slope: float, atr_pct: float, trending_score: float) -> Tuple[str, str, float]:
        """
        Classify the regime type with confidence
        Returns: (regime, regime_type, confidence)
        """
        # Strong trend detection
        if adx > self.strong_trend_adx:
            if slope > 0.01:
                regime = RegimeType.STRONG_BULL_TREND.value
                regime_type = "TREND"
                confidence = min(0.95, 0.7 + (adx - self.strong_trend_adx) / 50)
            elif slope < -0.01:
                regime = RegimeType.STRONG_BEAR_TREND.value
                regime_type = "TREND"
                confidence = min(0.95, 0.7 + (adx - self.strong_trend_adx) / 50)
            else:
                regime = RegimeType.RANGING_NEUTRAL.value
                regime_type = "RANGE"
                confidence = 0.5
        
        # Moderate trend detection
        elif adx > self.trend_adx_threshold:
            if slope > 0.005:
                regime = RegimeType.BULL_TREND.value
                regime_type = "TREND"
                confidence = 0.6 + (adx - self.trend_adx_threshold) / 50
            elif slope < -0.005:
                regime = RegimeType.BEAR_TREND.value
                regime_type = "TREND"
                confidence = 0.6 + (adx - self.trend_adx_threshold) / 50
            else:
                regime = RegimeType.RANGING_NEUTRAL.value
                regime_type = "RANGE"
                confidence = 0.4
        
        # Ranging market
        elif adx < 20:
            # Check for volatility expansion/contraction
            if atr_pct > ATR_PERCENT_HIGH * 1.5:
                regime = RegimeType.VOLATILE_EXPANSION.value
                regime_type = "VOLATILE"
                confidence = 0.7
            elif atr_pct < ATR_PERCENT_LOW * 1.5:
                regime = RegimeType.VOLATILE_CONTRACTION.value
                regime_type = "VOLATILE"
                confidence = 0.6
            else:
                # Determine ranging bias
                if slope > 0.003:
                    regime = RegimeType.RANGING_BULL_BIAS.value
                    regime_type = "RANGE"
                    confidence = 0.5
                elif slope < -0.003:
                    regime = RegimeType.RANGING_BEAR_BIAS.value
                    regime_type = "RANGE"
                    confidence = 0.5
                else:
                    regime = RegimeType.RANGING_NEUTRAL.value
                    regime_type = "RANGE"
                    confidence = 0.7
        else:
            regime = RegimeType.UNKNOWN.value
            regime_type = "UNKNOWN"
            confidence = 0.3
        
        return regime, regime_type, confidence
    
    def _calculate_directional_bias(self, df: pd.DataFrame, adx: float, slope: float, 
                                     trending_score: float, wyckoff_phase: str) -> Tuple[str, float]:
        """
        Calculate directional bias (-1 to 1) and label
        This is the CRITICAL method that DECIDES direction for the system
        
        Returns: (bias_label, bias_score)
        """
        bias_score = 0.0
        
        # Factor 1: Price slope (20% weight)
        slope_score = slope * 5  # Normalize slope to -1 to 1 range
        bias_score += slope_score * 0.20
        
        # Factor 2: EMA alignment (25% weight)
        if all(col in df for col in ['ema_8', 'ema_21', 'ema_50', 'ema_200']):
            price = df['close'].iloc[-1]
            e8 = df['ema_8'].iloc[-1]
            e21 = df['ema_21'].iloc[-1]
            e50 = df['ema_50'].iloc[-1]
            e200 = df['ema_200'].iloc[-1]
            
            ema_score = 0.0
            if price > e8 > e21 > e50 > e200:
                ema_score = 0.5
            elif price > e8 and price > e21:
                ema_score = 0.25
            elif price < e8 < e21 < e50 < e200:
                ema_score = -0.5
            elif price < e8 and price < e21:
                ema_score = -0.25
            
            bias_score += ema_score * 0.25
        
        # Factor 3: ADX/DMI direction (20% weight)
        if 'adx_pos' in df and 'adx_neg' in df:
            dmi_plus = df['adx_pos'].iloc[-1]
            dmi_minus = df['adx_neg'].iloc[-1]
            
            if dmi_plus > dmi_minus + DMI_PLUS_MINUS_THRESHOLD:
                dmi_score = 0.3
            elif dmi_plus > dmi_minus:
                dmi_score = 0.15
            elif dmi_minus > dmi_plus + DMI_PLUS_MINUS_THRESHOLD:
                dmi_score = -0.3
            elif dmi_minus > dmi_plus:
                dmi_score = -0.15
            else:
                dmi_score = 0.0
            
            bias_score += dmi_score * 0.20
        
        # Factor 4: Wyckoff phase (15% weight)
        wyckoff_score = 0.0
        if wyckoff_phase in ['MARKUP', 'ACCUMULATION']:
            wyckoff_score = 0.3
        elif wyckoff_phase in ['MARKDOWN', 'DISTRIBUTION']:
            wyckoff_score = -0.3
        
        bias_score += wyckoff_score * 0.15
        
        # Factor 5: RSI position (10% weight)
        if 'rsi' in df:
            rsi = df['rsi'].iloc[-1]
            if rsi > 60:
                rsi_score = 0.2
            elif rsi > 55:
                rsi_score = 0.1
            elif rsi < 40:
                rsi_score = -0.2
            elif rsi < 45:
                rsi_score = -0.1
            else:
                rsi_score = 0.0
            
            bias_score += rsi_score * 0.10
        
        # Factor 6: MACD position (10% weight)
        if 'macd_hist' in df:
            macd_hist = df['macd_hist'].iloc[-1]
            if macd_hist > 0:
                macd_score = 0.15 if macd_hist > df['macd_hist'].iloc[-2] else 0.05
            else:
                macd_score = -0.15 if macd_hist < df['macd_hist'].iloc[-2] else -0.05
            
            bias_score += macd_score * 0.10
        
        # Boost bias if trend is strong
        if adx > self.strong_trend_adx:
            bias_score = bias_score * (1 + min(0.3, (adx - self.strong_trend_adx) / 100))
        
        # Clamp to -1 to 1
        bias_score = max(-1.0, min(1.0, bias_score))
        
        # Determine bias label
        if bias_score > 0.6:
            bias = "STRONG_BULLISH"
        elif bias_score > 0.25:
            bias = "BULLISH"
        elif bias_score > -0.25:
            bias = "NEUTRAL"
        elif bias_score > -0.6:
            bias = "BEARISH"
        else:
            bias = "STRONG_BEARISH"
        
        return bias, bias_score
    
    def _detect_wyckoff_phase(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Detect Wyckoff market phase with confidence"""
        try:
            if len(df) < 100:
                return 'UNKNOWN', 0.0
            
            price = df['close'].values
            volume = df['volume'].values
            
            # Find local maxima and minima
            local_max = argrelextrema(price, np.greater, order=WYCKOFF_SWING_ORDER)[0]
            local_min = argrelextrema(price, np.less, order=WYCKOFF_SWING_ORDER)[0]
            
            if len(local_max) < 2 or len(local_min) < 2:
                return 'UNKNOWN', 0.0
            
            # Get recent swings (last 50 candles)
            recent_max_idx = [x for x in local_max if x > len(price) - 50]
            recent_min_idx = [x for x in local_min if x > len(price) - 50]
            
            recent_max = price[recent_max_idx] if recent_max_idx else []
            recent_min = price[recent_min_idx] if recent_min_idx else []
            
            # Volume trend
            recent_vol = volume[-50:]
            vol_trend = recent_vol[-10:].mean() / (recent_vol[:10].mean() + 1e-10) if len(recent_vol) >= 20 else 1
            
            if len(recent_max) >= 2 and len(recent_min) >= 2:
                last_max = recent_max[-1]
                prev_max = recent_max[-2]
                last_min = recent_min[-1]
                prev_min = recent_min[-2]
                
                # MARKUP: Higher highs, higher lows, volume increasing
                if last_max > prev_max and last_min > prev_min and vol_trend > WYCKOFF_VOLUME_THRESHOLD:
                    return 'MARKUP', 0.8
                
                # MARKDOWN: Lower highs, lower lows, volume increasing
                elif last_max < prev_max and last_min < prev_min and vol_trend > WYCKOFF_VOLUME_THRESHOLD:
                    return 'MARKDOWN', 0.8
                
                # ACCUMULATION: Range-bound, volume decreasing
                elif abs(last_max - prev_max) / (prev_max + 1e-10) < 0.05 and vol_trend < 0.9:
                    return 'ACCUMULATION', 0.7
                
                # DISTRIBUTION: Range-bound, volume increasing then weakening
                elif abs(last_max - prev_max) / (prev_max + 1e-10) < 0.05 and vol_trend > 1.2:
                    return 'DISTRIBUTION', 0.7
                
                # Re-accumulation after markup
                elif vol_trend < 0.8 and last_min > prev_min and abs(last_max - prev_max) / (prev_max + 1e-10) < 0.03:
                    return 'RE_ACCUMULATION', 0.6
                
                # Re-distribution after markdown
                elif vol_trend > 1.3 and last_max < prev_max and abs(last_min - prev_min) / (prev_min + 1e-10) < 0.03:
                    return 'RE_DISTRIBUTION', 0.6
            
            return 'UNKNOWN', 0.0
        except Exception:
            return 'UNKNOWN', 0.0
    
    def _get_dynamic_indicator_settings(self, regime: str, bias_score: float, atr_pct: float) -> Dict[str, Any]:
        """
        Get dynamic indicator settings based on regime
        These settings will be passed to the indicator analyzer
        """
        settings = {}
        
        # RSI adjustments based on regime
        if 'BULL' in regime:
            settings['rsi_oversold'] = RSI_BULL_TREND_OVERSOLD
            settings['rsi_overbought'] = RSI_BULL_TREND_OVERBOUGHT
            settings['rsi_period'] = max(8, RSI_PERIOD - 4)
        elif 'BEAR' in regime:
            settings['rsi_oversold'] = RSI_BEAR_TREND_OVERSOLD
            settings['rsi_overbought'] = RSI_BEAR_TREND_OVERBOUGHT
            settings['rsi_period'] = max(8, RSI_PERIOD - 4)
        elif 'RANGING' in regime:
            settings['rsi_oversold'] = RSI_OVERSOLD
            settings['rsi_overbought'] = RSI_OVERBOUGHT
            settings['rsi_period'] = RSI_PERIOD
        
        # MACD adjustments
        if 'TREND' in regime:
            settings['macd_fast'] = MACD_TRENDING_FAST
            settings['macd_slow'] = MACD_TRENDING_SLOW
        elif 'RANGING' in regime:
            settings['macd_fast'] = MACD_RANGING_FAST
            settings['macd_slow'] = MACD_RANGING_SLOW
        
        # Bollinger adjustments
        if 'RANGING' in regime:
            settings['bb_std'] = BB_STD + 0.2
        elif atr_pct > ATR_PERCENT_HIGH:
            settings['bb_std'] = BB_STD + 0.5
        elif atr_pct < ATR_PERCENT_LOW:
            settings['bb_std'] = max(1.5, BB_STD - 0.3)
        
        # Volume spike threshold adjustments
        if atr_pct > ATR_PERCENT_HIGH:
            settings['volume_spike_threshold'] = VOLUME_SPIKE_THRESHOLD + 0.5
        else:
            settings['volume_spike_threshold'] = VOLUME_SPIKE_THRESHOLD
        
        # ATR multiplier adjustments
        if atr_pct > ATR_PERCENT_HIGH:
            settings['atr_stop_multiplier'] = ATR_HIGH_VOL_MULTIPLIER
        elif atr_pct < ATR_PERCENT_LOW:
            settings['atr_stop_multiplier'] = ATR_LOW_VOL_MULTIPLIER
        else:
            settings['atr_stop_multiplier'] = ATR_STOP_MULTIPLIER
        
        return settings
    
    def _get_strategy_recommendations(self, regime: str, bias_score: float, trending_score: float) -> Tuple[List[str], List[str]]:
        """Get strategy recommendations based on regime"""
        recommended = []
        avoid = []
        
        if 'TREND' in regime or 'BULL' in regime or 'BEAR' in regime:
            recommended = [
                'Trend Following',
                'EMA Pullback',
                'MACD Crossover',
                'Breakout Strategies',
                'Higher Timeframe Continuation'
            ]
            avoid = [
                'Mean Reversion',
                'Range Trading',
                'VWAP Reversion'
            ]
        
        elif 'RANGING' in regime:
            recommended = [
                'Mean Reversion',
                'Support/Resistance Bounces',
                'Bollinger Bands',
                'VWAP Reversion',
                'RSI Boundaries'
            ]
            avoid = [
                'Trend Following',
                'Breakout Strategies',
                'Momentum Trading'
            ]
        
        elif 'VOLATILE' in regime:
            recommended = [
                'Volatility Breakout',
                'Momentum Trading',
                'Scalping',
                'Range Expansion'
            ]
            avoid = [
                'Mean Reversion',
                'Large Position Sizes',
                'Tight Stops'
            ]
        
        # Adjust based on bias strength
        if abs(bias_score) > 0.7:
            recommended.append('Strong Momentum Following')
        elif abs(bias_score) < 0.2:
            recommended.append('Wait for Clear Direction')
        
        return recommended[:5], avoid[:3]
    
    def _generate_regime_summary(self, regime: str, bias: str, bias_score: float,
                                   trending_score: float, wyckoff_phase: str, atr_pct: float) -> str:
        """Generate human-readable regime summary"""
        
        # Base summary
        if 'TREND' in regime:
            if 'BULL' in regime:
                strength = "strong" if "STRONG" in regime else "moderate"
                summary = f"{strength.capitalize()} bullish trend detected (ADX: trending, bias: {bias})"
            elif 'BEAR' in regime:
                strength = "strong" if "STRONG" in regime else "moderate"
                summary = f"{strength.capitalize()} bearish trend detected (ADX: trending, bias: {bias})"
            else:
                summary = f"Trending market with {bias.lower()} bias"
        
        elif 'RANGING' in regime:
            if 'BULL' in regime:
                summary = f"Ranging market with bullish bias - look for buys at support"
            elif 'BEAR' in regime:
                summary = f"Ranging market with bearish bias - look for sells at resistance"
            else:
                summary = f"Ranging market - suitable for mean reversion strategies"
        
        elif 'VOLATILE' in regime:
            summary = f"Volatile market - expect large moves, use wider stops"
        
        else:
            summary = f"Mixed market conditions, bias: {bias}"
        
        # Add Wyckoff phase if known
        if wyckoff_phase != 'UNKNOWN':
            summary += f" | Wyckoff: {wyckoff_phase}"
        
        # Add volatility context
        if atr_pct > ATR_PERCENT_HIGH:
            summary += f" | High volatility ({atr_pct:.1%})"
        elif atr_pct < ATR_PERCENT_LOW:
            summary += f" | Low volatility ({atr_pct:.1%})"
        
        # Add trend strength
        if trending_score > 0.7:
            summary += " | Strong trend"
        elif trending_score < 0.3:
            summary += " | Weak trend, ranging"
        
        return summary
    
    def _default_regime_result(self) -> RegimeResult:
        """Default regime result when analysis fails"""
        return RegimeResult(
            regime=RegimeType.UNKNOWN.value,
            regime_type="UNKNOWN",
            confidence=0.3,
            bias="NEUTRAL",
            bias_score=0.0,
            trend_strength=0.5,
            volatility_state="NORMAL",
            liquidity_state="NORMAL",
            adx=20.0,
            atr_pct=0.02,
            slope=0.0,
            is_squeeze=False,
            squeeze_intensity=0.0,
            wyckoff_phase="UNKNOWN",
            wyckoff_confidence=0.0,
            recommended_strategies=["Default Strategies"],
            avoid_strategies=[],
            indicator_settings={},
            summary="Unable to determine market regime - using neutral default",
            details={}
        )


# ============================================================================
# SIMPLE WRAPPER FOR BACKWARD COMPATIBILITY
# ============================================================================

def detect_market_regime(df: pd.DataFrame, symbol: str = "") -> RegimeResult:
    """
    Simple wrapper function for backward compatibility
    """
    detector = MarketRegimeAdvanced()
    return detector.detect_regime(df, symbol)