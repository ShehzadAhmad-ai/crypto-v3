# market_regime.py - ENHANCED FOR HUMAN-LIKE ANALYSIS
"""
Market Regime Detection - Enhanced Version
- Detects market regime with confidence scoring
- Wyckoff phase detection (Accumulation, Markup, Distribution, Markdown)
- Volatility and liquidity regimes
- Regime-specific recommendations
- All thresholds configurable from config.py
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from scipy import stats
from scipy.signal import argrelextrema
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from config import Config
from logger import log

# EPSILON for division safety
EPSILON = 1e-10


class RegimeType(Enum):
    """Primary regime classifications"""
    STRONG_BULL_TREND = "STRONG_BULL_TREND"
    BULL_TREND = "BULL_TREND"
    WEAK_BULL_TREND = "WEAK_BULL_TREND"
    RANGING_BULL_BIAS = "RANGING_BULL_BIAS"
    RANGING_NEUTRAL = "RANGING_NEUTRAL"
    RANGING_BEAR_BIAS = "RANGING_BEAR_BIAS"
    WEAK_BEAR_TREND = "WEAK_BEAR_TREND"
    BEAR_TREND = "BEAR_TREND"
    STRONG_BEAR_TREND = "STRONG_BEAR_TREND"
    VOLATILE_EXPANSION = "VOLATILE_EXPANSION"
    VOLATILE_CONTRACTION = "VOLATILE_CONTRACTION"
    UNKNOWN = "UNKNOWN"


class WyckoffPhase(Enum):
    """Wyckoff market phases"""
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    RE_ACCUMULATION = "RE_ACCUMULATION"
    RE_DISTRIBUTION = "RE_DISTRIBUTION"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeResult:
    """Complete regime analysis result"""
    # Primary regime
    regime: str
    regime_type: str
    regime_confidence: float  # 0-1, how confident we are in this classification
    
    # Directional bias
    bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    bias_score: float  # -1 (bearish) to +1 (bullish)
    
    # Strength metrics
    trend_strength: float  # 0-1
    volatility_state: str
    liquidity_state: str
    
    # Additional metrics
    adx: float
    atr_pct: float
    slope: float
    hurst: float
    trending_score: float
    risk_on_score: float
    is_squeeze: bool
    squeeze_intensity: float
    
    # Wyckoff phase
    wyckoff_phase: str
    wyckoff_confidence: float
    
    # Strategy recommendations
    recommended_strategies: List[str]
    avoid_strategies: List[str]
    
    # Dynamic indicator settings (for technical_analyzer)
    indicator_settings: Dict[str, Any]
    
    # Human-readable summary
    summary: str
    
    # Raw details for debugging
    details: Dict[str, Any]


class MarketRegimeDetector:
    """
    Enhanced market regime detection with configurable thresholds
    All thresholds can be set in config.py
    """
    
    def __init__(self):
        self.regime_history = []
        
        # Load thresholds from config (with defaults)
        self.trend_adx_threshold = getattr(Config, 'TREND_REGIME_ADX_THRESHOLD', 25.0)
        self.range_adx_threshold = getattr(Config, 'RANGE_REGIME_ADX_THRESHOLD', 20.0)
        self.high_vol_atr_pct = getattr(Config, 'MAX_VOLATILITY_ATR_PCT', 0.03)
        self.low_vol_atr_pct = getattr(Config, 'MIN_VOLATILITY_ATR_PCT', 0.002)
        
        # Additional configurable thresholds
        self.strong_trend_adx = getattr(Config, 'STRONG_TREND_ADX_THRESHOLD', 40.0)
        self.weak_trend_adx = getattr(Config, 'WEAK_TREND_ADX_THRESHOLD', 22.0)
        self.bullish_slope_threshold = getattr(Config, 'BULLISH_SLOPE_THRESHOLD', 0.01)
        self.bearish_slope_threshold = getattr(Config, 'BEARISH_SLOPE_THRESHOLD', -0.01)
        self.hurst_trend_threshold = getattr(Config, 'HURST_TREND_THRESHOLD', 0.55)
        
        log.info("MarketRegimeDetector initialized with configurable thresholds")
    
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Legacy method - returns dict for backward compatibility
        New code should use detect_regime_complete() for full analysis
        """
        if df is None or df.empty or len(df) < 100:
            return self._default_regime()
        
        try:
            result = self.detect_regime_complete(df)
            return {
                'regime': result.regime,
                'type': result.regime_type,
                'score': result.trend_strength,
                'atr_pct': result.atr_pct,
                'adx': result.adx,
                'slope': result.slope,
                'hurst': result.hurst,
                'volatility_regime': result.volatility_state,
                'liquidity_regime': result.liquidity_state,
                'is_bollinger_squeeze': result.is_squeeze,
                'squeeze_intensity': result.squeeze_intensity,
                'trending_score': result.trending_score,
                'risk_on_score': result.risk_on_score,
                'market_phase': result.wyckoff_phase,
                'details': result.details
            }
        except Exception as e:
            log.error(f"Error detecting regime: {e}")
            return self._default_regime()
    
    def detect_regime_complete(self, df: pd.DataFrame) -> RegimeResult:
        """
        Complete regime detection with all metrics for human-like analysis
        """
        if df is None or df.empty or len(df) < 100:
            return self._default_regime_result()
        
        try:
            # ===== STEP 1: CALCULATE BASE METRICS =====
            adx = float(df['adx'].iloc[-1]) if 'adx' in df else 20.0
            atr_pct = self._calculate_atr_pct(df)
            slope = self._calculate_slope(df)
            hurst = self._calculate_hurst_exponent(df['close'].values)
            
            # ===== STEP 2: REGIME METRICS =====
            vol_regime = self._detect_volatility_regime(df)
            liq_regime = self._detect_liquidity_regime(df)
            bb_squeeze, bb_intensity = self._detect_bollinger_squeeze(df)
            trending_score = self._calculate_trending_score(df, hurst, adx)
            risk_on_score = self._calculate_risk_on_score(df)
            wyckoff_phase, wyckoff_conf = self._detect_wyckoff_phase(df)
            
            # ===== STEP 3: DETERMINE REGIME TYPE =====
            regime, regime_type, regime_confidence = self._classify_regime(
                adx, slope, atr_pct, trending_score
            )
            
            # ===== STEP 4: DETERMINE DIRECTIONAL BIAS =====
            bias, bias_score = self._calculate_directional_bias(df, adx, slope, trending_score)
            
            # ===== STEP 5: DYNAMIC INDICATOR SETTINGS =====
            indicator_settings = self._get_dynamic_indicator_settings(regime, bias_score, atr_pct)
            
            # ===== STEP 6: STRATEGY RECOMMENDATIONS =====
            recommended, avoid = self._get_strategy_recommendations(regime, bias_score, trending_score)
            
            # ===== STEP 7: HUMAN-READABLE SUMMARY =====
            summary = self._generate_regime_summary(
                regime, bias, bias_score, trending_score, wyckoff_phase, atr_pct
            )
            
            # Create the result object
            result = RegimeResult(
                regime=regime,
                regime_type=regime_type,
                regime_confidence=regime_confidence,
                bias=bias,
                bias_score=bias_score,
                trend_strength=trending_score,
                volatility_state=vol_regime,
                liquidity_state=liq_regime,
                adx=round(adx, 1),
                atr_pct=round(atr_pct, 4),
                slope=round(slope, 4),
                hurst=round(hurst, 3),
                trending_score=round(trending_score, 3),
                risk_on_score=round(risk_on_score, 3),
                is_squeeze=bb_squeeze,
                squeeze_intensity=round(bb_intensity, 3),
                wyckoff_phase=wyckoff_phase,
                wyckoff_confidence=wyckoff_conf,
                recommended_strategies=recommended,
                avoid_strategies=avoid,
                indicator_settings=indicator_settings,
                summary=summary,
                details={
                    'adx': adx,
                    'atr_pct': atr_pct,
                    'slope': slope,
                    'hurst': hurst,
                    'trending_score': trending_score,
                    'risk_on_score': risk_on_score,
                    'vol_regime': vol_regime,
                    'liq_regime': liq_regime
                }
            )
            
            # ===== STORE IN HISTORY FOR HELPER METHODS =====
            # This enables is_trending(), is_ranging(), get_regime_summary()
            self.regime_history.append({
                'regime': regime,
                'type': regime_type,
                'score': trending_score,
                'trending_score': trending_score,
                'bias': bias,
                'bias_score': bias_score,
                'volatility_regime': vol_regime,
                'liquidity_regime': liq_regime,
                'is_bollinger_squeeze': bb_squeeze,
                'squeeze_intensity': bb_intensity,
                'risk_on_score': risk_on_score,
                'market_phase': wyckoff_phase,
                'timestamp': datetime.now()
            })
            
            # Keep history size manageable
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            return result
            
        except Exception as e:
            log.error(f"Error in complete regime detection: {e}")
            return self._default_regime_result()
    
    # ==================== PRIVATE METHODS ====================
    
    def _calculate_atr_pct(self, df: pd.DataFrame) -> float:
        """Calculate ATR as percentage of price - calculates ATR if missing"""
        try:
            # If ATR column exists, use it
            if 'atr' in df and not pd.isna(df['atr'].iloc[-1]) and df['close'].iloc[-1] > 0:
                return float(df['atr'].iloc[-1] / df['close'].iloc[-1])
            
            # Calculate ATR vectorized
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range components (vectorized)
            hl = high - low
            hc = np.abs(high - np.roll(close, 1))
            lc = np.abs(low - np.roll(close, 1))
            
            # True Range is max of the three
            tr = np.maximum(hl, np.maximum(hc, lc))
            
            # Calculate ATR with rolling 14-period mean
            atr = pd.Series(tr).rolling(window=14).mean().ffill()
            
            current_atr = atr.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_atr > 0 and current_price > 0:
                return float(current_atr / current_price)
            
            return 0.02
            
        except Exception as e:
            log.warning(f"Error calculating ATR: {e}, using fallback 0.02")
            return 0.02
    
    def _calculate_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate price slope over specified period"""
        try:
            if len(df) >= period:
                return (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_hurst_exponent(self, price_series: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst Exponent for trend persistence"""
        try:
            if len(price_series) < 100:
                return 0.5
            
            returns = np.diff(np.log(price_series + EPSILON))
            lags = range(2, min(max_lag, len(returns) // 4))
            tau = []
            
            for lag in lags:
                diff = np.array([returns[i] - returns[i - lag] for i in range(lag, len(returns))])
                tau.append(np.std(diff))
            
            if len(tau) > 0:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
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
            volume_sma = volume.rolling(20).mean()
            current_vol_ratio = volume.iloc[-1] / (volume_sma.iloc[-1] + EPSILON)
            volume_trend = volume.iloc[-10:].mean() / (volume.iloc[-20:-10].mean() + EPSILON) if len(volume) >= 20 else 1
            
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
    
    def _detect_bollinger_squeeze(self, df: pd.DataFrame, period: int = 20) -> Tuple[bool, float]:
        """Detect Bollinger Band squeeze"""
        try:
            if len(df) < period + 10:
                return False, 0.0
            
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            bandwidth = (upper - lower) / (sma + EPSILON)
            hist_bandwidth = bandwidth.rolling(period * 3).mean()
            current_bandwidth = bandwidth.iloc[-1]
            hist_bandwidth_val = hist_bandwidth.iloc[-1] if not pd.isna(hist_bandwidth.iloc[-1]) else current_bandwidth
            
            if hist_bandwidth_val > 0:
                squeeze_ratio = current_bandwidth / hist_bandwidth_val
                is_squeeze = squeeze_ratio < 0.7
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
    
    def _calculate_atr_pct(self, df: pd.DataFrame) -> float:
        """Calculate ATR as percentage of price - calculates ATR if missing"""
        try:
            # If ATR column exists and is valid, use it
            if 'atr' in df and not pd.isna(df['atr'].iloc[-1]) and df['close'].iloc[-1] > 0:
                return float(df['atr'].iloc[-1] / df['close'].iloc[-1])
            
            # Calculate ATR from OHLCV data (vectorized)
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range components
            hl = high - low
            hc = np.abs(high - np.roll(close, 1))
            lc = np.abs(low - np.roll(close, 1))
            
            # True Range is maximum of the three
            tr = np.maximum(hl, np.maximum(hc, lc))
            
            # ATR = 14-period rolling mean of True Range
            atr = pd.Series(tr).rolling(window=14).mean().ffill()
            
            current_atr = atr.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_atr > 0 and current_price > 0:
                return float(current_atr / current_price)
            
            # Fallback if calculation fails
            return 0.02
            
        except Exception as e:
            log.warning(f"Error calculating ATR: {e}, using fallback 0.02")
            return 0.02
        

        
    def _calculate_risk_on_score(self, df: pd.DataFrame) -> float:
        """Calculate risk-on score (0-1) - higher = risk-on environment"""
        try:
            if len(df) < 50:
                return 0.5
            
            scores = []
            
            # Volume trend
            if 'volume' in df:
                vol_trend = df['volume'].iloc[-5:].mean() / (df['volume'].iloc[-20:-5].mean() + EPSILON) if len(df) >= 20 else 1
                vol_score = min(1, max(0, (vol_trend - 0.7) / 0.8))
                scores.append(vol_score)
            
            # Volatility (lower volatility = more risk-on)
            # Calculate ATR if missing
            if 'atr' in df and not pd.isna(df['atr'].iloc[-1]):
                atr_value = df['atr'].iloc[-1]
            else:
                # Calculate ATR from OHLCV
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values
                
                hl = high - low
                hc = np.abs(high - np.roll(close, 1))
                lc = np.abs(low - np.roll(close, 1))
                tr = np.maximum(hl, np.maximum(hc, lc))
                atr_value = pd.Series(tr).rolling(window=14).mean().ffill().iloc[-1]
            
            atr_pct = atr_value / (df['close'].iloc[-1] + EPSILON)
            
            if atr_pct < 0.01:
                vol_score = 0.2
            elif atr_pct < 0.02:
                vol_score = 0.5
            elif atr_pct < 0.03:
                vol_score = 0.7
            elif atr_pct < 0.05:
                vol_score = 0.6
            else:
                vol_score = 0.3
            scores.append(vol_score)
            
            # RSI (higher RSI = more risk-on)
            if 'rsi' in df:
                rsi = df['rsi'].iloc[-1]
                if rsi > 70:
                    rsi_score = 0.9
                elif rsi > 60:
                    rsi_score = 0.8
                elif rsi > 50:
                    rsi_score = 0.6
                elif rsi > 40:
                    rsi_score = 0.4
                else:
                    rsi_score = 0.2
                scores.append(rsi_score)
            
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
            if slope > self.bullish_slope_threshold:
                regime = "STRONG_BULL_TREND"
                regime_type = "TREND"
                confidence = min(0.95, 0.7 + (adx - self.strong_trend_adx) / 50)
            elif slope < self.bearish_slope_threshold:
                regime = "STRONG_BEAR_TREND"
                regime_type = "TREND"
                confidence = min(0.95, 0.7 + (adx - self.strong_trend_adx) / 50)
            else:
                regime = "SIDEWAYS"
                regime_type = "CONSOLIDATION"
                confidence = 0.5
        
        # Moderate trend detection
        elif adx > self.trend_adx_threshold:
            if slope > self.bullish_slope_threshold:
                regime = "BULL_TREND"
                regime_type = "TREND"
                confidence = 0.6 + (adx - self.trend_adx_threshold) / 50
            elif slope < self.bearish_slope_threshold:
                regime = "BEAR_TREND"
                regime_type = "TREND"
                confidence = 0.6 + (adx - self.trend_adx_threshold) / 50
            else:
                regime = "SIDEWAYS"
                regime_type = "CONSOLIDATION"
                confidence = 0.4
        
        # Weak trend detection
        elif adx > self.weak_trend_adx:
            if slope > self.bullish_slope_threshold:
                regime = "WEAK_BULL_TREND"
                regime_type = "TREND"
                confidence = 0.5
            elif slope < self.bearish_slope_threshold:
                regime = "WEAK_BEAR_TREND"
                regime_type = "TREND"
                confidence = 0.5
            else:
                regime = "RANGING_NEUTRAL"
                regime_type = "RANGE"
                confidence = 0.6
        
        # Ranging market
        elif adx < self.range_adx_threshold:
            # Check for volatility expansion/contraction
            if atr_pct > self.high_vol_atr_pct * 1.5:
                regime = "VOLATILE_EXPANSION"
                regime_type = "VOLATILE"
                confidence = 0.7
            elif atr_pct < self.low_vol_atr_pct * 1.5:
                regime = "VOLATILE_CONTRACTION"
                regime_type = "VOLATILE"
                confidence = 0.6
            else:
                # Determine ranging bias
                if slope > 0.005:
                    regime = "RANGING_BULL_BIAS"
                    regime_type = "RANGE"
                    confidence = 0.5
                elif slope < -0.005:
                    regime = "RANGING_BEAR_BIAS"
                    regime_type = "RANGE"
                    confidence = 0.5
                else:
                    regime = "RANGING_NEUTRAL"
                    regime_type = "RANGE"
                    confidence = 0.7
        else:
            regime = "UNKNOWN"
            regime_type = "UNKNOWN"
            confidence = 0.3
        
        return regime, regime_type, confidence
    
    def _calculate_directional_bias(self, df: pd.DataFrame, adx: float, slope: float, trending_score: float) -> Tuple[str, float]:
        """
        Calculate directional bias (-1 to 1) and label
        Returns: (bias_label, bias_score)
        """
        # Start with slope contribution
        bias_score = slope * 5  # Normalize slope to -1 to 1 range
        
        # Add EMA alignment contribution
        if all(col in df for col in ['ema_8', 'ema_21', 'ema_50']):
            price = df['close'].iloc[-1]
            e8 = df['ema_8'].iloc[-1]
            e21 = df['ema_21'].iloc[-1]
            e50 = df['ema_50'].iloc[-1]
            
            if price > e8 > e21 > e50:
                bias_score += 0.3
            elif price < e8 < e21 < e50:
                bias_score -= 0.3
            elif price > e8 and price > e21:
                bias_score += 0.15
            elif price < e8 and price < e21:
                bias_score -= 0.15
        
        # Add ADX contribution (strong trend reinforces bias)
        if adx > self.trend_adx_threshold:
            bias_score = bias_score * (1 + min(0.5, (adx - self.trend_adx_threshold) / 50))
        
        # Clamp to -1 to 1
        bias_score = max(-1.0, min(1.0, bias_score))
        
        # Determine bias label
        if bias_score > 0.5:
            bias = "STRONG_BULLISH"
        elif bias_score > 0.2:
            bias = "BULLISH"
        elif bias_score > -0.2:
            bias = "NEUTRAL"
        elif bias_score > -0.5:
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
            
            local_max = argrelextrema(price, np.greater, order=10)[0]
            local_min = argrelextrema(price, np.less, order=10)[0]
            
            if len(local_max) < 2 or len(local_min) < 2:
                return 'UNKNOWN', 0.0
            
            recent_max_idx = [x for x in local_max if x > len(price) - 50]
            recent_min_idx = [x for x in local_min if x > len(price) - 50]
            
            recent_max = price[recent_max_idx] if recent_max_idx else []
            recent_min = price[recent_min_idx] if recent_min_idx else []
            
            recent_vol = volume[-50:]
            vol_trend = recent_vol[-10:].mean() / (recent_vol[:10].mean() + EPSILON) if len(recent_vol) >= 20 else 1
            
            if len(recent_max) >= 2 and len(recent_min) >= 2:
                last_max = recent_max[-1]
                prev_max = recent_max[-2]
                last_min = recent_min[-1]
                prev_min = recent_min[-2]
                
                # MARKUP: Higher highs, higher lows, volume increasing
                if last_max > prev_max and last_min > prev_min and vol_trend > 1.1:
                    return 'MARKUP', 0.8
                
                # MARKDOWN: Lower highs, lower lows, volume increasing
                elif last_max < prev_max and last_min < prev_min and vol_trend > 1.1:
                    return 'MARKDOWN', 0.8
                
                # ACCUMULATION: Range-bound, volume decreasing
                elif abs(last_max - prev_max) / (prev_max + EPSILON) < 0.05 and vol_trend < 0.9:
                    return 'ACCUMULATION', 0.7
                
                # DISTRIBUTION: Range-bound, volume increasing then weakening
                elif abs(last_max - prev_max) / (prev_max + EPSILON) < 0.05 and vol_trend > 1.2:
                    return 'DISTRIBUTION', 0.7
                
                # Re-accumulation after markup
                elif vol_trend < 0.8 and last_min > prev_min and abs(last_max - prev_max) / (prev_max + EPSILON) < 0.03:
                    return 'RE_ACCUMULATION', 0.6
                
                # Re-distribution after markdown
                elif vol_trend > 1.3 and last_max < prev_max and abs(last_min - prev_min) / (prev_min + EPSILON) < 0.03:
                    return 'RE_DISTRIBUTION', 0.6
            
            return 'UNKNOWN', 0.0
        except Exception:
            return 'UNKNOWN', 0.0
    
    def _get_dynamic_indicator_settings(self, regime: str, bias_score: float, atr_pct: float) -> Dict[str, Any]:
        """
        Get dynamic indicator settings based on regime
        All settings can be overridden in config.py
        """
        # Base settings (default)
        settings = {
            # RSI settings
            'rsi_period': getattr(Config, 'RSI_PERIOD', 14),
            'rsi_oversold': getattr(Config, 'RSI_OVERSOLD', 30),
            'rsi_overbought': getattr(Config, 'RSI_OVERBOUGHT', 70),
            
            # MACD settings
            'macd_fast': getattr(Config, 'MACD_FAST', 12),
            'macd_slow': getattr(Config, 'MACD_SLOW', 26),
            'macd_signal': getattr(Config, 'MACD_SIGNAL', 9),
            
            # EMA settings
            'ema_fast': getattr(Config, 'EMA_FAST', 8),
            'ema_medium': getattr(Config, 'EMA_MEDIUM', 21),
            'ema_slow': getattr(Config, 'EMA_SLOW', 50),
            'ema_very_slow': getattr(Config, 'EMA_VERY_SLOW', 200),
            
            # Bollinger Bands
            'bb_period': getattr(Config, 'BB_PERIOD', 20),
            'bb_std': getattr(Config, 'BB_STD', 2.0),
            
            # ADX settings
            'adx_period': getattr(Config, 'ADX_PERIOD', 14),
            'adx_trend_threshold': getattr(Config, 'ADX_TREND_THRESHOLD', 25),
            
            # Volume settings
            'volume_ma_period': getattr(Config, 'VOLUME_MA_PERIOD', 20),
            'volume_spike_threshold': getattr(Config, 'VOLUME_SPIKE_THRESHOLD', 1.5),
            
            # ATR settings
            'atr_period': getattr(Config, 'ATR_PERIOD', 14),
            'atr_multiplier_stop': getattr(Config, 'ATR_STOP_MULTIPLIER', 1.5),
            
            # Threshold adjustments
            'rsi_bull_threshold': 55,
            'rsi_bear_threshold': 45,
        }
        
        # Adjust based on regime
        if 'TREND' in regime or 'BULL' in regime or 'BEAR' in regime:
            # Trending market - faster indicators
            settings['rsi_period'] = max(8, settings['rsi_period'] - 4)
            settings['macd_fast'] = max(8, settings['macd_fast'] - 4)
            settings['macd_slow'] = max(20, settings['macd_slow'] - 6)
            settings['ema_fast'] = max(5, settings['ema_fast'] - 3)
            settings['ema_medium'] = max(13, settings['ema_medium'] - 8)
            
            # RSI thresholds shift with trend
            if 'BULL' in regime:
                settings['rsi_oversold'] = 40  # Higher oversold in bull trend
                settings['rsi_overbought'] = 85
                settings['rsi_bull_threshold'] = 50
            elif 'BEAR' in regime:
                settings['rsi_oversold'] = 20
                settings['rsi_overbought'] = 60  # Lower overbought in bear trend
                settings['rsi_bear_threshold'] = 50
        
        elif 'RANGING' in regime:
            # Ranging market - slower indicators, mean reversion focus
            settings['rsi_period'] = 14
            settings['rsi_oversold'] = 30
            settings['rsi_overbought'] = 70
            settings['macd_fast'] = 12
            settings['macd_slow'] = 26
            settings['ema_fast'] = 8
            settings['ema_medium'] = 21
            settings['bb_std'] = 2.2  # Wider bands for ranging
        
        elif 'VOLATILE' in regime:
            # Volatile market - more conservative settings
            settings['volume_spike_threshold'] = 2.0
            settings['atr_multiplier_stop'] = 2.0
            settings['bb_std'] = 2.5
        
        return settings
    
    def _get_strategy_recommendations(self, regime: str, bias_score: float, trending_score: float) -> Tuple[List[str], List[str]]:
        """Get strategy recommendations based on regime"""
        recommended = []
        avoid = []
        
        if 'TREND' in regime or 'BULL' in regime or 'BEAR' in regime:
            recommended = [
                'Trend Following',
                'EMA Trend',
                'MACD Crossover',
                'Breakout Strategies',
                'Smart Money (BOS/CHOCH)'
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
                'Bollinger Bands Mean Reversion',
                'VWAP Reversion',
                'RSI Oversold/Overbought'
            ]
            avoid = [
                'Trend Following',
                'Breakout Strategies',
                'Volatility Breakout'
            ]
        
        elif 'VOLATILE' in regime:
            recommended = [
                'Volatility Breakout',
                'Momentum Trading',
                'Smart Money (Liquidity Sweeps)',
                'Short-term Scalping'
            ]
            avoid = [
                'Mean Reversion',
                'Range Trading',
                'Large Position Sizes'
            ]
        
        # Adjust based on bias strength
        if abs(bias_score) > 0.7:
            recommended.append('Strong Momentum')
        elif abs(bias_score) < 0.2:
            recommended.append('Wait for Direction')
        
        return recommended[:5], avoid[:3]
    
    def _generate_regime_summary(self, regime: str, bias: str, bias_score: float, 
                                  trending_score: float, wyckoff_phase: str, atr_pct: float) -> str:
        """Generate human-readable regime summary"""
        
        # Base summary
        if 'TREND' in regime:
            if 'BULL' in regime:
                summary = f"Strong bullish trend detected (ADX: trending, bias: {bias})"
            elif 'BEAR' in regime:
                summary = f"Strong bearish trend detected (ADX: trending, bias: {bias})"
            else:
                summary = f"Trending market with {bias.lower()} bias"
        elif 'RANGING' in regime:
            summary = f"Ranging market with {bias.lower()} bias - suitable for mean reversion"
        elif 'VOLATILE' in regime:
            summary = f"Volatile market - expect large moves, use tighter stops"
        else:
            summary = f"Mixed market conditions, bias: {bias}"
        
        # Add Wyckoff phase if known
        if wyckoff_phase != 'UNKNOWN':
            summary += f" | Wyckoff phase: {wyckoff_phase}"
        
        # Add volatility context
        if atr_pct > 0.03:
            summary += f" | High volatility ({atr_pct:.1%})"
        elif atr_pct < 0.01:
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
            regime='UNKNOWN',
            regime_type='UNKNOWN',
            regime_confidence=0.3,
            bias='NEUTRAL',
            bias_score=0.0,
            trend_strength=0.5,
            volatility_state='NORMAL',
            liquidity_state='NORMAL',
            adx=20.0,
            atr_pct=0.02,
            slope=0.0,
            hurst=0.5,
            trending_score=0.5,
            risk_on_score=0.5,
            is_squeeze=False,
            squeeze_intensity=0.0,
            wyckoff_phase='UNKNOWN',
            wyckoff_confidence=0.0,
            recommended_strategies=['Default Strategies'],
            avoid_strategies=[],
            indicator_settings={},
            summary='Unable to determine market regime',
            details={}
        )
    
    def _default_regime(self) -> Dict[str, Any]:
        """Legacy default regime dict"""
        return {
            'regime': 'UNKNOWN',
            'type': 'UNKNOWN',
            'score': 0.5,
            'atr_pct': 0.02,
            'adx': 20.0,
            'slope': 0.0,
            'hurst': 0.5,
            'volatility_regime': 'NORMAL',
            'liquidity_regime': 'NORMAL',
            'is_bollinger_squeeze': False,
            'squeeze_intensity': 0.0,
            'trending_score': 0.5,
            'risk_on_score': 0.5,
            'market_phase': 'UNKNOWN',
            'details': {}
        }
    
    # ==================== LEGACY COMPATIBILITY METHODS ====================
    
    def is_trending(self) -> bool:
        if not self.regime_history:
            return False
        return self.regime_history[-1]['type'] == 'TREND'
    
    def is_ranging(self) -> bool:
        if not self.regime_history:
            return False
        return self.regime_history[-1]['type'] in ['RANGE', 'CONSOLIDATION']
    
    def get_regime_summary(self) -> str:
        """Get current regime as human-readable summary"""
        if not self.regime_history:
            return "No regime data available"
        
        latest = self.regime_history[-1]
        return f"Regime: {latest['regime']} | Trend Strength: {latest['trending_score']:.2f} | Volatility: {latest['volatility_regime']}"