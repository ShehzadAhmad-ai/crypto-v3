# divergence_detector.py - Advanced Divergence Detection Expert
"""
Advanced Divergence Detection Expert
- Detects regular and hidden divergences on multiple indicators
- RSI, MACD, OBV, Stochastic, CCI, MFI divergences
- Calculates divergence strength and confidence
- Provides divergence scoring for signal generation
- All thresholds from ta_config.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.signal import argrelextrema
from datetime import datetime

# Import configuration and core classes
from .ta_config import *
from .ta_core import DivergenceResult

# Try to import logger
try:
    from logger import log
except ImportError:
    import logging
    log = logging.getLogger(__name__)


class DivergenceDetector:
    """
    Advanced Divergence Detection Expert
    Detects regular and hidden divergences across multiple indicators
    """
    
    def __init__(self):
        """Initialize the divergence detector with thresholds from config"""
        self.lookback = DIVERGENCE_LOOKBACK
        self.min_swing_distance = DIVERGENCE_MIN_SWING_DISTANCE
        self.divergence_weights = DIVERGENCE_WEIGHTS.copy()
        
        # Track detected divergences for analysis
        self.divergence_history: List[DivergenceResult] = []
        
        log.info(f"DivergenceDetector initialized with lookback={self.lookback}")
    
    # ============================================================================
    # MAIN PUBLIC METHODS
    # ============================================================================
    
    def detect_all_divergences(self, df: pd.DataFrame) -> Tuple[List[DivergenceResult], float, bool, bool]:
        """
        Detect all types of divergences across multiple indicators
        
        Args:
            df: DataFrame with OHLCV data and indicator columns
        
        Returns:
            Tuple of (divergences list, overall score, has_bullish, has_bearish)
        """
        if df is None or df.empty or len(df) < self.lookback + 10:
            log.warning("Insufficient data for divergence detection")
            return [], 0.0, False, False
        
        try:
            divergences = []
            
            # RSI Divergences
            if 'rsi' in df.columns:
                rsi_divs = self._detect_rsi_divergences(df)
                divergences.extend(rsi_divs)
            
            # MACD Divergences
            if 'macd_hist' in df.columns:
                macd_divs = self._detect_macd_divergences(df)
                divergences.extend(macd_divs)
            
            # OBV Divergences
            if 'obv' in df.columns:
                obv_divs = self._detect_obv_divergences(df)
                divergences.extend(obv_divs)
            
            # Stochastic Divergences
            if 'stoch_k' in df.columns:
                stoch_divs = self._detect_stochastic_divergences(df)
                divergences.extend(stoch_divs)
            
            # CCI Divergences
            if 'cci' in df.columns:
                cci_divs = self._detect_cci_divergences(df)
                divergences.extend(cci_divs)
            
            # MFI Divergences
            if 'mfi' in df.columns:
                mfi_divs = self._detect_mfi_divergences(df)
                divergences.extend(mfi_divs)
            
            # Calculate overall divergence score
            overall_score = self._calculate_divergence_score(divergences)
            has_bullish = any(d.type == "BULLISH" for d in divergences)
            has_bearish = any(d.type == "BEARISH" for d in divergences)
            
            # Store in history
            for div in divergences:
                self.divergence_history.append(div)
            
            # Keep history manageable
            if len(self.divergence_history) > 100:
                self.divergence_history = self.divergence_history[-100:]
            
            log.debug(f"Detected {len(divergences)} divergences (Bullish: {has_bullish}, Bearish: {has_bearish})")
            
            return divergences, round(overall_score, 3), has_bullish, has_bearish
            
        except Exception as e:
            log.error(f"Error in divergence detection: {e}")
            return [], 0.0, False, False
    
    def get_best_divergence(self, divergences: List[DivergenceResult]) -> Optional[DivergenceResult]:
        """
        Get the strongest divergence from the list
        
        Args:
            divergences: List of DivergenceResult objects
        
        Returns:
            Strongest divergence or None
        """
        if not divergences:
            return None
        
        # Sort by strength and return the strongest
        return max(divergences, key=lambda d: d.strength)
    
    def get_divergence_bias(self, divergences: List[DivergenceResult]) -> float:
        """
        Get net divergence bias (-1 to 1)
        
        Returns:
            Positive = bullish divergence dominant, Negative = bearish divergence dominant
        """
        if not divergences:
            return 0.0
        
        bullish_strength = sum(d.strength for d in divergences if d.type == "BULLISH")
        bearish_strength = sum(d.strength for d in divergences if d.type == "BEARISH")
        total_strength = bullish_strength + bearish_strength
        
        if total_strength > 0:
            return (bullish_strength - bearish_strength) / total_strength
        return 0.0
    
    # ============================================================================
    # INDICATOR-SPECIFIC DIVERGENCE DETECTION
    # ============================================================================
    
    def _detect_rsi_divergences(self, df: pd.DataFrame) -> List[DivergenceResult]:
        """
        Detect RSI divergences (regular and hidden)
        
        Regular Bullish: Price makes lower low, RSI makes higher low
        Regular Bearish: Price makes higher high, RSI makes lower high
        Hidden Bullish: Price makes higher low, RSI makes lower low
        Hidden Bearish: Price makes lower high, RSI makes higher high
        """
        divergences = []
        price = df['close'].values
        rsi = df['rsi'].values
        
        # Find swing points
        price_highs_idx = argrelextrema(price, np.greater, order=self.min_swing_distance)[0]
        price_lows_idx = argrelextrema(price, np.less, order=self.min_swing_distance)[0]
        rsi_highs_idx = argrelextrema(rsi, np.greater, order=self.min_swing_distance)[0]
        rsi_lows_idx = argrelextrema(rsi, np.less, order=self.min_swing_distance)[0]
        
        # Regular Bullish Divergence: Price lower low, RSI higher low
        for i in range(1, len(price_lows_idx)):
            idx1 = price_lows_idx[i-1]
            idx2 = price_lows_idx[i]
            
            if idx2 < len(price) and idx1 < len(rsi) and idx2 < len(rsi):
                if price[idx2] < price[idx1] and rsi[idx2] > rsi[idx1]:
                    # Check if RSI low is actually a low
                    if idx2 in rsi_lows_idx or rsi[idx2] < rsi[idx2-1] and rsi[idx2] < rsi[idx2+1] if idx2+1 < len(rsi) else True:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], rsi[idx1], rsi[idx2], "BULLISH"
                        )
                        divergences.append(DivergenceResult(
                            indicator="RSI",
                            type="BULLISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx2]),
                            price_high=float(price[idx1]),
                            indicator_low=float(rsi[idx2]),
                            indicator_high=float(rsi[idx1]),
                            is_hidden=False
                        ))
        
        # Regular Bearish Divergence: Price higher high, RSI lower high
        for i in range(1, len(price_highs_idx)):
            idx1 = price_highs_idx[i-1]
            idx2 = price_highs_idx[i]
            
            if idx2 < len(price) and idx1 < len(rsi) and idx2 < len(rsi):
                if price[idx2] > price[idx1] and rsi[idx2] < rsi[idx1]:
                    if idx2 in rsi_highs_idx or (rsi[idx2] > rsi[idx2-1] and rsi[idx2] > rsi[idx2+1] if idx2+1 < len(rsi) else True):
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], rsi[idx1], rsi[idx2], "BEARISH"
                        )
                        divergences.append(DivergenceResult(
                            indicator="RSI",
                            type="BEARISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx1]),
                            price_high=float(price[idx2]),
                            indicator_low=float(rsi[idx2]),
                            indicator_high=float(rsi[idx1]),
                            is_hidden=False
                        ))
        
        # Hidden Bullish Divergence: Price higher low, RSI lower low
        for i in range(1, len(price_lows_idx)):
            for j in range(1, len(rsi_lows_idx)):
                idx1 = price_lows_idx[i-1]
                idx2 = price_lows_idx[i]
                ridx1 = rsi_lows_idx[j-1]
                ridx2 = rsi_lows_idx[j]
                
                if abs(idx2 - ridx2) <= 5:  # Within 5 bars
                    if price[idx2] > price[idx1] and rsi[ridx2] < rsi[ridx1]:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], rsi[ridx1], rsi[ridx2], "BULLISH"
                        ) * 0.8  # Hidden divergence slightly weaker
                        divergences.append(DivergenceResult(
                            indicator="RSI",
                            type="BULLISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx1]),
                            price_high=float(price[idx2]),
                            indicator_low=float(rsi[ridx2]),
                            indicator_high=float(rsi[ridx1]),
                            is_hidden=True
                        ))
                        break
        
        # Hidden Bearish Divergence: Price lower high, RSI higher high
        for i in range(1, len(price_highs_idx)):
            for j in range(1, len(rsi_highs_idx)):
                idx1 = price_highs_idx[i-1]
                idx2 = price_highs_idx[i]
                ridx1 = rsi_highs_idx[j-1]
                ridx2 = rsi_highs_idx[j]
                
                if abs(idx2 - ridx2) <= 5:
                    if price[idx2] < price[idx1] and rsi[ridx2] > rsi[ridx1]:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], rsi[ridx1], rsi[ridx2], "BEARISH"
                        ) * 0.8
                        divergences.append(DivergenceResult(
                            indicator="RSI",
                            type="BEARISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx2]),
                            price_high=float(price[idx1]),
                            indicator_low=float(rsi[ridx2]),
                            indicator_high=float(rsi[ridx1]),
                            is_hidden=True
                        ))
                        break
        
        return divergences
    
    def _detect_macd_divergences(self, df: pd.DataFrame) -> List[DivergenceResult]:
        """
        Detect MACD histogram divergences
        """
        divergences = []
        price = df['close'].values
        macd_hist = df['macd_hist'].values
        
        # Find swing points
        price_highs_idx = argrelextrema(price, np.greater, order=self.min_swing_distance)[0]
        price_lows_idx = argrelextrema(price, np.less, order=self.min_swing_distance)[0]
        macd_highs_idx = argrelextrema(macd_hist, np.greater, order=self.min_swing_distance)[0]
        macd_lows_idx = argrelextrema(macd_hist, np.less, order=self.min_swing_distance)[0]
        
        # Regular Bullish Divergence
        for i in range(1, len(price_lows_idx)):
            idx1 = price_lows_idx[i-1]
            idx2 = price_lows_idx[i]
            
            if idx2 < len(price) and idx1 < len(macd_hist) and idx2 < len(macd_hist):
                if price[idx2] < price[idx1] and macd_hist[idx2] > macd_hist[idx1]:
                    if idx2 in macd_lows_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], macd_hist[idx1], macd_hist[idx2], "BULLISH"
                        )
                        divergences.append(DivergenceResult(
                            indicator="MACD",
                            type="BULLISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx2]),
                            price_high=float(price[idx1]),
                            indicator_low=float(macd_hist[idx2]),
                            indicator_high=float(macd_hist[idx1]),
                            is_hidden=False
                        ))
        
        # Regular Bearish Divergence
        for i in range(1, len(price_highs_idx)):
            idx1 = price_highs_idx[i-1]
            idx2 = price_highs_idx[i]
            
            if idx2 < len(price) and idx1 < len(macd_hist) and idx2 < len(macd_hist):
                if price[idx2] > price[idx1] and macd_hist[idx2] < macd_hist[idx1]:
                    if idx2 in macd_highs_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], macd_hist[idx1], macd_hist[idx2], "BEARISH"
                        )
                        divergences.append(DivergenceResult(
                            indicator="MACD",
                            type="BEARISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx1]),
                            price_high=float(price[idx2]),
                            indicator_low=float(macd_hist[idx2]),
                            indicator_high=float(macd_hist[idx1]),
                            is_hidden=False
                        ))
        
        return divergences
    
    def _detect_obv_divergences(self, df: pd.DataFrame) -> List[DivergenceResult]:
        """
        Detect OBV divergences (volume confirmation)
        """
        divergences = []
        price = df['close'].values
        obv = df['obv'].values
        
        # Find swing points
        price_highs_idx = argrelextrema(price, np.greater, order=self.min_swing_distance)[0]
        price_lows_idx = argrelextrema(price, np.less, order=self.min_swing_distance)[0]
        obv_highs_idx = argrelextrema(obv, np.greater, order=self.min_swing_distance)[0]
        obv_lows_idx = argrelextrema(obv, np.less, order=self.min_swing_distance)[0]
        
        # Bullish Divergence: Price lower low, OBV higher low
        for i in range(1, len(price_lows_idx)):
            idx1 = price_lows_idx[i-1]
            idx2 = price_lows_idx[i]
            
            if idx2 < len(price) and idx1 < len(obv) and idx2 < len(obv):
                if price[idx2] < price[idx1] and obv[idx2] > obv[idx1]:
                    if idx2 in obv_lows_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], obv[idx1], obv[idx2], "BULLISH"
                        )
                        divergences.append(DivergenceResult(
                            indicator="OBV",
                            type="BULLISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx2]),
                            price_high=float(price[idx1]),
                            indicator_low=float(obv[idx2]),
                            indicator_high=float(obv[idx1]),
                            is_hidden=False
                        ))
        
        # Bearish Divergence: Price higher high, OBV lower high
        for i in range(1, len(price_highs_idx)):
            idx1 = price_highs_idx[i-1]
            idx2 = price_highs_idx[i]
            
            if idx2 < len(price) and idx1 < len(obv) and idx2 < len(obv):
                if price[idx2] > price[idx1] and obv[idx2] < obv[idx1]:
                    if idx2 in obv_highs_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], obv[idx1], obv[idx2], "BEARISH"
                        )
                        divergences.append(DivergenceResult(
                            indicator="OBV",
                            type="BEARISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx1]),
                            price_high=float(price[idx2]),
                            indicator_low=float(obv[idx2]),
                            indicator_high=float(obv[idx1]),
                            is_hidden=False
                        ))
        
        return divergences
    
    def _detect_stochastic_divergences(self, df: pd.DataFrame) -> List[DivergenceResult]:
        """
        Detect Stochastic divergences
        """
        divergences = []
        price = df['close'].values
        stoch = df['stoch_k'].values
        
        # Find swing points
        price_highs_idx = argrelextrema(price, np.greater, order=self.min_swing_distance)[0]
        price_lows_idx = argrelextrema(price, np.less, order=self.min_swing_distance)[0]
        stoch_highs_idx = argrelextrema(stoch, np.greater, order=self.min_swing_distance)[0]
        stoch_lows_idx = argrelextrema(stoch, np.less, order=self.min_swing_distance)[0]
        
        # Bullish Divergence
        for i in range(1, len(price_lows_idx)):
            idx1 = price_lows_idx[i-1]
            idx2 = price_lows_idx[i]
            
            if idx2 < len(price) and idx1 < len(stoch) and idx2 < len(stoch):
                if price[idx2] < price[idx1] and stoch[idx2] > stoch[idx1]:
                    if idx2 in stoch_lows_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], stoch[idx1], stoch[idx2], "BULLISH"
                        ) * 0.7  # Stochastic divergences slightly weaker
                        divergences.append(DivergenceResult(
                            indicator="Stochastic",
                            type="BULLISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx2]),
                            price_high=float(price[idx1]),
                            indicator_low=float(stoch[idx2]),
                            indicator_high=float(stoch[idx1]),
                            is_hidden=False
                        ))
        
        # Bearish Divergence
        for i in range(1, len(price_highs_idx)):
            idx1 = price_highs_idx[i-1]
            idx2 = price_highs_idx[i]
            
            if idx2 < len(price) and idx1 < len(stoch) and idx2 < len(stoch):
                if price[idx2] > price[idx1] and stoch[idx2] < stoch[idx1]:
                    if idx2 in stoch_highs_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], stoch[idx1], stoch[idx2], "BEARISH"
                        ) * 0.7
                        divergences.append(DivergenceResult(
                            indicator="Stochastic",
                            type="BEARISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx1]),
                            price_high=float(price[idx2]),
                            indicator_low=float(stoch[idx2]),
                            indicator_high=float(stoch[idx1]),
                            is_hidden=False
                        ))
        
        return divergences
    
    def _detect_cci_divergences(self, df: pd.DataFrame) -> List[DivergenceResult]:
        """
        Detect CCI divergences
        """
        divergences = []
        price = df['close'].values
        cci = df['cci'].values
        
        # Find swing points
        price_highs_idx = argrelextrema(price, np.greater, order=self.min_swing_distance)[0]
        price_lows_idx = argrelextrema(price, np.less, order=self.min_swing_distance)[0]
        cci_highs_idx = argrelextrema(cci, np.greater, order=self.min_swing_distance)[0]
        cci_lows_idx = argrelextrema(cci, np.less, order=self.min_swing_distance)[0]
        
        # Bullish Divergence
        for i in range(1, len(price_lows_idx)):
            idx1 = price_lows_idx[i-1]
            idx2 = price_lows_idx[i]
            
            if idx2 < len(price) and idx1 < len(cci) and idx2 < len(cci):
                if price[idx2] < price[idx1] and cci[idx2] > cci[idx1]:
                    if idx2 in cci_lows_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], cci[idx1], cci[idx2], "BULLISH"
                        ) * 0.6
                        divergences.append(DivergenceResult(
                            indicator="CCI",
                            type="BULLISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx2]),
                            price_high=float(price[idx1]),
                            indicator_low=float(cci[idx2]),
                            indicator_high=float(cci[idx1]),
                            is_hidden=False
                        ))
        
        # Bearish Divergence
        for i in range(1, len(price_highs_idx)):
            idx1 = price_highs_idx[i-1]
            idx2 = price_highs_idx[i]
            
            if idx2 < len(price) and idx1 < len(cci) and idx2 < len(cci):
                if price[idx2] > price[idx1] and cci[idx2] < cci[idx1]:
                    if idx2 in cci_highs_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], cci[idx1], cci[idx2], "BEARISH"
                        ) * 0.6
                        divergences.append(DivergenceResult(
                            indicator="CCI",
                            type="BEARISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx1]),
                            price_high=float(price[idx2]),
                            indicator_low=float(cci[idx2]),
                            indicator_high=float(cci[idx1]),
                            is_hidden=False
                        ))
        
        return divergences
    
    def _detect_mfi_divergences(self, df: pd.DataFrame) -> List[DivergenceResult]:
        """
        Detect MFI divergences (volume-weighted)
        """
        divergences = []
        price = df['close'].values
        mfi = df['mfi'].values
        
        # Find swing points
        price_highs_idx = argrelextrema(price, np.greater, order=self.min_swing_distance)[0]
        price_lows_idx = argrelextrema(price, np.less, order=self.min_swing_distance)[0]
        mfi_highs_idx = argrelextrema(mfi, np.greater, order=self.min_swing_distance)[0]
        mfi_lows_idx = argrelextrema(mfi, np.less, order=self.min_swing_distance)[0]
        
        # Bullish Divergence
        for i in range(1, len(price_lows_idx)):
            idx1 = price_lows_idx[i-1]
            idx2 = price_lows_idx[i]
            
            if idx2 < len(price) and idx1 < len(mfi) and idx2 < len(mfi):
                if price[idx2] < price[idx1] and mfi[idx2] > mfi[idx1]:
                    if idx2 in mfi_lows_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], mfi[idx1], mfi[idx2], "BULLISH"
                        ) * 0.65
                        divergences.append(DivergenceResult(
                            indicator="MFI",
                            type="BULLISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx2]),
                            price_high=float(price[idx1]),
                            indicator_low=float(mfi[idx2]),
                            indicator_high=float(mfi[idx1]),
                            is_hidden=False
                        ))
        
        # Bearish Divergence
        for i in range(1, len(price_highs_idx)):
            idx1 = price_highs_idx[i-1]
            idx2 = price_highs_idx[i]
            
            if idx2 < len(price) and idx1 < len(mfi) and idx2 < len(mfi):
                if price[idx2] > price[idx1] and mfi[idx2] < mfi[idx1]:
                    if idx2 in mfi_highs_idx:
                        strength = self._calculate_divergence_strength(
                            price[idx1], price[idx2], mfi[idx1], mfi[idx2], "BEARISH"
                        ) * 0.65
                        divergences.append(DivergenceResult(
                            indicator="MFI",
                            type="BEARISH",
                            strength=strength,
                            start_bar=idx1,
                            end_bar=idx2,
                            price_low=float(price[idx1]),
                            price_high=float(price[idx2]),
                            indicator_low=float(mfi[idx2]),
                            indicator_high=float(mfi[idx1]),
                            is_hidden=False
                        ))
        
        return divergences
    
    # ============================================================================
    # STRENGTH CALCULATION METHODS
    # ============================================================================
    
    def _calculate_divergence_strength(self, price1: float, price2: float,
                                        ind1: float, ind2: float,
                                        divergence_type: str) -> float:
        """
        Calculate the strength of a divergence
        
        Args:
            price1: First price point
            price2: Second price point
            ind1: First indicator value
            ind2: Second indicator value
            divergence_type: "BULLISH" or "BEARISH"
        
        Returns:
            Strength score between 0 and 1
        """
        # Calculate price divergence magnitude
        price_diff_pct = abs(price2 - price1) / price1
        
        # Calculate indicator divergence magnitude
        if divergence_type == "BULLISH":
            # For bullish divergence: price down, indicator up
            ind_divergence = (ind2 - ind1) / (abs(ind1) + 1e-10)
            price_divergence = -(price2 - price1) / price1
        else:
            # For bearish divergence: price up, indicator down
            ind_divergence = -(ind2 - ind1) / (abs(ind1) + 1e-10)
            price_divergence = (price2 - price1) / price1
        
        # Combine scores
        strength = (abs(price_divergence) * 0.4 + abs(ind_divergence) * 0.6)
        
        # Normalize to 0-1 range
        strength = min(1.0, strength * 5)  # 5x multiplier since typical divergences are 2-10%
        
        return max(0.1, min(1.0, strength))
    
    def _calculate_divergence_score(self, divergences: List[DivergenceResult]) -> float:
        """
        Calculate overall divergence score (0-1)
        Higher score = bullish divergence dominant
        """
        if not divergences:
            return 0.5
        
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        for div in divergences:
            weight = self.divergence_weights.get(div.indicator.lower(), 0.5)
            if div.type == "BULLISH":
                bullish_score += div.strength * weight
            else:
                bearish_score += div.strength * weight
            total_weight += weight
        
        if total_weight > 0:
            net_score = (bullish_score - bearish_score) / total_weight
            # Convert from -1 to 1 range to 0-1 range
            return (net_score + 1) / 2
        
        return 0.5
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def clear_history(self):
        """Clear divergence history"""
        self.divergence_history.clear()
        log.info("Divergence history cleared")
    
    def get_divergence_summary(self, divergences: List[DivergenceResult]) -> Dict[str, Any]:
        """
        Get a summary of divergences
        
        Returns:
            Dictionary with summary statistics
        """
        if not divergences:
            return {
                "total_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "strongest": None,
                "by_indicator": {}
            }
        
        bullish_count = sum(1 for d in divergences if d.type == "BULLISH")
        bearish_count = sum(1 for d in divergences if d.type == "BEARISH")
        
        by_indicator = {}
        for d in divergences:
            if d.indicator not in by_indicator:
                by_indicator[d.indicator] = {"bullish": 0, "bearish": 0}
            if d.type == "BULLISH":
                by_indicator[d.indicator]["bullish"] += 1
            else:
                by_indicator[d.indicator]["bearish"] += 1
        
        strongest = max(divergences, key=lambda d: d.strength)
        
        return {
            "total_count": len(divergences),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "strongest": {
                "indicator": strongest.indicator,
                "type": strongest.type,
                "strength": strongest.strength
            },
            "by_indicator": by_indicator
        }


# ============================================================================
# SIMPLE WRAPPER FUNCTIONS
# ============================================================================

def detect_divergences(df: pd.DataFrame) -> Tuple[List[DivergenceResult], float, bool, bool]:
    """
    Simple wrapper function for divergence detection
    """
    detector = DivergenceDetector()
    return detector.detect_all_divergences(df)


def get_divergence_bias(divergences: List[DivergenceResult]) -> float:
    """
    Get net divergence bias from divergence list
    """
    detector = DivergenceDetector()
    return detector.get_divergence_bias(divergences)