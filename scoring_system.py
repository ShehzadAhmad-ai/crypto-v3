import numpy as np
import pandas as pd
from typing import Dict, Tuple
from config import Config

class SignalScoringSystem:
    def __init__(self):
        self.indicator_weights = {
            'rsi': 0.15,
            'macd': 0.15,
            'ema_structure': 0.20,
            'vwap': 0.10,
            'volume_momentum': 0.15,
            'breakout': 0.10,
            'support_resistance': 0.10,
            'pattern': 0.05
        }

    def compute_indicator_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        if df is None or df.empty:
            return 0.0, {}
        details = {}
        # RSI
        rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
        rsi_score = 0.5
        if 30 <= rsi <= 40:
            rsi_score = 0.9
        elif rsi < 30:
            rsi_score = 0.8
        elif rsi > 70:
            rsi_score = 0.2
        details['rsi'] = {'value': rsi, 'score': rsi_score}
        # MACD
        macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df else 0
        macd_score = 0.5
        if macd_hist > 0:
            macd_score = 0.8
        elif macd_hist < 0:
            macd_score = 0.3
        details['macd'] = {'value': macd_hist, 'score': macd_score}
        # EMA structure
        ema_score = 0.5
        if all(col in df for col in ['ema_8','ema_21','ema_50']):
            price = df['close'].iloc[-1]; e8 = df['ema_8'].iloc[-1]; e21 = df['ema_21'].iloc[-1]; e50 = df['ema_50'].iloc[-1]
            if price > e8 > e21 > e50 or price < e8 < e21 < e50:
                ema_score = 1.0
        details['ema_structure'] = {'score': ema_score}
        # VWAP
        vwap_score = 0.5
        if 'vwap' in df:
            diff = (df['close'].iloc[-1] - df['vwap'].iloc[-1]) / df['vwap'].iloc[-1]
            if abs(diff) > 0.01:
                vwap_score = 0.8
        details['vwap'] = {'score': vwap_score}
        # Volume momentum
        vol_score = 0.5
        if 'volume_ratio' in df:
            vr = df['volume_ratio'].iloc[-1]
            if vr > 1.5:
                vol_score = 0.9
            elif vr > 1.2:
                vol_score = 0.7
        details['volume_momentum'] = {'score': vol_score}
        # Breakout
        breakout_score = 0.5
        if 'high' in df and 'low' in df:
            recent_high = df['high'].iloc[-10:-1].max() if len(df) >= 10 else df['high'].iloc[-1]
            recent_low = df['low'].iloc[-10:-1].min() if len(df) >= 10 else df['low'].iloc[-1]
            if df['high'].iloc[-1] > recent_high*1.002 or df['low'].iloc[-1] < recent_low*0.998:
                breakout_score = 0.9
        details['breakout'] = {'score': breakout_score}
        # Pattern
        pattern_score = 0.5
        if 'atr' in df:
            atr = df['atr'].iloc[-1]; atr_sma = df['atr'].rolling(20).mean().iloc[-1] if len(df)>=20 else atr
            if atr > atr_sma*1.3:
                pattern_score = 0.8
            elif atr < atr_sma*0.8:
                pattern_score = 0.7
        details['pattern'] = {'score': pattern_score}
        # Weighted total
        total = (rsi_score*self.indicator_weights['rsi'] + macd_score*self.indicator_weights['macd'] +
                 ema_score*self.indicator_weights['ema_structure'] + vwap_score*self.indicator_weights['vwap'] +
                 vol_score*self.indicator_weights['volume_momentum'] + breakout_score*self.indicator_weights['breakout'] +
                 pattern_score*self.indicator_weights['pattern'])
        confidence = min(0.99, total)
        return confidence, details