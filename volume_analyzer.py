# volume_analyzer.py - Advanced Volume Analysis
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

class VolumeAnalyzer:
    """
    Advanced volume analysis for confirmation signals
    """
    
    def __init__(self):
        self.volume_history = []
    
    def analyze_volume_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume trend and patterns
        """
        try:
            result = {
                'volume_trend': 'NEUTRAL',
                'volume_score': 0.5,
                'volume_ratio': 1.0,
                'reasons': []
            }
            
            if df is None or df.empty or len(df) < 20:
                return result
            
            # Get recent volume
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = float(df['volume'].iloc[-20:].mean())
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            result['volume_ratio'] = round(volume_ratio, 2)
            
            # Volume trend (last 5 vs previous 5)
            recent_volume = df['volume'].iloc[-5:].mean()
            previous_volume = df['volume'].iloc[-10:-5].mean()
            volume_trend_ratio = recent_volume / previous_volume if previous_volume > 0 else 1.0
            
            # Determine trend
            if volume_trend_ratio > 1.2:
                result['volume_trend'] = 'INCREASING'
                result['volume_score'] += 0.1
                result['reasons'].append(f"Volume increasing: {volume_trend_ratio:.2f}x")
            elif volume_trend_ratio < 0.8:
                result['volume_trend'] = 'DECREASING'
                result['volume_score'] -= 0.1
                result['reasons'].append(f"Volume decreasing: {volume_trend_ratio:.2f}x")
            else:
                result['volume_trend'] = 'STABLE'
            
            # Volume spike detection
            if volume_ratio > 2.0:
                result['volume_score'] += 0.15
                result['reasons'].append(f"Volume spike: {volume_ratio:.2f}x average")
            elif volume_ratio > 1.5:
                result['volume_score'] += 0.1
                result['reasons'].append(f"Above average volume: {volume_ratio:.2f}x")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyze_volume_trend: {e}")
            return {'volume_trend': 'NEUTRAL', 'volume_score': 0.5, 'volume_ratio': 1.0, 'reasons': []}
    
    def analyze_volume_price_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect divergence between price and volume
        """
        try:
            result = {
                'divergence': None,
                'divergence_score': 0.5,
                'reasons': []
            }
            
            if df is None or df.empty or len(df) < 20:
                return result
            
            # Price trend (last 10 periods)
            price_start = float(df['close'].iloc[-10])
            price_end = float(df['close'].iloc[-1])
            price_change = (price_end - price_start) / price_start
            
            # Volume trend
            volume_start = float(df['volume'].iloc[-10:].mean())
            volume_end = float(df['volume'].iloc[-5:].mean())
            volume_change = (volume_end - volume_start) / volume_start if volume_start > 0 else 0
            
            # Detect divergence
            if price_change > 0.02 and volume_change < -0.1:
                # Price up, volume down - bearish divergence
                result['divergence'] = 'BEARISH'
                result['divergence_score'] = 0.3
                result['reasons'].append(f"Bearish divergence: Price +{price_change:.1%}, Volume {volume_change:.1%}")
            elif price_change < -0.02 and volume_change > 0.1:
                # Price down, volume up - bullish divergence
                result['divergence'] = 'BULLISH'
                result['divergence_score'] = 0.7
                result['reasons'].append(f"Bullish divergence: Price {price_change:.1%}, Volume +{volume_change:.1%}")
            else:
                result['divergence'] = 'NONE'
                result['divergence_score'] = 0.5
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyze_volume_price_divergence: {e}")
            return {'divergence': None, 'divergence_score': 0.5, 'reasons': []}
    
    def analyze_volume_by_price(self, df: pd.DataFrame, bins: int = 10) -> Dict[str, Any]:
        """
        Analyze volume distribution by price levels
        """
        try:
            result = {
                'high_volume_nodes': [],
                'low_volume_nodes': [],
                'volume_profile_score': 0.5,
                'reasons': []
            }
            
            if df is None or df.empty or len(df) < 50:
                return result
            
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, bins)
            
            # Calculate volume in each bin
            volume_by_price = {}
            for i in range(len(price_bins) - 1):
                mask = (df['close'] >= price_bins[i]) & (df['close'] < price_bins[i+1])
                if mask.any():
                    volume = df.loc[mask, 'volume'].sum()
                    volume_by_price[(price_bins[i] + price_bins[i+1]) / 2] = volume
            
            if not volume_by_price:
                return result
            
            # Find high volume nodes (POC)
            sorted_by_volume = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
            high_volume_prices = [p for p, v in sorted_by_volume[:3]]
            
            # Find low volume nodes (gaps)
            low_volume_prices = [p for p, v in sorted_by_volume[-3:]]
            
            result['high_volume_nodes'] = high_volume_prices
            result['low_volume_nodes'] = low_volume_prices
            
            # Current price position
            current_price = float(df['close'].iloc[-1])
            
            # Check if price near high volume node
            for price in high_volume_prices:
                if abs(current_price - price) / current_price < 0.01:
                    result['volume_profile_score'] += 0.15
                    result['reasons'].append(f"Price near high volume node: {price:.4f}")
                    break
            
            # Check if price in low volume area (potential for fast move)
            for price in low_volume_prices:
                if abs(current_price - price) / current_price < 0.01:
                    result['volume_profile_score'] -= 0.1
                    result['reasons'].append(f"Price in low volume area: {price:.4f}")
                    break
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyze_volume_by_price: {e}")
            return result
    
    def generate_volume_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate combined volume signal
        """
        try:
            # Get all volume analyses
            trend = self.analyze_volume_trend(df)
            divergence = self.analyze_volume_price_divergence(df)
            profile = self.analyze_volume_by_price(df)
            
            # Combine scores
            score = (
                trend.get('volume_score', 0.5) * 0.4 +
                divergence.get('divergence_score', 0.5) * 0.3 +
                profile.get('volume_profile_score', 0.5) * 0.3
            )
            
            # Collect all reasons
            reasons = []
            reasons.extend(trend.get('reasons', []))
            reasons.extend(divergence.get('reasons', []))
            reasons.extend(profile.get('reasons', []))
            
            return {
                'score': min(0.99, max(0.01, score)),
                'volume_trend': trend.get('volume_trend', 'NEUTRAL'),
                'divergence': divergence.get('divergence'),
                'high_volume_nodes': profile.get('high_volume_nodes', []),
                'low_volume_nodes': profile.get('low_volume_nodes', []),
                'volume_ratio': trend.get('volume_ratio', 1.0),
                'reasons': reasons[:5]  # Top 5 reasons
            }
            
        except Exception as e:
            logging.error(f"Error in generate_volume_signal: {e}")
            return {'score': 0.5, 'reasons': []}
        


    def get_volume_score(self, df: pd.DataFrame) -> float:
        """
        Get aggregate volume score (0-1) for scoring layer
        Higher score = bullish volume, Lower score = bearish volume
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Score between 0 and 1
        """
        signal = self.generate_volume_signal(df)
        return signal.get('score', 0.5)

    def get_volume_bias(self, df: pd.DataFrame) -> float:
        """
        Get net volume bias (-1 to 1)
        Positive = bullish volume, Negative = bearish volume
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Bias score between -1 and 1
        """
        signal = self.generate_volume_signal(df)
        score = signal.get('score', 0.5)
        
        # Convert 0-1 score to -1 to 1 bias
        bias = (score - 0.5) * 2
        
        # Adjust based on divergence
        divergence = signal.get('divergence')
        if divergence == 'BULLISH':
            bias += 0.2
        elif divergence == 'BEARISH':
            bias -= 0.2
        
        return max(-1.0, min(1.0, bias))

    def get_volume_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive volume summary for scoring layer
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with volume summary
        """
        trend = self.analyze_volume_trend(df)
        divergence = self.analyze_volume_price_divergence(df)
        profile = self.analyze_volume_by_price(df)
        signal = self.generate_volume_signal(df)
        
        return {
            'volume_ratio': trend.get('volume_ratio', 1.0),
            'volume_trend': trend.get('volume_trend', 'NEUTRAL'),
            'volume_score': trend.get('volume_score', 0.5),
            'divergence': divergence.get('divergence'),
            'divergence_score': divergence.get('divergence_score', 0.5),
            'high_volume_nodes': profile.get('high_volume_nodes', []),
            'low_volume_nodes': profile.get('low_volume_nodes', []),
            'volume_profile_score': profile.get('volume_profile_score', 0.5),
            'combined_score': signal.get('score', 0.5),
            'reasons': signal.get('reasons', [])
        }

    def has_volume_spike(self, df: pd.DataFrame, threshold: float = 1.5) -> bool:
        """
        Check if there is a volume spike
        
        Args:
            df: DataFrame with OHLCV data
            threshold: Spike threshold (1.5 = 50% above average)
        
        Returns:
            True if volume spike detected
        """
        if df is None or df.empty or len(df) < 20:
            return False
        
        current_volume = float(df['volume'].iloc[-1])
        avg_volume = float(df['volume'].iloc[-20:].mean())
        
        return current_volume > avg_volume * threshold

    def is_volume_increasing(self, df: pd.DataFrame) -> bool:
        """
        Check if volume is increasing
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            True if volume trend is increasing
        """
        trend = self.analyze_volume_trend(df)
        return trend.get('volume_trend') == 'INCREASING'

    def has_volume_divergence(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Check if there is volume-price divergence
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (has_divergence, divergence_type)
        """
        divergence = self.analyze_volume_price_divergence(df)
        div_type = divergence.get('divergence')
        return div_type is not None and div_type != 'NONE', div_type