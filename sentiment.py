# sentiment.py - ENHANCED with REAL data from Binance
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

class Sentiment:
    """
    Enhanced Sentiment Analysis
    - Fear & Greed Index (public API - no key needed)
    - Funding rate sentiment (from Binance Futures)
    - Technical sentiment (from price action)
    - Volume sentiment (from volume analysis)
    """
    
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Store latest data from Binance
        self.funding_rates: Dict[str, float] = {}
        self.technical_data: Dict[str, Dict] = {}
        self.volume_data: Dict[str, float] = {}
        
        logging.info("Sentiment analyzer initialized")
    
    # ==================== PUBLIC METHODS ====================
    
    def get_combined_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get combined sentiment from all available sources
        Uses cached data from Binance if available
        """
        # Check cache
        if symbol in self.sentiment_cache:
            cached = self.sentiment_cache[symbol]
            if datetime.now() - cached['timestamp'] < self.cache_duration:
                return cached['data']
        
        sentiments = []
        sources = []
        components = {}
        
        # 1. FUNDING SENTIMENT (from Binance Futures)
        funding_score, funding_details = self._get_funding_sentiment(symbol)
        if funding_score != 0.5 or funding_details.get('has_data', False):
            sentiments.append(funding_score)
            sources.append('funding')
            components['funding'] = funding_details
        
        # 2. TECHNICAL SENTIMENT (from price action)
        tech_score, tech_details = self._get_technical_sentiment(symbol)
        if tech_score != 0.5 or tech_details.get('has_data', False):
            sentiments.append(tech_score)
            sources.append('technical')
            components['technical'] = tech_details
        
        # 3. VOLUME SENTIMENT (from volume analysis)
        volume_score, volume_details = self._get_volume_sentiment(symbol)
        if volume_score != 0.5 or volume_details.get('has_data', False):
            sentiments.append(volume_score)
            sources.append('volume')
            components['volume'] = volume_details
        
        # 4. FEAR & GREED INDEX (market-wide)
        fng = self.get_fear_greed_index()
        if fng['value'] != 50:  # Not default
            # Convert F&G (0-100) to sentiment score (0-1)
            fng_score = fng['value'] / 100
            sentiments.append(fng_score)
            sources.append('fear_greed')
            components['fear_greed'] = fng
        
        # Calculate composite score
        if sentiments:
            composite = sum(sentiments) / len(sentiments)
            
            # Determine signal
            if composite > 0.6:
                signal = 'BULLISH'
            elif composite < 0.4:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'
        else:
            composite = 0.5
            signal = 'NEUTRAL'
            sources = ['default']
        
        result = {
            'composite_score': round(composite, 3),
            'signal': signal,
            'sources': sources,
            'components': components,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        self.sentiment_cache[symbol] = {
            'data': result,
            'timestamp': datetime.now()
        }
        
        return result
    
    # ==================== DATA UPDATE METHODS ====================
    # Call these from pipeline with real Binance data
    
    def update_funding_data(self, symbol: str, funding_rate: float):
        """Update funding rate data from Binance Futures"""
        self.funding_rates[symbol] = funding_rate
        # Clear cache for this symbol
        if symbol in self.sentiment_cache:
            del self.sentiment_cache[symbol]
    
    def update_technical_data(self, symbol: str, rsi: float, price: float, 
                            ema_fast: float, ema_slow: float, trend: str):
        """Update technical indicator data"""
        self.technical_data[symbol] = {
            'rsi': rsi,
            'price': price,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'trend': trend,
            'timestamp': datetime.now()
        }
        # Clear cache for this symbol
        if symbol in self.sentiment_cache:
            del self.sentiment_cache[symbol]
    
    def update_volume_data(self, symbol: str, volume_ratio: float, 
                          volume_trend: str,divergence: Optional[str] = None):
        """Update volume analysis data"""
        self.volume_data[symbol] = {
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'divergence': divergence,
            'timestamp': datetime.now()
        }
        # Clear cache for this symbol
        if symbol in self.sentiment_cache:
            del self.sentiment_cache[symbol]
    
    # ==================== PRIVATE SENTIMENT METHODS ====================
    
    def _get_funding_sentiment(self, symbol: str) -> tuple[float, Dict]:
        """
        Get sentiment from funding rates
        Uses real data from Binance Futures
        """
        details = {'has_data': False, 'funding_rate': 0, 'bias': 'NEUTRAL'}
        
        if symbol not in self.funding_rates:
            return 0.5, details
        
        funding_rate = self.funding_rates[symbol]
        details['has_data'] = True
        details['funding_rate'] = round(funding_rate * 100, 4)  # Convert to percentage
        
        # Determine sentiment based on funding rate
        if funding_rate > 0.0001:  # 0.01% positive
            # Positive funding = longs paying = bullish sentiment
            score = 0.6 + min(0.3, funding_rate * 100)  # Max 0.9
            details['bias'] = 'BULLISH'
            details['reason'] = f"Positive funding ({funding_rate*100:.4f}%) indicates bullish sentiment"
        elif funding_rate < -0.0001:  # 0.01% negative
            # Negative funding = shorts paying = bearish sentiment
            score = 0.4 - min(0.3, abs(funding_rate) * 100)  # Min 0.1
            details['bias'] = 'BEARISH'
            details['reason'] = f"Negative funding ({abs(funding_rate)*100:.4f}%) indicates bearish sentiment"
        else:
            score = 0.5
            details['bias'] = 'NEUTRAL'
            details['reason'] = "Funding rate neutral"
        
        return min(0.95, max(0.05, score)), details
    
    def _get_technical_sentiment(self, symbol: str) -> tuple[float, Dict]:
        """
        Get sentiment from technical indicators
        Uses real data from technical_analyzer
        """
        details = {'has_data': False, 'rsi': 50, 'trend': 'NEUTRAL'}
        
        if symbol not in self.technical_data:
            return 0.5, details
        
        data = self.technical_data[symbol]
        details['has_data'] = True
        details['rsi'] = round(data['rsi'], 1)
        details['trend'] = data['trend']
        details['price'] = round(data['price'], 4)
        
        score = 0.5
        reasons = []
        
        # RSI contribution
        rsi = data['rsi']
        if rsi > 70:
            score += 0.15
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            score += 0.1
            reasons.append(f"RSI bullish ({rsi:.1f})")
        elif rsi < 30:
            score -= 0.15
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            score -= 0.1
            reasons.append(f"RSI bearish ({rsi:.1f})")
        
        # Trend contribution
        if data['trend'] == 'BULL':
            score += 0.1
            reasons.append("Price in bullish trend")
        elif data['trend'] == 'BEAR':
            score -= 0.1
            reasons.append("Price in bearish trend")
        
        # EMA alignment
        if 'ema_fast' in data and 'ema_slow' in data:
            if data['ema_fast'] > data['ema_slow']:
                score += 0.05
                reasons.append("EMA alignment bullish")
            elif data['ema_fast'] < data['ema_slow']:
                score -= 0.05
                reasons.append("EMA alignment bearish")
        
        details['reasons'] = reasons[:2]  # Top 2 reasons
        
        return min(0.95, max(0.05, score)), details
    
    def _get_volume_sentiment(self, symbol: str) -> tuple[float, Dict]:
        """
        Get sentiment from volume analysis
        Uses real data from volume_analyzer
        """
        details = {'has_data': False, 'volume_ratio': 1.0, 'trend': 'NEUTRAL'}
        
        if symbol not in self.volume_data:
            return 0.5, details
        
        data = self.volume_data[symbol]
        details['has_data'] = True
        details['volume_ratio'] = round(data['volume_ratio'], 2)
        details['trend'] = data['volume_trend']
        
        score = 0.5
        reasons = []
        
        # Volume ratio contribution
        vol_ratio = data['volume_ratio']
        if vol_ratio > 1.5:
            score += 0.15
            reasons.append(f"Volume spike ({vol_ratio:.1f}x)")
        elif vol_ratio > 1.2:
            score += 0.1
            reasons.append(f"Above average volume ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            score -= 0.1
            reasons.append(f"Very low volume ({vol_ratio:.1f}x)")
        
        # Volume trend contribution
        if data['volume_trend'] == 'INCREASING':
            score += 0.1
            reasons.append("Volume increasing")
        elif data['volume_trend'] == 'DECREASING':
            score -= 0.05
            reasons.append("Volume decreasing")
        
        # Divergence detection
        if data.get('divergence'):
            if data['divergence'] == 'BULLISH':
                score += 0.15
                reasons.append("Bullish volume-price divergence")
            elif data['divergence'] == 'BEARISH':
                score -= 0.15
                reasons.append("Bearish volume-price divergence")
        
        details['reasons'] = reasons[:2]
        
        return min(0.95, max(0.05, score)), details
    
    # ==================== FEAR & GREED INDEX ====================
    
    def get_fear_greed_index(self) -> Dict:
        """
        Get Crypto Fear & Greed Index
        Public API - no key required
        """
        try:
            response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and len(data['data']) > 0:
                    return {
                        'value': int(data['data'][0]['value']),
                        'classification': data['data'][0]['value_classification'],
                        'timestamp': data['data'][0]['timestamp']
                    }
        except Exception as e:
            logging.warning(f"Failed to fetch fear & greed index: {e}")
        
        return {
            'value': 50,
            'classification': 'Neutral',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_market_overview(self) -> Dict:
        """
        Get overall market sentiment
        """
        fng = self.get_fear_greed_index()
        
        # Determine overall sentiment
        if fng['value'] > 60:
            overall = 'BULLISH'
        elif fng['value'] < 40:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        return {
            'fear_greed': fng,
            'overall': overall,
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear sentiment cache"""
        self.sentiment_cache.clear()
        logging.info("Sentiment cache cleared")

    # ==================== SCORING METHODS FOR LIGHT CONFIRM ====================
    
    def get_sentiment_score(self, symbol: str) -> float:
        """
        Get overall sentiment score (0-1)
        Higher score = bullish sentiment
        Lower score = bearish sentiment
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Score between 0 and 1
        """
        try:
            sentiment = self.get_combined_sentiment(symbol)
            return sentiment.get('composite_score', 0.5)
            
        except Exception as e:
            logging.error(f"Error in get_sentiment_score: {e}")
            return 0.5
    
    def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive sentiment summary for light confirm pipeline
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Dictionary with sentiment summary
        """
        try:
            sentiment = self.get_combined_sentiment(symbol)
            
            # Get Fear & Greed separately
            fng = self.get_fear_greed_index()
            
            # Get component scores
            components = sentiment.get('components', {})
            
            funding = components.get('funding', {})
            technical = components.get('technical', {})
            volume = components.get('volume', {})
            
            return {
                'score': sentiment.get('composite_score', 0.5),
                'signal': sentiment.get('signal', 'NEUTRAL'),
                'sources': sentiment.get('sources', []),
                'fear_greed': {
                    'value': fng.get('value', 50),
                    'classification': fng.get('classification', 'Neutral'),
                    'score': fng.get('value', 50) / 100
                },
                'funding': {
                    'score': funding.get('score', 0.5),
                    'bias': funding.get('bias', 'NEUTRAL'),
                    'rate': funding.get('funding_rate', 0),
                    'reason': funding.get('reason', '')
                },
                'technical': {
                    'score': technical.get('score', 0.5),
                    'rsi': technical.get('rsi', 50),
                    'trend': technical.get('trend', 'NEUTRAL'),
                    'reasons': technical.get('reasons', [])
                },
                'volume': {
                    'score': volume.get('score', 0.5),
                    'ratio': volume.get('volume_ratio', 1.0),
                    'trend': volume.get('trend', 'NEUTRAL'),
                    'reasons': volume.get('reasons', [])
                },
                'reasons': [
                    f"Fear & Greed: {fng.get('value', 50)} - {fng.get('classification', 'Neutral')}",
                    f"Funding: {funding.get('bias', 'NEUTRAL')}",
                    f"Technical: {technical.get('trend', 'NEUTRAL')}",
                    f"Volume: {volume.get('trend', 'NEUTRAL')}"
                ]
            }
            
        except Exception as e:
            logging.error(f"Error in get_sentiment_summary: {e}")
            return {
                'score': 0.5,
                'signal': 'NEUTRAL',
                'reasons': ['Error in sentiment analysis']
            }
    
    def is_bullish(self, symbol: str, threshold: float = 0.6) -> bool:
        """
        Quick check if sentiment is bullish
        
        Args:
            symbol: Trading pair symbol
            threshold: Minimum score for bullish (default 0.6)
        
        Returns:
            True if bullish
        """
        try:
            score = self.get_sentiment_score(symbol)
            return score >= threshold
        except Exception:
            return False
    
    def is_bearish(self, symbol: str, threshold: float = 0.4) -> bool:
        """
        Quick check if sentiment is bearish
        
        Args:
            symbol: Trading pair symbol
            threshold: Maximum score for bearish (default 0.4)
        
        Returns:
            True if bearish
        """
        try:
            score = self.get_sentiment_score(symbol)
            return score <= threshold
        except Exception:
            return False
    
    def is_neutral(self, symbol: str, low: float = 0.4, high: float = 0.6) -> bool:
        """
        Quick check if sentiment is neutral
        
        Args:
            symbol: Trading pair symbol
            low: Lower bound for neutral (default 0.4)
            high: Upper bound for neutral (default 0.6)
        
        Returns:
            True if neutral
        """
        try:
            score = self.get_sentiment_score(symbol)
            return low < score < high
        except Exception:
            return True
    
    def get_fear_greed_score(self) -> float:
        """
        Get Fear & Greed score (0-1)
        0 = extreme fear, 1 = extreme greed
        """
        try:
            fng = self.get_fear_greed_index()
            return fng.get('value', 50) / 100
        except Exception:
            return 0.5
    
    def get_market_sentiment(self) -> str:
        """
        Get overall market sentiment (BULLISH/BEARISH/NEUTRAL)
        """
        try:
            fng = self.get_fear_greed_index()
            value = fng.get('value', 50)
            if value > 60:
                return 'BULLISH'
            elif value < 40:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'