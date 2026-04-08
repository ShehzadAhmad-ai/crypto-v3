# correlation_analyzer.py - Enhanced Correlation Analysis with BTC Dominance
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats

class CorrelationAnalyzer:
    """
    Advanced correlation analysis including BTC dominance and sector rotation
    """
    
    def __init__(self):
        self.correlation_history = []
        
        # Define sectors (you can expand these)
        self.sectors = {
            'L1': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'ADA/USDT'],
            'L2': ['MATIC/USDT', 'ARB/USDT', 'OP/USDT'],
            'DeFi': ['UNI/USDT', 'AAVE/USDT', 'MKR/USDT', 'COMP/USDT'],
            'Meme': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT'],
            'AI': ['FET/USDT', 'AGIX/USDT', 'OCEAN/USDT'],
            'Gaming': ['AXS/USDT', 'SAND/USDT', 'MANA/USDT']
        }
    
    def analyze_btc_dominance(self, btc_price: pd.Series, eth_price: Optional[pd.Series] = None, total_market_cap: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze BTC dominance and its impact on altcoins
        
        Args:
            btc_price: BTC price series
            total_market_cap: Total crypto market cap (if available)
        
        Returns:
            Dict with dominance analysis
        """
        try:
            result = {
                'btc_dominance_trend': 'NEUTRAL',
                'altcoin_season': False,
                'dominance_score': 0.5,
                'reasons': []
            }
            
            if len(btc_price) < 20:
                return result
            
            # If total market cap is available, calculate REAL dominance
            if total_market_cap is not None and len(total_market_cap) >= 20:
                # Estimate BTC market cap (price * supply) - using price as proxy
                btc_mcap = btc_price * 19_000_000  # Approximate BTC supply
                dominance = (btc_mcap / total_market_cap) * 100
                
                # Calculate dominance trend
                dom_20_ago = dominance.iloc[-20] if len(dominance) >= 20 else dominance.iloc[0]
                dom_now = dominance.iloc[-1]
                dom_change = ((dom_now - dom_20_ago) / dom_20_ago) * 100
                
                btc_trend = dom_change
            else:
                # Fallback to using ETH/BTC ratio as proxy for dominance
                # When ETH outperforms BTC, dominance typically decreases
                # This requires ETH price data - you'd need to pass it
                if eth_price is not None and len(eth_price) >= 20:
                    # ETH/BTC ratio is better proxy for dominance
                    eth_btc_ratio = eth_price / btc_price
                    ratio_change = (eth_btc_ratio.iloc[-1] / eth_btc_ratio.iloc[-20] - 1) * 100
                    # When ETH outperforms BTC (ratio up), dominance down
                    btc_trend = -ratio_change
                    result['reasons'].append("Using ETH/BTC ratio as dominance proxy")
                else:
                    btc_trend = (btc_price.iloc[-1] / btc_price.iloc[-20] - 1) * 100
                result['reasons'].append("Using price proxy for dominance (market cap unavailable)")
            
            # Determine if it's altcoin season (BTC dominance decreasing)
            if btc_trend < -3:  # 3% decrease in dominance
                result['btc_dominance_trend'] = 'DECREASING'
                result['altcoin_season'] = True
                result['dominance_score'] = 0.8
                result['reasons'].append(f"BTC dominance decreasing ({btc_trend:.1f}%) - Altcoin season likely")
            elif btc_trend > 3:  # 3% increase in dominance
                result['btc_dominance_trend'] = 'INCREASING'
                result['altcoin_season'] = False
                result['dominance_score'] = 0.3
                result['reasons'].append(f"BTC dominance increasing ({btc_trend:.1f}%) - BTC strength")
            else:
                result['btc_dominance_trend'] = 'NEUTRAL'
                result['dominance_score'] = 0.5
                result['reasons'].append("BTC dominance stable")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyze_btc_dominance: {e}")
            return {'btc_dominance_trend': 'NEUTRAL', 'altcoin_season': False, 'dominance_score': 0.5, 'reasons': []}
    def analyze_sector_rotation(self, price_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Analyze which sectors are currently strong/weak
        
        Args:
            price_data: Dict of symbol -> price series
        
        Returns:
            Dict with sector performance analysis
        """
        try:
            result = {
                'strongest_sector': 'UNKNOWN',
                'weakest_sector': 'UNKNOWN',
                'sector_performance': {},
                'rotation_score': 0.5,
                'reasons': []
            }
            
            if len(price_data) < 3:
                return result
            
            # Calculate 24h performance for each symbol
            symbol_perf = {}
            for symbol, prices in price_data.items():
                if len(prices) >= 2:
                    perf = (prices.iloc[-1] / prices.iloc[-2] - 1) * 100
                    symbol_perf[symbol] = perf
            
            # Calculate sector averages
            sector_perf = {}
            for sector, symbols in self.sectors.items():
                performances = [symbol_perf[s] for s in symbols if s in symbol_perf]
                if performances:
                    sector_perf[sector] = np.mean(performances)
            
            if not sector_perf:
                return result
            
            # Find strongest and weakest sectors
            strongest = max(sector_perf.items(), key=lambda x: x[1])
            weakest = min(sector_perf.items(), key=lambda x: x[1])
            
            result['strongest_sector'] = strongest[0]
            result['weakest_sector'] = weakest[0]
            result['sector_performance'] = sector_perf
            result['rotation_score'] = 0.5 + (strongest[1] - weakest[1]) / 100
            
            # Add reasons
            result['reasons'].append(f"Strongest sector: {strongest[0]} ({strongest[1]:.1f}%)")
            result['reasons'].append(f"Weakest sector: {weakest[0]} ({weakest[1]:.1f}%)")
            
            if strongest[1] > 5 and weakest[1] < -5:
                result['reasons'].append("Strong sector rotation detected")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyze_sector_rotation: {e}")
            return result
    
    def analyze_correlation_with_btc(self, symbol: str, btc_prices: pd.Series, 
                                      symbol_prices: pd.Series, window: int = 50) -> Dict[str, Any]:
        """
        Analyze correlation between a symbol and BTC
        
        Returns:
            Dict with correlation analysis
        """
        try:
            result = {
                'correlation': 0.0,
                'correlation_strength': 'NEUTRAL',
                'beta': 1.0,
                'reasons': []
            }
            
            if len(btc_prices) < window or len(symbol_prices) < window:
                return result
            
            # Align series
            common_idx = btc_prices.index.intersection(symbol_prices.index)
            if len(common_idx) < window:
                return result
            
            btc_aligned = btc_prices.loc[common_idx].iloc[-window:]
            sym_aligned = symbol_prices.loc[common_idx].iloc[-window:]
            
            # Calculate returns
            btc_returns = btc_aligned.pct_change().dropna()
            sym_returns = sym_aligned.pct_change().dropna()
            
            if len(btc_returns) < 10 or len(sym_returns) < 10:
                return result
            
            # Calculate correlation
            correlation = btc_returns.corr(sym_returns)
            
            # Calculate beta (simplified)
            covariance = btc_returns.cov(sym_returns)
            variance = btc_returns.var()
            beta = covariance / variance if variance > 0 else 1.0
            
            result['correlation'] = round(correlation, 3)
            result['beta'] = round(beta, 3)
            
            # Determine strength
            if abs(correlation) > 0.7:
                result['correlation_strength'] = 'STRONG'
            elif abs(correlation) > 0.4:
                result['correlation_strength'] = 'MODERATE'
            else:
                result['correlation_strength'] = 'WEAK'
            
            result['reasons'].append(f"BTC correlation: {correlation:.2f} ({result['correlation_strength']})")
            result['reasons'].append(f"Beta to BTC: {beta:.2f}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyze_correlation_with_btc: {e}")
            return result
    
    def generate_correlation_signal(self, symbol: str, btc_correlation: Dict[str, Any],
                                    btc_dominance: Dict[str, Any], 
                                    sector_rotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final correlation-based signal
        
        Returns:
            Dict with score and direction
        """
        try:
            score = 0.5
            direction = 'NEUTRAL'
            reasons = []
            
            # Factor 1: BTC correlation alignment
            if btc_correlation.get('correlation', 0) > 0.7:
                # High correlation with BTC
                if btc_dominance.get('btc_dominance_trend') == 'INCREASING':
                    score += 0.1
                    reasons.append("High BTC correlation with increasing dominance")
                elif btc_dominance.get('btc_dominance_trend') == 'DECREASING':
                    score -= 0.1
                    reasons.append("High BTC correlation but dominance decreasing")
            
            # Factor 2: Altcoin season
            if btc_dominance.get('altcoin_season', False):
                # Altcoin season - good for longs on alts
                score += 0.15
                reasons.append("Altcoin season detected")
            
            # Factor 3: Sector strength
            strongest_sector = sector_rotation.get('strongest_sector', 'UNKNOWN')
            if strongest_sector != 'UNKNOWN':
                # Check if our symbol belongs to strong sector
                for sector, symbols in self.sectors.items():
                    if symbol in symbols:
                        if sector == strongest_sector:
                            score += 0.2
                            reasons.append(f"Symbol in strongest sector: {sector}")
                        elif sector == sector_rotation.get('weakest_sector'):
                            score -= 0.15
                            reasons.append(f"Symbol in weakest sector: {sector}")
                        break
            
            return {
                'score': min(0.99, max(0.01, score)),
                'direction': direction,
                'reasons': reasons
            }
            
        except Exception as e:
            logging.error(f"Error in generate_correlation_signal: {e}")
            return {'score': 0.5, 'direction': 'NEUTRAL', 'reasons': []}
        

    # ==================== SCORING METHODS FOR LIGHT CONFIRM ====================
    
    def get_correlation_score(self, symbol: str, btc_prices: pd.Series, 
                               symbol_prices: pd.Series,
                               btc_dominance: Dict = None,
                               sector_rotation: Dict = None) -> float:
        """
        Get overall correlation score (0-1)
        Higher score = bullish correlation setup
        Lower score = bearish correlation setup
        
        Args:
            symbol: Trading pair symbol
            btc_prices: BTC price series
            symbol_prices: Symbol price series
            btc_dominance: BTC dominance analysis result (optional)
            sector_rotation: Sector rotation analysis result (optional)
        
        Returns:
            Score between 0 and 1
        """
        try:
            # Get correlation analysis
            correlation = self.analyze_correlation_with_btc(symbol, btc_prices, symbol_prices)
            
            # Get dominance analysis if not provided
            if btc_dominance is None:
                btc_dominance = self.analyze_btc_dominance(btc_prices)
            
            # Get sector rotation if not provided
            if sector_rotation is None:
                # Need price data for multiple symbols
                price_data = {symbol: symbol_prices}
                sector_rotation = self.analyze_sector_rotation(price_data)
            
            # Generate signal
            signal = self.generate_correlation_signal(symbol, correlation, btc_dominance, sector_rotation)
            
            return signal.get('score', 0.5)
            
        except Exception as e:
            logging.error(f"Error in get_correlation_score: {e}")
            return 0.5
    
    def get_correlation_summary(self, symbol: str, btc_prices: pd.Series,
                                 symbol_prices: pd.Series,
                                 btc_dominance: Dict = None,
                                 sector_rotation: Dict = None) -> Dict[str, Any]:
        """
        Get comprehensive correlation summary for light confirm pipeline
        
        Args:
            symbol: Trading pair symbol
            btc_prices: BTC price series
            symbol_prices: Symbol price series
            btc_dominance: BTC dominance analysis result (optional)
            sector_rotation: Sector rotation analysis result (optional)
        
        Returns:
            Dictionary with correlation summary
        """
        try:
            # Get correlation analysis
            correlation = self.analyze_correlation_with_btc(symbol, btc_prices, symbol_prices)
            
            # Get dominance analysis if not provided
            if btc_dominance is None:
                btc_dominance = self.analyze_btc_dominance(btc_prices)
            
            # Get sector rotation if not provided
            if sector_rotation is None:
                # Need price data for multiple symbols
                price_data = {symbol: symbol_prices}
                sector_rotation = self.analyze_sector_rotation(price_data)
            
            # Generate signal
            signal = self.generate_correlation_signal(symbol, correlation, btc_dominance, sector_rotation)
            
            # Determine symbol's sector
            symbol_sector = 'UNKNOWN'
            for sector, symbols in self.sectors.items():
                if symbol in symbols:
                    symbol_sector = sector
                    break
            
            return {
                'score': signal.get('score', 0.5),
                'correlation': correlation.get('correlation', 0),
                'correlation_strength': correlation.get('correlation_strength', 'NEUTRAL'),
                'beta': correlation.get('beta', 1.0),
                'btc_dominance': {
                    'trend': btc_dominance.get('btc_dominance_trend', 'NEUTRAL'),
                    'altcoin_season': btc_dominance.get('altcoin_season', False),
                    'dominance_score': btc_dominance.get('dominance_score', 0.5)
                },
                'sector': {
                    'name': symbol_sector,
                    'strongest': sector_rotation.get('strongest_sector', 'UNKNOWN'),
                    'weakest': sector_rotation.get('weakest_sector', 'UNKNOWN'),
                    'is_strongest': symbol_sector == sector_rotation.get('strongest_sector'),
                    'is_weakest': symbol_sector == sector_rotation.get('weakest_sector')
                },
                'reasons': signal.get('reasons', [])
            }
            
        except Exception as e:
            logging.error(f"Error in get_correlation_summary: {e}")
            return {
                'score': 0.5,
                'correlation': 0,
                'reasons': ['Error in correlation analysis']
            }
    
    def is_high_correlation(self, symbol: str, btc_prices: pd.Series,
                            symbol_prices: pd.Series, threshold: float = 0.7) -> bool:
        """
        Quick check if symbol has high correlation with BTC
        
        Args:
            symbol: Trading pair symbol
            btc_prices: BTC price series
            symbol_prices: Symbol price series
            threshold: Correlation threshold (default 0.7)
        
        Returns:
            True if high correlation
        """
        try:
            correlation = self.analyze_correlation_with_btc(symbol, btc_prices, symbol_prices)
            return abs(correlation.get('correlation', 0)) >= threshold
        except Exception:
            return False
    
    def is_altcoin_season(self, btc_prices: pd.Series, 
                          eth_price: Optional[pd.Series] = None,
                          total_market_cap: Optional[pd.Series] = None) -> bool:
        """
        Quick check if it's altcoin season
        
        Args:
            btc_prices: BTC price series
            eth_price: ETH price series (for proxy)
            total_market_cap: Total market cap (for accurate calculation)
        
        Returns:
            True if altcoin season
        """
        try:
            dominance = self.analyze_btc_dominance(btc_prices, eth_price, total_market_cap)
            return dominance.get('altcoin_season', False)
        except Exception:
            return False
    
    def get_symbol_sector(self, symbol: str) -> str:
        """
        Get the sector of a symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Sector name or 'UNKNOWN'
        """
        for sector, symbols in self.sectors.items():
            if symbol in symbols:
                return sector
        return 'UNKNOWN'
    
    def get_sector_strength(self, symbol: str, sector_rotation: Dict) -> float:
        """
        Get sector strength score (0-1) for a symbol
        
        Args:
            symbol: Trading pair symbol
            sector_rotation: Sector rotation analysis result
        
        Returns:
            Score between 0 and 1
        """
        try:
            symbol_sector = self.get_symbol_sector(symbol)
            sector_perf = sector_rotation.get('sector_performance', {})
            
            if symbol_sector in sector_perf:
                # Convert performance percentage to 0-1 score
                perf = sector_perf[symbol_sector]
                # Normalize: -10% to 10% range maps to 0-1
                score = (perf + 10) / 20
                return min(0.99, max(0.01, score))
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def get_btc_correlation_score(self, symbol: str, btc_prices: pd.Series,
                                   symbol_prices: pd.Series) -> float:
        """
        Get BTC correlation score (0-1)
        Higher score = positive correlation, Lower score = negative correlation
        
        Args:
            symbol: Trading pair symbol
            btc_prices: BTC price series
            symbol_prices: Symbol price series
        
        Returns:
            Score between 0 and 1
        """
        try:
            correlation = self.analyze_correlation_with_btc(symbol, btc_prices, symbol_prices)
            corr = correlation.get('correlation', 0)
            # Convert -1 to 1 range to 0-1
            return (corr + 1) / 2
        except Exception:
            return 0.5
    
    def get_beta_score(self, symbol: str, btc_prices: pd.Series,
                        symbol_prices: pd.Series) -> float:
        """
        Get beta score (0-1)
        Higher score = higher volatility relative to BTC
        
        Args:
            symbol: Trading pair symbol
            btc_prices: BTC price series
            symbol_prices: Symbol price series
        
        Returns:
            Score between 0 and 1
        """
        try:
            correlation = self.analyze_correlation_with_btc(symbol, btc_prices, symbol_prices)
            beta = correlation.get('beta', 1.0)
            # Normalize beta to 0-1 (beta 0-3 range)
            return min(0.99, max(0.01, beta / 3))
        except Exception:
            return 0.5
    
    def get_correlation_trend(self, btc_prices: pd.Series, 
                               symbol_prices: pd.Series,
                               windows: List[int] = [20, 50]) -> str:
        """
        Get correlation trend (INCREASING/DECREASING/STABLE)
        
        Args:
            btc_prices: BTC price series
            symbol_prices: Symbol price series
            windows: List of windows to compare
        
        Returns:
            Trend string
        """
        try:
            correlations = []
            for window in windows:
                if len(btc_prices) >= window and len(symbol_prices) >= window:
                    corr_result = self.analyze_correlation_with_btc(
                        'temp', btc_prices, symbol_prices, window
                    )
                    correlations.append(corr_result.get('correlation', 0))
            
            if len(correlations) >= 2:
                if correlations[-1] > correlations[0] + 0.1:
                    return 'INCREASING'
                elif correlations[-1] < correlations[0] - 0.1:
                    return 'DECREASING'
            
            return 'STABLE'
            
        except Exception:
            return 'STABLE'