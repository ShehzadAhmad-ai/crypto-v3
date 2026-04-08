# cross_asset_engine.py - Cross-Asset Correlation and Flow Analysis
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from scipy import stats

class AssetClass(Enum):
    CRYPTO = "CRYPTO"
    STABLE_COIN = "STABLE_COIN"
    DEFI = "DEFI"
    L1 = "L1"
    L2 = "L2"
    MEME = "MEME"
    AI = "AI"
    GAMING = "GAMING"

class CorrelationStrength(Enum):
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"

@dataclass
class CorrelationMatrix:
    """Correlation matrix between assets"""
    assets: List[str]
    matrix: pd.DataFrame
    timestamp: datetime
    avg_correlation: float
    regime: str  # 'HIGH_CORR', 'LOW_CORR', 'DECOUPLING'

@dataclass
class StablecoinFlow:
    """Stablecoin flow analysis"""
    symbol: str
    inflow_24h: float
    outflow_24h: float
    net_flow: float
    exchange_balances: Dict[str, float]
    supply_change_24h: float
    velocity: float  # turnover rate
    buying_pressure: float  # -1 to 1
    risk_sentiment: str  # 'RISK_ON', 'RISK_OFF', 'NEUTRAL'

@dataclass
class CrossAssetSignal:
    """Cross-asset trading signal"""
    asset: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    drivers: List[str]
    correlation_impact: float
    hedge_suggestion: Optional[str]
    pair_trade: Optional[Dict]

class CrossAssetEngine:
    """
    Cross-asset correlation and flow analysis engine
    Tracks relationships between assets and stablecoin flows
    """
    
    def __init__(self):
        self.correlation_history: List[CorrelationMatrix] = []
        self.stablecoin_flows: Dict[str, StablecoinFlow] = {}
        self.asset_classes: Dict[str, AssetClass] = {}
        
        # Define asset classes (simplified)
        self._init_asset_classes()
    
    def _init_asset_classes(self):
        """Initialize asset class mappings"""
        # Major L1s
        l1s = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'ADA/USDT']
        for asset in l1s:
            self.asset_classes[asset] = AssetClass.L1
        
        # L2s
        l2s = ['ARB/USDT', 'OP/USDT', 'MATIC/USDT']
        for asset in l2s:
            self.asset_classes[asset] = AssetClass.L2
        
        # DeFi
        defi = ['UNI/USDT', 'AAVE/USDT', 'MKR/USDT', 'COMP/USDT']
        for asset in defi:
            self.asset_classes[asset] = AssetClass.DEFI
        
        # Stablecoins
        stables = ['USDT/USDT', 'USDC/USDT', 'DAI/USDT']
        for asset in stables:
            self.asset_classes[asset] = AssetClass.STABLE_COIN
    
    def calculate_correlation_matrix(self, price_data: Dict[str, pd.Series], 
                                    window: int = 100) -> CorrelationMatrix:
        """
        Calculate correlation matrix between multiple assets
        """
        try:
            assets = list(price_data.keys())
            if len(assets) < 2:
                return self._default_correlation(assets)
            
            # Calculate returns
            returns_dict = {}
            for asset, prices in price_data.items():
                if len(prices) > window:
                    returns = prices.pct_change().dropna()
                    returns_dict[asset] = returns.iloc[-window:]
            
            # Align indices
            common_idx = None
            for asset, returns in returns_dict.items():
                if common_idx is None:
                    common_idx = set(returns.index)
                else:
                    common_idx = common_idx.intersection(set(returns.index))
            
            if not common_idx:
                return self._default_correlation(assets)
            
            common_idx = sorted(common_idx)
            
            # Build correlation matrix
            n = len(returns_dict)
            matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)
            
            corr_values = []
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i <= j:
                        if asset1 in returns_dict and asset2 in returns_dict:
                            r1 = returns_dict[asset1].loc[common_idx].values
                            r2 = returns_dict[asset2].loc[common_idx].values
                            corr = np.corrcoef(r1, r2)[0, 1]
                            if not np.isnan(corr):
                                matrix.loc[asset1, asset2] = corr
                                matrix.loc[asset2, asset1] = corr
                                corr_values.append(abs(corr))
                            else:
                                matrix.loc[asset1, asset2] = 0
                                matrix.loc[asset2, asset1] = 0
            
            avg_corr = np.mean(corr_values) if corr_values else 0
            
            # Determine regime
            if avg_corr > 0.7:
                regime = 'HIGH_CORR'
            elif avg_corr > 0.4:
                regime = 'MODERATE_CORR'
            elif avg_corr > 0.2:
                regime = 'LOW_CORR'
            else:
                regime = 'DECOUPLING'
            
            correlation = CorrelationMatrix(
                assets=assets,
                matrix=matrix,
                timestamp=datetime.now(),
                avg_correlation=avg_corr,
                regime=regime
            )
            
            self.correlation_history.append(correlation)
            return correlation
            
        except Exception as e:
            logging.error(f"Error in calculate_correlation_matrix: {e}")
            return self._default_correlation(assets)
    
    def _default_correlation(self, assets: List[str]) -> CorrelationMatrix:
        """Default correlation matrix"""
        n = len(assets)
        matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)
        for i in range(n):
            for j in range(n):
                matrix.iloc[i, j] = 1.0 if i == j else 0.0
        
        return CorrelationMatrix(
            assets=assets,
            matrix=matrix,
            timestamp=datetime.now(),
            avg_correlation=0,
            regime='UNKNOWN'
        )
    
    def analyze_stablecoin_flows(self, flow_data: Dict[str, Any]) -> Dict[str, StablecoinFlow]:
        """
        Analyze stablecoin flows for market direction signals
        """
        try:
            results = {}
            
            for stablecoin, data in flow_data.items():
                inflow = data.get('inflow_24h', 0)
                outflow = data.get('outflow_24h', 0)
                net_flow = inflow - outflow
                
                # Calculate buying pressure
                total_flow = inflow + outflow
                buying_pressure = (inflow - outflow) / total_flow if total_flow > 0 else 0
                
                # Determine risk sentiment
                if net_flow > 0 and buying_pressure > 0.3:
                    sentiment = 'RISK_ON'
                elif net_flow < 0 and buying_pressure < -0.3:
                    sentiment = 'RISK_OFF'
                else:
                    sentiment = 'NEUTRAL'
                
                flow = StablecoinFlow(
                    symbol=stablecoin,
                    inflow_24h=inflow,
                    outflow_24h=outflow,
                    net_flow=net_flow,
                    exchange_balances=data.get('exchange_balances', {}),
                    supply_change_24h=data.get('supply_change', 0),
                    velocity=data.get('velocity', 1.0),
                    buying_pressure=buying_pressure,
                    risk_sentiment=sentiment
                )
                
                results[stablecoin] = flow
                self.stablecoin_flows[stablecoin] = flow
            
            return results
            
        except Exception as e:
            logging.error(f"Error in analyze_stablecoin_flows: {e}")
            return {}
    
    def detect_regime_shifts(self, current_corr: CorrelationMatrix, 
                            lookback: int = 10) -> Dict[str, Any]:
        """
        Detect shifts in correlation regimes
        """
        try:
            if len(self.correlation_history) < lookback:
                return {'shift_detected': False, 'regime': 'INSUFFICIENT_DATA'}
            
            recent_corrs = [c.avg_correlation for c in self.correlation_history[-lookback:]]
            older_corrs = [c.avg_correlation for c in self.correlation_history[-lookback*2:-lookback]]
            
            if not older_corrs:
                return {'shift_detected': False, 'regime': current_corr.regime}
            
            recent_avg = np.mean(recent_corrs)
            older_avg = np.mean(older_corrs)
            
            change = recent_avg - older_avg
            change_pct = change / older_avg if older_avg != 0 else 0
            
            shift_detected = abs(change_pct) > 0.3  # 30% change
            
            return {
                'shift_detected': shift_detected,
                'regime': current_corr.regime,
                'previous_regime': self._classify_correlation(older_avg),
                'change': change,
                'change_pct': change_pct,
                'direction': 'INCREASING' if change > 0 else 'DECREASING'
            }
            
        except Exception as e:
            logging.error(f"Error in detect_regime_shifts: {e}")
            return {'shift_detected': False, 'regime': 'UNKNOWN'}
    
    def _classify_correlation(self, corr: float) -> str:
        """Classify correlation strength"""
        if corr > 0.7:
            return 'HIGH_CORR'
        elif corr > 0.4:
            return 'MODERATE_CORR'
        elif corr > 0.2:
            return 'LOW_CORR'
        else:
            return 'DECOUPLING'
    
    def find_hedge_assets(self, asset: str, correlation_matrix: CorrelationMatrix,
                         min_correlation: float = 0.7) -> List[str]:
        """
        Find assets that are highly correlated (potential hedges)
        """
        try:
            if asset not in correlation_matrix.matrix.index:
                return []
            
            correlations = correlation_matrix.matrix.loc[asset]
            hedges = []
            
            for other_asset, corr in correlations.items():
                if other_asset != asset and abs(corr) > min_correlation:
                    hedges.append(other_asset)
            
            return hedges
            
        except Exception:
            return []
    
    def detect_pair_trade_opportunities(self, correlation_matrix: CorrelationMatrix,
                                       price_data: Dict[str, pd.Series],
                                       zscore_threshold: float = 2.0) -> List[Dict]:
        """
        Detect pair trading opportunities when correlations break down
        """
        try:
            opportunities = []
            
            assets = list(price_data.keys())
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1 = assets[i]
                    asset2 = assets[j]
                    
                    if asset1 not in price_data or asset2 not in price_data:
                        continue
                    
                    # Get price series
                    p1 = price_data[asset1]
                    p2 = price_data[asset2]
                    
                    # Align indices
                    common_idx = p1.index.intersection(p2.index)
                    if len(common_idx) < 30:
                        continue
                    
                    p1_aligned = p1.loc[common_idx]
                    p2_aligned = p2.loc[common_idx]
                    
                    # Calculate ratio and z-score
                    ratio = p1_aligned / p2_aligned
                    zscore = (ratio - ratio.mean()) / ratio.std()
                    
                    current_z = zscore.iloc[-1]
                    
                    # Check for deviation
                    if abs(current_z) > zscore_threshold:
                        # Determine trade direction
                        if current_z > zscore_threshold:
                            # Asset1 overvalued relative to Asset2
                            opportunities.append({
                                'pair': f"{asset1}/{asset2}",
                                'asset1': asset1,
                                'asset2': asset2,
                                'zscore': current_z,
                                'trade': f"SHORT {asset1}, LONG {asset2}",
                                'confidence': min(1.0, abs(current_z) / 4)
                            })
                        else:
                            # Asset1 undervalued relative to Asset2
                            opportunities.append({
                                'pair': f"{asset1}/{asset2}",
                                'asset1': asset1,
                                'asset2': asset2,
                                'zscore': current_z,
                                'trade': f"LONG {asset1}, SHORT {asset2}",
                                'confidence': min(1.0, abs(current_z) / 4)
                            })
            
            return opportunities
            
        except Exception as e:
            logging.error(f"Error in detect_pair_trade_opportunities: {e}")
            return []
    
    def generate_cross_asset_signal(self, asset: str, correlation_matrix: CorrelationMatrix,
                                   stablecoin_flows: Dict[str, StablecoinFlow],
                                   price_data: pd.Series) -> CrossAssetSignal:
        """
        Generate trading signal based on cross-asset analysis
        """
        try:
            confidence = 0.5
            direction = 'HOLD'
            drivers = []
            
            # Check correlation with major assets
            if asset in correlation_matrix.matrix.index:
                btc_corr = correlation_matrix.matrix.loc[asset].get('BTC/USDT', 0)
                eth_corr = correlation_matrix.matrix.loc[asset].get('ETH/USDT', 0)
                
                if btc_corr > 0.7:
                    drivers.append(f"High BTC correlation: {btc_corr:.2f}")
                    confidence += 0.1
                if eth_corr > 0.7:
                    drivers.append(f"High ETH correlation: {eth_corr:.2f}")
                    confidence += 0.1
            
            # Check stablecoin flows
            usdt_flow = stablecoin_flows.get('USDT/USDT')
            if usdt_flow:
               if usdt_flow.buying_pressure > 0.3:
                   if direction == 'BUY':
                       confidence += 0.15
                       drivers.append(f"USDT buying pressure confirms bullish bias")
                   elif direction == 'SELL':
                      confidence -= 0.1
                      drivers.append(f"USDT buying pressure conflicts with bearish bias")
                   else:
                       confidence += 0.1
                       drivers.append(f"Strong USDT buying pressure: {usdt_flow.buying_pressure:.2f}")
               elif usdt_flow.buying_pressure < -0.3:
                   if direction == 'SELL':
                       confidence += 0.15
                       drivers.append(f"USDT selling pressure confirms bearish bias")
                   elif direction == 'BUY':
                       confidence -= 0.1
                       drivers.append(f"USDT selling pressure conflicts with bullish bias")
                   else:
                      confidence -= 0.1
                      drivers.append(f"Strong USDT selling pressure: {usdt_flow.buying_pressure:.2f}")
            
            # Determine direction based on recent price action
            if len(price_data) > 20:
                recent_return = (price_data.iloc[-1] / price_data.iloc[-20] - 1) * 100
                if recent_return > 5 and confidence > 0.6:
                    direction = 'BUY'
                    drivers.append(f"Strong performance: +{recent_return:.1f}%")
                elif recent_return < -5 and confidence > 0.6:
                    direction = 'SELL'
                    drivers.append(f"Weak performance: {recent_return:.1f}%")
            
            # Find hedge suggestion
            hedges = self.find_hedge_assets(asset, correlation_matrix)
            hedge_suggestion = hedges[0] if hedges else None
            
            return CrossAssetSignal(
                asset=asset,
                direction=direction,
                confidence=min(1.0, confidence),
                drivers=drivers,
                correlation_impact=confidence * 0.3,
                hedge_suggestion=hedge_suggestion,
                pair_trade=None
            )
            
        except Exception as e:
            logging.error(f"Error in generate_cross_asset_signal: {e}")
            return CrossAssetSignal(
                asset=asset,
                direction='HOLD',
                confidence=0,
                drivers=[],
                correlation_impact=0,
                hedge_suggestion=None,
                pair_trade=None
            )
    # ==================== BTC DOMINANCE ANALYSIS ====================
    
    def analyze_btc_dominance(self, btc_price: pd.Series, 
                               total_market_cap: Optional[pd.Series] = None,
                               eth_price: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze BTC dominance and its impact on altcoins
        
        Args:
            btc_price: BTC price series
            total_market_cap: Total crypto market cap (if available)
            eth_price: ETH price series (for ETH/BTC ratio proxy)
        
        Returns:
            Dict with dominance analysis
        """
        try:
            result = {
                'btc_dominance_trend': 'NEUTRAL',
                'altcoin_season': False,
                'dominance_score': 0.5,
                'btc_dominance': 50.0,
                'reasons': []
            }
            
            if len(btc_price) < 20:
                return result
            
            # Calculate BTC dominance using market cap if available
            if total_market_cap is not None and len(total_market_cap) >= 20:
                # Estimate BTC market cap (price * approximate supply)
                btc_supply = 19500000
                btc_mcap = btc_price * btc_supply
                dominance = (btc_mcap / total_market_cap) * 100
                
                # Get current dominance and trend
                current_dominance = dominance.iloc[-1]
                dominance_20_ago = dominance.iloc[-20] if len(dominance) >= 20 else dominance.iloc[0]
                dom_change = ((current_dominance - dominance_20_ago) / dominance_20_ago) * 100
                
                result['btc_dominance'] = current_dominance
                
                # Determine if altcoin season
                if dom_change < -3:  # 3% decrease in dominance
                    result['btc_dominance_trend'] = 'DECREASING'
                    result['altcoin_season'] = True
                    result['dominance_score'] = 0.8
                    result['reasons'].append(f"BTC dominance decreasing ({dom_change:.1f}%) - Altcoin season likely")
                elif dom_change > 3:  # 3% increase in dominance
                    result['btc_dominance_trend'] = 'INCREASING'
                    result['altcoin_season'] = False
                    result['dominance_score'] = 0.3
                    result['reasons'].append(f"BTC dominance increasing ({dom_change:.1f}%) - BTC strength")
                else:
                    result['reasons'].append("BTC dominance stable")
            
            # Fallback: use ETH/BTC ratio as proxy
            elif eth_price is not None and len(eth_price) >= 20:
                eth_btc_ratio = eth_price / btc_price
                ratio_change = (eth_btc_ratio.iloc[-1] / eth_btc_ratio.iloc[-20] - 1) * 100
                
                # When ETH outperforms BTC, altcoin season likely
                if ratio_change > 5:
                    result['altcoin_season'] = True
                    result['dominance_score'] = 0.75
                    result['reasons'].append(f"ETH/BTC ratio up {ratio_change:.1f}% - Altcoin season")
                elif ratio_change < -5:
                    result['altcoin_season'] = False
                    result['dominance_score'] = 0.35
                    result['reasons'].append(f"ETH/BTC ratio down {ratio_change:.1f}% - BTC dominance")
                else:
                    result['reasons'].append("ETH/BTC ratio stable")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in analyze_btc_dominance: {e}")
            return result
    
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
    
    def get_altcoin_season_score(self, btc_dominance_result: Dict, 
                                  sector_rotation_result: Dict) -> float:
        """
        Get combined altcoin season score (0-1)
        Higher score = altcoin season (bullish for alts)
        Lower score = BTC season (bearish for alts)
        """
        try:
            # Factor 1: BTC dominance
            dom_score = btc_dominance_result.get('dominance_score', 0.5)
            
            # Factor 2: Sector rotation
            rotation_score = sector_rotation_result.get('rotation_score', 0.5)
            
            # Combined score (60% dominance, 40% rotation)
            score = (dom_score * 0.6) + (rotation_score * 0.4)
            
            return min(0.99, max(0.01, score))
            
        except Exception:
            return 0.5
    
    def get_cross_asset_score(self, asset: str, btc_correlation: Dict,
                               btc_dominance: Dict, sector_rotation: Dict) -> float:
        """
        Get overall cross-asset score for a specific asset (0-1)
        """
        try:
            score = 0.5
            
            # Factor 1: Correlation with BTC
            corr = btc_correlation.get('correlation', 0)
            if corr > 0.7:
                # High correlation - follow BTC dominance
                if btc_dominance.get('altcoin_season', False):
                    score += 0.15  # Altcoin season + high correlation = bullish for alts
                else:
                    score -= 0.1
            else:
                # Low correlation - independent move
                score += 0.05
            
            # Factor 2: Sector strength
            strongest_sector = sector_rotation.get('strongest_sector', 'UNKNOWN')
            if strongest_sector != 'UNKNOWN':
                # Check if asset belongs to strong sector
                for sector, symbols in self.sectors.items():
                    if asset in symbols:
                        if sector == strongest_sector:
                            score += 0.2
                        elif sector == sector_rotation.get('weakest_sector'):
                            score -= 0.15
                        break
            
            return min(0.99, max(0.01, score))
            
        except Exception:
            return 0.5
    
    def get_cross_asset_summary(self, asset: str, btc_correlation: Dict,
                                  btc_dominance: Dict, sector_rotation: Dict) -> Dict[str, Any]:
        """
        Get comprehensive cross-asset summary for light confirm pipeline
        """
        return {
            'score': self.get_cross_asset_score(asset, btc_correlation, btc_dominance, sector_rotation),
            'btc_correlation': btc_correlation.get('correlation', 0),
            'btc_correlation_strength': btc_correlation.get('correlation_strength', 'NEUTRAL'),
            'beta': btc_correlation.get('beta', 1.0),
            'btc_dominance': btc_dominance.get('btc_dominance', 50),
            'btc_dominance_trend': btc_dominance.get('btc_dominance_trend', 'NEUTRAL'),
            'altcoin_season': btc_dominance.get('altcoin_season', False),
            'strongest_sector': sector_rotation.get('strongest_sector', 'UNKNOWN'),
            'weakest_sector': sector_rotation.get('weakest_sector', 'UNKNOWN'),
            'sector_performance': sector_rotation.get('sector_performance', {}),
            'reasons': btc_correlation.get('reasons', []) + 
                       btc_dominance.get('reasons', []) + 
                       sector_rotation.get('reasons', [])
        }