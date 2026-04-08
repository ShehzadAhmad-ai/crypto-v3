# funding_oi_engine.py - Funding Rate and Open Interest Analysis
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

class FundingBias(Enum):
    STRONG_LONG = "STRONG_LONG"  # High positive funding (longs paying)
    MODERATE_LONG = "MODERATE_LONG"
    NEUTRAL = "NEUTRAL"
    MODERATE_SHORT = "MODERATE_SHORT"
    STRONG_SHORT = "STRONG_SHORT"  # High negative funding (shorts paying)

class OITrend(Enum):
    STRONG_INCREASING = "STRONG_INCREASING"
    INCREASING = "INCREASING"
    FLAT = "FLAT"
    DECREASING = "DECREASING"
    STRONG_DECREASING = "STRONG_DECREASING"

class LiquidationCluster(Enum):
    MAJOR = "MAJOR"  # Many liquidations clustered
    MODERATE = "MODERATE"
    MINOR = "MINOR"
    NONE = "NONE"

@dataclass
class FundingMetrics:
    """Funding rate metrics"""
    current_rate: float
    annualized_rate: float
    avg_8h: float
    avg_24h: float
    bias: FundingBias
    divergence: float  # Price vs funding divergence (-1 to 1)
    predicted_next: float
    timestamp: datetime

@dataclass
class OpenInterestMetrics:
    """Open Interest metrics"""
    current_oi: float
    oi_change_1h: float
    oi_change_24h: float
    trend: OITrend
    volume_ratio: float  # OI / Volume
    concentration: float  # Concentration risk (0-1)
    long_short_ratio: Optional[float]
    timestamp: datetime

@dataclass
class LiquidationClusterMap:
    """Liquidation cluster mapping"""
    clusters: List[Dict[str, Any]]
    total_liq_volume: float
    largest_cluster_price: Optional[float]
    cluster_severity: LiquidationCluster
    cascade_risk: float  # 0-1
    timestamp: datetime

@dataclass
class OIFundingSignal:
    """Combined OI and Funding signal"""
    direction: str  # 'LONG', 'SHORT', 'HOLD'
    confidence: float
    funding_bias: FundingBias
    oi_trend: OITrend
    liquidation_risk: float
    entry_zone: Optional[Tuple[float, float]]
    stop_zone: Optional[float]
    reasons: List[str]

class FundingOIAnalyzer:
    """
    Analyzes funding rates and open interest for institutional insights
    """
    
    def __init__(self):
        self.funding_history: List[FundingMetrics] = []
        self.oi_history: List[OpenInterestMetrics] = []
        self.liquidation_history: List[LiquidationClusterMap] = []
        
        # Thresholds
        self.high_funding_threshold = 0.01  # 1% per 8h
        self.extreme_funding_threshold = 0.03  # 3% per 8h
        self.oi_surge_threshold = 0.2  # 20% increase
    
    def analyze_funding_rates(self, funding_data: Dict[str, Any]) -> FundingMetrics:
        """
        Analyze funding rates to detect market bias
        """
        try:
            current = funding_data.get('current_rate', 0)
            history_8h = funding_data.get('history_8h', [])
            history_24h = funding_data.get('history_24h', [])
            
            # Calculate averages
            avg_8h = np.mean(history_8h) if history_8h else current
            avg_24h = np.mean(history_24h) if history_24h else current
            
            # Annualize (8h funding * 3 = daily, * 365 = annual)
            annualized = current * 3 * 365
            
            # Determine bias
            if current > self.extreme_funding_threshold:
                bias = FundingBias.STRONG_LONG
            elif current > self.high_funding_threshold:
                bias = FundingBias.MODERATE_LONG
            elif current < -self.extreme_funding_threshold:
                bias = FundingBias.STRONG_SHORT
            elif current < -self.high_funding_threshold:
                bias = FundingBias.MODERATE_SHORT
            else:
                bias = FundingBias.NEUTRAL
            
            # Calculate divergence (funding vs price trend)
            divergence = self._calculate_funding_divergence(funding_data)
            
            # Predict next funding rate
            predicted = self._predict_next_funding(history_8h, current)
            
            metrics = FundingMetrics(
                current_rate=current,
                annualized_rate=annualized,
                avg_8h=avg_8h,
                avg_24h=avg_24h,
                bias=bias,
                divergence=divergence,
                predicted_next=predicted,
                timestamp=datetime.now()
            )
            
            self.funding_history.append(metrics)
            return metrics
            
        except Exception as e:
            logging.error(f"Error in analyze_funding_rates: {e}")
            return self._default_funding_metrics()
    
    def _calculate_funding_divergence(self, funding_data: Dict) -> float:
        """Calculate divergence between funding and price (-1 to 1)"""
        try:
            price_change = funding_data.get('price_change_24h', 0)
            funding_change = funding_data.get('funding_change_24h', 0)
            
            if abs(price_change) > 0.01 and abs(funding_change) > 0:
                # Positive divergence: price up but funding down (bearish)
                # Negative divergence: price down but funding up (bullish)
                if price_change > 0 and funding_change < 0:
                    return -min(1.0, abs(funding_change) / 0.01)
                elif price_change < 0 and funding_change > 0:
                    return min(1.0, funding_change / 0.01)
            
            return 0
            
        except Exception:
            return 0
    
    def _predict_next_funding(self, history: List[float], current: float) -> float:
        """Predict next funding rate"""
        try:
            if len(history) >= 3:
                # Simple trend extrapolation
                recent_trend = np.mean([history[-1] - history[-2], history[-2] - history[-3]])
                predicted = current + recent_trend
                # Cap at reasonable levels
                return max(-0.05, min(0.05, predicted))
            
            return current
            
        except Exception:
            return current
    
    def _default_funding_metrics(self) -> FundingMetrics:
        """Default funding metrics"""
        return FundingMetrics(
            current_rate=0,
            annualized_rate=0,
            avg_8h=0,
            avg_24h=0,
            bias=FundingBias.NEUTRAL,
            divergence=0,
            predicted_next=0,
            timestamp=datetime.now()
        )
    
    def analyze_open_interest(self, oi_data: Dict[str, Any]) -> OpenInterestMetrics:
        """
        Analyze open interest for market sentiment
        """
        try:
            current = oi_data.get('current_oi', 0)
            oi_1h_ago = oi_data.get('oi_1h_ago', current)
            oi_24h_ago = oi_data.get('oi_24h_ago', current)
            
            # Calculate changes
            oi_change_1h = (current - oi_1h_ago) / oi_1h_ago if oi_1h_ago > 0 else 0
            oi_change_24h = (current - oi_24h_ago) / oi_24h_ago if oi_24h_ago > 0 else 0
            
            # Determine trend
            if oi_change_24h > 0.3:
                trend = OITrend.STRONG_INCREASING
            elif oi_change_24h > 0.1:
                trend = OITrend.INCREASING
            elif oi_change_24h < -0.3:
                trend = OITrend.STRONG_DECREASING
            elif oi_change_24h < -0.1:
                trend = OITrend.DECREASING
            else:
                trend = OITrend.FLAT
            
            # Calculate volume ratio
            volume = oi_data.get('volume_24h', 0)
            volume_ratio = volume / current if current > 0 else 1
            
            # Calculate concentration (simplified)
            concentration = oi_data.get('concentration', 0.5)
            
            # Long/short ratio
            long_short_ratio = oi_data.get('long_short_ratio')
            
            metrics = OpenInterestMetrics(
                current_oi=current,
                oi_change_1h=oi_change_1h,
                oi_change_24h=oi_change_24h,
                trend=trend,
                volume_ratio=volume_ratio,
                concentration=concentration,
                long_short_ratio=long_short_ratio,
                timestamp=datetime.now()
            )
            
            self.oi_history.append(metrics)
            return metrics
            
        except Exception as e:
            logging.error(f"Error in analyze_open_interest: {e}")
            return self._default_oi_metrics()
    
    def _default_oi_metrics(self) -> OpenInterestMetrics:
        """Default OI metrics"""
        return OpenInterestMetrics(
            current_oi=0,
            oi_change_1h=0,
            oi_change_24h=0,
            trend=OITrend.FLAT,
            volume_ratio=1,
            concentration=0,
            long_short_ratio=None,
            timestamp=datetime.now()
        )
    
    def map_liquidation_clusters(self, liquidation_data: List[Dict]) -> LiquidationClusterMap:
        """
        Map liquidation clusters to identify key levels
        """
        try:
            if not liquidation_data:
                return self._default_liquidation_map()
            
            # Group liquidations by price level
            clusters = []
            price_levels = {}
            
            for liq in liquidation_data:
                price = liq.get('price', 0)
                volume = liq.get('volume', 0)
                
                # Round to nearest significant level
                rounded_price = round(price, -int(np.floor(np.log10(price))) + 2)
                
                if rounded_price not in price_levels:
                    price_levels[rounded_price] = {
                        'total_volume': 0,
                        'count': 0,
                        'avg_price': rounded_price
                    }
                
                price_levels[rounded_price]['total_volume'] += volume
                price_levels[rounded_price]['count'] += 1
            
            # Convert to clusters
            for price, data in price_levels.items():
                clusters.append({
                    'price': price,
                    'volume': data['total_volume'],
                    'count': data['count'],
                    'strength': min(1.0, data['total_volume'] / 1e7)  # Normalize
                })
            
            # Sort by volume
            clusters.sort(key=lambda x: x['volume'], reverse=True)
            
            total_volume = sum(c['volume'] for c in clusters)
            largest_cluster = clusters[0]['price'] if clusters else None
            
            # Determine severity
            if total_volume > 1e8:  # $100M+
                severity = LiquidationCluster.MAJOR
            elif total_volume > 1e7:  # $10M+
                severity = LiquidationCluster.MODERATE
            elif total_volume > 1e6:  # $1M+
                severity = LiquidationCluster.MINOR
            else:
                severity = LiquidationCluster.NONE
            
            # Calculate cascade risk
            cascade_risk = self._calculate_cascade_risk(clusters)
            
            return LiquidationClusterMap(
                clusters=clusters[:10],  # Top 10 clusters
                total_liq_volume=total_volume,
                largest_cluster_price=largest_cluster,
                cluster_severity=severity,
                cascade_risk=cascade_risk,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Error in map_liquidation_clusters: {e}")
            return self._default_liquidation_map()
    
    def _calculate_cascade_risk(self, clusters: List[Dict]) -> float:
        """Calculate risk of liquidation cascade"""
        try:
            if len(clusters) < 3:
                return 0
            
            # Check if clusters are close together
            prices = [c['price'] for c in clusters[:5]]
            if len(prices) < 2:
                return 0
            
            price_spread = max(prices) - min(prices)
            avg_price = np.mean(prices)
            spread_pct = price_spread / avg_price if avg_price > 0 else 1
            
            # Tight clusters = higher cascade risk
            if spread_pct < 0.01:  # Within 1%
                return 0.8
            elif spread_pct < 0.03:  # Within 3%
                return 0.5
            elif spread_pct < 0.05:  # Within 5%
                return 0.3
            else:
                return 0.1
            
        except Exception:
            return 0
    
    def _default_liquidation_map(self) -> LiquidationClusterMap:
        """Default liquidation map"""
        return LiquidationClusterMap(
            clusters=[],
            total_liq_volume=0,
            largest_cluster_price=None,
            cluster_severity=LiquidationCluster.NONE,
            cascade_risk=0,
            timestamp=datetime.now()
        )
    
    def generate_oi_funding_signal(self, funding: FundingMetrics, oi: OpenInterestMetrics,
                                  price: float, liquidation_map: Optional[LiquidationClusterMap] = None) -> OIFundingSignal:
        """
        Generate combined signal from OI and funding data
        """
        try:
            reasons = []
            confidence = 0.5
            direction = 'HOLD'
            
            # Funding analysis
            if funding.bias in [FundingBias.STRONG_LONG, FundingBias.MODERATE_LONG]:
                if funding.divergence < -0.3:
                    # High funding but price not following - bearish
                    direction = 'SHORT'
                    confidence += 0.2
                    reasons.append(f"Funding divergence: {funding.divergence:.2f}")
                elif funding.current_rate > self.high_funding_threshold:
                    # High funding, price following - cautious long
                    direction = 'LONG'
                    confidence += 0.1
                    reasons.append(f"High funding: {funding.current_rate:.4f}")
            
            elif funding.bias in [FundingBias.STRONG_SHORT, FundingBias.MODERATE_SHORT]:
                if funding.divergence > 0.3:
                    # Negative funding but price strong - bullish
                    direction = 'LONG'
                    confidence += 0.2
                    reasons.append(f"Funding divergence: {funding.divergence:.2f}")
                elif funding.current_rate < -self.high_funding_threshold:
                    # High negative funding, price following - cautious short
                    direction = 'SHORT'
                    confidence += 0.1
                    reasons.append(f"Negative funding: {funding.current_rate:.4f}")
            
            # OI analysis
            if oi.trend in [OITrend.STRONG_INCREASING, OITrend.INCREASING]:
                if direction == 'LONG':
                    confidence += 0.15
                    reasons.append(f"OI increasing: {oi.oi_change_24h:.1%}")
                elif direction == 'SHORT':
                    confidence += 0.1
                    reasons.append(f"OI increasing with short bias")
                else:
                    # OI increasing but no direction - momentum building
                    confidence += 0.1
                    reasons.append("OI building")
            
            elif oi.trend in [OITrend.STRONG_DECREASING, OITrend.DECREASING]:
                # OI decreasing - potential reversal
                confidence -= 0.1
                reasons.append(f"OI decreasing: {oi.oi_change_24h:.1%}")
            
            # Liquidation cluster analysis
            liquidation_risk = 0
            if liquidation_map and liquidation_map.largest_cluster_price:
                distance = abs(price - liquidation_map.largest_cluster_price) / price
                if distance < 0.02:  # Within 2% of major liquidation
                    liquidation_risk = 0.7
                    reasons.append(f"Near liquidation cluster at {liquidation_map.largest_cluster_price:.2f}")
                    
                    if direction == 'LONG' and price < liquidation_map.largest_cluster_price:
                        confidence -= 0.2  # Risk of long liquidation
                    elif direction == 'SHORT' and price > liquidation_map.largest_cluster_price:
                        confidence -= 0.2  # Risk of short liquidation
                elif distance < 0.05:
                    liquidation_risk = 0.3
            
            # Calculate entry zone
            entry_zone = None
            if liquidation_map and liquidation_map.largest_cluster_price:
                if direction == 'LONG':
                    entry_zone = (liquidation_map.largest_cluster_price * 0.995, 
                                 liquidation_map.largest_cluster_price * 1.005)
                elif direction == 'SHORT':
                    entry_zone = (liquidation_map.largest_cluster_price * 1.005,
                                 liquidation_map.largest_cluster_price * 0.995)
            
            # Stop zone based on liquidation clusters
            stop_zone = None
            if liquidation_map and liquidation_map.largest_cluster_price:
                if direction == 'LONG':
                    stop_zone = liquidation_map.largest_cluster_price * 0.99
                elif direction == 'SHORT':
                    stop_zone = liquidation_map.largest_cluster_price * 1.01
            
            return OIFundingSignal(
                direction=direction,
                confidence=min(1.0, max(0, confidence)),
                funding_bias=funding.bias,
                oi_trend=oi.trend,
                liquidation_risk=liquidation_risk,
                entry_zone=entry_zone,
                stop_zone=stop_zone,
                reasons=reasons
            )
            
        except Exception as e:
            logging.error(f"Error in generate_oi_funding_signal: {e}")
            return OIFundingSignal(
                direction='HOLD',
                confidence=0,
                funding_bias=FundingBias.NEUTRAL,
                oi_trend=OITrend.FLAT,
                liquidation_risk=0,
                entry_zone=None,
                stop_zone=None,
                reasons=[]
            )
        

    # ==================== SCORING METHODS FOR LIGHT CONFIRM ====================
    
    def get_funding_score(self, funding_data: Dict[str, Any]) -> float:
        """
        Get funding rate sentiment score (0-1)
        Higher score = bullish (negative funding or shorts paying)
        Lower score = bearish (positive funding or longs paying)
        """
        try:
            funding = self.analyze_funding_rates(funding_data)
            
            # Convert bias to score
            if funding.bias == FundingBias.STRONG_SHORT:
                return 0.9  # Strong negative funding = bullish
            elif funding.bias == FundingBias.MODERATE_SHORT:
                return 0.75
            elif funding.bias == FundingBias.NEUTRAL:
                return 0.5
            elif funding.bias == FundingBias.MODERATE_LONG:
                return 0.25
            elif funding.bias == FundingBias.STRONG_LONG:
                return 0.1
            else:
                return 0.5
                
        except Exception as e:
            logging.error(f"Error in get_funding_score: {e}")
            return 0.5
    
    def get_oi_score(self, oi_data: Dict[str, Any]) -> float:
        """
        Get open interest sentiment score (0-1)
        Higher score = bullish (OI increasing with price)
        Lower score = bearish (OI decreasing or increasing with divergence)
        """
        try:
            oi = self.analyze_open_interest(oi_data)
            
            # Base score on trend
            if oi.trend in [OITrend.STRONG_INCREASING, OITrend.INCREASING]:
                score = 0.7
            elif oi.trend in [OITrend.STRONG_DECREASING, OITrend.DECREASING]:
                score = 0.3
            else:
                score = 0.5
            
            # Adjust by volume ratio
            if oi.volume_ratio > 1.5:
                score += 0.1
            elif oi.volume_ratio < 0.5:
                score -= 0.1
            
            # Adjust by long/short ratio if available
            if oi.long_short_ratio is not None:
                if oi.long_short_ratio > 1.5:
                    score -= 0.1  # Too many longs = bearish
                elif oi.long_short_ratio < 0.7:
                    score += 0.1  # More shorts = bullish
            
            return min(0.99, max(0.01, score))
            
        except Exception as e:
            logging.error(f"Error in get_oi_score: {e}")
            return 0.5
    
    def get_funding_oi_score(self, funding_data: Dict[str, Any], 
                              oi_data: Dict[str, Any],
                              weights: Dict[str, float] = None) -> float:
        """
        Get combined funding and OI score (0-1)
        
        Args:
            funding_data: Funding rate data
            oi_data: Open interest data
            weights: Optional custom weights {'funding': 0.5, 'oi': 0.5}
        """
        try:
            if weights is None:
                weights = {'funding': 0.5, 'oi': 0.5}
            
            funding_score = self.get_funding_score(funding_data)
            oi_score = self.get_oi_score(oi_data)
            
            combined = (funding_score * weights['funding']) + (oi_score * weights['oi'])
            
            return min(0.99, max(0.01, combined))
            
        except Exception as e:
            logging.error(f"Error in get_funding_oi_score: {e}")
            return 0.5
    
    def get_funding_oi_summary(self, funding_data: Dict[str, Any],
                                oi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive funding and OI summary for light confirm pipeline
        """
        try:
            funding = self.analyze_funding_rates(funding_data)
            oi = self.analyze_open_interest(oi_data)
            
            funding_score = self.get_funding_score(funding_data)
            oi_score = self.get_oi_score(oi_data)
            combined_score = self.get_funding_oi_score(funding_data, oi_data)
            
            # Determine bias
            if funding.bias in [FundingBias.STRONG_SHORT, FundingBias.MODERATE_SHORT]:
                funding_bias = 'BULLISH'
            elif funding.bias in [FundingBias.STRONG_LONG, FundingBias.MODERATE_LONG]:
                funding_bias = 'BEARISH'
            else:
                funding_bias = 'NEUTRAL'
            
            if oi.trend in [OITrend.STRONG_INCREASING, OITrend.INCREASING]:
                oi_bias = 'BULLISH'
            elif oi.trend in [OITrend.STRONG_DECREASING, OITrend.DECREASING]:
                oi_bias = 'BEARISH'
            else:
                oi_bias = 'NEUTRAL'
            
            reasons = []
            if funding_score > 0.6:
                reasons.append(f"Funding: {funding.current_rate*100:.4f}% (bullish)")
            elif funding_score < 0.4:
                reasons.append(f"Funding: {funding.current_rate*100:.4f}% (bearish)")
            
            if oi_score > 0.6:
                reasons.append(f"OI: {oi.oi_change_24h:.1%} change (bullish)")
            elif oi_score < 0.4:
                reasons.append(f"OI: {oi.oi_change_24h:.1%} change (bearish)")
            
            return {
                'score': combined_score,
                'funding_score': funding_score,
                'oi_score': oi_score,
                'funding_rate': funding.current_rate,
                'funding_annualized': funding.annualized_rate,
                'funding_bias': funding_bias,
                'oi_value': oi.current_oi,
                'oi_change_24h': oi.oi_change_24h,
                'oi_trend': oi.trend.value,
                'oi_bias': oi_bias,
                'long_short_ratio': oi.long_short_ratio,
                'reasons': reasons
            }
            
        except Exception as e:
            logging.error(f"Error in get_funding_oi_summary: {e}")
            return {
                'score': 0.5,
                'funding_score': 0.5,
                'oi_score': 0.5,
                'reasons': ['Error in funding/OI analysis']
            }
    
    def is_funding_bullish(self, funding_data: Dict[str, Any]) -> bool:
        """Quick check if funding is bullish (negative funding)"""
        try:
            funding = self.analyze_funding_rates(funding_data)
            return funding.bias in [FundingBias.STRONG_SHORT, FundingBias.MODERATE_SHORT]
        except Exception:
            return False
    
    def is_funding_bearish(self, funding_data: Dict[str, Any]) -> bool:
        """Quick check if funding is bearish (positive funding)"""
        try:
            funding = self.analyze_funding_rates(funding_data)
            return funding.bias in [FundingBias.STRONG_LONG, FundingBias.MODERATE_LONG]
        except Exception:
            return False
    
    def is_oi_building(self, oi_data: Dict[str, Any], threshold: float = 0.1) -> bool:
        """Check if open interest is building (increasing)"""
        try:
            oi = self.analyze_open_interest(oi_data)
            return oi.oi_change_24h > threshold
        except Exception:
            return False
    
    def is_oi_declining(self, oi_data: Dict[str, Any], threshold: float = 0.1) -> bool:
        """Check if open interest is declining"""
        try:
            oi = self.analyze_open_interest(oi_data)
            return oi.oi_change_24h < -threshold
        except Exception:
            return False

    
    def detect_funding_arbitrage(self, funding_data: Dict[str, float]) -> List[Dict]:
        """
        Detect funding rate arbitrage opportunities across exchanges
        """
        try:
            opportunities = []
            
            exchanges = list(funding_data.keys())
            for i in range(len(exchanges)):
                for j in range(i + 1, len(exchanges)):
                    ex1 = exchanges[i]
                    ex2 = exchanges[j]
                    
                    rate1 = funding_data[ex1]
                    rate2 = funding_data[ex2]
                    
                    diff = abs(rate1 - rate2)
                    
                    if diff > 0.005:  # 0.5% difference
                        opportunities.append({
                            'exchange1': ex1,
                            'exchange2': ex2,
                            'rate1': rate1,
                            'rate2': rate2,
                            'spread': diff,
                            'annualized_spread': diff * 3 * 365,
                            'direction': 'LONG on lower, SHORT on higher'
                        })
            
            return opportunities
            
        except Exception:
            return []