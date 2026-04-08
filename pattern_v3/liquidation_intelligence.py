# liquidation_intelligence.py
"""
Liquidation Intelligence Module
Processes real liquidation data from Binance Futures API
Detects liquidation events, clusters, and cascade risk
"""
import sys
import os
# Add parent directory (main project folder) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from config import Config
from logger import log


@dataclass
class LiquidationEvent:
    """Single liquidation event"""
    timestamp: datetime
    symbol: str
    side: str           # 'BUY' or 'SELL'
    liquidation_type: str  # 'LONG_LIQUIDATION' or 'SHORT_LIQUIDATION'
    price: float
    quantity: float
    value: float
    strength: float     # 0-1


@dataclass
class LiquidationCluster:
    """Group of liquidations at similar price level"""
    price: float
    total_value: float
    event_count: int
    strength: float      # 0-1
    dominant_type: str   # 'LONG' or 'SHORT'
    timestamp: datetime


class LiquidationIntelligence:
    """
    Analyzes liquidation data to detect:
    - Long/short liquidation imbalances
    - Liquidation clusters (key levels)
    - Cascade risk
    - Liquidation-driven momentum
    """
    
    def __init__(self, phase5_fetcher=None):
        self.phase5_fetcher = phase5_fetcher
        self.liquidation_history: List[LiquidationEvent] = []
        
        # Load settings from config
        self.value_threshold = getattr(Config, 'SM_LIQUIDATION_VALUE_THRESHOLD', 100000)
        self.lookback_minutes = getattr(Config, 'SM_LIQUIDATION_LOOKBACK_MINUTES', 60)
        self.cluster_percent = getattr(Config, 'SM_LIQUIDATION_CLUSTER_PERCENT', 0.01)
        self.cascade_risk_threshold = getattr(Config, 'SM_CASCADE_RISK_THRESHOLD', 0.5)
        
        log.info("LiquidationIntelligence initialized")
    
    def set_fetcher(self, phase5_fetcher):
        """Set Phase5DataFetcher instance"""
        self.phase5_fetcher = phase5_fetcher
    
    def analyze(self, symbol: str, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze liquidation data for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            df: Optional OHLCV DataFrame for context
        
        Returns:
            Dictionary with liquidation analysis results
        """
        if not self.phase5_fetcher:
            log.debug("Phase5DataFetcher not set, returning default")
            return self._default_result()
        
        try:
            # Fetch recent liquidations
            liq_summary = self.phase5_fetcher.get_liquidation_summary(
                symbol, 
                minutes=self.lookback_minutes
            )
            
            # Get detailed liquidation events
            liq_events = self.phase5_fetcher.fetch_liquidations(symbol, limit=100)
            
            if not liq_events:
                return self._default_result()
            
            # ===== 1. PROCESS LIQUIDATION EVENTS =====
            events = self._process_events(liq_events, symbol)
            
            # ===== 2. ANALYZE FLOW =====
            flow_analysis = self._analyze_flow(events, liq_summary)
            
            # ===== 3. DETECT CLUSTERS =====
            clusters = self._detect_clusters(events, df)
            
            # ===== 4. CALCULATE CASCADE RISK =====
            cascade_risk = self._calculate_cascade_risk(events, clusters)
            
            # ===== 5. CALCULATE SCORE =====
            score = self._calculate_score(flow_analysis, cascade_risk, clusters)
            
            # ===== 6. DETERMINE DIRECTIONAL BIAS =====
            bias = self._determine_bias(flow_analysis, clusters)
            
            # ===== 7. BUILD REASONS =====
            reasons = self._build_reasons(flow_analysis, clusters, cascade_risk)
            
            # Store for history
            self.liquidation_history.extend(events)
            self._clean_history()
            
            return {
                'score': score,
                'bias': bias,
                'confidence': min(1.0, score * 1.2),
                'short_liq_value': flow_analysis.get('short_liq_value', 0),
                'long_liq_value': flow_analysis.get('long_liq_value', 0),
                'total_liq_value': flow_analysis.get('total_liq_value', 0),
                'dominant': flow_analysis.get('dominant', 'NONE'),
                'cascade_risk': cascade_risk,
                'clusters': clusters[:3],  # Top 3 clusters
                'reasons': reasons,
                'details': {
                    'events': len(events),
                    'flow_analysis': flow_analysis,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            log.error(f"Error in liquidation analysis for {symbol}: {e}")
            return self._default_result()
    
    def _process_events(self, liq_events: List[Dict], symbol: str) -> List[LiquidationEvent]:
        """Process raw liquidation events into LiquidationEvent objects"""
        events = []
        
        for event in liq_events:
            try:
                # Extract data
                timestamp = event.get('time')
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now()
                
                side = event.get('side', 'UNKNOWN')
                liquidation_type = event.get('liquidation_type', 'UNKNOWN')
                price = float(event.get('price', 0))
                quantity = float(event.get('quantity', 0))
                value = float(event.get('value', 0))
                
                # Calculate strength based on value
                if value > self.value_threshold * 2:
                    strength = 0.9
                elif value > self.value_threshold:
                    strength = 0.7
                elif value > self.value_threshold * 0.5:
                    strength = 0.5
                else:
                    strength = 0.3
                
                events.append(LiquidationEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side,
                    liquidation_type=liquidation_type,
                    price=price,
                    quantity=quantity,
                    value=value,
                    strength=strength
                ))
            except Exception as e:
                log.debug(f"Error processing liquidation event: {e}")
                continue
        
        return events
    
    def _analyze_flow(self, events: List[LiquidationEvent], 
                      summary: Dict) -> Dict[str, Any]:
        """Analyze liquidation flow (long vs short)"""
        # Use summary from Phase5DataFetcher first
        if summary:
            long_value = summary.get('long_liq_value', 0)
            short_value = summary.get('short_liq_value', 0)
            long_count = summary.get('total_long_liq', 0)
            short_count = summary.get('total_short_liq', 0)
        else:
            # Fallback to calculate from events
            long_value = sum(e.value for e in events if e.liquidation_type == 'LONG_LIQUIDATION')
            short_value = sum(e.value for e in events if e.liquidation_type == 'SHORT_LIQUIDATION')
            long_count = sum(1 for e in events if e.liquidation_type == 'LONG_LIQUIDATION')
            short_count = sum(1 for e in events if e.liquidation_type == 'SHORT_LIQUIDATION')
        
        total = long_value + short_value
        
        if total == 0:
            dominant = 'NONE'
            imbalance = 0
        elif long_value > short_value:
            dominant = 'LONG_LIQUIDATIONS'
            imbalance = (long_value - short_value) / total
        else:
            dominant = 'SHORT_LIQUIDATIONS'
            imbalance = (short_value - long_value) / total
        
        return {
            'long_liq_value': round(long_value, 2),
            'short_liq_value': round(short_value, 2),
            'total_liq_value': round(total, 2),
            'long_count': long_count,
            'short_count': short_count,
            'dominant': dominant,
            'imbalance': round(imbalance, 3),
            'is_bullish': dominant == 'SHORT_LIQUIDATIONS'  # Shorts liquidated = bullish
        }
    
    def _detect_clusters(self, events: List[LiquidationEvent], 
                         df: pd.DataFrame = None) -> List[Dict]:
        """Detect liquidation clusters at price levels"""
        if not events:
            return []
        
        # Group by price level
        price_groups = {}
        
        for event in events:
            if event.price <= 0:
                continue
            
            # Round price to nearest significant level
            rounded_price = round(event.price, -int(np.floor(np.log10(event.price))) + 2)
            
            if rounded_price not in price_groups:
                price_groups[rounded_price] = {
                    'total_value': 0,
                    'count': 0,
                    'long_value': 0,
                    'short_value': 0
                }
            
            price_groups[rounded_price]['total_value'] += event.value
            price_groups[rounded_price]['count'] += 1
            
            if event.liquidation_type == 'LONG_LIQUIDATION':
                price_groups[rounded_price]['long_value'] += event.value
            else:
                price_groups[rounded_price]['short_value'] += event.value
        
        # Convert to clusters and sort by value
        clusters = []
        current_price = 0
        
        # Get current price for context
        if df is not None and not df.empty:
            current_price = float(df['close'].iloc[-1])
        
        for price, data in price_groups.items():
            # Calculate strength (normalized by max value)
            strength = min(1.0, data['total_value'] / (self.value_threshold * 2))
            
            # Determine dominant type
            if data['long_value'] > data['short_value']:
                dominant_type = 'LONG'
                is_bullish = False  # Long liquidations are bearish
            else:
                dominant_type = 'SHORT'
                is_bullish = True   # Short liquidations are bullish
            
            clusters.append({
                'price': price,
                'total_value': round(data['total_value'], 2),
                'event_count': data['count'],
                'strength': round(strength, 3),
                'dominant_type': dominant_type,
                'is_bullish': is_bullish,
                'distance_from_current': abs(price - current_price) / current_price if current_price > 0 else 0
            })
        
        # Sort by total value (largest first)
        clusters.sort(key=lambda x: x['total_value'], reverse=True)
        
        return clusters
    
    def _calculate_cascade_risk(self, events: List[LiquidationEvent], 
                                clusters: List[Dict]) -> float:
        """
        Calculate risk of liquidation cascade
        Higher risk when multiple clusters are close together
        """
        if len(clusters) < 2:
            return 0.0
        
        # Get prices of top clusters
        top_prices = [c['price'] for c in clusters[:3]]
        
        if len(top_prices) < 2:
            return 0.0
        
        # Calculate spread between clusters
        price_spread = max(top_prices) - min(top_prices)
        avg_price = np.mean(top_prices)
        spread_pct = price_spread / avg_price if avg_price > 0 else 1
        
        # Tight clusters = higher cascade risk
        if spread_pct < 0.01:      # Within 1%
            cascade_risk = 0.8
        elif spread_pct < 0.02:    # Within 2%
            cascade_risk = 0.6
        elif spread_pct < 0.03:    # Within 3%
            cascade_risk = 0.4
        elif spread_pct < 0.05:    # Within 5%
            cascade_risk = 0.2
        else:
            cascade_risk = 0.0
        
        # Adjust by total liquidation value
        total_value = sum(c['total_value'] for c in clusters[:3])
        value_factor = min(1.0, total_value / (self.value_threshold * 5))
        
        return min(1.0, cascade_risk * (0.5 + value_factor * 0.5))
    
    def _calculate_score(self, flow: Dict, cascade_risk: float, 
                         clusters: List[Dict]) -> float:
        """Calculate overall liquidation score (0-1)"""
        score = 0.5  # Neutral base
        
        # Factor 1: Flow imbalance
        if flow.get('dominant') == 'SHORT_LIQUIDATIONS':
            # Shorts being squeezed = bullish
            score += flow.get('imbalance', 0) * 0.3
        elif flow.get('dominant') == 'LONG_LIQUIDATIONS':
            # Longs being squeezed = bearish
            score -= flow.get('imbalance', 0) * 0.3
        
        # Factor 2: Total liquidation value
        total_value = flow.get('total_liq_value', 0)
        if total_value > self.value_threshold * 5:
            score += 0.2
        elif total_value > self.value_threshold * 2:
            score += 0.1
        
        # Factor 3: Cascade risk (high cascade = momentum)
        if cascade_risk > 0.6:
            if flow.get('is_bullish'):
                score += 0.15
            else:
                score -= 0.15
        elif cascade_risk > 0.3:
            if flow.get('is_bullish'):
                score += 0.08
            else:
                score -= 0.08
        
        # Factor 4: Strong clusters near current price
        for cluster in clusters[:2]:
            if cluster.get('distance_from_current', 1) < 0.01:
                if cluster.get('is_bullish'):
                    score += 0.1
                else:
                    score -= 0.1
        
        # Clamp to 0-1
        return max(0.01, min(0.99, score))
    
    def _determine_bias(self, flow: Dict, clusters: List[Dict]) -> str:
        """Determine directional bias from liquidation data"""
        # If significant short liquidations (bullish)
        if flow.get('dominant') == 'SHORT_LIQUIDATIONS':
            if flow.get('imbalance', 0) > 0.3:
                return 'BULLISH'
            else:
                return 'SLIGHTLY_BULLISH'
        
        # If significant long liquidations (bearish)
        elif flow.get('dominant') == 'LONG_LIQUIDATIONS':
            if flow.get('imbalance', 0) > 0.3:
                return 'BEARISH'
            else:
                return 'SLIGHTLY_BEARISH'
        
        # Check clusters for nearby liquidations
        for cluster in clusters[:2]:
            if cluster.get('distance_from_current', 1) < 0.01:
                if cluster.get('is_bullish'):
                    return 'BULLISH'
                else:
                    return 'BEARISH'
        
        return 'NEUTRAL'
    
    def _build_reasons(self, flow: Dict, clusters: List[Dict], 
                       cascade_risk: float) -> List[str]:
        """Build human-readable reasons for the analysis"""
        reasons = []
        
        # Flow reasons
        total_value = flow.get('total_liq_value', 0)
        if total_value > 0:
            reasons.append(f"Total liquidations: ${total_value:,.0f}")
        
        if flow.get('dominant') == 'SHORT_LIQUIDATIONS':
            short_value = flow.get('short_liq_value', 0)
            reasons.append(f"${short_value:,.0f} short liquidations (bullish)")
        elif flow.get('dominant') == 'LONG_LIQUIDATIONS':
            long_value = flow.get('long_liq_value', 0)
            reasons.append(f"${long_value:,.0f} long liquidations (bearish)")
        
        # Cluster reasons
        if clusters:
            top_cluster = clusters[0]
            if top_cluster.get('strength', 0) > 0.6:
                reasons.append(
                    f"Major liquidation cluster at ${top_cluster['price']:,.0f} "
                    f"(${top_cluster['total_value']:,.0f})"
                )
        
        # Cascade risk
        if cascade_risk > 0.5:
            reasons.append(f"High cascade risk ({cascade_risk:.0%}) - potential for chain reaction")
        
        if not reasons:
            reasons.append("No significant liquidation activity detected")
        
        return reasons[:5]  # Top 5 reasons
    
    def _clean_history(self):
        """Clean old liquidation history"""
        cutoff = datetime.now() - timedelta(hours=24)
        self.liquidation_history = [
            e for e in self.liquidation_history 
            if e.timestamp > cutoff
        ]
    
    def _default_result(self) -> Dict[str, Any]:
        """Default result when no liquidation data available"""
        return {
            'score': 0.5,
            'bias': 'NEUTRAL',
            'confidence': 0.5,
            'short_liq_value': 0,
            'long_liq_value': 0,
            'total_liq_value': 0,
            'dominant': 'NONE',
            'cascade_risk': 0.0,
            'clusters': [],
            'reasons': ['No liquidation data available'],
            'details': {'error': 'No data'}
        }
    
    def get_liquidation_summary(self, symbol: str) -> Dict[str, Any]:
        """Quick summary of liquidation activity"""
        result = self.analyze(symbol)
        return {
            'symbol': symbol,
            'bias': result['bias'],
            'total_liq': result['total_liq_value'],
            'short_liq': result['short_liq_value'],
            'long_liq': result['long_liq_value'],
            'cascade_risk': result['cascade_risk'],
            'timestamp': datetime.now().isoformat()
        }