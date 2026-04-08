# market_maker_model.py - Institutional Market Maker Intelligence
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

class MarketMakerActivity(Enum):
    AGGRESSIVE_BUYING = "AGGRESSIVE_BUYING"
    PASSIVE_BUYING = "PASSIVE_BUYING"
    NEUTRAL = "NEUTRAL"
    PASSIVE_SELLING = "PASSIVE_SELLING"
    AGGRESSIVE_SELLING = "AGGRESSIVE_SELLING"

class SmartMoneyFootprint(Enum):
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    MARKUP = "MARKUP"
    MARKDOWN = "MARKDOWN"
    UNKNOWN = "UNKNOWN"

@dataclass
class MarketMakerPosition:
    """Market maker positioning data"""
    timestamp: datetime
    bid_volume: float
    ask_volume: float
    net_position: float  # Positive = long bias, Negative = short bias
    inventory_imbalance: float  # -1 to 1
    activity_level: float  # 0-1
    activity_type: MarketMakerActivity
    confidence: float  # 0-1

@dataclass
class SmartMoneyFlow:
    """Smart money flow detection"""
    footprint: SmartMoneyFootprint
    accumulation_price: Optional[float]
    distribution_price: Optional[float]
    volume_profile: Dict[str, float]
    order_block_strength: float
    liquidity_gradient: float
    institutional_bias: float  # -1 to 1

@dataclass
class OptionsFlowProxy:
    """Options flow proxy data"""
    put_call_ratio: float
    implied_volatility: float
    gamma_exposure: float
    dealer_hedge_flow: float  # -1 to 1
    max_pain_price: Optional[float]
    large_orders_detected: int
    unusual_activity_score: float  # 0-1

class MarketMakerModel:
    """
    Institutional Market Maker Model
    Detects market maker positioning and smart money footprint
    """
    
    def __init__(self):
        self.position_history: List[MarketMakerPosition] = []
        self.smart_money_history: List[SmartMoneyFlow] = []
        self.options_proxy_history: List[OptionsFlowProxy] = []
        
    def analyze_market_maker_activity(self, df: pd.DataFrame, order_book: Optional[Dict] = None) -> MarketMakerPosition:
        """
        Analyze market maker activity from order book and trade data
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return self._default_position()
            
            # Calculate bid-ask metrics
            if order_book:
                bids = order_book.get('bids', [])
                asks = order_book.get('asks', [])
                
                bid_volume = sum(b[1] for b in bids[:10]) if bids else 0
                ask_volume = sum(a[1] for a in asks[:10]) if asks else 0
                
                # Calculate net position (market maker inventory)
                net_position = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
                
                # Calculate inventory imbalance
                total_volume = bid_volume + ask_volume
                inventory_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
                
                # Detect activity level (how aggressive they are)
                spread = (asks[0][0] - bids[0][0]) / ((asks[0][0] + bids[0][0]) / 2) if asks and bids else 0.001
                activity_level = 1.0 - min(1.0, spread * 100)  # Tight spread = high activity
                
                # Determine activity type
                activity_type = self._determine_activity_type(net_position, inventory_imbalance, order_book)
                
                # Calculate confidence
                depth_consistency = self._calculate_depth_consistency(bids, asks)
                confidence = min(1.0, depth_consistency * 0.7 + abs(inventory_imbalance) * 0.3)
                
                position = MarketMakerPosition(
                    timestamp=datetime.now(),
                    bid_volume=bid_volume,
                    ask_volume=ask_volume,
                    net_position=net_position,
                    inventory_imbalance=inventory_imbalance,
                    activity_level=activity_level,
                    activity_type=activity_type,
                    confidence=confidence
                )
                
                self.position_history.append(position)
                return position
            
            return self._default_position()
            
        except Exception as e:
            logging.error(f"Error in analyze_market_maker_activity: {e}")
            return self._default_position()
    
    def _determine_activity_type(self, net_position: float, imbalance: float, 
                                order_book: Dict) -> MarketMakerActivity:
        """Determine market maker activity type"""
        # Look for large orders
        large_bids = order_book.get('large_bids', 0)
        large_asks = order_book.get('large_asks', 0)
        
        if net_position > 0.3 and large_bids > large_asks:
            return MarketMakerActivity.AGGRESSIVE_BUYING
        elif net_position > 0.1:
            return MarketMakerActivity.PASSIVE_BUYING
        elif net_position < -0.3 and large_asks > large_bids:
            return MarketMakerActivity.AGGRESSIVE_SELLING
        elif net_position < -0.1:
            return MarketMakerActivity.PASSIVE_SELLING
        else:
            return MarketMakerActivity.NEUTRAL
    
    def _calculate_depth_consistency(self, bids: List, asks: List) -> float:
        """Calculate consistency of order book depth"""
        if len(bids) < 5 or len(asks) < 5:
            return 0.5
        
        bid_sizes = [b[1] for b in bids[:5]]
        ask_sizes = [a[1] for a in asks[:5]]
        
        bid_std = np.std(bid_sizes) / (np.mean(bid_sizes) + 1e-8)
        ask_std = np.std(ask_sizes) / (np.mean(ask_sizes) + 1e-8)
        
        consistency = 1.0 - min(1.0, (bid_std + ask_std) / 2)
        return consistency
    
    def _default_position(self) -> MarketMakerPosition:
        """Default position when analysis fails"""
        return MarketMakerPosition(
            timestamp=datetime.now(),
            bid_volume=0,
            ask_volume=0,
            net_position=0,
            inventory_imbalance=0,
            activity_level=0,
            activity_type=MarketMakerActivity.NEUTRAL,
            confidence=0
        )
    
    def detect_smart_money_footprint(self, df: pd.DataFrame, microstructure: Dict) -> SmartMoneyFlow:
        """
        Detect smart money footprint using multiple indicators
        """
        try:
            if df is None or df.empty or len(df) < 50:
                return self._default_smart_money()
            
            # Analyze volume profile for accumulation/distribution
            volume_profile = self._analyze_volume_profile(df)
            
            # Check for order blocks
            order_blocks = microstructure.get('order_blocks', [])
            order_block_strength = self._calculate_order_block_strength(order_blocks)
            
            # Calculate liquidity gradient
            liquidity_gradient = self._calculate_liquidity_gradient(df)
            
            # Determine footprint
            footprint = self._determine_footprint(df, volume_profile, order_block_strength)
            
            # Find accumulation/distribution prices
            accumulation_price = self._find_accumulation_price(df, volume_profile)
            distribution_price = self._find_distribution_price(df, volume_profile)
            
            # Calculate institutional bias
            institutional_bias = self._calculate_institutional_bias(df, microstructure)
            
            return SmartMoneyFlow(
                footprint=footprint,
                accumulation_price=accumulation_price,
                distribution_price=distribution_price,
                volume_profile=volume_profile,
                order_block_strength=order_block_strength,
                liquidity_gradient=liquidity_gradient,
                institutional_bias=institutional_bias
            )
            
        except Exception as e:
            logging.error(f"Error in detect_smart_money_footprint: {e}")
            return self._default_smart_money()
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume profile for accumulation/distribution patterns"""
        try:
            if len(df) < 50:
                return {}
            
            # Calculate volume at price levels
            price = df['close'].values
            volume = df['volume'].values
            
            # Create price bins
            price_min, price_max = price.min(), price.max()
            bins = np.linspace(price_min, price_max, 20)
            
            # Sum volume in each bin
            volume_profile = {}
            for i in range(len(bins) - 1):
                mask = (price >= bins[i]) & (price < bins[i+1])
                if mask.any():
                    level_price = (bins[i] + bins[i+1]) / 2
                    volume_profile[level_price] = volume[mask].sum()
            
            return volume_profile
            
        except Exception:
            return {}
    
    def _calculate_order_block_strength(self, order_blocks: List) -> float:
        """Calculate overall order block strength"""
        if not order_blocks:
            return 0.0
        
        strengths = [ob.get('strength', 0.5) for ob in order_blocks[-5:]]
        return np.mean(strengths) if strengths else 0.0
    
    def _calculate_liquidity_gradient(self, df: pd.DataFrame) -> float:
        """Calculate liquidity gradient (-1 to 1)"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.0
            
            recent_vol = df['volume'].iloc[-5:].mean()
            older_vol = df['volume'].iloc[-20:-5].mean()
            
            if older_vol > 0:
                gradient = (recent_vol - older_vol) / older_vol
                return max(-1.0, min(1.0, gradient))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _determine_footprint(self, df: pd.DataFrame, volume_profile: Dict,
                           ob_strength: float) -> SmartMoneyFootprint:
        """Determine smart money footprint type"""
        try:
            if len(df) < 50:
                return SmartMoneyFootprint.UNKNOWN
            
            # Get recent price action
            recent_price = df['close'].iloc[-1]
            price_20_ago = df['close'].iloc[-20]
            price_50_ago = df['close'].iloc[-50] if len(df) >= 50 else recent_price
            
            # Volume analysis
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].iloc[-20:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Price trends
            short_trend = (recent_price - price_20_ago) / price_20_ago
            long_trend = (recent_price - price_50_ago) / price_50_ago
            
            # Accumulation: price range-bound, volume increasing
            if abs(short_trend) < 0.02 and volume_ratio > 1.2 and ob_strength > 0.6:
                return SmartMoneyFootprint.ACCUMULATION
            
            # Distribution: price range-bound, volume high, weakening
            if abs(short_trend) < 0.02 and volume_ratio > 1.5 and ob_strength < 0.4:
                return SmartMoneyFootprint.DISTRIBUTION
            
            # Markup: price rising, volume supporting
            if short_trend > 0.03 and volume_ratio > 1.1:
                return SmartMoneyFootprint.MARKUP
            
            # Markdown: price falling, volume supporting
            if short_trend < -0.03 and volume_ratio > 1.1:
                return SmartMoneyFootprint.MARKDOWN
            
            return SmartMoneyFootprint.UNKNOWN
            
        except Exception:
            return SmartMoneyFootprint.UNKNOWN
    
    def _find_accumulation_price(self, df: pd.DataFrame, volume_profile: Dict) -> Optional[float]:
        """Find potential accumulation price level"""
        try:
            if not volume_profile:
                return None
            
            # Find price level with highest volume (potential accumulation)
            max_volume_price = max(volume_profile, key=volume_profile.get)
            
            # Check if price is near this level
            current_price = df['close'].iloc[-1]
            if abs(current_price - max_volume_price) / max_volume_price < 0.02:
                return max_volume_price
            
            return None
            
        except Exception:
            return None
    
    def _find_distribution_price(self, df: pd.DataFrame, volume_profile: Dict) -> Optional[float]:
        """Find potential distribution price level"""
        try:
            if not volume_profile:
                return None
            
            # Find price level with high volume near recent highs
            recent_high = df['high'].iloc[-20:].max()
            candidates = [p for p in volume_profile.keys() if abs(p - recent_high) / recent_high < 0.02]
            
            if candidates:
                return max(candidates, key=lambda x: volume_profile.get(x, 0))
            
            return None
            
        except Exception:
            return None
    
    def _calculate_institutional_bias(self, df: pd.DataFrame, microstructure: Dict) -> float:
        """Calculate institutional bias (-1 to 1)"""
        try:
            bias = 0.0
            weight_sum = 0
            
            # Order block contribution
            obs = microstructure.get('order_blocks', [])
            if obs:
                ob_bias = 0
                for ob in obs[-3:]:
                    if ob.get('type') == 'BULLISH_OB':
                        ob_bias += ob.get('strength', 0.5)
                    elif ob.get('type') == 'BEARISH_OB':
                        ob_bias -= ob.get('strength', 0.5)
                
                bias += ob_bias * 0.4
                weight_sum += 0.4
            
            # BOS contribution
            bos = microstructure.get('bos', [])
            if bos:
                bos_bias = 0
                for b in bos[-3:]:
                    if b.get('type') == 'BULLISH_BOS':
                        bos_bias += b.get('strength', 0.5)
                    elif b.get('type') == 'BEARISH_BOS':
                        bos_bias -= b.get('strength', 0.5)
                
                bias += bos_bias * 0.3
                weight_sum += 0.3
            
            # Volume profile contribution
            if 'volume_ratio' in df.columns:
                vol_ratio = df['volume_ratio'].iloc[-1]
                if vol_ratio > 1.5:
                    bias += 0.2 * (1 if df['close'].iloc[-1] > df['open'].iloc[-1] else -1)
                    weight_sum += 0.2
            
            return bias / weight_sum if weight_sum > 0 else 0
            
        except Exception:
            return 0
    
    def _default_smart_money(self) -> SmartMoneyFlow:
        """Default smart money flow when analysis fails"""
        return SmartMoneyFlow(
            footprint=SmartMoneyFootprint.UNKNOWN,
            accumulation_price=None,
            distribution_price=None,
            volume_profile={},
            order_block_strength=0,
            liquidity_gradient=0,
            institutional_bias=0
        )
    
    def proxy_options_flow(self, df: pd.DataFrame, option_data: Optional[Dict] = None) -> OptionsFlowProxy:
        """
        Proxy options flow analysis (when real options data not available)
        Uses price action and volatility as proxy
        """
        try:
            if df is None or df.empty or len(df) < 30:
                return self._default_options_proxy()
            
            # Calculate put-call proxy using price action
            put_call_ratio = self._calculate_put_call_proxy(df)
            
            # Calculate implied volatility proxy
            implied_vol = self._calculate_implied_volatility_proxy(df)
            
            # Calculate gamma exposure proxy
            gamma_exposure = self._calculate_gamma_exposure_proxy(df)
            
            # Calculate dealer hedge flow
            dealer_flow = self._calculate_dealer_flow_proxy(df)
            
            # Find max pain proxy (price level with most volume)
            max_pain = self._find_max_pain_proxy(df)
            
            # Detect large/unusual orders
            large_orders, unusual_score = self._detect_unusual_activity(df)
            
            return OptionsFlowProxy(
                put_call_ratio=put_call_ratio,
                implied_volatility=implied_vol,
                gamma_exposure=gamma_exposure,
                dealer_hedge_flow=dealer_flow,
                max_pain_price=max_pain,
                large_orders_detected=large_orders,
                unusual_activity_score=unusual_score
            )
            
        except Exception as e:
            logging.error(f"Error in proxy_options_flow: {e}")
            return self._default_options_proxy()
    
    def _calculate_put_call_proxy(self, df: pd.DataFrame) -> float:
        """Calculate put-call ratio proxy using volume and price action"""
        try:
            if len(df) < 20:
                return 1.0
            
            # Down volume / Up volume as proxy for put/call
            up_volume = df[df['close'] > df['open']]['volume'].sum()
            down_volume = df[df['close'] < df['open']]['volume'].sum()
            
            if up_volume > 0:
                return down_volume / up_volume
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_implied_volatility_proxy(self, df: pd.DataFrame) -> float:
        """Calculate implied volatility proxy using historical volatility and skew"""
        try:
            if len(df) < 30:
                return 0.3
            
            # Historical volatility
            returns = df['close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)
            
            # Volatility skew proxy (using recent price moves)
            recent_returns = returns.iloc[-10:]
            skew = recent_returns.skew() if len(recent_returns) > 3 else 0
            
            # Implied vol typically higher than historical
            implied_vol = hist_vol * (1 + abs(skew) * 0.2)
            
            return min(2.0, max(0.1, implied_vol))
            
        except Exception:
            return 0.3
    
    def _calculate_gamma_exposure_proxy(self, df: pd.DataFrame) -> float:
        """Calculate gamma exposure proxy using price acceleration"""
        try:
            if len(df) < 20:
                return 0
            
            # Price acceleration as gamma proxy
            prices = df['close'].values[-20:]
            velocity = np.diff(prices)
            acceleration = np.diff(velocity)
            
            if len(acceleration) > 0:
                gamma_proxy = np.mean(acceleration) / np.mean(prices) * 100
                return max(-1.0, min(1.0, gamma_proxy))
            
            return 0
            
        except Exception:
            return 0
    
    def _calculate_dealer_flow_proxy(self, df: pd.DataFrame) -> float:
        """Calculate dealer hedge flow proxy using order flow"""
        try:
            if len(df) < 20:
                return 0
            
            # Use bid-ask volume imbalance as proxy
            up_volume = df[df['close'] > df['open']]['volume'].sum()
            down_volume = df[df['close'] < df['open']]['volume'].sum()
            total = up_volume + down_volume
            
            if total > 0:
                flow = (up_volume - down_volume) / total
                return max(-1.0, min(1.0, flow))
            
            return 0
            
        except Exception:
            return 0
    
    def _find_max_pain_proxy(self, df: pd.DataFrame) -> Optional[float]:
        """Find max pain proxy (price level with highest volume)"""
        try:
            if len(df) < 50:
                return None
            
            # Use volume profile to find level with most volume
            volume_profile = self._analyze_volume_profile(df)
            if volume_profile:
                return max(volume_profile, key=volume_profile.get)
            
            return None
            
        except Exception:
            return None
    
    def _detect_unusual_activity(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Detect unusual trading activity"""
        try:
            if len(df) < 50:
                return 0, 0
            
            large_orders = 0
            unusual_score = 0
            
            # Check for volume spikes
            volume = df['volume'].values
            volume_ma = pd.Series(volume).rolling(20).mean().values
            
            for i in range(-10, 0):
                if len(volume) > abs(i) and len(volume_ma) > abs(i):
                    if volume[i] > volume_ma[i] * 3:
                        large_orders += 1
                        unusual_score += 0.1
            
            # Check for price gaps
            for i in range(-5, 0):
                if len(df) > abs(i) + 1:
                    gap = abs(df['open'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
                    if gap > 0.02:  # 2% gap
                        unusual_score += 0.15
            
            unusual_score = min(1.0, unusual_score)
            
            return large_orders, unusual_score
            
        except Exception:
            return 0, 0
    
    def _default_options_proxy(self) -> OptionsFlowProxy:
        """Default options proxy when analysis fails"""
        return OptionsFlowProxy(
            put_call_ratio=1.0,
            implied_volatility=0.3,
            gamma_exposure=0,
            dealer_hedge_flow=0,
            max_pain_price=None,
            large_orders_detected=0,
            unusual_activity_score=0
        )