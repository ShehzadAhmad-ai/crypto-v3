"""
Risk Filters for Strategy Expert
Applies risk management filters to combined trades (Steps 6-7 of pipeline)

Step 6: Risk Filter
    if risk_reward < MIN_RISK_REWARD → SKIP

Step 7: Late Entry Filter
    if |current_price - entry| > ATR × multiplier → SKIP
    "Don't chase the trade"
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from strategy_expert.trade_combiner import CombinedTrade
from strategy_expert.strategy_config import get_pipeline_config, PipelineConfig


class FilterResult(Enum):
    """Result of applying risk filters"""
    PASS = "PASS"
    FAIL_RISK_REWARD = "FAIL_RISK_REWARD"
    FAIL_LATE_ENTRY = "FAIL_LATE_ENTRY"
    FAIL_MAX_SL = "FAIL_MAX_SL"
    FAIL_MAX_SPREAD = "FAIL_MAX_SPREAD"
    FAIL_VOLATILITY = "FAIL_VOLATILITY"
    FAIL_CONSOLIDATION = "FAIL_CONSOLIDATION"


@dataclass
class FilterStats:
    """Statistics from filter application"""
    result: FilterResult
    reason: str
    risk_reward: float = 0.0
    price_distance: float = 0.0
    price_distance_pct: float = 0.0
    atr: float = 0.0
    atr_multiplier: float = 1.0
    max_allowed_distance: float = 0.0
    current_price: float = 0.0
    entry_price: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'result': self.result.value,
            'reason': self.reason,
            'risk_reward': round(self.risk_reward, 2),
            'price_distance': round(self.price_distance, 4),
            'price_distance_pct': round(self.price_distance_pct, 2),
            'atr': round(self.atr, 4),
            'max_allowed_distance': round(self.max_allowed_distance, 4),
            'current_price': round(self.current_price, 4),
            'entry_price': round(self.entry_price, 4)
        }


class RiskFilter:
    """
    Risk management filter for trades
    
    Features:
    - Minimum risk/reward check
    - Late entry protection (don't chase)
    - Maximum stop loss distance
    - Maximum spread check
    - Volatility-based filters
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize Risk Filter
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or get_pipeline_config()
        self.min_risk_reward = self.config.min_risk_reward
        self.late_entry_multiplier = self.config.late_entry_atr_multiplier
        self.max_sl_distance_pct = 0.05  # Maximum 5% stop loss distance
        self.max_spread_pct = 0.002  # Maximum 0.2% spread
        self.min_volume_ratio = 0.5  # Minimum volume ratio (avoid illiquid)
    
    def apply_all_filters(self, combined_trade: CombinedTrade,
                         current_price: float,
                         indicators: Dict,
                         df: pd.DataFrame = None) -> Tuple[FilterResult, FilterStats]:
        """
        Apply all risk filters to a combined trade
        
        Args:
            combined_trade: Combined trade from TradeCombiner
            current_price: Current market price
            indicators: Technical indicators (contains ATR, volume_ratio, etc.)
            df: Optional DataFrame for additional checks
        
        Returns:
            Tuple of (FilterResult, FilterStats)
        """
        # Filter 1: Risk/Reward Check
        result, stats = self.check_risk_reward(combined_trade)
        if result != FilterResult.PASS:
            return result, stats
        
        # Filter 2: Late Entry Check
        result, stats = self.check_late_entry(combined_trade, current_price, indicators)
        if result != FilterResult.PASS:
            return result, stats
        
        # Filter 3: Maximum Stop Loss Distance
        result, stats = self.check_max_sl_distance(combined_trade)
        if result != FilterResult.PASS:
            return result, stats
        
        # Filter 4: Spread Check (if we have bid/ask)
        result, stats = self.check_spread(combined_trade, indicators)
        if result != FilterResult.PASS:
            return result, stats
        
        # Filter 5: Volume Check
        result, stats = self.check_volume(indicators)
        if result != FilterResult.PASS:
            return result, stats
        
        # Filter 6: Volatility Check
        result, stats = self.check_volatility(combined_trade, indicators)
        if result != FilterResult.PASS:
            return result, stats
        
        # All filters passed
        return FilterResult.PASS, FilterStats(
            result=FilterResult.PASS,
            reason="All risk filters passed",
            risk_reward=combined_trade.risk_reward
        )
    
    def check_risk_reward(self, combined_trade: CombinedTrade) -> Tuple[FilterResult, FilterStats]:
        """
        Check if trade meets minimum risk/reward ratio
        
        Args:
            combined_trade: Combined trade object
        
        Returns:
            Tuple of (FilterResult, FilterStats)
        """
        rr = combined_trade.risk_reward
        
        if rr >= self.min_risk_reward:
            return FilterResult.PASS, FilterStats(
                result=FilterResult.PASS,
                reason=f"Risk/Reward OK: {rr:.2f} >= {self.min_risk_reward}",
                risk_reward=rr
            )
        else:
            return FilterResult.FAIL_RISK_REWARD, FilterStats(
                result=FilterResult.FAIL_RISK_REWARD,
                reason=f"Risk/Reward too low: {rr:.2f} < {self.min_risk_reward}",
                risk_reward=rr
            )
    
    def check_late_entry(self, combined_trade: CombinedTrade,
                        current_price: float,
                        indicators: Dict) -> Tuple[FilterResult, FilterStats]:
        """
        Check if price hasn't moved too far from entry (don't chase)
        
        Args:
            combined_trade: Combined trade object
            current_price: Current market price
            indicators: Technical indicators (contains ATR)
        
        Returns:
            Tuple of (FilterResult, FilterStats)
        """
        entry = combined_trade.entry
        atr = indicators.get('atr', 0)
        
        if atr <= 0:
            # If ATR is zero, use percentage-based check
            distance_pct = abs(current_price - entry) / entry
            max_distance_pct = 0.01  # 1% max chase
            
            if distance_pct <= max_distance_pct:
                return FilterResult.PASS, FilterStats(
                    result=FilterResult.PASS,
                    reason=f"Late entry OK: {distance_pct:.2%} <= {max_distance_pct:.2%}",
                    price_distance=abs(current_price - entry),
                    price_distance_pct=distance_pct * 100,
                    current_price=current_price,
                    entry_price=entry
                )
            else:
                return FilterResult.FAIL_LATE_ENTRY, FilterStats(
                    result=FilterResult.FAIL_LATE_ENTRY,
                    reason=f"Price moved too far: {distance_pct:.2%} > {max_distance_pct:.2%}",
                    price_distance=abs(current_price - entry),
                    price_distance_pct=distance_pct * 100,
                    current_price=current_price,
                    entry_price=entry
                )
        
        # ATR-based check
        max_allowed_distance = atr * self.late_entry_multiplier
        actual_distance = abs(current_price - entry)
        distance_pct = (actual_distance / entry) * 100
        
        if actual_distance <= max_allowed_distance:
            return FilterResult.PASS, FilterStats(
                result=FilterResult.PASS,
                reason=f"Late entry OK: {actual_distance:.4f} <= {max_allowed_distance:.4f}",
                price_distance=actual_distance,
                price_distance_pct=distance_pct,
                atr=atr,
                atr_multiplier=self.late_entry_multiplier,
                max_allowed_distance=max_allowed_distance,
                current_price=current_price,
                entry_price=entry
            )
        else:
            return FilterResult.FAIL_LATE_ENTRY, FilterStats(
                result=FilterResult.FAIL_LATE_ENTRY,
                reason=f"Price moved too far: {actual_distance:.4f} > {max_allowed_distance:.4f}",
                price_distance=actual_distance,
                price_distance_pct=distance_pct,
                atr=atr,
                atr_multiplier=self.late_entry_multiplier,
                max_allowed_distance=max_allowed_distance,
                current_price=current_price,
                entry_price=entry
            )
    
    def check_max_sl_distance(self, combined_trade: CombinedTrade) -> Tuple[FilterResult, FilterStats]:
        """
        Check if stop loss is not too far (protect against excessive risk)
        
        Args:
            combined_trade: Combined trade object
        
        Returns:
            Tuple of (FilterResult, FilterStats)
        """
        entry = combined_trade.entry
        sl = combined_trade.stop_loss
        
        if combined_trade.direction == 'BUY':
            sl_distance_pct = (entry - sl) / entry
        else:  # SELL
            sl_distance_pct = (sl - entry) / entry
        
        if sl_distance_pct <= self.max_sl_distance_pct:
            return FilterResult.PASS, FilterStats(
                result=FilterResult.PASS,
                reason=f"SL distance OK: {sl_distance_pct:.2%} <= {self.max_sl_distance_pct:.2%}"
            )
        else:
            return FilterResult.FAIL_MAX_SL, FilterStats(
                result=FilterResult.FAIL_MAX_SL,
                reason=f"SL too far: {sl_distance_pct:.2%} > {self.max_sl_distance_pct:.2%}"
            )
    
    def check_spread(self, combined_trade: CombinedTrade,
                    indicators: Dict) -> Tuple[FilterResult, FilterStats]:
        """
        Check if spread is acceptable
        
        Args:
            combined_trade: Combined trade object
            indicators: Technical indicators (contains spread if available)
        
        Returns:
            Tuple of (FilterResult, FilterStats)
        """
        # If spread not available, assume it's okay
        spread = indicators.get('spread', 0)
        spread_pct = indicators.get('spread_pct', 0)
        
        if spread == 0:
            return FilterResult.PASS, FilterStats(
                result=FilterResult.PASS,
                reason="Spread check skipped (no data)"
            )
        
        if spread_pct <= self.max_spread_pct * 100:
            return FilterResult.PASS, FilterStats(
                result=FilterResult.PASS,
                reason=f"Spread OK: {spread_pct:.2f}% <= {self.max_spread_pct*100:.2f}%"
            )
        else:
            return FilterResult.FAIL_MAX_SPREAD, FilterStats(
                result=FilterResult.FAIL_MAX_SPREAD,
                reason=f"Spread too high: {spread_pct:.2f}% > {self.max_spread_pct*100:.2f}%"
            )
    
    def check_volume(self, indicators: Dict) -> Tuple[FilterResult, FilterStats]:
        """
        Check if volume is sufficient (avoid illiquid markets)
        
        Args:
            indicators: Technical indicators (contains volume_ratio)
        
        Returns:
            Tuple of (FilterResult, FilterStats)
        """
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        if volume_ratio >= self.min_volume_ratio:
            return FilterResult.PASS, FilterStats(
                result=FilterResult.PASS,
                reason=f"Volume OK: {volume_ratio:.2f}x >= {self.min_volume_ratio:.2f}x"
            )
        else:
            return FilterResult.FAIL_VOLATILITY, FilterStats(
                result=FilterResult.FAIL_VOLATILITY,
                reason=f"Volume too low: {volume_ratio:.2f}x < {self.min_volume_ratio:.2f}x"
            )
    
    def check_volatility(self, combined_trade: CombinedTrade,
                        indicators: Dict) -> Tuple[FilterResult, FilterStats]:
        """
        Check if volatility is reasonable for this trade
        
        Args:
            combined_trade: Combined trade object
            indicators: Technical indicators
        
        Returns:
            Tuple of (FilterResult, FilterStats)
        """
        atr = indicators.get('atr', 0)
        entry = combined_trade.entry
        
        if atr == 0:
            return FilterResult.PASS, FilterStats(
                result=FilterResult.PASS,
                reason="Volatility check skipped (no ATR)"
            )
        
        # Check if ATR is reasonable relative to entry
        atr_pct = atr / entry
        
        if atr_pct < 0.005:  # ATR < 0.5% - too low volatility
            return FilterResult.FAIL_VOLATILITY, FilterStats(
                result=FilterResult.FAIL_VOLATILITY,
                reason=f"Volatility too low: ATR {atr_pct:.2%} < 0.5%",
                atr=atr
            )
        
        if atr_pct > 0.1:  # ATR > 10% - too high volatility
            return FilterResult.FAIL_VOLATILITY, FilterStats(
                result=FilterResult.FAIL_VOLATILITY,
                reason=f"Volatility too high: ATR {atr_pct:.2%} > 10%",
                atr=atr
            )
        
        return FilterResult.PASS, FilterStats(
            result=FilterResult.PASS,
            reason=f"Volatility OK: ATR {atr_pct:.2%}",
            atr=atr
        )


class LateEntryFilter:
    """
    Specialized filter for late entry detection
    More advanced than the basic check in RiskFilter
    """
    
    def __init__(self, atr_multiplier: float = 1.0,
                 max_wait_bars: int = 5):
        """
        Initialize Late Entry Filter
        
        Args:
            atr_multiplier: How many ATRs price can move before considered late
            max_wait_bars: Maximum bars to wait for entry (for momentum trades)
        """
        self.atr_multiplier = atr_multiplier
        self.max_wait_bars = max_wait_bars
    
    def check(self, combined_trade: CombinedTrade,
              current_price: float,
              indicators: Dict,
              df: pd.DataFrame = None) -> Tuple[bool, str, Dict]:
        """
        Advanced late entry check
        
        Returns:
            Tuple of (passed, reason, details)
        """
        entry = combined_trade.entry
        atr = indicators.get('atr', 0)
        
        details = {
            'entry': entry,
            'current_price': current_price,
            'distance': 0,
            'distance_pct': 0,
            'atr': atr,
            'max_distance': 0
        }
        
        # Basic distance check
        distance = abs(current_price - entry)
        distance_pct = (distance / entry) * 100
        
        details['distance'] = distance
        details['distance_pct'] = distance_pct
        
        if atr > 0:
            max_distance = atr * self.atr_multiplier
            details['max_distance'] = max_distance
            
            if distance <= max_distance:
                return True, f"Price within {distance_pct:.2f}% ({distance:.4f} <= {max_distance:.4f})", details
            else:
                return False, f"Price moved too far: {distance_pct:.2f}% > {max_distance/entry*100:.2f}%", details
        
        # No ATR, use percentage
        max_distance_pct = 1.0  # 1% max chase
        if distance_pct <= max_distance_pct:
            return True, f"Price within {distance_pct:.2f}%", details
        else:
            return False, f"Price moved too far: {distance_pct:.2f}% > {max_distance_pct:.2f}%", details
    
    def check_momentum_shift(self, df: pd.DataFrame, 
                            entry_bar: int = None) -> Tuple[bool, str]:
        """
        Check if momentum has shifted since signal generation
        
        Args:
            df: OHLCV DataFrame
            entry_bar: Bar index where signal was generated
        
        Returns:
            Tuple of (passed, reason)
        """
        if df is None or len(df) < 10:
            return True, "Cannot check momentum shift (insufficient data)"
        
        if entry_bar is None:
            entry_bar = len(df) - 1
        
        # Check if price has reversed significantly
        if entry_bar < len(df) - 1:
            current_close = df['close'].iloc[-1]
            signal_close = df['close'].iloc[entry_bar]
            
            # If price has moved against direction more than 1 ATR, skip
            # This would require ATR calculation, simplified for now
            if abs(current_close - signal_close) / signal_close > 0.01:
                return False, f"Price reversed {abs(current_close - signal_close)/signal_close:.2%} since signal"
        
        return True, "Momentum intact"


class ConsolidationFilter:
    """
    Filter to avoid trading during consolidation (low volatility)
    """
    
    def __init__(self, min_range_pct: float = 0.01,  # 1% minimum range
                 min_atr_pct: float = 0.005):        # 0.5% minimum ATR
        self.min_range_pct = min_range_pct
        self.min_atr_pct = min_atr_pct
    
    def check(self, indicators: Dict, df: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Check if market is not in consolidation
        
        Args:
            indicators: Technical indicators
            df: Optional DataFrame for additional checks
        
        Returns:
            Tuple of (passed, reason)
        """
        # Check ATR
        atr = indicators.get('atr', 0)
        price = indicators.get('price', 0)
        
        if price > 0 and atr > 0:
            atr_pct = atr / price
            if atr_pct < self.min_atr_pct:
                return False, f"Consolidation detected: ATR {atr_pct:.2%} < {self.min_atr_pct:.2%}"
        
        # Check Bollinger Band width
        bb_width = indicators.get('bb_width', 0)
        if bb_width > 0:
            if bb_width < 0.02:  # 2% BB width = consolidation
                return False, f"Consolidation detected: BB width {bb_width:.2%} < 2%"
        
        return True, "No consolidation detected"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_risk_management(combined_trade: CombinedTrade,
                         current_price: float,
                         indicators: Dict,
                         config: PipelineConfig = None) -> Tuple[bool, str, Dict]:
    """
    Quick risk management check for a trade
    
    Args:
        combined_trade: Combined trade object
        current_price: Current market price
        indicators: Technical indicators
        config: Pipeline configuration
    
    Returns:
        Tuple of (passed, reason, stats)
    """
    risk_filter = RiskFilter(config)
    result, stats = risk_filter.apply_all_filters(
        combined_trade, current_price, indicators
    )
    
    passed = result == FilterResult.PASS
    return passed, stats.reason, stats.to_dict()


def calculate_max_position_size(combined_trade: CombinedTrade,
                                account_balance: float,
                                risk_per_trade: float = 0.02) -> float:
    """
    Calculate maximum position size based on risk
    
    Args:
        combined_trade: Combined trade object
        account_balance: Total account balance
        risk_per_trade: Risk per trade as percentage (0.02 = 2%)
    
    Returns:
        Position size in units
    """
    entry = combined_trade.entry
    sl = combined_trade.stop_loss
    direction = combined_trade.direction
    
    # Calculate risk amount in dollars
    risk_amount = account_balance * risk_per_trade
    
    # Calculate risk per unit
    if direction == 'BUY':
        risk_per_unit = entry - sl
    else:  # SELL
        risk_per_unit = sl - entry
    
    if risk_per_unit <= 0:
        return 0
    
    # Position size = risk_amount / risk_per_unit
    position_size = risk_amount / risk_per_unit
    
    return position_size