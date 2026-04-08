"""
Base class for all strategies in Strategy Expert module
All strategies must inherit from this class
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class StrategyOutput:
    """Standard output format for all strategies"""
    action: str                     # 'BUY', 'SELL', 'HOLD'
    confidence: float               # 0-1, raw strategy confidence
    entry: float                    # Entry price
    stop_loss: float                # Stop loss price
    take_profit: float              # Take profit price
    risk_reward: float              # Risk/Reward ratio
    reasons: List[str]              # Why this strategy recommends trade
    strategy_name: str              # Name of the strategy
    indicators_used: List[str] = field(default_factory=list)  # For debugging
    weight: float = 1.0
    def __post_init__(self):
        """Validate output data"""
        if self.action in ['BUY', 'SELL']:
            # Validate entry, SL, TP are positive
            assert self.entry > 0, f"Entry must be positive: {self.entry}"
            assert self.stop_loss > 0, f"Stop loss must be positive: {self.stop_loss}"
            assert self.take_profit > 0, f"Take profit must be positive: {self.take_profit}"
            
            # Validate SL and TP are on correct side
            if self.action == 'BUY':
                assert self.stop_loss < self.entry, f"BUY: SL ({self.stop_loss}) must be below entry ({self.entry})"
                assert self.take_profit > self.entry, f"BUY: TP ({self.take_profit}) must be above entry ({self.entry})"
            else:  # SELL
                assert self.stop_loss > self.entry, f"SELL: SL ({self.stop_loss}) must be above entry ({self.entry})"
                assert self.take_profit < self.entry, f"SELL: TP ({self.take_profit}) must be below entry ({self.entry})"
            
            # Validate risk/reward
            assert self.risk_reward >= 0, f"Risk/Reward must be positive: {self.risk_reward}"


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    Each strategy must implement:
        - generate_signal(): Core trading logic
        - calculate_entry_sl_tp(): Price levels for the trade
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = ""
        self.enabled = True
        self.weight = 1.0                     # Dynamic weight (updated by WeightManager)
        self.base_weight = 1.0                # Initial/base weight from config
        self.min_confidence = 0.6             # Minimum confidence to be considered
        self.min_rr = 1.5                     # Minimum risk/reward for this strategy
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, indicators: Dict,
                       market_regime: Dict, module_signals: Dict = None) -> Optional[StrategyOutput]:
        """
        Generate trading signal
        
        Args:
            df: Raw OHLCV DataFrame
            indicators: Pre-calculated indicators (or will be calculated if missing)
            market_regime: Market regime data (regime, bias, trend_strength)
            module_signals: Optional signals from other modules (for reference only)
        
        Returns:
            StrategyOutput with full trade setup, or None if no signal
        """
        pass
    
    @abstractmethod
    def calculate_entry_sl_tp(self, df: pd.DataFrame, indicators: Dict,
                              market_regime: Dict, action: str) -> Dict:
        """
        Calculate precise entry, stop loss, and take profit levels
        
        Args:
            df: Raw OHLCV DataFrame
            indicators: Technical indicators
            market_regime: Market regime data
            action: 'BUY' or 'SELL'
        
        Returns:
            Dict with 'entry', 'stop_loss', 'take_profit', 'risk_reward'
        """
        pass
    
    def get_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'weight': self.weight,
            'base_weight': self.base_weight,
            'min_confidence': self.min_confidence,
            'min_rr': self.min_rr
        }
    
    def calculate_rr(self, entry: float, sl: float, tp: float, action: str) -> float:
        """Calculate risk-reward ratio"""
        if action == 'BUY':
            risk = entry - sl
            reward = tp - entry
        else:  # SELL
            risk = sl - entry
            reward = entry - tp
        
        if risk <= 0:
            return 0.0
        return reward / risk
    
    def adjust_by_regime(self, confidence: float, action: str, 
                         market_regime: Dict) -> float:
        """
        Adjust confidence based on market regime alignment
        Can be called by strategies for regime-based confidence adjustment
        """
        regime_bias = market_regime.get('bias_score', 0)
        regime_name = market_regime.get('regime', 'UNKNOWN')
        
        # Boost if aligned
        if action == 'BUY' and regime_bias > 0:
            return min(0.95, confidence * 1.05)
        elif action == 'SELL' and regime_bias < 0:
            return min(0.95, confidence * 1.05)
        
        # Reduce if against regime
        if action == 'BUY' and regime_bias < -0.3:
            return confidence * 0.85
        elif action == 'SELL' and regime_bias > 0.3:
            return confidence * 0.85
        
        return confidence
    
    def get_atr(self, indicators: Dict, default: float = 0.01) -> float:
        """Get ATR with fallback"""
        return indicators.get('atr', default)
    
    def get_volume_ratio(self, indicators: Dict) -> float:
        """Get volume ratio with fallback"""
        return indicators.get('volume_ratio', 1.0)