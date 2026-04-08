# risk_manager.py - Complete Risk Management with Position Sizing & Protection Plans
"""
Complete Risk Management Module for Phase 8

Features:
- Position sizing with Kelly Criterion and multi-factor adjustment
- Portfolio heat management with sector and correlation risk
- Trade protection plans (trailing stops, partial TPs, breakeven)
- Adverse excursion monitoring
- Trade invalidation engine
- Risk of ruin calculations
- Dynamic risk scoring based on signal quality

Version: 3.0
Author: Risk Management Expert
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from config import Config
from logger import log

EPSILON = 1e-10


class ExitReason(Enum):
    """Reason for trade exit"""
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_EXIT = "TIME_EXIT"
    ADVERSE_EXCURSION = "ADVERSE_EXCURSION"
    INVALIDATION = "INVALIDATION"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"


class RiskLevel(Enum):
    """Risk level classification"""
    EXTREME = "EXTREME"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class SignalQuality(Enum):
    """Signal quality classification"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class TradeProtectionPlan:
    """Comprehensive trade protection plan"""
    trailing_stop: Dict[str, Any] = field(default_factory=dict)
    partial_tp_levels: List[Dict] = field(default_factory=list)
    breakeven_trigger: float = 0.02
    time_exit_bars: int = 48
    adverse_excursion_limit: float = 0.03
    invalidation_levels: List[float] = field(default_factory=list)


@dataclass
class RiskProfile:
    """Risk profile for a trade"""
    signal_quality: SignalQuality
    probability: float
    microstructure_aligned: bool
    htf_aligned: bool
    market_regime: str
    volatility_state: str
    kelly_fraction: float = 0.0
    risk_of_ruin: float = 0.0
    portfolio_heat: float = 0.0
    layer_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeProtection:
    """Trade protection settings for a specific trade"""
    trailing_activated: bool = False
    trailing_start_pct: float = 0.0
    trailing_distance: float = 0.0
    trailing_strategy: str = "atr"
    trailing_atr_multiple: float = 2.0
    breakeven_activated: bool = False
    breakeven_trigger: float = 0.02
    partial_tp_hit: List[float] = field(default_factory=list)
    time_exit_bars: int = 48
    adverse_excursion: float = 0.0
    invalidation_level: Optional[float] = None


@dataclass
class PortfolioHeat:
    """Portfolio heat metrics"""
    total_exposure: float
    sector_exposure: Dict[str, float]
    correlation_risk: float
    concentration_risk: float
    margin_used: float
    risk_of_ruin: float
    heat_score: float  # 0-100
    status: str  # 'COLD', 'WARM', 'HOT', 'EXTREME'


@dataclass
class PositionSizeResult:
    """Complete position sizing result"""
    position_size_usd: float          # Amount in dollars to trade
    position_size_units: float        # Amount in coin units
    risk_amount_usd: float            # Dollars at risk
    risk_percent: float               # Percentage of portfolio
    position_multiplier: float        # 0.5-1.5 based on quality
    kelly_used: bool
    factor_breakdown: Dict[str, float]


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment result"""
    is_allowed: bool
    reason: str
    risk_profile: RiskProfile
    position_size: PositionSizeResult
    protection_plan: TradeProtectionPlan
    portfolio_heat: PortfolioHeat
    warning_flags: List[str]


# ============================================================================
# POSITION SIZING CALCULATOR
# ============================================================================

def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float, 
                              kelly_fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing
    Kelly % = W - [(1-W)/R] where R = avg_win/avg_loss
    Returns fractional Kelly (default 25% of full Kelly)
    """
    try:
        if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
            return 0.0
        
        # Calculate win/loss ratio
        r = avg_win / avg_loss
        
        # Kelly formula
        kelly = win_rate - ((1 - win_rate) / r)
        
        # Use fractional Kelly
        kelly = max(0.0, min(0.25, kelly * kelly_fraction))
        
        return round(kelly, 4)
    except Exception:
        return 0.0


def calculate_risk_of_ruin(win_rate: float, risk_per_trade: float, 
                          account_size: float, simulations: int = 1000) -> Dict[str, float]:
    """
    Calculate risk of ruin using Monte Carlo simulation
    """
    try:
        if win_rate <= 0 or win_rate >= 1 or risk_per_trade <= 0:
            return {'ror': 1.0, 'safe': False}
        
        # Monte Carlo simulation
        np.random.seed(42)
        ruin_count = 0
        
        for _ in range(simulations):
            capital = account_size
            while capital > 0 and capital < account_size * 2:
                # Simulate trade (assumes 2:1 RR)
                if np.random.random() < win_rate:
                    capital += risk_per_trade * account_size * 2
                else:
                    capital -= risk_per_trade * account_size
                
                if capital <= 0:
                    ruin_count += 1
                    break
        
        mc_ror = ruin_count / simulations
        
        return {
            'ror': round(mc_ror, 4),
            'safe': mc_ror < 0.05,
            'edge': round(2 * win_rate - 1, 3)
        }
    except Exception:
        return {'ror': 1.0, 'safe': False}


class PositionSizingCalculator:
    """
    Advanced position sizing with multi-factor adjustment
    """
    
    def __init__(self):
        # Load configuration
        self.max_risk_per_trade = getattr(Config, 'MAX_RISK_PER_TRADE', 0.02)
        self.portfolio_value = getattr(Config, 'PORTFOLIO_VALUE', 10000)
        self.max_position_concentration = getattr(Config, 'MAX_POSITION_CONCENTRATION', 0.25)
        self.enable_kelly = getattr(Config, 'ENABLE_KELLY', True)
        self.kelly_fraction = getattr(Config, 'KELLY_FRACTION', 0.25)
        
        # Multipliers from config
        self.confidence_multipliers = getattr(Config, 'RISK_CONFIDENCE_MULTIPLIERS', {
            0.85: 1.3,
            0.75: 1.1,
            0.65: 1.0,
            0.55: 0.8,
            0.00: 0.5
        })
        
        self.win_rate_multipliers = getattr(Config, 'RISK_WIN_RATE_MULTIPLIERS', {
            0.70: 1.2,
            0.60: 1.1,
            0.50: 1.0,
            0.00: 0.8
        })
        
        self.regime_multipliers = getattr(Config, 'RISK_REGIME_MULTIPLIERS', {
            'STRONG_BULL_TREND': 1.15,
            'BULL_TREND': 1.0,
            'RANGING': 0.7,
            'VOLATILE': 0.6
        })
        
        self.volatility_multipliers = getattr(Config, 'RISK_VOLATILITY_MULTIPLIERS', {
            'LOW': 1.2,
            'NORMAL': 1.0,
            'HIGH': 0.7,
            'EXTREME': 0.5
        })
        
        self.grade_multipliers = getattr(Config, 'RISK_GRADE_MULTIPLIERS', {
            'A+': 1.5,
            'A': 1.3,
            'B': 1.1,
            'C': 0.8,
            'D': 0.5,
            'F': 0.3
        })
    
    def calculate(self,
                  confidence: float,
                  grade: str,
                  market_regime: str,
                  volatility_state: str,
                  win_rate: float = 0.5,
                  avg_win: float = 2.0,
                  avg_loss: float = 1.0,
                  correlation_reduction: float = 1.0,
                  portfolio_heat: float = 0.0,
                  entry_price: float = None,
                  stop_loss: float = None) -> PositionSizeResult:
        """
        Calculate optimal position size using multiple factors
        
        Args:
            confidence: Signal confidence (0-1)
            grade: Signal grade (A+, A, B, C, D, F)
            market_regime: Market regime string
            volatility_state: Volatility state (LOW, NORMAL, HIGH, EXTREME)
            win_rate: Historical win rate for Kelly
            avg_win: Average win amount for Kelly
            avg_loss: Average loss amount for Kelly
            correlation_reduction: Reduction factor from correlation risk
            portfolio_heat: Current portfolio heat score (0-100)
            entry_price: Entry price (for unit calculation)
            stop_loss: Stop loss price (for risk per unit)
        
        Returns:
            PositionSizeResult with all position details
        """
        
        # Step 1: Base risk
        base_risk = self.max_risk_per_trade
        
        # Step 2: Confidence factor
        confidence_factor = 1.0
        for threshold, multiplier in sorted(self.confidence_multipliers.items(), reverse=True):
            if confidence >= threshold:
                confidence_factor = multiplier
                break
        
        # Step 3: Win rate factor (for Kelly adjustment)
        win_rate_factor = 1.0
        for threshold, multiplier in sorted(self.win_rate_multipliers.items(), reverse=True):
            if win_rate >= threshold:
                win_rate_factor = multiplier
                break
        
        # Step 4: Regime factor
        regime_factor = self.regime_multipliers.get(market_regime, 1.0)
        
        # Step 5: Volatility factor
        volatility_factor = self.volatility_multipliers.get(volatility_state, 1.0)
        
        # Step 6: Grade factor
        grade_factor = self.grade_multipliers.get(grade, 1.0)
        
        # Step 7: Kelly adjustment
        kelly_used = False
        kelly_factor = 1.0
        
        if self.enable_kelly:
            kelly = calculate_kelly_criterion(win_rate, avg_win, avg_loss, self.kelly_fraction)
            if kelly > 0:
                kelly_factor = min(1.5, kelly / self.max_risk_per_trade)
                kelly_used = True
        
        # Step 8: Correlation reduction
        correlation_factor = correlation_reduction
        
        # Step 9: Portfolio heat reduction
        heat_factor = 1.0
        if portfolio_heat > 70:
            heat_factor = 0.7
        elif portfolio_heat > 50:
            heat_factor = 0.85
        elif portfolio_heat > 30:
            heat_factor = 0.95
        
        # Calculate final risk percentage
        risk_percent = (base_risk * 
                        confidence_factor * 
                        win_rate_factor * 
                        regime_factor * 
                        volatility_factor * 
                        grade_factor * 
                        kelly_factor * 
                        correlation_factor * 
                        heat_factor)
        
        # Cap risk
        risk_percent = min(0.05, max(0.005, risk_percent))
        
        # Calculate position multiplier (for output)
        position_multiplier = (confidence_factor * 
                               win_rate_factor * 
                               regime_factor * 
                               volatility_factor * 
                               grade_factor * 
                               kelly_factor)
        position_multiplier = min(1.5, max(0.5, position_multiplier))
        
        # Calculate position size
        risk_amount_usd = self.portfolio_value * risk_percent
        position_size_usd = risk_amount_usd  # This will be refined with risk per unit if SL provided
        
        # If stop loss provided, calculate units based on risk per unit
        position_size_units = 0
        if entry_price and stop_loss and entry_price > 0 and stop_loss > 0:
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit > 0:
                position_size_units = risk_amount_usd / risk_per_unit
                position_size_usd = position_size_units * entry_price
        
        # Cap by max position concentration
        max_position_usd = self.max_position_concentration * self.portfolio_value
        if position_size_usd > max_position_usd:
            position_size_usd = max_position_usd
            if entry_price and entry_price > 0:
                position_size_units = position_size_usd / entry_price
        
        # Factor breakdown for debugging
        factor_breakdown = {
            'base_risk': round(base_risk, 4),
            'confidence_factor': round(confidence_factor, 3),
            'win_rate_factor': round(win_rate_factor, 3),
            'regime_factor': round(regime_factor, 3),
            'volatility_factor': round(volatility_factor, 3),
            'grade_factor': round(grade_factor, 3),
            'kelly_factor': round(kelly_factor, 3),
            'correlation_factor': round(correlation_factor, 3),
            'heat_factor': round(heat_factor, 3),
            'final_risk_percent': round(risk_percent, 4)
        }
        
        return PositionSizeResult(
            position_size_usd=round(position_size_usd, 2),
            position_size_units=round(position_size_units, 8),
            risk_amount_usd=round(risk_amount_usd, 2),
            risk_percent=round(risk_percent, 4),
            position_multiplier=round(position_multiplier, 2),
            kelly_used=kelly_used,
            factor_breakdown=factor_breakdown
        )


# ============================================================================
# PORTFOLIO HEAT MANAGER
# ============================================================================

class PortfolioHeatManager:
    """Manages portfolio heat and exposure"""
    
    def __init__(self):
        self.max_total_exposure = getattr(Config, 'MAX_PORTFOLIO_EXPOSURE', 2.0)
        self.max_sector_exposure = getattr(Config, 'MAX_SECTOR_EXPOSURE', 0.5)
        self.positions = []
        
        # Sector mapping
        self.sector_map = {
            'BTCUSDT': 'L1', 'ETHUSDT': 'L1', 'SOLUSDT': 'L1',
            'BNBUSDT': 'Exchange', 'UNIUSDT': 'DeFi', 'AAVEUSDT': 'DeFi',
            'DOGEUSDT': 'Meme', 'SHIBUSDT': 'Meme',
            'FETUSDT': 'AI', 'AGIXUSDT': 'AI',
            'AXSUSDT': 'Gaming', 'SANDUSDT': 'Gaming'
        }
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        for key, sector in self.sector_map.items():
            if key in symbol.upper():
                return sector
        return 'Other'
    
    def calculate_portfolio_heat(self, portfolio_value: float, 
                                positions: List[Dict], 
                                correlation_data: Dict = None) -> PortfolioHeat:
        """
        Calculate portfolio heat score and risk metrics
        """
        try:
            if not positions:
                return PortfolioHeat(
                    total_exposure=0.0,
                    sector_exposure={},
                    correlation_risk=0.0,
                    concentration_risk=0.0,
                    margin_used=0.0,
                    risk_of_ruin=0.0,
                    heat_score=0.0,
                    status='COLD'
                )
            
            # Calculate total exposure
            total_exposure = sum(p.get('position_value', 0) for p in positions) / portfolio_value
            
            # Calculate sector exposure
            sector_exposure = {}
            for p in positions:
                sector = self.get_sector(p.get('symbol', ''))
                value = p.get('position_value', 0)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + value
            
            for sector in sector_exposure:
                sector_exposure[sector] /= portfolio_value
            
            # Concentration risk (Herfindahl index)
            weights = [p.get('position_value', 0) / portfolio_value for p in positions]
            hhi = sum(w ** 2 for w in weights)
            concentration_risk = min(1.0, hhi * 2)
            
            # Correlation risk
            correlation_risk = 0.0
            if correlation_data and 'matrix' in correlation_data and len(positions) > 1:
                symbols = [p.get('symbol') for p in positions]
                matrix = correlation_data['matrix']
                
                corrs = []
                for i, sym1 in enumerate(symbols):
                    for j, sym2 in enumerate(symbols):
                        if i < j and sym1 in matrix.index and sym2 in matrix.columns:
                            corrs.append(abs(matrix.loc[sym1, sym2]))
                
                correlation_risk = np.mean(corrs) if corrs else 0.0
            
            # Margin used (simplified)
            margin_used = total_exposure * 0.1
            
            # Risk of ruin
            avg_win_rate = np.mean([p.get('win_rate', 0.5) for p in positions])
            avg_risk = np.mean([p.get('risk_percent', 0.02) for p in positions])
            ror_result = calculate_risk_of_ruin(avg_win_rate, avg_risk, portfolio_value)
            risk_of_ruin = ror_result['ror']
            
            # Combined heat score (0-100)
            heat_score = (
                min(100, total_exposure * 50) * 0.3 +
                concentration_risk * 100 * 0.2 +
                correlation_risk * 100 * 0.2 +
                risk_of_ruin * 100 * 0.3
            )
            
            # Status
            if heat_score < 30:
                status = 'COLD'
            elif heat_score < 60:
                status = 'WARM'
            elif heat_score < 80:
                status = 'HOT'
            else:
                status = 'EXTREME'
            
            return PortfolioHeat(
                total_exposure=round(total_exposure, 3),
                sector_exposure={k: round(v, 3) for k, v in sector_exposure.items()},
                correlation_risk=round(correlation_risk, 3),
                concentration_risk=round(concentration_risk, 3),
                margin_used=round(margin_used, 3),
                risk_of_ruin=round(risk_of_ruin, 4),
                heat_score=round(heat_score, 1),
                status=status
            )
            
        except Exception as e:
            log.debug(f"Error calculating portfolio heat: {e}")
            return PortfolioHeat(
                total_exposure=0.0,
                sector_exposure={},
                correlation_risk=0.0,
                concentration_risk=0.0,
                margin_used=0.0,
                risk_of_ruin=0.0,
                heat_score=0.0,
                status='UNKNOWN'
            )


# ============================================================================
# CORRELATION RISK FILTER
# ============================================================================

class CorrelationRiskFilter:
    """Filter trades based on correlation with existing positions"""
    
    def __init__(self):
        self.threshold = getattr(Config, 'CORRELATION_RISK_THRESHOLD', 0.7)
        self.reduction = getattr(Config, 'CORRELATION_REDUCTION', 0.5)
    
    def check_correlation_risk(self, symbol: str, correlation_data: Dict, 
                               existing_positions: List[Dict]) -> Tuple[bool, str, float, float]:
        """
        Check if new trade would create excessive correlation risk
        Returns: (allowed, reason, max_correlation, reduction_factor)
        """
        try:
            if not existing_positions or not correlation_data:
                return True, "No existing positions or correlation data", 0.0, 1.0
            
            matrix = correlation_data.get('matrix')
            if matrix is None:
                return True, "No correlation matrix", 0.0, 1.0
            
            max_correlation = 0.0
            correlated_with = []
            
            for pos in existing_positions:
                pos_symbol = pos.get('symbol')
                if pos_symbol in matrix.index and symbol in matrix.columns:
                    corr = abs(matrix.loc[pos_symbol, symbol])
                    if corr > max_correlation:
                        max_correlation = corr
                    
                    if corr > self.threshold:
                        correlated_with.append(f"{pos_symbol}({corr:.2f})")
            
            if max_correlation > self.threshold:
                return False, f"High correlation with: {', '.join(correlated_with)}", max_correlation, 0.0
            elif max_correlation > self.threshold * 0.7:
                reduction = self.reduction
                return True, f"Moderate correlation ({max_correlation:.2f}) - reduce size by {reduction:.0%}", max_correlation, reduction
            else:
                return True, f"Low correlation ({max_correlation:.2f})", max_correlation, 1.0
                
        except Exception as e:
            return True, f"Error checking correlation: {e}", 0.0, 1.0


# ============================================================================
# TRADE PROTECTION ENGINE
# ============================================================================

class TradeProtectionEngine:
    """Advanced trade protection with multiple strategies"""
    
    def __init__(self):
        self.strategies = {
            'atr': self._trail_by_atr,
            'percentage': self._trail_by_percentage,
            'parabolic': self._trail_by_parabolic,
            'swing': self._trail_by_swing
        }
        
        # Trade protection plans by signal quality
        self.protection_plans = {
            SignalQuality.VERY_HIGH: TradeProtectionPlan(
                trailing_stop={'strategy': 'atr', 'atr_multiple': 2.0, 'activation_pct': 0.01},
                partial_tp_levels=[
                    {'pct': 0.33, 'fraction': 0.25},
                    {'pct': 0.66, 'fraction': 0.35},
                    {'pct': 1.0, 'fraction': 0.40}
                ],
                breakeven_trigger=0.015,
                time_exit_bars=72,
                adverse_excursion_limit=0.04
            ),
            SignalQuality.HIGH: TradeProtectionPlan(
                trailing_stop={'strategy': 'atr', 'atr_multiple': 2.0, 'activation_pct': 0.02},
                partial_tp_levels=[
                    {'pct': 0.5, 'fraction': 0.5},
                    {'pct': 1.0, 'fraction': 0.5}
                ],
                breakeven_trigger=0.02,
                time_exit_bars=48,
                adverse_excursion_limit=0.03
            ),
            SignalQuality.MEDIUM: TradeProtectionPlan(
                trailing_stop={'strategy': 'atr', 'atr_multiple': 2.5, 'activation_pct': 0.03},
                partial_tp_levels=[
                    {'pct': 1.0, 'fraction': 1.0}
                ],
                breakeven_trigger=0.025,
                time_exit_bars=36,
                adverse_excursion_limit=0.025
            ),
            SignalQuality.LOW: TradeProtectionPlan(
                trailing_stop={'strategy': 'percentage', 'trail_pct': 0.01, 'activation_pct': 0.04},
                partial_tp_levels=[],
                breakeven_trigger=0.03,
                time_exit_bars=24,
                adverse_excursion_limit=0.02
            )
        }
    
    def get_protection_plan(self, signal_quality: SignalQuality) -> TradeProtectionPlan:
        """Get protection plan for signal quality"""
        return self.protection_plans.get(signal_quality, self.protection_plans[SignalQuality.MEDIUM])
    
    def calculate_trailing_stop(self, df: pd.DataFrame, entry_price: float, 
                               current_price: float, direction: str,
                               strategy: str = 'atr', params: Dict = None,
                               last_stop: float = None) -> Dict[str, Any]:
        """
        Calculate trailing stop using specified strategy
        
        Returns:
            Dict with trailing_stop, activated, distance
        """
        if params is None:
            params = {}
        
        params['last_stop'] = last_stop
        
        if strategy in self.strategies:
            return self.strategies[strategy](df, entry_price, current_price, direction, params)
        else:
            return self._trail_by_atr(df, entry_price, current_price, direction, params)
    
    def _trail_by_atr(self, df, entry_price, current_price, direction, params):
        """Trail by ATR multiple"""
        try:
            atr_multiple = params.get('atr_multiple', 2.0)
            atr = float(df['atr'].iloc[-1]) if 'atr' in df else current_price * 0.02
            last_stop = params.get('last_stop')
            
            if direction.upper() == 'BUY':
                trail_stop = current_price - (atr * atr_multiple)
                if last_stop:
                    trail_stop = max(trail_stop, last_stop)
            else:
                trail_stop = current_price + (atr * atr_multiple)
                if last_stop:
                    trail_stop = min(trail_stop, last_stop)
            
            distance_pct = abs(current_price - trail_stop) / current_price
            activated = abs(current_price - entry_price) / entry_price > params.get('activation_pct', 0.02)
            
            return {
                'trailing_stop': round(trail_stop, 6),
                'distance': round(distance_pct, 4),
                'activated': activated,
                'strategy': 'atr'
            }
        except Exception:
            return {'trailing_stop': None, 'activated': False}
    
    def _trail_by_percentage(self, df, entry_price, current_price, direction, params):
        """Trail by fixed percentage"""
        try:
            trail_pct = params.get('trail_pct', 0.02)
            last_stop = params.get('last_stop')
            
            if direction.upper() == 'BUY':
                trail_stop = current_price * (1 - trail_pct)
                if last_stop:
                    trail_stop = max(trail_stop, last_stop)
            else:
                trail_stop = current_price * (1 + trail_pct)
                if last_stop:
                    trail_stop = min(trail_stop, last_stop)
            
            activated = abs(current_price - entry_price) / entry_price > params.get('activation_pct', 0.03)
            
            return {
                'trailing_stop': round(trail_stop, 6),
                'distance': trail_pct,
                'activated': activated,
                'strategy': 'percentage'
            }
        except Exception:
            return {'trailing_stop': None, 'activated': False}
    
    def _trail_by_parabolic(self, df, entry_price, current_price, direction, params):
        """Trail using Parabolic SAR"""
        try:
            if 'psar' not in df.columns:
                return self._trail_by_atr(df, entry_price, current_price, direction, params)
            
            psar = float(df['psar'].iloc[-1])
            last_stop = params.get('last_stop')
            
            if direction.upper() == 'BUY':
                trail_stop = max(psar, last_stop) if last_stop else psar
            else:
                trail_stop = min(psar, last_stop) if last_stop else psar
            
            activated = True
            
            return {
                'trailing_stop': round(trail_stop, 6),
                'distance': abs(current_price - trail_stop) / current_price,
                'activated': activated,
                'strategy': 'parabolic'
            }
        except Exception:
            return {'trailing_stop': None, 'activated': False}
    
    def _trail_by_swing(self, df, entry_price, current_price, direction, params):
        """Trail using swing points"""
        try:
            lookback = params.get('lookback', 20)
            recent = df.iloc[-lookback:]
            last_stop = params.get('last_stop')
            
            if direction.upper() == 'BUY':
                swing_low = recent['low'].min()
                trail_stop = swing_low * 0.995
                if last_stop:
                    trail_stop = max(trail_stop, last_stop)
            else:
                swing_high = recent['high'].max()
                trail_stop = swing_high * 1.005
                if last_stop:
                    trail_stop = min(trail_stop, last_stop)
            
            activated = True
            
            return {
                'trailing_stop': round(trail_stop, 6),
                'distance': abs(current_price - trail_stop) / current_price,
                'activated': activated,
                'strategy': 'swing'
            }
        except Exception:
            return {'trailing_stop': None, 'activated': False}


# ============================================================================
# ADVERSE EXCURSION MONITOR
# ============================================================================

class AdverseExcursionMonitor:
    """Monitor adverse price movement against position"""
    
    def __init__(self):
        self.max_adverse_pct = getattr(Config, 'ADVERSE_EXCURSION_LIMIT', 0.03)
    
    def calculate_adverse_excursion(self, df: pd.DataFrame, entry_price: float,
                                   entry_index: int, direction: str) -> Dict[str, Any]:
        """
        Calculate Maximum Adverse Excursion (MAE)
        """
        try:
            if entry_index >= len(df) - 1:
                return {'mae': 0.0, 'current_ae': 0.0, 'signal': 'HOLD'}
            
            post_entry = df.iloc[entry_index + 1:]
            
            if direction.upper() == 'BUY':
                lowest = post_entry['low'].min()
                mae = (entry_price - lowest) / entry_price
                current_ae = (entry_price - df['low'].iloc[-1]) / entry_price
            else:
                highest = post_entry['high'].max()
                mae = (highest - entry_price) / entry_price
                current_ae = (df['high'].iloc[-1] - entry_price) / entry_price
            
            if mae > self.max_adverse_pct:
                signal = "EXIT"
            elif current_ae > self.max_adverse_pct * 0.8:
                signal = "CAUTION"
            else:
                signal = "HOLD"
            
            return {
                'mae': round(mae, 4),
                'current_ae': round(current_ae, 4),
                'mae_vs_threshold': round(mae / self.max_adverse_pct, 2),
                'signal': signal,
                'threshold': self.max_adverse_pct
            }
            
        except Exception as e:
            return {'mae': 0.0, 'current_ae': 0.0, 'signal': 'ERROR', 'error': str(e)}


# ============================================================================
# TRADE INVALIDATION ENGINE
# ============================================================================

class TradeInvalidationEngine:
    """Detect when trade thesis is invalidated"""
    
    def __init__(self):
        self.invalidation_reasons = []
    
    def check_invalidation(self, df: pd.DataFrame, entry_price: float,
                          direction: str, setup_type: str,
                          invalidation_levels: List[float] = None) -> Dict[str, Any]:
        """
        Check if trade should be invalidated
        """
        try:
            current_price = float(df['close'].iloc[-1])
            invalidation_reasons = []
            is_invalidated = False
            
            # 1. Price breaks invalidation level
            if invalidation_levels:
                for level in invalidation_levels:
                    if direction.upper() == 'BUY' and current_price < level:
                        is_invalidated = True
                        invalidation_reasons.append(f"Price broke support at {level:.6f}")
                    elif direction.upper() == 'SELL' and current_price > level:
                        is_invalidated = True
                        invalidation_reasons.append(f"Price broke resistance at {level:.6f}")
            
            # 2. Structure break
            if setup_type in ['TREND_FOLLOW', 'BREAKOUT']:
                if direction.upper() == 'BUY':
                    recent_low = df['low'].iloc[-10:].min()
                    if current_price < recent_low:
                        is_invalidated = True
                        invalidation_reasons.append(f"Structure broken - below recent low {recent_low:.6f}")
                else:
                    recent_high = df['high'].iloc[-10:].max()
                    if current_price > recent_high:
                        is_invalidated = True
                        invalidation_reasons.append(f"Structure broken - above recent high {recent_high:.6f}")
            
            # 3. Volume confirmation failure
            if 'volume_ratio' in df.columns:
                vol_ratio = df['volume_ratio'].iloc[-1]
                if vol_ratio < 0.5:
                    invalidation_reasons.append(f"Low volume confirmation ({vol_ratio:.2f})")
            
            return {
                'is_invalidated': is_invalidated,
                'reasons': invalidation_reasons,
                'confidence': 1.0 if is_invalidated else 0.0,
                'current_price': current_price
            }
            
        except Exception as e:
            return {'is_invalidated': False, 'reasons': [f"Error: {e}"], 'confidence': 0.0}


# ============================================================================
# RISK PROFILE CREATOR
# ============================================================================

def create_risk_profile(
    layer_scores: Dict[str, float],
    microstructure_aligned: bool,
    htf_aligned: bool,
    market_regime: str,
    volatility_state: str,
    historical_stats: Dict = None
) -> RiskProfile:
    """
    Create comprehensive risk profile from all available data
    
    Args:
        layer_scores: Scores from all technical layers
        microstructure_aligned: Whether microstructure aligns
        htf_aligned: Whether higher timeframes align
        market_regime: Current market regime
        volatility_state: Current volatility state
        historical_stats: Historical performance stats for Kelly
    
    Returns:
        RiskProfile with all risk metrics
    """
    # Calculate average score from layers
    if layer_scores:
        avg_score = np.mean(list(layer_scores.values()))
    else:
        avg_score = 0.5
    
    # Determine signal quality
    if avg_score > 0.8 and microstructure_aligned and htf_aligned:
        quality = SignalQuality.VERY_HIGH
    elif avg_score > 0.7:
        quality = SignalQuality.HIGH
    elif avg_score > 0.6:
        quality = SignalQuality.MEDIUM
    else:
        quality = SignalQuality.LOW
    
    # Calculate Kelly fraction
    kelly = 0.0
    if historical_stats:
        win_rate = historical_stats.get('win_rate', 0.5)
        avg_win = historical_stats.get('avg_win', 2.0)
        avg_loss = historical_stats.get('avg_loss', 1.0)
        kelly = calculate_kelly_criterion(win_rate, avg_win, avg_loss, 0.25)
    
    # Calculate risk of ruin
    ror_result = calculate_risk_of_ruin(
        historical_stats.get('win_rate', 0.5) if historical_stats else 0.5,
        0.02,  # Default risk per trade
        10000  # Default portfolio value
    )
    
    return RiskProfile(
        signal_quality=quality,
        probability=avg_score,
        microstructure_aligned=microstructure_aligned,
        htf_aligned=htf_aligned,
        market_regime=market_regime,
        volatility_state=volatility_state,
        kelly_fraction=kelly,
        risk_of_ruin=ror_result['ror'],
        portfolio_heat=0.0,
        layer_scores=layer_scores
    )


# ============================================================================
# HELPER FUNCTIONS (Legacy compatibility)
# ============================================================================

def validate_risk_reward(risk_reward: float, min_rr: float = None) -> Tuple[bool, str]:
    """Validate risk/reward ratio"""
    min_rr = min_rr or getattr(Config, 'MIN_RISK_REWARD', 1.5)
    if risk_reward >= min_rr:
        return True, f"rr_ok ({risk_reward:.2f} >= {min_rr:.2f})"
    return False, f"rr_too_low ({risk_reward:.2f} < {min_rr:.2f})"


def calculate_breakeven_stop(entry_price: float, current_price: float, 
                            direction: str, buffer_pct: float = 0.001) -> Optional[float]:
    """Calculate breakeven stop level"""
    try:
        move_pct = abs(current_price - entry_price) / entry_price
        if move_pct < 0.02:
            return None
        
        if direction.upper() == 'BUY':
            return round(entry_price * (1 + buffer_pct), 6)
        else:
            return round(entry_price * (1 - buffer_pct), 6)
            
    except Exception:
        return None


def calculate_time_based_exit(entry_time: pd.Timestamp, current_time: pd.Timestamp,
                             max_hold_bars: int, timeframe: str) -> Tuple[bool, str]:
    """Check if trade should exit based on time held"""
    try:
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
        }.get(timeframe, 5)
        
        bars_held = (current_time - entry_time).total_seconds() / 60 / tf_minutes
        
        if bars_held >= max_hold_bars:
            return True, f"Max hold time reached ({bars_held:.0f} bars)"
        elif bars_held >= max_hold_bars * 0.8:
            return False, f"Approaching time exit ({bars_held:.0f}/{max_hold_bars} bars)"
        else:
            return False, f"Within time limit ({bars_held:.0f}/{max_hold_bars} bars)"
            
    except Exception:
        return False, "Time exit check failed"


def calculate_partial_take_profits(entry_price: float, tp_price: float, 
                                  levels: List[Dict] = None) -> List[Dict[str, Any]]:
    """Calculate partial take profit levels"""
    if levels is None:
        levels = [
            {'pct': 0.33, 'fraction': 0.25},
            {'pct': 0.66, 'fraction': 0.35},
            {'pct': 1.0, 'fraction': 0.40}
        ]
    
    plans = []
    if entry_price is None or tp_price is None or entry_price == tp_price:
        return plans
    
    total_move = tp_price - entry_price
    
    for i, level in enumerate(levels):
        level_price = entry_price + total_move * level['pct']
        
        plans.append({
            'level': i + 1,
            'price': round(level_price, 6),
            'fraction': level['fraction'],
            'cumulative_fraction': sum(l['fraction'] for l in levels[:i+1]),
            'pct_of_move': level['pct']
        })
    
    return plans