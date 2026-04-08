# signal_filters.py - Enhanced with Vectorization
import math
import numpy as np
from typing import Dict, Any, Tuple, List
from config import Config

def filter_signal(scored: Dict[str, Any], df, features: Dict[str, Any], 
                  order_book: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
    """
    Apply conservative filters to a scored setup.
    Enhanced with vectorization and new filters.
    """
    reasons: List[str] = []
    
    # Basic safety
    if scored is None:
        return False, ["no scored setup"]

    price = float(df['close'].iloc[-1]) if df is not None and not df.empty else None
    mc = features.get('market_context', {}) if features else {}
    trend = mc.get('trend', 'NEUTRAL')
    trend_strength = float(mc.get('trend_strength', 0.0))
    atr_pct = float(mc.get('atr_pct', 0.02))
    
    # Vectorized volume check
    vol_ratio = 1.0
    current_volume = 0.0
    if Config.VECTORIZED and 'volume_ratio' in df.columns and 'volume' in df.columns:
        vol_ratio = float(df['volume_ratio'].iloc[-1])
        current_volume = float(df['volume'].iloc[-1])
    else:
        # Fallback
        if 'volume_ratio' in df and not df.empty:
            vol_ratio = float(df['volume_ratio'].iloc[-1])
        if 'volume' in df and not df.empty:
            current_volume = float(df['volume'].iloc[-1])
    
    # CRITICAL: Reject zero volume
    if current_volume <= 0:
        reasons.append("zero_volume_detected")
        return False, reasons
    
    spread = order_book.get('spread') if order_book else None
    price_zones = features.get('price_zones', {}) if features else {}
    dirn = scored.get('direction', 'BUY')

    # ===== FILTER 1: Trend Conflict =====
    if trend in ('BULL', 'BEAR'):
        if (dirn == 'BUY' and trend == 'BEAR') or (dirn == 'SELL' and trend == 'BULL'):
            if trend_strength >= 0.45:
                reasons.append(f"trend_conflict (trend={trend} strength={trend_strength:.2f})")
                return False, reasons
            else:
                reasons.append(f"weak_trend_conflict (trend={trend} strength={trend_strength:.2f})")

    # ===== FILTER 2: Volatility =====
    setup_type = scored.get('type', '').upper()
    if 'BREAKOUT' in setup_type or 'MOMENTUM' in setup_type:
        if atr_pct < Config.MIN_VOLATILITY_ATR_PCT:
            reasons.append(f"low_volatility (atr_pct={atr_pct:.4f})")
            return False, reasons

    # ===== FILTER 3: Spread =====
    if spread is not None:
        if spread > getattr(Config, 'SPREAD_THRESHOLD', 0.005) * 3:
            reasons.append(f"high_spread ({spread:.5f})")
            return False, reasons
        elif spread > getattr(Config, 'SPREAD_THRESHOLD', 0.005) * 1.5:
            reasons.append(f"wide_spread_warning ({spread:.5f})")

    # ===== FILTER 4: S/R Proximity (Vectorized) =====
    supports = np.array(price_zones.get('supports', []) or [])
    resistances = np.array(price_zones.get('resistances', []) or [])
    prox_pct = 0.008  # 0.8%
    
    if price is not None:
        if dirn == 'BUY' and len(resistances) > 0:
            # Vectorized proximity check
            distances = np.abs(resistances - price) / max(price, 1e-8)
            if np.any(distances <= prox_pct):
                reasons.append(f"resistance_proximity (nearest: {resistances[np.argmin(distances)]:.6f})")
                return False, reasons
                
        if dirn == 'SELL' and len(supports) > 0:
            distances = np.abs(supports - price) / max(price, 1e-8)
            if np.any(distances <= prox_pct):
                reasons.append(f"support_proximity (nearest: {supports[np.argmin(distances)]:.6f})")
                return False, reasons

    # ===== FILTER 5: Overextended Move =====
    try:
        roc_5 = float(df['roc_5'].iloc[-1]) if 'roc_5' in df else 0.0
        roc_10 = float(df['roc_10'].iloc[-1]) if 'roc_10' in df else 0.0
        
        if dirn == 'BUY':
            if roc_5 > 0.12 or roc_10 > 0.20:
                reasons.append(f"overextended_up (roc_5={roc_5:.3f}, roc_10={roc_10:.3f})")
                return False, reasons
        elif dirn == 'SELL':
            if roc_5 < -0.12 or roc_10 < -0.20:
                reasons.append(f"overextended_down (roc_5={roc_5:.3f}, roc_10={roc_10:.3f})")
                return False, reasons
    except Exception:
        pass

    # ===== FILTER 6: Volume Confirmation =====
    min_vol = getattr(Config, 'MIN_VOLUME_RATIO', 0.8)
    
    if vol_ratio < min_vol and scored.get('type') not in ('VWAP_MEAN_REVERT',):
        reasons.append(f"low_volume_ratio ({vol_ratio:.2f} < {min_vol})")
        return False, reasons

    # ===== FILTER 7: Grade Gating =====
    grade = scored.get('grade', 'LOW')
    if grade in ['LOW', 'D']:
        reasons.append(f"low_grade ({grade})")
        return False, reasons

    # ===== NEW FILTER 8: Liquidity Check =====
    if 'liquidity_intel' in features:
        liquidity = features['liquidity_intel']
        if liquidity.get('liquidity_score', 1.0) < 0.3:
            reasons.append("low_liquidity")
            return False, reasons

    # ===== NEW FILTER 9: Order Flow Imbalance =====
    if order_book and 'bid_ask_imbalance' in order_book:
        imbalance = order_book.get('bid_ask_imbalance', 0)
        if dirn == 'BUY' and imbalance < -0.5:
            reasons.append(f"order_flow_conflict (imbalance={imbalance:.2f})")
            return False, reasons
        elif dirn == 'SELL' and imbalance > 0.5:
            reasons.append(f"order_flow_conflict (imbalance={imbalance:.2f})")
            return False, reasons

    # ===== NEW FILTER 10: Market Regime =====
    composite_regime = mc.get('composite_regime', 'UNKNOWN')
    if 'VOLATILE' in composite_regime and 'BREAKOUT' not in setup_type:
        reasons.append("volatile_regime_caution")
        # Don't reject, just warn

    # If we reached here, pass
    return True, ["passed_filters"] + reasons