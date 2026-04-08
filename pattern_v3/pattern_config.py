# """
# pattern_config.py - Central Configuration for Pattern V4 System

# ALL hard-coded thresholds REMOVED.
# Replaced with continuous scoring parameters, dynamic weights, and learning config.

# Version: 4.0
# Author: Pattern Intelligence System
# """

# from typing import Dict, List, Tuple, Any, Optional
# from dataclasses import dataclass, field


# # ============================================================================
# # CONFIGURATION CLASS - All Settings in One Place
# # ============================================================================

# @dataclass
# class PatternConfigV4:
#     """
#     Complete configuration for Pattern V4 System.
#     NO hard-coded thresholds - only weights, scales, and learning parameters.
#     """

#     # ===== FEATURE TOGGLES =====
#     features: Dict[str, bool] = field(default_factory=lambda: {
#         'enable_liquidity_analysis': True,
#         'enable_trap_detection': True,
#         'enable_clustering': True,
#         'enable_regime_adjustment': True,
#         'enable_confidence_calibration': True,
#         'enable_time_decay': True,
#         'enable_latency_check': True,
#         'enable_fail_fast': True,
#         'enable_harmonic_patterns': True,
#         'enable_structure_patterns': True,
#         'enable_volume_patterns': True,
#         'enable_divergence_patterns': True,
#         'enable_adaptive_swings': True,
#         'enable_volume_profile': True,
#         'enable_order_block_detection': True,
#         'enable_fvg_detection': True,
#     })

#     # ===== TRAP CONFIG =====
#     trap_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'convert_traps': True,
#         'min_trap_strength': 0.60,
#         'bull_trap_boost': 1.20,
#         'bear_trap_boost': 1.20,
#         'inducement_threshold': 2.0,
#         'sweep_failure_threshold': 1.5,
#     })
    
#     # ===== CLUSTERING CONFIG =====
#     clustering_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'min_cluster_size': 2,
#         'distance_threshold': 0.02,
#         'cluster_boost_per_pattern': 0.05,
#         'max_cluster_boost': 0.15,
#         'conflict_resolution': 'strongest_wins',
#         'conflict_threshold': 0.20,
#     })
    
#     # ===== LIQUIDITY CONFIG =====
#     liquidity_config: Dict[str, Any] = field(default_factory=lambda: {
#         'require_sweep': False,
#         'min_sweep_strength': 0.60,
#         'min_liquidity_score': 0.50,
#         'max_inducements': 2,
#         'max_sweep_failures': 1,
#         'min_stop_hunt_probability': 0.30,
#         'sweep_boost_multiplier': 1.15,
#     })
    
#     # ===== SWING CONFIG =====
#     swing_config: Dict[str, Any] = field(default_factory=lambda: {
#         'method': 'adaptive',
#         'fixed_window': 3,
#         'adaptive_atr_multiplier': 0.5,
#         'min_window': 3,
#         'max_window': 10,
#         'use_volume_confirmation': True,
#         'volume_threshold': 1.2,
#         'min_swing_strength': 0.3,
#     })
    
#     # ========================================================================
#     # 1. PATTERN SIMILARITY WEIGHTS (For 23 pattern types)
#     # ========================================================================
    
#     pattern_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
#         # ===== HEAD & SHOULDERS / INVERSE H&S (7 components) =====
#         'Head_Shoulders': {
#             'shoulder_symmetry': 0.20,
#             'head_prominence': 0.18,
#             'neckline_quality': 0.12,
#             'structure_clarity': 0.15,
#             'volume_pattern': 0.15,
#             'fib_ratio': 0.12,
#             'time_symmetry': 0.08,
#         },
#         'Inverse_Head_Shoulders': {
#             'shoulder_symmetry': 0.20,
#             'head_prominence': 0.18,
#             'neckline_quality': 0.12,
#             'structure_clarity': 0.15,
#             'volume_pattern': 0.15,
#             'fib_ratio': 0.12,
#             'time_symmetry': 0.08,
#         },
        
#         # ===== DOUBLE TOP/BOTTOM (5 components) =====
#         'Double_Top': {
#             'price_similarity': 0.35,
#             'valley_strength': 0.25,
#             'volume_pattern': 0.20,
#             'time_symmetry': 0.10,
#             'breakout_volume': 0.10,
#         },
#         'Double_Bottom': {
#             'price_similarity': 0.35,
#             'valley_strength': 0.25,
#             'volume_pattern': 0.20,
#             'time_symmetry': 0.10,
#             'breakout_volume': 0.10,
#         },
        
#         # ===== TRIPLE TOP/BOTTOM (5 components) =====
#         'Triple_Top': {
#             'price_similarity': 0.30,
#             'valley_strength': 0.20,
#             'volume_pattern': 0.20,
#             'time_symmetry': 0.15,
#             'structure_clarity': 0.15,
#         },
#         'Triple_Bottom': {
#             'price_similarity': 0.30,
#             'valley_strength': 0.20,
#             'volume_pattern': 0.20,
#             'time_symmetry': 0.15,
#             'structure_clarity': 0.15,
#         },
        
#         # ===== FLAG & PENNANT (4 components) =====
#         'Flag': {
#             'pole_strength': 0.40,
#             'tightness': 0.30,
#             'volume_decrease': 0.20,
#             'breakout_volume': 0.10,
#         },
#         'Pennant': {
#             'pole_strength': 0.35,
#             'convergence': 0.35,
#             'volume_decrease': 0.20,
#             'breakout_volume': 0.10,
#         },
        
#         # ===== WEDGES (4 components) =====
#         'Rising_Wedge': {
#             'convergence': 0.40,
#             'slope_quality': 0.25,
#             'volume_pattern': 0.20,
#             'breakout_position': 0.15,
#         },
#         'Falling_Wedge': {
#             'convergence': 0.40,
#             'slope_quality': 0.25,
#             'volume_pattern': 0.20,
#             'breakout_position': 0.15,
#         },
        
#         # ===== TRIANGLES (4 components) =====
#         'Ascending_Triangle': {
#             'flat_resistance': 0.35,
#             'rising_support': 0.30,
#             'volume_contraction': 0.20,
#             'breakout_volume': 0.15,
#         },
#         'Descending_Triangle': {
#             'flat_support': 0.35,
#             'falling_resistance': 0.30,
#             'volume_contraction': 0.20,
#             'breakout_volume': 0.15,
#         },
#         'Symmetrical_Triangle': {
#             'convergence': 0.40,
#             'symmetry': 0.30,
#             'volume_contraction': 0.20,
#             'breakout_position': 0.10,
#         },
        
#         # ===== HARMONIC PATTERNS (5 components) =====
#         'Gartley': {
#             'ab_retrace': 0.25,
#             'bc_retrace': 0.20,
#             'xd_extension': 0.25,
#             'fib_accuracy': 0.20,
#             'structure_clarity': 0.10,
#         },
#         'Bat': {
#             'ab_retrace': 0.25,
#             'bc_retrace': 0.20,
#             'xd_extension': 0.25,
#             'fib_accuracy': 0.20,
#             'structure_clarity': 0.10,
#         },
#         'Crab': {
#             'ab_retrace': 0.20,
#             'bc_retrace': 0.15,
#             'xd_extension': 0.30,
#             'fib_accuracy': 0.25,
#             'structure_clarity': 0.10,
#         },
#         'Butterfly': {
#             'ab_retrace': 0.25,
#             'bc_retrace': 0.20,
#             'xd_extension': 0.25,
#             'fib_accuracy': 0.20,
#             'structure_clarity': 0.10,
#         },
#         'Three_Drives': {
#             'drive_ratio': 0.35,
#             'alternation': 0.25,
#             'rsi_divergence': 0.20,
#             'structure_clarity': 0.20,
#         },
        
#         # ===== CUP & HANDLE (4 components) =====
#         'Cup_Handle': {
#             'cup_depth': 0.30,
#             'cup_symmetry': 0.25,
#             'handle_depth': 0.25,
#             'volume_pattern': 0.20,
#         },
        
#         # ===== ADAM & EVE (4 components) =====
#         'Adam_Eve': {
#             'price_similarity': 0.35,
#             'shape_quality': 0.30,
#             'peak_strength': 0.20,
#             'volume_pattern': 0.15,
#         },
        
#         # ===== QUASIMODO (4 components) =====
#         'Quasimodo': {
#             'structure_quality': 0.35,
#             'reversal_strength': 0.30,
#             'volume_confirmation': 0.20,
#             'breakout_quality': 0.15,
#         },
        
#         # ===== VCP (4 components) =====
#         'VCP': {
#             'contraction_count': 0.30,
#             'volatility_reduction': 0.35,
#             'volume_contraction': 0.20,
#             'breakout_volume': 0.15,
#         },
        
#         # ===== WOLFE WAVE (4 components) =====
#         'Wolfe_Wave': {
#             'wave_alternation': 0.35,
#             'wave_symmetry': 0.30,
#             'target_alignment': 0.20,
#             'structure_clarity': 0.15,
#         },
        
#         # ===== DIVERGENCE (3 components) =====
#         'Divergence': {
#             'price_swing_magnitude': 0.35,
#             'rsi_divergence_strength': 0.40,
#             'volume_confirmation': 0.25,
#         },
        
#         # Default pattern
#         'DEFAULT': {
#             'structure_clarity': 0.50,
#             'volume_confirmation': 0.30,
#             'price_action': 0.20,
#         },
#     })
    
#     # ========================================================================
#     # 2. DYNAMIC CONTEXT WEIGHTS (By Market Regime)
#     # ========================================================================
    
#     dynamic_context_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
#         'STRONG_TREND': {
#             'trend_alignment': 0.40,
#             'support_resistance': 0.15,
#             'volume_confirmation': 0.20,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#         'BULL_TREND': {
#             'trend_alignment': 0.35,
#             'support_resistance': 0.20,
#             'volume_confirmation': 0.20,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#         'WEAK_BULL_TREND': {
#             'trend_alignment': 0.30,
#             'support_resistance': 0.25,
#             'volume_confirmation': 0.20,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#         'RANGING': {
#             'trend_alignment': 0.15,
#             'support_resistance': 0.40,
#             'volume_confirmation': 0.15,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.10,
#         },
#         'WEAK_BEAR_TREND': {
#             'trend_alignment': 0.30,
#             'support_resistance': 0.25,
#             'volume_confirmation': 0.20,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#         'BEAR_TREND': {
#             'trend_alignment': 0.35,
#             'support_resistance': 0.20,
#             'volume_confirmation': 0.20,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#         'STRONG_BEAR_TREND': {
#             'trend_alignment': 0.40,
#             'support_resistance': 0.15,
#             'volume_confirmation': 0.20,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#         'VOLATILE': {
#             'trend_alignment': 0.20,
#             'support_resistance': 0.15,
#             'volume_confirmation': 0.15,
#             'volatility_condition': 0.35,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#         'NEUTRAL': {
#             'trend_alignment': 0.30,
#             'support_resistance': 0.25,
#             'volume_confirmation': 0.20,
#             'volatility_condition': 0.10,
#             'liquidity_sweep': 0.10,
#             'volume_profile': 0.05,
#         },
#     })
    
#     # ========================================================================
#     # 3. PATTERN VALIDITY BY REGIME (Multipliers)
#     # ========================================================================
    
#     pattern_regime_validity: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
#         'Head_Shoulders': {
#             'STRONG_TREND': 0.60, 'BULL_TREND': 0.70, 'WEAK_BULL_TREND': 0.80,
#             'RANGING': 1.20, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
#             'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
#         },
#         'Inverse_Head_Shoulders': {
#             'STRONG_TREND': 0.65, 'BULL_TREND': 0.75, 'WEAK_BULL_TREND': 0.85,
#             'RANGING': 1.20, 'WEAK_BEAR_TREND': 0.80, 'BEAR_TREND': 0.70, 
#             'STRONG_BEAR_TREND': 0.60, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
#         },
#         'Double_Top': {
#             'STRONG_TREND': 0.60, 'BULL_TREND': 0.70, 'WEAK_BULL_TREND': 0.80,
#             'RANGING': 1.15, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
#             'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#         'Double_Bottom': {
#             'STRONG_TREND': 0.65, 'BULL_TREND': 0.75, 'WEAK_BULL_TREND': 0.85,
#             'RANGING': 1.15, 'WEAK_BEAR_TREND': 0.80, 'BEAR_TREND': 0.70, 
#             'STRONG_BEAR_TREND': 0.60, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#         'Triple_Top': {
#             'STRONG_TREND': 0.55, 'BULL_TREND': 0.65, 'WEAK_BULL_TREND': 0.75,
#             'RANGING': 1.25, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
#             'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
#         },
#         'Triple_Bottom': {
#             'STRONG_TREND': 0.60, 'BULL_TREND': 0.70, 'WEAK_BULL_TREND': 0.80,
#             'RANGING': 1.25, 'WEAK_BEAR_TREND': 0.80, 'BEAR_TREND': 0.70, 
#             'STRONG_BEAR_TREND': 0.60, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
#         },
#         'Flag': {
#             'STRONG_TREND': 1.30, 'BULL_TREND': 1.20, 'WEAK_BULL_TREND': 1.10,
#             'RANGING': 0.70, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.15, 
#             'STRONG_BEAR_TREND': 1.25, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
#         },
#         'Pennant': {
#             'STRONG_TREND': 1.25, 'BULL_TREND': 1.15, 'WEAK_BULL_TREND': 1.05,
#             'RANGING': 0.75, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.15, 
#             'STRONG_BEAR_TREND': 1.25, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
#         },
#         'Rising_Wedge': {
#             'STRONG_TREND': 0.70, 'BULL_TREND': 0.80, 'WEAK_BULL_TREND': 0.90,
#             'RANGING': 1.15, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.10, 
#             'STRONG_BEAR_TREND': 1.15, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#         'Falling_Wedge': {
#             'STRONG_TREND': 1.15, 'BULL_TREND': 1.10, 'WEAK_BULL_TREND': 1.05,
#             'RANGING': 1.15, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
#             'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#         'Ascending_Triangle': {
#             'STRONG_TREND': 1.20, 'BULL_TREND': 1.15, 'WEAK_BULL_TREND': 1.05,
#             'RANGING': 1.10, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
#             'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#         'Descending_Triangle': {
#             'STRONG_TREND': 0.70, 'BULL_TREND': 0.80, 'WEAK_BULL_TREND': 0.90,
#             'RANGING': 1.10, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.15, 
#             'STRONG_BEAR_TREND': 1.20, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#         'Symmetrical_Triangle': {
#             'STRONG_TREND': 0.90, 'BULL_TREND': 0.95, 'WEAK_BULL_TREND': 1.00,
#             'RANGING': 1.20, 'WEAK_BEAR_TREND': 1.00, 'BEAR_TREND': 0.95, 
#             'STRONG_BEAR_TREND': 0.90, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
#         },
#         'Cup_Handle': {
#             'STRONG_TREND': 1.15, 'BULL_TREND': 1.10, 'WEAK_BULL_TREND': 1.05,
#             'RANGING': 1.10, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
#             'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#         'VCP': {
#             'STRONG_TREND': 1.20, 'BULL_TREND': 1.15, 'WEAK_BULL_TREND': 1.05,
#             'RANGING': 1.00, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
#             'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
#         },
#         'Wolfe_Wave': {
#             'STRONG_TREND': 0.85, 'BULL_TREND': 0.90, 'WEAK_BULL_TREND': 0.95,
#             'RANGING': 1.20, 'WEAK_BEAR_TREND': 0.95, 'BEAR_TREND': 0.90, 
#             'STRONG_BEAR_TREND': 0.85, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
#         },
#         'Divergence': {
#             'STRONG_TREND': 0.70, 'BULL_TREND': 0.80, 'WEAK_BULL_TREND': 0.90,
#             'RANGING': 1.30, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
#             'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
#         },
#         'DEFAULT': {
#             'STRONG_TREND': 0.85, 'BULL_TREND': 0.90, 'WEAK_BULL_TREND': 0.95,
#             'RANGING': 1.10, 'WEAK_BEAR_TREND': 0.95, 'BEAR_TREND': 0.90, 
#             'STRONG_BEAR_TREND': 0.85, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
#         },
#     })
    
#     # ========================================================================
#     # 4. SIMILARITY SCALES (How quickly similarity drops)
#     # ========================================================================
    
#     similarity_scales: Dict[str, float] = field(default_factory=lambda: {
#         'price_scale': 0.05,
#         'neckline_slope_scale': 0.02,
#         'valley_depth_scale': 0.03,
#         'head_prominence_scale': 0.05,
#         'time_scale': 10,
#         'time_symmetry_scale': 0.30,
#         'volume_scale': 1.5,
#         'volume_contraction_scale': 0.5,
#         'fib_tolerance': 0.05,
#     })
    
#     # ========================================================================
#     # 5. FIBONACCI RATIOS FOR HARMONIC PATTERNS
#     # ========================================================================
    
#     fibonacci_ratios: Dict[str, float] = field(default_factory=lambda: {
#         '0.382': 0.382, '0.5': 0.5, '0.618': 0.618, '0.786': 0.786,
#         '0.886': 0.886, '1.13': 1.13, '1.27': 1.27, '1.414': 1.414,
#         '1.618': 1.618, '2.0': 2.0, '2.24': 2.24, '2.618': 2.618,
#     })
    
#     # Ideal Fibonacci ratios for each harmonic pattern
#     harmonic_ideal_ratios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
#         'Gartley': {'ab': 0.618, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 0.786},
#         'Bat': {'ab_min': 0.382, 'ab_max': 0.5, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 0.886},
#         'Crab': {'ab_min': 0.382, 'ab_max': 0.618, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 1.618},
#         'Butterfly': {'ab': 0.786, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 1.27},
#         'Three_Drives': {'drive_ratio_min': 0.5, 'drive_ratio_max': 0.9},
#     })
    
#     # ========================================================================
#     # 6. COMPLETION THRESHOLDS (For early entry)
#     # ========================================================================
    
#     completion_thresholds: Dict[str, float] = field(default_factory=lambda: {
#         'Head_Shoulders': 0.70,
#         'Inverse_Head_Shoulders': 0.70,
#         'Double_Top': 0.65,
#         'Double_Bottom': 0.65,
#         'Triple_Top': 0.70,
#         'Triple_Bottom': 0.70,
#         'Flag': 0.75,
#         'Pennant': 0.75,
#         'Rising_Wedge': 0.80,
#         'Falling_Wedge': 0.80,
#         'Ascending_Triangle': 0.80,
#         'Descending_Triangle': 0.80,
#         'Symmetrical_Triangle': 0.80,
#         'Cup_Handle': 0.70,
#         'VCP': 0.75,
#         'Wolfe_Wave': 0.70,
#         'DEFAULT': 0.60,
#     })
    
#     # ========================================================================
#     # 7. TIME DECAY CONFIGURATION
#     # ========================================================================
    
#     time_decay: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'decay_rate': 0.95,
#         'half_life_bars': 14,
#         'min_weight': 0.30,
#         'apply_to_swings': True,
#         'apply_to_volume': False,
#     })
    
#     # ========================================================================
#     # 8. MULTI-TIMEFRAME CONFLUENCE
#     # ========================================================================
    
#     mtf_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'timeframe_weights': {
#             '1m': 0.05, '5m': 0.10, '15m': 0.15,
#             '30m': 0.20, '1h': 0.25, '2h': 0.28,
#             '4h': 0.30, '1d': 0.15, '1w': 0.10,
#         },
#         'same_pattern_boost': 1.15,
#         'aligned_trend_boost': 1.10,
#         'conflicting_penalty': 0.85,
#         'max_boost': 1.30,
#         'min_boost': 0.70,
#         'required_tf_count': 2,
#     })
    
#     # ========================================================================
#     # 9. FALSE BREAKOUT DETECTION
#     # ========================================================================
    
#     false_breakout_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'volume_threshold': 1.2,
#         'wick_ratio_threshold': 0.60,
#         'reversal_bars': 2,
#         'min_breakout_volume': 1.5,
#         'penalties': {
#             'volume_too_low': -0.20,
#             'wick_too_long': -0.25,
#             'immediate_reversal': -0.30,
#             'htf_resistance_nearby': -0.15,
#             'htf_support_nearby': -0.15,
#             'low_volume_breakout': -0.15,
#         },
#         'features_to_track': [
#             'volume_too_low', 'wick_too_long', 'immediate_reversal',
#             'htf_resistance_nearby', 'htf_support_nearby'
#         ],
#     })
    
#     # ========================================================================
#     # 10. LEARNING & SELF-IMPROVEMENT
#     # ========================================================================
    
#     learning_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'min_trades_for_adjustment': 20,
#         'weight_adjustment_rate': 0.05,
#         'max_weight_multiplier': 1.30,
#         'min_weight_multiplier': 0.70,
#         'target_win_rate': 0.55,
#         'decay_old_trades': True,
#         'max_history_trades': 500,
#         'performance_file': 'pattern_performance.json',
#         'failure_analysis_enabled': True,
#         'max_failures_to_remember': 100,
#         'pattern_learning_rates': {
#             'Head_Shoulders': 0.06,
#             'Double_Top': 0.05,
#             'Flag': 0.04,
#             'DEFAULT': 0.05,
#         },
#     })
    
#     # ========================================================================
#     # 11. RISK PENALTIES
#     # ========================================================================
    
#     risk_penalties: Dict[str, float] = field(default_factory=lambda: {
#         'high_volatility': 0.05,
#         'low_liquidity': 0.05,
#         'active_trap': 0.05,
#         'conflicting_patterns': 0.05,
#         'poor_volume': 0.05,
#         'late_entry': 0.10,
#         'overextended': 0.05,
#     })
    
#     # ========================================================================
#     # 12. STOP LOSS CONFIGURATION
#     # ========================================================================
    
#     stop_loss_config: Dict[str, Any] = field(default_factory=lambda: {
#         'atr_multiplier_min': 1.2,
#         'atr_multiplier_max': 2.0,
#         'structure_stop_weight': 0.6,
#         'volatility_stop_weight': 0.3,
#         'liquidity_stop_weight': 0.1,
#         'max_stop_distance': 0.05,
#     })
    
#     # ========================================================================
#     # 13. PATTERN LIFECYCLE CONFIGURATION
#     # ========================================================================
    
#     lifecycle_config: Dict[str, Any] = field(default_factory=lambda: {
#         'time_decay_rate': 0.05,
#         'decay_half_life': 20,
#         'min_decay_factor': 0.60,
#         'breakout_confirmation_bars': 2,
#         'retest_confirmation_bars': 1,
#         'retest_tolerance': 0.005,
#         'min_formation_bars': 5,
#         'max_formation_bars': 100,
#     })
#     # Add these to your PatternConfigV4 class
# # Add these to your PatternConfigV4 class
#     min_trade_confidence: float = 0.35   # Minimum confidence to generate signal (35%)
#     min_trade_rr: float = 1.2           # Minimum risk/reward to generate signal (1.0)
#     # ========================================================================
#     # 14. DECISION THRESHOLDS
#     # ========================================================================
    
#     decision_thresholds: Dict[str, float] = field(default_factory=lambda: {
#         'min_confidence_to_trade': 0.30,
#         'min_rr_to_trade': 1.08,
#         'min_rr_for_strong': 1.08,
#         'max_pattern_age_bars': 50,
#         'max_latency_bars': 3,
#         'max_breakout_latency': 3,
#         'max_retest_latency': 1,
#         'strong_entry_threshold': 0.80,
#         'enter_now_threshold': 0.70,
#         'wait_retest_threshold': 0.60,
#     })
    
#     # ========================================================================
#     # 15. FAIL FAST CONFIGURATION
#     # ========================================================================
    
#     fail_fast_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'max_failures_per_symbol': 5,
#         'cooldown_bars_after_failure': 10,
#         'log_all_failures': True,
#         'failure_analysis_enabled': True,
#     })
    
#     # ========================================================================
#     # 16. CONFIDENCE FORMULA WEIGHTS
#     # ========================================================================
    
#     confidence_weights: Dict[str, float] = field(default_factory=lambda: {
#         'similarity_weight': 0.50,
#         'context_weight': 0.30,
#         'mtf_weight': 0.10,
#         'regime_weight': 0.05,
#         'learning_weight': 0.05,
#     })
    
#     # ========================================================================
#     # 17. ACTION THRESHOLDS (Based on final confidence)
#     # ========================================================================
    
#     action_thresholds: Dict[str, float] = field(default_factory=lambda: {
#         'strong_entry': 0.69,
#         'enter_now': 0.65,
#         'wait_retest': 0.30,
#         'skip': 0.20,
#     })
    
#     # ========================================================================
#     # 18. GRADE THRESHOLDS
#     # ========================================================================
    
#     grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
#         'A+': 0.82, 'A': 0.77, 'B+': 0.72, 'B': 0.67,
#         'B-': 0.62, 'C+': 0.57, 'C': 0.52, 'D': 0.32, 'F': 0.20
#     })
    
#     # ========================================================================
#     # 19. POSITION MULTIPLIERS BY GRADE
#     # ========================================================================
    
#     position_multipliers: Dict[str, float] = field(default_factory=lambda: {
#         'A+': 1.6, 'A': 1.4, 'B+': 1.2, 'B': 1.0,
#         'B-': 0.9, 'C+': 0.8, 'C': 0.6, 'D': 0.3, 'F': 0.0
#     })
    
#     # ========================================================================
#     # 20. INVALIDATION THRESHOLDS (When to reject pattern)
#     # ========================================================================
    
#     invalidation_config: Dict[str, Any] = field(default_factory=lambda: {
#         'min_structure_clarity': 0.40,
#         'min_swing_count': 5,
#         'max_wick_ratio': 0.60,
#         'min_formation_bars': 5,
#         'max_confidence_degradation': 0.20,
#         'max_age_bars': 100,
#     })
    
#     # ========================================================================
#     # 21. VOLUME PROFILE CONFIGURATION
#     # ========================================================================
    
#     volume_profile_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enabled': True,
#         'bins': 20,
#         'lookback_bars': 100,
#         'poc_strength_threshold': 0.7,
#         'high_volume_node_threshold': 0.8,
#         'low_volume_node_threshold': 0.2,
#     })
    
#     # ========================================================================
#     # 22. PERFORMANCE & LOGGING
#     # ========================================================================
    
#     performance_config: Dict[str, Any] = field(default_factory=lambda: {
#         'enable_debug_logging': True,
#         'enable_detailed_metadata': True,
#         'max_patterns_per_symbol': 20,
#         'log_similarity_components': True,
#         'log_context_components': True,
#         'log_mtf_details': True,
#         'cache_results': True,
#         'cache_ttl_seconds': 60,
#     })


# # ============================================================================
# # GLOBAL CONFIGURATION INSTANCE
# # ============================================================================

# CONFIG = PatternConfigV4()


# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================

# def get_config() -> PatternConfigV4:
#     """Get the global configuration instance"""
#     return CONFIG


# def get_pattern_weights(pattern_name: str) -> Dict[str, float]:
#     """Get similarity weights for a specific pattern"""
#     weights = CONFIG.pattern_weights.get(pattern_name)
#     if weights is None:
#         for key in CONFIG.pattern_weights:
#             if key.lower() in pattern_name.lower() or pattern_name.lower() in key.lower():
#                 return CONFIG.pattern_weights[key]
#     return weights or CONFIG.pattern_weights.get('DEFAULT', {'structure_clarity': 1.0})


# def get_dynamic_context_weights(regime: str) -> Dict[str, float]:
#     """Get context weights for a specific market regime"""
#     return CONFIG.dynamic_context_weights.get(regime, CONFIG.dynamic_context_weights['NEUTRAL'])


# def get_pattern_regime_validity(pattern_name: str, regime: str) -> float:
#     """Get regime validity multiplier for a pattern"""
#     pattern_validity = CONFIG.pattern_regime_validity.get(pattern_name)
#     if pattern_validity is None:
#         pattern_validity = CONFIG.pattern_regime_validity.get('DEFAULT', {})
#     return pattern_validity.get(regime, 1.0)


# def get_completion_threshold(pattern_name: str) -> float:
#     """Get completion threshold for early entry"""
#     return CONFIG.completion_thresholds.get(pattern_name, CONFIG.completion_thresholds['DEFAULT'])


# def get_mtf_weight(timeframe: str) -> float:
#     """Get multi-timeframe weight for a specific timeframe"""
#     return CONFIG.mtf_config['timeframe_weights'].get(timeframe, 0.10)


# def get_false_breakout_penalty(feature: str) -> float:
#     """Get penalty for a specific false breakout feature"""
#     return CONFIG.false_breakout_config['penalties'].get(feature, -0.10)


# def get_grade(confidence: float) -> str:
#     """Get grade based on confidence"""
#     for grade, threshold in sorted(CONFIG.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
#         if confidence >= threshold:
#             return grade
#     return 'F'


# def get_position_multiplier(grade: str) -> float:
#     """Get position multiplier based on grade"""
#     return CONFIG.position_multipliers.get(grade, 1.0)


# def get_action(confidence: float, completion_pct: float = 1.0) -> str:
#     """Get action based on confidence and completion"""
#     thresholds = CONFIG.action_thresholds
    
#     if confidence >= thresholds['strong_entry']:
#         return "STRONG_ENTRY"
#     elif confidence >= thresholds['enter_now']:
#         if completion_pct >= 0.95:
#             return "ENTER_NOW"
#         else:
#             return "WAIT_FOR_RETEST"
#     elif confidence >= thresholds['wait_retest']:
#         return "WAIT_FOR_RETEST"
#     else:
#         return "SKIP"


# def validate_config() -> List[str]:
#     """Validate configuration and return any issues"""
#     issues = []
    
#     # Check pattern weights sum to approximately 1.0 for each pattern
#     for pattern_name, weights in CONFIG.pattern_weights.items():
#         total = sum(weights.values())
#         if abs(total - 1.0) > 0.05:
#             issues.append(f"Pattern {pattern_name} weights sum to {total:.2f}, should be 1.0")
    
#     # Check context weights sum to 1.0 for each regime
#     for regime, weights in CONFIG.dynamic_context_weights.items():
#         total = sum(weights.values())
#         if abs(total - 1.0) > 0.01:
#             issues.append(f"Regime {regime} context weights sum to {total:.2f}, should be 1.0")
    
#     # Check confidence weights sum to 1.0
#     conf_total = sum(CONFIG.confidence_weights.values())
#     if abs(conf_total - 1.0) > 0.01:
#         issues.append(f"Confidence weights sum to {conf_total:.2f}, should be 1.0")
    
#     # Check grade thresholds are descending
#     prev = 1.0
#     for grade, threshold in sorted(CONFIG.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
#         if threshold > prev:
#             issues.append(f"Grade {grade} threshold {threshold} > previous {prev}")
#         prev = threshold
    
#     return issues


# # ============================================================================
# # EXPORTS
# # ============================================================================

# __all__ = [
#     'PatternConfigV4',
#     'CONFIG',
#     'get_config',
#     'get_pattern_weights',
#     'get_dynamic_context_weights',
#     'get_pattern_regime_validity',
#     'get_completion_threshold',
#     'get_mtf_weight',
#     'get_false_breakout_penalty',
#     'get_grade',
#     'get_position_multiplier',
#     'get_action',
#     'validate_config',
# ]


































































"""
pattern_config.py - Central Configuration for Pattern V4 System

ALL hard-coded thresholds REMOVED.
Replaced with continuous scoring parameters, dynamic weights, and learning config.

Version: 4.0
Author: Pattern Intelligence System
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


# ============================================================================
# CONFIGURATION CLASS - All Settings in One Place
# ============================================================================

@dataclass
class PatternConfigV4:
    """
    Complete configuration for Pattern V4 System.
    NO hard-coded thresholds - only weights, scales, and learning parameters.
    """

    # ===== FEATURE TOGGLES =====
    features: Dict[str, bool] = field(default_factory=lambda: {
        'enable_liquidity_analysis': True,
        'enable_trap_detection': True,
        'enable_clustering': True,
        'enable_regime_adjustment': True,
        'enable_confidence_calibration': True,
        'enable_time_decay': True,
        'enable_latency_check': True,
        'enable_fail_fast': True,
        'enable_harmonic_patterns': True,
        'enable_structure_patterns': True,
        'enable_volume_patterns': True,
        'enable_divergence_patterns': True,
        'enable_adaptive_swings': True,
        'enable_volume_profile': True,
        'enable_order_block_detection': True,
        'enable_fvg_detection': True,
    })

    # ===== TRAP CONFIG =====
    trap_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'convert_traps': True,
        'min_trap_strength': 0.60,
        'bull_trap_boost': 1.20,
        'bear_trap_boost': 1.20,
        'inducement_threshold': 2.0,
        'sweep_failure_threshold': 1.5,
    })
    
    # ===== CLUSTERING CONFIG =====
    clustering_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'min_cluster_size': 2,
        'distance_threshold': 0.02,
        'cluster_boost_per_pattern': 0.05,
        'max_cluster_boost': 0.15,
        'conflict_resolution': 'strongest_wins',
        'conflict_threshold': 0.20,
    })
    
    # ===== LIQUIDITY CONFIG =====
    liquidity_config: Dict[str, Any] = field(default_factory=lambda: {
        'require_sweep': False,
        'min_sweep_strength': 0.60,
        'min_liquidity_score': 0.50,
        'max_inducements': 2,
        'max_sweep_failures': 1,
        'min_stop_hunt_probability': 0.30,
        'sweep_boost_multiplier': 1.15,
    })
    
    # ===== SWING CONFIG =====
    swing_config: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'adaptive',
        'fixed_window': 3,
        'adaptive_atr_multiplier': 0.5,
        'min_window': 3,
        'max_window': 10,
        'use_volume_confirmation': True,
        'volume_threshold': 1.2,
        'min_swing_strength': 0.3,
    })
    
    # ========================================================================
    # 1. PATTERN SIMILARITY WEIGHTS (For 23 pattern types)
    # ========================================================================
    
    pattern_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # ===== HEAD & SHOULDERS / INVERSE H&S (7 components) =====
        'Head_Shoulders': {
            'shoulder_symmetry': 0.20,
            'head_prominence': 0.18,
            'neckline_quality': 0.12,
            'structure_clarity': 0.15,
            'volume_pattern': 0.15,
            'fib_ratio': 0.12,
            'time_symmetry': 0.08,
        },
        'Inverse_Head_Shoulders': {
            'shoulder_symmetry': 0.20,
            'head_prominence': 0.18,
            'neckline_quality': 0.12,
            'structure_clarity': 0.15,
            'volume_pattern': 0.15,
            'fib_ratio': 0.12,
            'time_symmetry': 0.08,
        },
        
        # ===== DOUBLE TOP/BOTTOM (5 components) =====
        'Double_Top': {
            'price_similarity': 0.35,
            'valley_strength': 0.25,
            'volume_pattern': 0.20,
            'time_symmetry': 0.10,
            'breakout_volume': 0.10,
        },
        'Double_Bottom': {
            'price_similarity': 0.35,
            'valley_strength': 0.25,
            'volume_pattern': 0.20,
            'time_symmetry': 0.10,
            'breakout_volume': 0.10,
        },
        
        # ===== TRIPLE TOP/BOTTOM (5 components) =====
        'Triple_Top': {
            'price_similarity': 0.30,
            'valley_strength': 0.20,
            'volume_pattern': 0.20,
            'time_symmetry': 0.15,
            'structure_clarity': 0.15,
        },
        'Triple_Bottom': {
            'price_similarity': 0.30,
            'valley_strength': 0.20,
            'volume_pattern': 0.20,
            'time_symmetry': 0.15,
            'structure_clarity': 0.15,
        },
        
        # ===== FLAG & PENNANT (4 components) =====
        'Flag': {
            'pole_strength': 0.40,
            'tightness': 0.30,
            'volume_decrease': 0.20,
            'breakout_volume': 0.10,
        },
        'Pennant': {
            'pole_strength': 0.35,
            'convergence': 0.35,
            'volume_decrease': 0.20,
            'breakout_volume': 0.10,
        },
        
        # ===== WEDGES (4 components) =====
        'Rising_Wedge': {
            'convergence': 0.40,
            'slope_quality': 0.25,
            'volume_pattern': 0.20,
            'breakout_position': 0.15,
        },
        'Falling_Wedge': {
            'convergence': 0.40,
            'slope_quality': 0.25,
            'volume_pattern': 0.20,
            'breakout_position': 0.15,
        },
        
        # ===== TRIANGLES (4 components) =====
        'Ascending_Triangle': {
            'flat_resistance': 0.35,
            'rising_support': 0.30,
            'volume_contraction': 0.20,
            'breakout_volume': 0.15,
        },
        'Descending_Triangle': {
            'flat_support': 0.35,
            'falling_resistance': 0.30,
            'volume_contraction': 0.20,
            'breakout_volume': 0.15,
        },
        'Symmetrical_Triangle': {
            'convergence': 0.40,
            'symmetry': 0.30,
            'volume_contraction': 0.20,
            'breakout_position': 0.10,
        },
        
        # ===== HARMONIC PATTERNS (5 components) =====
        'Gartley': {
            'ab_retrace': 0.25,
            'bc_retrace': 0.20,
            'xd_extension': 0.25,
            'fib_accuracy': 0.20,
            'structure_clarity': 0.10,
        },
        'Bat': {
            'ab_retrace': 0.25,
            'bc_retrace': 0.20,
            'xd_extension': 0.25,
            'fib_accuracy': 0.20,
            'structure_clarity': 0.10,
        },
        'Crab': {
            'ab_retrace': 0.20,
            'bc_retrace': 0.15,
            'xd_extension': 0.30,
            'fib_accuracy': 0.25,
            'structure_clarity': 0.10,
        },
        'Butterfly': {
            'ab_retrace': 0.25,
            'bc_retrace': 0.20,
            'xd_extension': 0.25,
            'fib_accuracy': 0.20,
            'structure_clarity': 0.10,
        },
        'Three_Drives': {
            'drive_ratio': 0.35,
            'alternation': 0.25,
            'rsi_divergence': 0.20,
            'structure_clarity': 0.20,
        },
        
        # ===== CUP & HANDLE (4 components) =====
        'Cup_Handle': {
            'cup_depth': 0.30,
            'cup_symmetry': 0.25,
            'handle_depth': 0.25,
            'volume_pattern': 0.20,
        },
        
        # ===== ADAM & EVE (4 components) =====
        'Adam_Eve': {
            'price_similarity': 0.35,
            'shape_quality': 0.30,
            'peak_strength': 0.20,
            'volume_pattern': 0.15,
        },
        
        # ===== QUASIMODO (4 components) =====
        'Quasimodo': {
            'structure_quality': 0.35,
            'reversal_strength': 0.30,
            'volume_confirmation': 0.20,
            'breakout_quality': 0.15,
        },
        
        # ===== VCP (4 components) =====
        'VCP': {
            'contraction_count': 0.30,
            'volatility_reduction': 0.35,
            'volume_contraction': 0.20,
            'breakout_volume': 0.15,
        },
        
        # ===== WOLFE WAVE (4 components) =====
        'Wolfe_Wave': {
            'wave_alternation': 0.35,
            'wave_symmetry': 0.30,
            'target_alignment': 0.20,
            'structure_clarity': 0.15,
        },
        
        # ===== DIVERGENCE (3 components) =====
        'Divergence': {
            'price_swing_magnitude': 0.35,
            'rsi_divergence_strength': 0.40,
            'volume_confirmation': 0.25,
        },
        
        # Default pattern
        'DEFAULT': {
            'structure_clarity': 0.50,
            'volume_confirmation': 0.30,
            'price_action': 0.20,
        },
    })
    
    # ========================================================================
    # 2. DYNAMIC CONTEXT WEIGHTS (By Market Regime)
    # ========================================================================
    
    dynamic_context_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'STRONG_TREND': {
            'trend_alignment': 0.40,
            'support_resistance': 0.15,
            'volume_confirmation': 0.20,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
        'BULL_TREND': {
            'trend_alignment': 0.35,
            'support_resistance': 0.20,
            'volume_confirmation': 0.20,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
        'WEAK_BULL_TREND': {
            'trend_alignment': 0.30,
            'support_resistance': 0.25,
            'volume_confirmation': 0.20,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
        'RANGING': {
            'trend_alignment': 0.15,
            'support_resistance': 0.40,
            'volume_confirmation': 0.15,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.10,
        },
        'WEAK_BEAR_TREND': {
            'trend_alignment': 0.30,
            'support_resistance': 0.25,
            'volume_confirmation': 0.20,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
        'BEAR_TREND': {
            'trend_alignment': 0.35,
            'support_resistance': 0.20,
            'volume_confirmation': 0.20,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
        'STRONG_BEAR_TREND': {
            'trend_alignment': 0.40,
            'support_resistance': 0.15,
            'volume_confirmation': 0.20,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
        'VOLATILE': {
            'trend_alignment': 0.20,
            'support_resistance': 0.15,
            'volume_confirmation': 0.15,
            'volatility_condition': 0.35,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
        'NEUTRAL': {
            'trend_alignment': 0.30,
            'support_resistance': 0.25,
            'volume_confirmation': 0.20,
            'volatility_condition': 0.10,
            'liquidity_sweep': 0.10,
            'volume_profile': 0.05,
        },
    })
    
    # ========================================================================
    # 3. PATTERN VALIDITY BY REGIME (Multipliers)
    # ========================================================================
    
    pattern_regime_validity: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Head_Shoulders': {
            'STRONG_TREND': 0.60, 'BULL_TREND': 0.70, 'WEAK_BULL_TREND': 0.80,
            'RANGING': 1.20, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
            'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
        },
        'Inverse_Head_Shoulders': {
            'STRONG_TREND': 0.65, 'BULL_TREND': 0.75, 'WEAK_BULL_TREND': 0.85,
            'RANGING': 1.20, 'WEAK_BEAR_TREND': 0.80, 'BEAR_TREND': 0.70, 
            'STRONG_BEAR_TREND': 0.60, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
        },
        'Double_Top': {
            'STRONG_TREND': 0.60, 'BULL_TREND': 0.70, 'WEAK_BULL_TREND': 0.80,
            'RANGING': 1.15, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
            'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
        'Double_Bottom': {
            'STRONG_TREND': 0.65, 'BULL_TREND': 0.75, 'WEAK_BULL_TREND': 0.85,
            'RANGING': 1.15, 'WEAK_BEAR_TREND': 0.80, 'BEAR_TREND': 0.70, 
            'STRONG_BEAR_TREND': 0.60, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
        'Triple_Top': {
            'STRONG_TREND': 0.55, 'BULL_TREND': 0.65, 'WEAK_BULL_TREND': 0.75,
            'RANGING': 1.25, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
            'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
        },
        'Triple_Bottom': {
            'STRONG_TREND': 0.60, 'BULL_TREND': 0.70, 'WEAK_BULL_TREND': 0.80,
            'RANGING': 1.25, 'WEAK_BEAR_TREND': 0.80, 'BEAR_TREND': 0.70, 
            'STRONG_BEAR_TREND': 0.60, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
        },
        'Flag': {
            'STRONG_TREND': 1.30, 'BULL_TREND': 1.20, 'WEAK_BULL_TREND': 1.10,
            'RANGING': 0.70, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.15, 
            'STRONG_BEAR_TREND': 1.25, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
        },
        'Pennant': {
            'STRONG_TREND': 1.25, 'BULL_TREND': 1.15, 'WEAK_BULL_TREND': 1.05,
            'RANGING': 0.75, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.15, 
            'STRONG_BEAR_TREND': 1.25, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
        },
        'Rising_Wedge': {
            'STRONG_TREND': 0.70, 'BULL_TREND': 0.80, 'WEAK_BULL_TREND': 0.90,
            'RANGING': 1.15, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.10, 
            'STRONG_BEAR_TREND': 1.15, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
        'Falling_Wedge': {
            'STRONG_TREND': 1.15, 'BULL_TREND': 1.10, 'WEAK_BULL_TREND': 1.05,
            'RANGING': 1.15, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
            'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
        'Ascending_Triangle': {
            'STRONG_TREND': 1.20, 'BULL_TREND': 1.15, 'WEAK_BULL_TREND': 1.05,
            'RANGING': 1.10, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
            'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
        'Descending_Triangle': {
            'STRONG_TREND': 0.70, 'BULL_TREND': 0.80, 'WEAK_BULL_TREND': 0.90,
            'RANGING': 1.10, 'WEAK_BEAR_TREND': 1.05, 'BEAR_TREND': 1.15, 
            'STRONG_BEAR_TREND': 1.20, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
        'Symmetrical_Triangle': {
            'STRONG_TREND': 0.90, 'BULL_TREND': 0.95, 'WEAK_BULL_TREND': 1.00,
            'RANGING': 1.20, 'WEAK_BEAR_TREND': 1.00, 'BEAR_TREND': 0.95, 
            'STRONG_BEAR_TREND': 0.90, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
        },
        'Cup_Handle': {
            'STRONG_TREND': 1.15, 'BULL_TREND': 1.10, 'WEAK_BULL_TREND': 1.05,
            'RANGING': 1.10, 'WEAK_BEAR_TREND': 0.85, 'BEAR_TREND': 0.75, 
            'STRONG_BEAR_TREND': 0.65, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
        'VCP': {
            'STRONG_TREND': 1.20, 'BULL_TREND': 1.15, 'WEAK_BULL_TREND': 1.05,
            'RANGING': 1.00, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
            'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
        },
        'Wolfe_Wave': {
            'STRONG_TREND': 0.85, 'BULL_TREND': 0.90, 'WEAK_BULL_TREND': 0.95,
            'RANGING': 1.20, 'WEAK_BEAR_TREND': 0.95, 'BEAR_TREND': 0.90, 
            'STRONG_BEAR_TREND': 0.85, 'VOLATILE': 0.80, 'NEUTRAL': 1.00,
        },
        'Divergence': {
            'STRONG_TREND': 0.70, 'BULL_TREND': 0.80, 'WEAK_BULL_TREND': 0.90,
            'RANGING': 1.30, 'WEAK_BEAR_TREND': 0.90, 'BEAR_TREND': 0.80, 
            'STRONG_BEAR_TREND': 0.70, 'VOLATILE': 0.90, 'NEUTRAL': 1.00,
        },
        'DEFAULT': {
            'STRONG_TREND': 0.85, 'BULL_TREND': 0.90, 'WEAK_BULL_TREND': 0.95,
            'RANGING': 1.10, 'WEAK_BEAR_TREND': 0.95, 'BEAR_TREND': 0.90, 
            'STRONG_BEAR_TREND': 0.85, 'VOLATILE': 0.85, 'NEUTRAL': 1.00,
        },
    })
    
    # ========================================================================
    # 4. SIMILARITY SCALES (How quickly similarity drops)
    # ========================================================================
    
    similarity_scales: Dict[str, float] = field(default_factory=lambda: {
        'price_scale': 0.05,
        'neckline_slope_scale': 0.02,
        'valley_depth_scale': 0.03,
        'head_prominence_scale': 0.05,
        'time_scale': 10,
        'time_symmetry_scale': 0.30,
        'volume_scale': 1.5,
        'volume_contraction_scale': 0.5,
        'fib_tolerance': 0.05,
    })
    
    # ========================================================================
    # 5. FIBONACCI RATIOS FOR HARMONIC PATTERNS
    # ========================================================================
    
    fibonacci_ratios: Dict[str, float] = field(default_factory=lambda: {
        '0.382': 0.382, '0.5': 0.5, '0.618': 0.618, '0.786': 0.786,
        '0.886': 0.886, '1.13': 1.13, '1.27': 1.27, '1.414': 1.414,
        '1.618': 1.618, '2.0': 2.0, '2.24': 2.24, '2.618': 2.618,
    })
    
    # Ideal Fibonacci ratios for each harmonic pattern
    harmonic_ideal_ratios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Gartley': {'ab': 0.618, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 0.786},
        'Bat': {'ab_min': 0.382, 'ab_max': 0.5, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 0.886},
        'Crab': {'ab_min': 0.382, 'ab_max': 0.618, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 1.618},
        'Butterfly': {'ab': 0.786, 'bc_min': 0.382, 'bc_max': 0.886, 'xd': 1.27},
        'Three_Drives': {'drive_ratio_min': 0.5, 'drive_ratio_max': 0.9},
    })
    
    # ========================================================================
    # 6. COMPLETION THRESHOLDS (For early entry)
    # ========================================================================
    
    completion_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'Head_Shoulders': 0.70,
        'Inverse_Head_Shoulders': 0.70,
        'Double_Top': 0.65,
        'Double_Bottom': 0.65,
        'Triple_Top': 0.70,
        'Triple_Bottom': 0.70,
        'Flag': 0.75,
        'Pennant': 0.75,
        'Rising_Wedge': 0.80,
        'Falling_Wedge': 0.80,
        'Ascending_Triangle': 0.80,
        'Descending_Triangle': 0.80,
        'Symmetrical_Triangle': 0.80,
        'Cup_Handle': 0.70,
        'VCP': 0.75,
        'Wolfe_Wave': 0.70,
        'DEFAULT': 0.60,
    })
    
    # ========================================================================
    # 7. TIME DECAY CONFIGURATION
    # ========================================================================
    
    time_decay: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'decay_rate': 0.95,
        'half_life_bars': 14,
        'min_weight': 0.30,
        'apply_to_swings': True,
        'apply_to_volume': False,
    })
    
    # ========================================================================
    # 8. MULTI-TIMEFRAME CONFLUENCE
    # ========================================================================
    
    mtf_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'timeframe_weights': {
            '1m': 0.05, '5m': 0.10, '15m': 0.15,
            '30m': 0.20, '1h': 0.25, '2h': 0.28,
            '4h': 0.30, '1d': 0.15, '1w': 0.10,
        },
        'same_pattern_boost': 1.15,
        'aligned_trend_boost': 1.10,
        'conflicting_penalty': 0.85,
        'max_boost': 1.30,
        'min_boost': 0.70,
        'required_tf_count': 2,
    })
    
    # ========================================================================
    # 9. FALSE BREAKOUT DETECTION
    # ========================================================================
    
    false_breakout_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'volume_threshold': 1.2,
        'wick_ratio_threshold': 0.60,
        'reversal_bars': 2,
        'min_breakout_volume': 1.5,
        'penalties': {
            'volume_too_low': -0.10,
            'wick_too_long': -0.20,
            'immediate_reversal': -0.25,
            'htf_resistance_nearby': -0.15,
            'htf_support_nearby': -0.15,
            'low_volume_breakout': -0.15,
        },
        'features_to_track': [
            'volume_too_low', 'wick_too_long', 'immediate_reversal',
            'htf_resistance_nearby', 'htf_support_nearby'
        ],
    })
    
    # ========================================================================
    # 10. LEARNING & SELF-IMPROVEMENT
    # ========================================================================
    
    learning_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'min_trades_for_adjustment': 20,
        'weight_adjustment_rate': 0.05,
        'max_weight_multiplier': 1.30,
        'min_weight_multiplier': 0.70,
        'target_win_rate': 0.55,
        'decay_old_trades': True,
        'max_history_trades': 500,
        'performance_file': 'pattern_performance.json',
        'failure_analysis_enabled': True,
        'max_failures_to_remember': 100,
        'pattern_learning_rates': {
            'Head_Shoulders': 0.06,
            'Double_Top': 0.05,
            'Flag': 0.04,
            'DEFAULT': 0.05,
        },
    })
    
    # ========================================================================
    # 11. RISK PENALTIES
    # ========================================================================
    
    risk_penalties: Dict[str, float] = field(default_factory=lambda: {
        'high_volatility': 0.05,
        'low_liquidity': 0.05,
        'active_trap': 0.05,
        'conflicting_patterns': 0.05,
        'poor_volume': 0.05,
        'late_entry': 0.10,
        'overextended': 0.05,
    })
    
    # ========================================================================
    # 12. STOP LOSS CONFIGURATION
    # ========================================================================
    
    stop_loss_config: Dict[str, Any] = field(default_factory=lambda: {
        'atr_multiplier_min': 1.2,
        'atr_multiplier_max': 2.0,
        'structure_stop_weight': 0.6,
        'volatility_stop_weight': 0.3,
        'liquidity_stop_weight': 0.1,
        'max_stop_distance': 0.05,
    })
    
    # ========================================================================
    # 13. PATTERN LIFECYCLE CONFIGURATION
    # ========================================================================
    
    lifecycle_config: Dict[str, Any] = field(default_factory=lambda: {
        'time_decay_rate': 0.05,
        'decay_half_life': 20,
        'min_decay_factor': 0.60,
        'breakout_confirmation_bars': 2,
        'retest_confirmation_bars': 1,
        'retest_tolerance': 0.005,
        'min_formation_bars': 5,
        'max_formation_bars': 100,
    })

    # ========================================================================
    # 14. TRADE FILTER THRESHOLDS (only things that block a signal)
    # ========================================================================
    min_trade_confidence: float = 0.35   # Minimum confidence to generate signal
    min_trade_rr: float = 1.2            # Minimum risk/reward to generate signal

    # ========================================================================
    # 14b. DECISION THRESHOLDS
    # ========================================================================
    
    decision_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_confidence_to_trade': 0.30,
        'min_rr_to_trade': 1.08,
        'min_rr_for_strong': 1.08,
        'max_pattern_age_bars': 50,
        'max_latency_bars': 3,
        'max_breakout_latency': 3,
        'max_retest_latency': 1,
        'strong_entry_threshold': 0.80,
        'enter_now_threshold': 0.70,
        'wait_retest_threshold': 0.60,
    })
    
    # ========================================================================
    # 15. FAIL FAST CONFIGURATION
    # ========================================================================
    
    fail_fast_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'max_failures_per_symbol': 5,
        'cooldown_bars_after_failure': 10,
        'log_all_failures': True,
        'failure_analysis_enabled': True,
    })
    
    # ========================================================================
    # 16. CONFIDENCE FORMULA WEIGHTS
    # ========================================================================
    
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'similarity_weight': 0.50,
        'context_weight': 0.30,
        'mtf_weight': 0.10,
        'regime_weight': 0.05,
        'learning_weight': 0.05,
    })
    
    # ========================================================================
    # 17. ACTION THRESHOLDS (Based on final confidence)
    # ========================================================================
    
    action_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'strong_entry': 0.69,
        'enter_now': 0.65,
        'wait_retest': 0.30,
        'skip': 0.20,
    })
    
    # ========================================================================
    # 18. GRADE THRESHOLDS
    # ========================================================================
    
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'A+': 0.82, 'A': 0.77, 'B+': 0.72, 'B': 0.67,
        'B-': 0.62, 'C+': 0.57, 'C': 0.52, 'D': 0.32, 'F': 0.15
    })
    
    # ========================================================================
    # 19. POSITION MULTIPLIERS BY GRADE
    # ========================================================================
    
    position_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'A+': 1.6, 'A': 1.4, 'B+': 1.2, 'B': 1.0,
        'B-': 0.9, 'C+': 0.8, 'C': 0.6, 'D': 0.3, 'F': 0.12
    })
    
    # ========================================================================
    # 20. INVALIDATION THRESHOLDS (When to reject pattern)
    # ========================================================================
    
    invalidation_config: Dict[str, Any] = field(default_factory=lambda: {
        'min_structure_clarity': 0.40,
        'min_swing_count': 5,
        'max_wick_ratio': 0.60,
        'min_formation_bars': 5,
        'max_confidence_degradation': 0.20,
        'max_age_bars': 100,
    })
    
    # ========================================================================
    # 21. VOLUME PROFILE CONFIGURATION
    # ========================================================================
    
    volume_profile_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'bins': 20,
        'lookback_bars': 100,
        'poc_strength_threshold': 0.7,
        'high_volume_node_threshold': 0.8,
        'low_volume_node_threshold': 0.2,
    })
    
    # ========================================================================
    # 22. PERFORMANCE & LOGGING
    # ========================================================================
    
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_debug_logging': True,
        'enable_detailed_metadata': True,
        'max_patterns_per_symbol': 20,
        'log_similarity_components': True,
        'log_context_components': True,
        'log_mtf_details': True,
        'cache_results': True,
        'cache_ttl_seconds': 60,
    })


# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

CONFIG = PatternConfigV4()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config() -> PatternConfigV4:
    """Get the global configuration instance"""
    return CONFIG


def get_pattern_weights(pattern_name: str) -> Dict[str, float]:
    """Get similarity weights for a specific pattern"""
    weights = CONFIG.pattern_weights.get(pattern_name)
    if weights is None:
        for key in CONFIG.pattern_weights:
            if key.lower() in pattern_name.lower() or pattern_name.lower() in key.lower():
                return CONFIG.pattern_weights[key]
    return weights or CONFIG.pattern_weights.get('DEFAULT', {'structure_clarity': 1.0})


def get_dynamic_context_weights(regime: str) -> Dict[str, float]:
    """Get context weights for a specific market regime"""
    return CONFIG.dynamic_context_weights.get(regime, CONFIG.dynamic_context_weights['NEUTRAL'])


def get_pattern_regime_validity(pattern_name: str, regime: str) -> float:
    """Get regime validity multiplier for a pattern"""
    pattern_validity = CONFIG.pattern_regime_validity.get(pattern_name)
    if pattern_validity is None:
        pattern_validity = CONFIG.pattern_regime_validity.get('DEFAULT', {})
    return pattern_validity.get(regime, 1.0)


def get_completion_threshold(pattern_name: str) -> float:
    """Get completion threshold for early entry"""
    return CONFIG.completion_thresholds.get(pattern_name, CONFIG.completion_thresholds['DEFAULT'])


def get_mtf_weight(timeframe: str) -> float:
    """Get multi-timeframe weight for a specific timeframe"""
    return CONFIG.mtf_config['timeframe_weights'].get(timeframe, 0.10)


def get_false_breakout_penalty(feature: str) -> float:
    """Get penalty for a specific false breakout feature"""
    return CONFIG.false_breakout_config['penalties'].get(feature, -0.10)


def get_grade(confidence: float) -> str:
    """Get grade based on confidence"""
    for grade, threshold in sorted(CONFIG.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
        if confidence >= threshold:
            return grade
    return 'F'


def get_position_multiplier(grade: str) -> float:
    """Get position multiplier based on grade"""
    return CONFIG.position_multipliers.get(grade, 1.0)


def get_action(confidence: float, completion_pct: float = 1.0) -> str:
    """Get action based on confidence and completion"""
    thresholds = CONFIG.action_thresholds
    
    if confidence >= thresholds['strong_entry']:
        return "STRONG_ENTRY"
    elif confidence >= thresholds['enter_now']:
        if completion_pct >= 0.95:
            return "ENTER_NOW"
        else:
            return "WAIT_FOR_RETEST"
    elif confidence >= thresholds['wait_retest']:
        return "WAIT_FOR_RETEST"
    else:
        return "SKIP"


def validate_config() -> List[str]:
    """Validate configuration and return any issues"""
    issues = []
    
    # Check pattern weights sum to approximately 1.0 for each pattern
    for pattern_name, weights in CONFIG.pattern_weights.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 0.05:
            issues.append(f"Pattern {pattern_name} weights sum to {total:.2f}, should be 1.0")
    
    # Check context weights sum to 1.0 for each regime
    for regime, weights in CONFIG.dynamic_context_weights.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            issues.append(f"Regime {regime} context weights sum to {total:.2f}, should be 1.0")
    
    # Check confidence weights sum to 1.0
    conf_total = sum(CONFIG.confidence_weights.values())
    if abs(conf_total - 1.0) > 0.01:
        issues.append(f"Confidence weights sum to {conf_total:.2f}, should be 1.0")
    
    # Check grade thresholds are descending
    prev = 1.0
    for grade, threshold in sorted(CONFIG.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
        if threshold > prev:
            issues.append(f"Grade {grade} threshold {threshold} > previous {prev}")
        prev = threshold
    
    return issues


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PatternConfigV4',
    'CONFIG',
    'get_config',
    'get_pattern_weights',
    'get_dynamic_context_weights',
    'get_pattern_regime_validity',
    'get_completion_threshold',
    'get_mtf_weight',
    'get_false_breakout_penalty',
    'get_grade',
    'get_position_multiplier',
    'get_action',
    'validate_config',
]