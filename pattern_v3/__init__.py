"""
pattern_v4 - Advanced Pattern Intelligence System

Complete pattern detection and analysis system with:
- 25+ pattern types (Harmonic, Structure, Volume, Divergence, Wave)
- Continuous similarity scoring (0-1) - NO hard-coded thresholds
- Real context analysis (trend, location, volatility, momentum)
- Liquidity intelligence (sweeps, inducements, stop hunts)
- Trap detection and conversion
- Multi-pattern clustering with feedback loop
- Meta scoring with confidence calibration
- Multi-timeframe confluence analysis
- Self-learning from historical performance
- Complete trade execution (entry/stop/target with RR)

Version: 4.0
Author: Pattern Intelligence System
"""

# Core classes
from .pattern_core import (
    PatternV4,
    PatternDirection,
    PatternType,
    PatternStage,
    ActionType,
    Grade,
    PatternSimilarity,
    ContextScore,
    HTFConfluence,
    PatternEvolution,
    FalseBreakoutRisk,
    LiquidityAnalysis,
    TrapAnalysis,
    TradeSetup,
    PatternClustering,
    PatternLifecycle,
    SwingPoint,
    create_pattern_v4,
)

# Configuration
from .pattern_config import (
    CONFIG,
    PatternConfigV4,
    get_config,
    get_pattern_weights,
    get_dynamic_context_weights,
    get_pattern_regime_validity,
    get_completion_threshold,
    get_mtf_weight,
    get_false_breakout_penalty,
    get_grade,
    get_position_multiplier,
    get_action,
    validate_config,
)

# Detection Engine
from .pattern_detection import (
    PatternDetectionEngineV4,
    AdaptiveSwingDetectorV4,
    HeadShouldersDetectorV4,
    DoubleTopBottomDetectorV4,
    TripleTopBottomDetectorV4,
    FlagPennantDetectorV4,
    RisingWedgeDetectorV4,
    FallingWedgeDetectorV4,
    AscendingTriangleDetectorV4,
    DescendingTriangleDetectorV4,
    SymmetricalTriangleDetectorV4,
    CupHandleDetectorV4,
    AdamEveDetectorV4,
    QuasimodoDetectorV4,
    GartleyDetectorV4,
    BatDetectorV4,
    CrabDetectorV4,
    ButterflyDetectorV4,
    ThreeDrivesDetectorV4,
    VCPDetectorV4,
    WolfeWaveDetectorV4,
    DivergenceDetectorV4,
)

# Intelligence Engine
from .pattern_intelligence import (
    PatternIntelligenceEngineV4,
    ContextAnalyzerV4,
    LiquidityAnalyzerV4,
    TrapDetectorV4,
    QualityCalculatorV4,
)

# Execution Engine
from .pattern_execution import (
    PatternExecutionEngineV4,
    DraftExecutionEngineV4,
    FinalExecutionAdjustmentV4,
    RetestConfirmationTrackerV4,
    LatencyDetectorV4,
    CompletionCalculatorV4,
)

# Clustering Engine
from .pattern_clustering import (
    PatternClusteringEngineV4Orchestrator,
    PatternClusteringEngineV4,
    ConfluenceScorerV4,
    PatternClusterV4,
    ClusteringConfigV4,
)

# Scoring Engine
from .pattern_scoring import (
    PatternScoringEngineV4,
    RiskPenaltyCalculatorV4,
    FalseBreakoutPenaltyV4,
    ConfidenceCalculatorV4,
    GradeAssignerV4,
    ActionDeterminerV4,
    get_score_summary,
)

# Learning Engine
from .pattern_learning import (
    PatternLearningEngineV4,
    PatternPerformanceV4,
    PatternWeightOptimizerV4,
)

# Multi-Timeframe Engine
from .pattern_mtf import (
    MultiTimeframeAnalyzerV4,
    MTFConfluenceResult,
    TIMEFRAME_ORDER,
    TIMEFRAME_WEIGHTS,
    get_higher_timeframes,
    get_lower_timeframes,
)

# Factory (Main Orchestrator)
from .pattern_factory import (
    PatternFactoryV4,
    PatternFactoryLegacyV4,
    FailFastSystemV4,
    test_pattern_factory_v4,
)

# Debug
from .pattern_debug import (
    PatternDebuggerV4,
    get_debugger_v4,
    quick_log,
    log_error,
    debug_log_v4,
    format_similarity_bar,
    format_confidence_bar,
    format_grade_with_icon,
)

# Adapter (for expert system integration)
from .adapter_pattern_v3 import (
    ExpertSignal,
    pattern_v4_to_expert_signal,
    pattern_v3_to_expert_signal,  # Legacy alias for backward compatibility
    pattern_v4_batch_to_expert_signals,
    validate_expert_signal,
    get_signal_summary,
)


# ============================================================================
# LEGACY SUPPORT
# ============================================================================

# For backward compatibility with existing code that imports from pattern_v3
PatternV3 = PatternV4
PatternFactory = PatternFactoryV4
PatternFactoryLegacy = PatternFactoryLegacyV4
PatternDetectionEngine = PatternDetectionEngineV4
PatternIntelligenceEngine = PatternIntelligenceEngineV4
PatternExecutionEngine = PatternExecutionEngineV4
PatternClusteringEngine = PatternClusteringEngineV4Orchestrator
PatternScoringEngine = PatternScoringEngineV4
PatternDebugger = PatternDebuggerV4
get_debugger = get_debugger_v4


# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "4.0.0"
__author__ = "Pattern Intelligence System"
__all__ = [
    # Core
    'PatternV4',
    'PatternV3',  # Legacy alias
    'PatternDirection',
    'PatternType',
    'PatternStage',
    'ActionType',
    'Grade',
    'PatternSimilarity',
    'ContextScore',
    'HTFConfluence',
    'PatternEvolution',
    'FalseBreakoutRisk',
    'LiquidityAnalysis',
    'TrapAnalysis',
    'TradeSetup',
    'PatternClustering',
    'PatternLifecycle',
    'SwingPoint',
    'create_pattern_v4',
    
    # Config
    'CONFIG',
    'PatternConfigV4',
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
    
    # Detection
    'PatternDetectionEngineV4',
    'PatternDetectionEngine',  # Legacy alias
    'AdaptiveSwingDetectorV4',
    'HeadShouldersDetectorV4',
    'DoubleTopBottomDetectorV4',
    'TripleTopBottomDetectorV4',
    'FlagPennantDetectorV4',
    'RisingWedgeDetectorV4',
    'FallingWedgeDetectorV4',
    'AscendingTriangleDetectorV4',
    'DescendingTriangleDetectorV4',
    'SymmetricalTriangleDetectorV4',
    'CupHandleDetectorV4',
    'AdamEveDetectorV4',
    'QuasimodoDetectorV4',
    'GartleyDetectorV4',
    'BatDetectorV4',
    'CrabDetectorV4',
    'ButterflyDetectorV4',
    'ThreeDrivesDetectorV4',
    'VCPDetectorV4',
    'WolfeWaveDetectorV4',
    'DivergenceDetectorV4',
    
    # Intelligence
    'PatternIntelligenceEngineV4',
    'PatternIntelligenceEngine',  # Legacy alias
    'ContextAnalyzerV4',
    'LiquidityAnalyzerV4',
    'TrapDetectorV4',
    'QualityCalculatorV4',
    
    # Execution
    'PatternExecutionEngineV4',
    'PatternExecutionEngine',  # Legacy alias
    'DraftExecutionEngineV4',
    'FinalExecutionAdjustmentV4',
    'RetestConfirmationTrackerV4',
    'LatencyDetectorV4',
    'CompletionCalculatorV4',
    
    # Clustering
    'PatternClusteringEngineV4Orchestrator',
    'PatternClusteringEngine',  # Legacy alias
    'PatternClusteringEngineV4',
    'ConfluenceScorerV4',
    'PatternClusterV4',
    'ClusteringConfigV4',
    
    # Scoring
    'PatternScoringEngineV4',
    'PatternScoringEngine',  # Legacy alias
    'RiskPenaltyCalculatorV4',
    'FalseBreakoutPenaltyV4',
    'ConfidenceCalculatorV4',
    'GradeAssignerV4',
    'ActionDeterminerV4',
    'get_score_summary',
    
    # Learning
    'PatternLearningEngineV4',
    'PatternPerformanceV4',
    'PatternWeightOptimizerV4',
    
    # MTF
    'MultiTimeframeAnalyzerV4',
    'MTFConfluenceResult',
    'TIMEFRAME_ORDER',
    'TIMEFRAME_WEIGHTS',
    'get_higher_timeframes',
    'get_lower_timeframes',
    
    # Factory
    'PatternFactoryV4',
    'PatternFactory',  # Legacy alias
    'PatternFactoryLegacyV4',
    'PatternFactoryLegacy',  # Legacy alias
    'FailFastSystemV4',
    'test_pattern_factory_v4',
    
    # Debug
    'PatternDebuggerV4',
    'PatternDebugger',  # Legacy alias
    'get_debugger_v4',
    'get_debugger',  # Legacy alias
    'quick_log',
    'log_error',
    'debug_log_v4',
    'format_similarity_bar',
    'format_confidence_bar',
    'format_grade_with_icon',
    
    # Adapter
    'ExpertSignal',
    'pattern_v4_to_expert_signal',
    'pattern_v3_to_expert_signal',  # Legacy alias
    'pattern_v4_batch_to_expert_signals',
    'validate_expert_signal',
    'get_signal_summary',
]