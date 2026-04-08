"""
pattern_debug.py - Complete Debug Logging for Pattern V4 System

Logs every step, every score, every decision to file with detailed breakdowns.
Supports all 23 pattern types with similarity component logging.

Version: 4.0
Author: Pattern Intelligence System
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
import logging


# ============================================================================
# DEBUG CONFIGURATION
# ============================================================================

class DebugConfigV4:
    """Configuration for debug system"""
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.debug_dir = self.base_dir / "debug_report"
        self.log_level = logging.DEBUG
        self.save_json = True
        self.save_tracebacks = True
        self.max_log_entries = 10000
        self.log_similarity_components = True
        self.log_context_components = True
        self.log_mtf_details = True
        self.log_evolution = True
        
        # Auto-create directory
        self.debug_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MAIN DEBUG LOGGER
# ============================================================================

class PatternDebuggerV4:
    """
    Singleton debug logger for Pattern V4 system.
    Captures ALL events, errors, scores, and performance metrics.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if PatternDebuggerV4._initialized:
            return
        
        self.config = DebugConfigV4()
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.session_time = datetime.now().strftime("%H%M%S")
        self.log_file = self.config.debug_dir / f"{self.today}_{self.session_time}.log"
        self.json_file = self.config.debug_dir / f"{self.today}_{self.session_time}.json"
        self.error_file = self.config.debug_dir / f"errors_{self.today}.json"
        
        # Setup file logging
        self._setup_file_logging()
        
        # Storage for current session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.events = []
        self.errors = []
        self.warnings = []
        self.step_times = {}
        self.pattern_results = []
        self.current_symbol = ""
        self.current_timeframe = ""
        
        PatternDebuggerV4._initialized = True
        
        self.log("=" * 80)
        self.log(f"PATTERN V4 DEBUG SESSION STARTED: {self.session_id}")
        self.log(f"Session Time: {datetime.now().isoformat()}")
        self.log("=" * 80)
    
    def _setup_file_logging(self):
        """Setup Python logging to file"""
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        )
        
        self.file_logger = logging.getLogger('pattern_v4_debug')
        self.file_logger.setLevel(logging.DEBUG)
        self.file_logger.addHandler(file_handler)
        self.file_logger.propagate = False
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message to both file and memory"""
        timestamp = datetime.now().isoformat()
        
        # Store in memory
        self.events.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        
        # Trim if too many
        if len(self.events) > self.config.max_log_entries:
            self.events = self.events[-self.config.max_log_entries:]
        
        # Write to file logger
        if level == "DEBUG":
            self.file_logger.debug(message)
        elif level == "INFO":
            self.file_logger.info(message)
        elif level == "WARNING":
            self.file_logger.warning(message)
        elif level == "ERROR":
            self.file_logger.error(message)
        elif level == "CRITICAL":
            self.file_logger.critical(message)
        
        # Also print to console for important messages
        if level in ["ERROR", "CRITICAL", "WARNING"]:
            print(f"[PatternV4-{level}] {message[:200]}")
    
    def log_step(self, step_name: str, start_time: float = None):
        """Log a processing step with timing"""
        import time
        
        if start_time is None:
            # Start timing
            self.step_times[step_name] = {'start': time.time()}
            self.log(f"▶ STEP START: {step_name}", "DEBUG")
        else:
            # End timing
            elapsed = time.time() - start_time
            if step_name in self.step_times:
                self.step_times[step_name]['end'] = time.time()
                self.step_times[step_name]['elapsed'] = elapsed
            self.log(f"◼ STEP END: {step_name} - {elapsed:.3f}s", "DEBUG")
        
        return time.time() if start_time is None else None
    
    def log_error(self, location: str, error: Exception, context: Dict = None):
        """Log an error with full traceback"""
        timestamp = datetime.now().isoformat()
        error_info = {
            'timestamp': timestamp,
            'location': location,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.errors.append(error_info)
        self.log(f"❌ ERROR in {location}: {type(error).__name__} - {str(error)[:200]}", "ERROR")
        
        # Save to error file immediately
        self._save_errors()
        
        return error_info
    
    def log_warning(self, location: str, message: str, context: Dict = None):
        """Log a warning"""
        timestamp = datetime.now().isoformat()
        warning_info = {
            'timestamp': timestamp,
            'location': location,
            'message': message,
            'context': context or {}
        }
        
        self.warnings.append(warning_info)
        self.log(f"⚠️ WARNING in {location}: {message}", "WARNING")
    
    def log_pattern_detection(self, pattern_data: Dict):
        """
        Log a detected pattern with all similarity components.
        This is the main logging function for V4.
        """
        pattern_name = pattern_data.get('pattern_name', 'Unknown')
        direction = pattern_data.get('direction', 'NEUTRAL')
        similarity = pattern_data.get('similarity', 0.0)
        components = pattern_data.get('components', {})
        
        self.log("")
        self.log("─" * 80)
        self.log(f"🔍 PATTERN DETECTED: {pattern_name} ({direction})")
        self.log("─" * 80)
        
        # Similarity score with bar
        bar_length = int(similarity * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        self.log(f"📐 SIMILARITY SCORE: {similarity:.1%} {bar}")
        
        # Log components if available
        if components and self.config.log_similarity_components:
            self.log("   ├── Component Scores:")
            for i, (name, value) in enumerate(components.items()):
                prefix = "   └──" if i == len(components) - 1 else "   ├──"
                self.log(f"   {prefix} {name}: {value:.1%}")
        
        return pattern_data
    
    def log_context_score(self, context_score: float, components: Dict, regime: str):
        """Log context scoring details"""
        self.log("")
        self.log("🏪 CONTEXT SCORE: {:.1%}".format(context_score))
        self.log(f"   Regime: {regime}")
        
        if components and self.config.log_context_components:
            self.log("   ├── Components:")
            for i, (name, value) in enumerate(components.items()):
                prefix = "   └──" if i == len(components) - 1 else "   ├──"
                self.log(f"   {prefix} {name}: {value:.1%}")
    
    def log_mtf_confluence(self, mtf_data: Dict):
        """Log multi-timeframe confluence details"""
        if not mtf_data or not self.config.log_mtf_details:
            return
        
        self.log("")
        self.log("📈 HTF CONFLUENCE:")
        
        boost = mtf_data.get('boost_factor', 1.0)
        bar_length = int((boost - 0.7) / 0.6 * 40)  # 0.7-1.3 range
        bar = "█" * max(0, min(40, bar_length))
        self.log(f"   Boost Factor: {boost:.2f}x {bar}")
        
        tf_scores = mtf_data.get('timeframe_scores', {})
        if tf_scores:
            self.log("   ├── Timeframe Scores:")
            for i, (tf, score) in enumerate(tf_scores.items()):
                prefix = "   └──" if i == len(tf_scores) - 1 else "   ├──"
                self.log(f"   {prefix} {tf}: {score:.1%}")
        
        same_pattern_tfs = mtf_data.get('same_pattern_timeframes', [])
        if same_pattern_tfs:
            self.log(f"   └── Same pattern on: {', '.join(same_pattern_tfs)}")
    
    def log_evolution(self, evolution: Dict):
        """Log pattern evolution tracking"""
        if not evolution or not self.config.log_evolution:
            return
        
        self.log("")
        self.log("🔄 PATTERN EVOLUTION:")
        self.log(f"   Initial Confidence: {evolution.get('initial_confidence', 0):.1%}")
        self.log(f"   Current Confidence: {evolution.get('current_confidence', 0):.1%}")
        self.log(f"   Trend: {evolution.get('confidence_trend', 'STABLE')}")
        self.log(f"   Completion: {evolution.get('completion_pct', 0):.1%}")
    
    def log_false_breakout_risk(self, risk_data: Dict):
        """Log false breakout risk assessment"""
        if not risk_data:
            return
        
        risk_score = risk_data.get('risk_score', 0.5)
        risk_level = "LOW" if risk_score < 0.3 else "MEDIUM" if risk_score < 0.6 else "HIGH"
        
        self.log("")
        self.log(f"⚠️ FALSE BREAKOUT RISK: {risk_level} ({risk_score:.1%})")
        
        features = risk_data.get('features', {})
        if features:
            risky_features = [f for f, present in features.items() if present]
            if risky_features:
                self.log(f"   └── Risk factors: {', '.join(risky_features)}")
    
    def log_final_confidence(self, final_confidence: float, grade: str, action: str):
        """Log final confidence calculation"""
        self.log("")
        self.log("✨ FINAL CONFIDENCE: {:.1%}".format(final_confidence))
        
        # Grade with color indicator
        grade_colors = {
            'A+': '🟢', 'A': '🟢', 'B+': '🟡', 'B': '🟡',
            'B-': '🟠', 'C+': '🟠', 'C': '🔴', 'D': '🔴', 'F': '⚫'
        }
        color = grade_colors.get(grade, '⚪')
        
        self.log(f"   Grade: {color} {grade}")
        
        action_icons = {
            'STRONG_ENTRY': '💪',
            'ENTER_NOW': '✅',
            'WAIT_FOR_RETEST': '⏳',
            'SKIP': '⏭️'
        }
        icon = action_icons.get(action, '📊')
        self.log(f"   Action: {icon} {action}")
    
    def log_trade_setup(self, trade_setup: Dict):
        """Log trade setup details"""
        self.log("")
        self.log("💰 TRADE SETUP:")
        
        entry = trade_setup.get('entry', 0)
        stop = trade_setup.get('stop_loss', 0)
        target = trade_setup.get('take_profit', 0)
        rr = trade_setup.get('risk_reward', 0)
        
        self.log(f"   Entry: {entry:.4f}")
        self.log(f"   Stop Loss: {stop:.4f}")
        self.log(f"   Take Profit: {target:.4f}")
        self.log(f"   Risk/Reward: {rr:.2f}")
        
        retest_level = trade_setup.get('retest_level')
        if retest_level:
            self.log(f"   Retest Level: {retest_level:.4f}")
    
    def log_pattern_rejection(self, pattern_name: str, reason: str):
        """Log when a pattern is rejected"""
        self.log(f"❌ REJECTED: {pattern_name} - {reason}", "WARNING")
    
    def log_pattern_invalidation(self, pattern_name: str, reason: str):
        """Log when a pattern is invalidated"""
        self.log(f"🚫 INVALIDATED: {pattern_name} - {reason}", "WARNING")
    
    def log_competition_results(self, ranked_patterns: List[Dict], selected: Dict):
        """Log pattern competition ranking"""
        self.log("")
        self.log("─" * 80)
        self.log("📊 PATTERN COMPETITION RANKING")
        self.log("─" * 80)
        
        for i, pattern in enumerate(ranked_patterns[:5], 1):
            name = pattern.get('pattern_name', 'Unknown')
            direction = pattern.get('direction', '?')
            confidence = pattern.get('final_confidence', 0)
            selected_mark = "← SELECTED" if pattern == selected else ""
            self.log(f"   {i}. {name} ({direction}): {confidence:.1%} {selected_mark}")
    
    def log_learning_insights(self, insights: Dict):
        """Log learning and self-improvement insights"""
        if not insights:
            return
        
        self.log("")
        self.log("─" * 80)
        self.log("📚 LEARNING INSIGHTS")
        self.log("─" * 80)
        
        pattern_perf = insights.get('pattern_performance', {})
        if pattern_perf:
            self.log("   Pattern Performance:")
            for pattern, perf in list(pattern_perf.items())[:5]:
                win_rate = perf.get('win_rate', 0)
                trades = perf.get('wins', 0) + perf.get('losses', 0)
                self.log(f"      {pattern}: {win_rate:.1%} ({trades} trades)")
        
        failures = insights.get('top_failure_reasons', [])
        if failures:
            self.log("   Top Failure Reasons:")
            for reason, count in failures[:3]:
                self.log(f"      - {reason}: {count} occurrences")
        
        recommendations = insights.get('recommendations', [])
        if recommendations:
            self.log("   Recommendations:")
            for rec in recommendations[:3]:
                self.log(f"      💡 {rec}")
    
    def log_summary(self, stats: Dict):
        """Log session summary"""
        self.log("")
        self.log("═" * 80)
        self.log("📈 SESSION SUMMARY")
        self.log("═" * 80)
        
        self.log(f"   Symbol: {self.current_symbol}")
        self.log(f"   Timeframe: {self.current_timeframe}")
        self.log(f"   Total patterns detected: {stats.get('total_detected', 0)}")
        self.log(f"   After invalidation: {stats.get('after_invalidation', 0)}")
        self.log(f"   Tradeable signals: {stats.get('tradeable', 0)}")
        self.log(f"   Processing time: {stats.get('processing_time_ms', 0):.0f}ms")
        
        if stats.get('avg_similarity', 0) > 0:
            self.log(f"   Avg similarity score: {stats.get('avg_similarity', 0):.1%}")
        
        if stats.get('avg_context_score', 0) > 0:
            self.log(f"   Avg context score: {stats.get('avg_context_score', 0):.1%}")
    
    def log_performance_stats(self, stats: Dict):
        """Log performance statistics"""
        self.log("")
        self.log("⚡ PERFORMANCE STATS:")
        self.log(f"   Patterns processed: {stats.get('total_patterns_processed', 0)}")
        self.log(f"   Patterns accepted: {stats.get('total_patterns_accepted', 0)}")
        self.log(f"   Acceptance rate: {stats.get('acceptance_rate', 0):.1%}")
        self.log(f"   Avg processing time: {stats.get('avg_processing_time_ms', 0):.1f}ms")
    
    def start_session(self, symbol: str, timeframe: str, df_shape: Tuple[int, int]):
        """Start a new analysis session for a symbol"""
        self.current_symbol = symbol
        self.current_timeframe = timeframe
        self.session_start = datetime.now()
        
        self.log("")
        self.log("═" * 80)
        self.log(f"🔬 ANALYZING: {symbol} | {timeframe} | {df_shape[0]} bars")
        self.log("═" * 80)
        
        # Log input data summary
        self.log(f"📊 INPUT DATA SUMMARY:")
        self.log(f"   DataFrame shape: {df_shape}")
        self.log(f"   Session started: {self.session_start.isoformat()}")
    
    def end_session(self, decisions_count: int):
        """End the current analysis session"""
        elapsed = (datetime.now() - self.session_start).total_seconds()
        
        self.log("")
        self.log("═" * 80)
        self.log(f"📊 SESSION COMPLETE: {self.current_symbol}")
        self.log(f"   Decisions generated: {decisions_count}")
        self.log(f"   Session duration: {elapsed:.2f}s")
        self.log("═" * 80)
        
        # Save session data
        self._save_session()
    
    def _save_session(self):
        """Save all session data to JSON"""
        session_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': self.current_symbol,
            'timeframe': self.current_timeframe,
            'events': self.events[-500:],
            'errors': self.errors,
            'warnings': self.warnings,
            'step_times': self.step_times,
            'pattern_results': self.pattern_results,
        }
        
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            self.log(f"💾 Session saved to {self.json_file}", "DEBUG")
        except Exception as e:
            self.log_error("Save session", e)
    
    def _save_errors(self):
        """Save errors to separate file"""
        try:
            error_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'errors': self.errors[-100:],
                'warnings': self.warnings[-100:]
            }
            with open(self.error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, default=str)
        except Exception:
            pass
    
    def get_summary(self) -> Dict:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'total_events': len(self.events),
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'total_decisions': len(self.pattern_results),
            'step_times': self.step_times,
        }
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("PATTERN V4 DEBUG SUMMARY")
        print("=" * 60)
        print(f"Session: {summary['session_id']}")
        print(f"Events: {summary['total_events']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Warnings: {summary['total_warnings']}")
        print(f"Decisions: {summary['total_decisions']}")
        print("-" * 60)
        print(f"Log file: {self.log_file}")
        print(f"JSON file: {self.json_file}")
        print("=" * 60)


# ============================================================================
# DECORATORS FOR AUTO-LOGGING
# ============================================================================

def debug_log_v4(level="INFO"):
    """Decorator to automatically log function calls and results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            debugger = PatternDebuggerV4()
            
            # Build argument preview
            arg_preview = {}
            for i, arg in enumerate(args[:2]):
                if hasattr(arg, 'shape'):
                    arg_preview[f'arg{i}'] = f"DataFrame{arg.shape}"
                elif isinstance(arg, str):
                    arg_preview[f'arg{i}'] = arg[:50]
                else:
                    arg_preview[f'arg{i}'] = str(type(arg).__name__)
            
            for key, value in list(kwargs.items())[:3]:
                if hasattr(value, 'shape'):
                    arg_preview[key] = f"DataFrame{value.shape}"
                else:
                    arg_preview[key] = str(value)[:30]
            
            debugger.log(f"🔧 CALLING {func.__name__}", "DEBUG")
            
            try:
                result = func(*args, **kwargs)
                
                # Log result preview
                result_preview = None
                if isinstance(result, list):
                    result_preview = f"list[{len(result)}]"
                elif isinstance(result, dict):
                    result_preview = f"dict[{len(result)}]"
                elif hasattr(result, 'shape'):
                    result_preview = f"DataFrame{result.shape}"
                else:
                    result_preview = str(result)[:50]
                
                debugger.log(f"✅ {func.__name__} returned: {result_preview}", "DEBUG")
                return result
                
            except Exception as e:
                debugger.log_error(func.__name__, e, {'args': arg_preview, 'kwargs': kwargs})
                raise
                
        return wrapper
    return decorator


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_debugger = None

def get_debugger_v4() -> PatternDebuggerV4:
    """Get the global debugger instance"""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = PatternDebuggerV4()
    return _global_debugger


def quick_log(message: str, level: str = "INFO"):
    """Quick log function"""
    get_debugger_v4().log(message, level)


def log_error(location: str, error: Exception):
    """Quick error log"""
    get_debugger_v4().log_error(location, error)


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_similarity_bar(score: float, width: int = 40) -> str:
    """Format similarity score as a progress bar"""
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def format_confidence_bar(confidence: float, width: int = 30) -> str:
    """Format confidence as a progress bar with color indicators"""
    filled = int(confidence * width)
    
    if confidence >= 0.8:
        color = "🟢"  # Green for strong
    elif confidence >= 0.65:
        color = "🟡"  # Yellow for medium
    elif confidence >= 0.5:
        color = "🟠"  # Orange for weak
    else:
        color = "🔴"  # Red for poor
    
    return color + " " + ("█" * filled + "░" * (width - filled))


def format_grade_with_icon(grade: str) -> str:
    """Format grade with appropriate icon"""
    icons = {
        'A+': '🏆', 'A': '📈', 'B+': '📊', 'B': '📉',
        'B-': '⚠️', 'C+': '⚠️', 'C': '❌', 'D': '❌', 'F': '⛔'
    }
    return f"{icons.get(grade, '📊')} {grade}"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'DebugConfigV4',
    'PatternDebuggerV4',
    'get_debugger_v4',
    'quick_log',
    'log_error',
    'debug_log_v4',
    'format_similarity_bar',
    'format_confidence_bar',
    'format_grade_with_icon',
]