"""
smc_debug.py - Complete Debug Logging System for SMC Expert V3

Features:
- Daily log files with timestamps
- Tracks every step of analysis
- Records signal generation and rejection reasons
- Logs fallbacks and errors
- JSON export for analysis
- Color-coded console output (optional)
"""

import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import wraps
import sys


class SMCDebugger:
    """
    Centralized debug logger for SMC Expert
    
    Usage:
        debug = SMCDebugger(symbol="BTC/USDT")
        debug.log_step("Starting analysis")
        debug.log_signal_generated(signal_data)
        debug.log_rejection("No order blocks found", "order_block_detection")
        debug.log_error(exception, "price_calculation")
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, symbol: str = None, log_dir: str = "logs/smc_expert"):
        if hasattr(self, '_initialized'):
            return
        
        self.symbol = symbol
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"{self.today}.log"
        self.json_file = self.log_dir / f"{self.today}.json"
        self.error_file = self.log_dir / f"{self.today}_errors.json"
        
        # Setup file logging
        self._setup_logging()
        
        # Session storage
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_steps = []
        self.signals_generated = []
        self.rejections = []
        self.errors = []
        self.fallbacks_used = []
        
        self._initialized = True
        
        self.log("=" * 80)
        self.log(f"SMC EXPERT DEBUG SESSION STARTED: {self.session_id}")
        self.log(f"Symbol: {symbol if symbol else 'N/A'}")
        self.log("=" * 80)
    
    def _setup_logging(self):
        """Setup Python logging to file"""
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        )
        
        self.file_logger = logging.getLogger('smc_expert_debug')
        self.file_logger.setLevel(logging.DEBUG)
        self.file_logger.addHandler(file_handler)
        self.file_logger.propagate = False
    
    def log(self, message: str, level: str = "INFO", print_console: bool = True):
        """Log a message to file and optionally console"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Add to memory
        self.analysis_steps.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        
        # Write to file logger
        if level == "DEBUG":
            self.file_logger.debug(message)
        elif level == "INFO":
            self.file_logger.info(message)
        elif level == "WARNING":
            self.file_logger.warning(message)
        elif level == "ERROR":
            self.file_logger.error(message)
        
        # Print to console if requested
        if print_console:
            if level == "ERROR":
                print(f"❌ {message[:200]}")
            elif level == "WARNING":
                print(f"⚠️ {message[:200]}")
            elif level == "SUCCESS":
                print(f"✅ {message[:200]}")
            elif level == "DEBUG" and False:  # Disable debug console by default
                pass
            else:
                print(f"📊 {message[:200]}")
    
    def log_step(self, step_name: str, details: Dict = None):
        """Log the start of a processing step"""
        self.log(f"▶ STEP: {step_name}", "INFO")
        if details:
            self.log(f"   Details: {json.dumps(details, default=str)[:200]}", "DEBUG")
    
    def log_step_complete(self, step_name: str, result: Any = None, duration_ms: float = None):
        """Log completion of a processing step"""
        msg = f"◼ STEP COMPLETE: {step_name}"
        if duration_ms:
            msg += f" ({duration_ms:.1f}ms)"
        if result:
            msg += f" -> {str(result)[:100]}"
        self.log(msg, "INFO")
    
    def log_signal_generated(self, signal: Dict):
        """Log when a trading signal is generated"""
        self.signals_generated.append({
            'timestamp': datetime.now().isoformat(),
            'signal': signal
        })
        
        direction = signal.get('direction', 'UNKNOWN')
        confidence = signal.get('confidence', 0)
        action = signal.get('action', 'UNKNOWN')
        grade = signal.get('grade', 'N/A')
        
        self.log(f"🎯 SIGNAL GENERATED: {direction} | Confidence: {confidence:.1%} | Action: {action} | Grade: {grade}", "SUCCESS")
        
        # Log full signal details
        self.log(f"   Entry: {signal.get('entry', 'N/A')}", "INFO")
        self.log(f"   Stop Loss: {signal.get('stop_loss', 'N/A')}", "INFO")
        self.log(f"   Take Profit: {signal.get('take_profit', 'N/A')}", "INFO")
        self.log(f"   Risk/Reward: {signal.get('risk_reward', 0):.2f}", "INFO")
        self.log(f"   Reason: {signal.get('decision_reason', 'N/A')[:150]}", "INFO")
    
    def log_rejection(self, reason: str, stage: str, details: Dict = None):
        """Log when a potential signal is rejected"""
        self.rejections.append({
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'reason': reason,
            'details': details
        })
        
        self.log(f"❌ REJECTED at {stage}: {reason}", "WARNING")
        if details:
            self.log(f"   Details: {json.dumps(details, default=str)[:150]}", "DEBUG")
    
    def log_error(self, error: Exception, location: str, context: Dict = None):
        """Log an error with full traceback"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        })
        
        self.log(f"💥 ERROR in {location}: {type(error).__name__} - {str(error)[:100]}", "ERROR")
        
        # Save errors immediately
        self._save_errors()
    
    def log_fallback(self, fallback_name: str, original_method: str, reason: str):
        """Log when a fallback method is used"""
        self.fallbacks_used.append({
            'timestamp': datetime.now().isoformat(),
            'fallback': fallback_name,
            'original_method': original_method,
            'reason': reason
        })
        
        self.log(f"🔄 FALLBACK: {fallback_name} (instead of {original_method}) - {reason}", "WARNING")
    
    def log_component_result(self, component: str, success: bool, result: Any = None, 
                              duration_ms: float = None, details: Dict = None):
        """Log result of a component analysis"""
        status = "✅" if success else "❌"
        msg = f"{status} COMPONENT: {component}"
        if duration_ms:
            msg += f" ({duration_ms:.1f}ms)"
        if result is not None:
            if isinstance(result, (int, float)):
                msg += f" -> {result}"
            elif isinstance(result, list):
                msg += f" -> {len(result)} items"
        
        self.log(msg, "INFO" if success else "WARNING")
        if details:
            self.log(f"   Details: {json.dumps(details, default=str)[:150]}", "DEBUG")
    
    def log_dataframe(self, name: str, df, context: str = ""):
        """Log DataFrame information for debugging"""
        if df is None:
            self.log(f"📊 DATAFRAME {name}: None", "DEBUG")
            return
        
        try:
            info = {
                'name': name,
                'shape': df.shape,
                'columns': list(df.columns)[:15],
                'has_nan': df.isna().any().any(),
                'last_price': float(df['close'].iloc[-1]) if 'close' in df else None
            }
            self.log(f"📊 DATAFRAME {name}: {df.shape[0]}x{df.shape[1]} | Columns: {info['columns'][:5]}...", "DEBUG")
        except Exception as e:
            self.log(f"📊 DATAFRAME {name}: Error logging - {e}", "DEBUG")
    
    def start_analysis(self, symbol: str, timeframe: str, df_shape: tuple):
        """Start a new analysis session for a symbol"""
        self.symbol = symbol
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log("=" * 80)
        self.log(f"🔬 ANALYZING: {symbol} | {timeframe} | {df_shape[0]} bars")
        self.log("=" * 80)
    
    def end_analysis(self, signal_generated: bool = False):
        """End the current analysis session"""
        status = "SIGNAL GENERATED" if signal_generated else "NO SIGNAL"
        self.log(f"📊 ANALYSIS COMPLETE: {self.symbol} - {status}")
        self.log("=" * 80)
        
        # Save session data
        self._save_session()
    
    def _save_session(self):
        """Save all session data to JSON"""
        session_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'analysis_steps': self.analysis_steps[-500:],
            'signals_generated': self.signals_generated,
            'rejections': self.rejections,
            'errors': self.errors,
            'fallbacks_used': self.fallbacks_used
        }
        
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            self.log(f"💾 Session saved to {self.json_file}", "DEBUG")
        except Exception as e:
            self.log(f"Error saving session: {e}", "ERROR")
    
    def _save_errors(self):
        """Save errors to separate file"""
        try:
            error_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'errors': self.errors,
                'rejections': self.rejections
            }
            with open(self.error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, default=str)
        except Exception:
            pass
    
    def get_summary(self) -> Dict:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'symbol': self.symbol,
            'total_steps': len(self.analysis_steps),
            'total_errors': len(self.errors),
            'total_rejections': len(self.rejections),
            'total_signals': len(self.signals_generated),
            'total_fallbacks': len(self.fallbacks_used),
            'log_file': str(self.log_file),
            'json_file': str(self.json_file)
        }
    
    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("SMC EXPERT DEBUG SUMMARY")
        print("=" * 60)
        print(f"Session: {summary['session_id']}")
        print(f"Symbol: {summary['symbol']}")
        print(f"Steps: {summary['total_steps']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Rejections: {summary['total_rejections']}")
        print(f"Signals: {summary['total_signals']}")
        print(f"Fallbacks: {summary['total_fallbacks']}")
        print("-" * 60)
        print(f"Log file: {summary['log_file']}")
        print(f"JSON file: {summary['json_file']}")
        print("=" * 60)


# ============================================================================
# DECORATOR FOR AUTO-LOGGING
# ============================================================================

def debug_log(component: str):
    """Decorator to automatically log function calls and results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            debug = SMCDebugger()
            
            # Get symbol from args if available
            symbol = "UNKNOWN"
            for arg in args:
                if hasattr(arg, 'symbol'):
                    symbol = arg.symbol
                    break
            
            debug.log(f"🔧 CALLING {component}.{func.__name__}", "DEBUG")
            
            try:
                import time
                start = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                
                # Log result summary
                if result:
                    if isinstance(result, dict) and result.get('direction'):
                        debug.log(f"✅ {component}.{func.__name__} returned {result.get('direction')} signal ({duration:.1f}ms)", "DEBUG")
                    elif isinstance(result, list):
                        debug.log(f"✅ {component}.{func.__name__} returned {len(result)} items ({duration:.1f}ms)", "DEBUG")
                    else:
                        debug.log(f"✅ {component}.{func.__name__} completed ({duration:.1f}ms)", "DEBUG")
                else:
                    debug.log(f"⚠️ {component}.{func.__name__} returned None ({duration:.1f}ms)", "DEBUG")
                
                return result
            except Exception as e:
                debug.log_error(e, f"{component}.{func.__name__}")
                raise
        return wrapper
    return decorator


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_debugger = None

def get_debugger(symbol: str = None) -> SMCDebugger:
    """Get the global debugger instance"""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = SMCDebugger(symbol)
    elif symbol and _global_debugger.symbol != symbol:
        # New symbol, create new session but keep same instance
        _global_debugger.start_analysis(symbol, "N/A", (0,))
    return _global_debugger


def quick_log(message: str, level: str = "INFO"):
    """Quick log function"""
    get_debugger().log(message, level)