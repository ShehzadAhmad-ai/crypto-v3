# cooldown_manager.py
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from logger import log
from config import Config

class CooldownManager:
    """Prevent signal spam for same symbols - now with direction support"""
    
    def __init__(self):
        self.last_signals: Dict[str, datetime] = {}
        self.cooldown_minutes = Config.SYMBOL_COOLDOWN_MINUTES
    
    def _get_key(self, symbol: str, direction: Optional[str] = None) -> str:
        """Generate cache key - with direction if provided"""
        if direction:
            return f"{symbol}_{direction}"
        return symbol
    
    def can_trade(self, symbol: str, direction: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if symbol/direction is in cooldown
        If direction is provided, checks per-direction cooldown
        If no direction, checks symbol-level cooldown
        Returns: (allowed, reason)
        """
        now = datetime.now()
        key = self._get_key(symbol, direction)
        
        if key in self.last_signals:
            last_time = self.last_signals[key]
            elapsed = (now - last_time).total_seconds() / 60
            
            if elapsed < self.cooldown_minutes:
                wait = self.cooldown_minutes - elapsed
                if direction:
                    return False, f"Cooldown active for {direction} ({wait:.1f} minutes remaining)"
                else:
                    return False, f"Symbol cooldown active ({wait:.1f} minutes remaining)"
        
        return True, "Cooldown OK"
    
    def register_signal(self, symbol: str, direction: Optional[str] = None):
        """
        Record that a signal was generated for this symbol/direction
        If direction provided, registers per-direction cooldown
        Also registers symbol-level cooldown for backward compatibility
        """
        # Register direction-specific if provided
        if direction:
            dir_key = self._get_key(symbol, direction)
            self.last_signals[dir_key] = datetime.now()
            log.debug(f"Registered cooldown for {symbol} {direction}")
        
        # Always register symbol-level cooldown
        sym_key = self._get_key(symbol)
        self.last_signals[sym_key] = datetime.now()
        log.debug(f"Registered symbol cooldown for {symbol}")
    
    def get_remaining_cooldown(self, symbol: str, direction: Optional[str] = None) -> float:
        """Get remaining cooldown in minutes"""
        key = self._get_key(symbol, direction)
        
        if key not in self.last_signals:
            return 0.0
        
        elapsed = (datetime.now() - self.last_signals[key]).total_seconds() / 60
        remaining = max(0.0, self.cooldown_minutes - elapsed)
        return remaining
    
    def can_trade_opposite(self, symbol: str, current_direction: str) -> Tuple[bool, str]:
        """
        Special check for opposite direction trades
        Returns True if enough time has passed since last trade in opposite direction
        """
        opposite = 'SELL' if current_direction == 'BUY' else 'BUY'
        
        # Check opposite direction cooldown
        opp_key = self._get_key(symbol, opposite)
        if opp_key in self.last_signals:
            last_time = self.last_signals[opp_key]
            elapsed = (datetime.now() - last_time).total_seconds() / 60
            
            # Opposite direction trades need same cooldown
            if elapsed < self.cooldown_minutes:
                wait = self.cooldown_minutes - elapsed
                return False, f"Opposite direction cooldown active ({wait:.1f} minutes remaining)"
        
        return True, "Opposite direction OK"
    
    def clear_cooldown(self, symbol: str, direction: Optional[str] = None):
        """Manually clear cooldown for a symbol/direction"""
        if direction:
            key = self._get_key(symbol, direction)
            if key in self.last_signals:
                del self.last_signals[key]
                log.debug(f"Cleared cooldown for {symbol} {direction}")
        else:
            # Clear all keys for this symbol
            keys_to_delete = []
            for k in self.last_signals:
                if k.startswith(symbol):
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del self.last_signals[k]
            log.debug(f"Cleared all cooldown for {symbol}")
    
    def get_all_cooldowns(self) -> Dict[str, float]:
        """Get all active cooldowns with remaining time"""
        now = datetime.now()
        result = {}
        
        for key, last_time in self.last_signals.items():
            elapsed = (now - last_time).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                result[key] = round(self.cooldown_minutes - elapsed, 1)
        
        return result