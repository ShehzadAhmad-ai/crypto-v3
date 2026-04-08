# logger.py
import logging
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from config import Config

class SystemLogger:
    """Centralized logging system with rotation"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('TradingSystem')
        self.logger.setLevel(logging.DEBUG if Config.DEBUG else logging.INFO)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # File handler with rotation (10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(
            f'logs/system_{datetime.now().strftime("%Y%m%d")}.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        if Config.DEBUG:
            console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        
        self._initialized = True
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

# Global instance
log = SystemLogger()