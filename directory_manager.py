# directory_manager.py
import os
from datetime import datetime
from typing import List, Dict
from logger import log

class DirectoryManager:
    """Manages all system directories"""
    
    REQUIRED_DIRS = [
        'signals/raw',
        'signals/confirmed',
        'signals/final',
        'signals/performance/trades',
        'signals/performance/daily',
        'signals/performance/monthly',
        'logs'
    ]
    
    def __init__(self):
        self.base_path = os.getcwd()
        self.created_dirs = []
    
    def create_all(self) -> bool:
        """Create all required directories"""
        try:
            for dir_path in self.REQUIRED_DIRS:
                full_path = os.path.join(self.base_path, dir_path)
                os.makedirs(full_path, exist_ok=True)
                self.created_dirs.append(full_path)
            
            log.info(f"Created/verified {len(self.created_dirs)} directories")
            return True
            
        except Exception as e:
            log.error(f"Failed to create directories: {e}")
            return False
    
    def get_today_file(self, category: str) -> str:
        """
        Get today's file path for a category
        category: 'raw', 'confirmed', 'final', 'performance'
        """
        if category not in ['raw', 'confirmed', 'final', 'performance']:
            raise ValueError(f"Invalid category: {category}")
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}.json"
        
        return os.path.join(self.base_path, 'signals', category, filename)
    
    def list_signals(self, category: str, date: str = None) -> List[str]:
        """List all signal files in a category"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        dir_path = os.path.join(self.base_path, 'signals', category)
        pattern = f"{date}.json"
        
        files = []
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(pattern)]
        
        return files
    
    def get_performance_file(self) -> str:
        """Get performance tracking file"""
        return os.path.join(self.base_path, 'signals', 'performance', 'performance.json')