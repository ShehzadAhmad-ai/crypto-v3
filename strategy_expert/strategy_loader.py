"""
Strategy Loader for Strategy Expert
Auto-loads all strategies from the strategies/ folder
No subfolders needed - all strategies in one place
"""

import os
import importlib
import inspect
from typing import Dict, List, Optional, Type
from pathlib import Path

from strategy_expert.base_strategy import BaseStrategy


class StrategyLoader:
    """
    Automatically loads all strategy classes from the strategies folder
    
    Features:
    - Scans all .py files in strategies/ folder
    - Auto-imports and instantiates strategy classes
    - Skips __init__.py and legacy files
    - Provides filtered lists (enabled/disabled)
    """
    
    def __init__(self, strategies_path: Optional[Path] = None):
        """
        Initialize Strategy Loader
        
        Args:
            strategies_path: Path to strategies folder (default: strategy_expert/strategies)
        """
        if strategies_path is None:
            self.strategies_path = Path(__file__).parent / "strategies"
        else:
            self.strategies_path = Path(strategies_path)
        
        self.strategies: Dict[str, BaseStrategy] = {}
        self.load_errors: Dict[str, str] = {}
        self.loaded_count = 0
    
    def load_all_strategies(self) -> Dict[str, BaseStrategy]:
        """
        Scan strategies folder and load all strategy classes
        Returns dict of strategy_name: strategy_instance
        """
        self.strategies = {}
        self.load_errors = {}
        
        if not self.strategies_path.exists():
            print(f"⚠️  Strategies folder not found: {self.strategies_path}")
            print(f"   Creating folder: {self.strategies_path}")
            self.strategies_path.mkdir(parents=True, exist_ok=True)
            return self.strategies
        
        print(f"\n📂 Scanning strategies folder: {self.strategies_path}")
        print("-" * 50)
        
        # Scan all .py files in the strategies folder
        for file_path in self.strategies_path.glob("*.py"):
            if file_path.name.startswith('__'):
                continue
            
            if file_path.name.startswith('legacy'):
                continue
            
            module_name = file_path.stem
            
            try:
                # Import the module
                module = importlib.import_module(
                    f"strategy_expert.strategies.{module_name}"
                )
                
                # Find all classes that inherit from BaseStrategy
                found_in_file = 0
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        
                        # Create instance
                        strategy_instance = obj()
                        self.strategies[strategy_instance.name] = strategy_instance
                        found_in_file += 1
                        
                        # Print with weight info if available
                        weight_info = f" (weight: {strategy_instance.base_weight})" if hasattr(strategy_instance, 'base_weight') else ""
                        print(f"  ✅ Loaded: {strategy_instance.name}{weight_info}")
                
                if found_in_file == 0:
                    print(f"  ⚠️  {file_path.name}: No strategy classes found")
                
                self.loaded_count += found_in_file
                
            except ImportError as e:
                error_msg = f"Import error: {e}"
                self.load_errors[module_name] = error_msg
                print(f"  ❌ {module_name}: {error_msg}")
                
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                self.load_errors[module_name] = error_msg
                print(f"  ❌ {module_name}: {error_msg}")
        
        print("-" * 50)
        print(f"📊 Total loaded: {len(self.strategies)} strategies")
        
        if self.load_errors:
            print(f"⚠️  Errors: {len(self.load_errors)} strategies failed to load")
        
        return self.strategies
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a specific strategy by name"""
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all loaded strategies"""
        return self.strategies
    
    def get_strategy_names(self) -> List[str]:
        """Get list of all strategy names"""
        return list(self.strategies.keys())
    
    def get_enabled_strategies(self, config_loader) -> List[BaseStrategy]:
        """
        Get only enabled strategies based on config
        
        Args:
            config_loader: StrategyExpertConfigLoader instance
        
        Returns:
            List of enabled strategy instances
        """
        enabled = []
        for name, strategy in self.strategies.items():
            if config_loader.is_strategy_enabled(name):
                # Apply base weight from config
                strat_config = config_loader.get_strategy_config(name)
                strategy.base_weight = strat_config.base_weight
                strategy.weight = strat_config.base_weight
                strategy.min_confidence = strat_config.min_confidence
                strategy.min_rr = strat_config.min_rr
                enabled.append(strategy)
        
        return enabled
    
    def get_disabled_strategies(self, config_loader) -> List[str]:
        """Get list of disabled strategy names"""
        disabled = []
        for name in self.strategies.keys():
            if not config_loader.is_strategy_enabled(name):
                disabled.append(name)
        return disabled
    
    def reload_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """
        Reload a specific strategy (useful for development)
        
        Args:
            strategy_name: Name of the strategy to reload
        
        Returns:
            Reloaded strategy instance or None
        """
        # Find the module file
        module_file = None
        for name, strategy in self.strategies.items():
            if name == strategy_name:
                module_name = strategy.__class__.__module__
                module_parts = module_name.split('.')
                if len(module_parts) >= 2:
                    file_name = module_parts[-1] + '.py'
                    module_file = self.strategies_path / file_name
                break
        
        if not module_file or not module_file.exists():
            print(f"❌ Cannot find module for: {strategy_name}")
            return None
        
        try:
            # Reload the module
            module_name = f"strategy_expert.strategies.{module_file.stem}"
            
            if module_name in importlib.sys.modules:
                importlib.reload(importlib.sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
            
            # Find the strategy class again
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseStrategy) and 
                    obj != BaseStrategy):
                    
                    new_strategy = obj()
                    self.strategies[new_strategy.name] = new_strategy
                    print(f"✅ Reloaded: {new_strategy.name}")
                    return new_strategy
            
        except Exception as e:
            print(f"❌ Error reloading {strategy_name}: {e}")
        
        return None
    
    def print_summary(self):
        """Print summary of loaded strategies"""
        print("\n" + "="*60)
        print("STRATEGY LOADER SUMMARY")
        print("="*60)
        print(f"📁 Folder: {self.strategies_path}")
        print(f"📊 Total Strategies: {len(self.strategies)}")
        
        if self.strategies:
            print("\n📋 Loaded Strategies:")
            for name, strategy in sorted(self.strategies.items()):
                weight_info = f" (weight: {strategy.base_weight})" if hasattr(strategy, 'base_weight') else ""
                print(f"   • {name}{weight_info}")
        
        if self.load_errors:
            print("\n⚠️  Load Errors:")
            for name, error in self.load_errors.items():
                print(f"   • {name}: {error}")
        
        print("="*60)


# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def test_loader():
    """Quick test to verify loader works"""
    print("Testing Strategy Loader...")
    loader = StrategyLoader()
    strategies = loader.load_all_strategies()
    
    print(f"\n✅ Loaded {len(strategies)} strategies:")
    for name in strategies.keys():
        print(f"   - {name}")
    
    return loader


if __name__ == "__main__":
    test_loader()