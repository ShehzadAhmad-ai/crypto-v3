# test_pattern_v3.py - Standalone Pattern V3 Test with Debugging
"""
Standalone test for Pattern V3 Expert with full debugging.
Fetches real data and runs only Pattern V3.
"""

import sys
import io
import os
from datetime import datetime

# Fix UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 80)
print(" PATTERN V3 EXPERT - STANDALONE TEST WITH DEBUGGING")
print("=" * 80)
print()

# Import required modules
try:
    from config import Config
    print("✅ Config loaded")
    print(f"   Primary Timeframe: {Config.PRIMARY_TIMEFRAME}")
    print(f"   Portfolio Value: ${Config.PORTFOLIO_VALUE:,.2f}")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

try:
    from data_fetcher import DataFetcher
    print("✅ DataFetcher loaded")
except Exception as e:
    print(f"❌ DataFetcher error: {e}")
    sys.exit(1)

try:
    from pattern_v3.pattern_factory import PatternFactory, test_pattern_factory_with_debug
    print("✅ Pattern V3 loaded")
except Exception as e:
    print(f"❌ Pattern V3 import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def main():
    """Run Pattern V3 test"""
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = Config.PRIMARY_TIMEFRAME
    lookback = Config.LOOKBACK_PERIOD
    
    print("\n" + "-" * 50)
    print("TEST CONFIGURATION")
    print("-" * 50)
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {timeframe}")
    print(f"Lookback: {lookback} candles")
    print("-" * 50)
    
    all_decisions = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"🔬 TESTING: {symbol}")
        print(f"{'='*60}")
        
        # Run test using the debug test function
        decisions = test_pattern_factory_with_debug(
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback
        )
        
        all_decisions[symbol] = decisions
    
    # Final summary
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY")
    print("=" * 80)
    
    for symbol, decisions in all_decisions.items():
        if decisions:
            print(f"\n✅ {symbol}: {len(decisions)} decisions")
            for d in decisions:
                print(f"   - {d['pattern_name']} ({d['direction']}) | Grade: {d['grade']} | RR: {d['risk_reward']:.2f}")
        else:
            print(f"\n❌ {symbol}: No decisions")
    
    print("\n" + "=" * 80)
    print("✅ Test Complete!")
    print("📁 Check pattern_v3/debug_report/ for detailed debug logs")
    print("=" * 80)


if __name__ == "__main__":
    main()