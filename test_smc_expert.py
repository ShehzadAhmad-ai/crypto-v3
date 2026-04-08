"""
SMC Expert V3 - Complete Test Script for TradingSystemV3
Place this file in the root folder: TradingSystemV3/
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import time
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the system
from data_fetcher import DataFetcher
from config import Config

# Import SMC Expert from its folder
from smc_expert.smc_factory import SMCFactory
from smc_expert.smc_config import CONFIG
from smc_expert.smc_debug import get_debugger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Top 20 coins to test (USDT pairs)
TEST_SYMBOLS = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'XRP/USDT',
    'SOL/USDT',
    'ADA/USDT',
    'DOGE/USDT',
    'AVAX/USDT',
    'DOT/USDT',
    'LINK/USDT',
    'MATIC/USDT',
    'SHIB/USDT',
    'TRX/USDT',
    'ICP/USDT',
    'LEO/USDT',
    'ETC/USDT',
    'NEAR/USDT',
    'ATOM/USDT',
    'APT/USDT',
    'ARB/USDT'
]

# Timeframes configuration
TIMEFRAMES = {
    'current': '5m',
    'htf_list': ['15m', '1h', '4h', '1d']
}

# Data limits
LIMIT_CURRENT = 500
LIMIT_HTF = 300


class SMCTester:
    """Complete tester for SMC Expert V3"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.smc = SMCFactory()
        self.results: List[Dict] = []
        self.errors: List[Dict] = []
        
        # Summary stats
        self.summary = {
            'total_symbols': 0,
            'signals_generated': 0,
            'errors_encountered': 0,
            'avg_confidence': 0,
            'avg_rr': 0,
            'grade_distribution': {},
            'pattern_distribution': {},
            'order_blocks_total': 0,
            'fvgs_total': 0,
            'sweeps_total': 0
        }
    
    def fetch_all_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all timeframe data for a symbol
        """
        data = {}
        
        try:
            # Fetch current timeframe (5m)
            logger.info(f"  📊 Fetching 5m data for {symbol}...")
            df_current = self.data_fetcher.fetch_ohlcv(
                symbol, 
                timeframe=TIMEFRAMES['current'], 
                limit=LIMIT_CURRENT
            )
            
            if df_current.empty:
                logger.warning(f"  ⚠️ No 5m data for {symbol}")
                return {}
            
            data['5m'] = df_current
            logger.info(f"  ✅ 5m data: {len(df_current)} candles")
            
            # Fetch HTF data
            for tf in TIMEFRAMES['htf_list']:
                logger.info(f"  📊 Fetching {tf} data for {symbol}...")
                df_htf = self.data_fetcher.fetch_ohlcv(
                    symbol,
                    timeframe=tf,
                    limit=LIMIT_HTF
                )
                
                if not df_htf.empty:
                    data[tf] = df_htf
                    logger.info(f"  ✅ {tf} data: {len(df_htf)} candles")
                else:
                    logger.warning(f"  ⚠️ No {tf} data for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"  ❌ Error fetching data for {symbol}: {e}")
            self.errors.append({
                'symbol': symbol,
                'stage': 'data_fetch',
                'error': str(e)
            })
            return {}
    
    def test_symbol(self, symbol: str, index: int, total: int) -> Dict:
        """
        Test SMC Expert on a single symbol
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"[{index}/{total}] TESTING: {symbol}")
        logger.info(f"{'='*70}")
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_tf': TIMEFRAMES['current'],
            'signals': [],
            'order_blocks_count': 0,
            'fvgs_count': 0,
            'sweeps_count': 0,
            'errors': []
        }
        
        try:
            # Fetch all data
            data = self.fetch_all_data(symbol)
            
            if not data or '5m' not in data:
                result['errors'].append("No 5m data available")
                return result
            
            # Prepare HTF data dictionary
            htf_data = {}
            for tf in TIMEFRAMES['htf_list']:
                if tf in data:
                    htf_data[tf] = data[tf]
            
            logger.info(f"  🔧 HTF data available: {list(htf_data.keys())}")
            
            # Run SMC analysis
            logger.info(f"  🧠 Running SMC analysis...")
            start_time = time.time()
            
            signal = self.smc.analyze(
                df=data['5m'],
                symbol=symbol,
                timeframe=TIMEFRAMES['current'],
                htf_data=htf_data if htf_data else None
            )
            
            duration = (time.time() - start_time) * 1000
            
            if signal:
                logger.info(f"  ✅✅✅ SIGNAL GENERATED! ✅✅✅")
                logger.info(f"     📈 Direction: {signal['direction']}")
                logger.info(f"     💰 Entry: {signal['entry']:.4f}")
                logger.info(f"     🛑 Stop Loss: {signal['stop_loss']:.4f}")
                logger.info(f"     🎯 Take Profit: {signal['take_profit']:.4f}")
                logger.info(f"     📊 Risk/Reward: {signal['risk_reward']:.2f}")
                logger.info(f"     ⭐ Confidence: {signal['confidence']:.1%}")
                logger.info(f"     🏆 Grade: {signal['grade']}")
                logger.info(f"     ⚡ Action: {signal['action']}")
                logger.info(f"     🔮 Pattern: {signal['pattern_name']}")
                logger.info(f"     ⏱️ Time: {duration:.0f}ms")
                
                result['signals'].append(signal)
                
                # Update summary
                self.summary['signals_generated'] += 1
                
                grade = signal.get('grade', 'F')
                self.summary['grade_distribution'][grade] = self.summary['grade_distribution'].get(grade, 0) + 1
                
                pattern = signal.get('pattern_name', 'UNKNOWN')
                self.summary['pattern_distribution'][pattern] = self.summary['pattern_distribution'].get(pattern, 0) + 1
                
                # Update confidence and RR averages
                all_confidences = []
                all_rrs = []
                for r in self.results:
                    for s in r.get('signals', []):
                        all_confidences.append(s.get('confidence', 0))
                        all_rrs.append(s.get('risk_reward', 0))
                if all_confidences:
                    self.summary['avg_confidence'] = np.mean(all_confidences)
                if all_rrs:
                    self.summary['avg_rr'] = np.mean(all_rrs)
                
            else:
                logger.info(f"  ❌ NO SIGNAL - No setup found")
                logger.info(f"     ⏱️ Time: {duration:.0f}ms")
                logger.info(f"     💡 Check debug logs for details")
            
            return result
            
        except Exception as e:
            logger.error(f"  💥 ERROR testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
            result['errors'].append(str(e))
            self.errors.append({
                'symbol': symbol,
                'stage': 'smc_analysis',
                'error': str(e)
            })
            
            return result
    
    def run_full_test(self, symbols: List[str]) -> pd.DataFrame:
        """
        Run complete test on all symbols
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 SMC EXPERT V3 - COMPLETE TEST")
        logger.info("="*80)
        logger.info(f"📊 Symbols to test: {len(symbols)}")
        logger.info(f"⏱️ Current TF: {TIMEFRAMES['current']}")
        logger.info(f"📈 HTF TFs: {TIMEFRAMES['htf_list']}")
        logger.info(f"💾 Current limit: {LIMIT_CURRENT} candles")
        logger.info(f"💾 HTF limit: {LIMIT_HTF} candles")
        logger.info("="*80)
        
        self.summary['total_symbols'] = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            result = self.test_symbol(symbol, i, len(symbols))
            self.results.append(result)
            
            # Small delay between symbols to avoid rate limits
            time.sleep(0.5)
        
        # Calculate final summary
        self._calculate_final_summary()
        
        return self._generate_report()
    
    def _calculate_final_summary(self):
        """Calculate final summary statistics"""
        all_confidences = []
        all_rrs = []
        
        for result in self.results:
            for signal in result['signals']:
                all_confidences.append(signal.get('confidence', 0))
                all_rrs.append(signal.get('risk_reward', 0))
        
        if all_confidences:
            self.summary['avg_confidence'] = np.mean(all_confidences)
        if all_rrs:
            self.summary['avg_rr'] = np.mean(all_rrs)
        
        self.summary['errors_encountered'] = len(self.errors)
    
    def _generate_report(self) -> pd.DataFrame:
        """Generate test report"""
        report_data = []
        
        for result in self.results:
            if result['signals']:
                for signal in result['signals']:
                    report_data.append({
                        'Symbol': result['symbol'],
                        'Timestamp': result['timestamp'],
                        'Direction': signal['direction'],
                        'Entry': signal['entry'],
                        'Stop Loss': signal['stop_loss'],
                        'Take Profit': signal['take_profit'],
                        'Risk/Reward': signal['risk_reward'],
                        'Confidence': f"{signal['confidence']:.1%}",
                        'Grade': signal['grade'],
                        'Action': signal['action'],
                        'Pattern': signal['pattern_name'],
                        'Errors': len(result['errors'])
                    })
            else:
                report_data.append({
                    'Symbol': result['symbol'],
                    'Timestamp': result['timestamp'],
                    'Direction': 'NO SIGNAL',
                    'Entry': '-',
                    'Stop Loss': '-',
                    'Take Profit': '-',
                    'Risk/Reward': '-',
                    'Confidence': '-',
                    'Grade': '-',
                    'Action': 'SKIP',
                    'Pattern': '-',
                    'Errors': len(result['errors'])
                })
        
        return pd.DataFrame(report_data)
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"Total Symbols Tested:     {self.summary['total_symbols']}")
        print(f"Total Timeframes:         {1 + len(TIMEFRAMES['htf_list'])}")
        print(f"Signals Generated:        {self.summary['signals_generated']}")
        print(f"Errors Encountered:       {self.summary['errors_encountered']}")
        
        if self.summary['total_symbols'] > 0:
            success_rate = (self.summary['signals_generated'] / self.summary['total_symbols'] * 100)
            print(f"Success Rate:             {success_rate:.1f}%")
        
        print("-"*40)
        print(f"Average Confidence:       {self.summary['avg_confidence']:.1%}")
        print(f"Average RR:               {self.summary['avg_rr']:.2f}")
        print("-"*40)
        
        print("Grade Distribution:")
        if self.summary['grade_distribution']:
            for grade, count in sorted(self.summary['grade_distribution'].items()):
                print(f"  {grade}: {count}")
        else:
            print("  No signals generated")
        
        print("-"*40)
        print("Pattern Distribution:")
        if self.summary['pattern_distribution']:
            for pattern, count in sorted(self.summary['pattern_distribution'].items()):
                print(f"  {pattern}: {count}")
        else:
            print("  No patterns detected")
        
        print("="*80)
        
        # Debug log location
        print(f"\n📁 Debug logs saved to: logs/smc_expert/")
        print(f"   Check the .json file for detailed analysis")
    
    def save_results(self, df_report: pd.DataFrame):
        """Save results to CSV files"""
        # Save to signals folder if it exists
        signals_dir = "signals/smc_test"
        os.makedirs(signals_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main report
        report_file = os.path.join(signals_dir, f"smc_test_report_{timestamp}.csv")
        df_report.to_csv(report_file, index=False)
        print(f"\n📁 Results saved to: {report_file}")
        
        # Save detailed results
        details_file = os.path.join(signals_dir, f"smc_test_details_{timestamp}.json")
        with open(details_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 Details saved to: {details_file}")
        
        # Save errors if any
        if self.errors:
            errors_file = os.path.join(signals_dir, f"smc_test_errors_{timestamp}.csv")
            df_errors = pd.DataFrame(self.errors)
            df_errors.to_csv(errors_file, index=False)
            print(f"📁 Errors saved to: {errors_file}")
    
    def quick_verify(self) -> Dict:
        """
        Quick verification that SMC Expert imports correctly
        """
        print("\n" + "="*80)
        print("🔧 QUICK VERIFICATION")
        print("="*80)
        
        checks = {}
        
        # Check imports
        try:
            from smc_expert.smc_core import Direction, Candle, OrderBlock, FVG
            checks['smc_core'] = True
            print("✅ smc_core.py - OK")
        except Exception as e:
            checks['smc_core'] = False
            print(f"❌ smc_core.py - {e}")
        
        try:
            from smc_expert.smc_blocks import OrderBlockDetector
            checks['smc_blocks'] = True
            print("✅ smc_blocks.py - OK")
        except Exception as e:
            checks['smc_blocks'] = False
            print(f"❌ smc_blocks.py - {e}")
        
        try:
            from smc_expert.smc_gaps import FVGManager
            checks['smc_gaps'] = True
            print("✅ smc_gaps.py - OK")
        except Exception as e:
            checks['smc_gaps'] = False
            print(f"❌ smc_gaps.py - {e}")
        
        try:
            from smc_expert.smc_patterns import ICTPatternManager
            checks['smc_patterns'] = True
            print("✅ smc_patterns.py - OK")
        except Exception as e:
            checks['smc_patterns'] = False
            print(f"❌ smc_patterns.py - {e}")
        
        try:
            from smc_expert.smc_factory import SMCFactory
            checks['smc_factory'] = True
            print("✅ smc_factory.py - OK")
        except Exception as e:
            checks['smc_factory'] = False
            print(f"❌ smc_factory.py - {e}")
        
        print("="*80)
        
        all_ok = all(checks.values())
        if all_ok:
            print("✅ All SMC modules imported successfully!")
        else:
            print("❌ Some modules failed to import. Check the errors above.")
        
        return checks


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("🚀 SMC EXPERT V3 - VALIDATION TEST")
    print("="*80)
    print("This test will:")
    print("  1. Verify all SMC modules import correctly")
    print("  2. Fetch real data for 20+ coins")
    print("  3. Test with 5m + HTFs (15m, 1h, 4h, 1d)")
    print("  4. Generate trading signals")
    print("  5. Save results for analysis")
    print("="*80)
    
    tester = SMCTester()
    
    # Step 1: Quick verification
    print("\n🔧 STEP 1: Module Verification")
    checks = tester.quick_verify()
    
    if not all(checks.values()):
        print("\n❌ Cannot proceed - SMC modules not working properly.")
        print("   Make sure all SMC files are in the smc_expert/ folder.")
        return
    
    # Step 2: Ask about full test
    print("\n📊 STEP 2: Full Test")
    print("-"*40)
    
    response = input("\nRun full test on 20 coins? (y/n): ")
    
    if response.lower() == 'y':
        # Run full test
        df_report = tester.run_full_test(TEST_SYMBOLS)
        
        # Print summary
        tester.print_summary()
        
        # Save results
        tester.save_results(df_report)
        
        # Print sample signals
        signals_df = df_report[df_report['Direction'] != 'NO SIGNAL']
        if not signals_df.empty:
            print("\n📈 SAMPLE SIGNALS GENERATED:")
            print(signals_df[['Symbol', 'Direction', 'Entry', 'Risk/Reward', 'Confidence', 'Grade']].head(10).to_string())
        else:
            print("\n⚠️ No signals generated in this test run")
            print("   Possible reasons:")
            print("   - Market conditions are not favorable")
            print("   - Try during kill zones (London 8-9 AM UTC, NY 2-3 PM UTC)")
            print("   - Check debug logs for specific rejections")
            print("\n   To see detailed rejections, check:")
            print("   logs/smc_expert/YYYY-MM-DD.json")
    else:
        print("\nSkipping full test. Quick verification only.")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()