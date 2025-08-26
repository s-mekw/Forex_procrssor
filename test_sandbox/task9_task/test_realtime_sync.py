"""
RCIリアルタイムチャートのデータ同期テスト
修正後の動作確認用
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import time
import MetaTrader5 as mt5
from rci_realtime_chart import RCIRealtimeChart
from utils.config_loader import load_config
import threading

def test_realtime_sync(duration_seconds=30):
    """リアルタイムデータ同期のテスト"""
    print("="*60)
    print("Testing RCI Realtime Chart Data Synchronization")
    print(f"Duration: {duration_seconds} seconds")
    print("="*60)
    
    # MT5初期化
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    
    try:
        # チャートマネージャーを初期化
        config = load_config()
        chart = RCIRealtimeChart(config)
        
        # 初期データロード
        print("\n1. Loading initial data...")
        chart.fetch_initial_data()
        
        # データ同期の確認
        print("\n2. Initial data sync check:")
        ohlc_len = len(chart.ohlc_data)
        for period in config.all_rci_periods:
            rci_len = len(chart.rci_data[period])
            status = "✓ OK" if ohlc_len == rci_len else "✗ MISMATCH"
            print(f"  Period {period}: OHLC={ohlc_len}, RCI={rci_len} {status}")
        
        # リアルタイム受信開始
        print("\n3. Starting realtime data reception...")
        chart.start_realtime()
        
        # 指定時間動作させる
        print(f"\n4. Monitoring for {duration_seconds} seconds...")
        print("   (Debug logs will appear below)")
        print("-" * 40)
        
        start_time = time.time()
        last_check = start_time
        check_interval = 5  # 5秒ごとにチェック
        
        while time.time() - start_time < duration_seconds:
            time.sleep(1)
            
            # 定期的にデータ同期をチェック
            if time.time() - last_check >= check_interval:
                with chart.data_lock:
                    ohlc_len = len(chart.ohlc_data)
                    sync_ok = True
                    for period in config.all_rci_periods:
                        rci_len = len(chart.rci_data[period])
                        if ohlc_len != rci_len:
                            sync_ok = False
                            print(f"\n[SYNC CHECK] Period {period}: OHLC={ohlc_len}, RCI={rci_len} ✗ MISMATCH")
                    
                    if sync_ok:
                        print(f"\n[SYNC CHECK] All periods synchronized. OHLC length: {ohlc_len} ✓")
                
                last_check = time.time()
        
        print("-" * 40)
        
        # リアルタイム受信停止
        print("\n5. Stopping realtime data reception...")
        chart.stop_realtime()
        
        # 最終データ同期確認
        print("\n6. Final data sync check:")
        with chart.data_lock:
            ohlc_len = len(chart.ohlc_data)
            all_synced = True
            for period in config.all_rci_periods:
                rci_len = len(chart.rci_data[period])
                status = "✓ OK" if ohlc_len == rci_len else "✗ MISMATCH"
                print(f"  Period {period}: OHLC={ohlc_len}, RCI={rci_len} {status}")
                if ohlc_len != rci_len:
                    all_synced = False
        
        # 統計情報表示
        print("\n7. Statistics:")
        print(f"  Ticks received: {chart.stats['ticks_received']}")
        print(f"  Bars completed: {chart.stats['bars_completed']}")
        
        # 結果
        print("\n" + "="*60)
        if all_synced:
            print("✓ TEST PASSED: Data synchronization is working correctly!")
        else:
            print("✗ TEST FAILED: Data synchronization issues detected!")
        print("="*60)
        
        return all_synced
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    # 30秒間のテスト（必要に応じて調整）
    success = test_realtime_sync(duration_seconds=30)
    sys.exit(0 if success else 1)