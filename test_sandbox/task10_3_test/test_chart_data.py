"""
チャートデータの内容を詳しく確認するテストスクリプト
特にM5 RCIに価格データが混入していないか確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import asyncio
import MetaTrader5 as mt5
from pipeline_chart_dashboard import PipelineChartManager
import time

def test_chart_data():
    """チャートデータの内容を確認"""
    
    # 設定ファイルのパス
    config_path = Path(__file__).parent / 'task10_3_config.toml'
    
    # チャートマネージャー初期化
    try:
        manager = PipelineChartManager(config_path=str(config_path))
        print(f"✅ Initialized for symbol: {manager.symbol}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # 初期データ確認
    print("\n=== INITIAL DATA CHECK ===")
    
    # M1 OHLCデータ
    if manager.chart_data.m1_ohlc is not None:
        m1_ohlc = manager.chart_data.m1_ohlc
        print(f"\n📊 M1 OHLC:")
        print(f"  Bars: {len(m1_ohlc)}")
        print(f"  Close range: {m1_ohlc['close'].min():.5f} - {m1_ohlc['close'].max():.5f}")
        print(f"  Last 3 closes: {m1_ohlc['close'][-3:].to_list()}")
    
    # M5 OHLCデータ
    if manager.chart_data.m5_ohlc is not None:
        m5_ohlc = manager.chart_data.m5_ohlc
        print(f"\n📊 M5 OHLC:")
        print(f"  Bars: {len(m5_ohlc)}")
        print(f"  Close range: {m5_ohlc['close'].min():.5f} - {m5_ohlc['close'].max():.5f}")
        print(f"  Last 3 closes: {m5_ohlc['close'][-3:].to_list()}")
    
    # M1 RCIデータ
    print(f"\n📈 M1 RCI:")
    for period, values in manager.chart_data.m1_rci.items():
        if len(values) > 0:
            print(f"  Period {period}: {len(values)} values, range: {min(values):.2f} - {max(values):.2f}")
            print(f"    Last 3 values: {values[-3:]}")
    
    # M5 RCIデータ（重要）
    print(f"\n📈 M5 RCI (IMPORTANT):")
    for period, values in manager.chart_data.m5_rci.items():
        if len(values) > 0:
            print(f"  Period {period}: {len(values)} values")
            print(f"    Range: {min(values):.2f} - {max(values):.2f}")
            print(f"    First 5 values: {values[:5]}")
            print(f"    Last 5 values: {values[-5:]}")
            
            # 異常値チェック
            suspicious = [v for v in values if abs(v) > 150]
            if suspicious:
                print(f"    ⚠️ SUSPICIOUS VALUES (>150): {suspicious[:5]}")
                print(f"    ❌ This indicates price data contamination!")
            else:
                print(f"    ✅ All values within RCI range (-100 to 100)")
    
    # チャート作成時のデータを確認
    print("\n=== CHART CREATION DATA ===")
    
    # create_chartメソッドを呼び出してデータを確認
    with manager.data_lock:
        display_bars_m1 = manager.display_bars_m1
        display_bars_m5 = manager.display_bars_m5
        
        m1_ohlc = manager.chart_data.m1_ohlc.tail(display_bars_m1) if manager.chart_data.m1_ohlc is not None else None
        m5_ohlc = manager.chart_data.m5_ohlc.tail(display_bars_m5) if manager.chart_data.m5_ohlc is not None and not manager.chart_data.m5_ohlc.is_empty() else None
        m1_rci = {k: v[-display_bars_m1:] if len(v) > display_bars_m1 else v 
                 for k, v in manager.chart_data.m1_rci.items()}
        m5_rci = {k: v[-display_bars_m5:] if len(v) > display_bars_m5 else v 
                 for k, v in manager.chart_data.m5_rci.items()}
    
    print(f"\nDisplay bars M1: {display_bars_m1}, M5: {display_bars_m5}")
    
    if m5_ohlc is not None:
        print(f"\nM5 OHLC for display:")
        print(f"  Bars: {len(m5_ohlc)}")
        print(f"  Close range: {m5_ohlc['close'].min():.5f} - {m5_ohlc['close'].max():.5f}")
    
    print(f"\nM5 RCI for display:")
    for period, values in m5_rci.items():
        if len(values) > 0:
            print(f"  Period {period}: {len(values)} values")
            print(f"    Range: {min(values):.2f} - {max(values):.2f}")
            if max(values) > 150:
                print(f"    ❌ PRICE DATA DETECTED IN M5 RCI!")
                print(f"    Sample: {values[:3]}")
    
    # クリーンアップ
    if mt5.initialize():
        mt5.shutdown()
    
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_chart_data()