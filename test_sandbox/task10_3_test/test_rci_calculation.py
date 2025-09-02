"""
RCI計算のテストスクリプト
M5 RCIの値が正しく計算されているか確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import MetaTrader5 as mt5
import polars as pl
import numpy as np
from datetime import datetime
from src.data_processing.rci import RCICalculatorEngine

def test_m5_rci_calculation():
    """M5 RCIの計算をテスト"""
    
    # MT5初期化
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return
    
    # M5データ取得
    symbol = "EURJPY#"
    rates_m5 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M5,
        0,
        200  # 200本取得
    )
    
    if rates_m5 is None or len(rates_m5) == 0:
        print("❌ Failed to fetch M5 data")
        mt5.shutdown()
        return
    
    print(f"✅ Fetched {len(rates_m5)} M5 bars")
    
    # DataFrame作成
    df_m5 = pl.DataFrame({
        "timestamp": [datetime.fromtimestamp(r['time']) for r in rates_m5],
        "open": np.array([r['open'] for r in rates_m5], dtype=np.float32),
        "high": np.array([r['high'] for r in rates_m5], dtype=np.float32),
        "low": np.array([r['low'] for r in rates_m5], dtype=np.float32),
        "close": np.array([r['close'] for r in rates_m5], dtype=np.float32),
        "volume": np.array([r['tick_volume'] for r in rates_m5], dtype=np.float32)
    })
    
    # 価格データの範囲を確認
    print(f"\n📊 M5 OHLC Data Range:")
    print(f"  Close: min={df_m5['close'].min():.5f}, max={df_m5['close'].max():.5f}")
    print(f"  High:  min={df_m5['high'].min():.5f}, max={df_m5['high'].max():.5f}")
    print(f"  Low:   min={df_m5['low'].min():.5f}, max={df_m5['low'].max():.5f}")
    
    # RCI計算
    rci_engine = RCICalculatorEngine()
    periods = [24, 33, 48, 66, 108]
    
    print(f"\n🔬 Calculating RCI for periods: {periods}")
    
    for period in periods:
        if len(df_m5) >= period:
            # 最新のperiod本でRCI計算
            window_df = df_m5[-period:]
            
            result = rci_engine.calculate_multiple(
                data=window_df,
                periods=[period],
                mode="batch"
            )
            
            if f"rci_{period}" in result.columns:
                rci_values = result[f"rci_{period}"].to_list()
                
                # RCIの範囲を確認
                print(f"\n📈 RCI[{period}]:")
                print(f"  Values: {len(rci_values)}")
                
                # None値をフィルタリング
                valid_values = [v for v in rci_values if v is not None]
                if valid_values:
                    print(f"  Range: min={min(valid_values):.2f}, max={max(valid_values):.2f}")
                    if rci_values[-1] is not None:
                        print(f"  Latest value: {rci_values[-1]:.2f}")
                    else:
                        print(f"  Latest value: None")
                else:
                    print(f"  ⚠️ All values are None!")
                    print(f"  Raw values: {rci_values}")
                
                # 異常値チェック
                if valid_values:
                    if max(valid_values) > 100 or min(valid_values) < -100:
                        print(f"  ⚠️ WARNING: Values outside -100 to 100 range!")
                        print(f"  Sample values: {valid_values[:5]}")
                    else:
                        print(f"  ✅ All values within valid range (-100 to 100)")
    
    # クリーンアップ
    mt5.shutdown()
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_m5_rci_calculation()