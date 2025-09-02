"""
M5 RCIデータの内容を詳しく調査するデバッグスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import MetaTrader5 as mt5
import polars as pl
import numpy as np
from datetime import datetime
from src.data_processing.rci import RCICalculatorEngine

def debug_m5_rci():
    """M5 RCIデータの詳細デバッグ"""
    
    # MT5初期化
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return
    
    symbol = "EURJPY#"
    
    # M5データ取得（少量）
    rates_m5 = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M5,
        0,
        50  # 50本のみ
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
    
    # 最新のいくつかの価格データを表示
    print("\n📊 Latest M5 OHLC Data (last 5 bars):")
    for i in range(-5, 0):
        print(f"  Bar {i}: O={df_m5['open'][i]:.5f}, H={df_m5['high'][i]:.5f}, L={df_m5['low'][i]:.5f}, C={df_m5['close'][i]:.5f}")
    
    # RCI計算（期間24のみ）
    rci_engine = RCICalculatorEngine()
    period = 24
    
    print(f"\n🔬 Calculating RCI for period {period}")
    
    # calculate_rci_historyメソッドと同じロジック
    rci_history = []
    for i in range(period, len(df_m5) + 1):
        window_df = df_m5[i-period:i]
        
        result = rci_engine.calculate_multiple(
            data=window_df,
            periods=[period],
            mode="batch"
        )
        
        if f"rci_{period}" in result.columns:
            # 全ての値を確認
            rci_column = result[f"rci_{period}"]
            print(f"\n  Window {i-period}:{i}")
            print(f"    RCI column length: {len(rci_column)}")
            print(f"    RCI column values: {rci_column.to_list()}")
            
            # 最後の値だけを取得（通常の処理）
            rci_value = rci_column[-1]
            if rci_value is not None:
                rci_history.append(float(rci_value))
                print(f"    Selected value: {float(rci_value):.2f}")
            else:
                print(f"    Selected value: None")
    
    print(f"\n📈 Final RCI history for period {period}:")
    print(f"  Total values: {len(rci_history)}")
    if rci_history:
        print(f"  Range: min={min(rci_history):.2f}, max={max(rci_history):.2f}")
        print(f"  First 5 values: {rci_history[:5]}")
        print(f"  Last 5 values: {rci_history[-5:]}")
        
        # 価格のような値があるかチェック
        suspicious = [v for v in rci_history if abs(v) > 150]
        if suspicious:
            print(f"  ⚠️ SUSPICIOUS VALUES (>150): {suspicious}")
    
    mt5.shutdown()
    print("\n✅ Debug completed")

if __name__ == "__main__":
    debug_m5_rci()