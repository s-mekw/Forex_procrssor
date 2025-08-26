"""
RCIデータ同期のテストスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from src.data_processing.rci import DifferentialRCICalculator

def test_data_sync():
    """データ同期のテスト"""
    print("="*60)
    print("Testing RCI Data Synchronization")
    print("="*60)
    
    # テスト用のOHLCデータを作成
    times = [datetime.now() - timedelta(minutes=i) for i in range(20, 0, -1)]
    prices = [1.10 + 0.001 * np.sin(i * 0.5) for i in range(20)]
    
    ohlc_data = pl.DataFrame({
        "time": times,
        "close": prices
    })
    
    print(f"\nInitial OHLC data: {len(ohlc_data)} bars")
    
    # RCI計算器を初期化
    calculator = DifferentialRCICalculator(period=9)
    rci_data = []
    
    # 初期データ処理（最後のバーを除く）
    print("\n1. Processing initial complete bars:")
    for i, price in enumerate(prices[:-1]):
        rci = calculator.add(price)
        rci_data.append(rci)
        if rci is not None and i >= 15:
            print(f"  Bar {i+1}: Price={price:.5f}, RCI={rci:.2f}")
    
    # 最後のバー（未完成）
    print("\n2. Processing incomplete bar:")
    last_price = prices[-1]
    preview_rci = calculator.preview(last_price)
    rci_data.append(preview_rci)
    print(f"  Incomplete bar: Price={last_price:.5f}, Preview RCI={preview_rci:.2f}")
    
    print(f"\nData lengths - OHLC: {len(ohlc_data)}, RCI: {len(rci_data)}")
    assert len(ohlc_data) == len(rci_data), "Data length mismatch!"
    
    # 新しいバーのシミュレーション
    print("\n3. Simulating new bar completion:")
    new_time = datetime.now()
    new_price = 1.105
    
    # バー完成時の処理
    print("  - Completing current bar")
    completed_rci = calculator.add(last_price)
    rci_data[-1] = completed_rci  # 最後の値を更新
    print(f"    Completed: Price={last_price:.5f}, RCI={completed_rci:.2f}")
    
    # 新しい未完成バー
    print("  - Starting new incomplete bar")
    ohlc_data = pl.concat([
        ohlc_data,
        pl.DataFrame({"time": [new_time], "close": [new_price]})
    ])
    preview_rci = calculator.preview(new_price)
    rci_data.append(preview_rci)
    print(f"    New incomplete: Price={new_price:.5f}, Preview RCI={preview_rci:.2f}")
    
    print(f"\nFinal data lengths - OHLC: {len(ohlc_data)}, RCI: {len(rci_data)}")
    assert len(ohlc_data) == len(rci_data), "Data length mismatch after update!"
    
    # RCIデータの検証
    print("\n4. Validating RCI data:")
    none_count = sum(1 for x in rci_data if x is None)
    valid_count = sum(1 for x in rci_data if x is not None)
    print(f"  None values: {none_count}")
    print(f"  Valid values: {valid_count}")
    print(f"  Total: {len(rci_data)}")
    
    # 最後の10個のRCI値を表示
    print("\n5. Last 10 RCI values:")
    for i, rci in enumerate(rci_data[-10:], start=len(rci_data)-9):
        status = "incomplete" if i == len(rci_data) else "complete"
        if rci is not None:
            print(f"  Index {i}: {rci:.2f} ({status})")
        else:
            print(f"  Index {i}: None ({status})")

def test_bar_replacement():
    """バー置き換えのテスト"""
    print("\n" + "="*60)
    print("Testing Bar Replacement Logic")
    print("="*60)
    
    calculator = DifferentialRCICalculator(period=9)
    
    # 初期データ
    prices = [1.10 + i * 0.001 for i in range(10)]
    rci_data = []
    
    # 完成バーを追加
    for price in prices[:-1]:
        rci = calculator.add(price)
        rci_data.append(rci)
    
    # 未完成バー
    incomplete_price = prices[-1]
    preview_rci = calculator.preview(incomplete_price)
    rci_data.append(preview_rci)
    
    print(f"\nBefore replacement:")
    print(f"  RCI data length: {len(rci_data)}")
    print(f"  Last RCI (preview): {rci_data[-1]:.2f}" if rci_data[-1] else "  Last RCI: None")
    
    # バーを完成させる（置き換え）
    final_price = incomplete_price + 0.0005
    final_rci = calculator.add(final_price)
    rci_data[-1] = final_rci  # 置き換え
    
    print(f"\nAfter replacement:")
    print(f"  RCI data length: {len(rci_data)} (should be same)")
    print(f"  Last RCI (final): {rci_data[-1]:.2f}" if rci_data[-1] else "  Last RCI: None")
    
    # 新しいバーを追加
    new_price = 1.111
    new_rci = calculator.add(new_price)
    rci_data.append(new_rci)
    
    print(f"\nAfter new bar:")
    print(f"  RCI data length: {len(rci_data)} (should be +1)")
    print(f"  Last RCI: {rci_data[-1]:.2f}" if rci_data[-1] else "  Last RCI: None")

if __name__ == "__main__":
    test_data_sync()
    test_bar_replacement()
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)