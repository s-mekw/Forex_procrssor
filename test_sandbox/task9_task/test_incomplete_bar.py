"""
未完成バー処理のテストスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.data_processing.rci import DifferentialRCICalculator
import numpy as np

def test_incomplete_bar_handling():
    """未完成バーの処理をテスト"""
    print("="*60)
    print("Testing Incomplete Bar Handling")
    print("="*60)
    
    # 期間9のRCI計算器を作成
    calculator = DifferentialRCICalculator(period=9)
    
    # 初期データ（完成バー）を追加
    initial_prices = [
        1.10000, 1.10010, 1.10005, 1.10015, 1.10020,
        1.10012, 1.10025, 1.10018  # 8個の完成バー
    ]
    
    print("\n1. Adding initial complete bars (8 bars):")
    for i, price in enumerate(initial_prices):
        rci = calculator.add(price)
        if rci is not None:
            print(f"  Bar {i+1}: {price:.5f} -> RCI: {rci:.2f}")
        else:
            print(f"  Bar {i+1}: {price:.5f} -> RCI: Not ready")
    
    # 未完成バーのシミュレーション
    print("\n2. Simulating incomplete bar (9th bar):")
    incomplete_prices = [1.10030, 1.10032, 1.10028, 1.10035]  # ティックごとの価格変動
    
    for tick_price in incomplete_prices:
        preview_rci = calculator.preview(tick_price)
        print(f"  Tick update: {tick_price:.5f} -> Preview RCI: {preview_rci:.2f}")
    
    # バー完成
    print("\n3. Completing the bar:")
    final_price = 1.10033
    completed_rci = calculator.add(final_price)
    print(f"  Bar completed: {final_price:.5f} -> RCI: {completed_rci:.2f}")
    
    # 次の未完成バー
    print("\n4. Next incomplete bar (10th bar):")
    next_incomplete = [1.10040, 1.10038, 1.10042]
    
    for tick_price in next_incomplete:
        preview_rci = calculator.preview(tick_price)
        print(f"  Tick update: {tick_price:.5f} -> Preview RCI: {preview_rci:.2f}")
    
    # バッファの状態確認
    print("\n5. Buffer state:")
    buffer_state = calculator.get_buffer_state()
    print(f"  Period: {buffer_state['period']}")
    print(f"  Buffer size: {buffer_state['buffer_size']}")
    print(f"  Calculation count: {buffer_state['calculation_count']}")
    print(f"  Last 5 prices: {[f'{p:.5f}' for p in buffer_state['current_prices'][-5:]]}")

def test_preview_logic():
    """preview()メソッドのロジックテスト"""
    print("\n" + "="*60)
    print("Testing Preview Logic (Old vs New)")
    print("="*60)
    
    calculator = DifferentialRCICalculator(period=9)
    
    # 9個の価格でウィンドウをフルにする
    prices = [1.10 + i * 0.001 for i in range(9)]
    for price in prices:
        calculator.add(price)
    
    print(f"\nInitial prices: {[f'{p:.3f}' for p in prices]}")
    print(f"Last calculated RCI: {calculator.last_rci:.2f}")
    
    # 未完成バーのpreview
    incomplete_price = 1.11000
    preview_rci = calculator.preview(incomplete_price)
    
    print(f"\nPreview with {incomplete_price:.5f}: RCI = {preview_rci:.2f}")
    print("This should use prices[1:] + [incomplete_price]")
    print(f"Actual window: [{', '.join([f'{p:.3f}' for p in prices[1:]])}] + [{incomplete_price:.3f}]")
    
    # 実際にaddした場合
    actual_rci = calculator.add(incomplete_price)
    print(f"\nActual add with {incomplete_price:.5f}: RCI = {actual_rci:.2f}")
    print("This shifts the window: removes oldest, adds newest")

if __name__ == "__main__":
    test_incomplete_bar_handling()
    test_preview_logic()
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)