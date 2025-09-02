"""
RCI修正の動作確認テストスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.data_processing.rci import DifferentialRCICalculator
import numpy as np

def test_preview_method():
    """previewメソッドの動作確認"""
    print("="*60)
    print("Testing DifferentialRCICalculator.preview() method")
    print("="*60)
    
    # 期間9のRCI計算器を作成
    calculator = DifferentialRCICalculator(period=9)
    
    # テスト用の価格データ
    test_prices = [
        1.10000, 1.10010, 1.10005, 1.10015, 1.10020,
        1.10012, 1.10025, 1.10018, 1.10030
    ]
    
    print("\nAdding prices sequentially...")
    for i, price in enumerate(test_prices):
        rci = calculator.add(price)
        if rci is not None:
            print(f"Price {i+1}: {price:.5f} -> RCI: {rci:.2f}")
        else:
            print(f"Price {i+1}: {price:.5f} -> RCI: Not ready (need {9} prices)")
    
    # previewメソッドのテスト
    print("\n" + "="*60)
    print("Testing preview with different prices:")
    print("="*60)
    
    test_preview_prices = [1.09990, 1.10020, 1.10040]
    
    for preview_price in test_preview_prices:
        preview_rci = calculator.preview(preview_price)
        print(f"Preview price: {preview_price:.5f} -> RCI: {preview_rci:.2f}")
    
    # 最後のRCIが変わっていないことを確認
    print(f"\nLast calculated RCI (unchanged): {calculator.last_rci:.2f}")
    
    # 実際に新しい価格を追加
    print("\n" + "="*60)
    print("Adding new price normally:")
    print("="*60)
    
    new_price = 1.10035
    new_rci = calculator.add(new_price)
    print(f"Added price: {new_price:.5f} -> RCI: {new_rci:.2f}")
    
    # バッファの状態確認
    print("\n" + "="*60)
    print("Buffer state:")
    print("="*60)
    buffer_state = calculator.get_buffer_state()
    print(f"Period: {buffer_state['period']}")
    print(f"Buffer size: {buffer_state['buffer_size']}")
    print(f"Calculation count: {buffer_state['calculation_count']}")
    print(f"Current prices: {[f'{p:.5f}' for p in buffer_state['current_prices'][-5:]]}")

def test_consistency():
    """previewと実際のaddの一貫性テスト"""
    print("\n" + "="*60)
    print("Testing consistency between preview and add")
    print("="*60)
    
    calculator = DifferentialRCICalculator(period=9)
    
    # 初期データを追加
    prices = np.random.randn(20) * 0.001 + 1.10
    
    for price in prices[:9]:
        calculator.add(float(price))
    
    # 同じ価格でpreviewとaddを比較
    test_price = 1.10050
    
    # preview結果
    preview_result = calculator.preview(test_price)
    print(f"Preview result for {test_price:.5f}: {preview_result:.2f}")
    
    # 実際に追加
    add_result = calculator.add(test_price)
    print(f"Add result for {test_price:.5f}: {add_result:.2f}")
    
    # 差を確認
    if preview_result is not None and add_result is not None:
        diff = abs(preview_result - add_result)
        print(f"Difference: {diff:.6f}")
        if diff < 0.001:
            print("✓ Preview and add results are consistent!")
        else:
            print("✗ Warning: Preview and add results differ significantly!")

if __name__ == "__main__":
    test_preview_method()
    test_consistency()
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)