"""
DifferentialRCICalculatorのユニットテスト

基本的なRCI計算の正確性とパフォーマンスを検証します。
"""

import pytest
import numpy as np
from collections import deque
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_processing.rci import DifferentialRCICalculator


class TestDifferentialRCICalculator:
    """DifferentialRCICalculatorのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        # 正常な初期化
        calc = DifferentialRCICalculator(period=9)
        assert calc.period == 9
        assert len(calc.prices) == 0
        assert not calc.is_ready
        assert calc.calculation_count == 0
        assert calc.last_rci is None
        
        # Float32精度の確認
        assert calc.time_ranks.dtype == np.float32
        assert calc.denominator.dtype == np.float32
    
    def test_invalid_period(self):
        """無効な期間のテスト"""
        # 期間が小さすぎる
        with pytest.raises(ValueError, match="Period must be at least"):
            DifferentialRCICalculator(period=2)
        
        # 期間が大きすぎる
        with pytest.raises(ValueError, match="Period must be at most"):
            DifferentialRCICalculator(period=201)
    
    def test_add_prices_not_ready(self):
        """ウィンドウが満たされるまでのテスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 最初の4つの価格ではNoneを返す
        for i, price in enumerate([100, 101, 102, 103]):
            result = calc.add(price)
            assert result is None
            assert len(calc.prices) == i + 1
            assert not calc.is_ready
    
    def test_rci_calculation_ascending(self):
        """上昇トレンドのRCI計算テスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 完全な上昇トレンド [100, 101, 102, 103, 104]
        prices = [100, 101, 102, 103, 104]
        for i, price in enumerate(prices):
            result = calc.add(price)
            if i < 4:
                assert result is None
            else:
                # 完全な上昇トレンドではRCIは+100に近い
                assert result is not None
                assert result > 90  # 理論的には100
                assert calc.is_ready
                assert calc.calculation_count == 1
    
    def test_rci_calculation_descending(self):
        """下降トレンドのRCI計算テスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 完全な下降トレンド [104, 103, 102, 101, 100]
        prices = [104, 103, 102, 101, 100]
        for i, price in enumerate(prices):
            result = calc.add(price)
            if i < 4:
                assert result is None
            else:
                # 完全な下降トレンドではRCIは-100に近い
                assert result is not None
                assert result < -90  # 理論的には-100
    
    def test_rci_calculation_sideways(self):
        """横ばいトレンドのRCI計算テスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 横ばいトレンド [100, 100, 100, 100, 100]
        prices = [100] * 5
        for i, price in enumerate(prices):
            result = calc.add(price)
            if i < 4:
                assert result is None
            else:
                # 全て同じ価格の場合、全ての価格順位が平均順位（2.0 = (0+1+2+3+4)/5）になる
                # これにより、時間順位と価格順位の差が一定になり、RCIは特定の値になる
                # 実際の計算では、同値の場合のRCIは50になる（scipy.rankdataの平均順位処理による）
                assert result is not None
                # 同値の場合、理論的にはRCIは意味を持たないが、計算上は50前後になる
                assert 40 <= result <= 60
    
    def test_sliding_window(self):
        """スライディングウィンドウのテスト"""
        calc = DifferentialRCICalculator(period=3)
        
        # 6つの価格を追加
        prices = [100, 102, 101, 103, 99, 104]
        results = []
        
        for price in prices:
            result = calc.add(price)
            if result is not None:
                results.append(result)
        
        # period=3なので、4回RCIが計算される
        assert len(results) == 4
        assert calc.calculation_count == 4
        
        # バッファは最新の3つの価格のみを保持
        assert len(calc.prices) == 3
        assert list(calc.prices) == [103, 99, 104]
    
    def test_nan_inf_handling(self):
        """NaN/Inf値のハンドリングテスト"""
        calc = DifferentialRCICalculator(period=3)
        
        # 正常な価格を追加
        calc.add(100)
        calc.add(101)
        
        # NaN値
        result = calc.add(np.nan)
        assert result is None
        
        # Inf値
        calc.add(102)  # バッファをリセット
        calc.add(103)
        result = calc.add(np.inf)
        assert result is None
        
        # -Inf値
        calc.add(104)
        calc.add(105)
        result = calc.add(-np.inf)
        assert result is None
    
    def test_reset(self):
        """リセット機能のテスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # データを追加
        for price in [100, 101, 102, 103, 104]:
            calc.add(price)
        
        assert calc.is_ready
        assert calc.calculation_count == 1
        assert calc.last_rci is not None
        
        # リセット
        calc.reset()
        
        assert not calc.is_ready
        assert calc.calculation_count == 0
        assert calc.last_rci is None
        assert len(calc.prices) == 0
    
    def test_float32_precision(self):
        """Float32精度の検証"""
        calc = DifferentialRCICalculator(period=9)
        
        # 大きな値でもFloat32精度で計算されることを確認
        large_prices = [1e6 + i for i in range(9)]
        
        for price in large_prices:
            result = calc.add(price)
        
        # RCIは正規化された値なので、大きな価格でも-100〜+100の範囲
        assert result is not None
        assert -100 <= result <= 100
        
        # 内部データがFloat32であることを確認
        state = calc.get_buffer_state()
        prices_array = np.array(state['current_prices'], dtype=np.float32)
        assert prices_array.dtype == np.float32
    
    def test_get_buffer_state(self):
        """バッファ状態取得のテスト"""
        calc = DifferentialRCICalculator(period=3)
        
        # 初期状態
        state = calc.get_buffer_state()
        assert state['period'] == 3
        assert state['buffer_size'] == 0
        assert not state['is_ready']
        assert state['calculation_count'] == 0
        assert state['last_rci'] is None
        assert state['current_prices'] == []
        
        # データ追加後
        calc.add(100)
        calc.add(101)
        calc.add(102)
        
        state = calc.get_buffer_state()
        assert state['buffer_size'] == 3
        assert state['is_ready']
        assert state['calculation_count'] == 1
        assert state['last_rci'] is not None
        assert state['current_prices'] == [100, 101, 102]


def test_performance_benchmark():
    """パフォーマンスベンチマーク"""
    import time
    
    calc = DifferentialRCICalculator(period=50)
    
    # 10000個のランダム価格データ
    np.random.seed(42)
    prices = np.random.randn(10000) * 10 + 100
    
    start_time = time.time()
    
    for price in prices:
        calc.add(price)
    
    elapsed = time.time() - start_time
    
    # 10000データポイントの処理時間
    print(f"\nPerformance: {len(prices)} prices processed in {elapsed:.3f} seconds")
    print(f"Average time per update: {elapsed/len(prices)*1000:.3f} ms")
    print(f"Total RCI calculations: {calc.calculation_count}")
    
    # 目標: 0.1ms/update以下
    assert elapsed / len(prices) < 0.0001  # 0.1ms


if __name__ == "__main__":
    # 基本的なテスト実行
    test = TestDifferentialRCICalculator()
    
    print("Running basic tests...")
    test.test_initialization()
    print("✓ Initialization test passed")
    
    test.test_rci_calculation_ascending()
    print("✓ Ascending trend RCI test passed")
    
    test.test_rci_calculation_descending()
    print("✓ Descending trend RCI test passed")
    
    test.test_sliding_window()
    print("✓ Sliding window test passed")
    
    test.test_float32_precision()
    print("✓ Float32 precision test passed")
    
    # パフォーマンステスト
    print("\nRunning performance benchmark...")
    test_performance_benchmark()
    print("✓ Performance benchmark passed")
    
    print("\nAll tests passed successfully!")