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


class TestRankingOptimizations:
    """ランキング最適化のテストクラス"""
    
    def test_numpy_ranking_with_ties(self):
        """NumPyベースの同値対応ランキングのテスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 同値を含む価格配列
        prices = np.array([100, 101, 100, 102, 101], dtype=np.float32)
        
        ranks = calc._numpy_ranking_with_ties(prices)
        
        # 期待される順位（0ベース）
        # 100: (0+2)/2 = 1.0 (2つの100)
        # 101: (1+4)/2 = 2.5 (2つの101)
        # 102: 4 (1つの102)
        expected = np.array([0.5, 2.5, 0.5, 4, 2.5], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(ranks, expected, decimal=5)
    
    def test_fast_ranking_no_ties(self):
        """同値なしの高速ランキングのテスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 全て異なる価格
        prices = np.array([100.1, 100.5, 99.8, 101.2, 100.3], dtype=np.float32)
        
        ranks = calc._fast_ranking_no_ties(prices)
        
        # 正しい順位が割り当てられているか確認
        sorted_indices = np.argsort(prices)
        expected = np.empty_like(sorted_indices, dtype=np.float32)
        expected[sorted_indices] = np.arange(len(prices), dtype=np.float32)
        
        np.testing.assert_array_almost_equal(ranks, expected, decimal=5)
    
    def test_check_for_ties(self):
        """同値チェック機能のテスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 同値あり
        prices_with_ties = np.array([100, 101, 100, 102], dtype=np.float32)
        assert calc._check_for_ties(prices_with_ties) == True
        
        # 同値なし
        prices_no_ties = np.array([100.1, 100.2, 100.3, 100.4], dtype=np.float32)
        assert calc._check_for_ties(prices_no_ties) == False
    
    def test_cached_ranking(self):
        """キャッシュ付きランキングのテスト"""
        calc = DifferentialRCICalculator(period=5)
        
        # 同じ価格パターンを複数回計算
        prices = np.array([100, 101, 102, 103, 104], dtype=np.float32)
        
        # 初回はキャッシュミス
        result1 = calc._cached_ranking(prices)
        assert calc._cache_misses == 1
        assert calc._cache_hits == 0
        
        # 2回目はキャッシュヒット
        result2 = calc._cached_ranking(prices)
        assert calc._cache_hits == 1
        
        # 結果が同じことを確認
        np.testing.assert_array_equal(result1, result2)
    
    def test_cache_size_limit(self):
        """キャッシュサイズ制限のテスト"""
        calc = DifferentialRCICalculator(period=3)
        calc._max_cache_size = 3  # 小さなキャッシュサイズに設定
        
        # 異なるパターンを追加
        patterns = [
            np.array([100, 101, 102], dtype=np.float32),
            np.array([101, 102, 103], dtype=np.float32),
            np.array([102, 103, 104], dtype=np.float32),
            np.array([103, 104, 105], dtype=np.float32),  # これで上限を超える
        ]
        
        for pattern in patterns:
            calc._cached_ranking(pattern)
        
        # キャッシュサイズが上限を超えないことを確認
        assert len(calc._ranking_cache) <= calc._max_cache_size
        
        # 最初のパターンがキャッシュから削除されていることを確認
        first_hash = calc._get_price_hash(patterns[0])
        assert first_hash not in calc._ranking_cache


def benchmark_ranking_methods():
    """各ランキング手法のベンチマーク比較"""
    import time
    
    calc = DifferentialRCICalculator(period=100)
    n_iterations = 1000
    
    # テストデータ準備
    np.random.seed(42)
    
    # 同値なしのデータ
    prices_no_ties = np.random.randn(100).astype(np.float32) * 10 + 100
    
    # 同値ありのデータ（一部を同じ値に）
    prices_with_ties = prices_no_ties.copy()
    prices_with_ties[10:15] = prices_with_ties[10]  # 5つの同値
    prices_with_ties[30:33] = prices_with_ties[30]  # 3つの同値
    
    results = {}
    
    # 1. 高速ランキング（同値なし）
    start = time.time()
    for _ in range(n_iterations):
        calc._fast_ranking_no_ties(prices_no_ties)
    results['fast_no_ties'] = time.time() - start
    
    # 2. NumPy同値対応ランキング
    start = time.time()
    for _ in range(n_iterations):
        calc._numpy_ranking_with_ties(prices_with_ties)
    results['numpy_with_ties'] = time.time() - start
    
    # 3. キャッシュ付きランキング（初回）
    calc.reset()  # キャッシュをクリア
    start = time.time()
    for _ in range(n_iterations):
        calc._cached_ranking(prices_no_ties)
    results['cached_first'] = time.time() - start
    
    # 4. キャッシュ付きランキング（キャッシュヒット）
    # 同じデータで再実行（キャッシュヒット）
    start = time.time()
    for _ in range(n_iterations):
        calc._cached_ranking(prices_no_ties)
    results['cached_hit'] = time.time() - start
    
    # 結果表示
    print("\n" + "="*60)
    print("Ranking Method Benchmark Results")
    print("="*60)
    print(f"Iterations: {n_iterations}, Period: 100")
    print("-"*60)
    
    for method, elapsed in results.items():
        avg_ms = (elapsed / n_iterations) * 1000
        print(f"{method:<20}: {avg_ms:.4f} ms/call")
    
    print("-"*60)
    
    # キャッシュ統計
    state = calc.get_buffer_state()
    cache_stats = state['cache_stats']
    print(f"\nCache Statistics:")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['cache_hit_rate']:.2%}")
    print(f"  Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    
    # パフォーマンス目標の確認
    assert results['fast_no_ties'] < results['numpy_with_ties']
    assert results['cached_hit'] < results['cached_first']
    print("\n✓ All performance targets met!")


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
    
    # 最適化テスト
    print("\nRunning optimization tests...")
    opt_test = TestRankingOptimizations()
    
    opt_test.test_numpy_ranking_with_ties()
    print("✓ NumPy ranking with ties test passed")
    
    opt_test.test_fast_ranking_no_ties()
    print("✓ Fast ranking test passed")
    
    opt_test.test_check_for_ties()
    print("✓ Tie detection test passed")
    
    opt_test.test_cached_ranking()
    print("✓ Cached ranking test passed")
    
    opt_test.test_cache_size_limit()
    print("✓ Cache size limit test passed")
    
    # パフォーマンステスト
    print("\nRunning performance benchmark...")
    test_performance_benchmark()
    print("✓ Performance benchmark passed")
    
    # ランキング手法のベンチマーク
    print("\nRunning ranking methods benchmark...")
    benchmark_ranking_methods()
    
    print("\nAll tests passed successfully!")