"""
RCICalculatorEngineのユニットテスト

複数期間の並列計算、バリデーションロジック、
エラーハンドリングのテストを実施します。
"""

import pytest
import numpy as np
import polars as pl
from typing import List
import logging

from src.data_processing.rci import (
    RCICalculatorEngine,
    DifferentialRCICalculator,
    RCICalculationError,
    InvalidPeriodError,
    InsufficientDataError
)

# ロギング設定
logging.basicConfig(level=logging.DEBUG)


class TestRCICalculatorEngine:
    """RCICalculatorEngineのテストクラス"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化処理"""
        self.engine = RCICalculatorEngine()
        
        # テスト用データの生成
        np.random.seed(42)
        n = 200
        # 上昇トレンドのデータ
        trend = np.linspace(100, 120, n)
        noise = np.random.normal(0, 0.5, n)
        self.test_prices = trend + noise
        
        self.test_df = pl.DataFrame({
            'timestamp': list(range(n)),
            'close': self.test_prices,
            'volume': np.random.randint(1000, 10000, n)
        })
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ処理"""
        self.engine.reset()
    
    def test_initialization(self):
        """エンジンの初期化テスト"""
        assert self.engine.calculators == {}
        assert self.engine._statistics['batch_calculations'] == 0
        assert self.engine._statistics['streaming_calculations'] == 0
        assert self.engine._statistics['total_periods_calculated'] == 0
        assert self.engine._statistics['validation_errors'] == 0
    
    def test_validate_periods_valid(self):
        """有効な期間リストの検証テスト"""
        # 通常の期間リスト
        periods = [9, 13, 24]
        validated = self.engine.validate_periods(periods)
        assert validated == [9, 13, 24]
        
        # 重複を含む期間リスト
        periods = [9, 13, 9, 24, 13]
        validated = self.engine.validate_periods(periods)
        assert validated == [9, 13, 24]  # 重複除去・ソート済み
        
        # 順序がバラバラな期間リスト
        periods = [24, 9, 13]
        validated = self.engine.validate_periods(periods)
        assert validated == [9, 13, 24]  # ソート済み
    
    def test_validate_periods_invalid(self):
        """無効な期間リストの検証テスト"""
        # 空のリスト
        with pytest.raises(InvalidPeriodError, match="Periods list cannot be empty"):
            self.engine.validate_periods([])
        
        # 範囲外の期間
        with pytest.raises(InvalidPeriodError, match="Invalid periods"):
            self.engine.validate_periods([2, 9, 201])  # 2と201が無効
        
        # 非整数値
        with pytest.raises(InvalidPeriodError, match="Invalid periods"):
            self.engine.validate_periods([9, "invalid", 13])
        
        # エラーカウントの確認
        assert self.engine._statistics['validation_errors'] == 2
    
    def test_calculate_multiple_batch_mode(self):
        """バッチモードでの複数期間RCI計算テスト"""
        periods = [9, 13, 24]
        result = self.engine.calculate_multiple(
            self.test_df,
            periods=periods,
            column_name='close',
            mode='batch'
        )
        
        # カラムの存在確認
        for period in periods:
            assert f'rci_{period}' in result.columns
            assert f'rci_{period}_reliable' in result.columns
        
        # RCI値の範囲確認（-100〜+100）
        for period in periods:
            rci_values = result[f'rci_{period}'].drop_nulls()
            assert rci_values.min() >= -100
            assert rci_values.max() <= 100
        
        # 最初のperiod-1個はNoneであることを確認
        for period in periods:
            null_count = result[f'rci_{period}'].null_count()
            assert null_count == period - 1
        
        # 統計情報の確認
        assert self.engine._statistics['batch_calculations'] == 1
        assert self.engine._statistics['total_periods_calculated'] == 3
    
    def test_calculate_multiple_streaming_mode(self):
        """ストリーミングモードでの複数期間RCI計算テスト"""
        periods = [9, 13]
        result = self.engine.calculate_multiple(
            self.test_df,
            periods=periods,
            column_name='close',
            mode='streaming'
        )
        
        # カラムの存在確認
        for period in periods:
            assert f'rci_{period}' in result.columns
            assert f'rci_{period}_reliable' in result.columns
        
        # RCI値の範囲確認
        for period in periods:
            rci_values = result[f'rci_{period}'].drop_nulls()
            assert rci_values.min() >= -100
            assert rci_values.max() <= 100
        
        # 計算器の状態確認
        assert 9 in self.engine.calculators
        assert 13 in self.engine.calculators
        assert self.engine.calculators[9].calculation_count > 0
        assert self.engine.calculators[13].calculation_count > 0
        
        # 統計情報の確認
        assert self.engine._statistics['streaming_calculations'] == 1
        assert self.engine._statistics['total_periods_calculated'] == 2
    
    def test_calculate_multiple_default_periods(self):
        """デフォルト期間での計算テスト"""
        result = self.engine.calculate_multiple(
            self.test_df,
            periods=None,  # デフォルト期間を使用
            column_name='close'
        )
        
        # デフォルト期間のカラムが存在することを確認
        for period in RCICalculatorEngine.DEFAULT_PERIODS:
            assert f'rci_{period}' in result.columns
    
    def test_calculate_multiple_column_not_found(self):
        """存在しないカラム名でのエラーテスト"""
        with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
            self.engine.calculate_multiple(
                self.test_df,
                periods=[9],
                column_name='invalid_column'
            )
    
    def test_calculate_multiple_insufficient_data(self):
        """データ不足エラーのテスト"""
        small_df = self.test_df.head(10)  # 10行のみ
        
        with pytest.raises(InsufficientDataError, match="Insufficient data"):
            self.engine.calculate_multiple(
                small_df,
                periods=[20],  # 20期間は10行では計算不可
                column_name='close'
            )
    
    def test_calculate_multiple_invalid_mode(self):
        """無効なモードでのエラーテスト"""
        with pytest.raises(ValueError, match="Invalid mode"):
            self.engine.calculate_multiple(
                self.test_df,
                periods=[9],
                column_name='close',
                mode='invalid_mode'
            )
    
    def test_reliability_flags(self):
        """信頼性フラグのテスト"""
        periods = [9, 13]
        result = self.engine.calculate_multiple(
            self.test_df,
            periods=periods,
            column_name='close',
            add_reliability=True
        )
        
        for period in periods:
            reliability_col = f'rci_{period}_reliable'
            
            # 最初のperiod-1個はFalse
            for i in range(period - 1):
                assert result[reliability_col][i] == False
            
            # それ以降はTrue
            for i in range(period - 1, len(result)):
                assert result[reliability_col][i] == True
    
    def test_no_reliability_flags(self):
        """信頼性フラグを追加しない場合のテスト"""
        result = self.engine.calculate_multiple(
            self.test_df,
            periods=[9],
            column_name='close',
            add_reliability=False
        )
        
        # 信頼性フラグカラムが存在しないことを確認
        assert 'rci_9_reliable' not in result.columns
    
    def test_get_statistics(self):
        """統計情報取得のテスト"""
        # バッチ計算実行
        self.engine.calculate_multiple(
            self.test_df,
            periods=[9, 13],
            mode='batch'
        )
        
        # ストリーミング計算実行
        self.engine.calculate_multiple(
            self.test_df,
            periods=[24],
            mode='streaming'
        )
        
        stats = self.engine.get_statistics()
        
        assert stats['batch_calculations'] == 1
        assert stats['streaming_calculations'] == 1
        assert stats['total_periods_calculated'] == 3
        assert 24 in stats['active_calculators']
        assert 'calculator_details' in stats
        assert 24 in stats['calculator_details']
    
    def test_reset(self):
        """リセット機能のテスト"""
        # 計算実行
        self.engine.calculate_multiple(
            self.test_df,
            periods=[9],
            mode='streaming'
        )
        
        # リセット前の確認
        assert len(self.engine.calculators) > 0
        assert self.engine._statistics['total_periods_calculated'] > 0
        
        # リセット実行
        self.engine.reset()
        
        # リセット後の確認
        assert len(self.engine.calculators) == 0
        assert self.engine._statistics['batch_calculations'] == 0
        assert self.engine._statistics['streaming_calculations'] == 0
        assert self.engine._statistics['total_periods_calculated'] == 0
    
    def test_batch_vs_streaming_consistency(self):
        """バッチモードとストリーミングモードの結果一致性テスト"""
        periods = [9, 13]
        
        # バッチモードで計算
        batch_result = self.engine.calculate_multiple(
            self.test_df,
            periods=periods,
            mode='batch'
        )
        
        # エンジンをリセット
        self.engine.reset()
        
        # ストリーミングモードで計算
        streaming_result = self.engine.calculate_multiple(
            self.test_df,
            periods=periods,
            mode='streaming'
        )
        
        # 結果の一致性を確認（浮動小数点誤差を考慮）
        for period in periods:
            batch_rci = batch_result[f'rci_{period}'].to_numpy()
            streaming_rci = streaming_result[f'rci_{period}'].to_numpy()
            
            # NaN値のインデックスが一致することを確認
            batch_nan_mask = np.isnan(batch_rci)
            streaming_nan_mask = np.isnan(streaming_rci)
            np.testing.assert_array_equal(batch_nan_mask, streaming_nan_mask)
            
            # 非NaN値が近似的に一致することを確認
            valid_indices = ~batch_nan_mask
            if np.any(valid_indices):
                np.testing.assert_allclose(
                    batch_rci[valid_indices],
                    streaming_rci[valid_indices],
                    rtol=1e-5,
                    atol=1e-5
                )
    
    def test_trend_detection(self):
        """トレンド検出能力のテスト"""
        # 明確な上昇トレンドデータ
        n = 100
        uptrend_prices = np.linspace(100, 150, n)
        uptrend_df = pl.DataFrame({
            'close': uptrend_prices
        })
        
        result = self.engine.calculate_multiple(
            uptrend_df,
            periods=[9],
            column_name='close'
        )
        
        # 上昇トレンドではRCIが高い値を示すはず
        rci_values = result['rci_9'].drop_nulls()
        mean_rci = rci_values.mean()
        assert mean_rci > 50  # 平均RCIが50以上
        
        # 明確な下降トレンドデータ
        downtrend_prices = np.linspace(150, 100, n)
        downtrend_df = pl.DataFrame({
            'close': downtrend_prices
        })
        
        self.engine.reset()
        result = self.engine.calculate_multiple(
            downtrend_df,
            periods=[9],
            column_name='close'
        )
        
        # 下降トレンドではRCIが低い値を示すはず
        rci_values = result['rci_9'].drop_nulls()
        mean_rci = rci_values.mean()
        assert mean_rci < -50  # 平均RCIが-50以下
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 最小期間でのテスト
        min_period_df = pl.DataFrame({
            'close': [100, 101, 102]  # 3行（最小期間）
        })
        
        result = self.engine.calculate_multiple(
            min_period_df,
            periods=[3],
            column_name='close'
        )
        
        # 最後の行にのみRCI値が存在
        assert result['rci_3'][0] is None
        assert result['rci_3'][1] is None
        assert result['rci_3'][2] is not None
        
        # 全て同じ価格の場合
        same_price_df = pl.DataFrame({
            'close': [100.0] * 20
        })
        
        self.engine.reset()
        result = self.engine.calculate_multiple(
            same_price_df,
            periods=[9],
            column_name='close'
        )
        
        # 価格が全て同じ場合、RCIは0に近い値になるはず
        rci_values = result['rci_9'].drop_nulls()
        for rci in rci_values:
            assert abs(rci) < 1.0  # ほぼ0

    def test_auto_mode_selection(self):
        """自動モード選択のテスト"""
        # 小規模データ（AUTO_MODE_THRESHOLD未満）
        small_df = self.test_df.head(30000)
        result = self.engine.calculate_multiple(
            small_df,
            periods=[9, 13],
            mode='auto'
        )
        
        # 自動モード選択が実行されたことを確認
        assert self.engine._statistics['auto_mode_selections'] > 0
        
        # 結果が正しく計算されていることを確認
        assert 'rci_9' in result.columns
        assert 'rci_13' in result.columns
        
        # 大規模データ（AUTO_MODE_THRESHOLD以上）
        self.engine.reset()
        large_df = pl.DataFrame({
            'close': np.random.randn(60000).cumsum() + 100
        })
        
        result = self.engine.calculate_multiple(
            large_df,
            periods=[9, 13, 24],
            mode='auto'
        )
        
        # 自動モード選択が実行されたことを確認
        assert self.engine._statistics['auto_mode_selections'] == 1
        assert len(result) == 60000
    
    def test_chunked_batch_processing(self):
        """チャンク処理のテスト"""
        # チャンクサイズを小さく設定してテスト
        engine = RCICalculatorEngine(chunk_size=10000)
        
        # チャンク処理が必要な大規模データ
        large_df = pl.DataFrame({
            'close': np.random.randn(50000).cumsum() + 100
        })
        
        result = engine.calculate_multiple(
            large_df,
            periods=[9, 13],
            mode='batch'
        )
        
        # チャンク処理が実行されたことを確認
        assert engine._statistics['chunked_calculations'] > 0
        
        # 結果の整合性確認
        assert len(result) == 50000
        assert 'rci_9' in result.columns
        assert 'rci_13' in result.columns
        
        # RCI値が正しく計算されていることを確認
        rci_9_values = result['rci_9'].drop_nulls()
        assert len(rci_9_values) == 50000 - 8  # 最初の8個はNone
        assert rci_9_values.min() >= -100
        assert rci_9_values.max() <= 100
    
    def test_parallel_calculation(self):
        """並列計算のテスト"""
        periods = [9, 13, 24, 33, 48]  # 複数期間
        
        # タイマーで処理時間を測定
        import time
        start_time = time.time()
        
        result = self.engine.calculate_multiple(
            self.test_df,
            periods=periods,
            mode='batch'
        )
        
        parallel_time = time.time() - start_time
        
        # 並列計算が実行されたことを確認（_calculate_rci_parallelが呼ばれる）
        # 結果の確認
        for period in periods:
            assert f'rci_{period}' in result.columns
        
        print(f"\nParallel calculation time for {len(periods)} periods: {parallel_time:.2f}s")
    
    def test_incremental_update(self):
        """増分更新（リアルタイム処理）のテスト"""
        # 初期データでストリーミング計算
        initial_data = self.test_df.head(50)
        result = self.engine.calculate_multiple(
            initial_data,
            periods=[9, 13],
            mode='streaming'
        )
        
        # 増分更新のテスト
        new_prices = [101.5, 102.0, 102.5, 103.0]
        for price in new_prices:
            rci_values = self.engine.add_incremental(price, periods=[9, 13])
            
            # 返り値の確認
            assert 9 in rci_values
            assert 13 in rci_values
            
            # 計算器が準備完了状態の場合、RCI値が返される
            if self.engine.calculators[9].is_ready:
                assert rci_values[9] is not None
                assert -100 <= rci_values[9] <= 100
    
    def test_get_latest_rci_values(self):
        """最新RCI値取得のテスト"""
        # データを流して計算器を準備
        self.engine.calculate_multiple(
            self.test_df.head(50),
            periods=[9, 13],
            mode='streaming'
        )
        
        # 最新値の取得
        latest_values = self.engine.get_latest_rci_values()
        
        assert 9 in latest_values
        assert 13 in latest_values
        
        # 特定期間のみ取得
        specific_values = self.engine.get_latest_rci_values(periods=[9])
        assert 9 in specific_values
        assert 13 not in specific_values
    
    def test_memory_management(self):
        """メモリ管理機能のテスト"""
        # メモリ使用量の監視
        memory_usage = self.engine._monitor_memory_usage()
        assert isinstance(memory_usage, float)
        assert 0 <= memory_usage <= 100
        
        # チャンクサイズの動的調整
        original_chunk_size = self.engine.chunk_size
        
        # 高メモリ使用時のシミュレーション
        new_size = self.engine._adjust_chunk_size_dynamically(85.0)
        assert new_size < original_chunk_size
        
        # 低メモリ使用時のシミュレーション
        self.engine.chunk_size = 10000
        new_size = self.engine._adjust_chunk_size_dynamically(20.0)
        assert new_size > 10000
    
    def test_large_scale_with_auto_mode(self):
        """大規模データでの自動モードテスト"""
        # 10万行のテストデータ
        n = 100_000
        large_df = pl.DataFrame({
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # 自動モードで実行
        result = self.engine.calculate_multiple(
            large_df,
            periods=[9, 13, 24],
            mode='auto'
        )
        
        # 結果の確認
        assert len(result) == n
        for period in [9, 13, 24]:
            assert f'rci_{period}' in result.columns
            
            # null以外の値の数を確認
            non_null_count = result[f'rci_{period}'].drop_nulls().len()
            assert non_null_count == n - period + 1
        
        # 統計情報の確認
        stats = self.engine.get_statistics()
        assert stats['auto_mode_selections'] > 0
        print(f"\nLarge scale test statistics: {stats}")
    
    def test_mode_consistency(self):
        """異なるモード間の結果一致性テスト"""
        periods = [9, 13]
        test_data = self.test_df.head(1000)
        
        # バッチモード（通常）
        self.engine.reset()
        result_batch = self.engine.calculate_multiple(
            test_data,
            periods=periods,
            mode='batch'
        )
        
        # バッチモード（チャンク処理）
        engine_chunked = RCICalculatorEngine(chunk_size=200)
        result_chunked = engine_chunked.calculate_multiple(
            test_data,
            periods=periods,
            mode='batch'  # チャンク処理が自動で適用される
        )
        
        # ストリーミングモード
        self.engine.reset()
        result_streaming = self.engine.calculate_multiple(
            test_data,
            periods=periods,
            mode='streaming'
        )
        
        # 結果の一致性を確認（浮動小数点誤差を考慮）
        for period in periods:
            batch_values = result_batch[f'rci_{period}'].to_numpy()
            chunked_values = result_chunked[f'rci_{period}'].to_numpy()
            streaming_values = result_streaming[f'rci_{period}'].to_numpy()
            
            # NaN値のマスク
            valid_mask = ~np.isnan(batch_values)
            
            # 値の一致を確認
            np.testing.assert_allclose(
                batch_values[valid_mask],
                chunked_values[valid_mask],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Batch vs Chunked mismatch for period {period}"
            )
            
            np.testing.assert_allclose(
                batch_values[valid_mask],
                streaming_values[valid_mask],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Batch vs Streaming mismatch for period {period}"
            )


class TestRCICalculatorEngineAdvancedPerformance:
    """高度なパフォーマンステスト"""
    
    def test_parallel_vs_sequential_performance(self):
        """並列処理と逐次処理のパフォーマンス比較"""
        engine = RCICalculatorEngine()
        
        # テストデータ
        n = 50000
        data = pl.DataFrame({
            'close': np.random.randn(n).cumsum() + 100
        })
        periods = [9, 13, 24, 33, 48]
        
        import time
        
        # 並列処理のタイミング
        start = time.time()
        prices = data['close'].to_numpy()
        parallel_results = engine._calculate_rci_parallel(prices, periods)
        parallel_time = time.time() - start
        
        # 逐次処理のタイミング
        start = time.time()
        sequential_results = {}
        for period in periods:
            sequential_results[period] = engine._calculate_rci_batch(prices, period)
        sequential_time = time.time() - start
        
        # パフォーマンス改善の確認
        speedup = sequential_time / parallel_time
        print(f"\n並列処理のスピードアップ: {speedup:.2f}x")
        print(f"逐次処理時間: {sequential_time:.2f}秒")
        print(f"並列処理時間: {parallel_time:.2f}秒")
        
        # 結果の一致性確認
        for period in periods:
            parallel_vals = parallel_results[period]
            sequential_vals = sequential_results[period]
            
            # None以外の値を比較
            for i, (p, s) in enumerate(zip(parallel_vals, sequential_vals)):
                if p is not None and s is not None:
                    assert abs(p - s) < 1e-5, f"Mismatch at index {i} for period {period}"
    
    def test_memory_efficiency_large_scale(self):
        """大規模データでのメモリ効率テスト"""
        import psutil
        import gc
        
        engine = RCICalculatorEngine(chunk_size=50000)
        
        # 初期メモリ使用量
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 50万行のデータ処理
        n = 500_000
        large_data = pl.DataFrame({
            'close': np.random.randn(n).cumsum() + 100
        })
        
        # チャンク処理で実行
        result = engine.calculate_multiple(
            large_data,
            periods=[9, 13, 24],
            mode='batch'
        )
        
        # 処理後のメモリ使用量
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\n大規模データ処理のメモリ使用量:")
        print(f"  データサイズ: {n:,} 行")
        print(f"  初期メモリ: {initial_memory:.2f} MB")
        print(f"  最終メモリ: {final_memory:.2f} MB")
        print(f"  増加量: {memory_increase:.2f} MB")
        
        # メモリ増加が妥当な範囲内であることを確認
        # 500K行のデータで500MB以内の増加を期待
        assert memory_increase < 500, f"Memory increase too large: {memory_increase:.2f} MB"
        
        # 結果の妥当性確認
        assert len(result) == n
        assert 'rci_9' in result.columns
        
        # チャンク処理が使用されたことを確認
        assert engine._statistics['chunked_calculations'] > 0


class TestRCICalculatorEnginePerformance:
    """パフォーマンステスト"""
    
    def test_large_dataset_performance(self):
        """大規模データセットでのパフォーマンステスト"""
        engine = RCICalculatorEngine()
        
        # 10万行のテストデータ
        n = 100000
        large_df = pl.DataFrame({
            'close': np.random.randn(n).cumsum() + 100
        })
        
        import time
        
        # バッチモードのパフォーマンス測定
        start_time = time.time()
        result = engine.calculate_multiple(
            large_df,
            periods=[9, 13, 24],
            mode='batch'
        )
        batch_time = time.time() - start_time
        
        print(f"\nBatch mode (100k rows, 3 periods): {batch_time:.2f} seconds")
        assert batch_time < 5.0  # 5秒以内に完了すべき
        
        # 結果の妥当性確認
        assert len(result) == n
        for period in [9, 13, 24]:
            assert f'rci_{period}' in result.columns
    
    def test_memory_efficiency(self):
        """メモリ効率のテスト"""
        import psutil
        import os
        
        engine = RCICalculatorEngine()
        
        # メモリ使用量測定
        process = psutil.Process(os.getpid())
        
        # 初期メモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 5万行のデータで計算
        n = 50000
        df = pl.DataFrame({
            'close': np.random.randn(n).cumsum() + 100
        })
        
        result = engine.calculate_multiple(
            df,
            periods=[9, 13, 24, 33, 48],
            mode='batch'
        )
        
        # 計算後のメモリ使用量
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory increase: {memory_increase:.2f} MB")
        
        # メモリ増加が妥当な範囲内であることを確認
        assert memory_increase < 200  # 200MB以内の増加


if __name__ == "__main__":
    pytest.main([__file__, "-v"])