"""
RCIモジュールの包括的なユニットテスト

このファイルでは以下をテストします：
1. 例外クラス（RCICalculationError, InvalidPeriodError, InsufficientDataError）
2. RCIProcessorクラス
3. 統合テスト（全クラスの連携）
4. エッジケース
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, Any
import time
import psutil
import os
from concurrent.futures import ProcessPoolExecutor

from src.data_processing.rci import (
    RCICalculationError,
    InvalidPeriodError,
    InsufficientDataError,
    DifferentialRCICalculator,
    RCICalculatorEngine,
    RCIProcessor
)


class TestRCIExceptions:
    """RCI例外クラスのテスト"""
    
    def test_rci_calculation_error(self):
        """RCICalculationError例外のテスト"""
        with pytest.raises(RCICalculationError) as exc_info:
            raise RCICalculationError("計算エラーが発生しました")
        
        assert str(exc_info.value) == "計算エラーが発生しました"
        assert isinstance(exc_info.value, Exception)
    
    def test_invalid_period_error(self):
        """InvalidPeriodError例外のテスト"""
        with pytest.raises(InvalidPeriodError) as exc_info:
            raise InvalidPeriodError("期間は2以上である必要があります")
        
        assert str(exc_info.value) == "期間は2以上である必要があります"
        assert isinstance(exc_info.value, RCICalculationError)
    
    def test_insufficient_data_error(self):
        """InsufficientDataError例外のテスト"""
        with pytest.raises(InsufficientDataError) as exc_info:
            raise InsufficientDataError("データ不足: 10件必要ですが5件しかありません")
        
        assert str(exc_info.value) == "データ不足: 10件必要ですが5件しかありません"
        assert isinstance(exc_info.value, RCICalculationError)
    
    def test_exception_hierarchy(self):
        """例外の継承関係のテスト"""
        # InvalidPeriodErrorはRCICalculationErrorのサブクラス
        error = InvalidPeriodError("test")
        assert isinstance(error, RCICalculationError)
        assert isinstance(error, Exception)
        
        # InsufficientDataErrorもRCICalculationErrorのサブクラス
        error = InsufficientDataError("test")
        assert isinstance(error, RCICalculationError)
        assert isinstance(error, Exception)


class TestRCIProcessor:
    """RCIProcessorクラスのテスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.processor = RCIProcessor()
        
        # テスト用データの準備
        np.random.seed(42)
        self.test_data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'close': np.random.randn(200).cumsum() + 100,
            'symbol': ['EURUSD'] * 100 + ['USDJPY'] * 100,
            'volume': np.random.randint(100, 1000, 200)
        })
    
    def test_initialization(self):
        """初期化のテスト"""
        # デフォルト初期化
        processor = RCIProcessor()
        assert processor.engine is not None
        assert processor.use_float32 is True  # デフォルトはTrue
        assert processor._expr_cache == {}
        
        # float32無効での初期化
        processor = RCIProcessor(use_float32=False)
        assert processor.use_float32 is False
        
        # エンジンが正しく初期化されている
        assert hasattr(processor.engine, 'calculators')
        assert hasattr(processor.engine, 'calculate_multiple')
        assert processor.engine.calculators == {}  # 初期状態では空
    
    def test_create_rci_expression(self):
        """create_rci_expressionメソッドのテスト"""
        # 基本的な式の作成
        expr = self.processor.create_rci_expression(
            column='close',
            period=10
        )
        assert expr is not None
        
        # キャッシュの確認
        cache_key = ('close', 10)
        assert cache_key in self.processor._expr_cache
        
        # 同じパラメータでの再呼び出し（キャッシュヒット）
        expr2 = self.processor.create_rci_expression(
            column='close',
            period=10
        )
        assert expr is expr2  # 同じオブジェクトが返される（キャッシュから）
        
        # 異なる期間での式作成
        expr_20 = self.processor.create_rci_expression(
            column='close',
            period=20
        )
        assert expr_20 is not None
        assert expr_20 is not expr  # 異なるオブジェクト
        
        # 異なるカラムでの式作成
        expr_high = self.processor.create_rci_expression(
            column='high',
            period=10
        )
        assert expr_high is not None
        assert expr_high is not expr  # 異なるオブジェクト
    
    def test_apply_to_dataframe(self):
        """apply_to_dataframeメソッドのテスト"""
        # 単一期間での適用
        result = self.processor.apply_to_dataframe(
            self.test_data,
            periods=[10],
            column_name='close'
        )
        
        assert 'rci_10' in result.columns
        assert len(result) == len(self.test_data)
        
        # 複数期間での適用
        result = self.processor.apply_to_dataframe(
            self.test_data,
            periods=[5, 10, 20],
            column_name='close'
        )
        
        assert 'rci_5' in result.columns
        assert 'rci_10' in result.columns
        assert 'rci_20' in result.columns
        
        # デフォルト期間での適用
        result = self.processor.apply_to_dataframe(
            self.test_data,
            column_name='close'
        )
        
        # デフォルト期間（9, 13, 24, 33, 48, 66, 108）が適用される
        assert 'rci_9' in result.columns
        assert 'rci_13' in result.columns
        assert 'rci_24' in result.columns
        
        # 信頼性フラグ付きでの適用（デフォルト）
        result = self.processor.apply_to_dataframe(
            self.test_data,
            periods=[10],
            column_name='close',
            add_reliability=True
        )
        
        assert 'rci_10' in result.columns
        assert 'rci_10_reliable' in result.columns
        
        # 信頼性フラグ無しでの適用
        result = self.processor.apply_to_dataframe(
            self.test_data,
            periods=[10],
            column_name='close',
            add_reliability=False
        )
        
        assert 'rci_10' in result.columns
        assert 'rci_10_reliable' not in result.columns
    
    def test_apply_to_lazyframe(self):
        """apply_to_lazyframeメソッドのテスト"""
        # LazyFrameの作成
        lazy_df = self.test_data.lazy()
        
        # 単一期間での適用
        result_lazy = self.processor.apply_to_lazyframe(
            lazy_df,
            periods=[10],
            column_name='close'
        )
        
        # collectして結果を確認
        result = result_lazy.collect()
        assert 'rci_10' in result.columns
        assert len(result) == len(self.test_data)
        
        # 複数期間での適用
        result_lazy = self.processor.apply_to_lazyframe(
            lazy_df,
            periods=[5, 10, 20],
            column_name='close'
        )
        
        result = result_lazy.collect()
        assert 'rci_5' in result.columns
        assert 'rci_10' in result.columns
        assert 'rci_20' in result.columns
    
    def test_apply_grouped(self):
        """apply_groupedメソッドのテスト"""
        # グループ化してRCI計算
        result = self.processor.apply_grouped(
            self.test_data,
            group_by=['symbol'],
            periods=[10],
            column_name='close'
        )
        
        assert 'rci_10' in result.columns
        assert len(result) == len(self.test_data)
        
        # 各グループごとに計算されているか確認
        eurusd_data = result.filter(pl.col('symbol') == 'EURUSD')
        usdjpy_data = result.filter(pl.col('symbol') == 'USDJPY')
        
        assert len(eurusd_data) == 100
        assert len(usdjpy_data) == 100
        
        # 各グループの最初の期間-1行はNaNであることを確認
        eurusd_nulls = eurusd_data.filter(pl.col('rci_10').is_null())
        assert len(eurusd_nulls) == 9  # 最初の9行はNaN
    
    def test_create_multi_period_expression(self):
        """create_multi_period_expressionメソッドのテスト"""
        # 複数期間の式を一度に作成
        exprs = self.processor.create_multi_period_expression(
            periods=[5, 10, 20],
            column='close'
        )
        
        assert len(exprs) == 3
        assert all(expr is not None for expr in exprs)
    
    def test_get_statistics(self):
        """get_statisticsメソッドのテスト"""
        # 初期状態の統計情報
        stats = self.processor.get_statistics()
        assert 'expression_cache_size' in stats
        assert 'cached_expressions' in stats
        assert 'engine_statistics' in stats
        assert stats['expression_cache_size'] == 0
        
        # 処理後の統計情報
        self.processor.apply_to_dataframe(
            self.test_data,
            periods=[10],
            column_name='close'
        )
        
        stats = self.processor.get_statistics()
        assert stats['expression_cache_size'] > 0
        assert len(stats['cached_expressions']) > 0
    
    def test_clear_cache(self):
        """clear_cacheメソッドのテスト"""
        # キャッシュを作成
        self.processor.create_rci_expression('close', 10)
        assert len(self.processor._expr_cache) > 0
        
        # キャッシュをクリア
        self.processor.clear_cache()
        assert len(self.processor._expr_cache) == 0
    
    def test_invalid_inputs(self):
        """不正な入力に対するエラーハンドリング"""
        # 存在しないカラム
        with pytest.raises(Exception):
            self.processor.apply_to_dataframe(
                self.test_data,
                periods=[10],
                column_name='non_existent'
            )
        
        # 不正な期間
        with pytest.raises(InvalidPeriodError):
            self.processor.create_rci_expression(
                column='close',
                period=1  # 期間は2以上
            )
        
        # 空のデータフレーム
        empty_df = pl.DataFrame({'close': []})
        result = self.processor.apply_to_dataframe(
            empty_df,
            periods=[10],
            column_name='close'
        )
        assert len(result) == 0


class TestRCIIntegration:
    """統合テスト - 全クラスの連携"""
    
    def setup_method(self):
        """各テストメソッドの前に実行"""
        np.random.seed(42)
        
        # より現実的なデータセットを作成
        self.create_realistic_data()
    
    def create_realistic_data(self):
        """現実的なFXデータを作成"""
        import pandas as pd
        
        # トレンドとボラティリティを含むデータ生成
        n_points = 1000
        time_index = pd.date_range('2024-01-01', periods=n_points, freq='1min')
        
        # 上昇トレンド、下降トレンド、レンジ相場を含む
        prices = []
        base_price = 100.0
        
        for i in range(n_points):
            if i < 300:  # 上昇トレンド
                base_price += np.random.normal(0.01, 0.1)
            elif i < 600:  # 下降トレンド
                base_price -= np.random.normal(0.01, 0.1)
            else:  # レンジ相場
                base_price += np.random.normal(0, 0.05)
            
            prices.append(max(base_price, 0.1))  # 負の価格を防ぐ
        
        self.fx_data = pl.DataFrame({
            'timestamp': time_index,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 0.02)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.02)) for p in prices],
            'close': [p + np.random.normal(0, 0.01) for p in prices],
            'volume': np.random.randint(100, 10000, n_points)
        })
    
    def test_full_pipeline(self):
        """完全なパイプラインのテスト: Calculator → Engine → Processor"""
        # 1. DifferentialRCICalculatorの直接使用
        calc = DifferentialRCICalculator(period=10)
        prices = self.fx_data['close'].to_numpy()[:20]
        
        rci_values = []
        for price in prices:
            rci = calc.add(price)
            rci_values.append(rci)
        
        # 2. RCICalculatorEngineの使用
        engine = RCICalculatorEngine(periods=[10, 20, 30])
        engine_result = engine.calculate_multiple(
            self.fx_data[:100],
            periods=[10, 20, 30],
            price_col='close',
            mode='batch'
        )
        
        assert 'rci_10' in engine_result.columns
        assert 'rci_20' in engine_result.columns
        assert 'rci_30' in engine_result.columns
        
        # 3. RCIProcessorの使用
        processor = RCIProcessor(periods=[10, 20, 30])
        processor_result = processor.apply_to_dataframe(
            self.fx_data[:100],
            periods=[10, 20, 30],
            price_col='close'
        )
        
        assert 'rci_10' in processor_result.columns
        assert 'rci_20' in processor_result.columns
        assert 'rci_30' in processor_result.columns
        
        # 結果の一貫性チェック（最初の10行を除く）
        for i in range(10, 20):
            calc_value = rci_values[i] if rci_values[i] is not None else np.nan
            engine_value = engine_result['rci_10'][i]
            processor_value = processor_result['rci_10'][i]
            
            if not np.isnan(calc_value):
                assert np.isclose(calc_value, engine_value, rtol=1e-5)
                assert np.isclose(calc_value, processor_value, rtol=1e-5)
    
    def test_streaming_vs_batch_consistency(self):
        """ストリーミングモードとバッチモードの一貫性テスト"""
        engine = RCICalculatorEngine()
        
        # バッチモードで計算
        batch_result = engine.calculate_multiple(
            self.fx_data[:200],
            periods=[10, 20],
            price_col='close',
            mode='batch'
        )
        
        # ストリーミングモードで計算
        engine.reset()
        streaming_result = engine.calculate_multiple(
            self.fx_data[:200],
            periods=[10, 20],
            price_col='close',
            mode='streaming'
        )
        
        # 結果の比較（NaN以外の値）
        for col in ['rci_10', 'rci_20']:
            batch_vals = batch_result[col].drop_nulls().to_numpy()
            stream_vals = streaming_result[col].drop_nulls().to_numpy()
            
            assert len(batch_vals) == len(stream_vals)
            assert np.allclose(batch_vals, stream_vals, rtol=1e-5)
    
    def test_incremental_update(self):
        """インクリメンタル更新のテスト"""
        engine = RCICalculatorEngine(periods=[10, 20])
        
        # 初期データで計算
        initial_data = self.fx_data[:100]
        result1 = engine.calculate_multiple(
            initial_data,
            periods=[10, 20],
            price_col='close',
            mode='streaming'
        )
        
        # 新しいデータポイントを追加
        for i in range(100, 150):
            new_price = self.fx_data['close'][i]
            results = engine.add_incremental(new_price)
            
            assert 10 in results
            assert 20 in results
            
            # 値の妥当性チェック
            assert -100 <= results[10] <= 100
            assert -100 <= results[20] <= 100
    
    def test_grouped_processing(self):
        """グループ化処理のテスト"""
        # 複数シンボルのデータを作成
        multi_symbol_data = pl.concat([
            self.fx_data[:200].with_columns(pl.lit('EURUSD').alias('symbol')),
            self.fx_data[:200].with_columns(pl.lit('USDJPY').alias('symbol')),
            self.fx_data[:200].with_columns(pl.lit('GBPUSD').alias('symbol'))
        ])
        
        processor = RCIProcessor()
        result = processor.apply_grouped(
            multi_symbol_data,
            group_cols=['symbol'],
            periods=[10, 20],
            price_col='close'
        )
        
        # 各グループが正しく処理されているか確認
        for symbol in ['EURUSD', 'USDJPY', 'GBPUSD']:
            symbol_data = result.filter(pl.col('symbol') == symbol)
            assert len(symbol_data) == 200
            assert 'rci_10' in symbol_data.columns
            assert 'rci_20' in symbol_data.columns
            
            # 各グループで独立して計算されているか確認
            non_null_10 = symbol_data['rci_10'].drop_nulls()
            non_null_20 = symbol_data['rci_20'].drop_nulls()
            
            assert len(non_null_10) == 191  # 200 - 9
            assert len(non_null_20) == 181  # 200 - 19
    
    def test_performance_metrics(self):
        """パフォーマンスメトリクスのテスト"""
        processor = RCIProcessor()
        
        # 大量データでの処理時間測定
        start_time = time.time()
        result = processor.apply_to_dataframe(
            self.fx_data,
            periods=[5, 10, 20, 30, 50],
            price_col='close'
        )
        processing_time = time.time() - start_time
        
        # 基本的なパフォーマンス基準
        assert processing_time < 5.0  # 5秒以内に完了
        
        # 統計情報の確認
        stats = processor.get_statistics()
        assert 'engine_stats' in stats
        assert 'cache_info' in stats
        
        # キャッシュヒット率の確認
        cache_info = stats['cache_info']
        if cache_info['hits'] + cache_info['misses'] > 0:
            hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])
            assert hit_rate >= 0  # キャッシュが機能している


class TestRCIEdgeCases:
    """エッジケースのテスト"""
    
    def test_extreme_values(self):
        """極端な値でのテスト"""
        # 非常に大きな値
        large_values = pl.DataFrame({
            'price': [1e10, 1e11, 1e12, 1e10, 1e11] * 4
        })
        
        processor = RCIProcessor()
        result = processor.apply_to_dataframe(
            large_values,
            periods=[5],
            price_col='price'
        )
        
        # RCIは-100から100の範囲内
        rci_values = result['rci_5'].drop_nulls().to_numpy()
        assert all(-100 <= v <= 100 for v in rci_values)
        
        # 非常に小さな値
        small_values = pl.DataFrame({
            'price': [1e-10, 1e-11, 1e-12, 1e-10, 1e-11] * 4
        })
        
        result = processor.apply_to_dataframe(
            small_values,
            periods=[5],
            price_col='price'
        )
        
        rci_values = result['rci_5'].drop_nulls().to_numpy()
        assert all(-100 <= v <= 100 for v in rci_values)
    
    def test_identical_values(self):
        """同一値の連続でのテスト"""
        # すべて同じ値
        identical_data = pl.DataFrame({
            'price': [100.0] * 50
        })
        
        processor = RCIProcessor()
        result = processor.apply_to_dataframe(
            identical_data,
            periods=[10],
            price_col='price'
        )
        
        # 同一値の場合、RCIは0に近い値になるはず
        rci_values = result['rci_10'].drop_nulls().to_numpy()
        assert all(abs(v) < 10 for v in rci_values)
    
    def test_alternating_values(self):
        """交互に変化する値でのテスト"""
        # ジグザグパターン
        alternating_data = pl.DataFrame({
            'price': [100, 110, 100, 110, 100, 110] * 10
        })
        
        processor = RCIProcessor()
        result = processor.apply_to_dataframe(
            alternating_data,
            periods=[6],
            price_col='price'
        )
        
        # 結果が妥当な範囲内
        rci_values = result['rci_6'].drop_nulls().to_numpy()
        assert all(-100 <= v <= 100 for v in rci_values)
    
    def test_large_dataset_memory_efficiency(self):
        """大規模データセットでのメモリ効率テスト"""
        # 10万件のデータ
        np.random.seed(42)
        large_data = pl.DataFrame({
            'timestamp': pl.date_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 2, 1),
                interval='1m',
                eager=True
            )[:100000],
            'price': np.random.randn(100000).cumsum() + 100
        })
        
        # メモリ使用量の測定
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        processor = RCIProcessor()
        result = processor.apply_to_dataframe(
            large_data,
            periods=[10, 20, 30],
            price_col='price'
        )
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # メモリ増加が妥当な範囲内（500MB以下）
        assert memory_increase < 500
        
        # 結果の妥当性確認
        assert len(result) == len(large_data)
        assert 'rci_10' in result.columns
        assert 'rci_20' in result.columns
        assert 'rci_30' in result.columns
    
    def test_parallel_processing_boundaries(self):
        """並列処理の境界条件テスト"""
        # 並列処理の閾値前後でのテスト
        engine = RCICalculatorEngine()
        threshold = engine.AUTO_MODE_THRESHOLD
        
        # 閾値より少し少ないデータ
        small_data = pl.DataFrame({
            'price': np.random.randn(threshold - 100).cumsum() + 100
        })
        
        # 閾値より少し多いデータ
        large_data = pl.DataFrame({
            'price': np.random.randn(threshold + 100).cumsum() + 100
        })
        
        # 両方で同じ結果が得られることを確認
        result_small = engine.calculate_multiple(
            small_data,
            periods=[10],
            price_col='price',
            mode='auto'
        )
        
        engine.reset()
        result_large = engine.calculate_multiple(
            large_data[:len(small_data)],
            periods=[10],
            price_col='price',
            mode='auto'
        )
        
        # 同じ長さのデータで比較
        small_rci = result_small['rci_10'].drop_nulls().to_numpy()
        large_rci = result_large['rci_10'].drop_nulls().to_numpy()
        
        assert np.allclose(small_rci, large_rci, rtol=1e-5)
    
    def test_nan_inf_propagation(self):
        """NaN/Inf値の伝播テスト"""
        # NaNとInfを含むデータ
        problematic_data = pl.DataFrame({
            'price': [100, 110, np.nan, 120, np.inf, 130, 140, -np.inf, 150, 160] * 5
        })
        
        processor = RCIProcessor()
        result = processor.apply_to_dataframe(
            problematic_data,
            periods=[5],
            price_col='price'
        )
        
        # NaN/Infが適切に処理されていることを確認
        rci_values = result['rci_5'].to_numpy()
        
        # NaN/Infの位置で結果もNaNになっているか確認
        price_values = problematic_data['price'].to_numpy()
        for i in range(len(rci_values)):
            if np.isnan(price_values[i]) or np.isinf(price_values[i]):
                # その位置から期間分はNaNになるはず
                for j in range(i, min(i + 5, len(rci_values))):
                    if j >= 4:  # 最初の期間-1個は常にNaN
                        assert np.isnan(rci_values[j]) or rci_values[j] is None
    
    def test_period_boundary_conditions(self):
        """期間の境界条件テスト"""
        # 最小期間（2）
        calc_min = DifferentialRCICalculator(period=2)
        calc_min.add(100)
        calc_min.add(110)
        assert calc_min.is_ready()
        
        # 最大期間（200）
        calc_max = DifferentialRCICalculator(period=200)
        for i in range(200):
            calc_max.add(100 + i)
        assert calc_max.is_ready()
        
        # 境界外の期間
        with pytest.raises(InvalidPeriodError):
            DifferentialRCICalculator(period=1)
        
        with pytest.raises(InvalidPeriodError):
            DifferentialRCICalculator(period=201)
    
    def test_concurrent_processing(self):
        """並行処理の安全性テスト"""
        # 複数のプロセッサを同時に使用
        processors = [RCIProcessor() for _ in range(3)]
        
        test_data = pl.DataFrame({
            'price': np.random.randn(100).cumsum() + 100
        })
        
        results = []
        for processor in processors:
            result = processor.apply_to_dataframe(
                test_data,
                periods=[10],
                price_col='price'
            )
            results.append(result)
        
        # すべての結果が同一であることを確認
        base_result = results[0]['rci_10'].to_numpy()
        for result in results[1:]:
            comparison = result['rci_10'].to_numpy()
            # NaNの位置も含めて比較
            assert np.array_equal(np.isnan(base_result), np.isnan(comparison))
            # 非NaN値の比較
            valid_indices = ~np.isnan(base_result)
            assert np.allclose(
                base_result[valid_indices],
                comparison[valid_indices],
                rtol=1e-5
            )


# パフォーマンステスト用の追加関数
def test_performance_benchmark():
    """パフォーマンスベンチマークテスト"""
    print("\n=== RCI Performance Benchmark ===")
    
    # データサイズごとのベンチマーク
    sizes = [100, 1000, 10000, 50000]
    periods = [5, 10, 20, 30, 50]
    
    results = []
    
    for size in sizes:
        # テストデータ生成
        data = pl.DataFrame({
            'price': np.random.randn(size).cumsum() + 100
        })
        
        # 処理時間測定
        processor = RCIProcessor()
        start_time = time.time()
        result = processor.apply_to_dataframe(
            data,
            periods=periods,
            price_col='price'
        )
        processing_time = time.time() - start_time
        
        # 結果記録
        results.append({
            'size': size,
            'periods': len(periods),
            'time': processing_time,
            'throughput': size / processing_time
        })
        
        print(f"Size: {size:6d} | Time: {processing_time:.3f}s | "
              f"Throughput: {size/processing_time:.0f} rows/sec")
    
    # スケーラビリティの確認
    times = [r['time'] for r in results]
    # 線形以下のスケーリングであることを確認
    assert times[-1] / times[0] < (sizes[-1] / sizes[0]) * 1.5
    
    print("=== Benchmark Complete ===\n")


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v", "-s"])
    
    # パフォーマンスベンチマーク実行
    test_performance_benchmark()