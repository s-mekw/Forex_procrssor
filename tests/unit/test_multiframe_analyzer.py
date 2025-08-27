"""
MultiTimeframeAnalyzerクラスのユニットテスト

このファイルでは以下をテストします：
1. MultiTimeframeAnalyzerの初期化と設定
2. 短期RCI（1分足）の計算
3. 長期RCI（5分足）の計算
4. データ統合ロジック
5. 並列処理と逐次処理
6. ストリーミング分析
7. エラーハンドリングとエッジケース
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import time

from src.data_processing.analyzer import (
    MultiTimeframeAnalyzer,
    MultiTimeframeAnalysisError,
    AnalysisConfigurationError,
    DataAlignmentError,
)
from src.data_processing.rci import InvalidPeriodError


class TestMultiTimeframeAnalyzerInitialization:
    """MultiTimeframeAnalyzerの初期化テスト"""

    def test_default_initialization(self):
        """デフォルト値での初期化テスト"""
        analyzer = MultiTimeframeAnalyzer()
        
        assert analyzer.short_term_periods == [9, 13, 24, 33, 48, 66, 108]
        assert analyzer.long_term_periods == [24, 33, 48, 66, 108]
        assert analyzer.long_timeframe == "5T"
        assert analyzer.incomplete_bar_handling == "drop"
        assert analyzer.use_parallel is True
        assert analyzer.max_workers is None

    def test_custom_initialization(self):
        """カスタム値での初期化テスト"""
        short_periods = [5, 10, 15]
        long_periods = [20, 30, 40]
        
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=short_periods,
            long_term_periods=long_periods,
            long_timeframe="15T",
            incomplete_bar_handling="keep",
            use_parallel=False,
            max_workers=4
        )
        
        assert analyzer.short_term_periods == short_periods
        assert analyzer.long_term_periods == long_periods
        assert analyzer.long_timeframe == "15T"
        assert analyzer.incomplete_bar_handling == "keep"
        assert analyzer.use_parallel is False
        assert analyzer.max_workers == 4

    def test_component_initialization(self):
        """内部コンポーネントの初期化テスト"""
        analyzer = MultiTimeframeAnalyzer()
        
        # TimeframeConverterが初期化されていることを確認
        assert analyzer.timeframe_converter is not None
        assert analyzer.timeframe_converter.target_timeframe == "5T"
        assert analyzer.timeframe_converter.align_to_boundary is True
        
        # RCIエンジンが初期化されていることを確認
        assert analyzer.short_term_engine is not None
        assert analyzer.long_term_engine is not None

    def test_invalid_period_validation(self):
        """無効な期間での初期化エラーテスト"""
        # 期間が小さすぎる
        with pytest.raises(AnalysisConfigurationError) as exc_info:
            MultiTimeframeAnalyzer(short_term_periods=[1])  # 最小値は2
        assert "無効な期間" in str(exc_info.value)
        
        # 期間が大きすぎる
        with pytest.raises(AnalysisConfigurationError) as exc_info:
            MultiTimeframeAnalyzer(long_term_periods=[201])  # 最大値は200
        assert "無効な期間" in str(exc_info.value)

    def test_invalid_timeframe_error(self):
        """無効なタイムフレームでの初期化エラーテスト"""
        with pytest.raises(AnalysisConfigurationError) as exc_info:
            MultiTimeframeAnalyzer(long_timeframe="2T")  # サポートされていない
        assert "コンポーネント初期化エラー" in str(exc_info.value)

    def test_get_analyzer_info(self):
        """アナライザー情報取得のテスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13],
            long_term_periods=[24, 33],
            long_timeframe="15T",
            use_parallel=False
        )
        
        info = analyzer.get_analyzer_info()
        assert info['short_term_periods'] == [9, 13]
        assert info['long_term_periods'] == [24, 33]
        assert info['long_timeframe'] == "15T"
        assert info['incomplete_bar_handling'] == "drop"
        assert info['use_parallel'] is False


class TestMultiTimeframeAnalysis:
    """マルチタイムフレーム分析のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.create_test_data()
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13],
            long_term_periods=[24, 33],
            use_parallel=False  # テストでは逐次処理を使用
        )

    def create_test_data(self):
        """テスト用の1分足データを作成"""
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        n_minutes = 200  # 200分のデータ
        
        timestamps = [base_time + timedelta(minutes=i) for i in range(n_minutes)]
        
        # トレンドのあるデータを生成
        base_price = 100.0
        prices = []
        for i in range(n_minutes):
            if i < 50:  # 上昇トレンド
                trend = i * 0.1
            elif i < 100:  # 下降トレンド
                trend = 50 * 0.1 - (i - 50) * 0.1
            else:  # レンジ相場
                trend = 0
            
            noise = np.random.normal(0, 0.5)
            prices.append(base_price + trend + noise)
        
        self.test_data = pl.DataFrame({
            'timestamp': timestamps,
            'open': [p + np.random.normal(0, 0.1) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 1000, n_minutes)
        })

    def test_analyze_basic(self):
        """基本的な分析機能のテスト"""
        result = self.analyzer.analyze(self.test_data)
        
        # 結果の形状確認
        assert len(result) == len(self.test_data)
        
        # 必須カラムの存在確認
        assert 'timestamp' in result.columns
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        
        # 短期RCIカラムの存在確認
        assert 'short_rci_9' in result.columns
        assert 'short_rci_13' in result.columns
        
        # 長期RCIカラムの存在確認
        assert 'long_rci_24' in result.columns
        assert 'long_rci_33' in result.columns

    def test_analyze_with_intermediate_results(self):
        """中間結果を含む分析のテスト"""
        results = self.analyzer.analyze(self.test_data, return_intermediate=True)
        
        assert isinstance(results, dict)
        assert 'combined' in results
        assert 'short_term' in results
        assert 'long_term' in results
        assert 'long_term_ohlcv' in results
        
        # 各結果の妥当性確認
        combined = results['combined']
        assert len(combined) == len(self.test_data)
        
        short_term = results['short_term']
        assert len(short_term) == len(self.test_data)
        assert 'short_rci_9' in short_term.columns
        assert 'short_rci_13' in short_term.columns
        
        long_term = results['long_term']
        assert 'long_rci_24' in long_term.columns
        assert 'long_rci_33' in long_term.columns
        
        long_ohlcv = results['long_term_ohlcv']
        assert len(long_ohlcv) == len(self.test_data) // 5  # 5分足なので1/5

    def test_short_term_rci_calculation(self):
        """短期RCI計算の正確性テスト"""
        # 小さなデータセットで確認
        small_data = self.test_data[:20]
        
        result = self.analyzer.analyze(small_data)
        
        # 短期RCIの値が適切な範囲内にあることを確認
        short_rci_9 = result['short_rci_9'].drop_nulls().to_numpy()
        short_rci_13 = result['short_rci_13'].drop_nulls().to_numpy()
        
        # RCIは-100から100の範囲内
        assert all(-100 <= v <= 100 for v in short_rci_9)
        assert all(-100 <= v <= 100 for v in short_rci_13)
        
        # 最初の期間-1行はNaN
        assert result['short_rci_9'][:8].null_count() == 8
        assert result['short_rci_13'][:12].null_count() == 12

    def test_long_term_rci_calculation(self):
        """長期RCI計算の正確性テスト"""
        result = self.analyzer.analyze(self.test_data)
        
        # 長期RCIの値が適切な範囲内にあることを確認
        long_rci_24 = result.select('long_rci_24').drop_nulls()
        if len(long_rci_24) > 0:
            values = long_rci_24['long_rci_24'].to_numpy()
            assert all(-100 <= v <= 100 for v in values)

    def test_timestamp_alignment(self):
        """タイムスタンプアライメントのテスト"""
        result = self.analyzer.analyze(self.test_data)
        
        # タイムスタンプが元のデータと一致することを確認
        assert result['timestamp'].to_list() == self.test_data['timestamp'].to_list()
        
        # 長期RCIが正しい期間にアライメントされていることを確認
        # 5分足のRCIは5分間同じ値を持つ
        for i in range(0, len(result) - 5, 5):
            if not result['long_rci_24'][i] is None:
                # 同じ5分期間内のRCI値が同じ
                period_values = [result['long_rci_24'][i + j] for j in range(5)]
                non_null_values = [v for v in period_values if v is not None]
                if len(non_null_values) > 1:
                    assert len(set(non_null_values)) == 1

    def test_parallel_vs_sequential_processing(self):
        """並列処理と逐次処理の結果一致性テスト"""
        # 逐次処理
        analyzer_seq = MultiTimeframeAnalyzer(
            short_term_periods=[9],
            long_term_periods=[24],
            use_parallel=False
        )
        result_seq = analyzer_seq.analyze(self.test_data[:100])
        
        # 並列処理
        analyzer_par = MultiTimeframeAnalyzer(
            short_term_periods=[9],
            long_term_periods=[24],
            use_parallel=True,
            max_workers=2
        )
        result_par = analyzer_par.analyze(self.test_data[:100])
        
        # 結果が一致することを確認
        assert len(result_seq) == len(result_par)
        
        # 短期RCIの比較
        seq_short = result_seq['short_rci_9'].drop_nulls().to_numpy()
        par_short = result_par['short_rci_9'].drop_nulls().to_numpy()
        assert np.allclose(seq_short, par_short, rtol=1e-5)
        
        # 長期RCIの比較
        seq_long = result_seq.select('long_rci_24').drop_nulls()
        par_long = result_par.select('long_rci_24').drop_nulls()
        if len(seq_long) > 0 and len(par_long) > 0:
            assert np.allclose(
                seq_long['long_rci_24'].to_numpy(),
                par_long['long_rci_24'].to_numpy(),
                rtol=1e-5
            )


class TestStreamingAnalysis:
    """ストリーミング分析のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13],
            long_term_periods=[24, 33]
        )
        
        # ストリーミング用の履歴データ
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        n_minutes = 100
        timestamps = [base_time + timedelta(minutes=i) for i in range(n_minutes)]
        prices = np.random.randn(n_minutes).cumsum() + 100
        
        self.history = pl.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.random.rand(n_minutes),
            'low': prices - np.random.rand(n_minutes),
            'close': prices + np.random.randn(n_minutes) * 0.1,
            'volume': np.random.randint(100, 1000, n_minutes)
        })

    def test_streaming_analysis_basic(self):
        """基本的なストリーミング分析のテスト"""
        # 新しいバー
        new_bar = {
            'timestamp': datetime(2024, 1, 1, 10, 40, 0),
            'open': 105.0,
            'high': 106.0,
            'low': 104.5,
            'close': 105.5,
            'volume': 500
        }
        
        result = self.analyzer.analyze_streaming(new_bar, self.history)
        
        assert 'timestamp' in result
        assert 'short_rci' in result
        assert 'long_rci' in result
        assert 'is_new_long_bar' in result
        
        # 短期RCIが計算されていることを確認
        assert len(result['short_rci']) > 0
        for period, value in result['short_rci'].items():
            assert -100 <= value <= 100

    def test_streaming_on_5min_boundary(self):
        """5分境界でのストリーミング分析のテスト"""
        # 5分境界のバー（10:45:00）
        new_bar = {
            'timestamp': datetime(2024, 1, 1, 10, 45, 0),
            'open': 105.0,
            'high': 106.0,
            'low': 104.5,
            'close': 105.5,
            'volume': 500
        }
        
        result = self.analyzer.analyze_streaming(new_bar, self.history)
        
        # 5分境界なので新しい長期バーが完成
        assert result['is_new_long_bar'] is True
        
        # 長期RCIが計算されていることを確認
        if len(result['long_rci']) > 0:
            for period, value in result['long_rci'].items():
                assert -100 <= value <= 100

    def test_streaming_not_on_boundary(self):
        """5分境界でない場合のストリーミング分析のテスト"""
        # 5分境界でないバー（10:42:00）
        new_bar = {
            'timestamp': datetime(2024, 1, 1, 10, 42, 0),
            'open': 105.0,
            'high': 106.0,
            'low': 104.5,
            'close': 105.5,
            'volume': 500
        }
        
        result = self.analyzer.analyze_streaming(new_bar, self.history)
        
        # 5分境界でないので新しい長期バーは完成しない
        assert result['is_new_long_bar'] is False
        
        # 長期RCIは計算されない（または空）
        assert len(result['long_rci']) == 0

    def test_streaming_with_insufficient_history(self):
        """履歴データ不足でのストリーミング分析のテスト"""
        # 少ない履歴データ
        small_history = self.history[:5]
        
        new_bar = {
            'timestamp': datetime(2024, 1, 1, 9, 5, 0),
            'open': 100.0,
            'high': 101.0,
            'low': 99.5,
            'close': 100.5,
            'volume': 200
        }
        
        result = self.analyzer.analyze_streaming(new_bar, small_history)
        
        # 短期RCIは期間が足りない場合計算されない
        assert len(result['short_rci']) == 0  # 9期間必要だが6本しかない

    def test_streaming_memory_efficiency(self):
        """ストリーミング時のメモリ効率テスト"""
        # 大量の履歴データ
        large_history = pl.concat([self.history] * 50)  # 5000本のデータ
        
        new_bar = {
            'timestamp': datetime(2024, 1, 2, 9, 0, 0),
            'open': 110.0,
            'high': 111.0,
            'low': 109.5,
            'close': 110.5,
            'volume': 600
        }
        
        # analyze_streamingは履歴データのサイズを制限する
        result = self.analyzer.analyze_streaming(new_bar, large_history)
        
        # 結果が正常に返されることを確認
        assert 'short_rci' in result
        assert 'long_rci' in result


class TestDataValidation:
    """データ検証のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.analyzer = MultiTimeframeAnalyzer()

    def test_missing_columns_validation(self):
        """必須カラムが不足している場合のテスト"""
        # タイムスタンプカラムが不足
        invalid_data = pl.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [100]
        })
        
        with pytest.raises(ValueError) as exc_info:
            self.analyzer.analyze(invalid_data)
        assert "必須カラムが不足" in str(exc_info.value)

    def test_empty_dataframe_validation(self):
        """空のデータフレームの検証テスト"""
        empty_data = pl.DataFrame({
            'timestamp': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        })
        
        with pytest.raises(ValueError) as exc_info:
            self.analyzer.analyze(empty_data)
        assert "データが空です" in str(exc_info.value)

    def test_unsorted_timestamps(self):
        """ソートされていないタイムスタンプの処理テスト"""
        # 逆順のタイムスタンプ
        timestamps = pd.date_range('2024-01-01', periods=50, freq='1min')[::-1]
        data = pl.DataFrame({
            'timestamp': timestamps,
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102] * 50,
            'volume': [100] * 50
        })
        
        # 警告が出るが処理は続行される
        result = self.analyzer.analyze(data)
        assert len(result) == 50


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.analyzer = MultiTimeframeAnalyzer()

    def test_rci_calculation_error_propagation(self):
        """RCI計算エラーの伝播テスト"""
        # NaN値を含むデータ
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min'),
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [float('nan')] * 25 + [102] * 25,  # 前半がNaN
            'volume': [100] * 50
        })
        
        # エラーは発生せず、NaN部分の結果もNaNになる
        result = self.analyzer.analyze(data)
        assert len(result) == 50

    def test_timeframe_conversion_error(self):
        """タイムフレーム変換エラーのテスト"""
        # 不正なタイムフレームでアナライザーを初期化しようとする
        with pytest.raises(AnalysisConfigurationError):
            MultiTimeframeAnalyzer(long_timeframe="invalid")

    def test_parallel_processing_timeout(self):
        """並列処理のタイムアウトテスト"""
        # モックを使用してタイムアウトをシミュレート
        with patch('src.data_processing.analyzer.ThreadPoolExecutor') as mock_executor:
            mock_future = Mock()
            mock_future.result.side_effect = TimeoutError("Timeout")
            
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            analyzer = MultiTimeframeAnalyzer(use_parallel=True)
            
            data = pl.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
                'open': [100] * 10,
                'high': [105] * 10,
                'low': [95] * 10,
                'close': [102] * 10,
                'volume': [100] * 10
            })
            
            with pytest.raises(MultiTimeframeAnalysisError):
                analyzer.analyze(data)


class TestIntegrationWithComponents:
    """コンポーネント統合のテスト"""

    def test_timeframe_converter_integration(self):
        """TimeframeConverterとの統合テスト"""
        analyzer = MultiTimeframeAnalyzer(
            long_timeframe="15T",  # 15分足
            incomplete_bar_handling="keep"
        )
        
        # 50分のデータ（3本の完全な15分足 + 5分の不完全）
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:00', periods=50, freq='1min'),
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102 + i * 0.01 for i in range(50)],
            'volume': [100] * 50
        })
        
        results = analyzer.analyze(data, return_intermediate=True)
        
        # 15分足データの確認
        long_ohlcv = results['long_term_ohlcv']
        assert len(long_ohlcv) == 4  # keepモードなので不完全バーも含む

    def test_rci_engine_integration(self):
        """RCIエンジンとの統合テスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13, 24],
            long_term_periods=[24, 33, 48]
        )
        
        # 十分なデータで確実にRCIが計算される
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=300, freq='1min'),
            'open': np.random.randn(300).cumsum() + 100,
            'high': np.random.randn(300).cumsum() + 105,
            'low': np.random.randn(300).cumsum() + 95,
            'close': np.random.randn(300).cumsum() + 102,
            'volume': np.random.randint(100, 1000, 300)
        })
        
        result = analyzer.analyze(data)
        
        # すべてのRCI期間が結果に含まれることを確認
        for period in [9, 13, 24]:
            assert f'short_rci_{period}' in result.columns
        for period in [24, 33, 48]:
            assert f'long_rci_{period}' in result.columns


class TestPerformance:
    """パフォーマンステスト"""

    def test_analysis_performance(self):
        """分析パフォーマンステスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13, 24],
            long_term_periods=[24, 33, 48],
            use_parallel=True
        )
        
        # 1万本の1分足データ
        n_bars = 10000
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1min'),
            'open': np.random.randn(n_bars).cumsum() + 100,
            'high': np.random.randn(n_bars).cumsum() + 105,
            'low': np.random.randn(n_bars).cumsum() + 95,
            'close': np.random.randn(n_bars).cumsum() + 102,
            'volume': np.random.randint(100, 1000, n_bars)
        })
        
        # 分析時間の測定
        start_time = time.time()
        result = analyzer.analyze(data)
        elapsed_time = time.time() - start_time
        
        # 10秒以内に完了すること
        assert elapsed_time < 10.0
        
        # 結果の妥当性確認
        assert len(result) == n_bars

    def test_parallel_processing_speedup(self):
        """並列処理の高速化効果テスト"""
        # 大きめのデータセット
        n_bars = 5000
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1min'),
            'open': np.random.randn(n_bars).cumsum() + 100,
            'high': np.random.randn(n_bars).cumsum() + 105,
            'low': np.random.randn(n_bars).cumsum() + 95,
            'close': np.random.randn(n_bars).cumsum() + 102,
            'volume': np.random.randint(100, 1000, n_bars)
        })
        
        # 逐次処理の時間測定
        analyzer_seq = MultiTimeframeAnalyzer(use_parallel=False)
        start_time = time.time()
        result_seq = analyzer_seq.analyze(data)
        time_seq = time.time() - start_time
        
        # 並列処理の時間測定
        analyzer_par = MultiTimeframeAnalyzer(use_parallel=True, max_workers=2)
        start_time = time.time()
        result_par = analyzer_par.analyze(data)
        time_par = time.time() - start_time
        
        # 並列処理の方が高速（または同等）
        # 小さなデータセットではオーバーヘッドのため逐次処理の方が速い場合もある
        assert time_par <= time_seq * 1.2  # 20%の余裕を持たせる
        
        # 結果が同じ
        assert len(result_seq) == len(result_par)


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_single_bar_analysis(self):
        """単一バーの分析テスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[2],  # 最小期間
            long_term_periods=[2]
        )
        
        data = pl.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 9, 0, 0)],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [100]
        })
        
        # データ不足でもエラーにならない
        result = analyzer.analyze(data)
        assert len(result) == 1
        # RCIはすべてNaN
        assert result['short_rci_2'][0] is None

    def test_exact_period_boundary(self):
        """期間境界ぴったりのデータテスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[10],
            long_term_periods=[10]
        )
        
        # ちょうど50分のデータ（10本の5分足）
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min'),
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': list(range(100, 150)),  # 上昇トレンド
            'volume': [100] * 50
        })
        
        result = analyzer.analyze(data)
        
        # 短期RCIは10本目から計算される
        assert result['short_rci_10'][:9].null_count() == 9
        assert result['short_rci_10'][9] is not None

    def test_all_identical_values(self):
        """すべて同一値のデータテスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9],
            long_term_periods=[24]
        )
        
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'open': [100] * 200,
            'high': [100] * 200,
            'low': [100] * 200,
            'close': [100] * 200,
            'volume': [100] * 200
        })
        
        result = analyzer.analyze(data)
        
        # RCIは0に近い値になる
        short_rci = result['short_rci_9'].drop_nulls().to_numpy()
        assert all(abs(v) < 10 for v in short_rci)

    def test_extreme_values(self):
        """極端な値のテスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[5],
            long_term_periods=[10]
        )
        
        # 非常に大きな値と小さな値
        prices = [1e-10, 1e10, 1e-10, 1e10] * 50
        
        data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'open': prices,
            'high': [p * 1.1 for p in prices],
            'low': [p * 0.9 for p in prices],
            'close': prices,
            'volume': [100] * 200
        })
        
        result = analyzer.analyze(data)
        
        # RCIは-100から100の範囲内
        short_rci = result['short_rci_5'].drop_nulls().to_numpy()
        assert all(-100 <= v <= 100 for v in short_rci)


class TestBufferManagement:
    """バッファ管理機能のテスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行"""
        self.analyzer = MultiTimeframeAnalyzer(max_history_bars=100)
    
    def test_add_new_bar_basic(self):
        """基本的なバー追加のテスト"""
        # 初期状態の確認
        assert self.analyzer.get_buffer_size() == 0
        
        # 新しいバーを追加
        new_bar = {
            'timestamp': datetime(2024, 1, 1, 9, 0, 0),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000
        }
        self.analyzer.add_new_bar(new_bar)
        
        # バッファサイズが増加することを確認
        assert self.analyzer.get_buffer_size() == 1
        
        # 追加したバーがバッファに存在することを確認
        buffer_df = self.analyzer.get_buffer_as_dataframe()
        assert buffer_df is not None
        assert len(buffer_df) == 1
        assert buffer_df['close'][0] == 102.0
    
    def test_buffer_size_limit(self):
        """バッファサイズ制限のテスト"""
        # max_history_bars=10で再初期化
        analyzer = MultiTimeframeAnalyzer(max_history_bars=10)
        
        # 15個のバーを追加
        for i in range(15):
            bar = {
                'timestamp': datetime(2024, 1, 1, 9, i, 0),
                'open': 100.0 + i,
                'high': 105.0 + i,
                'low': 95.0 + i,
                'close': 102.0 + i,
                'volume': 1000 + i
            }
            analyzer.add_new_bar(bar)
        
        # バッファサイズが10を超えないことを確認
        assert analyzer.get_buffer_size() == 10
        
        # 最新10個のバーが保持されていることを確認
        buffer_df = analyzer.get_buffer_as_dataframe()
        assert len(buffer_df) == 10
        # 最初のバーがインデックス5のもの（0-4は削除された）
        assert buffer_df['close'][0] == 107.0  # 102.0 + 5
        # 最後のバーがインデックス14のもの
        assert buffer_df['close'][9] == 116.0  # 102.0 + 14
    
    def test_buffer_size_management(self):
        """古いデータが適切に削除されることのテスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=5)
        
        # 最初に5個のバーを追加
        for i in range(5):
            bar = {
                'timestamp': datetime(2024, 1, 1, 9, i, 0),
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 100.0 + i,  # 識別用に異なる値を設定
                'volume': 1000
            }
            analyzer.add_new_bar(bar)
        
        assert analyzer.get_buffer_size() == 5
        
        # 新しいバーを1つ追加
        new_bar = {
            'timestamp': datetime(2024, 1, 1, 9, 5, 0),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 105.0,
            'volume': 1000
        }
        analyzer.add_new_bar(new_bar)
        
        # サイズは5のまま
        assert analyzer.get_buffer_size() == 5
        
        # 最初のバー（close=100.0）が削除され、新しいバーが追加されていることを確認
        buffer_df = analyzer.get_buffer_as_dataframe()
        assert buffer_df['close'][0] == 101.0  # 最も古いバーが削除された
        assert buffer_df['close'][4] == 105.0  # 新しいバーが最後に
    
    def test_get_buffer_size(self):
        """バッファサイズが正しく取得できることのテスト"""
        assert self.analyzer.get_buffer_size() == 0
        
        # 複数のバーを追加してサイズを確認
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(50):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000
            }
            self.analyzer.add_new_bar(bar)
            assert self.analyzer.get_buffer_size() == min(i + 1, 100)  # max_history_bars=100


class TestAnalysisReadiness:
    """分析準備状態のテスト"""
    
    def test_is_ready_with_sufficient_data(self):
        """十分なデータがある場合の判定テスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=500)
        
        # 初期状態では準備未完了
        assert analyzer.is_ready() is False
        
        # 199個のバーを追加（まだ不足）
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(199):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000
            }
            analyzer.add_new_bar(bar)
        
        assert analyzer.is_ready() is False
        
        # 200個目のバーを追加（準備完了）
        bar = {
            'timestamp': base_time + timedelta(minutes=199),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000
        }
        analyzer.add_new_bar(bar)
        
        assert analyzer.is_ready() is True
    
    def test_is_ready_with_insufficient_data(self):
        """データ不足の場合の判定テスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=1000)
        
        # 100個のバーを追加（不足）
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(100):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000
            }
            analyzer.add_new_bar(bar)
        
        # 200個未満なので準備未完了
        assert analyzer.is_ready() is False
        assert analyzer.get_buffer_size() == 100
    
    def test_is_ready_boundary_case(self):
        """境界値（ちょうど200バー）のテスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=200)
        
        # ちょうど200個のバーを追加
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(200):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000
            }
            analyzer.add_new_bar(bar)
        
        # 境界値で準備完了
        assert analyzer.is_ready() is True
        assert analyzer.get_buffer_size() == 200
        
        # さらにバーを追加してもmax_history_barsで制限
        analyzer.add_new_bar({
            'timestamp': base_time + timedelta(minutes=200),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000
        })
        
        assert analyzer.is_ready() is True
        assert analyzer.get_buffer_size() == 200  # サイズは200のまま


class TestDataFrameConversion:
    """DataFrame変換のテスト"""
    
    def test_get_buffer_as_dataframe_with_data(self):
        """データがある場合の変換テスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=100)
        
        # 10個のバーを追加
        timestamps = []
        closes = []
        for i in range(10):
            timestamp = datetime(2024, 1, 1, 9, i, 0)
            close = 100.0 + i
            bar = {
                'timestamp': timestamp,
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': close,
                'volume': 1000 + i
            }
            analyzer.add_new_bar(bar)
            timestamps.append(timestamp)
            closes.append(close)
        
        # DataFrameとして取得
        df = analyzer.get_buffer_as_dataframe()
        assert df is not None
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        
        # データの内容を確認
        assert df['timestamp'].to_list() == timestamps
        assert df['close'].to_list() == closes
    
    def test_get_buffer_as_dataframe_empty(self):
        """空バッファの場合の処理テスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=100)
        
        # バッファが空の場合
        df = analyzer.get_buffer_as_dataframe()
        assert df is None
    
    def test_dataframe_column_types(self):
        """変換後のカラム型の確認テスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=100)
        
        # サンプルバーを追加
        bar = {
            'timestamp': datetime(2024, 1, 1, 9, 0, 0),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000
        }
        analyzer.add_new_bar(bar)
        
        df = analyzer.get_buffer_as_dataframe()
        assert df is not None
        
        # カラムの存在確認
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in df.columns
        
        # データ型の確認（timestampは日時型、その他は数値型）
        assert df.dtypes[0] == pl.Datetime  # timestamp
        for i in range(1, 6):
            assert df.dtypes[i] in [pl.Float64, pl.Int64, pl.Int32, pl.Float32]  # 数値型


class TestInternalBufferMode:
    """内部バッファモードのテスト"""
    
    def test_analyze_streaming_internal_buffer(self):
        """内部バッファを使用した分析テスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9, 13],
            long_term_periods=[24, 33],
            max_history_bars=500
        )
        
        # 250個のバーを追加（分析可能な状態にする）
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(250):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 100.0 + np.random.randn() * 0.5,
                'high': 105.0 + np.random.randn() * 0.5,
                'low': 95.0 + np.random.randn() * 0.5,
                'close': 102.0 + i * 0.01,  # トレンド付き
                'volume': 1000 + np.random.randint(0, 100)
            }
            analyzer.add_new_bar(bar)
        
        assert analyzer.is_ready() is True
        
        # 内部バッファモードで分析（historyパラメータなし）
        result = analyzer.analyze_streaming()
        
        assert result is not None
        # 'status'は正常時には含まれない（not_readyやno_dataの場合のみ）
        assert 'timestamp' in result  # 代わりにtimestampをチェック
        
        # RCI結果の確認
        assert 'short_rci' in result
        assert 'long_rci' in result
        assert isinstance(result['short_rci'], dict)
        assert isinstance(result['long_rci'], dict)
        
        # 短期RCIが計算されていることを確認
        assert len(result['short_rci']) > 0
        for period in [9, 13]:
            assert period in result['short_rci']
            assert -100 <= result['short_rci'][period] <= 100
    
    def test_analyze_streaming_not_ready(self):
        """準備未完了時の応答テスト"""
        analyzer = MultiTimeframeAnalyzer(max_history_bars=500)
        
        # 少数のバーを追加（準備未完了）
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        for i in range(50):
            bar = {
                'timestamp': base_time + timedelta(minutes=i),
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000
            }
            analyzer.add_new_bar(bar)
        
        assert analyzer.is_ready() is False
        
        # 内部バッファモードで分析
        result = analyzer.analyze_streaming()
        
        assert result is not None
        assert 'status' in result
        assert result['status'] == 'not_ready'
        assert 'buffer_size' in result
        assert result['buffer_size'] == 50
        assert 'required_bars' in result
        assert result['required_bars'] == 200
    
    def test_analyze_streaming_backward_compatibility(self):
        """後方互換性の確認テスト"""
        analyzer = MultiTimeframeAnalyzer(
            short_term_periods=[9],
            long_term_periods=[24],
            max_history_bars=500
        )
        
        # 外部から履歴データを準備
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        n_minutes = 100
        timestamps = [base_time + timedelta(minutes=i) for i in range(n_minutes)]
        prices = np.random.randn(n_minutes).cumsum() + 100
        
        history = pl.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.random.rand(n_minutes),
            'low': prices - np.random.rand(n_minutes),
            'close': prices + np.random.randn(n_minutes) * 0.1,
            'volume': np.random.randint(100, 1000, n_minutes).astype(np.int64)  # 明示的にInt64に変換
        })
        
        new_bar = {
            'timestamp': datetime(2024, 1, 1, 10, 40, 0),
            'open': 105.0,
            'high': 106.0,
            'low': 104.5,
            'close': 105.5,
            'volume': 500
        }
        
        # 外部履歴を使用した分析（後方互換性）
        result = analyzer.analyze_streaming(new_bar, history, min_history_bars=50)
        
        assert result is not None
        assert 'timestamp' in result
        assert 'short_rci' in result
        assert 'long_rci' in result
        
        # 短期RCIが計算されていることを確認
        assert len(result['short_rci']) > 0
        assert 9 in result['short_rci']


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v", "-s"])