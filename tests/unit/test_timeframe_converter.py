"""
TimeframeConverterクラスのユニットテスト

このファイルでは以下をテストします：
1. TimeframeConverterの初期化と設定
2. タイムフレーム変換ロジック（1分足→5分足、15分足など）
3. 不完全バーの処理（drop、keep、preview）
4. ストリーミングデータの変換
5. データ検証機能
6. エラーハンドリングとエッジケース
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.data_processing.timeframe_converter import (
    TimeframeConverter,
    TimeframeConversionError,
    InvalidTimeframeError,
    InsufficientBarsError,
)


class TestTimeframeConverterInitialization:
    """TimeframeConverterの初期化テスト"""

    def test_valid_initialization(self):
        """正常な初期化のテスト"""
        converter = TimeframeConverter("5T")
        assert converter.source_timeframe == "1T"
        assert converter.target_timeframe == "5T"
        assert converter.align_to_boundary is True
        assert converter._interval == "5m"

    def test_all_supported_timeframes(self):
        """すべてのサポートされたタイムフレームの初期化テスト"""
        supported = ["5T", "15T", "30T", "1H", "4H", "1D"]
        for tf in supported:
            converter = TimeframeConverter(tf)
            assert converter.target_timeframe == tf

    def test_invalid_timeframe_error(self):
        """サポートされていないタイムフレームでのエラーテスト"""
        with pytest.raises(InvalidTimeframeError) as exc_info:
            TimeframeConverter("2T")  # サポートされていない
        assert "Unsupported timeframe" in str(exc_info.value)

    def test_invalid_source_timeframe(self):
        """ソースタイムフレームが1T以外の場合のエラーテスト"""
        with pytest.raises(InvalidTimeframeError) as exc_info:
            TimeframeConverter("5T", source_timeframe="5T")
        assert "Source timeframe must be '1T'" in str(exc_info.value)

    def test_align_to_boundary_option(self):
        """境界整列オプションのテスト"""
        converter_aligned = TimeframeConverter("5T", align_to_boundary=True)
        assert converter_aligned.align_to_boundary is True
        
        converter_not_aligned = TimeframeConverter("5T", align_to_boundary=False)
        assert converter_not_aligned.align_to_boundary is False

    def test_polars_interval_conversion(self):
        """Polarsのinterval形式への変換テスト"""
        mappings = {
            "5T": "5m",
            "15T": "15m",
            "30T": "30m",
            "1H": "1h",
            "4H": "4h",
            "1D": "1d",
        }
        
        for tf, expected in mappings.items():
            converter = TimeframeConverter(tf)
            assert converter._interval == expected


class TestTimeframeConversion:
    """タイムフレーム変換ロジックのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        # テスト用の1分足データを生成
        self.create_test_data()

    def create_test_data(self):
        """テスト用の1分足OHLCVデータを作成"""
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        n_minutes = 60  # 1時間分のデータ
        
        timestamps = [base_time + timedelta(minutes=i) for i in range(n_minutes)]
        
        # リアリスティックな価格データを生成
        base_price = 100.0
        prices = []
        for i in range(n_minutes):
            # トレンドとノイズを含む価格変動
            trend = i * 0.01  # 上昇トレンド
            noise = np.random.normal(0, 0.1)
            prices.append(base_price + trend + noise)
        
        self.test_data = pl.DataFrame({
            'timestamp': timestamps,
            'open': [p + np.random.normal(0, 0.05) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.1)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.1)) for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 1000, n_minutes)
        })

    def test_convert_1min_to_5min(self):
        """1分足から5分足への変換テスト"""
        converter = TimeframeConverter("5T")
        result = converter.convert(self.test_data)
        
        # 期待される行数（60分÷5分=12本）
        assert len(result) == 12
        
        # カラムの存在確認
        assert 'timestamp' in result.columns
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert 'bar_count' in result.columns
        
        # 各5分足バーが5本の1分足から構成されていることを確認
        assert all(result['bar_count'] == 5)
        
        # タイムスタンプが5分間隔であることを確認
        timestamps = result['timestamp'].to_list()
        for i in range(len(timestamps) - 1):
            diff = timestamps[i + 1] - timestamps[i]
            assert diff == timedelta(minutes=5)

    def test_convert_1min_to_15min(self):
        """1分足から15分足への変換テスト"""
        converter = TimeframeConverter("15T")
        result = converter.convert(self.test_data)
        
        # 期待される行数（60分÷15分=4本）
        assert len(result) == 4
        
        # 各15分足バーが15本の1分足から構成されていることを確認
        assert all(result['bar_count'] == 15)

    def test_convert_1min_to_1hour(self):
        """1分足から1時間足への変換テスト"""
        converter = TimeframeConverter("1H")
        result = converter.convert(self.test_data)
        
        # 期待される行数（60分÷60分=1本）
        assert len(result) == 1
        
        # 1時間足バーが60本の1分足から構成されていることを確認
        assert result['bar_count'][0] == 60

    def test_ohlc_aggregation_correctness(self):
        """OHLC集約の正確性テスト"""
        # 簡単なデータで正確性を確認
        simple_data = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:00', periods=5, freq='1min'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [101, 102, 103, 104, 105],
            'volume': [100, 200, 300, 400, 500]
        })
        
        converter = TimeframeConverter("5T")
        result = converter.convert(simple_data)
        
        assert len(result) == 1
        assert result['open'][0] == 100  # First価格
        assert result['high'][0] == 109  # Max価格
        assert result['low'][0] == 95   # Min価格
        assert result['close'][0] == 105  # Last価格
        assert result['volume'][0] == 1500  # Sum of volumes

    def test_incomplete_bar_handling_drop(self):
        """不完全バー処理（drop）のテスト"""
        # 62分のデータ（12本の完全な5分足バー + 2分の不完全バー）
        timestamps = pd.date_range('2024-01-01 09:00', periods=62, freq='1min')
        data = pl.DataFrame({
            'timestamp': timestamps,
            'open': [100] * 62,
            'high': [105] * 62,
            'low': [95] * 62,
            'close': [102] * 62,
            'volume': [100] * 62
        })
        
        converter = TimeframeConverter("5T")
        result = converter.convert(data, incomplete_bar_handling="drop")
        
        # 不完全バーは削除されるので12本のみ
        assert len(result) == 12

    def test_incomplete_bar_handling_keep(self):
        """不完全バー処理（keep）のテスト"""
        # 62分のデータ
        timestamps = pd.date_range('2024-01-01 09:00', periods=62, freq='1min')
        data = pl.DataFrame({
            'timestamp': timestamps,
            'open': [100] * 62,
            'high': [105] * 62,
            'low': [95] * 62,
            'close': [102] * 62,
            'volume': [100] * 62
        })
        
        converter = TimeframeConverter("5T")
        result = converter.convert(data, incomplete_bar_handling="keep")
        
        # 不完全バーも保持されるので13本
        assert len(result) == 13
        # 最後のバーは2本の1分足から構成
        assert result['bar_count'][-1] == 2

    def test_incomplete_bar_handling_preview(self):
        """不完全バー処理（preview）のテスト"""
        # 62分のデータ
        timestamps = pd.date_range('2024-01-01 09:00', periods=62, freq='1min')
        data = pl.DataFrame({
            'timestamp': timestamps,
            'open': [100] * 62,
            'high': [105] * 62,
            'low': [95] * 62,
            'close': [102] * 62,
            'volume': [100] * 62
        })
        
        converter = TimeframeConverter("5T")
        result = converter.convert(data, incomplete_bar_handling="preview")
        
        # プレビューモードでは不完全バーにマークが付く
        assert len(result) == 13
        assert 'is_complete' in result.columns
        # 最初の12本は完全
        assert all(result['is_complete'][:12])
        # 最後の1本は不完全
        assert not result['is_complete'][12]


class TestStreamingConversion:
    """ストリーミングデータ変換のテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行"""
        # ストリーミング用のデータを準備
        self.prepare_streaming_data()

    def prepare_streaming_data(self):
        """ストリーミング用データの準備"""
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # 22分のデータ（4本の完全な5分足 + 2分の不完全）
        timestamps = [base_time + timedelta(minutes=i) for i in range(22)]
        
        self.streaming_data = pl.DataFrame({
            'timestamp': timestamps,
            'open': [100 + i * 0.1 for i in range(22)],
            'high': [105 + i * 0.1 for i in range(22)],
            'low': [95 + i * 0.1 for i in range(22)],
            'close': [102 + i * 0.1 for i in range(22)],
            'volume': [100 + i * 10 for i in range(22)]
        })

    def test_convert_streaming_basic(self):
        """ストリーミング変換の基本テスト"""
        converter = TimeframeConverter("5T")
        complete, incomplete = converter.convert_streaming(self.streaming_data)
        
        # 完成バーは4本
        assert complete is not None
        assert len(complete) == 4
        assert 'is_complete' not in complete.columns
        
        # 不完全バーは1本（2分のデータ）
        assert incomplete is not None
        assert len(incomplete) == 1
        assert 'is_complete' not in incomplete.columns

    def test_convert_streaming_with_last_timestamp(self):
        """最後の完成タイムスタンプを指定したストリーミング変換テスト"""
        converter = TimeframeConverter("5T")
        
        # 最初の2本の5分足を既に処理済みとする
        last_complete = datetime(2024, 1, 1, 9, 5, 0)  # 9:05の5分足まで処理済み
        
        complete, incomplete = converter.convert_streaming(
            self.streaming_data,
            last_complete_timestamp=last_complete
        )
        
        # 新しい完成バーは2本（9:10と9:15の5分足）
        assert complete is not None
        assert len(complete) == 2
        assert complete['timestamp'][0] == datetime(2024, 1, 1, 9, 10, 0)

    def test_convert_streaming_no_complete_bars(self):
        """完成バーがない場合のストリーミング変換テスト"""
        # 3分のデータのみ（5分足が完成しない）
        short_data = self.streaming_data[:3]
        
        converter = TimeframeConverter("5T")
        complete, incomplete = converter.convert_streaming(short_data)
        
        # 完成バーはNone
        assert complete is None
        # 不完全バーが1本
        assert incomplete is not None
        assert len(incomplete) == 1


class TestDataValidation:
    """データ検証機能のテスト"""

    def test_validate_empty_dataframe(self):
        """空のデータフレームの検証テスト"""
        converter = TimeframeConverter("5T")
        empty_df = pl.DataFrame()
        
        errors = converter.validate_data(empty_df)
        assert len(errors) > 0
        assert "DataFrame is empty" in errors

    def test_validate_missing_timestamp(self):
        """タイムスタンプカラムが欠けている場合の検証テスト"""
        converter = TimeframeConverter("5T")
        df = pl.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [100]
        })
        
        errors = converter.validate_data(df, timestamp_col='timestamp')
        assert any("Timestamp column" in e for e in errors)

    def test_validate_missing_price_columns(self):
        """価格カラムが欠けている場合の検証テスト"""
        converter = TimeframeConverter("5T")
        df = pl.DataFrame({
            'timestamp': [datetime.now()],
            'volume': [100]
        })
        
        price_cols = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        }
        
        errors = converter.validate_data(df, price_cols=price_cols)
        assert len(errors) >= 4  # 4つの価格カラムが不足

    def test_validate_null_values(self):
        """NULL値を含むデータの検証テスト"""
        converter = TimeframeConverter("5T")
        df = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'open': [100, 101, None, 103, 104],
            'high': [105, 106, 107, None, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [100, 200, 300, 400, 500]
        })
        
        price_cols = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        }
        
        errors = converter.validate_data(df, price_cols=price_cols)
        assert any("null values" in e for e in errors)

    def test_validate_duplicate_timestamps(self):
        """重複タイムスタンプの検証テスト"""
        converter = TimeframeConverter("5T")
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        df = pl.DataFrame({
            'timestamp': [base_time, base_time, base_time + timedelta(minutes=1)],
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [100, 200, 300]
        })
        
        errors = converter.validate_data(df)
        assert any("duplicate timestamps" in e for e in errors)

    def test_validate_valid_data(self):
        """正常なデータの検証テスト"""
        converter = TimeframeConverter("5T")
        df = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [100, 200, 300, 400, 500]
        })
        
        errors = converter.validate_data(df)
        assert len(errors) == 0  # エラーなし


class TestGetInfo:
    """get_infoメソッドのテスト"""

    def test_get_info_basic(self):
        """基本的な情報取得のテスト"""
        converter = TimeframeConverter("5T")
        info = converter.get_info()
        
        assert info['source_timeframe'] == "1T"
        assert info['target_timeframe'] == "5T"
        assert info['target_description'] == "5分足"
        assert info['align_to_boundary'] is True
        assert info['polars_interval'] == "5m"
        assert info['expected_bars_per_period'] == 5

    def test_get_info_all_timeframes(self):
        """すべてのタイムフレームでの情報取得テスト"""
        expected_bars = {
            "5T": 5,
            "15T": 15,
            "30T": 30,
            "1H": 60,
            "4H": 240,
            "1D": 1440
        }
        
        for tf, expected in expected_bars.items():
            converter = TimeframeConverter(tf)
            info = converter.get_info()
            assert info['expected_bars_per_period'] == expected


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_empty_dataframe_error(self):
        """空のデータフレームでのエラーテスト"""
        converter = TimeframeConverter("5T")
        empty_df = pl.DataFrame()
        
        with pytest.raises(InsufficientBarsError) as exc_info:
            converter.convert(empty_df)
        assert "Input dataframe is empty" in str(exc_info.value)

    def test_missing_timestamp_column_error(self):
        """タイムスタンプカラムが存在しない場合のエラーテスト"""
        converter = TimeframeConverter("5T")
        df = pl.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [100]
        })
        
        with pytest.raises(TimeframeConversionError) as exc_info:
            converter.convert(df, timestamp_col='timestamp')
        assert "Timestamp column" in str(exc_info.value)

    def test_missing_price_column_error(self):
        """価格カラムが存在しない場合のエラーテスト"""
        converter = TimeframeConverter("5T")
        df = pl.DataFrame({
            'timestamp': [datetime.now()],
            'volume': [100]
        })
        
        with pytest.raises(TimeframeConversionError) as exc_info:
            converter.convert(df)
        assert "Price column" in str(exc_info.value)

    def test_exception_hierarchy(self):
        """例外の継承関係のテスト"""
        # InvalidTimeframeErrorはTimeframeConversionErrorのサブクラス
        error = InvalidTimeframeError("test")
        assert isinstance(error, TimeframeConversionError)
        assert isinstance(error, Exception)
        
        # InsufficientBarsErrorもTimeframeConversionErrorのサブクラス
        error = InsufficientBarsError("test")
        assert isinstance(error, TimeframeConversionError)
        assert isinstance(error, Exception)


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_single_bar_conversion(self):
        """単一バーの変換テスト"""
        converter = TimeframeConverter("5T")
        df = pl.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 9, 0, 0)],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [100]
        })
        
        # keepモードでは1本の不完全バーが生成される
        result = converter.convert(df, incomplete_bar_handling="keep")
        assert len(result) == 1
        assert result['bar_count'][0] == 1

    def test_exact_boundary_data(self):
        """境界ぴったりのデータの変換テスト"""
        # ちょうど30分のデータ
        timestamps = pd.date_range('2024-01-01 09:00', periods=30, freq='1min')
        df = pl.DataFrame({
            'timestamp': timestamps,
            'open': [100] * 30,
            'high': [105] * 30,
            'low': [95] * 30,
            'close': [102] * 30,
            'volume': [100] * 30
        })
        
        # 5分足変換で6本のバーが生成される
        converter = TimeframeConverter("5T")
        result = converter.convert(df)
        assert len(result) == 6
        assert all(result['bar_count'] == 5)
        
        # 15分足変換で2本のバーが生成される
        converter = TimeframeConverter("15T")
        result = converter.convert(df)
        assert len(result) == 2
        assert all(result['bar_count'] == 15)
        
        # 30分足変換で1本のバーが生成される
        converter = TimeframeConverter("30T")
        result = converter.convert(df)
        assert len(result) == 1
        assert result['bar_count'][0] == 30

    def test_large_dataset(self):
        """大規模データセットの変換テスト"""
        # 1日分のデータ（1440分）
        timestamps = pd.date_range('2024-01-01', periods=1440, freq='1min')
        prices = np.random.randn(1440).cumsum() + 100
        
        df = pl.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.random.rand(1440),
            'low': prices - np.random.rand(1440),
            'close': prices + np.random.randn(1440) * 0.1,
            'volume': np.random.randint(100, 1000, 1440)
        })
        
        # 5分足変換（288本）
        converter = TimeframeConverter("5T")
        result = converter.convert(df)
        assert len(result) == 288
        
        # 1時間足変換（24本）
        converter = TimeframeConverter("1H")
        result = converter.convert(df)
        assert len(result) == 24
        
        # 日足変換（1本）
        converter = TimeframeConverter("1D")
        result = converter.convert(df)
        assert len(result) == 1

    def test_custom_column_names(self):
        """カスタムカラム名での変換テスト"""
        df = pl.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'o': [100] * 10,
            'h': [105] * 10,
            'l': [95] * 10,
            'c': [102] * 10,
            'v': [100] * 10
        })
        
        converter = TimeframeConverter("5T")
        result = converter.convert(
            df,
            timestamp_col='time',
            price_cols={
                'open': 'o',
                'high': 'h',
                'low': 'l',
                'close': 'c'
            },
            volume_col='v'
        )
        
        assert len(result) == 2
        assert 'time' in result.columns
        assert 'open' in result.columns  # 標準名に変換される
        assert 'volume' in result.columns

    def test_additional_columns_aggregation(self):
        """追加カラムの集約テスト"""
        df = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [100] * 10,
            'spread': [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
            'trades': list(range(10))
        })
        
        converter = TimeframeConverter("5T")
        result = converter.convert(
            df,
            additional_cols={
                'spread': 'mean',
                'trades': 'sum'
            }
        )
        
        assert len(result) == 2
        assert 'spread' in result.columns
        assert 'trades' in result.columns
        
        # 平均スプレッド確認
        assert result['spread'][0] == pytest.approx(0.3, rel=1e-5)
        # 合計取引数確認
        assert result['trades'][0] == sum(range(5))


class TestPerformance:
    """パフォーマンステスト"""

    def test_conversion_performance(self):
        """変換パフォーマンステスト"""
        import time
        
        # 10万本の1分足データ
        n_bars = 100000
        timestamps = pd.date_range('2024-01-01', periods=n_bars, freq='1min')
        prices = np.random.randn(n_bars).cumsum() + 100
        
        df = pl.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.random.rand(n_bars),
            'low': prices - np.random.rand(n_bars),
            'close': prices + np.random.randn(n_bars) * 0.1,
            'volume': np.random.randint(100, 1000, n_bars)
        })
        
        converter = TimeframeConverter("5T")
        
        # 変換時間の測定
        start_time = time.time()
        result = converter.convert(df)
        elapsed_time = time.time() - start_time
        
        # 10万本の変換が5秒以内で完了すること
        assert elapsed_time < 5.0
        
        # 結果の妥当性確認
        expected_bars = n_bars // 5
        assert len(result) == expected_bars

    def test_memory_efficiency(self):
        """メモリ効率のテスト"""
        import psutil
        import os
        
        # 現在のメモリ使用量
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大規模データの変換
        n_bars = 50000
        timestamps = pd.date_range('2024-01-01', periods=n_bars, freq='1min')
        df = pl.DataFrame({
            'timestamp': timestamps,
            'open': np.random.rand(n_bars) * 100,
            'high': np.random.rand(n_bars) * 100,
            'low': np.random.rand(n_bars) * 100,
            'close': np.random.rand(n_bars) * 100,
            'volume': np.random.randint(100, 1000, n_bars)
        })
        
        converter = TimeframeConverter("5T")
        result = converter.convert(df)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # メモリ増加が妥当な範囲内（200MB以下）
        assert memory_increase < 200
        
        # 結果の妥当性確認
        assert len(result) == n_bars // 5


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v", "-s"])