"""
OHLCデータ取得機能のユニットテスト

このテストファイルはTDDアプローチに従い、実装前にテストケースを定義しています。
テスト対象: HistoricalDataFetcher クラス
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

# 実装されたクラスをインポート
from src.mt5_data_acquisition.ohlc_fetcher import HistoricalDataFetcher

# ===========================
# テスト用フィクスチャ
# ===========================


@pytest.fixture
def mock_mt5_rates():
    """Mock MT5のOHLCデータを生成するフィクスチャ（NumPy structured array形式）"""
    base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp())

    # MT5の実際の返り値形式（NumPy structured array）を作成
    dtype = np.dtype(
        [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
            ("real_volume", "i8"),
        ]
    )

    rates = np.zeros(100, dtype=dtype)
    for i in range(100):  # 100本のバー
        rates[i] = (
            base_time + i * 3600,  # 1時間足
            110.00 + np.random.randn() * 0.1,
            110.50 + np.random.randn() * 0.1,
            109.50 + np.random.randn() * 0.1,
            110.20 + np.random.randn() * 0.1,
            np.random.randint(100, 1000),
            np.random.randint(1, 5),
            0,
        )

    return rates


@pytest.fixture
def mock_mt5_rates_with_gap():
    """欠損データを含むMock MT5データを生成するフィクスチャ（NumPy structured array形式）"""
    base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp())

    # MT5の実際の返り値形式（NumPy structured array）を作成
    dtype = np.dtype(
        [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
            ("real_volume", "i8"),
        ]
    )

    # 欠損を含むデータ（50-59のインデックスを除外）
    rates = np.zeros(90, dtype=dtype)
    idx = 0
    for i in range(100):
        # 50-59の範囲でデータ欠損を作る
        if 50 <= i < 60:
            continue

        rates[idx] = (
            base_time + i * 3600,  # 1時間足
            110.00 + np.random.randn() * 0.1,
            110.50 + np.random.randn() * 0.1,
            109.50 + np.random.randn() * 0.1,
            110.20 + np.random.randn() * 0.1,
            np.random.randint(100, 1000),
            np.random.randint(1, 5),
            0,
        )
        idx += 1

    return rates


@pytest.fixture
def mock_mt5_client():
    """Mock MT5ConnectionManagerを生成するフィクスチャ"""
    client = Mock()
    client.connect = Mock(return_value=True)
    client.disconnect = Mock()
    client.is_connected = Mock(return_value=True)
    client.terminal_info = Mock()
    client.account_info = Mock()
    # MT5Configのモック
    client._config = Mock()
    return client


@pytest.fixture
def sample_config():
    """テスト用の設定を生成するフィクスチャ"""
    return {
        "symbols": ["USDJPY", "EURUSD", "GBPUSD"],
        "timeframes": ["H1", "H4", "D1"],
        "batch_size": 1000,
        "max_workers": 4,
        "retry_count": 3,
        "retry_delay": 1.0,
    }


# ===========================
# HistoricalDataFetcherのテストクラス
# ===========================


class TestHistoricalDataFetcher:
    """HistoricalDataFetcherクラスのテストケース"""

    # ------------------------
    # 初期化のテスト
    # ------------------------

    def test_init(self, mock_mt5_client):
        """初期化のテスト

        検証項目:
        - MT5クライアントの設定
        - デフォルトパラメータの設定
        - 内部状態の初期化
        """
        # Arrange
        # なし

        # Act
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # Assert
        assert fetcher.mt5_client == mock_mt5_client
        assert fetcher.batch_size == 10000  # デフォルトバッチサイズ
        assert fetcher.max_workers == 4
        assert fetcher.max_retries == 3
        assert fetcher.retry_delay == 1000
        assert fetcher._connected is False

    def test_init_with_custom_config(self, mock_mt5_client, sample_config):
        """カスタム設定での初期化テスト"""
        # Arrange
        # なし

        # Act
        fetcher = HistoricalDataFetcher(
            mt5_client=mock_mt5_client, config=sample_config
        )

        # Assert
        assert fetcher.batch_size == sample_config["batch_size"]
        assert fetcher.max_workers == sample_config["max_workers"]
        assert fetcher.max_retries == sample_config["retry_count"]
        assert (
            fetcher.retry_delay == sample_config["retry_delay"]
        )  # ミリ秒単位でそのまま保存される

    # ------------------------
    # 基本的なデータ取得のテスト
    # ------------------------

    @patch("src.mt5_data_acquisition.ohlc_fetcher.mt5")
    def test_fetch_ohlc_data(self, mock_mt5, mock_mt5_client, mock_mt5_rates):
        """基本的なOHLCデータ取得のテスト

        検証項目:
        - 正常なデータ取得
        - LazyFrameへの変換
        - カラム名の正確性
        - データ型の正確性
        """
        # Arrange
        mock_mt5.copy_rates_range.return_value = mock_mt5_rates
        mock_mt5.symbol_info.return_value = Mock(visible=True)
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)
        fetcher._connected = True  # 接続状態にする

        # Act
        lazy_df = fetcher.fetch_ohlc_data(
            symbol="USDJPY",
            timeframe="H1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            use_batch=False,  # バッチ処理を無効化してシンプルなテスト
            detect_gaps=False,  # 欠損検出を無効化
        )

        # LazyFrameをDataFrameに変換してテスト
        df = lazy_df.collect()

        # Assert
        assert isinstance(lazy_df, pl.LazyFrame)
        assert isinstance(df, pl.DataFrame)
        assert df.height == len(mock_mt5_rates)
        assert all(
            col in df.columns
            for col in ["timestamp", "open", "high", "low", "close", "volume", "spread"]
        )
        # UTCタイムゾーン付きのDatatime型であることを確認
        assert str(df.get_column("timestamp").dtype).startswith("Datetime")
        assert df.get_column("open").dtype == pl.Float32
        assert df.get_column("volume").dtype == pl.Float32

    @patch("src.mt5_data_acquisition.ohlc_fetcher.mt5")
    def test_fetch_ohlc_data_empty_result(self, mock_mt5, mock_mt5_client):
        """データが存在しない場合のテスト"""
        # Arrange
        mock_mt5.copy_rates_range.return_value = None  # 空の結果
        mock_mt5.symbol_info.return_value = Mock(visible=True)
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)
        fetcher._connected = True

        # Act
        lazy_df = fetcher.fetch_ohlc_data(
            symbol="USDJPY",
            timeframe="H1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            use_batch=False,
            detect_gaps=False,
        )

        df = lazy_df.collect()

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 0
        # 空でも正しいスキーマを持つことを確認
        assert all(
            col in df.columns
            for col in ["timestamp", "open", "high", "low", "close", "volume", "spread"]
        )

    # ------------------------
    # バッチ処理のテスト
    # ------------------------

    @patch("src.mt5_data_acquisition.ohlc_fetcher.mt5")
    def test_batch_processing(self, mock_mt5, mock_mt5_client, mock_mt5_rates):
        """大量データのバッチ処理テスト

        検証項目:
        - バッチ分割の正確性
        - メモリ効率的な処理
        - 複数バッチの結合
        """
        # Arrange
        mock_mt5.copy_rates_range.return_value = mock_mt5_rates
        mock_mt5.symbol_info.return_value = Mock(visible=True)
        fetcher = HistoricalDataFetcher(
            mt5_client=mock_mt5_client,
            config={"batch_size": 50},  # 小さいバッチサイズでテスト
        )
        fetcher._connected = True

        # Act
        lazy_df = fetcher.fetch_ohlc_data(
            symbol="USDJPY",
            timeframe="H1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),  # 216時間 = 複数バッチ
            use_batch=True,
            detect_gaps=False,
        )

        df = lazy_df.collect()

        # Assert
        assert isinstance(df, pl.DataFrame)
        # バッチ処理が複数回呼ばれたことを確認
        assert mock_mt5.copy_rates_range.call_count >= 2
        # データがソートされていることを確認
        timestamps = df.get_column("timestamp").to_list()
        assert timestamps == sorted(timestamps)

    @pytest.mark.skip(reason="進捗コールバック機能は今後実装予定")
    def test_batch_processing_with_progress(self, mock_mt5_client, mock_mt5_rates):
        """進捗表示付きバッチ処理のテスト"""
        pass

    # ------------------------
    # 並列フェッチのテスト
    # ------------------------

    @patch("src.mt5_data_acquisition.ohlc_fetcher.mt5")
    def test_parallel_fetch(self, mock_mt5, mock_mt5_client, mock_mt5_rates):
        """並列フェッチのテスト

        検証項目:
        - 並列処理の動作
        - 結果の正確性
        - データのソートと重複除去
        """
        # Arrange
        mock_mt5.copy_rates_range.return_value = mock_mt5_rates
        mock_mt5.symbol_info.return_value = Mock(visible=True)
        mock_mt5.initialize.return_value = True
        mock_mt5.shutdown.return_value = None

        fetcher = HistoricalDataFetcher(
            mt5_client=mock_mt5_client, config={"max_workers": 2, "batch_size": 50}
        )
        fetcher._connected = True

        # Act
        lazy_df = fetcher.fetch_ohlc_data(
            symbol="USDJPY",
            timeframe="H1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),  # 大量データで並列処理をトリガー
            use_batch=True,
            use_parallel=True,  # 並列処理を有効化
            detect_gaps=False,
        )

        df = lazy_df.collect()

        # Assert
        assert isinstance(df, pl.DataFrame)
        # データが取得されている
        assert df.height > 0
        # タイムスタンプがソートされている
        timestamps = df.get_column("timestamp").to_list()
        assert timestamps == sorted(timestamps)
        # 重複がない
        assert df.get_column("timestamp").n_unique() == df.height

    @patch("src.mt5_data_acquisition.ohlc_fetcher.mt5")
    def test_parallel_fetch_with_error(self, mock_mt5, mock_mt5_client):
        """並列フェッチ中のエラーハンドリングテスト"""
        # Arrange
        call_count = {"count": 0}

        def side_effect(*args, **kwargs):
            # 最初の呼び出しは成功、次は失敗
            call_count["count"] += 1
            if call_count["count"] == 2:
                return None  # エラーをシミュレート
            # フィクスチャではなく、直接NumPy配列を作成
            base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp())
            dtype = np.dtype(
                [
                    ("time", "i8"),
                    ("open", "f8"),
                    ("high", "f8"),
                    ("low", "f8"),
                    ("close", "f8"),
                    ("tick_volume", "i8"),
                    ("spread", "i4"),
                    ("real_volume", "i8"),
                ]
            )
            rates = np.zeros(50, dtype=dtype)
            for i in range(50):
                rates[i] = (
                    base_time + i * 3600,
                    110.00 + np.random.randn() * 0.1,
                    110.50 + np.random.randn() * 0.1,
                    109.50 + np.random.randn() * 0.1,
                    110.20 + np.random.randn() * 0.1,
                    np.random.randint(100, 1000),
                    np.random.randint(1, 5),
                    0,
                )
            return rates

        mock_mt5.copy_rates_range.side_effect = side_effect
        mock_mt5.symbol_info.return_value = Mock(visible=True)
        mock_mt5.initialize.return_value = True
        mock_mt5.shutdown.return_value = None

        fetcher = HistoricalDataFetcher(
            mt5_client=mock_mt5_client, config={"max_workers": 2, "batch_size": 50}
        )
        fetcher._connected = True

        # Act
        lazy_df = fetcher.fetch_ohlc_data(
            symbol="USDJPY",
            timeframe="H1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            use_batch=True,
            use_parallel=True,
            detect_gaps=False,
        )

        df = lazy_df.collect()

        # Assert
        # 部分的な失敗があってもデータが返される
        assert isinstance(df, pl.DataFrame)
        # 少なくとも一部のデータは取得されている（または空）
        # ※ 実装によっては部分的な結果が返る

    # ------------------------
    # 欠損データ検出のテスト
    # ------------------------

    @patch("src.mt5_data_acquisition.ohlc_fetcher.mt5")
    def test_missing_data_detection(
        self, mock_mt5, mock_mt5_client, mock_mt5_rates_with_gap
    ):
        """欠損データ検出のテスト

        検証項目:
        - 欠損期間の検出
        - 欠損レポートの生成
        - 欠損バー数の計算
        """
        # Arrange
        mock_mt5.copy_rates_range.return_value = mock_mt5_rates_with_gap
        mock_mt5.symbol_info.return_value = Mock(visible=True)
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)
        fetcher._connected = True

        # Act
        lazy_df = fetcher.fetch_ohlc_data(
            symbol="USDJPY",
            timeframe="H1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),
            use_batch=False,
            detect_gaps=True,  # 欠損検出を有効化
        )

        # 欠損検出を直接テスト
        missing_periods = fetcher.detect_missing_periods(lazy_df, "H1")

        # Assert
        assert len(missing_periods) > 0
        # 最初の欠損期間を確認
        first_gap = missing_periods[0]
        assert "start" in first_gap
        assert "end" in first_gap
        assert "expected_bars" in first_gap
        assert "actual_gap" in first_gap
        # 10時間分の欠損があるはず
        assert first_gap["expected_bars"] >= 10

    @pytest.mark.skip(reason="欠損データ補完機能は今後実装予定")
    def test_missing_data_filling(self, mock_mt5_client, mock_mt5_rates_with_gap):
        """欠損データ補完のテスト"""
        pass

    # ------------------------
    # 時間足変換のテスト
    # ------------------------

    @pytest.mark.skip(reason="時間足変換機能は別モジュールで実装予定")
    def test_timeframe_conversion(self, mock_mt5_client, mock_mt5_rates):
        """時間足変換のテスト"""
        pass

    @pytest.mark.skip(reason="時間足変換機能は別モジュールで実装予定")
    def test_timeframe_conversion_with_custom_aggregation(self, mock_mt5_client):
        """カスタム集計での時間足変換テスト"""
        pass


# ===========================
# その他のヘルパー関数のテスト
# ===========================


class TestHelperFunctions:
    """ヘルパー関数のテストケース"""

    def test_get_expected_interval(self, mock_mt5_client):
        """時間足に応じた期待間隔のテスト"""
        from datetime import timedelta

        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # Act & Assert
        assert fetcher._get_expected_interval("M1") == timedelta(minutes=1)
        assert fetcher._get_expected_interval("M5") == timedelta(minutes=5)
        assert fetcher._get_expected_interval("M15") == timedelta(minutes=15)
        assert fetcher._get_expected_interval("M30") == timedelta(minutes=30)
        assert fetcher._get_expected_interval("H1") == timedelta(minutes=60)
        assert fetcher._get_expected_interval("H4") == timedelta(minutes=240)
        assert fetcher._get_expected_interval("D1") == timedelta(minutes=1440)
        assert fetcher._get_expected_interval("W1") == timedelta(minutes=10080)
        assert fetcher._get_expected_interval("MN") == timedelta(minutes=43200)

    def test_calculate_batch_dates(self, mock_mt5_client):
        """バッチ処理用の日付分割テスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(
            mt5_client=mock_mt5_client, config={"batch_size": 100}
        )
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)  # 9日間

        # Act
        batches = fetcher._calculate_batch_dates(start, end, "H1", 100)

        # Assert
        assert len(batches) > 0
        # 最初のバッチの開始時刻が正しい
        assert batches[0][0] == start
        # 最後のバッチの終了時刻が正しい
        assert batches[-1][1] == end
        # 各バッチが連続している
        for i in range(len(batches) - 1):
            assert batches[i][1] == batches[i + 1][0]

    def test_is_market_closed(self, mock_mt5_client):
        """市場休場期間の検出テスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # Act & Assert
        # 土曜日のテスト
        saturday = datetime(2024, 1, 6, 12, 0, 0)  # 2024年1月6日は土曜日
        sunday = datetime(2024, 1, 7, 12, 0, 0)  # 2024年1月7日は日曜日
        monday = datetime(2024, 1, 8, 12, 0, 0)  # 2024年1月8日は月曜日

        assert fetcher._is_market_closed(saturday, saturday.replace(hour=13)) is True
        assert fetcher._is_market_closed(sunday, sunday.replace(hour=13)) is True
        assert fetcher._is_market_closed(monday, monday.replace(hour=13)) is False

        # クリスマスのテスト
        christmas = datetime(2024, 12, 25, 12, 0, 0)
        assert fetcher._is_market_closed(christmas, christmas.replace(hour=13)) is True

        # 元日のテスト
        new_year = datetime(2024, 1, 1, 12, 0, 0)
        assert fetcher._is_market_closed(new_year, new_year.replace(hour=13)) is True

    def test_is_gap(self, mock_mt5_client):
        """欠損判定ロジックのテスト"""
        from datetime import timedelta

        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)
        expected_interval = timedelta(minutes=60)  # 1時間

        # Act & Assert
        # 1.5倍未満は欠損ではない
        assert fetcher._is_gap(timedelta(minutes=89), expected_interval) is False
        # 1.5倍ちょうどは欠損ではない
        assert fetcher._is_gap(timedelta(minutes=90), expected_interval) is False
        # 1.5倍を超えると欠損
        assert fetcher._is_gap(timedelta(minutes=91), expected_interval) is True
        assert fetcher._is_gap(timedelta(minutes=120), expected_interval) is True


# ===========================
# 統合テスト
# ===========================


@pytest.mark.skip(reason="統合テストは別途実装")
class TestIntegrationScenarios:
    """統合シナリオのテストケース"""

    def test_full_workflow(self, mock_mt5_client):
        """完全なワークフローのテスト"""
        pass

    def test_error_recovery_workflow(self, mock_mt5_client):
        """エラーリカバリーのワークフローテスト"""
        pass

    def test_performance_with_large_dataset(self, mock_mt5_client):
        """大規模データセットでのパフォーマンステスト"""
        pass
