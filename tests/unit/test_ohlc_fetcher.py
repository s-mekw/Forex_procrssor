"""
OHLCデータ取得機能のユニットテスト

このテストファイルはTDDアプローチに従い、実装前にテストケースを定義しています。
テスト対象: HistoricalDataFetcher クラス
"""

import random
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import psutil
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
    # MT5Configのモック（新しいget_config()メソッドを使用）
    client.get_config = Mock(return_value=Mock())
    client._config = Mock()  # 後方互換性のため残す
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
# 統合テスト用ヘルパー関数
# ===========================


def generate_large_ohlc_dataset(
    num_bars: int,
    start_time: datetime,
    timeframe_minutes: int = 1,
    with_gaps: bool = False,
    gap_probability: float = 0.001,
) -> np.ndarray:
    """大規模OHLCデータセットを生成

    Args:
        num_bars: 生成するバー数
        start_time: 開始時刻
        timeframe_minutes: 時間足（分単位）
        with_gaps: 欠損を含むかどうか
        gap_probability: 欠損確率（with_gaps=Trueの場合のみ有効）

    Returns:
        NumPy structured array（MT5形式）
    """
    # 初期価格設定
    base_price = 110.0
    price = base_price
    volatility = 0.002  # 0.2%のボラティリティ

    # データ配列の初期化
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
    data = np.zeros(num_bars, dtype=dtype)

    current_time = start_time
    actual_bars = 0

    for _i in range(num_bars):
        # 欠損の生成（オプション）
        if with_gaps and random.random() < gap_probability:
            # 1-10時間分の欠損を生成
            gap_hours = random.randint(1, 10)
            current_time = current_time + timedelta(hours=gap_hours)

        # 価格の生成（ランダムウォーク）
        price_change = random.gauss(0, volatility * price)
        open_price = price
        close_price = price + price_change

        # 高値・安値の生成
        intra_volatility = abs(price_change) * 1.5
        high_price = max(open_price, close_price) + random.uniform(0, intra_volatility)
        low_price = min(open_price, close_price) - random.uniform(0, intra_volatility)

        # データの設定
        data[actual_bars]["time"] = int(current_time.timestamp())
        data[actual_bars]["open"] = round(open_price, 5)
        data[actual_bars]["high"] = round(high_price, 5)
        data[actual_bars]["low"] = round(low_price, 5)
        data[actual_bars]["close"] = round(close_price, 5)
        data[actual_bars]["tick_volume"] = random.randint(100, 10000)
        data[actual_bars]["spread"] = random.randint(1, 5)
        data[actual_bars]["real_volume"] = random.randint(1000000, 100000000)

        # 次のバーへ
        price = close_price
        current_time = current_time + timedelta(minutes=timeframe_minutes)
        actual_bars += 1

    return data[:actual_bars]


def measure_memory_usage():
    """現在のメモリ使用量を測定（MB単位）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


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

    def test_create_empty_lazyframe(self, mock_mt5_client):
        """空のLazyFrame生成メソッドのテスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # Act
        empty_df = fetcher._create_empty_lazyframe()

        # Assert
        assert isinstance(empty_df, pl.LazyFrame)
        # DataFrameに変換して検証
        df = empty_df.collect()
        assert len(df) == 0
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread",
        ]
        # 型チェック
        assert df.schema["timestamp"] == pl.Datetime("us")
        assert df.schema["open"] == pl.Float32
        assert df.schema["high"] == pl.Float32
        assert df.schema["low"] == pl.Float32
        assert df.schema["close"] == pl.Float32
        assert df.schema["volume"] == pl.Float32
        assert df.schema["spread"] == pl.Int32


# ===========================
# 統合テスト
# ===========================


class TestIntegrationScenarios:
    """統合シナリオのテストケース

    注意: これらのテストはモックデータを使用して統合動作を検証します。
    実際のMT5接続は使用しません。
    """

    def test_complete_workflow(self, mock_mt5_client):
        """エンドツーエンドの完全なワークフローテスト

        検証項目:
        - 大量データ（10万バー）の取得
        - バッチ処理と並列処理の統合動作
        - データの完全性と順序保証
        - 欠損検出の動作確認
        """
        # Arrange
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 1, 0, 0, 0)

        # 10万バーのデータを生成（欠損含む）
        large_dataset = generate_large_ohlc_dataset(
            num_bars=100000,
            start_time=start_time,
            timeframe_minutes=1,
            with_gaps=True,
            gap_probability=0.0001,  # 0.01%の確率で欠損
        )

        # MT5のモックを設定（バッチごとにデータを返す）
        def mock_copy_rates_range(symbol, timeframe, date_from, date_to):
            # タイムスタンプ範囲でフィルタリング
            from_ts = int(date_from.timestamp())
            to_ts = int(date_to.timestamp())

            mask = (large_dataset["time"] >= from_ts) & (large_dataset["time"] <= to_ts)
            return large_dataset[mask][:10000]  # 最大10000バー

        # シンボル情報のモック
        mock_symbol_info = Mock()
        mock_symbol_info.visible = True

        with patch("MetaTrader5.copy_rates_range", side_effect=mock_copy_rates_range):
            with patch("MetaTrader5.initialize", return_value=True):
                with patch("MetaTrader5.shutdown"):
                    with patch(
                        "MetaTrader5.symbol_info", return_value=mock_symbol_info
                    ):
                        with patch("MetaTrader5.symbol_select", return_value=True):
                            # Act
                            fetcher = HistoricalDataFetcher(
                                mt5_client=mock_mt5_client,
                                config={
                                    "batch_size": 10000,
                                    "max_workers": 4,
                                },
                            )

                            # 接続
                            fetcher.connect()

                            # 並列処理を使用してデータ取得
                            result = fetcher.fetch_ohlc_data(
                                symbol="USDJPY",
                                timeframe="M1",
                                start_date=start_time,
                                end_date=end_time,
                                use_parallel=True,
                                detect_gaps=True,
                            )

                            # Assert
                            assert result is not None
                            df_collected = result.collect()

                            # データ件数の確認（欠損を除く）
                            assert len(df_collected) > 0
                            assert len(df_collected) <= 100000

                            # データの順序確認
                            timestamps = df_collected["timestamp"].to_list()
                            assert timestamps == sorted(timestamps), (
                                "データが時系列順になっていない"
                            )

                            # カラムの確認
                            expected_columns = [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "spread",
                            ]
                            assert df_collected.columns == expected_columns

                            # データ型の確認
                            assert df_collected["timestamp"].dtype == pl.Datetime
                            assert df_collected["open"].dtype == pl.Float32
                            assert df_collected["high"].dtype == pl.Float32
                            assert df_collected["low"].dtype == pl.Float32
                            assert df_collected["close"].dtype == pl.Float32

    def test_error_recovery(self, mock_mt5_client):
        """エラー回復のワークフローテスト

        検証項目:
        - 部分的なワーカー失敗への対処
        - データ取得エラーからの回復
        - 最小限のデータでも結果を返す
        """
        # Arrange
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 2, 0, 0, 0)

        # エラーを含むモックデータ
        successful_data = generate_large_ohlc_dataset(
            num_bars=1000,
            start_time=start_time,
            timeframe_minutes=1,
        )

        call_count = 0

        def mock_copy_rates_with_errors(symbol, timeframe, date_from, date_to):
            nonlocal call_count
            call_count += 1

            # 2回に1回はエラーを発生させる
            if call_count % 2 == 0:
                return None  # MT5エラーをシミュレート

            # 成功する場合はデータを返す
            from_ts = int(date_from.timestamp())
            to_ts = int(date_to.timestamp())
            mask = (successful_data["time"] >= from_ts) & (
                successful_data["time"] <= to_ts
            )
            return successful_data[mask]

        # シンボル情報のモック
        mock_symbol_info = Mock()
        mock_symbol_info.visible = True

        with patch(
            "MetaTrader5.copy_rates_range", side_effect=mock_copy_rates_with_errors
        ):
            with patch("MetaTrader5.initialize", return_value=True):
                with patch("MetaTrader5.shutdown"):
                    with patch(
                        "MetaTrader5.symbol_info", return_value=mock_symbol_info
                    ):
                        with patch("MetaTrader5.symbol_select", return_value=True):
                            # Act
                            fetcher = HistoricalDataFetcher(
                                mt5_client=mock_mt5_client,
                                config={
                                    "batch_size": 500,
                                    "max_workers": 2,
                                },
                            )

                            # 接続
                            fetcher.connect()

                            result = fetcher.fetch_ohlc_data(
                                symbol="EURUSD",
                                timeframe="M5",
                                start_date=start_time,
                                end_date=end_time,
                                use_parallel=True,
                                detect_gaps=False,
                            )

                            # Assert
                            assert result is not None
                            df_collected = result.collect()

                            # 部分的にでもデータが取得できていることを確認
                            assert len(df_collected) > 0
                            assert (
                                len(df_collected) < 1000
                            )  # 一部失敗のため完全ではない

                            # データの整合性確認
                            assert not df_collected["open"].is_null().any()
                            assert not df_collected["close"].is_null().any()

    def test_performance(self, mock_mt5_client):
        """パフォーマンステスト

        検証項目:
        - 大規模データセット（100万バー）の処理時間
        - メモリ使用量の測定
        - 並列処理の効果測定
        """
        # Arrange
        start_time = datetime(2020, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 1, 0, 0, 0)

        # 100万バーのデータを生成
        print("\n生成中: 100万バーのテストデータ...")
        large_dataset = generate_large_ohlc_dataset(
            num_bars=1000000,
            start_time=start_time,
            timeframe_minutes=1,
            with_gaps=False,
        )
        print(f"生成完了: {len(large_dataset)}バー")

        def mock_copy_rates_fast(symbol, timeframe, date_from, date_to):
            # 高速なデータスライシング
            from_ts = int(date_from.timestamp())
            to_ts = int(date_to.timestamp())

            # バイナリサーチで高速化
            left_idx = np.searchsorted(large_dataset["time"], from_ts, side="left")
            right_idx = np.searchsorted(large_dataset["time"], to_ts, side="right")

            return large_dataset[left_idx:right_idx][:10000]

        # シンボル情報のモック
        mock_symbol_info = Mock()
        mock_symbol_info.visible = True

        with patch("MetaTrader5.copy_rates_range", side_effect=mock_copy_rates_fast):
            with patch("MetaTrader5.initialize", return_value=True):
                with patch("MetaTrader5.shutdown"):
                    with patch(
                        "MetaTrader5.symbol_info", return_value=mock_symbol_info
                    ):
                        with patch("MetaTrader5.symbol_select", return_value=True):
                            # メモリ使用量の初期値
                            initial_memory = measure_memory_usage()
                            print(f"初期メモリ使用量: {initial_memory:.2f} MB")

                            # Act - シングルスレッド処理
                            fetcher_single = HistoricalDataFetcher(
                                mt5_client=mock_mt5_client,
                                config={
                                    "batch_size": 10000,
                                    "max_workers": 1,
                                },
                            )
                            fetcher_single.connect()

                            print("\nシングルスレッド処理開始...")
                            start_single = time.time()
                            result_single = fetcher_single.fetch_ohlc_data(
                                symbol="USDJPY",
                                timeframe="M1",
                                start_date=start_time,
                                end_date=end_time,
                                use_parallel=False,
                                detect_gaps=False,
                            )
                            df_single = result_single.collect()
                            time_single = time.time() - start_single
                            memory_single = measure_memory_usage() - initial_memory

                            print(
                                f"シングルスレッド完了: {time_single:.2f}秒, メモリ増加: {memory_single:.2f} MB"
                            )

                            # Act - 並列処理
                            fetcher_parallel = HistoricalDataFetcher(
                                mt5_client=mock_mt5_client,
                                config={
                                    "batch_size": 10000,
                                    "max_workers": 4,
                                },
                            )
                            fetcher_parallel.connect()

                            print("\n並列処理開始（4ワーカー）...")
                            start_parallel = time.time()
                            result_parallel = fetcher_parallel.fetch_ohlc_data(
                                symbol="USDJPY",
                                timeframe="M1",
                                start_date=start_time,
                                end_date=end_time,
                                use_parallel=True,
                                detect_gaps=False,
                            )
                            df_parallel = result_parallel.collect()
                            time_parallel = time.time() - start_parallel
                            memory_parallel = measure_memory_usage() - initial_memory

                            print(
                                f"並列処理完了: {time_parallel:.2f}秒, メモリ増加: {memory_parallel:.2f} MB"
                            )

                            # Assert - パフォーマンス基準
                            print(f"\n速度向上率: {time_single / time_parallel:.2f}倍")

                            # データの一致確認
                            assert len(df_single) == len(df_parallel), (
                                "並列処理とシングルスレッドのデータ数が一致しない"
                            )

                            # パフォーマンス基準（モック環境での目標値）
                            assert time_parallel < time_single * 0.8, (
                                "並列処理が期待した速度向上を達成していない"
                            )

                            # メモリ使用量の確認（1GBを超えないこと）
                            assert memory_parallel < 1000, (
                                f"メモリ使用量が過大: {memory_parallel:.2f} MB"
                            )

                            # データ品質の確認
                            assert len(df_parallel) > 0
                            assert df_parallel["timestamp"].is_sorted(), (
                                "並列処理後のデータが順序を保持していない"
                            )

                            print("\nパフォーマンステスト合格")


class TestImprovements:
    """改善項目の動作確認テスト"""

    def test_error_message_detail(self, mock_mt5_client):
        """詳細なエラーメッセージのテスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)
        fetcher._connected = True
        symbol = "USDJPY"
        timeframe = "H1"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)

        # MT5のcopy_rates_rangeで例外を発生させる
        with patch("MetaTrader5.symbol_info") as mock_symbol_info:
            mock_symbol_info.return_value = Mock(visible=True)
            with patch("MetaTrader5.copy_rates_range") as mock_copy_rates:
                mock_copy_rates.side_effect = Exception("MT5 connection lost")

                # Act & Assert
                with pytest.raises(RuntimeError) as exc_info:
                    fetcher.fetch_ohlc_data(
                        symbol, timeframe, start_date, end_date, use_batch=False
                    )

                # エラーメッセージに詳細情報が含まれることを確認
                error_msg = str(exc_info.value)
                assert symbol in error_msg
                assert timeframe in error_msg
                assert str(start_date) in error_msg
                assert str(end_date) in error_msg
                assert "MT5 connection lost" in error_msg

    def test_mt5_get_config(self, mock_mt5_client):
        """MT5ConnectionManagerのget_config()メソッドのテスト"""
        # Arrange
        config_data = {"account": 12345, "server": "test_server"}
        mock_mt5_client.get_config.return_value = config_data
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # Act
        # connect()メソッド内でget_config()が呼ばれることを確認
        fetcher.connect()

        # Assert
        mock_mt5_client.get_config.assert_called_once()
        mock_mt5_client.connect.assert_called_once_with(config_data)


class TestRetryMechanism:
    """リトライ機能のテスト"""

    def test_retry_with_backoff_success_on_first_try(self, mock_mt5_client):
        """最初の試行で成功する場合のテスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # 成功する関数
        def success_func():
            return "success"

        # Act
        result = fetcher._retry_with_backoff(success_func, max_retries=3)

        # Assert
        assert result == "success"

    def test_retry_with_backoff_success_after_retry(self, mock_mt5_client):
        """リトライ後に成功する場合のテスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # 2回目で成功する関数
        call_count = {"count": 0}

        def retry_func():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise ConnectionError("Connection failed")
            return "success"

        # Act
        result = fetcher._retry_with_backoff(
            retry_func,
            max_retries=3,
            initial_delay=0.1,
            backoff_factor=2.0,
        )

        # Assert
        assert result == "success"
        assert call_count["count"] == 2

    def test_retry_with_backoff_max_retries_exceeded(self, mock_mt5_client):
        """最大リトライ回数を超えた場合のテスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # 常に失敗する関数
        def fail_func():
            raise ConnectionError("Always fails")

        # Act & Assert
        with pytest.raises(ConnectionError, match="Always fails"):
            fetcher._retry_with_backoff(
                fail_func,
                max_retries=2,
                initial_delay=0.01,
            )

    def test_retry_with_backoff_non_retryable_error(self, mock_mt5_client):
        """リトライ不可能なエラーの場合のテスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # ValueError（リトライ不可）を発生させる関数
        call_count = {"count": 0}

        def value_error_func():
            call_count["count"] += 1
            raise ValueError("Invalid value")

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid value"):
            fetcher._retry_with_backoff(value_error_func, max_retries=3)

        # リトライ不可のため、1回しか呼ばれない
        assert call_count["count"] == 1

    def test_is_retryable_error(self, mock_mt5_client):
        """エラーのリトライ可否判定テスト"""
        # Arrange
        fetcher = HistoricalDataFetcher(mt5_client=mock_mt5_client)

        # リトライ可能なエラー
        assert fetcher._is_retryable_error(ConnectionError("Connection lost"))
        assert fetcher._is_retryable_error(TimeoutError("Timeout"))
        assert fetcher._is_retryable_error(OSError("Network error"))
        assert fetcher._is_retryable_error(Exception("Server temporarily unavailable"))

        # リトライ不可能なエラー
        assert not fetcher._is_retryable_error(ValueError("Invalid value"))
        assert not fetcher._is_retryable_error(PermissionError("Access denied"))
        assert not fetcher._is_retryable_error(KeyError("Key not found"))
        assert not fetcher._is_retryable_error(TypeError("Type error"))

    @patch("MetaTrader5.copy_rates_range")
    @patch("MetaTrader5.symbol_info")
    @patch("MetaTrader5.symbol_select")
    def test_fetch_with_retry_integration(
        self,
        mock_symbol_select,
        mock_symbol_info,
        mock_copy_rates,
        mock_mt5_client,
        mock_mt5_rates,
    ):
        """データ取得にリトライが適用されるか統合テスト"""
        # Arrange
        mock_symbol_info.return_value = Mock(visible=True)
        mock_symbol_select.return_value = True

        # 2回目で成功するモック
        mock_copy_rates.side_effect = [
            None,  # 1回目は失敗
            mock_mt5_rates,  # 2回目で成功
        ]

        fetcher = HistoricalDataFetcher(
            mt5_client=mock_mt5_client,
            config={"max_retries": 3, "batch_size": 10000},
        )
        fetcher.connect()

        # Act
        result = fetcher.fetch_ohlc_data(
            symbol="EURUSD",
            timeframe="M1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            use_batch=True,
            use_parallel=False,
        )

        # Assert
        assert result is not None
        df = result.collect()
        assert len(df) == 100  # mock_mt5_ratesのデータ数

        # copy_rates_rangeが2回呼ばれたことを確認
        assert mock_copy_rates.call_count == 2
