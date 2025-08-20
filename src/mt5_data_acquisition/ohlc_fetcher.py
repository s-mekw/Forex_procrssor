"""
履歴OHLCデータ取得モジュール

MT5から履歴OHLCデータを効率的に取得するためのHistoricalDataFetcherクラスを提供。
10,000バー単位のバッチ処理と並列フェッチ機能により、大量の履歴データを高速に取得可能。
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from typing import Any

import MetaTrader5 as mt5
import polars as pl

from src.common.config import ConfigManager
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager

# ロガー設定
logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """MT5から履歴OHLCデータを取得するクラス

    大量の履歴データを効率的に取得するため、以下の機能を提供：
    - 10,000バー単位のバッチ処理
    - ThreadPoolExecutorによる並列データ取得
    - 複数時間足（M1〜MN）のサポート
    - データ欠損の自動検出とログ記録
    - Polars LazyFrameによるメモリ効率的な処理

    Attributes:
        DEFAULT_BATCH_SIZE: デフォルトのバッチサイズ（10,000バー）
        DEFAULT_MAX_WORKERS: デフォルトの並列ワーカー数（4）
        DEFAULT_MAX_RETRIES: デフォルトの最大リトライ回数（3）
        DEFAULT_RETRY_DELAY: デフォルトのリトライ間隔（ミリ秒）
    """

    DEFAULT_BATCH_SIZE = 10000
    DEFAULT_MAX_WORKERS = 4
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1000

    # MT5時間足マッピング
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN": mt5.TIMEFRAME_MN1,
    }

    def __init__(
        self,
        mt5_client: MT5ConnectionManager | None = None,
        config: dict | None = None,
    ):
        """初期化メソッド

        Args:
            mt5_client: MT5接続マネージャーのインスタンス（オプション）
            config: 設定辞書（オプション）
                - batch_size: バッチサイズ（デフォルト: 10000）
                - max_workers: 並列ワーカー数（デフォルト: 4）
                - max_retries: 最大リトライ回数（デフォルト: 3）
                - retry_delay: リトライ間隔（ミリ秒、デフォルト: 1000）
        """
        # MT5クライアントの設定
        if mt5_client is None:
            config_manager = ConfigManager()
            app_config = config_manager.get_config()
            self.mt5_client = MT5ConnectionManager(app_config.mt5)
        else:
            self.mt5_client = mt5_client

        # 設定の読み込み
        self.config = config or {}
        self.batch_size = self.config.get("batch_size", self.DEFAULT_BATCH_SIZE)
        self.max_workers = self.config.get("max_workers", self.DEFAULT_MAX_WORKERS)
        self.max_retries = self.config.get("max_retries", self.DEFAULT_MAX_RETRIES)
        self.retry_delay = self.config.get("retry_delay", self.DEFAULT_RETRY_DELAY)

        # 内部状態
        self._connected = False
        self._terminal_info = None
        self._account_info = None

        logger.info(
            f"HistoricalDataFetcher initialized with batch_size={self.batch_size}, "
            f"max_workers={self.max_workers}"
        )

    def connect(self) -> bool:
        """MT5に接続（リトライ機能付き）

        Returns:
            bool: 接続成功の場合True、失敗の場合False
        """
        try:
            if self._connected:
                logger.debug("Already connected to MT5")
                return True

            # リトライロジックを適用してMT5接続を試行
            def _connect_internal():
                if not self.mt5_client.connect(self.mt5_client._config):
                    raise ConnectionError("Failed to connect to MT5")
                return True

            # リトライ付きで接続を試行
            result = self._retry_with_backoff(
                _connect_internal,
                max_retries=self.max_retries,
                initial_delay=1.0,
                backoff_factor=2.0,
            )

            if result:
                self._connected = True
                self._terminal_info = self.mt5_client.terminal_info
                self._account_info = self.mt5_client.account_info
                logger.info("Successfully connected to MT5")
                return True

            return False

        except Exception as e:
            logger.error(f"Error connecting to MT5 after retries: {e}")
            return False

    def disconnect(self) -> None:
        """MT5から切断"""
        if self._connected:
            self.mt5_client.disconnect()
            self._connected = False
            self._terminal_info = None
            self._account_info = None
            logger.info("Disconnected from MT5")

    def is_connected(self) -> bool:
        """接続状態を確認

        Returns:
            bool: 接続中の場合True、切断中の場合False
        """
        return self._connected and self.mt5_client.is_connected()

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了ポイント"""
        self.disconnect()

    def __del__(self):
        """デストラクタ"""
        if hasattr(self, "_connected") and self._connected:
            self.disconnect()

    def fetch_ohlc_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_batch: bool = True,
        use_parallel: bool = False,
        detect_gaps: bool = True,
    ) -> pl.LazyFrame:
        """MT5から指定期間のOHLCデータを取得

        Args:
            symbol: 通貨ペア（例: "EURUSD"）
            timeframe: 時間足（"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"）
            start_date: 開始日時（UTC）
            end_date: 終了日時（UTC）
            use_batch: バッチ処理を使用するか（デフォルト: True）
            use_parallel: 並列処理を使用するか（デフォルト: False）
            detect_gaps: 欠損期間を検出するか（デフォルト: True）

        Returns:
            pl.LazyFrame: OHLCデータを含むPolars LazyFrame
                - timestamp: タイムスタンプ（UTC）
                - open: 始値
                - high: 高値
                - low: 安値
                - close: 終値
                - volume: 出来高
                - spread: スプレッド

        Raises:
            ValueError: 無効なシンボルまたは時間足が指定された場合
            ConnectionError: MT5に接続していない場合
            RuntimeError: データ取得に失敗した場合
        """
        # 接続確認
        if not self.is_connected():
            raise ConnectionError("Not connected to MT5. Call connect() first.")

        # 時間足の検証と変換
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid options: {list(self.TIMEFRAME_MAP.keys())}"
            )
        mt5_timeframe = self.TIMEFRAME_MAP[timeframe]

        # シンボル情報の取得
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found in MT5")

        # シンボルが取引可能か確認
        if not symbol_info.visible:
            # シンボルを表示
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Failed to select symbol {symbol}")

        # 日時をUTCタイムゾーンに変換（MT5はUTCを使用）
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        logger.info(
            f"Fetching OHLC data for {symbol} {timeframe} "
            f"from {start_date} to {end_date}"
        )

        # データ量を推定してバッチ処理が必要か判定
        total_minutes = (end_date - start_date).total_seconds() / 60
        timeframe_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN": 43200,
        }

        estimated_bars = total_minutes / timeframe_minutes.get(timeframe, 1)

        # 並列処理の判定（大量データかつ並列処理が有効な場合）
        if use_parallel and estimated_bars > self.batch_size * 2:
            logger.info(
                f"Using parallel processing (estimated {estimated_bars:.0f} bars)"
            )

            # 並列処理でデータ取得
            result_df = self._fetch_parallel(
                symbol, mt5_timeframe, start_date, end_date, timeframe
            )

            # 欠損検出を実行（有効な場合）
            if detect_gaps:
                missing_periods = self.detect_missing_periods(result_df, timeframe)
                if missing_periods:
                    logger.warning(
                        f"Found {len(missing_periods)} missing periods in the data"
                    )

            return result_df

        # バッチ処理の判定（10,000バー以上の場合、または明示的に指定された場合）
        elif use_batch and (estimated_bars > self.batch_size or self.batch_size > 0):
            logger.info(f"Using batch processing (estimated {estimated_bars:.0f} bars)")

            # バッチ処理でデータ取得
            batch_results = self._fetch_in_batches(
                symbol, mt5_timeframe, start_date, end_date, timeframe
            )

            if not batch_results:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                # 空のLazyFrameを返す
                return pl.LazyFrame(
                    {
                        "timestamp": [],
                        "open": [],
                        "high": [],
                        "low": [],
                        "close": [],
                        "volume": [],
                        "spread": [],
                    }
                ).cast(
                    {
                        "timestamp": pl.Datetime("us"),
                        "open": pl.Float32,
                        "high": pl.Float32,
                        "low": pl.Float32,
                        "close": pl.Float32,
                        "volume": pl.Float32,
                        "spread": pl.Int32,
                    }
                )

            # バッチ結果を結合（LazyFrameのまま処理）
            combined_df = pl.concat(batch_results, how="vertical")

            # タイムスタンプでソートして重複を除外
            result_df = combined_df.unique("timestamp").sort("timestamp")

            # 欠損検出を実行（有効な場合）
            if detect_gaps:
                missing_periods = self.detect_missing_periods(result_df, timeframe)
                if missing_periods:
                    logger.warning(
                        f"Found {len(missing_periods)} missing periods in the data"
                    )

            return result_df

        else:
            # 通常の単一リクエスト処理
            try:
                # MT5からOHLCデータを取得
                rates = mt5.copy_rates_range(
                    symbol, mt5_timeframe, start_date, end_date
                )

                if rates is None or len(rates) == 0:
                    logger.warning(f"No data returned for {symbol} {timeframe}")
                    # 空のLazyFrameを返す
                    return pl.LazyFrame(
                        {
                            "timestamp": [],
                            "open": [],
                            "high": [],
                            "low": [],
                            "close": [],
                            "volume": [],
                            "spread": [],
                        }
                    ).cast(
                        {
                            "timestamp": pl.Datetime("us"),
                            "open": pl.Float32,
                            "high": pl.Float32,
                            "low": pl.Float32,
                            "close": pl.Float32,
                            "volume": pl.Float32,
                            "spread": pl.Int32,
                        }
                    )

                logger.info(f"Retrieved {len(rates)} bars")

                # NumPy structured arrayをPolars LazyFrameに変換
                df = pl.DataFrame(
                    {
                        "timestamp": [
                            datetime.fromtimestamp(r[0], tz=UTC) for r in rates
                        ],
                        "open": rates["open"].astype("float32"),
                        "high": rates["high"].astype("float32"),
                        "low": rates["low"].astype("float32"),
                        "close": rates["close"].astype("float32"),
                        "volume": rates["tick_volume"].astype("float32"),
                        "spread": rates["spread"].astype("int32"),
                    }
                )

                # LazyFrameに変換
                result_df = df.lazy()

                # 欠損検出を実行（有効な場合）
                if detect_gaps:
                    missing_periods = self.detect_missing_periods(result_df, timeframe)
                    if missing_periods:
                        logger.warning(
                            f"Found {len(missing_periods)} missing periods in the data"
                        )

                return result_df

            except Exception as e:
                logger.error(f"Error fetching OHLC data: {e}")
                raise RuntimeError(f"Failed to fetch OHLC data: {e}") from e

    def _calculate_batch_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        batch_size: int,
    ) -> list[tuple[datetime, datetime]]:
        """バッチ処理用に日付範囲を分割

        Args:
            start_date: 開始日時（UTC）
            end_date: 終了日時（UTC）
            timeframe: 時間足
            batch_size: バッチサイズ（バー数）

        Returns:
            List[tuple[datetime, datetime]]: 分割された日付範囲のリスト
        """
        # 時間足ごとのバー間隔（分）
        timeframe_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN": 43200,  # 約30日
        }

        if timeframe not in timeframe_minutes:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # バッチ間隔を計算（分）
        batch_interval_minutes = timeframe_minutes[timeframe] * batch_size
        batch_interval = timedelta(minutes=batch_interval_minutes)

        # バッチ境界を計算
        batch_dates = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + batch_interval, end_date)
            batch_dates.append((current_start, current_end))

            # 次のバッチの開始時刻（1バー分のオーバーラップ）
            current_start = current_end

        return batch_dates

    def _fetch_in_batches(
        self,
        symbol: str,
        mt5_timeframe: int,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> list[pl.LazyFrame]:
        """10,000バー単位でデータを分割取得（リトライ機能付き）

        Args:
            symbol: 通貨ペア
            mt5_timeframe: MT5時間足定数
            start_date: 開始日時（UTC）
            end_date: 終了日時（UTC）
            timeframe: 時間足文字列

        Returns:
            List[pl.LazyFrame]: バッチごとのLazyFrameリスト
        """
        # バッチ日付範囲を計算
        batch_dates = self._calculate_batch_dates(
            start_date, end_date, timeframe, self.batch_size
        )

        logger.info(
            f"Fetching data in {len(batch_dates)} batches "
            f"(batch_size={self.batch_size})"
        )

        # 各バッチを取得
        batch_results = []
        for i, (batch_start, batch_end) in enumerate(batch_dates, 1):
            logger.info(
                f"Fetching batch {i}/{len(batch_dates)}: {batch_start} to {batch_end}"
            )

            try:
                # MT5からバッチデータを取得（リトライ付き）
                def _fetch_batch_data(batch_idx=i, bs=batch_start, be=batch_end):
                    rates = mt5.copy_rates_range(symbol, mt5_timeframe, bs, be)
                    if rates is None:
                        raise ConnectionError(
                            f"Failed to fetch rates for batch {batch_idx}: {bs} to {be}"
                        )
                    return rates

                # リトライ付きでデータ取得
                rates = self._retry_with_backoff(
                    _fetch_batch_data,
                    max_retries=2,  # バッチごとのリトライは少なめに
                    initial_delay=0.5,
                    backoff_factor=2.0,
                )

                if len(rates) > 0:
                    # NumPy structured arrayをPolars LazyFrameに変換
                    df = pl.DataFrame(
                        {
                            "timestamp": [
                                datetime.fromtimestamp(r[0], tz=UTC) for r in rates
                            ],
                            "open": rates["open"].astype("float32"),
                            "high": rates["high"].astype("float32"),
                            "low": rates["low"].astype("float32"),
                            "close": rates["close"].astype("float32"),
                            "volume": rates["tick_volume"].astype("float32"),
                            "spread": rates["spread"].astype("int32"),
                        }
                    )
                    batch_results.append(df.lazy())
                    logger.info(f"Batch {i}: Retrieved {len(rates)} bars")
                else:
                    logger.warning(f"Batch {i}: No data returned")

            except Exception as e:
                # リトライ後も失敗した場合
                logger.error(f"Error fetching batch {i} after retries: {e}")
                # バッチ失敗時は空のLazyFrameを追加して続行
                empty_df = pl.LazyFrame(
                    {
                        "timestamp": [],
                        "open": [],
                        "high": [],
                        "low": [],
                        "close": [],
                        "volume": [],
                        "spread": [],
                    }
                ).cast(
                    {
                        "timestamp": pl.Datetime("us"),
                        "open": pl.Float32,
                        "high": pl.Float32,
                        "low": pl.Float32,
                        "close": pl.Float32,
                        "volume": pl.Float32,
                        "spread": pl.Int32,
                    }
                )
                batch_results.append(empty_df)

        return batch_results

    def _split_time_range(
        self,
        start_date: datetime,
        end_date: datetime,
        num_workers: int,
    ) -> list[tuple[datetime, datetime]]:
        """時間範囲を均等分割

        Args:
            start_date: 開始日時（UTC）
            end_date: 終了日時（UTC）
            num_workers: ワーカー数

        Returns:
            List[tuple[datetime, datetime]]: 分割された日付範囲のリスト
        """
        # 全期間を秒単位で計算
        total_seconds = (end_date - start_date).total_seconds()

        # ワーカーあたりの秒数
        seconds_per_worker = total_seconds / num_workers

        # 分割された範囲を生成
        time_chunks = []
        for i in range(num_workers):
            chunk_start = start_date + timedelta(seconds=i * seconds_per_worker)

            # 最後のワーカーの場合は終了日時まで
            if i == num_workers - 1:
                chunk_end = end_date
            else:
                chunk_end = start_date + timedelta(seconds=(i + 1) * seconds_per_worker)

            time_chunks.append((chunk_start, chunk_end))

        return time_chunks

    def _fetch_worker(
        self,
        worker_id: int,
        symbol: str,
        mt5_timeframe: int,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pl.LazyFrame | None:
        """各ワーカーでのデータ取得処理（リトライ機能付き）

        Args:
            worker_id: ワーカーID（ログ用）
            symbol: 通貨ペア
            mt5_timeframe: MT5時間足定数
            start_date: 開始日時（UTC）
            end_date: 終了日時（UTC）
            timeframe: 時間足文字列

        Returns:
            pl.LazyFrame | None: 取得したデータまたはNone（エラー時）
        """
        logger.info(
            f"Worker {worker_id}: Fetching {symbol} {timeframe} "
            f"from {start_date} to {end_date}"
        )

        try:
            # MT5接続の確立（リトライ付き）
            def _initialize_mt5():
                if not mt5.initialize():
                    raise ConnectionError(
                        f"Worker {worker_id}: Failed to initialize MT5"
                    )
                return True

            # リトライ付きでMT5を初期化
            self._retry_with_backoff(
                _initialize_mt5,
                max_retries=self.max_retries,
                initial_delay=1.0,
                backoff_factor=2.0,
            )

            # バッチ処理でデータ取得
            batch_results = []
            batch_dates = self._calculate_batch_dates(
                start_date, end_date, timeframe, self.batch_size
            )

            for batch_start, batch_end in batch_dates:
                try:
                    # copy_rates_rangeをリトライ付きで実行
                    def _fetch_rates(bs=batch_start, be=batch_end):
                        rates = mt5.copy_rates_range(symbol, mt5_timeframe, bs, be)
                        if rates is None:
                            raise ConnectionError(
                                f"Failed to fetch rates for {bs} to {be}"
                            )
                        return rates

                    rates = self._retry_with_backoff(
                        _fetch_rates,
                        max_retries=2,  # バッチごとのリトライは少なめに
                        initial_delay=0.5,
                        backoff_factor=2.0,
                    )

                    if len(rates) > 0:
                        df = pl.DataFrame(
                            {
                                "timestamp": [
                                    datetime.fromtimestamp(r[0], tz=UTC) for r in rates
                                ],
                                "open": rates["open"].astype("float32"),
                                "high": rates["high"].astype("float32"),
                                "low": rates["low"].astype("float32"),
                                "close": rates["close"].astype("float32"),
                                "volume": rates["tick_volume"].astype("float32"),
                                "spread": rates["spread"].astype("int32"),
                            }
                        )
                        batch_results.append(df.lazy())
                        logger.debug(
                            f"Worker {worker_id}: Batch {batch_start} to {batch_end} - "
                            f"{len(rates)} bars retrieved"
                        )

                except Exception as e:
                    # リトライ後も失敗した場合はログを記録して続行
                    logger.error(
                        f"Worker {worker_id}: Failed to fetch batch "
                        f"{batch_start} to {batch_end} after retries: {e}"
                    )
                    continue

            # MT5接続を閉じる
            mt5.shutdown()

            if batch_results:
                # バッチ結果を結合
                combined = pl.concat(batch_results, how="vertical")
                logger.info(f"Worker {worker_id}: Completed successfully")
                return combined
            else:
                logger.warning(f"Worker {worker_id}: No data retrieved")
                return None

        except Exception as e:
            logger.error(f"Worker {worker_id}: Fatal error: {e}")
            # エラー時もMT5接続を確実に閉じる
            try:
                mt5.shutdown()
            except Exception:
                pass
            return None

    def _fetch_parallel(
        self,
        symbol: str,
        mt5_timeframe: int,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pl.LazyFrame:
        """並列処理でデータ取得

        Args:
            symbol: 通貨ペア
            mt5_timeframe: MT5時間足定数
            start_date: 開始日時（UTC）
            end_date: 終了日時（UTC）
            timeframe: 時間足文字列

        Returns:
            pl.LazyFrame: 結合済みのデータ
        """
        # ワーカー数を決定（データ量に応じて調整）
        total_minutes = (end_date - start_date).total_seconds() / 60
        timeframe_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN": 43200,
        }
        estimated_bars = total_minutes / timeframe_minutes.get(timeframe, 1)

        # ワーカー数を動的に調整（最大でmax_workers、最小で1）
        num_workers = min(
            self.max_workers, max(1, int(estimated_bars / (self.batch_size * 2)))
        )

        logger.info(
            f"Starting parallel fetch with {num_workers} workers "
            f"for {estimated_bars:.0f} estimated bars"
        )

        # 時間範囲を分割
        time_chunks = self._split_time_range(start_date, end_date, num_workers)

        # ThreadPoolExecutorで並列実行
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 各ワーカーにタスクを送信
            futures = {}
            for i, (chunk_start, chunk_end) in enumerate(time_chunks):
                future = executor.submit(
                    self._fetch_worker,
                    i + 1,  # worker_id
                    symbol,
                    mt5_timeframe,
                    chunk_start,
                    chunk_end,
                    timeframe,
                )
                futures[future] = i + 1

            # 結果を収集
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    result = future.result(timeout=300)  # 5分のタイムアウト
                    if result is not None:
                        results.append(result)
                        logger.info(f"Worker {worker_id} result collected")
                    else:
                        logger.warning(f"Worker {worker_id} returned no data")
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed: {e}")

        # 結果の検証と結合
        if not results:
            logger.error("All workers failed to retrieve data")
            # 空のLazyFrameを返す
            return pl.LazyFrame(
                {
                    "timestamp": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                    "spread": [],
                }
            ).cast(
                {
                    "timestamp": pl.Datetime("us"),
                    "open": pl.Float32,
                    "high": pl.Float32,
                    "low": pl.Float32,
                    "close": pl.Float32,
                    "volume": pl.Float32,
                    "spread": pl.Int32,
                }
            )

        logger.info(f"Combining results from {len(results)} successful workers")

        # 全結果を結合し、タイムスタンプ順にソート、重複を除去
        combined = pl.concat(results, how="vertical_relaxed")
        return combined.unique("timestamp").sort("timestamp")

    def detect_missing_periods(
        self,
        df: pl.LazyFrame,
        timeframe: str,
    ) -> list[dict]:
        """取得したOHLCデータから欠損期間を検出

        Args:
            df: OHLCデータを含むLazyFrame
            timeframe: 時間足（"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"）

        Returns:
            List[dict]: 欠損期間のリスト
                - start: 欠損開始時刻（datetime）
                - end: 欠損終了時刻（datetime）
                - expected_bars: 期待されるバー数（int）
                - actual_gap: 実際のギャップ時間（timedelta）
        """
        # LazyFrameをDataFrameに変換（タイムスタンプのみ）
        df_collected = df.select("timestamp").collect()

        # データが空の場合は空リストを返す
        if df_collected.height == 0:
            logger.info("No data to check for missing periods")
            return []

        # タイムスタンプをソート
        df_sorted = df_collected.sort("timestamp")
        timestamps = df_sorted["timestamp"].to_list()

        # 時間足に応じた期待間隔を取得
        expected_interval = self._get_expected_interval(timeframe)

        # 欠損期間を検出
        missing_periods = []
        for i in range(1, len(timestamps)):
            prev_time = timestamps[i - 1]
            curr_time = timestamps[i]
            actual_gap = curr_time - prev_time

            # ギャップが期待間隔の1.5倍を超える場合を欠損と判定
            if self._is_gap(actual_gap, expected_interval):
                # 市場休場時間でない場合のみ欠損として記録
                if not self._is_market_closed(prev_time, curr_time):
                    expected_bars = int(
                        actual_gap.total_seconds() / expected_interval.total_seconds()
                    )
                    missing_periods.append(
                        {
                            "start": prev_time,
                            "end": curr_time,
                            "expected_bars": expected_bars,
                            "actual_gap": actual_gap,
                        }
                    )

                    logger.warning(
                        f"Missing data detected: {prev_time} to {curr_time} "
                        f"({expected_bars} bars, gap: {actual_gap})"
                    )

        # 欠損率を計算して情報として出力
        if missing_periods:
            total_bars = len(timestamps)
            total_missing = sum(p["expected_bars"] for p in missing_periods)
            missing_rate = total_missing / (total_bars + total_missing) * 100
            logger.info(
                f"Data completeness: {missing_rate:.2f}% missing "
                f"({total_missing} bars out of {total_bars + total_missing})"
            )
        else:
            logger.info("No missing periods detected")

        return missing_periods

    def _get_expected_interval(self, timeframe: str) -> timedelta:
        """時間足に応じた期待される間隔を返す

        Args:
            timeframe: 時間足（"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"）

        Returns:
            timedelta: 期待される間隔
        """
        interval_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN": 43200,  # 約30日
        }

        if timeframe not in interval_minutes:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        return timedelta(minutes=interval_minutes[timeframe])

    def _is_market_closed(self, start_time: datetime, end_time: datetime) -> bool:
        """市場休場時間かどうかを判定

        Args:
            start_time: 開始時刻（UTC）
            end_time: 終了時刻（UTC）

        Returns:
            bool: 市場休場時間の場合True
        """
        # 週末の判定（土曜日00:00 UTC から月曜日00:00 UTC）
        # 実際の市場は金曜日22:00 UTCから月曜日00:00 UTCまで休場だが、
        # ブローカーによって異なるため、土曜日全日と日曜日全日を休場とする

        # 開始時刻と終了時刻の間に週末が含まれているかチェック
        current = start_time
        while current <= end_time:
            # 土曜日（5）または日曜日（6）の場合
            if current.weekday() in [5, 6]:
                return True
            # 1日ずつチェック（長期間の場合は効率化の余地あり）
            current += timedelta(days=1)
            # 終了時刻を超えたら終了
            if current > end_time:
                break

        # 主要な祝日のチェック（オプション）
        # クリスマス（12/25）と元日（1/1）
        for date in [start_time, end_time]:
            if (date.month == 12 and date.day == 25) or (
                date.month == 1 and date.day == 1
            ):
                return True

        return False

    def _is_gap(self, actual_gap: timedelta, expected_interval: timedelta) -> bool:
        """欠損判定ロジック

        Args:
            actual_gap: 実際の時間差
            expected_interval: 期待される間隔

        Returns:
            bool: 欠損と判定される場合True
        """
        # 期待間隔の1.5倍を超える場合を欠損と判定
        # （市場の一時的な停止や軽微な遅延を許容）
        threshold = expected_interval * 1.5
        return actual_gap > threshold

    def _retry_with_backoff(
        self,
        func: Callable[..., Any],
        *args,
        max_retries: int = None,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        **kwargs,
    ) -> Any:
        """エクスポネンシャルバックオフでリトライを実行

        Args:
            func: 実行する関数
            *args: 関数の位置引数
            max_retries: 最大リトライ回数（デフォルト: self.max_retries）
            initial_delay: 初回リトライまでの待機時間（秒）
            backoff_factor: バックオフ係数（待機時間の倍率）
            max_delay: 最大待機時間（秒）
            **kwargs: 関数のキーワード引数

        Returns:
            Any: 関数の実行結果

        Raises:
            最後のリトライでも失敗した場合は元の例外を再発生
        """
        if max_retries is None:
            max_retries = self.max_retries

        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                # 関数を実行
                result = func(*args, **kwargs)

                # 成功した場合
                if attempt > 0:
                    logger.info(
                        f"Retry successful after {attempt} attempt(s) for {func.__name__}"
                    )
                return result

            except Exception as e:
                last_exception = e

                # リトライ可能なエラーかチェック
                if not self._is_retryable_error(e):
                    logger.error(
                        f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise

                # 最後の試行の場合
                if attempt == max_retries:
                    logger.error(
                        f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                        f"Last error: {type(e).__name__}: {e}"
                    )
                    raise

                # リトライログ
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: "
                    f"{type(e).__name__}: {e}. Retrying in {delay:.1f} seconds..."
                )

                # バックオフ待機
                time.sleep(delay)

                # 次回の待機時間を計算（最大値を超えないように）
                delay = min(delay * backoff_factor, max_delay)

        # ここには到達しないはずだが、念のため
        if last_exception:
            raise last_exception

    def _is_retryable_error(self, error: Exception) -> bool:
        """エラーがリトライ可能かどうかを判定

        Args:
            error: 発生した例外

        Returns:
            bool: リトライ可能な場合True

        リトライ可能なエラー:
            - ConnectionError: ネットワーク接続エラー
            - TimeoutError: タイムアウトエラー
            - OSError: システムレベルのエラー（一時的な場合）
            - MT5エラー: 一時的なMT5サーバーエラー

        リトライ不可能なエラー:
            - ValueError: 無効なパラメータ
            - PermissionError: 権限エラー
            - KeyError: キーエラー
            - その他のプログラムエラー
        """
        # リトライ不可能なエラーを先にチェック（優先度高）
        non_retryable_types = (
            ValueError,
            PermissionError,
            KeyError,
            AttributeError,
            TypeError,
        )

        if isinstance(error, non_retryable_types):
            return False

        # リトライ可能なエラータイプ
        retryable_types = (
            ConnectionError,
            TimeoutError,
            OSError,
        )

        # エラータイプをチェック
        if isinstance(error, retryable_types):
            # PermissionErrorはOSErrorのサブクラスなので除外
            if isinstance(error, PermissionError):
                return False
            return True

        # MT5固有のエラーをチェック（エラーメッセージで判定）
        error_message = str(error).lower()
        retryable_messages = [
            "connection",
            "timeout",
            "network",
            "server",
            "temporary",
            "unavailable",
            "busy",
            "failed to connect",
            "no connection",
        ]

        for msg in retryable_messages:
            if msg in error_message:
                return True

        # デフォルトはリトライ可能とする（安全側に倒す）
        return True
