"""
マルチタイムフレーム分析エンジン

短期（1分足）と長期（5分足）のRCIを並列計算し、
トレンド分析のための統合データを生成します。
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from src.data_processing.rci import RCICalculatorEngine
from src.data_processing.timeframe_converter import (
    TimeframeConverter,
)

logger = logging.getLogger(__name__)


class MultiTimeframeAnalysisError(Exception):
    """マルチタイムフレーム分析固有のエラー"""

    pass


class AnalysisConfigurationError(MultiTimeframeAnalysisError):
    """分析設定エラー"""

    pass


class DataAlignmentError(MultiTimeframeAnalysisError):
    """データ整列エラー"""

    pass


class MultiTimeframeAnalyzer:
    """
    マルチタイムフレーム分析クラス

    1分足データから短期・長期のRCI指標を計算し、
    包括的なトレンド分析データを生成します。

    Attributes:
        short_term_periods: 短期RCI（1分足）の計算期間
        long_term_periods: 長期RCI（5分足）の計算期間
        long_timeframe: 長期分析のタイムフレーム
        incomplete_bar_handling: 不完全なバーの処理方法
        use_parallel: 並列処理を使用するかどうか
        max_workers: 並列処理の最大ワーカー数

    Example:
        >>> analyzer = MultiTimeframeAnalyzer()
        >>> result = analyzer.analyze(df_1min)
        >>> print(result.columns)
        ['timestamp', 'open', 'high', 'low', 'close', 'volume',
         'short_rci_9', 'short_rci_13', ..., 'long_rci_24', 'long_rci_33', ...]
    """

    # デフォルトのRCI期間設定
    DEFAULT_SHORT_PERIODS = [9, 13, 24, 33, 48, 66, 108]  # 1分足用
    DEFAULT_LONG_PERIODS = [24, 33, 48, 66, 108]  # 5分足用（120→24, 165→33, ...）

    def __init__(
        self,
        short_term_periods: list[int] | None = None,
        long_term_periods: list[int] | None = None,
        long_timeframe: str = "5T",
        incomplete_bar_handling: str = "drop",
        use_parallel: bool = True,
        max_workers: int | None = None,
        max_history_bars: int = 5000,
    ):
        """
        マルチタイムフレーム分析エンジンを初期化します。

        Args:
            short_term_periods: 短期RCI（1分足）の計算期間リスト
            long_term_periods: 長期RCI（5分足）の計算期間リスト
            long_timeframe: 長期分析のタイムフレーム（デフォルト: 5分足）
            incomplete_bar_handling: 不完全バーの処理方法（"drop", "keep", "preview"）
            use_parallel: 並列処理を使用するか
            max_workers: 並列処理の最大ワーカー数（Noneで自動設定）
            max_history_bars: 保持する最大履歴バー数（デフォルト: 5000）

        Raises:
            AnalysisConfigurationError: 無効な設定パラメータの場合
        """
        self.short_term_periods = short_term_periods or self.DEFAULT_SHORT_PERIODS
        self.long_term_periods = long_term_periods or self.DEFAULT_LONG_PERIODS
        self.long_timeframe = long_timeframe
        self.incomplete_bar_handling = incomplete_bar_handling
        self.use_parallel = use_parallel
        self.max_workers = max_workers

        # バッファ管理プロパティの追加
        self._data_buffer: list[dict[str, Any]] = []
        self._max_history_bars = max_history_bars
        self._min_required_bars = 200  # 分析に必要な最小バー数

        # コンポーネントの初期化
        try:
            self.timeframe_converter = TimeframeConverter(
                target_timeframe=long_timeframe,
                align_to_boundary=True,
            )
            self.short_term_engine = RCICalculatorEngine()
            self.long_term_engine = RCICalculatorEngine()
        except Exception as e:
            raise AnalysisConfigurationError(f"コンポーネント初期化エラー: {e}") from e

        # 期間設定の検証
        self._validate_periods()

        logger.info(
            f"MultiTimeframeAnalyzer initialized: "
            f"short_periods={self.short_term_periods}, "
            f"long_periods={self.long_term_periods}, "
            f"long_timeframe={self.long_timeframe}, "
            f"max_history_bars={self._max_history_bars}"
        )

    def _validate_periods(self) -> None:
        """期間設定を検証します。"""
        all_periods = self.short_term_periods + self.long_term_periods

        # RCIエンジンの制限値をチェック
        invalid_periods = [
            p
            for p in all_periods
            if p < RCICalculatorEngine.MIN_PERIOD or p > RCICalculatorEngine.MAX_PERIOD
        ]

        if invalid_periods:
            raise AnalysisConfigurationError(
                f"無効な期間が含まれています: {invalid_periods}. "
                f"有効範囲: {RCICalculatorEngine.MIN_PERIOD}-{RCICalculatorEngine.MAX_PERIOD}"
            )

    def analyze(
        self,
        data: pl.DataFrame,
        return_intermediate: bool = False,
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """
        マルチタイムフレーム分析を実行します。

        Args:
            data: 1分足のOHLCVデータ（timestamp, open, high, low, close, volume）
            return_intermediate: 中間結果（短期・長期の個別結果）も返すか

        Returns:
            統合されたRCI分析結果（DataFrame）
            return_intermediate=Trueの場合は辞書形式:
            {
                "combined": 統合結果,
                "short_term": 短期RCI結果,
                "long_term": 長期RCI結果,
                "long_term_ohlcv": 5分足OHLCV
            }

        Raises:
            MultiTimeframeAnalysisError: 分析処理エラー
            ValueError: 入力データが不正な場合
        """
        # 入力データの検証
        self._validate_input_data(data)

        try:
            # 並列処理か逐次処理かを選択
            if self.use_parallel:
                results = self._analyze_parallel(data)
            else:
                results = self._analyze_sequential(data)

            # 結果の統合
            combined = self._merge_results(
                original_data=data,
                short_term_rci=results["short_term"],
                long_term_rci=results["long_term"],
                long_term_data=results["long_term_data"],
            )

            if return_intermediate:
                return {
                    "combined": combined,
                    "short_term": results["short_term"],
                    "long_term": results["long_term"],
                    "long_term_ohlcv": results["long_term_data"],
                }

            return combined

        except Exception as e:
            logger.error(f"マルチタイムフレーム分析エラー: {e}")
            raise MultiTimeframeAnalysisError(f"分析処理に失敗しました: {e}") from e

    def _analyze_parallel(self, data: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """
        並列処理による分析を実行します。

        Args:
            data: 1分足データ

        Returns:
            短期・長期のRCI結果を含む辞書
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            # 短期RCI計算タスク
            futures["short_term"] = executor.submit(
                self._calculate_short_term_rci, data
            )

            # 長期RCI計算タスク（タイムフレーム変換を含む）
            futures["long_term_full"] = executor.submit(
                self._calculate_long_term_rci_with_conversion, data
            )

            # 結果の収集
            for key, future in futures.items():
                try:
                    result = future.result(timeout=30)  # 30秒のタイムアウト
                    if key == "long_term_full":
                        results["long_term"] = result["rci"]
                        results["long_term_data"] = result["data"]
                    else:
                        results[key] = result
                except Exception as e:
                    logger.error(f"並列処理エラー ({key}): {e}")
                    raise

        return results

    def _analyze_sequential(self, data: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """
        逐次処理による分析を実行します。

        Args:
            data: 1分足データ

        Returns:
            短期・長期のRCI結果を含む辞書
        """
        # 短期RCI計算
        short_term = self._calculate_short_term_rci(data)

        # 長期RCI計算（タイムフレーム変換を含む）
        long_term_result = self._calculate_long_term_rci_with_conversion(data)

        return {
            "short_term": short_term,
            "long_term": long_term_result["rci"],
            "long_term_data": long_term_result["data"],
        }

    def _calculate_short_term_rci(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        短期RCI（1分足）を計算します。

        Args:
            data: 1分足データ

        Returns:
            短期RCI結果（timestamp, short_rci_9, short_rci_13, ...）
        """
        logger.debug(f"短期RCI計算開始: periods={self.short_term_periods}")

        # RCI計算
        rci_results = self.short_term_engine.calculate_multiple(
            data=data,
            periods=self.short_term_periods,
            column_name="close",
            mode="batch",
        )

        # カラム名をプレフィックス付きに変更
        rename_map = {f"rci_{p}": f"short_rci_{p}" for p in self.short_term_periods}
        rci_results = rci_results.rename(rename_map)

        logger.debug(f"短期RCI計算完了: shape={rci_results.shape}")
        return rci_results

    def _calculate_long_term_rci_with_conversion(
        self, data: pl.DataFrame
    ) -> dict[str, pl.DataFrame]:
        """
        長期RCI（5分足）を計算します（タイムフレーム変換を含む）。

        Args:
            data: 1分足データ

        Returns:
            辞書形式の結果:
            - "rci": 長期RCI結果
            - "data": 5分足OHLCV
        """
        logger.debug(f"長期RCI計算開始: timeframe={self.long_timeframe}")

        # タイムフレーム変換（1分足→5分足）
        long_term_data = self.timeframe_converter.convert(
            df=data,
            incomplete_bar_handling=self.incomplete_bar_handling,
        )

        # 長期データが少なすぎる場合の処理
        min_required_bars = max(self.long_term_periods) + 1
        if len(long_term_data) < min_required_bars:
            logger.warning(
                f"長期データ不足: {len(long_term_data)} < {min_required_bars}"
            )
            # 空のDataFrameを返す
            empty_rci = pl.DataFrame({"timestamp": []})
            for period in self.long_term_periods:
                empty_rci = empty_rci.with_columns(
                    pl.lit(None).alias(f"long_rci_{period}")
                )
            return {"rci": empty_rci, "data": long_term_data}

        # RCI計算
        rci_results = self.long_term_engine.calculate_multiple(
            data=long_term_data,
            periods=self.long_term_periods,
            column_name="close",
            mode="batch",
        )

        # カラム名をプレフィックス付きに変更
        rename_map = {f"rci_{p}": f"long_rci_{p}" for p in self.long_term_periods}
        rci_results = rci_results.rename(rename_map)

        logger.debug(f"長期RCI計算完了: shape={rci_results.shape}")
        return {"rci": rci_results, "data": long_term_data}

    def _merge_results(
        self,
        original_data: pl.DataFrame,
        short_term_rci: pl.DataFrame,
        long_term_rci: pl.DataFrame,
        long_term_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        短期・長期のRCI結果を統合します。

        Args:
            original_data: 元の1分足データ
            short_term_rci: 短期RCI結果
            long_term_rci: 長期RCI結果
            long_term_data: 5分足データ

        Returns:
            統合された結果
        """
        logger.debug("結果の統合開始")

        # ベースデータ（1分足のOHLCV）
        result = original_data.select(
            ["timestamp", "open", "high", "low", "close", "volume"]
        )

        # 短期RCIの結合
        result = result.join(short_term_rci, on="timestamp", how="left")

        # 長期RCIの結合（タイムスタンプのアライメント）
        if len(long_term_rci) > 0:
            result = self._align_long_term_rci(result, long_term_rci, long_term_data)

        # NaN値の処理（オプション）
        # 初期のバーではRCI計算ができないため、Noneが含まれる
        # これは正常な動作なので、そのまま保持

        logger.debug(f"結果統合完了: shape={result.shape}")
        return result

    def _align_long_term_rci(
        self,
        base_df: pl.DataFrame,
        long_term_rci: pl.DataFrame,
        long_term_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        長期RCIを1分足データにアライメントします。

        5分足のRCI値を対応する1分足のタイムスタンプに割り当てます。
        各5分足バーのRCI値は、その期間内のすべての1分足バーで同じ値になります。

        Args:
            base_df: ベースとなる1分足データ
            long_term_rci: 長期RCI結果
            long_term_data: 5分足データ

        Returns:
            長期RCIが結合された結果
        """
        # 長期RCIのカラムだけを抽出（timestamp含む）
        long_rci_cols = ["timestamp"] + [
            col for col in long_term_rci.columns if "long_rci" in col
        ]
        long_term_rci_only = long_term_rci.select(long_rci_cols)

        # 各1分足タイムスタンプに対応する5分足期間を特定
        # 1分足のタイムスタンプを5分足の期間に切り下げ
        # Polarsのtruncateは"5T"形式を受け付けないため変換
        truncate_arg = self._convert_timeframe_for_truncate(self.long_timeframe)
        base_with_period = base_df.with_columns(
            (pl.col("timestamp").dt.truncate(truncate_arg)).alias("long_period")
        )

        # 5分足RCIを1分足データに結合
        # 各5分足期間のRCI値が、その期間内のすべての1分足バーに割り当てられる
        result = base_with_period.join(
            long_term_rci_only.rename({"timestamp": "long_period"}),
            on="long_period",
            how="left",
        ).drop("long_period")

        return result

    def _validate_input_data(self, data: pl.DataFrame) -> None:
        """
        入力データを検証します。

        Args:
            data: 検証するデータ

        Raises:
            ValueError: データが不正な場合
        """
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")

        if len(data) == 0:
            raise ValueError("データが空です")

        # タイムスタンプがソートされているか確認
        if not data["timestamp"].is_sorted():
            logger.warning(
                "タイムスタンプがソートされていません。自動的にソートします。"
            )

    def add_new_bar(self, bar: dict[str, Any]) -> None:
        """
        新しいバーをバッファに追加し、サイズを管理します。

        Args:
            bar: 追加するバーデータ（timestamp, open, high, low, close, volume）
        """
        # None値のチェック
        if bar is None:
            logger.warning("None値のバーが渡されました。スキップします。")
            return
        
        # OHLC妥当性チェック
        try:
            high = float(bar.get("high", 0))
            low = float(bar.get("low", 0))
            close = float(bar.get("close", 0))
            open_price = float(bar.get("open", 0))
            
            # High < Low のチェック
            if high > 0 and low > 0 and high < low:
                logger.warning(
                    f"Invalid bar: high ({high}) < low ({low}). "
                    f"Bar timestamp: {bar.get('timestamp')}"
                )
                return
            
            # Close が High/Low の範囲外のチェック
            if high > 0 and low > 0 and close > 0:
                if not (low <= close <= high):
                    logger.warning(
                        f"Invalid bar: close ({close}) out of high/low range [{low}, {high}]. "
                        f"Bar timestamp: {bar.get('timestamp')}"
                    )
                    return
            
            # Open が High/Low の範囲外のチェック
            if high > 0 and low > 0 and open_price > 0:
                if not (low <= open_price <= high):
                    logger.warning(
                        f"Invalid bar: open ({open_price}) out of high/low range [{low}, {high}]. "
                        f"Bar timestamp: {bar.get('timestamp')}"
                    )
                    return
                    
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type in bar data: {e}. Bar data: {bar}")
            return

        self._data_buffer.append(bar)
        self._manage_buffer_size()

    def _manage_buffer_size(self) -> None:
        """
        バッファサイズを最大値以内に維持します。
        """
        if len(self._data_buffer) > self._max_history_bars:
            self._data_buffer = self._data_buffer[-self._max_history_bars :]

    def get_buffer_size(self) -> int:
        """
        現在のバッファサイズを返します。

        Returns:
            バッファ内のバー数
        """
        return len(self._data_buffer)

    def is_ready(self) -> bool:
        """
        分析準備が完了しているかを返します。

        Returns:
            最小バー数以上のデータがある場合True
        """
        return len(self._data_buffer) >= self._min_required_bars

    def get_buffer_as_dataframe(self) -> pl.DataFrame | None:
        """
        バッファをDataFrameとして取得します。

        Returns:
            バッファのDataFrame形式、バッファが空の場合はNone
        """
        if not self._data_buffer:
            return None
        return pl.DataFrame(self._data_buffer)

    def analyze_streaming(
        self,
        new_bar: dict[str, Any] | None = None,
        history: pl.DataFrame | None = None,
        min_history_bars: int = 200,
    ) -> dict[str, Any]:
        """
        ストリーミングデータのマルチタイムフレーム分析を実行します。

        新しい1分足バーを受信するたびに、短期・長期のRCIを更新します。
        後方互換性のため、外部履歴データも引き続きサポートします。

        Args:
            new_bar: 新しい1分足バー（dict形式）（内部バッファ使用時は省略可）
            history: 過去のOHLCVデータ（外部履歴使用時）
            min_history_bars: 最小履歴バー数（外部履歴使用時）

        Returns:
            最新のRCI値を含む辞書、または準備未完了ステータス

        Raises:
            MultiTimeframeAnalysisError: 分析処理エラー
        """
        # 後方互換性の維持：外部履歴が提供された場合
        if history is not None:
            return self._analyze_with_external_history(
                new_bar, history, min_history_bars
            )

        # 新しい動作：内部バッファを使用
        if not self.is_ready():
            return {
                "timestamp": new_bar.get("timestamp") if new_bar else None,
                "status": "not_ready",
                "buffer_size": self.get_buffer_size(),
                "required_bars": self._min_required_bars,
            }

        history_df = self.get_buffer_as_dataframe()
        if history_df is None:
            return {"status": "no_data"}

        # 既存のRCI計算ロジックを利用
        return self._calculate_rci_metrics(history_df)

    def _analyze_with_external_history(
        self, new_bar: dict[str, Any], history: pl.DataFrame, min_history_bars: int
    ) -> dict[str, Any]:
        """
        既存の外部履歴を使用した分析（後方互換性）。

        Args:
            new_bar: 新しい1分足バー（dict形式）
            history: 過去のOHLCVデータ
            min_history_bars: 最小履歴バー数

        Returns:
            最新のRCI値を含む辞書
        """
        # 新しいバーを履歴に追加
        new_row = pl.DataFrame([new_bar])
        updated_history = pl.concat([history, new_row])

        # 履歴サイズの制限（メモリ効率化）
        max_bars = max(self.short_term_periods + [self.long_term_periods[-1] * 5]) * 2
        if len(updated_history) > max_bars:
            updated_history = updated_history[-max_bars:]

        try:
            # 短期RCI計算（最新値のみ）
            short_rci = {}
            for period in self.short_term_periods:
                if len(updated_history) >= period:
                    recent_data = updated_history[-period:]
                    rci_value = self._calculate_single_rci(
                        recent_data["close"].to_numpy(), period
                    )
                    short_rci[period] = rci_value

            # 5分足バーが完成したかチェック
            is_new_long_bar = self._is_new_long_bar_complete(new_bar["timestamp"])

            # 長期RCI計算（5分足バー完成時のみ）
            long_rci = {}
            if is_new_long_bar:
                # 5分足データに変換（convert_streamingはタプルを返すが、完成バーのみ使用）
                long_term_data, _ = self.timeframe_converter.convert_streaming(
                    df=updated_history, incomplete_bar_handling="drop"
                )

                for period in self.long_term_periods:
                    if len(long_term_data) >= period:
                        recent_data = long_term_data[-period:]
                        rci_value = self._calculate_single_rci(
                            recent_data["close"].to_numpy(), period
                        )
                        long_rci[period] = rci_value

            return {
                "timestamp": new_bar["timestamp"],
                "short_rci": short_rci,
                "long_rci": long_rci,
                "is_new_long_bar": is_new_long_bar,
            }

        except Exception as e:
            logger.error(f"外部履歴分析エラー: {e}")
            raise MultiTimeframeAnalysisError(f"外部履歴分析に失敗: {e}") from e

    def _calculate_rci_metrics(self, history_df: pl.DataFrame) -> dict[str, Any]:
        """
        RCIメトリクスの計算（内部バッファ用）。

        Args:
            history_df: 履歴データのDataFrame

        Returns:
            最新のRCI値を含む辞書
        """
        try:
            # 最新のタイムスタンプを取得
            latest_timestamp = history_df["timestamp"][-1]

            # 短期RCI計算（最新値のみ）
            short_rci = {}
            for period in self.short_term_periods:
                if len(history_df) >= period:
                    recent_data = history_df[-period:]
                    rci_value = self._calculate_single_rci(
                        recent_data["close"].to_numpy(), period
                    )
                    short_rci[period] = rci_value

            # 5分足バーが完成したかチェック
            is_new_long_bar = self._is_new_long_bar_complete(latest_timestamp)

            # 長期RCI計算（5分足バー完成時のみ）
            long_rci = {}
            if is_new_long_bar:
                # 5分足データに変換
                long_term_data, _ = self.timeframe_converter.convert_streaming(
                    df=history_df, incomplete_bar_handling="drop"
                )

                for period in self.long_term_periods:
                    if len(long_term_data) >= period:
                        recent_data = long_term_data[-period:]
                        rci_value = self._calculate_single_rci(
                            recent_data["close"].to_numpy(), period
                        )
                        long_rci[period] = rci_value

            return {
                "timestamp": latest_timestamp,
                "short_rci": short_rci,
                "long_rci": long_rci,
                "is_new_long_bar": is_new_long_bar,
            }

        except Exception as e:
            logger.error(f"RCI計算エラー: {e}")
            raise MultiTimeframeAnalysisError(f"RCI計算に失敗: {e}") from e

    def _is_new_long_bar_complete(self, timestamp: datetime) -> bool:
        """
        新しい5分足バーが完成したかチェックします。
        
        境界を超えたかどうかを判定する方式に変更。
        （以前の timestamp.second == 0 の条件は実際のティックデータでは
        ほぼ満たされないため）

        Args:
            timestamp: チェックするタイムスタンプ

        Returns:
            5分足バーが完成した場合True
        """
        # タイムフレームのパース（"5T" -> 5分）
        interval_minutes = int(self.long_timeframe.rstrip("T"))
        
        # 現在のバーの開始分を計算
        current_bar_minute = (timestamp.minute // interval_minutes) * interval_minutes
        
        # 初回チェック時は前回のバー時刻を記録
        if not hasattr(self, '_last_bar_minute'):
            self._last_bar_minute = current_bar_minute
            return False
        
        # バー境界を超えたかチェック
        if current_bar_minute != self._last_bar_minute:
            logger.debug(f"M5 bar boundary crossed: {self._last_bar_minute:02d}:00 -> {current_bar_minute:02d}:00 at {timestamp}")
            self._last_bar_minute = current_bar_minute
            return True
        
        return False

    def _calculate_single_rci(self, prices: np.ndarray, period: int) -> float:
        """
        単一期間のRCIを計算します（簡易版）。

        Args:
            prices: 価格データ
            period: 計算期間

        Returns:
            RCI値（-100 to 100）
        """
        n = len(prices)
        if n != period:
            raise ValueError(f"データ長が期間と一致しません: {n} != {period}")

        # 価格順位と時間順位を計算
        price_ranks = np.argsort(np.argsort(prices)) + 1
        time_ranks = np.arange(1, n + 1)

        # スピアマンの順位相関係数を計算
        d = price_ranks - time_ranks
        rho = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))

        # RCIに変換（-100 to 100）
        return rho * 100

    def _convert_timeframe_for_truncate(self, timeframe: str) -> str:
        """
        タイムフレーム形式をPolarsのtruncate用に変換します。

        Args:
            timeframe: 元のタイムフレーム形式（例: "5T", "15T", "1H"）

        Returns:
            Polars truncate用の形式（例: "5m", "15m", "1h"）
        """
        mapping = {
            "5T": "5m",
            "15T": "15m",
            "30T": "30m",
            "1H": "1h",
            "4H": "4h",
            "1D": "1d",
        }
        return mapping.get(timeframe, "5m")  # デフォルトは5分

    def get_analyzer_info(self) -> dict[str, Any]:
        """
        アナライザーの設定情報を返します。

        Returns:
            設定情報を含む辞書
        """
        return {
            "short_term_periods": self.short_term_periods,
            "long_term_periods": self.long_term_periods,
            "long_timeframe": self.long_timeframe,
            "incomplete_bar_handling": self.incomplete_bar_handling,
            "use_parallel": self.use_parallel,
            "max_workers": self.max_workers,
        }
