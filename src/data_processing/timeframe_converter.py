"""
タイムフレーム変換モジュール

1分足データを高次タイムフレーム（5分足など）にリサンプリングする機能を提供。
Polarsの`group_by_dynamic`を使用した効率的な集約処理を実装。
"""

import logging
from datetime import datetime
from typing import Any, Literal

import polars as pl

logger = logging.getLogger(__name__)


class TimeframeConversionError(Exception):
    """タイムフレーム変換固有のエラー"""

    pass


class InvalidTimeframeError(TimeframeConversionError):
    """無効なタイムフレームパラメータエラー"""

    pass


class InsufficientBarsError(TimeframeConversionError):
    """変換に必要なバー数が不足しているエラー"""

    pass


class TimeframeConverter:
    """
    タイムフレーム変換クラス

    1分足データを高次タイムフレーム（5分足、15分足など）に効率的にリサンプリングします。
    Polarsの`group_by_dynamic`を使用し、OHLCVデータの適切な集約を行います。

    Attributes:
        source_timeframe: ソースタイムフレーム（常に"1T"）
        target_timeframe: ターゲットタイムフレーム（"5T", "15T", "30T", "1H"）
        align_to_boundary: タイムスタンプを境界に整列させるかどうか

    Example:
        >>> converter = TimeframeConverter("5T")
        >>> df_5min = converter.convert(df_1min)
    """

    SUPPORTED_TIMEFRAMES = {
        "5T": "5分足",
        "15T": "15分足",
        "30T": "30分足",
        "1H": "1時間足",
        "4H": "4時間足",
        "1D": "日足",
    }

    def __init__(
        self,
        target_timeframe: Literal["5T", "15T", "30T", "1H", "4H", "1D"],
        source_timeframe: str = "1T",
        align_to_boundary: bool = True,
    ):
        """
        TimeframeConverterを初期化

        Args:
            target_timeframe: 変換先のタイムフレーム
            source_timeframe: 元のタイムフレーム（デフォルト: 1分足）
            align_to_boundary: タイムスタンプを境界に整列させるか

        Raises:
            InvalidTimeframeError: サポートされていないタイムフレームが指定された場合
        """
        if target_timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise InvalidTimeframeError(
                f"Unsupported timeframe: {target_timeframe}. "
                f"Supported: {list(self.SUPPORTED_TIMEFRAMES.keys())}"
            )

        if source_timeframe != "1T":
            raise InvalidTimeframeError(
                f"Source timeframe must be '1T' (1-minute), got: {source_timeframe}"
            )

        self.source_timeframe = source_timeframe
        self.target_timeframe = target_timeframe
        self.align_to_boundary = align_to_boundary

        # Polarsのgroup_by_dynamic用のinterval文字列を生成
        self._interval = self._convert_to_polars_interval(target_timeframe)

        logger.info(
            f"TimeframeConverter initialized: {source_timeframe} -> {target_timeframe}, "
            f"align_to_boundary={align_to_boundary}"
        )

    def _convert_to_polars_interval(self, timeframe: str) -> str:
        """
        タイムフレーム文字列をPolarsのinterval形式に変換

        Args:
            timeframe: タイムフレーム文字列（"5T", "1H"など）

        Returns:
            Polarsのinterval文字列（"5m", "1h"など）
        """
        mapping = {
            "5T": "5m",
            "15T": "15m",
            "30T": "30m",
            "1H": "1h",
            "4H": "4h",
            "1D": "1d",
        }
        return mapping.get(timeframe, timeframe)

    def convert(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "timestamp",
        price_cols: dict[str, str] | None = None,
        volume_col: str | None = "volume",
        additional_cols: dict[str, str] | None = None,
        incomplete_bar_handling: Literal["drop", "keep", "preview"] = "drop",
    ) -> pl.DataFrame:
        """
        1分足データを高次タイムフレームに変換

        Args:
            df: 入力データフレーム（1分足データ）
            timestamp_col: タイムスタンプカラム名
            price_cols: 価格カラムのマッピング（デフォルト: OHLC）
            volume_col: 出来高カラム名
            additional_cols: 追加集約カラムと集約方法のマッピング
            incomplete_bar_handling: 不完全なバーの処理方法
                - "drop": 不完全なバーを削除
                - "keep": 不完全なバーを保持
                - "preview": 不完全なバーをプレビューとしてマーク

        Returns:
            変換後のデータフレーム

        Raises:
            InsufficientBarsError: データが不足している場合
            TimeframeConversionError: 変換中にエラーが発生した場合
        """
        # 入力検証
        if df.is_empty():
            raise InsufficientBarsError("Input dataframe is empty")

        if timestamp_col not in df.columns:
            raise TimeframeConversionError(
                f"Timestamp column '{timestamp_col}' not found"
            )

        # デフォルトの価格カラム設定
        if price_cols is None:
            price_cols = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
            }

        # 価格カラムの存在確認
        for col_type, col_name in price_cols.items():
            if col_name not in df.columns:
                raise TimeframeConversionError(
                    f"Price column '{col_name}' ({col_type}) not found"
                )

        try:
            # タイムスタンプが datetime型であることを確認
            if not isinstance(df[timestamp_col].dtype, pl.Datetime):
                df = df.with_columns(pl.col(timestamp_col).cast(pl.Datetime))

            # ソート（タイムスタンプ順）
            df = df.sort(timestamp_col)

            # 集約式のリストを構築
            agg_exprs = []

            # OHLC集約
            if "open" in price_cols:
                agg_exprs.append(pl.col(price_cols["open"]).first().alias("open"))
            if "high" in price_cols:
                agg_exprs.append(pl.col(price_cols["high"]).max().alias("high"))
            if "low" in price_cols:
                agg_exprs.append(pl.col(price_cols["low"]).min().alias("low"))
            if "close" in price_cols:
                agg_exprs.append(pl.col(price_cols["close"]).last().alias("close"))

            # Volume集約
            if volume_col and volume_col in df.columns:
                agg_exprs.append(pl.col(volume_col).sum().alias("volume"))

            # カウント（バー数）
            agg_exprs.append(pl.count().alias("bar_count"))

            # 追加カラムの集約
            if additional_cols:
                for col_name, agg_method in additional_cols.items():
                    if col_name in df.columns:
                        if agg_method == "mean":
                            agg_exprs.append(pl.col(col_name).mean().alias(col_name))
                        elif agg_method == "sum":
                            agg_exprs.append(pl.col(col_name).sum().alias(col_name))
                        elif agg_method == "last":
                            agg_exprs.append(pl.col(col_name).last().alias(col_name))
                        elif agg_method == "first":
                            agg_exprs.append(pl.col(col_name).first().alias(col_name))
                        else:
                            logger.warning(
                                f"Unknown aggregation method '{agg_method}' for column '{col_name}'"
                            )

            # group_by_dynamicを使用したリサンプリング
            resampled = df.group_by_dynamic(
                timestamp_col,
                every=self._interval,
                label="left" if self.align_to_boundary else "datapoint",
                include_boundaries=False,
                closed="left",
            ).agg(agg_exprs)

            # 不完全なバーの処理
            if incomplete_bar_handling != "keep":
                # 期待されるバー数を計算
                expected_bars = self._get_expected_bars_per_period()

                if incomplete_bar_handling == "drop":
                    # 不完全なバーを削除
                    resampled = resampled.filter(pl.col("bar_count") == expected_bars)
                elif incomplete_bar_handling == "preview":
                    # 不完全なバーをマーク
                    resampled = resampled.with_columns(
                        (pl.col("bar_count") == expected_bars).alias("is_complete")
                    )

            # カラムの並び替え（標準的な順序）
            column_order = [timestamp_col]
            for col in ["open", "high", "low", "close", "volume"]:
                if col in resampled.columns:
                    column_order.append(col)

            # その他のカラムを追加
            for col in resampled.columns:
                if col not in column_order:
                    column_order.append(col)

            resampled = resampled.select(column_order)

            logger.debug(
                f"Converted {len(df)} {self.source_timeframe} bars to "
                f"{len(resampled)} {self.target_timeframe} bars"
            )

            return resampled

        except Exception as e:
            logger.error(f"Error during timeframe conversion: {str(e)}")
            raise TimeframeConversionError(
                f"Failed to convert timeframe: {str(e)}"
            ) from e

    def _get_expected_bars_per_period(self) -> int:
        """
        各タイムフレームで期待されるバー数を取得

        Returns:
            期待されるバー数
        """
        mapping = {
            "5T": 5,  # 5分足 = 5本の1分足
            "15T": 15,  # 15分足 = 15本の1分足
            "30T": 30,  # 30分足 = 30本の1分足
            "1H": 60,  # 1時間足 = 60本の1分足
            "4H": 240,  # 4時間足 = 240本の1分足
            "1D": 1440,  # 日足 = 1440本の1分足
        }
        return mapping.get(self.target_timeframe, 1)

    def convert_streaming(
        self,
        df: pl.DataFrame,
        last_complete_timestamp: datetime | None = None,
        **kwargs,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """
        ストリーミングデータ向けの変換（完成バーと未完成バーを分離）

        Args:
            df: 入力データフレーム（1分足データ）
            last_complete_timestamp: 最後の完成バーのタイムスタンプ
            **kwargs: convertメソッドに渡す追加引数

        Returns:
            (完成バーのDataFrame, 未完成バーのDataFrame（プレビュー）)
        """
        # プレビューモードで変換
        kwargs["incomplete_bar_handling"] = "preview"
        converted = self.convert(df, **kwargs)

        if "is_complete" in converted.columns:
            # 完成バーと未完成バーを分離
            complete_bars = converted.filter(pl.col("is_complete")).drop("is_complete")
            incomplete_bars = converted.filter(~pl.col("is_complete")).drop(
                "is_complete"
            )

            # last_complete_timestampが指定されている場合、それ以降のバーのみを返す
            if last_complete_timestamp is not None:
                timestamp_col = kwargs.get("timestamp_col", "timestamp")
                complete_bars = complete_bars.filter(
                    pl.col(timestamp_col) > last_complete_timestamp
                )

            return (
                complete_bars if not complete_bars.is_empty() else None,
                incomplete_bars if not incomplete_bars.is_empty() else None,
            )
        else:
            return converted, None

    def validate_data(
        self,
        df: pl.DataFrame,
        timestamp_col: str = "timestamp",
        price_cols: dict[str, str] | None = None,
    ) -> list[str]:
        """
        入力データの検証

        Args:
            df: 検証するデータフレーム
            timestamp_col: タイムスタンプカラム名
            price_cols: 価格カラムのマッピング

        Returns:
            検証エラーメッセージのリスト（エラーがない場合は空リスト）
        """
        errors = []

        # 空のデータフレームチェック
        if df.is_empty():
            errors.append("DataFrame is empty")
            return errors

        # タイムスタンプカラムの存在チェック
        if timestamp_col not in df.columns:
            errors.append(f"Timestamp column '{timestamp_col}' not found")

        # 価格カラムのチェック
        if price_cols:
            for col_type, col_name in price_cols.items():
                if col_name not in df.columns:
                    errors.append(f"Price column '{col_name}' ({col_type}) not found")
                elif col_name in df.columns:
                    # NaN/NULL値のチェック
                    null_count = df[col_name].null_count()
                    if null_count > 0:
                        errors.append(
                            f"Column '{col_name}' contains {null_count} null values"
                        )

        # タイムスタンプの重複チェック
        if timestamp_col in df.columns:
            duplicates = df.filter(df.duplicated(subset=[timestamp_col]))
            if len(duplicates) > 0:
                errors.append(f"Found {len(duplicates)} duplicate timestamps")

        return errors

    def get_info(self) -> dict[str, Any]:
        """
        コンバータの設定情報を取得

        Returns:
            設定情報の辞書
        """
        return {
            "source_timeframe": self.source_timeframe,
            "target_timeframe": self.target_timeframe,
            "target_description": self.SUPPORTED_TIMEFRAMES.get(self.target_timeframe),
            "align_to_boundary": self.align_to_boundary,
            "polars_interval": self._interval,
            "expected_bars_per_period": self._get_expected_bars_per_period(),
        }
