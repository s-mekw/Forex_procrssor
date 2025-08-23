"""
テクニカル指標計算エンジン

Polarsの組み込み機能を使用して高速なテクニカル指標計算を提供します。
Float32型での処理によりメモリ効率を最適化します。
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


class TechnicalIndicatorEngine:
    """
    テクニカル指標計算エンジン

    Polarsのネイティブ機能を使用してEMA（指数移動平均）などの
    テクニカル指標を高速に計算します。
    """

    def __init__(self, ema_periods: list[int] | None = None) -> None:
        """
        エンジンを初期化します。

        Args:
            ema_periods: EMA計算期間のリスト（デフォルト: [5, 20, 50, 100, 200]）

        Raises:
            ValueError: 無効な期間が指定された場合
        """
        if ema_periods is None:
            self.ema_periods = [5, 20, 50, 100, 200]
        else:
            # 期間の検証
            if not all(isinstance(p, int) and p > 0 for p in ema_periods):
                raise ValueError("期間は正の整数である必要があります")
            self.ema_periods = ema_periods

        logger.info(
            f"TechnicalIndicatorEngine initialized with periods: {self.ema_periods}"
        )

    def calculate_ema(
        self, df: pl.DataFrame, price_column: str = "close", group_by: str | None = None
    ) -> pl.DataFrame:
        """
        EMA（指数移動平均）を計算します。

        Args:
            df: 入力データフレーム
            price_column: 価格列の名前（デフォルト: "close"）
            group_by: グループ化する列名（複数シンボル対応）

        Returns:
            EMA列が追加されたDataFrame

        Raises:
            ValueError: 無効なデータが渡された場合
        """
        # データ検証
        if df.is_empty():
            raise ValueError("空のDataFrameが渡されました")

        if price_column not in df.columns:
            raise ValueError(f"{price_column}列が必要です")

        # Float32への変換
        if df[price_column].dtype != pl.Float32:
            df = df.with_columns(pl.col(price_column).cast(pl.Float32))

        result = df

        # 各期間のEMAを計算
        for period in self.ema_periods:
            col_name = f"ema_{period}"

            if group_by:
                # グループごとにEMAを計算
                result = result.with_columns(
                    pl.col(price_column)
                    .ewm_mean(span=period, adjust=False)
                    .over(group_by)
                    .cast(pl.Float32)
                    .alias(col_name)
                )
            else:
                # 全体でEMAを計算
                result = result.with_columns(
                    pl.col(price_column)
                    .ewm_mean(span=period, adjust=False)
                    .cast(pl.Float32)
                    .alias(col_name)
                )

            logger.debug(f"Calculated EMA with period {period}")

        return result

    def update_ema_incremental(
        self,
        existing_df: pl.DataFrame,
        new_data: pl.DataFrame,
        price_column: str = "close",
    ) -> pl.DataFrame:
        """
        既存のEMAデータに新しいデータを増分的に追加します。

        Args:
            existing_df: 既存のEMA計算済みデータ
            new_data: 新しいデータ
            price_column: 価格列の名前

        Returns:
            更新されたDataFrame
        """
        # 新しいデータの検証
        if new_data.is_empty():
            logger.warning("新しいデータが空です")
            return existing_df

        # 新しいデータにEMA列を追加（初期値はNULL）
        for period in self.ema_periods:
            col_name = f"ema_{period}"
            if col_name not in new_data.columns:
                new_data = new_data.with_columns(
                    pl.lit(None).cast(pl.Float32).alias(col_name)
                )

        # データを結合
        combined = pl.concat([existing_df, new_data], how="vertical")

        # EMAを再計算（効率的な方法）
        # 既存のEMA値を考慮して、新しいデータポイントのみを更新
        for period in self.ema_periods:
            col_name = f"ema_{period}"

            if col_name in existing_df.columns:
                # 既存のEMA列がある場合、最後の値から継続計算
                # ウィンドウサイズを制限して効率化
                window_size = min(len(combined), period * 3)  # 期間の3倍まで

                # 最後のwindow_sizeの行だけを使って再計算
                if len(combined) > window_size:
                    # 古いデータは保持、新しい部分だけ再計算
                    old_part = combined[: len(combined) - window_size]
                    new_part = combined[len(combined) - window_size :]

                    new_part = new_part.with_columns(
                        pl.col(price_column)
                        .ewm_mean(span=period, adjust=False)
                        .cast(pl.Float32)
                        .alias(col_name)
                    )

                    combined = pl.concat([old_part, new_part], how="vertical")
                else:
                    # 全体を再計算
                    combined = combined.with_columns(
                        pl.col(price_column)
                        .ewm_mean(span=period, adjust=False)
                        .cast(pl.Float32)
                        .alias(col_name)
                    )
            else:
                # 新規計算
                combined = combined.with_columns(
                    pl.col(price_column)
                    .ewm_mean(span=period, adjust=False)
                    .cast(pl.Float32)
                    .alias(col_name)
                )

        logger.info(f"Incrementally updated EMA for {len(new_data)} new rows")
        return combined

    def calculate_multiple_indicators(
        self, df: pl.DataFrame, indicators: list[str] | None = None
    ) -> pl.DataFrame:
        """
        複数のテクニカル指標を一括計算します。

        Args:
            df: 入力データフレーム
            indicators: 計算する指標のリスト

        Returns:
            指標が追加されたDataFrame
        """
        if indicators is None:
            indicators = ["ema"]

        result = df

        for indicator in indicators:
            if indicator.lower() == "ema":
                result = self.calculate_ema(result)
            # 将来的に他の指標を追加
            # elif indicator.lower() == "rsi":
            #     result = self.calculate_rsi(result)
            # elif indicator.lower() == "macd":
            #     result = self.calculate_macd(result)
            else:
                logger.warning(f"Unknown indicator: {indicator}")

        return result

    def validate_calculation(self, df: pl.DataFrame) -> bool:
        """
        計算結果の妥当性を検証します。

        Args:
            df: 検証するDataFrame

        Returns:
            検証が成功した場合True
        """
        # EMA列の存在確認
        ema_columns = [f"ema_{p}" for p in self.ema_periods]
        missing_columns = [col for col in ema_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing EMA columns: {missing_columns}")
            return False

        # 値の妥当性確認
        for col in ema_columns:
            if col in df.columns:
                # NaN/Inf のチェック
                if df[col].is_infinite().any() or df[col].is_nan().any():
                    logger.error(f"Invalid values (NaN/Inf) found in {col}")
                    return False

                # 範囲チェック（価格の妥当な範囲）
                values = df[col].drop_nulls()
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()

                    if min_val <= 0 or max_val > 10000:
                        logger.warning(
                            f"Suspicious values in {col}: min={min_val}, max={max_val}"
                        )

        return True
