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

    def calculate_rsi(
        self,
        df: pl.DataFrame,
        period: int = 14,
        price_column: str = "close",
        group_by: str | None = None,
    ) -> pl.DataFrame:
        """
        RSI（相対力指数）を計算します。
        
        RSIは0〜100の範囲で、価格の上昇・下落の勢いを測る指標です。
        - RSI > 70: 買われすぎ
        - RSI < 30: 売られすぎ
        
        計算式:
        1. 価格変化 = 現在価格 - 前回価格
        2. 上昇幅 = 価格変化 (価格変化 > 0の場合)、0 (それ以外)
        3. 下落幅 = |価格変化| (価格変化 < 0の場合)、0 (それ以外)
        4. 平均上昇幅 = 上昇幅のEMA(period)
        5. 平均下落幅 = 下落幅のEMA(period)
        6. RS = 平均上昇幅 / 平均下落幅
        7. RSI = 100 - (100 / (1 + RS))
        
        Args:
            df: 入力データフレーム
            period: RSI計算期間（デフォルト: 14）
            price_column: 価格列の名前（デフォルト: "close"）
            group_by: グループ化する列名（複数シンボル対応）
        
        Returns:
            RSI列が追加されたDataFrame
        
        Raises:
            ValueError: 無効なデータが渡された場合
        """
        # データ検証
        if df.is_empty():
            raise ValueError("空のDataFrameが渡されました")
        
        if price_column not in df.columns:
            raise ValueError(f"{price_column}列が必要です")
        
        if period <= 0:
            raise ValueError("期間は正の整数である必要があります")
        
        # Float32への変換
        if df[price_column].dtype != pl.Float32:
            df = df.with_columns(pl.col(price_column).cast(pl.Float32))
        
        result = df
        col_name = f"rsi_{period}"
        
        if group_by:
            # グループごとにRSIを計算
            # 価格変化を計算
            result = result.with_columns(
                pl.col(price_column)
                .diff()
                .over(group_by)
                .alias("price_change")
            )
            
            # 上昇幅と下落幅を分離
            result = result.with_columns([
                pl.when(pl.col("price_change") > 0)
                .then(pl.col("price_change"))
                .otherwise(0.0)
                .alias("gain"),
                
                pl.when(pl.col("price_change") < 0)
                .then(-pl.col("price_change"))  # 絶対値
                .otherwise(0.0)
                .alias("loss"),
            ])
            
            # EMA（指数移動平均）で平均上昇幅と平均下落幅を計算
            # alpha = 1 / period （RSIの標準的な計算方法）
            alpha = 1.0 / period
            
            result = result.with_columns([
                pl.col("gain")
                .ewm_mean(alpha=alpha, adjust=False)
                .over(group_by)
                .alias("avg_gain"),
                
                pl.col("loss")
                .ewm_mean(alpha=alpha, adjust=False)
                .over(group_by)
                .alias("avg_loss"),
            ])
        else:
            # 全体でRSIを計算
            # 価格変化を計算
            result = result.with_columns(
                pl.col(price_column).diff().alias("price_change")
            )
            
            # 上昇幅と下落幅を分離
            result = result.with_columns([
                pl.when(pl.col("price_change") > 0)
                .then(pl.col("price_change"))
                .otherwise(0.0)
                .alias("gain"),
                
                pl.when(pl.col("price_change") < 0)
                .then(-pl.col("price_change"))  # 絶対値
                .otherwise(0.0)
                .alias("loss"),
            ])
            
            # EMA（指数移動平均）で平均上昇幅と平均下落幅を計算
            alpha = 1.0 / period
            
            result = result.with_columns([
                pl.col("gain")
                .ewm_mean(alpha=alpha, adjust=False)
                .alias("avg_gain"),
                
                pl.col("loss")
                .ewm_mean(alpha=alpha, adjust=False)
                .alias("avg_loss"),
            ])
        
        # RSIを計算
        # RS = avg_gain / avg_loss
        # RSI = 100 - (100 / (1 + RS))
        # avg_lossが0の場合の処理も含める
        result = result.with_columns(
            pl.when((pl.col("avg_loss") == 0) & (pl.col("avg_gain") == 0))
            .then(50.0)  # 価格変化がない場合はRSI=50（中立）
            .when(pl.col("avg_loss") == 0)
            .then(100.0)  # 下落がない場合はRSI=100
            .when(pl.col("avg_gain") == 0)
            .then(0.0)  # 上昇がない場合はRSI=0
            .otherwise(
                100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss"))))
            )
            .cast(pl.Float32)
            .alias(col_name)
        )
        
        # 一時列を削除
        result = result.drop(["price_change", "gain", "loss", "avg_gain", "avg_loss"])
        
        logger.debug(f"Calculated RSI with period {period}")
        return result

    def calculate_macd(
        self,
        df: pl.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_column: str = "close",
        group_by: str | None = None,
    ) -> pl.DataFrame:
        """
        MACD（移動平均収束拡散）を計算します。
        
        MACDは2つのEMAの差を使用してトレンドの方向と強さを測る指標です。
        - MACD Line: 短期EMA - 長期EMA
        - Signal Line: MACD LineのEMA
        - MACD Histogram: MACD Line - Signal Line
        
        計算式:
        1. MACD Line = EMA(fast_period) - EMA(slow_period)
        2. Signal Line = EMA(MACD Line, signal_period)
        3. MACD Histogram = MACD Line - Signal Line
        
        Args:
            df: 入力データフレーム
            fast_period: 短期EMA期間（デフォルト: 12）
            slow_period: 長期EMA期間（デフォルト: 26）
            signal_period: シグナルラインのEMA期間（デフォルト: 9）
            price_column: 価格列の名前（デフォルト: "close"）
            group_by: グループ化する列名（複数シンボル対応）
        
        Returns:
            MACD関連列が追加されたDataFrame
        
        Raises:
            ValueError: 無効なデータが渡された場合
        """
        # データ検証
        if df.is_empty():
            raise ValueError("空のDataFrameが渡されました")
        
        if price_column not in df.columns:
            raise ValueError(f"{price_column}列が必要です")
        
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("期間は正の整数である必要があります")
        
        if fast_period >= slow_period:
            raise ValueError("短期期間は長期期間より小さい必要があります")
        
        # Float32への変換
        if df[price_column].dtype != pl.Float32:
            df = df.with_columns(pl.col(price_column).cast(pl.Float32))
        
        result = df
        
        if group_by:
            # グループごとにMACDを計算
            # 短期EMAと長期EMAを計算
            result = result.with_columns([
                pl.col(price_column)
                .ewm_mean(span=fast_period, adjust=False)
                .over(group_by)
                .cast(pl.Float32)
                .alias("ema_fast"),
                
                pl.col(price_column)
                .ewm_mean(span=slow_period, adjust=False)
                .over(group_by)
                .cast(pl.Float32)
                .alias("ema_slow"),
            ])
            
            # MACD Lineを計算
            result = result.with_columns(
                (pl.col("ema_fast") - pl.col("ema_slow"))
                .cast(pl.Float32)
                .alias("macd_line")
            )
            
            # Signal Line（MACD LineのEMA）を計算
            result = result.with_columns(
                pl.col("macd_line")
                .ewm_mean(span=signal_period, adjust=False)
                .over(group_by)
                .cast(pl.Float32)
                .alias("signal_line")
            )
        else:
            # 全体でMACDを計算
            # 短期EMAと長期EMAを計算
            result = result.with_columns([
                pl.col(price_column)
                .ewm_mean(span=fast_period, adjust=False)
                .cast(pl.Float32)
                .alias("ema_fast"),
                
                pl.col(price_column)
                .ewm_mean(span=slow_period, adjust=False)
                .cast(pl.Float32)
                .alias("ema_slow"),
            ])
            
            # MACD Lineを計算
            result = result.with_columns(
                (pl.col("ema_fast") - pl.col("ema_slow"))
                .cast(pl.Float32)
                .alias("macd_line")
            )
            
            # Signal Line（MACD LineのEMA）を計算
            result = result.with_columns(
                pl.col("macd_line")
                .ewm_mean(span=signal_period, adjust=False)
                .cast(pl.Float32)
                .alias("signal_line")
            )
        
        # MACD Histogramを計算
        result = result.with_columns(
            (pl.col("macd_line") - pl.col("signal_line"))
            .cast(pl.Float32)
            .alias("macd_histogram")
        )
        
        # 一時列を削除
        result = result.drop(["ema_fast", "ema_slow"])
        
        logger.debug(
            f"Calculated MACD with periods: fast={fast_period}, "
            f"slow={slow_period}, signal={signal_period}"
        )
        return result

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
            elif indicator.lower() == "rsi":
                result = self.calculate_rsi(result)
            elif indicator.lower() == "macd":
                result = self.calculate_macd(result)
            # 将来的に他の指標を追加
            # elif indicator.lower() == "bollinger":
            #     result = self.calculate_bollinger_bands(result)
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
