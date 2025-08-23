"""
テクニカル指標計算エンジン

Polarsの組み込み機能を使用して高速なテクニカル指標計算を提供します。
Float32型での処理によりメモリ効率を最適化します。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

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

        # メタデータの初期化
        self._metadata: dict[str, Any] = {
            "indicators": {},
            "statistics": {
                "total_rows_processed": 0,
                "total_processing_time": 0.0,
                "last_update": None,
            },
        }

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

        # メタデータの更新
        self._update_metadata(
            "ema",
            {
                "calculated": True,
                "periods": self.ema_periods,
                "timestamp": datetime.now().isoformat(),
            },
            len(df),
        )

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
        
        # メタデータの更新
        self._update_metadata(
            "rsi",
            {
                "calculated": True,
                "period": period,
                "timestamp": datetime.now().isoformat(),
            },
            len(df),
        )
        
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
                .alias("macd_signal")
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
                .alias("macd_signal")
            )
        
        # MACD Histogramを計算
        result = result.with_columns(
            (pl.col("macd_line") - pl.col("macd_signal"))
            .cast(pl.Float32)
            .alias("macd_histogram")
        )
        
        # 一時列を削除
        result = result.drop(["ema_fast", "ema_slow"])
        
        logger.debug(
            f"Calculated MACD with periods: fast={fast_period}, "
            f"slow={slow_period}, signal={signal_period}"
        )
        
        # メタデータの更新
        self._update_metadata(
            "macd",
            {
                "calculated": True,
                "params": {
                    "fast": fast_period,
                    "slow": slow_period,
                    "signal": signal_period,
                },
                "timestamp": datetime.now().isoformat(),
            },
            len(df),
        )
        
        return result

    def calculate_bollinger_bands(
        self,
        df: pl.DataFrame,
        period: int = 20,
        num_std: float = 2.0,
        price_column: str = "close",
        group_by: str | None = None,
    ) -> pl.DataFrame:
        """
        ボリンジャーバンドを計算します。
        
        ボリンジャーバンドは価格のボラティリティを視覚化し、
        サポート・レジスタンスレベルの特定に使用されます。
        
        計算式:
        - Middle Band = SMA(period)
        - Upper Band = Middle Band + (num_std × 標準偏差)
        - Lower Band = Middle Band - (num_std × 標準偏差)
        - Band Width = Upper Band - Lower Band
        - %B = (Close - Lower Band) / (Upper Band - Lower Band)
        
        Args:
            df: 価格データを含むDataFrame
            period: 移動平均の期間（デフォルト: 20）
            num_std: 標準偏差の倍数（デフォルト: 2.0）
            price_column: 価格列名（デフォルト: "close"）
            group_by: グループ化する列名（複数シンボル対応）
        
        Returns:
            ボリンジャーバンド列が追加されたDataFrame
            - bb_upper: 上部バンド
            - bb_middle: 中央バンド（SMA）
            - bb_lower: 下部バンド
            - bb_width: バンド幅
            - bb_percent: %B（バンド内での価格位置）
        
        Raises:
            ValueError: 無効なパラメータが指定された場合
        """
        # パラメータの検証
        if period <= 0:
            raise ValueError(f"期間は正の値である必要があります: {period}")
        if num_std <= 0:
            raise ValueError(f"標準偏差倍数は正の値である必要があります: {num_std}")
        if price_column not in df.columns:
            raise ValueError(f"価格列が見つかりません: {price_column}")
        
        # Float32への変換
        if df[price_column].dtype != pl.Float32:
            df = df.with_columns(pl.col(price_column).cast(pl.Float32))
        
        # グループ化が必要な場合
        if group_by and group_by in df.columns:
            # グループごとに計算
            result = df.with_columns([
                # 中央バンド（SMA）
                pl.col(price_column)
                .rolling_mean(window_size=period)
                .over(group_by)
                .alias("bb_middle")
                .cast(pl.Float32),
                
                # 標準偏差
                pl.col(price_column)
                .rolling_std(window_size=period)
                .over(group_by)
                .alias("bb_std")
                .cast(pl.Float32),
            ])
        else:
            # 全体で計算
            result = df.with_columns([
                # 中央バンド（SMA）
                pl.col(price_column)
                .rolling_mean(window_size=period)
                .alias("bb_middle")
                .cast(pl.Float32),
                
                # 標準偏差
                pl.col(price_column)
                .rolling_std(window_size=period)
                .alias("bb_std")
                .cast(pl.Float32),
            ])
        
        # 上部バンドと下部バンドを計算
        result = result.with_columns([
            # 上部バンド
            (pl.col("bb_middle") + (pl.col("bb_std") * num_std))
            .alias("bb_upper")
            .cast(pl.Float32),
            
            # 下部バンド
            (pl.col("bb_middle") - (pl.col("bb_std") * num_std))
            .alias("bb_lower")
            .cast(pl.Float32),
        ])
        
        # バンド幅と%Bを計算
        result = result.with_columns([
            # バンド幅
            (pl.col("bb_upper") - pl.col("bb_lower"))
            .alias("bb_width")
            .cast(pl.Float32),
            
            # %B（バンド内での価格位置）
            # 0 = 下部バンド、0.5 = 中央バンド、1 = 上部バンド
            pl.when(pl.col("bb_upper") != pl.col("bb_lower"))
            .then(
                (pl.col(price_column) - pl.col("bb_lower")) / 
                (pl.col("bb_upper") - pl.col("bb_lower"))
            )
            .otherwise(0.5)  # バンド幅が0の場合は中央
            .alias("bb_percent")
            .cast(pl.Float32),
        ])
        
        # 一時列を削除
        result = result.drop("bb_std")
        
        logger.debug(
            f"Calculated Bollinger Bands with period={period}, "
            f"num_std={num_std}, group_by={group_by}"
        )
        
        # メタデータの更新
        self._update_metadata(
            "bollinger",
            {
                "calculated": True,
                "params": {
                    "period": period,
                    "num_std": num_std,
                },
                "timestamp": datetime.now().isoformat(),
            },
            len(df),
        )
        
        return result

    def calculate_all_indicators(
        self,
        df: pl.DataFrame,
        price_column: str = "close",
        volume_column: str | None = None,
        group_by: str | None = None,
        include_indicators: list[str] | None = None,
        ema_periods: list[int] | None = None,
        rsi_period: int = 14,
        macd_params: dict | None = None,
        bollinger_params: dict | None = None,
    ) -> pl.DataFrame:
        """
        全指標を効率的に一括計算します。
        
        共通計算の再利用とメモリ最適化により、個別計算よりも高速に処理します。
        
        Args:
            df: 入力データフレーム
            price_column: 価格列の名前（デフォルト: "close"）
            volume_column: ボリューム列の名前（任意）
            group_by: グループ化する列名（複数シンボル対応）
            include_indicators: 計算する指標のリスト
                              （None の場合は全指標を計算）
                              例: ["ema", "rsi", "macd", "bollinger"]
            ema_periods: EMA計算期間のリスト
            rsi_period: RSI計算期間
            macd_params: MACDパラメータ {"fast": 12, "slow": 26, "signal": 9}
            bollinger_params: ボリンジャーバンドパラメータ {"period": 20, "num_std": 2.0}
        
        Returns:
            全指標が追加されたDataFrame
        
        Raises:
            ValueError: 無効なデータが渡された場合
        """
        # 開始時刻を記録
        start_time = time.time()
        
        # データ検証
        if df.is_empty():
            raise ValueError("空のDataFrameが渡されました")
        
        if price_column not in df.columns:
            raise ValueError(f"{price_column}列が必要です")
        
        # Float32への変換（メモリ効率）
        if df[price_column].dtype != pl.Float32:
            df = df.with_columns(pl.col(price_column).cast(pl.Float32))
        
        # デフォルト値の設定
        if include_indicators is None:
            include_indicators = ["ema", "rsi", "macd", "bollinger"]
        
        if ema_periods is None:
            ema_periods = self.ema_periods
        
        if macd_params is None:
            macd_params = {"fast": 12, "slow": 26, "signal": 9}
        
        if bollinger_params is None:
            bollinger_params = {"period": 20, "num_std": 2.0}
        
        # 結果DataFrame（最初はコピー）
        result = df.clone()
        
        # ===== 共通計算の最適化 =====
        # EMAの事前計算（他の指標でも使用される可能性があるため）
        ema_cache = {}
        
        # EMA計算が必要な全ての期間を収集
        required_ema_periods = set()
        
        if "ema" in include_indicators:
            required_ema_periods.update(ema_periods)
        
        if "macd" in include_indicators:
            required_ema_periods.add(macd_params["fast"])
            required_ema_periods.add(macd_params["slow"])
        
        # 必要なEMAを一括計算してキャッシュ
        if group_by:
            # グループ処理の場合、partition_byを使用して各グループごとに処理
            for period in sorted(required_ema_periods):
                col_name = f"_ema_cache_{period}"
                result = result.with_columns(
                    pl.col(price_column)
                    .ewm_mean(span=period, adjust=False)
                    .over(group_by)
                    .cast(pl.Float32)
                    .alias(col_name)
                )
                ema_cache[period] = pl.col(col_name)
                logger.debug(f"Pre-calculated EMA for period {period}")
        else:
            # グループなしの場合、通常の処理
            for period in sorted(required_ema_periods):
                col_name = f"_ema_cache_{period}"
                ema_expr = (
                    pl.col(price_column)
                    .ewm_mean(span=period, adjust=False)
                    .cast(pl.Float32)
                )
                ema_cache[period] = ema_expr
                logger.debug(f"Pre-calculated EMA for period {period}")
        
        # ===== 各指標の計算（最適化済み） =====
        
        # 1. EMA（指数移動平均）
        if "ema" in include_indicators:
            ema_columns = {}
            for period in ema_periods:
                col_name = f"ema_{period}"
                if period in ema_cache:
                    # キャッシュから取得
                    ema_columns[col_name] = ema_cache[period].alias(col_name)
                else:
                    # 新規計算
                    if group_by:
                        ema_columns[col_name] = (
                            pl.col(price_column)
                            .ewm_mean(span=period, adjust=False)
                            .over(group_by)
                            .cast(pl.Float32)
                            .alias(col_name)
                        )
                    else:
                        ema_columns[col_name] = (
                            pl.col(price_column)
                            .ewm_mean(span=period, adjust=False)
                            .cast(pl.Float32)
                            .alias(col_name)
                        )
            
            # 一括で列を追加（効率的）
            result = result.with_columns(list(ema_columns.values()))
            
            # メタデータ更新
            self._update_metadata(
                "ema",
                {
                    "calculated": True,
                    "periods": ema_periods,
                    "timestamp": datetime.now().isoformat(),
                },
                len(df),
            )
            logger.info(f"Calculated EMA for {len(ema_periods)} periods")
        
        # 2. RSI（相対力指数）
        if "rsi" in include_indicators:
            # RSI計算
            alpha = 1.0 / rsi_period
            
            if group_by:
                # グループごとに価格変化を計算
                result = result.with_columns(
                    pl.col(price_column).diff().over(group_by).alias("_price_change")
                )
                
                # 上昇幅と下落幅を分離
                result = result.with_columns([
                    pl.when(pl.col("_price_change") > 0)
                    .then(pl.col("_price_change"))
                    .otherwise(0.0)
                    .alias("_gain"),
                    
                    pl.when(pl.col("_price_change") < 0)
                    .then(-pl.col("_price_change"))
                    .otherwise(0.0)
                    .alias("_loss"),
                ])
                
                # EMAでの平均を計算
                result = result.with_columns([
                    pl.col("_gain")
                    .ewm_mean(alpha=alpha, adjust=False)
                    .over(group_by)
                    .alias("_avg_gain"),
                    
                    pl.col("_loss")
                    .ewm_mean(alpha=alpha, adjust=False)
                    .over(group_by)
                    .alias("_avg_loss"),
                ])
                
                # 一時列を削除
                result = result.drop(["_price_change", "_gain", "_loss"])
            else:
                # グループなしの場合
                result = result.with_columns(
                    pl.col(price_column).diff().alias("_price_change")
                )
                
                # 上昇幅と下落幅を分離
                result = result.with_columns([
                    pl.when(pl.col("_price_change") > 0)
                    .then(pl.col("_price_change"))
                    .otherwise(0.0)
                    .alias("_gain"),
                    
                    pl.when(pl.col("_price_change") < 0)
                    .then(-pl.col("_price_change"))
                    .otherwise(0.0)
                    .alias("_loss"),
                ])
                
                # EMAでの平均を計算
                result = result.with_columns([
                    pl.col("_gain")
                    .ewm_mean(alpha=alpha, adjust=False)
                    .alias("_avg_gain"),
                    
                    pl.col("_loss")
                    .ewm_mean(alpha=alpha, adjust=False)
                    .alias("_avg_loss"),
                ])
                
                # 一時列を削除
                result = result.drop(["_price_change", "_gain", "_loss"])
            
            result = result.with_columns([
                pl.when(pl.col("_avg_loss") != 0)
                .then(100 - (100 / (1 + (pl.col("_avg_gain") / pl.col("_avg_loss")))))
                .otherwise(50.0)
                .alias(f"rsi_{rsi_period}")
                .cast(pl.Float32)
            ])
            
            # 一時列を削除（メモリ最適化）
            result = result.drop(["_avg_gain", "_avg_loss"])
            
            # メタデータ更新
            self._update_metadata(
                "rsi",
                {
                    "calculated": True,
                    "period": rsi_period,
                    "timestamp": datetime.now().isoformat(),
                },
                len(df),
            )
            logger.info(f"Calculated RSI with period {rsi_period}")
        
        # 3. MACD（移動平均収束拡散）
        if "macd" in include_indicators:
            fast_period = macd_params["fast"]
            slow_period = macd_params["slow"]
            signal_period = macd_params["signal"]
            
            # キャッシュからEMAを取得または計算
            if fast_period in ema_cache:
                ema_fast = ema_cache[fast_period]
            else:
                if group_by:
                    ema_fast = (
                        pl.col(price_column)
                        .ewm_mean(span=fast_period, adjust=False)
                        .over(group_by)
                        .cast(pl.Float32)
                    )
                else:
                    ema_fast = (
                        pl.col(price_column)
                        .ewm_mean(span=fast_period, adjust=False)
                        .cast(pl.Float32)
                    )
            
            if slow_period in ema_cache:
                ema_slow = ema_cache[slow_period]
            else:
                if group_by:
                    ema_slow = (
                        pl.col(price_column)
                        .ewm_mean(span=slow_period, adjust=False)
                        .over(group_by)
                        .cast(pl.Float32)
                    )
                else:
                    ema_slow = (
                        pl.col(price_column)
                        .ewm_mean(span=slow_period, adjust=False)
                        .cast(pl.Float32)
                    )
            
            # MACD計算
            result = result.with_columns([
                (ema_fast - ema_slow).alias("macd_line").cast(pl.Float32)
            ])
            
            # Signal Line計算
            if group_by:
                result = result.with_columns([
                    pl.col("macd_line")
                    .ewm_mean(span=signal_period, adjust=False)
                    .over(group_by)
                    .cast(pl.Float32)
                    .alias("macd_signal")
                ])
            else:
                result = result.with_columns([
                    pl.col("macd_line")
                    .ewm_mean(span=signal_period, adjust=False)
                    .cast(pl.Float32)
                    .alias("macd_signal")
                ])
            
            # Histogram計算
            result = result.with_columns([
                (pl.col("macd_line") - pl.col("macd_signal"))
                .alias("macd_histogram")
                .cast(pl.Float32)
            ])
            
            # メタデータ更新
            self._update_metadata(
                "macd",
                {
                    "calculated": True,
                    "params": {
                        "fast": fast_period,
                        "slow": slow_period,
                        "signal": signal_period,
                    },
                    "timestamp": datetime.now().isoformat(),
                },
                len(df),
            )
            logger.info(f"Calculated MACD with periods {fast_period}/{slow_period}/{signal_period}")
        
        # 4. ボリンジャーバンド
        if "bollinger" in include_indicators:
            period = bollinger_params["period"]
            num_std = bollinger_params["num_std"]
            
            # 中央バンドと標準偏差を同時に計算
            if group_by:
                result = result.with_columns([
                    pl.col(price_column)
                    .rolling_mean(window_size=period)
                    .over(group_by)
                    .cast(pl.Float32)
                    .alias("bb_middle"),
                    
                    pl.col(price_column)
                    .rolling_std(window_size=period)
                    .over(group_by)
                    .cast(pl.Float32)
                    .alias("_bb_std")
                ])
            else:
                result = result.with_columns([
                    pl.col(price_column)
                    .rolling_mean(window_size=period)
                    .cast(pl.Float32)
                    .alias("bb_middle"),
                    
                    pl.col(price_column)
                    .rolling_std(window_size=period)
                    .cast(pl.Float32)
                    .alias("_bb_std")
                ])
            
            # 上部・下部バンド、バンド幅、%Bを一括計算
            result = result.with_columns([
                (pl.col("bb_middle") + (pl.col("_bb_std") * num_std))
                .alias("bb_upper")
                .cast(pl.Float32),
                
                (pl.col("bb_middle") - (pl.col("_bb_std") * num_std))
                .alias("bb_lower")
                .cast(pl.Float32),
            ])
            
            result = result.with_columns([
                (pl.col("bb_upper") - pl.col("bb_lower"))
                .alias("bb_width")
                .cast(pl.Float32),
                
                pl.when(pl.col("bb_upper") != pl.col("bb_lower"))
                .then(
                    (pl.col(price_column) - pl.col("bb_lower")) / 
                    (pl.col("bb_upper") - pl.col("bb_lower"))
                )
                .otherwise(0.5)
                .alias("bb_percent")
                .cast(pl.Float32),
            ])
            
            # 一時列を削除（メモリ最適化）
            result = result.drop("_bb_std")
            
            # メタデータ更新
            self._update_metadata(
                "bollinger",
                {
                    "calculated": True,
                    "params": {
                        "period": period,
                        "num_std": num_std,
                    },
                    "timestamp": datetime.now().isoformat(),
                },
                len(df),
            )
            logger.info(f"Calculated Bollinger Bands with period {period}")
        
        # EMAキャッシュ列を削除（メモリ最適化）
        cache_columns = [col for col in result.columns if col.startswith("_ema_cache_")]
        if cache_columns:
            result = result.drop(cache_columns)
            logger.debug(f"Dropped {len(cache_columns)} EMA cache columns")
        
        # 処理時間を記録
        elapsed_time = time.time() - start_time
        self._last_process_time = elapsed_time
        
        # 統計情報の更新
        stats = self._metadata["statistics"]
        stats["total_processing_time"] += elapsed_time
        
        logger.info(
            f"Batch calculation completed: {len(include_indicators)} indicators, "
            f"{len(df)} rows, {elapsed_time:.4f} seconds"
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
            elif indicator.lower() == "bollinger":
                result = self.calculate_bollinger_bands(result)
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

    def _update_metadata(
        self, indicator_name: str, indicator_info: dict[str, Any], rows_processed: int
    ) -> None:
        """
        メタデータを更新します（内部使用）。

        Args:
            indicator_name: 指標名
            indicator_info: 指標情報
            rows_processed: 処理した行数
        """
        # 処理時間の計測（簡易的な実装）
        if not hasattr(self, "_last_process_time"):
            self._last_process_time = 0.0

        # 指標メタデータの更新
        self._metadata["indicators"][indicator_name] = indicator_info

        # 統計情報の更新
        stats = self._metadata["statistics"]
        stats["total_rows_processed"] += rows_processed
        stats["total_processing_time"] += self._last_process_time
        stats["last_update"] = datetime.now().isoformat()

    def get_metadata(self) -> dict[str, Any]:
        """
        現在のメタデータを取得します。

        Returns:
            メタデータのディクショナリ
            - indicators: 計算済み指標の情報
            - statistics: 処理統計情報
        """
        return self._metadata.copy()

    def clear_metadata(self) -> None:
        """
        メタデータをクリアします。
        """
        self._metadata = {
            "indicators": {},
            "statistics": {
                "total_rows_processed": 0,
                "total_processing_time": 0.0,
                "last_update": None,
            },
        }
        logger.info("Metadata cleared")

    def get_calculated_indicators(self) -> list[str]:
        """
        計算済みの指標名のリストを取得します。

        Returns:
            計算済み指標名のリスト
        """
        return list(self._metadata["indicators"].keys())

    def is_indicator_calculated(self, indicator_name: str) -> bool:
        """
        特定の指標が計算済みかどうかを確認します。

        Args:
            indicator_name: 確認する指標名

        Returns:
            計算済みの場合True
        """
        return indicator_name in self._metadata["indicators"]

    def get_indicator_params(self, indicator_name: str) -> dict[str, Any] | None:
        """
        特定の指標で使用されたパラメータを取得します。

        Args:
            indicator_name: 指標名

        Returns:
            パラメータの辞書、または指標が計算されていない場合None
        """
        if indicator_name in self._metadata["indicators"]:
            indicator_info = self._metadata["indicators"][indicator_name]
            if "params" in indicator_info:
                return indicator_info["params"]
            elif "periods" in indicator_info:
                return {"periods": indicator_info["periods"]}
            elif "period" in indicator_info:
                return {"period": indicator_info["period"]}
        return None

    def get_processing_statistics(self) -> dict[str, Any]:
        """
        処理統計情報を取得します。

        Returns:
            統計情報の辞書
        """
        return self._metadata["statistics"].copy()
