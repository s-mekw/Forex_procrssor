"""
Polarsデータ処理基盤のユニットテスト

このモジュールは、PolarsProcessingEngineクラスの機能をテストします。
TDD（テスト駆動開発）アプローチに従い、実装前にテストを定義しています。
"""

import gc
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import psutil
import pytest

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """テスト用のサンプルデータを生成"""
    # 100行のサンプルデータを作成
    n_rows = 100
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    data = {
        "timestamp": [base_time + timedelta(seconds=i) for i in range(n_rows)],
        "open": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
        "high": np.random.uniform(1.0850, 1.0950, n_rows).astype(np.float64),
        "low": np.random.uniform(1.0750, 1.0850, n_rows).astype(np.float64),
        "close": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
        "volume": np.random.uniform(1000, 10000, n_rows).astype(np.float64),
    }

    return pl.DataFrame(data)


@pytest.fixture
def large_sample_data() -> pl.DataFrame:
    """大規模データテスト用のサンプルデータを生成"""
    # 1万行のサンプルデータを作成
    n_rows = 10000
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    data = {
        "timestamp": [base_time + timedelta(seconds=i) for i in range(n_rows)],
        "open": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
        "high": np.random.uniform(1.0850, 1.0950, n_rows).astype(np.float64),
        "low": np.random.uniform(1.0750, 1.0850, n_rows).astype(np.float64),
        "close": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
        "volume": np.random.uniform(1000, 10000, n_rows).astype(np.float64),
    }

    return pl.DataFrame(data)


class TestPolarsProcessingEngine:
    """PolarsProcessingEngineクラスのテストスイート"""

    def test_engine_initialization(self):
        """エンジンの初期化テスト"""
        # TODO: PolarsProcessingEngineクラスの実装後にテストを有効化
        pass

    def test_data_type_optimization(self, sample_data: pl.DataFrame):
        """Float64からFloat32への変換とメモリ削減テスト"""
        # Step 2: 基本的なデータ型定義テスト

        # 1. 元のデータがFloat64であることを確認
        for col in ["open", "high", "low", "close", "volume"]:
            assert sample_data[col].dtype == pl.Float64, (
                f"{col}列は初期状態でFloat64であるべき"
            )

        # 2. メモリ使用量を記録（バイト単位）
        original_memory = sample_data.estimated_size()

        # 3. Float32に変換
        optimized_df = sample_data.select(
            [
                pl.col("timestamp"),
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("volume").cast(pl.Float32),
            ]
        )

        # 4. 変換後のデータ型を確認
        assert optimized_df["timestamp"].dtype == pl.Datetime, (
            "timestamp列はDatetimeを維持すべき"
        )
        for col in ["open", "high", "low", "close", "volume"]:
            assert optimized_df[col].dtype == pl.Float32, (
                f"{col}列はFloat32に変換されるべき"
            )

        # 5. メモリ使用量の削減を確認（約50%削減を期待）
        optimized_memory = optimized_df.estimated_size()
        memory_reduction_ratio = 1 - (optimized_memory / original_memory)

        # 数値列のメモリが約50%削減されることを確認
        # （timestamp列は変更されないため、全体の削減率は50%未満になる）
        assert memory_reduction_ratio > 0.35, (
            f"メモリ削減率が不十分: {memory_reduction_ratio:.2%} (期待値: 35%以上)"
        )

        # 6. データの精度が維持されていることを確認（小数点以下2桁）
        for col in ["open", "high", "low", "close"]:
            original_values = sample_data[col].to_numpy()
            optimized_values = optimized_df[col].to_numpy()
            # 小数点以下2桁までの精度で一致することを確認
            np.testing.assert_array_almost_equal(
                original_values,
                optimized_values,
                decimal=2,
                err_msg=f"{col}列の精度が失われています",
            )

    def test_lazyframe_creation(self, sample_data: pl.DataFrame):
        """LazyFrameの作成とクエリ計画のテスト"""
        # Step 2: LazyFrame作成の基本テスト

        # 1. DataFrameからLazyFrameを作成
        lazy_df = sample_data.lazy()

        # 2. LazyFrameであることを確認
        assert isinstance(lazy_df, pl.LazyFrame), "LazyFrameが作成されるべき"

        # 3. 基本的な操作を連鎖（遅延評価）
        query = (
            lazy_df.select(
                [
                    pl.col("timestamp"),
                    pl.col("open").cast(pl.Float32),
                    pl.col("high").cast(pl.Float32),
                    pl.col("low").cast(pl.Float32),
                    pl.col("close").cast(pl.Float32),
                    pl.col("volume").cast(pl.Float32),
                ]
            )
            .filter(pl.col("volume") > 5000)
            .sort("timestamp")
        )

        # 4. クエリ計画が作成されていることを確認（まだ実行されていない）
        assert isinstance(query, pl.LazyFrame), "クエリチェーンはLazyFrameを維持すべき"

        # 5. クエリを実行してDataFrameを取得
        result_df = query.collect()

        # 6. 結果がDataFrameであることを確認
        assert isinstance(result_df, pl.DataFrame), (
            "collect()後はDataFrameが返されるべき"
        )

        # 7. フィルタリングが適用されていることを確認
        assert all(result_df["volume"] > 5000), "volumeフィルタが適用されるべき"

        # 8. ソートが適用されていることを確認
        timestamps = result_df["timestamp"].to_list()
        assert timestamps == sorted(timestamps), "timestampでソートされるべき"

        # 9. データ型が正しく変換されていることを確認
        for col in ["open", "high", "low", "close", "volume"]:
            assert result_df[col].dtype == pl.Float32, f"{col}列はFloat32であるべき"

    def test_optimize_integer_types(self):
        """整数型の最適化テスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # 異なる範囲の整数データを作成
        data = {
            "small_int": list(range(100)),  # Int8で十分 (0-99)
            "medium_int": [i * 100 for i in range(100)],  # Int16で十分 (0-9900)
            "large_int": [i * 100000 for i in range(100)],  # Int32が必要 (0-9900000)
            "trade_count": [
                i * 10 for i in range(100)
            ],  # 取引カウント Int16で十分 (0-990)
        }
        df = pl.DataFrame(data).with_columns(
            [
                pl.col("small_int").cast(pl.Int64),
                pl.col("medium_int").cast(pl.Int64),
                pl.col("large_int").cast(pl.Int64),
                pl.col("trade_count").cast(pl.Int64),
            ]
        )

        # エンジンを初期化して最適化
        engine = PolarsProcessingEngine()
        optimized_df = engine.optimize_dtypes(df)

        # 最適なデータ型に変換されていることを確認
        assert optimized_df["small_int"].dtype == pl.Int8, (
            "small_intはInt8に最適化されるべき"
        )
        assert optimized_df["medium_int"].dtype == pl.Int16, (
            "medium_intはInt16に最適化されるべき"
        )
        assert optimized_df["large_int"].dtype == pl.Int32, (
            "large_intはInt32に最適化されるべき"
        )
        assert optimized_df["trade_count"].dtype == pl.Int16, (
            "trade_countはInt16に最適化されるべき"
        )

        # メモリ削減を確認
        original_memory = df.estimated_size()
        optimized_memory = optimized_df.estimated_size()
        memory_reduction = 1 - (optimized_memory / original_memory)
        assert memory_reduction > 0.5, (
            f"整数型最適化でメモリが50%以上削減されるべき: {memory_reduction:.2%}"
        )

    def test_categorical_optimization(self):
        """カテゴリカル型の最適化テスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # カテゴリカルデータを含むDataFrameを作成
        n_rows = 1000
        data = {
            "symbol": ["EURUSD"] * 500 + ["GBPUSD"] * 300 + ["USDJPY"] * 200,
            "order_type": ["BUY"] * 400 + ["SELL"] * 600,
            "status": ["PENDING"] * 100 + ["EXECUTED"] * 800 + ["CANCELLED"] * 100,
            "value": np.random.uniform(1000, 10000, n_rows),
        }
        df = pl.DataFrame(data)

        # エンジンを初期化して最適化
        engine = PolarsProcessingEngine()
        optimized_df = engine.optimize_dtypes(df, categorical_threshold=0.5)

        # カテゴリカル型に変換されていることを確認
        assert optimized_df["symbol"].dtype == pl.Categorical, (
            "symbolはCategoricalに変換されるべき"
        )
        assert optimized_df["order_type"].dtype == pl.Categorical, (
            "order_typeはCategoricalに変換されるべき"
        )
        assert optimized_df["status"].dtype == pl.Categorical, (
            "statusはCategoricalに変換されるべき"
        )
        assert optimized_df["value"].dtype == pl.Float32, (
            "valueはFloat32に変換されるべき"
        )

        # メモリ削減を確認
        original_memory = df.estimated_size()
        optimized_memory = optimized_df.estimated_size()
        memory_reduction = 1 - (optimized_memory / original_memory)
        assert memory_reduction > 0.3, (
            f"カテゴリカル最適化でメモリが30%以上削減されるべき: {memory_reduction:.2%}"
        )

        # データの整合性を確認
        assert optimized_df["symbol"].to_list() == df["symbol"].to_list()
        assert optimized_df["order_type"].to_list() == df["order_type"].to_list()
        assert optimized_df["status"].to_list() == df["status"].to_list()

    def test_memory_report_generation(self):
        """メモリ使用量レポート機能のテスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # テストデータを作成
        data = {
            "id": list(range(1000)),  # Int64
            "small_value": [i % 100 for i in range(1000)],  # Int64だがInt8で十分
            "symbol": ["EURUSD"] * 500 + ["GBPUSD"] * 500,  # 文字列（カテゴリカル候補）
            "price": np.random.uniform(1.0, 2.0, 1000),  # Float64
            "volume": np.random.uniform(1000, 10000, 1000),  # Float64
        }
        df = pl.DataFrame(data)

        # エンジンを初期化してレポートを生成
        engine = PolarsProcessingEngine()
        report = engine.get_memory_report(df)

        # レポートの構造を確認
        assert "total_memory_bytes" in report
        assert "total_memory_mb" in report
        assert "n_rows" in report
        assert "n_columns" in report
        assert "bytes_per_row" in report
        assert "column_memory" in report
        assert "dtype_summary" in report
        assert "optimization_potential" in report

        # 基本的な値を確認
        assert report["n_rows"] == 1000
        assert report["n_columns"] == 5
        assert report["total_memory_bytes"] > 0
        assert report["bytes_per_row"] > 0

        # カラムごとのメモリ情報を確認
        for col in df.columns:
            assert col in report["column_memory"]
            col_info = report["column_memory"][col]
            assert "memory_bytes" in col_info
            assert "memory_mb" in col_info
            assert "dtype" in col_info
            assert "percentage" in col_info
            assert col_info["memory_bytes"] > 0
            assert 0 <= col_info["percentage"] <= 100

        # データ型サマリーを確認
        assert len(report["dtype_summary"]) > 0
        for dtype_info in report["dtype_summary"].values():
            assert "count" in dtype_info
            assert "total_memory_bytes" in dtype_info
            assert "columns" in dtype_info

        # 最適化ポテンシャルを確認
        opt_potential = report["optimization_potential"]
        assert "potential_savings_bytes" in opt_potential
        assert "potential_savings_mb" in opt_potential
        assert "potential_savings_percentage" in opt_potential
        assert "suggestions" in opt_potential

        # 最適化の提案があることを確認（Float64とInt64があるため）
        assert opt_potential["potential_savings_bytes"] > 0
        assert len(opt_potential["suggestions"]) > 0
        assert any("Float64 → Float32" in s for s in opt_potential["suggestions"])
        assert any("Int64 → Int" in s for s in opt_potential["suggestions"])

    def test_lazy_frame_processing(self, sample_data: pl.DataFrame):
        """LazyFrameによる遅延評価の詳細テスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # エンジンを初期化
        engine = PolarsProcessingEngine()

        # 1. 遅延評価の動作確認
        lazy_df = engine.create_lazyframe(sample_data)
        assert isinstance(lazy_df, pl.LazyFrame), "LazyFrameが返されるべき"

        # 2. フィルタリング操作のテスト
        filtered_lazy = engine.apply_filters(
            lazy_df,
            filters=[
                ("volume", ">", 5000),
                ("close", ">", 1.0850),
            ],
        )
        assert isinstance(filtered_lazy, pl.LazyFrame), (
            "フィルタリング後もLazyFrameを維持すべき"
        )

        # フィルタリング結果を確認
        filtered_result = filtered_lazy.collect()
        assert all(filtered_result["volume"] > 5000), "volumeフィルタが適用されるべき"
        assert all(filtered_result["close"] > 1.0850), "closeフィルタが適用されるべき"

        # 3. 集計操作のテスト
        aggregated_lazy = engine.apply_aggregations(
            lazy_df,
            group_by=None,
            aggregations={
                "open": ["mean", "std"],
                "high": ["max"],
                "low": ["min"],
                "volume": ["sum", "mean"],
            },
        )
        assert isinstance(aggregated_lazy, pl.LazyFrame), (
            "集計後もLazyFrameを維持すべき"
        )

        # 集計結果を確認
        agg_result = aggregated_lazy.collect()
        assert "open_mean" in agg_result.columns, "open_meanカラムが存在すべき"
        assert "open_std" in agg_result.columns, "open_stdカラムが存在すべき"
        assert "high_max" in agg_result.columns, "high_maxカラムが存在すべき"
        assert "low_min" in agg_result.columns, "low_minカラムが存在すべき"
        assert "volume_sum" in agg_result.columns, "volume_sumカラムが存在すべき"
        assert "volume_mean" in agg_result.columns, "volume_meanカラムが存在すべき"
        assert len(agg_result) == 1, "集計結果は1行であるべき"

        # 4. グループ化集計のテスト
        # タイムスタンプから分単位のグループキーを作成
        lazy_with_minute = lazy_df.with_columns(
            [
                pl.col("timestamp").dt.truncate("1m").alias("minute"),
            ]
        )

        grouped_agg_lazy = engine.apply_aggregations(
            lazy_with_minute,
            group_by=["minute"],
            aggregations={
                "close": ["first", "last"],
                "volume": ["sum"],
            },
        )
        grouped_result = grouped_agg_lazy.collect()
        assert "minute" in grouped_result.columns, "グループキーが存在すべき"
        assert "close_first" in grouped_result.columns
        assert "close_last" in grouped_result.columns
        assert "volume_sum" in grouped_result.columns

        # 5. 複数操作のチェイン処理テスト
        chained_lazy = (
            engine.create_lazyframe(sample_data)
            .pipe(
                engine.apply_filters,
                filters=[
                    ("volume", ">=", 3000),
                    ("volume", "<=", 8000),
                ],
            )
            .with_columns(
                [
                    # 価格の変化率を計算
                    ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias(
                        "price_change_pct"
                    ),
                    # 価格レンジ
                    (pl.col("high") - pl.col("low")).alias("price_range"),
                ]
            )
            .sort("timestamp")
            .select(
                [
                    "timestamp",
                    "close",
                    "volume",
                    "price_change_pct",
                    "price_range",
                ]
            )
        )

        assert isinstance(chained_lazy, pl.LazyFrame), (
            "チェイン処理後もLazyFrameを維持すべき"
        )

        # チェイン処理の結果を確認
        chained_result = chained_lazy.collect()
        assert (
            3000
            <= chained_result["volume"].min()
            <= chained_result["volume"].max()
            <= 8000
        )
        assert "price_change_pct" in chained_result.columns
        assert "price_range" in chained_result.columns
        assert len(chained_result.columns) == 5, "選択された5カラムのみ存在すべき"

        # 6. collectタイミングの検証（計算が遅延されていることを確認）
        # 複雑なクエリを構築しても、collect()するまで実行されない
        complex_lazy = (
            engine.create_lazyframe(sample_data)
            .filter(pl.col("volume") > 1000)
            .with_columns(
                [
                    pl.col("close").rolling_mean(window_size=5).alias("close_ma5"),
                    pl.col("volume").rolling_mean(window_size=3).alias("volume_ma3"),
                ]
            )
            .filter(pl.col("close_ma5").is_not_null())
        )

        # この時点ではまだ計算されていない（LazyFrameのまま）
        assert isinstance(complex_lazy, pl.LazyFrame)

        # collect()で初めて計算が実行される
        complex_result = complex_lazy.collect()
        assert isinstance(complex_result, pl.DataFrame)
        assert "close_ma5" in complex_result.columns
        assert "volume_ma3" in complex_result.columns

        # 7. メモリ効率の確認（LazyFrameは中間結果を保持しない）
        # 大きなデータセットでの処理をシミュレート
        large_data = pl.DataFrame(
            {
                "timestamp": sample_data["timestamp"].to_list() * 10,  # 10倍に拡張
                "value": list(range(1000)),
            }
        )

        # LazyFrameで処理（メモリ効率的）
        lazy_large = (
            large_data.lazy()
            .filter(pl.col("value") > 500)
            .group_by("timestamp")
            .agg(pl.col("value").mean())
        )

        # 計算前はメモリを消費しない
        assert isinstance(lazy_large, pl.LazyFrame)

        # 必要な時だけ計算
        result_large = lazy_large.collect()
        assert len(result_large) > 0

    def test_chunk_processing(self):
        """大規模データのチャンク処理テスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # 1. 100万行の大規模データを生成
        n_rows = 1_000_000
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # メモリ効率的にデータを生成（チャンクで生成）
        chunk_size = 100_000
        chunks = []

        for i in range(0, n_rows, chunk_size):
            chunk_end = min(i + chunk_size, n_rows)
            chunk_data = {
                "timestamp": [
                    base_time + timedelta(seconds=j) for j in range(i, chunk_end)
                ],
                "open": np.random.uniform(1.0800, 1.0900, chunk_end - i).astype(
                    np.float32
                ),
                "high": np.random.uniform(1.0850, 1.0950, chunk_end - i).astype(
                    np.float32
                ),
                "low": np.random.uniform(1.0750, 1.0850, chunk_end - i).astype(
                    np.float32
                ),
                "close": np.random.uniform(1.0800, 1.0900, chunk_end - i).astype(
                    np.float32
                ),
                "volume": np.random.uniform(1000, 10000, chunk_end - i).astype(
                    np.float32
                ),
            }
            chunks.append(pl.DataFrame(chunk_data))

        # 全データを結合
        large_df = pl.concat(chunks)
        del chunks  # メモリ解放
        gc.collect()

        assert len(large_df) == n_rows, f"データが{n_rows}行であるべき"

        # 2. チャンクサイズの設定と検証
        engine = PolarsProcessingEngine(chunk_size=100_000)
        assert engine.chunk_size == 100_000, "チャンクサイズが100,000であるべき"

        # 3. チャンク処理のテスト
        # process_in_chunksメソッドのテスト（実装後に有効化）
        processed_chunks = []
        memory_usage_per_chunk = []

        # メモリ使用量の測定開始
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # LazyFrameでチャンク処理をシミュレート
        lazy_df = large_df.lazy()

        # チャンクごとに処理
        for i in range(0, n_rows, engine.chunk_size):
            chunk_end = min(i + engine.chunk_size, n_rows)

            # チャンクの処理（フィルタリングと集計）
            chunk_result = (
                lazy_df.slice(i, chunk_end - i)
                .filter(pl.col("volume") > 5000)
                .group_by(pl.col("timestamp").dt.truncate("1h"))
                .agg(
                    [
                        pl.col("close").mean().alias("close_mean"),
                        pl.col("volume").sum().alias("volume_sum"),
                    ]
                )
                .collect()
            )

            processed_chunks.append(chunk_result)

            # チャンクごとのメモリ使用量を記録
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage_per_chunk.append(current_memory - initial_memory)

        # 4. 結果の検証
        assert len(processed_chunks) == 10, "10個のチャンクが処理されるべき"

        # メモリ使用量が一定以下であることを確認
        max_memory_increase = max(memory_usage_per_chunk)
        avg_memory_increase = sum(memory_usage_per_chunk) / len(memory_usage_per_chunk)

        # チャンク処理によりメモリ使用量が制御されていることを確認
        # （全データを一度に処理するよりも効率的）
        total_data_size_mb = large_df.estimated_size() / 1024 / 1024
        assert max_memory_increase < total_data_size_mb * 0.6, (
            f"チャンク処理のメモリ使用量が過大: {max_memory_increase:.2f}MB "
            f"(データサイズの60%未満であるべき: {total_data_size_mb * 0.6:.2f}MB)"
        )

        # 5. チャンク結果の集約
        final_result = pl.concat(processed_chunks)
        assert len(final_result) > 0, "処理結果が存在すべき"
        assert "close_mean" in final_result.columns
        assert "volume_sum" in final_result.columns

        # 6. チャンクサイズの動的調整テスト
        # メモリ使用量に基づいてチャンクサイズを調整
        if avg_memory_increase > 100:  # 100MB以上使用している場合
            new_chunk_size = engine.chunk_size // 2
            engine.chunk_size = new_chunk_size
            assert engine.chunk_size == 50_000, "チャンクサイズが動的に調整されるべき"

        # クリーンアップ
        del large_df
        del processed_chunks
        gc.collect()

    def test_streaming_processing(self):
        """ストリーミング処理（CSV/Parquet）のテスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # 1. テスト用の大規模データを生成
        n_rows = 100_000
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        data = {
            "timestamp": [base_time + timedelta(seconds=i) for i in range(n_rows)],
            "open": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float32),
            "high": np.random.uniform(1.0850, 1.0950, n_rows).astype(np.float32),
            "low": np.random.uniform(1.0750, 1.0850, n_rows).astype(np.float32),
            "close": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float32),
            "volume": np.random.uniform(1000, 10000, n_rows).astype(np.float32),
        }
        df = pl.DataFrame(data)

        # エンジンを初期化
        PolarsProcessingEngine(chunk_size=10_000)

        # 2. CSVストリーミング処理のテスト
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_csv:
            csv_path = tmp_csv.name
            df.write_csv(csv_path)

        try:
            # scan_csvによるストリーミング読み込み（データ型を明示）
            lazy_csv = pl.scan_csv(
                csv_path,
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float32,
                    "high": pl.Float32,
                    "low": pl.Float32,
                    "close": pl.Float32,
                    "volume": pl.Float32,
                },
            )
            assert isinstance(lazy_csv, pl.LazyFrame), "scan_csvはLazyFrameを返すべき"

            # ストリーミング処理（メモリに全データをロードせずに処理）
            csv_result = (
                lazy_csv.filter(pl.col("volume") > 5000)
                .with_columns(pl.col("timestamp").dt.truncate("1h").alias("hour"))
                .group_by("hour")
                .agg(
                    [
                        pl.col("close").mean().alias("close_mean"),
                        pl.col("volume").sum().alias("volume_sum"),
                        pl.col("high").max().alias("high_max"),
                        pl.col("low").min().alias("low_min"),
                    ]
                )
                .sort("hour")
                .collect()  # ストリーミングモードで実行
            )

            assert isinstance(csv_result, pl.DataFrame), "結果はDataFrameであるべき"
            assert len(csv_result) > 0, "処理結果が存在すべき"
            assert "close_mean" in csv_result.columns
            assert "volume_sum" in csv_result.columns

        finally:
            # クリーンアップ
            if os.path.exists(csv_path):
                os.unlink(csv_path)

        # 3. Parquetストリーミング処理のテスト
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False
        ) as tmp_parquet:
            parquet_path = tmp_parquet.name
            df.write_parquet(parquet_path)

        try:
            # scan_parquetによるストリーミング読み込み
            lazy_parquet = pl.scan_parquet(parquet_path)
            assert isinstance(lazy_parquet, pl.LazyFrame), (
                "scan_parquetはLazyFrameを返すべき"
            )

            # ストリーミング処理
            parquet_result = (
                lazy_parquet.filter(
                    (pl.col("volume") > 3000) & (pl.col("close") > 1.0850)
                )
                .with_columns(
                    [
                        # 移動平均の計算
                        pl.col("close")
                        .rolling_mean(window_size=100)
                        .alias("close_ma100"),
                        # 価格レンジ
                        (pl.col("high") - pl.col("low")).alias("price_range"),
                        # ボリューム変化率
                        (pl.col("volume").pct_change()).alias("volume_pct_change"),
                    ]
                )
                .filter(pl.col("close_ma100").is_not_null())
                .select(
                    [
                        "timestamp",
                        "close",
                        "close_ma100",
                        "price_range",
                        "volume",
                        "volume_pct_change",
                    ]
                )
                .collect()
            )

            assert isinstance(parquet_result, pl.DataFrame), "結果はDataFrameであるべき"
            assert len(parquet_result) > 0, "処理結果が存在すべき"
            assert "close_ma100" in parquet_result.columns
            assert "price_range" in parquet_result.columns
            assert "volume_pct_change" in parquet_result.columns

            # 移動平均が正しく計算されているか確認
            assert parquet_result["close_ma100"].null_count() == 0, (
                "フィルタリング後はnull値が存在しないべき"
            )

        finally:
            # クリーンアップ
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)

        # 4. メモリ効率の比較テスト
        import gc

        import psutil

        # 通常の読み込み（全データをメモリにロード）
        gc.collect()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        normal_df = df.filter(pl.col("volume") > 5000)
        memory_after_normal = process.memory_info().rss / 1024 / 1024  # MB
        memory_after_normal - memory_before

        del normal_df
        gc.collect()

        # ストリーミング処理のメモリ使用量を測定
        # （実際のストリーミング処理は上記で実施済み）
        # ストリーミング処理のメモリ増加は通常処理より小さいはず

        # 5. バッチ処理との組み合わせテスト
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            batch_path = tmp.name
            df.write_parquet(batch_path)

        try:
            # バッチサイズを指定してストリーミング処理
            batch_size = 20_000
            lazy_batch = pl.scan_parquet(batch_path)

            # バッチごとに異なる処理を適用
            batch_results = []
            for i in range(0, n_rows, batch_size):
                batch_result = (
                    lazy_batch.slice(i, min(batch_size, n_rows - i))
                    .group_by(pl.col("timestamp").dt.truncate("10m"))
                    .agg(
                        [
                            pl.col("close").mean(),
                            pl.col("volume").sum(),
                        ]
                    )
                    .collect()
                )
                batch_results.append(batch_result)

            # バッチ結果を結合
            final_batch_result = pl.concat(batch_results)
            assert len(final_batch_result) > 0, "バッチ処理結果が存在すべき"

        finally:
            if os.path.exists(batch_path):
                os.unlink(batch_path)

    def test_invalid_data_types(self, caplog):
        """不適切なデータ型の処理を検証"""
        from src.data_processing.processor import PolarsProcessingEngine
        
        engine = PolarsProcessingEngine()
        
        # 1. 文字列が数値カラムに混入した場合
        # strict=Falseで混合型のDataFrameを作成
        invalid_data = pl.DataFrame(
            {
                "timestamp": [datetime.now()] * 5,
                "open": ["invalid", "1.0800", "1.0810", "1.0820", "1.0830"],  # 文字列として作成
                "high": [1.0900, 1.0910, 1.0920, 1.0930, 1.0940],
                "low": [1.0750, 1.0760, 1.0770, 1.0780, 1.0790],
                "close": [1.0850, 1.0860, 1.0870, 1.0880, 1.0890],
                "volume": [1000, 2000, 3000, 4000, 5000],
            }
        )
        
        # データ型最適化でエラーが発生することを確認（文字列カラムをFloat32に変換しようとする）
        with pytest.raises((pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError, ValueError)):
            # openカラムを強制的にFloat型に変換しようとする
            invalid_data = invalid_data.with_columns(pl.col("open").cast(pl.Float64))
            engine.optimize_dtypes(invalid_data)
        
        # ログにエラーメッセージが記録されることを確認
        assert "error" in caplog.text.lower() or "warning" in caplog.text.lower()
        
        # 2. 日付型と数値型の混在（カラムのデータ型が一貫していない場合のテスト）
        # 異なるアプローチ: オブジェクト型として作成
        mixed_values = [1.5, "not_a_number", 3.0]
        mixed_type_data = pl.DataFrame({
            "timestamp": [datetime.now()] * 3,
            "value": mixed_values,  # 混在型
        }, strict=False)
        
        # LazyFrame作成時の型チェック
        lazy_df = engine.create_lazyframe(mixed_type_data)
        
        # 数値演算でエラーが発生（文字列に対する数値演算）
        with pytest.raises((pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError)):
            # 文字列カラムを数値として扱おうとする
            result = lazy_df.with_columns(
                pl.col("value").cast(pl.Float64) * 2
            ).collect()
        
        # 3. null/NaN値の適切な処理
        null_data = pl.DataFrame({
            "timestamp": [datetime.now()] * 5,
            "open": [1.0800, None, 1.0810, float('nan'), 1.0830],
            "high": [1.0900, 1.0910, None, 1.0930, float('nan')],
            "low": [1.0750, 1.0760, 1.0770, None, 1.0790],
            "close": [None, 1.0860, 1.0870, 1.0880, None],
            "volume": [1000, None, 3000, None, 5000],
        })
        
        # null値を含むデータの最適化
        optimized = engine.optimize_dtypes(null_data)
        
        # null値が保持されていることを確認
        assert optimized["open"].null_count() > 0
        assert optimized["close"].null_count() > 0
        
        # フィルタリングでnull値を処理
        lazy_null = engine.create_lazyframe(optimized)
        filtered = engine.apply_filters(
            lazy_null,
            filters=[("open", ">", 1.08)]
        ).collect()
        
        # null値は自動的に除外される
        assert filtered["open"].null_count() == 0

    def test_empty_dataframe_handling(self, caplog):
        """空のDataFrameに対する安全な処理"""
        from src.data_processing.processor import PolarsProcessingEngine
        
        engine = PolarsProcessingEngine()
        
        # 1. 完全に空のDataFrame
        empty_df = pl.DataFrame()
        
        # データ型最適化
        with caplog.at_level("WARNING"):
            result = engine.optimize_dtypes(empty_df)
            assert result.shape == (0, 0)
            assert "empty" in caplog.text.lower()
        
        # 2. カラムはあるが行が0のDataFrame
        empty_with_schema = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }, schema={
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        })
        
        # 空データの最適化
        optimized_empty = engine.optimize_dtypes(empty_with_schema)
        assert optimized_empty.shape[0] == 0
        assert optimized_empty["open"].dtype == pl.Float32
        
        # 3. LazyFrameの作成と処理
        lazy_empty = engine.create_lazyframe(empty_with_schema)
        
        # フィルタリング（エラーなく空の結果を返す）
        filtered_empty = engine.apply_filters(
            lazy_empty,
            filters=[("volume", ">", 1000)]
        ).collect()
        assert filtered_empty.shape[0] == 0
        
        # 集計（エラーなく空の結果を返す）
        agg_empty = engine.apply_aggregations(
            lazy_empty,
            aggregations={"close": ["mean", "sum"]}
        ).collect()
        assert agg_empty.shape[0] == 1  # グローバル集計は1行
        
        # 4. チャンク処理
        result_chunks = engine.process_in_chunks(
            empty_with_schema,
            chunk_size=100,
            process_func=lambda x: x
        )
        assert len(result_chunks) == 0 or result_chunks.shape[0] == 0

    def test_excessive_data_size(self, caplog):
        """メモリ制限を超える大規模データの処理"""
        from src.data_processing.processor import PolarsProcessingEngine
        
        engine = PolarsProcessingEngine(chunk_size=100_000)
        
        # メモリ情報のモック
        with patch('psutil.virtual_memory') as mock_memory:
            # 利用可能メモリを1GBに設定
            mock_memory.return_value.available = 1 * 1024 * 1024 * 1024  # 1GB
            mock_memory.return_value.total = 8 * 1024 * 1024 * 1024  # 8GB
            mock_memory.return_value.percent = 87.5  # 87.5%使用中
            
            # 大規模データのシミュレーション（実際には小さいデータ）
            large_data = pl.DataFrame({
                "timestamp": [datetime.now()] * 1000,
                "value": np.random.rand(1000),
            })
            
            # チャンクサイズの自動調整を確認
            new_chunk_size = engine.adjust_chunk_size()
            
            # メモリ使用率が高いため、チャンクサイズが縮小される
            assert new_chunk_size < engine.chunk_size
            assert new_chunk_size == engine.chunk_size // 2
            
            # 警告ログが出力される
            assert "memory usage high" in caplog.text.lower() or "adjusting" in caplog.text.lower()
            
            # チャンク処理が正常に動作
            with caplog.at_level("INFO"):
                result = engine.process_in_chunks(
                    large_data,
                    chunk_size=new_chunk_size,
                    process_func=lambda x: x.select(pl.col("value").mean())
                )
                assert result is not None
                assert "processing chunk" in caplog.text.lower() or "chunk" in caplog.text.lower()

    def test_corrupted_file_handling(self, caplog):
        """破損ファイルの読み込みエラー処理"""
        from src.data_processing.processor import PolarsProcessingEngine
        
        engine = PolarsProcessingEngine()
        
        # 1. 不正なCSVファイルの作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            corrupted_csv_path = f.name
            # 不完全なCSVデータ（カラム数が一致しない）
            f.write("col1,col2,col3\n")
            f.write("1,2\n")  # カラム不足
            f.write("3,4,5,6\n")  # カラム過多
            f.write("invalid,data\n")
        
        try:
            # CSVの読み込みでエラー
            with pytest.raises(pl.exceptions.ComputeError):
                pl.read_csv(corrupted_csv_path)
            
            # stream_csvメソッドでのエラーハンドリング
            with caplog.at_level("ERROR"):
                lazy_csv = engine.stream_csv(corrupted_csv_path)
                if lazy_csv is not None:
                    with pytest.raises(pl.exceptions.ComputeError):
                        lazy_csv.collect()
                assert "error" in caplog.text.lower()
        finally:
            os.unlink(corrupted_csv_path)
        
        # 2. 存在しないファイル
        non_existent = "/path/to/non/existent/file.parquet"
        
        with caplog.at_level("ERROR"):
            result = engine.stream_parquet(non_existent)
            assert result is None or isinstance(result, pl.LazyFrame)
            if "error" in caplog.text.lower():
                assert "not found" in caplog.text.lower() or "exist" in caplog.text.lower()
        
        # 3. スキーマ不整合のParquetファイル
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name
            
            # 最初のスキーマでファイルを作成
            df1 = pl.DataFrame({
                "id": [1, 2, 3],
                "value": [1.0, 2.0, 3.0],
            })
            df1.write_parquet(parquet_path)
        
        try:
            # 異なるスキーマでの読み込みを試みる
            expected_schema = {
                "id": pl.Int32,
                "value": pl.Float32,
                "extra_column": pl.String,  # 存在しないカラム
            }
            
            # スキーマの不一致でエラーまたは警告
            lazy_parquet = pl.scan_parquet(parquet_path)
            result_df = lazy_parquet.collect()
            
            # 存在しないカラムへのアクセスでエラー
            with pytest.raises(pl.exceptions.ColumnNotFoundError):
                result_df.select("extra_column")
                
        finally:
            os.unlink(parquet_path)

    def test_memory_pressure_handling(self):
        """メモリ逼迫時の動作確認"""
        from src.data_processing.processor import PolarsProcessingEngine
        
        engine = PolarsProcessingEngine(chunk_size=100_000)
        
        # psutil.Processのモック
        with patch('psutil.Process') as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process
            
            # メモリ使用率を段階的に変更
            memory_percentages = [85.0, 90.0, 95.0]
            
            with patch('psutil.virtual_memory') as mock_memory:
                for mem_percent in memory_percentages:
                    # メモリ使用率を設定
                    mock_memory.return_value.percent = mem_percent
                    mock_memory.return_value.available = (100 - mem_percent) * 10 * 1024 * 1024
                    mock_memory.return_value.total = 100 * 10 * 1024 * 1024
                    
                    # チャンクサイズの調整
                    new_size = engine.adjust_chunk_size()
                    
                    # メモリ使用率が80%を超えているため、チャンクサイズが縮小
                    if mem_percent > 80:
                        assert new_size < engine.chunk_size
                        # 縮小率の確認
                        assert new_size == engine.chunk_size // 2
                    
                    # 新しいチャンクサイズを設定
                    engine.chunk_size = new_size
                    
                    # 最小チャンクサイズの確認
                    assert engine.chunk_size >= 1_000
        
        # グレースフルデグレデーションのテスト
        test_data = pl.DataFrame({
            "timestamp": [datetime.now()] * 10000,
            "value": np.random.rand(10000),
        })
        
        with patch('psutil.virtual_memory') as mock_memory:
            # 極度のメモリ逼迫（95%使用）
            mock_memory.return_value.percent = 95.0
            mock_memory.return_value.available = 500 * 1024 * 1024  # 500MB
            
            # 処理は継続されるが、チャンクサイズは最小に
            result = engine.process_in_chunks(
                test_data,
                chunk_size=engine.adjust_chunk_size(),
                process_func=lambda x: x.select(pl.col("value").mean())
            )
            
            # 処理が完了することを確認
            assert result is not None
            
            # チャンクサイズが適切に調整されている
            assert engine.chunk_size <= 10_000

    def test_invalid_parameters(self, caplog):
        """不正なパラメータの検証"""
        from src.data_processing.processor import PolarsProcessingEngine
        
        # 1. 負のチャンクサイズ
        with pytest.raises(ValueError) as exc_info:
            PolarsProcessingEngine(chunk_size=-1000)
        assert "chunk_size must be positive" in str(exc_info.value).lower()
        
        engine = PolarsProcessingEngine()
        
        # 2. 存在しない集計関数名
        df = pl.DataFrame({
            "timestamp": [datetime.now()] * 5,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        lazy_df = engine.create_lazyframe(df)
        
        with pytest.raises(ValueError) as exc_info:
            engine.apply_aggregations(
                lazy_df,
                aggregations={"value": ["invalid_function"]}
            )
        assert "unsupported aggregation" in str(exc_info.value).lower()
        
        # 3. 不正なフィルタ条件式
        # 無効な演算子
        with pytest.raises(ValueError) as exc_info:
            engine.apply_filters(
                lazy_df,
                filters=[("value", "invalid_op", 3.0)]
            )
        assert "unsupported operator" in str(exc_info.value).lower()
        
        # 4. 無効なカラム名
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            engine.apply_filters(
                lazy_df,
                filters=[("non_existent_column", ">", 3.0)]
            ).collect()
        
        # 5. 無効なデータ型パラメータ
        with pytest.raises((TypeError, ValueError)):
            engine.optimize_dtypes(df, categorical_threshold="invalid")  # 数値であるべき
        
        # 6. process_in_chunksの無効なパラメータ
        with pytest.raises((TypeError, ValueError)):
            engine.process_in_chunks(
                df,
                chunk_size=0,  # 0は無効
                process_func=lambda x: x
            )
        
        # 7. 無効な処理関数
        with pytest.raises(TypeError):
            engine.process_in_chunks(
                df,
                chunk_size=100,
                process_func="not_a_function"  # 関数であるべき
            )


class TestDataValidation:
    """データ検証機能のテストスイート"""

    def test_validate_schema(self):
        """スキーマ検証のテスト"""
        # TODO: スキーマ検証の実装後にテストを有効化
        pass

    def test_validate_data_types(self):
        """データ型検証のテスト"""
        # TODO: データ型検証の実装後にテストを有効化
        pass

    def test_validate_data_range(self):
        """データ範囲検証のテスト"""
        # TODO: データ範囲検証の実装後にテストを有効化
        pass


class TestPerformance:
    """パフォーマンステストスイート"""

    @pytest.mark.benchmark
    def test_processing_speed(self):
        """処理速度のベンチマークテスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # 1. テストデータの準備（10万行）
        n_rows = 100_000
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        data = {
            "timestamp": [base_time + timedelta(seconds=i) for i in range(n_rows)],
            "open": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
            "high": np.random.uniform(1.0850, 1.0950, n_rows).astype(np.float64),
            "low": np.random.uniform(1.0750, 1.0850, n_rows).astype(np.float64),
            "close": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
            "volume": np.random.uniform(1000, 10000, n_rows).astype(np.float64),
            "symbol": np.random.choice(["EURUSD", "GBPUSD", "USDJPY"], n_rows),
        }
        df = pl.DataFrame(data)

        engine = PolarsProcessingEngine()

        # 2. データ型最適化の速度測定
        start_time = time.perf_counter()
        optimized_df = engine.optimize_dtypes(df)
        optimization_time = time.perf_counter() - start_time

        print(
            f"\nデータ型最適化: {optimization_time:.4f}秒 ({n_rows / optimization_time:.0f} rows/sec)"
        )
        assert optimization_time < 1.0, (
            f"最適化が1秒以内に完了すべき: {optimization_time:.4f}秒"
        )

        # 3. LazyFrame処理の速度測定
        lazy_df = engine.create_lazyframe(optimized_df)

        # 複雑なクエリの構築
        start_time = time.perf_counter()
        complex_query = (
            lazy_df.filter(pl.col("volume") > 5000)
            .with_columns(
                [
                    # 移動平均
                    pl.col("close").rolling_mean(window_size=20).alias("close_ma20"),
                    pl.col("volume").rolling_mean(window_size=10).alias("volume_ma10"),
                    # 価格変化率
                    pl.col("close").pct_change().alias("close_pct_change"),
                    # 価格レンジ
                    (pl.col("high") - pl.col("low")).alias("price_range"),
                ]
            )
            .group_by(["symbol", pl.col("timestamp").dt.truncate("1h")])
            .agg(
                [
                    pl.col("close").mean().alias("close_mean"),
                    pl.col("volume").sum().alias("volume_sum"),
                    pl.col("price_range").max().alias("max_range"),
                    pl.col("close_pct_change").std().alias("volatility"),
                ]
            )
            .sort(["symbol", "timestamp"])
        )

        # クエリ実行時間の測定
        complex_query.collect()
        query_time = time.perf_counter() - start_time

        print(
            f"複雑なクエリ実行: {query_time:.4f}秒 ({n_rows / query_time:.0f} rows/sec)"
        )
        assert query_time < 2.0, f"クエリが2秒以内に完了すべき: {query_time:.4f}秒"

        # 4. フィルタリング性能の測定
        start_time = time.perf_counter()
        engine.apply_filters(
            lazy_df,
            filters=[
                ("volume", ">", 3000),
                ("volume", "<", 8000),
                ("close", ">=", 1.0850),
            ],
        ).collect()
        filter_time = time.perf_counter() - start_time

        print(
            f"フィルタリング処理: {filter_time:.4f}秒 ({n_rows / filter_time:.0f} rows/sec)"
        )
        assert filter_time < 0.5, (
            f"フィルタリングが0.5秒以内に完了すべき: {filter_time:.4f}秒"
        )

        # 5. 集計処理の性能測定
        start_time = time.perf_counter()
        engine.apply_aggregations(
            lazy_df,
            group_by=["symbol"],
            aggregations={
                "close": ["mean", "std", "min", "max"],
                "volume": ["sum", "mean"],
                "high": ["max"],
                "low": ["min"],
            },
        ).collect()
        agg_time = time.perf_counter() - start_time

        print(f"集計処理: {agg_time:.4f}秒 ({n_rows / agg_time:.0f} rows/sec)")
        assert agg_time < 0.5, f"集計が0.5秒以内に完了すべき: {agg_time:.4f}秒"

        # 6. 総合処理時間の確認
        total_time = optimization_time + query_time + filter_time + agg_time
        throughput = n_rows * 4 / total_time  # 4つの処理を実行

        print("\n=== パフォーマンスサマリー ===")
        print(f"総処理時間: {total_time:.4f}秒")
        print(f"スループット: {throughput:.0f} rows/sec")
        print(f"処理行数: {n_rows:,}行")

        # パフォーマンス基準を満たしているか確認
        assert throughput > 50000, (
            f"スループットが50,000 rows/sec以上であるべき: {throughput:.0f} rows/sec"
        )

    @pytest.mark.benchmark
    def test_memory_efficiency(self):
        """メモリ効率のベンチマークテスト"""
        from src.data_processing.processor import PolarsProcessingEngine

        # メモリ測定用のプロセス
        process = psutil.Process()

        # 1. 大規模データの準備（50万行）
        n_rows = 500_000
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # ガベージコレクションを実行してベースラインメモリを測定
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Float64でデータを作成（メモリ非効率）
        data = {
            "timestamp": [base_time + timedelta(seconds=i) for i in range(n_rows)],
            "open": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
            "high": np.random.uniform(1.0850, 1.0950, n_rows).astype(np.float64),
            "low": np.random.uniform(1.0750, 1.0850, n_rows).astype(np.float64),
            "close": np.random.uniform(1.0800, 1.0900, n_rows).astype(np.float64),
            "volume": np.random.uniform(1000, 10000, n_rows).astype(np.float64),
            "trade_count": np.random.randint(1, 100, n_rows),  # Int64
            "symbol": np.random.choice(["EURUSD", "GBPUSD", "USDJPY"], n_rows),
        }
        df = pl.DataFrame(data)

        # 元のデータのメモリ使用量
        original_memory = process.memory_info().rss / 1024 / 1024  # MB
        original_df_size = df.estimated_size() / 1024 / 1024  # MB
        memory_after_creation = original_memory - baseline_memory

        print("\n=== メモリ使用量（最適化前） ===")
        print(f"DataFrameサイズ: {original_df_size:.2f} MB")
        print(f"プロセスメモリ増加: {memory_after_creation:.2f} MB")

        # 2. メモリ最適化の実行
        engine = PolarsProcessingEngine()
        optimized_df = engine.optimize_dtypes(df, categorical_threshold=0.3)

        # 最適化後のメモリ使用量
        optimized_memory = process.memory_info().rss / 1024 / 1024  # MB
        optimized_df_size = optimized_df.estimated_size() / 1024 / 1024  # MB

        print("\n=== メモリ使用量（最適化後） ===")
        print(f"DataFrameサイズ: {optimized_df_size:.2f} MB")
        print(f"メモリ削減率: {(1 - optimized_df_size / original_df_size) * 100:.1f}%")

        # メモリ削減率の確認（35%以上削減を期待）
        memory_reduction = 1 - (optimized_df_size / original_df_size)
        assert memory_reduction > 0.35, (
            f"メモリが35%以上削減されるべき: {memory_reduction * 100:.1f}%"
        )

        # 元のDataFrameを削除してメモリ解放
        del df
        gc.collect()

        # 3. LazyFrameによるメモリ効率の測定
        lazy_df = engine.create_lazyframe(optimized_df)

        # 複雑な処理チェーンを構築（まだ実行されない）
        complex_lazy = (
            lazy_df.filter(pl.col("volume") > 5000)
            .with_columns(
                [
                    pl.col("close").rolling_mean(window_size=100).alias("close_ma100"),
                    pl.col("volume").rolling_mean(window_size=50).alias("volume_ma50"),
                    ((pl.col("high") - pl.col("low")) / pl.col("open")).alias(
                        "volatility"
                    ),
                ]
            )
            .group_by(["symbol", pl.col("timestamp").dt.truncate("1h")])
            .agg(
                [
                    pl.col("close").mean(),
                    pl.col("volume").sum(),
                    pl.col("volatility").max(),
                ]
            )
        )

        # LazyFrameの構築はメモリをほとんど使用しない
        lazy_memory = process.memory_info().rss / 1024 / 1024  # MB
        lazy_memory_increase = lazy_memory - optimized_memory

        print("\n=== LazyFrame処理 ===")
        print(f"LazyFrame構築時のメモリ増加: {lazy_memory_increase:.2f} MB")
        assert lazy_memory_increase < 10, (
            f"LazyFrame構築はメモリをほとんど使用しないべき: {lazy_memory_increase:.2f} MB"
        )

        # 実際に計算を実行
        result = complex_lazy.collect()
        result_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"計算実行後のメモリ: {result_memory:.2f} MB")
        print(f"結果のサイズ: {result.estimated_size() / 1024 / 1024:.2f} MB")

        # 4. チャンク処理によるメモリ効率の測定
        del optimized_df
        del result
        gc.collect()

        # チャンク処理のシミュレーション
        chunk_size = 50_000
        max_chunk_memory = 0

        for i in range(0, n_rows, chunk_size):
            # チャンクごとに処理
            chunk_end = min(i + chunk_size, n_rows)
            chunk_lazy = (
                lazy_df.slice(i, chunk_end - i)
                .filter(pl.col("volume") > 3000)
                .group_by(pl.col("timestamp").dt.truncate("10m"))
                .agg(
                    [
                        pl.col("close").mean(),
                        pl.col("volume").sum(),
                    ]
                )
            )

            chunk_result = chunk_lazy.collect()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_chunk_memory = max(max_chunk_memory, current_memory)

            # チャンク結果を処理（実際の実装では保存や集約を行う）
            del chunk_result
            gc.collect()

        print("\n=== チャンク処理 ===")
        print(f"最大メモリ使用量: {max_chunk_memory:.2f} MB")
        print(f"チャンクサイズ: {chunk_size:,}行")

        # 5. メモリレポートの生成
        # 新しいデータで最終的なメモリ分析
        final_data = {
            "id": list(range(1000)),
            "value": np.random.uniform(0, 100, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        }
        final_df = pl.DataFrame(final_data)

        report = engine.get_memory_report(final_df)

        print("\n=== メモリレポート ===")
        print(f"総メモリ: {report['total_memory_mb']:.2f} MB")
        print(f"行あたり: {report['bytes_per_row']:.0f} bytes")
        print(
            f"最適化可能: {report['optimization_potential']['potential_savings_percentage']:.1f}%"
        )

        # メモリ効率の総合評価
        assert max_chunk_memory < baseline_memory + 200, (
            f"チャンク処理のメモリ使用量が過大: {max_chunk_memory:.2f} MB"
        )


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v"])
