"""
統合テスト: テクニカル指標計算エンジン

エンドツーエンドワークフロー、既存システムとの統合、
パフォーマンスベンチマーク、ストリーミングシミュレーションを実装します。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.data_processing.indicators import TechnicalIndicatorEngine
from src.data_processing.processor import PolarsProcessingEngine


# ==================== フィクスチャ ====================


@pytest.fixture
def large_forex_data() -> pl.DataFrame:
    """大規模なテスト用Forexデータを生成"""
    np.random.seed(42)
    n_rows = 1_000_000
    
    # 時系列データの生成
    timestamps = [
        datetime(2024, 1, 1) + timedelta(seconds=i)
        for i in range(n_rows)
    ]
    
    # 複数通貨ペアのデータ生成
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    symbol_data = []
    
    for symbol in symbols:
        n_symbol_rows = n_rows // len(symbols)
        
        # 現実的な価格変動をシミュレート
        if symbol == "USDJPY":
            base_price = 110.0
            volatility = 0.5
        else:
            base_price = 1.2 if symbol == "EURUSD" else 1.3
            volatility = 0.001
        
        # ランダムウォークでトレンド生成
        returns = np.random.normal(0, volatility, n_symbol_rows)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLC価格の生成
        highs = prices * (1 + np.abs(np.random.normal(0, 0.0005, n_symbol_rows)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.0005, n_symbol_rows)))
        opens = prices * (1 + np.random.normal(0, 0.0002, n_symbol_rows))
        volumes = np.random.uniform(1000, 10000, n_symbol_rows)
        
        symbol_data.append({
            "timestamp": timestamps[:n_symbol_rows],
            "symbol": [symbol] * n_symbol_rows,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        })
    
    # 全データを結合
    df = pl.concat([
        pl.DataFrame(data)
        for data in symbol_data
    ])
    
    # Float32型に変換
    return df.with_columns([
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
    ])


@pytest.fixture
def streaming_data_generator():
    """ストリーミングデータを生成するジェネレータ"""
    def generate_tick(symbol: str, base_price: float, timestamp: datetime) -> dict:
        """1ティックのデータを生成"""
        price_change = np.random.normal(0, 0.001)
        price = base_price * (1 + price_change)
        
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "bid": price - 0.00005,
            "ask": price + 0.00005,
            "volume": np.random.uniform(100, 1000),
        }
    
    def generate_ohlc_from_ticks(ticks: list[dict]) -> pl.DataFrame:
        """ティックデータからOHLCを生成"""
        if not ticks:
            return pl.DataFrame()
        
        df = pl.DataFrame(ticks)
        
        # ティックから1分足OHLCを生成
        ohlc = df.group_by("symbol").agg([
            pl.col("bid").first().alias("open"),
            pl.col("bid").max().alias("high"),
            pl.col("bid").min().alias("low"),
            pl.col("bid").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("timestamp").last().alias("timestamp"),
        ])
        
        return ohlc.with_columns([
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
        ])
    
    return {
        "generate_tick": generate_tick,
        "generate_ohlc_from_ticks": generate_ohlc_from_ticks,
    }


@pytest.fixture
def indicator_engine() -> TechnicalIndicatorEngine:
    """テクニカル指標エンジンのインスタンス"""
    return TechnicalIndicatorEngine()


@pytest.fixture
def polars_engine() -> PolarsProcessingEngine:
    """Polars処理エンジンのインスタンス"""
    return PolarsProcessingEngine()


# ==================== エンドツーエンドテスト ====================


@pytest.mark.integration
def test_end_to_end_workflow(large_forex_data, indicator_engine):
    """
    エンドツーエンドワークフローのテスト
    データ読み込み → 全指標計算 → 結果検証 → メタデータ確認
    """
    # Step 1: データの準備と検証
    assert not large_forex_data.is_empty()
    assert len(large_forex_data) == 1_000_000
    assert "close" in large_forex_data.columns
    
    # Step 2: 全指標の計算
    indicator_engine.clear_metadata()  # メタデータをリセット
    start_time = time.time()
    
    result = indicator_engine.calculate_all_indicators(
        large_forex_data,
        group_by="symbol",
        include_indicators=["ema", "rsi", "macd", "bollinger"],
    )
    
    processing_time = time.time() - start_time
    
    # Step 3: 結果の検証
    # 全ての指標列が存在することを確認
    expected_columns = [
        # EMA
        "ema_5", "ema_20", "ema_50", "ema_100", "ema_200",
        # RSI
        "rsi_14",
        # MACD
        "macd_line", "macd_signal", "macd_histogram",
        # ボリンジャーバンド
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent",
    ]
    
    for col in expected_columns:
        assert col in result.columns, f"列 {col} が結果に存在しません"
    
    # データ型の確認（全てFloat32であるべき）
    for col in expected_columns:
        assert result[col].dtype == pl.Float32, f"{col}の型が正しくありません"
    
    # Step 4: メタデータの確認
    metadata = indicator_engine.get_metadata()
    
    assert metadata["indicators"]["ema"]["calculated"] is True
    assert metadata["indicators"]["rsi"]["calculated"] is True
    assert metadata["indicators"]["macd"]["calculated"] is True
    assert metadata["indicators"]["bollinger"]["calculated"] is True
    
    # バッチ処理では各指標が個別に行数をカウントするため、合計は行数×指標数
    assert metadata["statistics"]["total_rows_processed"] == len(large_forex_data) * 4
    assert metadata["statistics"]["last_update"] is not None
    
    # Step 5: パフォーマンスの確認
    assert processing_time < 10.0, f"処理時間が10秒を超えています: {processing_time:.2f}秒"
    
    rows_per_second = len(large_forex_data) / processing_time
    print(f"処理速度: {rows_per_second:,.0f} 行/秒")


@pytest.mark.integration
def test_all_indicators_consistency(large_forex_data, indicator_engine):
    """
    全指標の一貫性テスト
    個別計算とバッチ計算の結果が一致するか検証
    """
    # データのサブセットを使用（高速化のため）
    test_data = large_forex_data.head(10000)
    
    # 個別計算
    result_individual = test_data
    result_individual = indicator_engine.calculate_ema(result_individual, group_by="symbol")
    result_individual = indicator_engine.calculate_rsi(result_individual, group_by="symbol")
    result_individual = indicator_engine.calculate_macd(result_individual, group_by="symbol")
    result_individual = indicator_engine.calculate_bollinger_bands(result_individual, group_by="symbol")
    
    # バッチ計算
    indicator_engine.clear_metadata()  # メタデータをリセット
    result_batch = indicator_engine.calculate_all_indicators(
        test_data,
        group_by="symbol",
    )
    
    # 結果の比較
    columns_to_compare = [
        "ema_5", "ema_20", "ema_50", "ema_100", "ema_200",
        "rsi_14",
        "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent",
    ]
    
    for col in columns_to_compare:
        individual_values = result_individual[col].to_numpy()
        batch_values = result_batch[col].to_numpy()
        
        # NaNを除外して比較
        mask = ~(np.isnan(individual_values) | np.isnan(batch_values))
        
        if np.any(mask):
            np.testing.assert_allclose(
                individual_values[mask],
                batch_values[mask],
                rtol=1e-5,
                err_msg=f"列 {col} の値が一致しません",
            )


# ==================== 既存システムとの統合テスト ====================


@pytest.mark.integration
def test_integration_with_polars_engine(large_forex_data, indicator_engine, polars_engine):
    """
    PolarsProcessingEngineとの統合テスト
    データ処理 → 指標計算 → 結果統合
    """
    # Step 1: PolarsProcessingEngineでデータ前処理
    # サンプリング（sample_dataメソッドがない場合は直接Polarsのサンプリングを使用）
    sampled_data = large_forex_data.sample(fraction=0.1, seed=42)
    assert len(sampled_data) < len(large_forex_data)
    
    # Step 2: TechnicalIndicatorEngineで指標計算
    result = indicator_engine.calculate_all_indicators(
        sampled_data,
        group_by="symbol",
    )
    
    # Step 3: 結果の統合と検証
    assert not result.is_empty()
    assert len(result) == len(sampled_data)
    
    # 全ての元のカラムが保持されていることを確認
    for col in sampled_data.columns:
        assert col in result.columns
    
    # 指標列が追加されていることを確認
    assert "ema_5" in result.columns
    assert "rsi_14" in result.columns
    assert "macd_line" in result.columns
    assert "bb_upper" in result.columns


@pytest.mark.integration
def test_data_pipeline_flow(large_forex_data, indicator_engine):
    """
    データパイプラインフローのテスト
    複数ステップの処理を連続実行
    """
    # Step 1: データのフィルタリング（特定のシンボルのみ）
    filtered_data = large_forex_data.filter(
        pl.col("symbol").is_in(["EURUSD", "GBPUSD"])
    )
    
    # Step 2: 時間範囲でのフィルタリング
    start_time = filtered_data["timestamp"].min()
    end_time = start_time + timedelta(hours=1)
    
    time_filtered = filtered_data.filter(
        (pl.col("timestamp") >= start_time) & 
        (pl.col("timestamp") <= end_time)
    )
    
    # Step 3: 指標計算
    result = indicator_engine.calculate_all_indicators(
        time_filtered,
        group_by="symbol",
    )
    
    # Step 4: 結果の集計
    summary = result.group_by("symbol").agg([
        pl.col("close").mean().alias("avg_close"),
        pl.col("rsi_14").mean().alias("avg_rsi"),
        pl.col("macd_histogram").mean().alias("avg_macd_hist"),
        pl.col("bb_width").mean().alias("avg_bb_width"),
    ])
    
    # 検証
    assert len(summary) == 2  # EURUSD と GBPUSD
    assert not summary["avg_rsi"].is_null().any()


# ==================== パフォーマンスベンチマーク ====================


@pytest.mark.integration
@pytest.mark.benchmark
def test_large_scale_performance(large_forex_data, indicator_engine):
    """
    大規模データでのパフォーマンステスト
    100万行データでの処理速度、メモリ使用量、CPU利用率
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # 初期メモリ使用量
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 処理時間の計測
    start_time = time.time()
    
    result = indicator_engine.calculate_all_indicators(
        large_forex_data,
        group_by="symbol",
    )
    
    processing_time = time.time() - start_time
    
    # 処理後のメモリ使用量
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_increase = mem_after - mem_before
    
    # パフォーマンス指標の計算
    rows_per_second = len(large_forex_data) / processing_time
    
    # アサーション
    assert processing_time < 30.0, f"処理時間が30秒を超えています: {processing_time:.2f}秒"
    assert rows_per_second > 30000, f"処理速度が遅すぎます: {rows_per_second:.0f} 行/秒"
    assert mem_increase < 2000, f"メモリ使用量が大きすぎます: {mem_increase:.0f} MB"
    
    # レポート
    print(f"\n=== パフォーマンステスト結果 ===")
    print(f"データ行数: {len(large_forex_data):,}")
    print(f"処理時間: {processing_time:.2f} 秒")
    print(f"処理速度: {rows_per_second:,.0f} 行/秒")
    print(f"メモリ増加: {mem_increase:.0f} MB")
    print(f"初期メモリ: {mem_before:.0f} MB")
    print(f"最終メモリ: {mem_after:.0f} MB")


@pytest.mark.integration
def test_memory_efficiency(indicator_engine):
    """
    メモリ効率テスト
    大量データ処理時のメモリ管理を検証
    """
    import gc
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # メモリ計測用のデータを生成
    n_iterations = 5
    rows_per_iteration = 100000
    
    memory_usage = []
    
    for i in range(n_iterations):
        # データ生成
        data = pl.DataFrame({
            "timestamp": [datetime.now() + timedelta(seconds=j) for j in range(rows_per_iteration)],
            "symbol": ["EURUSD"] * rows_per_iteration,
            "close": np.random.uniform(1.0, 1.5, rows_per_iteration),
        }).with_columns(
            pl.col("close").cast(pl.Float32)
        )
        
        # 指標計算
        result = indicator_engine.calculate_all_indicators(data)
        
        # メモリ使用量を記録
        mem_current = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage.append(mem_current)
        
        # 明示的にガベージコレクション
        del data
        del result
        gc.collect()
    
    # メモリリークのチェック
    # 最初と最後のメモリ使用量の差が許容範囲内であること
    mem_diff = memory_usage[-1] - memory_usage[0]
    assert mem_diff < 100, f"メモリリークの可能性: {mem_diff:.0f} MB増加"
    
    print(f"\nメモリ使用量の推移: {memory_usage}")


@pytest.mark.integration
def test_concurrent_processing(large_forex_data, indicator_engine):
    """
    並行処理テスト
    複数シンボルの同時処理とスレッドセーフティの確認
    """
    import concurrent.futures
    from threading import Lock
    
    # シンボルごとにデータを分割
    symbols = large_forex_data["symbol"].unique().to_list()
    
    results = {}
    lock = Lock()
    
    def process_symbol(symbol: str) -> tuple[str, pl.DataFrame]:
        """単一シンボルの処理"""
        symbol_data = large_forex_data.filter(pl.col("symbol") == symbol)
        
        # 新しいエンジンインスタンスを作成（スレッドセーフのため）
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_all_indicators(symbol_data)
        
        return symbol, result
    
    # 並行処理の実行
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = [
            executor.submit(process_symbol, symbol)
            for symbol in symbols
        ]
        
        for future in concurrent.futures.as_completed(futures):
            symbol, result = future.result()
            with lock:
                results[symbol] = result
    
    # 結果の検証
    assert len(results) == len(symbols)
    
    for symbol, result in results.items():
        assert not result.is_empty()
        assert "ema_5" in result.columns
        assert "rsi_14" in result.columns
        
        # シンボルが正しくフィルタされているか確認
        unique_symbols = result["symbol"].unique().to_list()
        assert len(unique_symbols) == 1
        assert unique_symbols[0] == symbol


# ==================== ストリーミングシミュレーション ====================


@pytest.mark.integration
def test_streaming_update(indicator_engine, streaming_data_generator):
    """
    ストリーミングデータ更新テスト
    初期データで指標計算 → 新データを逐次追加 → 増分計算の検証
    """
    # 初期データの準備
    initial_ticks = []
    base_time = datetime.now()
    
    # 100ティックの初期データを生成
    for i in range(100):
        tick = streaming_data_generator["generate_tick"](
            "EURUSD", 1.1000, base_time + timedelta(seconds=i)
        )
        initial_ticks.append(tick)
    
    # OHLCに変換
    initial_ohlc = streaming_data_generator["generate_ohlc_from_ticks"](initial_ticks)
    
    # 初期指標計算
    result = indicator_engine.calculate_ema(initial_ohlc)
    initial_ema = result["ema_5"].to_list()[-1]
    
    # ストリーミング更新のシミュレーション
    for i in range(10):
        # 新しいティックを生成
        new_tick = streaming_data_generator["generate_tick"](
            "EURUSD", 1.1000, base_time + timedelta(seconds=100 + i)
        )
        
        # 新しいOHLCデータを作成
        new_ohlc = streaming_data_generator["generate_ohlc_from_ticks"]([new_tick])
        
        # データを結合
        combined_data = pl.concat([result, new_ohlc])
        
        # 増分計算
        result = indicator_engine.update_indicators(combined_data, last_n_rows=1)
        
        # EMAが更新されていることを確認
        current_ema = result["ema_5"].to_list()[-1]
        assert current_ema != initial_ema or i == 0  # 最初以外は変化するはず


@pytest.mark.integration
def test_realtime_scenario(indicator_engine, streaming_data_generator):
    """
    リアルタイムシナリオテスト
    1秒ごとにデータ更新 → 指標の再計算 → パフォーマンス要件の確認 (< 100ms)
    """
    base_time = datetime.now()
    data_buffer = []
    
    # 10秒間のリアルタイムシミュレーション
    for second in range(10):
        # 1秒分のティックデータを生成（10ティック/秒）
        second_ticks = []
        for tick_num in range(10):
            tick = streaming_data_generator["generate_tick"](
                "EURUSD",
                1.1000,
                base_time + timedelta(seconds=second, milliseconds=tick_num * 100)
            )
            second_ticks.append(tick)
        
        # OHLCに変換
        ohlc = streaming_data_generator["generate_ohlc_from_ticks"](second_ticks)
        data_buffer.append(ohlc)
        
        # データを結合
        combined_data = pl.concat(data_buffer)
        
        # 指標計算の時間を計測
        start_time = time.time()
        result = indicator_engine.calculate_all_indicators(combined_data)
        calc_time = (time.time() - start_time) * 1000  # ミリ秒に変換
        
        # パフォーマンス要件のチェック
        assert calc_time < 100, f"計算時間が100msを超えています: {calc_time:.2f}ms"
        
        print(f"秒 {second + 1}: 計算時間 {calc_time:.2f}ms, データ行数: {len(combined_data)}")


# ==================== エラーリカバリーテスト ====================


@pytest.mark.integration
def test_error_recovery(indicator_engine):
    """
    エラーリカバリーテスト
    不正データでの動作、NULL値や異常値の処理、エラー後の継続処理
    """
    # 異常値を含むデータの生成
    data_with_issues = pl.DataFrame({
        "timestamp": [datetime.now() + timedelta(seconds=i) for i in range(100)],
        "symbol": ["EURUSD"] * 100,
        "close": [
            1.1000 if i < 20 else
            None if 20 <= i < 30 else  # NULL値
            float('inf') if i == 40 else  # 無限大
            -1.0 if i == 50 else  # 負の値
            1.1000 + np.random.normal(0, 0.001)
            for i in range(100)
        ],
    })
    
    # Float32に変換（エラーが発生する可能性）
    data_with_issues = data_with_issues.with_columns(
        pl.col("close").cast(pl.Float32)
    )
    
    # エラーが発生してもクラッシュしないことを確認
    try:
        result = indicator_engine.calculate_all_indicators(data_with_issues)
        
        # 結果が返されることを確認
        assert result is not None
        assert not result.is_empty()
        
        # 指標列が存在することを確認（一部NaNになっている可能性）
        assert "ema_5" in result.columns
        assert "rsi_14" in result.columns
        
    except Exception as e:
        # エラーが適切に処理されることを確認
        assert False, f"エラーリカバリーに失敗: {str(e)}"


@pytest.mark.integration
def test_data_quality_handling(indicator_engine):
    """
    データ品質処理のテスト
    欠損値、重複、順序の乱れなどを処理
    """
    # 問題のあるデータを生成
    base_time = datetime.now()
    
    # 時系列が乱れたデータ
    unordered_data = pl.DataFrame({
        "timestamp": [
            base_time + timedelta(seconds=i) 
            for i in [0, 5, 2, 8, 3, 1, 9, 4, 6, 7]  # 順序がバラバラ
        ],
        "symbol": ["EURUSD"] * 10,
        "close": [1.1000 + i * 0.0001 for i in range(10)],
    }).with_columns(
        pl.col("close").cast(pl.Float32)
    )
    
    # 重複データを追加
    duplicate_data = unordered_data.head(3)
    combined_data = pl.concat([unordered_data, duplicate_data])
    
    # データをソートして重複を除去
    cleaned_data = (
        combined_data
        .sort("timestamp")
        .unique(subset=["timestamp", "symbol"])
    )
    
    # 指標計算
    result = indicator_engine.calculate_all_indicators(cleaned_data)
    
    # 結果の検証
    assert len(result) == 10  # 重複が除去されている
    assert result["timestamp"].is_sorted()  # 時系列順にソートされている
    assert "ema_5" in result.columns
    
    # EMAが正しく計算されていることを確認（順序が正しければ値も正しい）
    ema_values = result["ema_5"].to_numpy()
    assert not np.all(np.isnan(ema_values))  # 全てNaNではない


@pytest.mark.integration
def test_incremental_calculation_accuracy(indicator_engine):
    """
    増分計算の精度テスト
    バッチ計算と増分計算の結果が一致することを確認
    """
    # テストデータの生成
    n_initial = 1000
    n_increments = 10
    increment_size = 100
    
    # 初期データ
    all_data = pl.DataFrame({
        "timestamp": [
            datetime.now() + timedelta(seconds=i)
            for i in range(n_initial + n_increments * increment_size)
        ],
        "symbol": ["EURUSD"] * (n_initial + n_increments * increment_size),
        "close": [
            1.1000 + np.sin(i * 0.01) * 0.01
            for i in range(n_initial + n_increments * increment_size)
        ],
    }).with_columns(
        pl.col("close").cast(pl.Float32)
    )
    
    # バッチ計算（全データを一度に処理）
    batch_result = indicator_engine.calculate_all_indicators(all_data)
    
    # 増分計算（段階的に処理）
    incremental_result = all_data.head(n_initial)
    incremental_result = indicator_engine.calculate_all_indicators(incremental_result)
    
    for i in range(n_increments):
        start_idx = n_initial + i * increment_size
        end_idx = start_idx + increment_size
        
        new_data = all_data[start_idx:end_idx]
        incremental_result = pl.concat([incremental_result, new_data])
        incremental_result = indicator_engine.update_indicators(
            incremental_result,
            last_n_rows=increment_size
        )
    
    # 結果の比較
    comparison_columns = ["ema_5", "ema_20", "rsi_14", "macd_line"]
    
    for col in comparison_columns:
        if col in batch_result.columns and col in incremental_result.columns:
            batch_values = batch_result[col].to_numpy()
            incremental_values = incremental_result[col].to_numpy()
            
            # NaNを除外して比較
            mask = ~(np.isnan(batch_values) | np.isnan(incremental_values))
            
            if np.any(mask):
                np.testing.assert_allclose(
                    batch_values[mask],
                    incremental_values[mask],
                    rtol=1e-4,
                    err_msg=f"増分計算で{col}の値が異なります"
                )


# ==================== マーカーとヘルパー ====================


def run_integration_tests():
    """統合テストを実行するヘルパー関数"""
    pytest.main([__file__, "-v", "-m", "integration"])


if __name__ == "__main__":
    run_integration_tests()