"""
テクニカル指標計算エンジンのユニットテスト

このモジュールは、TechnicalIndicatorEngineクラスのテストを提供します。
EMA（指数移動平均）計算の精度と増分計算機能をテストします。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_price_data() -> pl.DataFrame:
    """テスト用の価格データを生成
    
    実際の為替データに近い価格変動を持つデータを生成します。
    """
    np.random.seed(42)
    n_rows = 500  # EMA計算に十分なデータ量
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # リアリスティックな価格変動を生成（ランダムウォーク）
    base_price = 1.1000
    returns = np.random.normal(0, 0.0001, n_rows)  # 平均0、標準偏差0.01%のリターン
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCデータを生成
    opens = prices * (1 + np.random.uniform(-0.0001, 0.0001, n_rows))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.uniform(0, 0.0002, n_rows)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.uniform(0, 0.0002, n_rows)))
    closes = prices
    volumes = np.random.uniform(1000, 10000, n_rows)
    
    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["EURUSD"] * n_rows,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }).with_columns([
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
    ])


@pytest.fixture
def small_price_data() -> pl.DataFrame:
    """小さなテスト用データセット（精度検証用）"""
    # 既知のEMA計算結果を検証するための小さなデータセット
    return pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
        "symbol": ["EURUSD"] * 10,
        "close": [1.1000, 1.1010, 1.1005, 1.1015, 1.1020, 
                  1.1025, 1.1018, 1.1022, 1.1030, 1.1035],
    }).with_columns([
        pl.col("close").cast(pl.Float32),
    ])


@pytest.fixture
def incremental_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """増分計算テスト用のデータ（既存データと新規データ）"""
    base_time = datetime(2024, 1, 1)
    
    # 既存データ（100行）
    existing = pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(100)],
        "symbol": ["EURUSD"] * 100,
        "close": [1.1000 + 0.0001 * np.sin(i/10) for i in range(100)],
    }).with_columns([
        pl.col("close").cast(pl.Float32),
    ])
    
    # 新規データ（10行）
    new = pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=100+i) for i in range(10)],
        "symbol": ["EURUSD"] * 10,
        "close": [1.1000 + 0.0001 * np.sin((100+i)/10) for i in range(10)],
    }).with_columns([
        pl.col("close").cast(pl.Float32),
    ])
    
    return existing, new


class TestTechnicalIndicatorEngine:
    """TechnicalIndicatorEngineクラスのテストスイート"""
    
    def test_engine_initialization(self):
        """エンジンの初期化テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # デフォルト設定での初期化
        engine = TechnicalIndicatorEngine()
        assert engine is not None
        assert hasattr(engine, 'ema_periods')
        assert engine.ema_periods == [5, 20, 50, 100, 200]
        
        # カスタム期間での初期化
        custom_periods = [10, 30, 60]
        engine_custom = TechnicalIndicatorEngine(ema_periods=custom_periods)
        assert engine_custom.ema_periods == custom_periods
    
    def test_ema_calculation_basic(self, sample_price_data: pl.DataFrame):
        """基本的なEMA計算のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(sample_price_data)
        
        # 結果の検証
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_price_data)
        
        # EMA列が追加されていることを確認
        expected_columns = ["ema_5", "ema_20", "ema_50", "ema_100", "ema_200"]
        for col in expected_columns:
            assert col in result.columns
            assert result[col].dtype == pl.Float32
        
        # 元のデータ列も保持されていることを確認
        for col in sample_price_data.columns:
            assert col in result.columns
    
    def test_ema_calculation_accuracy(self, small_price_data: pl.DataFrame):
        """EMA計算の精度テスト
        
        手動計算した既知の値と比較して精度を検証します。
        """
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine(ema_periods=[3])  # 検証しやすい短い期間
        result = engine.calculate_ema(small_price_data)
        
        # EMA(3)の手動計算
        # alpha = 2 / (3 + 1) = 0.5
        # EMA[i] = alpha * price[i] + (1 - alpha) * EMA[i-1]
        alpha = 2 / (3 + 1)
        manual_ema = [float(small_price_data["close"][0])]
        
        for i in range(1, len(small_price_data)):
            ema_value = (alpha * float(small_price_data["close"][i]) + 
                        (1 - alpha) * manual_ema[-1])
            manual_ema.append(ema_value)
        
        # 計算結果の比較（浮動小数点誤差を考慮）
        calculated_ema = result["ema_3"].to_list()
        for i in range(len(manual_ema)):
            assert abs(calculated_ema[i] - manual_ema[i]) < 1e-5, (
                f"EMA値が一致しません。インデックス {i}: "
                f"計算値={calculated_ema[i]}, 期待値={manual_ema[i]}"
            )
    
    def test_ema_with_null_values(self):
        """NULL値を含むデータでのEMA計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # NULL値を含むデータ
        data_with_nulls = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
            "symbol": ["EURUSD"] * 10,
            "close": [1.1000, None, 1.1005, 1.1015, None, 
                     1.1025, 1.1018, None, 1.1030, 1.1035],
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(data_with_nulls)
        
        # NULL値が適切に処理されていることを確認
        assert isinstance(result, pl.DataFrame)
        assert "ema_5" in result.columns
        
        # EMA列にNULL値が含まれることを確認（NULL値は伝播する）
        assert result["ema_5"].null_count() <= data_with_nulls["close"].null_count()
    
    def test_incremental_ema_update(self, incremental_data: tuple[pl.DataFrame, pl.DataFrame]):
        """増分EMA更新のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        existing, new = incremental_data
        engine = TechnicalIndicatorEngine()
        
        # 既存データでEMAを計算
        existing_with_ema = engine.calculate_ema(existing)
        
        # 増分更新
        updated = engine.update_ema_incremental(existing_with_ema, new)
        
        # 検証
        assert isinstance(updated, pl.DataFrame)
        assert len(updated) == len(existing) + len(new)
        
        # EMA列が存在することを確認
        for period in engine.ema_periods:
            col_name = f"ema_{period}"
            assert col_name in updated.columns
            
            # 最後の値が更新されていることを確認
            last_existing_ema = float(existing_with_ema[col_name][-1])
            last_updated_ema = float(updated[col_name][-1])
            assert last_existing_ema != last_updated_ema, (
                f"{col_name}が更新されていません"
            )
    
    def test_multiple_symbols(self):
        """複数シンボルのEMA計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 複数シンボルのデータ
        n_rows_per_symbol = 50
        base_time = datetime(2024, 1, 1)
        
        data = []
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            base_price = {"EURUSD": 1.1000, "GBPUSD": 1.2500, "USDJPY": 110.00}[symbol]
            for i in range(n_rows_per_symbol):
                data.append({
                    "timestamp": base_time + timedelta(minutes=i),
                    "symbol": symbol,
                    "close": base_price * (1 + 0.0001 * np.sin(i/10)),
                })
        
        df = pl.DataFrame(data).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(df, group_by="symbol")
        
        # 各シンボルごとにEMAが計算されていることを確認
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            symbol_data = result.filter(pl.col("symbol") == symbol)
            assert len(symbol_data) == n_rows_per_symbol
            assert "ema_5" in symbol_data.columns
            
            # EMA値が適切な範囲内にあることを確認
            ema_values = symbol_data["ema_5"].drop_nulls()
            assert ema_values.min() > 0
            assert ema_values.max() < 1000  # 妥当な範囲内
    
    def test_performance_large_dataset(self, benchmark_timer):
        """大規模データセットでのパフォーマンステスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 100万行のデータを生成
        n_rows = 1_000_000
        base_time = datetime(2024, 1, 1)
        
        # 効率的なデータ生成
        timestamps = [base_time + timedelta(seconds=i) for i in range(n_rows)]
        prices = 1.1000 + 0.0001 * np.sin(np.arange(n_rows) / 1000)
        
        large_data = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": ["EURUSD"] * n_rows,
            "close": prices,
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        
        # パフォーマンス測定
        with benchmark_timer() as timer:
            result = engine.calculate_ema(large_data)
        
        # 結果の検証
        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_rows
        
        # パフォーマンス基準（1秒以内に完了すべき）
        assert timer.elapsed < 1.0, f"EMA計算が遅すぎます: {timer.elapsed:.2f}秒"
        
        print(f"100万行のEMA計算時間: {timer.elapsed:.4f}秒")
    
    def test_error_handling_invalid_data(self):
        """無効なデータでのエラーハンドリングテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 空のDataFrame
        empty_df = pl.DataFrame()
        with pytest.raises(ValueError, match="空のDataFrame"):
            engine.calculate_ema(empty_df)
        
        # close列がないDataFrame
        no_close_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "symbol": ["EURUSD"],
            "open": [1.1000],
        })
        with pytest.raises(ValueError, match="close列が必要"):
            engine.calculate_ema(no_close_df)
        
        # 無効な期間
        with pytest.raises(ValueError, match="期間は正の整数"):
            TechnicalIndicatorEngine(ema_periods=[0, -1, 5])
    
    def test_memory_efficiency(self):
        """メモリ効率のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        import psutil
        import gc
        
        # メモリ使用量の測定
        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大規模データの処理
        n_rows = 500_000
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(n_rows)],
            "symbol": ["EURUSD"] * n_rows,
            "close": np.random.uniform(1.0, 1.2, n_rows),
        }).with_columns([
            pl.col("close").cast(pl.Float32),  # Float32で メモリ効率化
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(data)
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # メモリ増加が妥当な範囲内であることを確認（500MB以下）
        assert memory_increase < 500, f"メモリ使用量が大きすぎます: {memory_increase:.2f}MB"
        
        # Float32が維持されていることを確認
        for col in result.columns:
            if col.startswith("ema_"):
                assert result[col].dtype == pl.Float32
    
    @pytest.mark.parametrize("period", [5, 20, 50, 100, 200])
    def test_ema_convergence(self, period: int):
        """各EMA期間での収束性テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 一定値のデータでEMAが収束することを確認
        constant_value = 1.1500
        n_rows = period * 10  # 期間の10倍のデータ
        
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_rows)],
            "symbol": ["EURUSD"] * n_rows,
            "close": [constant_value] * n_rows,
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine(ema_periods=[period])
        result = engine.calculate_ema(data)
        
        # 最後のEMA値が元の値に収束していることを確認
        col_name = f"ema_{period}"
        last_ema = float(result[col_name][-1])
        
        # 収束誤差は0.01%以内
        tolerance = constant_value * 0.0001
        assert abs(last_ema - constant_value) < tolerance, (
            f"EMA({period})が収束していません: "
            f"期待値={constant_value}, 実際={last_ema}"
        )
    
    def test_ema_with_custom_column(self):
        """カスタム列でのEMA計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # カスタム列を持つデータ
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
            "symbol": ["EURUSD"] * 50,
            "custom_price": [1.1000 + 0.0001 * i for i in range(50)],
        }).with_columns([
            pl.col("custom_price").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(data, price_column="custom_price")
        
        # EMAが計算されていることを確認
        assert "ema_5" in result.columns
        assert result["ema_5"].null_count() < len(result)  # 一部の値は計算されている
    
    def test_pipeline_integration_readiness(self, sample_price_data: pl.DataFrame):
        """パイプライン統合の準備テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # メソッドチェーン可能な設計であることを確認
        result = (
            sample_price_data
            .pipe(engine.calculate_ema)
            .filter(pl.col("ema_5").is_not_null())
            .select(["timestamp", "symbol", "close", "ema_5", "ema_20"])
        )
        
        assert isinstance(result, pl.DataFrame)
        assert len(result.columns) == 5
        assert "ema_5" in result.columns
        assert "ema_20" in result.columns


@pytest.mark.unit
class TestEdgeCases:
    """エッジケースのテストクラス"""
    
    def test_single_row_data(self):
        """1行のみのデータでのEMA計算"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        single_row = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "symbol": ["EURUSD"],
            "close": [1.1000],
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(single_row)
        
        # 1行でも処理できることを確認
        assert len(result) == 1
        assert "ema_5" in result.columns
        # 最初の値はcloseと同じになるはず
        assert float(result["ema_5"][0]) == float(single_row["close"][0])
    
    def test_extreme_values(self):
        """極端な値でのEMA計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 極端な値を含むデータ
        extreme_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
            "symbol": ["EURUSD"] * 10,
            "close": [1.1000, 1.1000, 10000.0, 1.1000, 0.0001, 
                     1.1000, 1.1000, 1.1000, 1.1000, 1.1000],
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(extreme_data)
        
        # 計算が完了することを確認
        assert isinstance(result, pl.DataFrame)
        assert "ema_5" in result.columns
        
        # EMA値が妥当な範囲に収まることを確認（スムージング効果）
        ema_values = result["ema_5"].drop_nulls()
        assert ema_values.max() < 10000.0  # 極端な値より小さい
        assert ema_values.min() > 0.0001   # 極端な値より大きい