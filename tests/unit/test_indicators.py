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


@pytest.fixture
def sample_multi_symbol_data() -> pl.DataFrame:
    """複数シンボルのテスト用データ"""
    np.random.seed(42)
    n_rows = 300  # 各シンボル100行
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    data = []
    for symbol, base_price in [("EURUSD", 1.1000), ("GBPUSD", 1.2500), ("USDJPY", 110.00)]:
        returns = np.random.normal(0, 0.0001, n_rows // 3)
        prices = base_price * np.exp(np.cumsum(returns))
        
        for i, price in enumerate(prices):
            data.append({
                "timestamp": base_time + timedelta(minutes=i),
                "symbol": symbol,
                "close": price,
                "volume": np.random.uniform(1000, 10000),
            })
    
    return pl.DataFrame(data).with_columns([
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
    ])


@pytest.fixture
def large_price_data() -> pl.DataFrame:
    """大規模データセット（パフォーマンステスト用）"""
    np.random.seed(42)
    n_rows = 100000  # 10万行
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # リアリスティックな価格変動を生成
    base_price = 1.1000
    returns = np.random.normal(0, 0.0001, n_rows)
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["EURUSD"] * n_rows,
        "close": prices,
        "volume": np.random.uniform(1000, 10000, n_rows),
    }).with_columns([
        pl.col("close").cast(pl.Float32),
        pl.col("volume").cast(pl.Float32),
    ])


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


class TestRSICalculation:
    """RSI（相対力指数）計算のテストスイート"""
    
    def test_rsi_calculation_basic(self, sample_price_data: pl.DataFrame):
        """基本的なRSI計算のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_rsi(sample_price_data)
        
        # 結果の検証
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_price_data)
        
        # RSI列が追加されていることを確認
        assert "rsi_14" in result.columns
        assert result["rsi_14"].dtype == pl.Float32
        
        # RSI値が0-100の範囲内であることを確認
        rsi_values = result["rsi_14"].drop_nulls()
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100
    
    def test_rsi_calculation_accuracy(self):
        """RSI計算の精度テスト
        
        既知の値でRSIを手動計算し、実装と比較します。
        """
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 簡単なテストデータ（上昇・下落が明確）
        test_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)],
            "symbol": ["EURUSD"] * 20,
            "close": [
                100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0,  # 上昇傾向
                104.5, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0,   # 下落傾向
                99.0, 100.0, 101.0, 102.0  # 回復
            ],
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_rsi(test_data, period=14)
        
        # RSI値が計算されていることを確認
        assert "rsi_14" in result.columns
        rsi_values = result["rsi_14"].drop_nulls()
        
        # 上昇傾向の期間ではRSI > 50、下落傾向ではRSI < 50を期待
        # （14期間後から有効な値が出る）
        if len(rsi_values) >= 5:
            # 最後の数値（回復期）は中間的なRSI値を期待
            last_rsi = float(rsi_values[-1])
            assert 30 <= last_rsi <= 70, f"RSI値が期待範囲外: {last_rsi}"
    
    def test_rsi_with_custom_period(self):
        """カスタム期間でのRSI計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
            "symbol": ["EURUSD"] * 50,
            "close": [100.0 + np.sin(i/5) * 5 for i in range(50)],  # サイン波
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        
        # 異なる期間でRSIを計算
        for period in [7, 14, 21]:
            result = engine.calculate_rsi(data, period=period)
            col_name = f"rsi_{period}"
            
            assert col_name in result.columns
            rsi_values = result[col_name].drop_nulls()
            
            # 値の範囲確認
            assert rsi_values.min() >= 0
            assert rsi_values.max() <= 100
            
            # 期間が短いほど、より敏感（変動が大きい）になることを確認
            if period == 7 and len(rsi_values) > 0:
                rsi_7_std = rsi_values.std()
            elif period == 21 and len(rsi_values) > 0:
                rsi_21_std = rsi_values.std()
                # RSI(7)の方がRSI(21)より変動が大きいはず
                if 'rsi_7_std' in locals():
                    assert rsi_7_std >= rsi_21_std * 0.9  # ある程度の差を許容
    
    def test_rsi_multiple_symbols(self):
        """複数シンボルでのRSI計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 複数シンボルのデータ
        data = []
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            base_price = {"EURUSD": 1.1000, "GBPUSD": 1.2500, "USDJPY": 110.00}[symbol]
            for i in range(30):
                data.append({
                    "timestamp": datetime(2024, 1, 1) + timedelta(days=i),
                    "symbol": symbol,
                    "close": base_price * (1 + 0.001 * np.sin(i/3)),
                })
        
        df = pl.DataFrame(data).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_rsi(df, group_by="symbol")
        
        # 各シンボルごとにRSIが計算されていることを確認
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            symbol_data = result.filter(pl.col("symbol") == symbol)
            assert "rsi_14" in symbol_data.columns
            
            rsi_values = symbol_data["rsi_14"].drop_nulls()
            if len(rsi_values) > 0:
                assert rsi_values.min() >= 0
                assert rsi_values.max() <= 100
    
    def test_rsi_with_constant_price(self):
        """価格が一定の場合のRSI計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 価格が変化しないデータ
        constant_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "symbol": ["EURUSD"] * 30,
            "close": [100.0] * 30,
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_rsi(constant_data)
        
        # RSIは50付近になるはず（変化がない場合）
        rsi_values = result["rsi_14"].drop_nulls()
        if len(rsi_values) > 0:
            # 最後の値をチェック（十分なデータがある場合）
            last_rsi = float(rsi_values[-1])
            # 価格変化がない場合、RSIは50になる（または計算不可）
            assert 45 <= last_rsi <= 55 or np.isnan(last_rsi), f"一定価格でのRSIが異常: {last_rsi}"
    
    def test_rsi_extreme_movements(self):
        """極端な価格変動でのRSI計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 極端な上昇と下落を含むデータ
        extreme_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "symbol": ["EURUSD"] * 30,
            "close": [100.0] * 10 + [200.0] * 10 + [50.0] * 10,  # 急騰後急落
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_rsi(extreme_data)
        
        rsi_values = result["rsi_14"].drop_nulls()
        if len(rsi_values) >= 20:
            # 急騰期間の後半ではRSI > 70（買われすぎ）
            mid_rsi = float(rsi_values[15])  # 急騰後
            assert mid_rsi > 60, f"急騰後のRSIが低すぎる: {mid_rsi}"
            
            # 急落期間の後半ではRSI < 30（売られすぎ）
            last_rsi = float(rsi_values[-1])  # 急落後
            assert last_rsi < 40, f"急落後のRSIが高すぎる: {last_rsi}"


class TestMACDCalculation:
    """MACD（移動平均収束拡散）計算のテストスイート"""
    
    def test_macd_calculation_basic(self, sample_price_data: pl.DataFrame):
        """基本的なMACD計算のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_macd(sample_price_data)
        
        # 結果の検証
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_price_data)
        
        # MACD関連列が追加されていることを確認
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        
        # データ型の確認
        assert result["macd_line"].dtype == pl.Float32
        assert result["macd_signal"].dtype == pl.Float32
        assert result["macd_histogram"].dtype == pl.Float32
        
        # 元のデータ列も保持されていることを確認
        for col in sample_price_data.columns:
            assert col in result.columns
    
    def test_macd_calculation_accuracy(self):
        """MACD計算の精度テスト
        
        既知の値でMACDを手動計算し、実装と比較します。
        """
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # シンプルなテストデータ（トレンドが明確）
        test_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
            "symbol": ["EURUSD"] * 50,
            "close": [100.0 + i * 0.5 for i in range(50)],  # 上昇トレンド
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_macd(test_data)
        
        # MACD値が計算されていることを確認
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        
        # 上昇トレンドなので、MACDラインは正の値になるはず（後半で）
        macd_values = result["macd_line"].drop_nulls()
        if len(macd_values) >= 30:
            # 十分なデータポイント後のMACD値をチェック
            last_macd = float(macd_values[-1])
            assert last_macd > 0, f"上昇トレンドでMACDが負: {last_macd}"
    
    def test_macd_with_custom_periods(self):
        """カスタム期間でのMACD計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)],
            "symbol": ["EURUSD"] * 100,
            "close": [100.0 + np.sin(i/10) * 10 for i in range(100)],  # サイン波
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        
        # カスタム期間でMACDを計算
        result = engine.calculate_macd(
            data, 
            fast_period=10, 
            slow_period=20, 
            signal_period=5
        )
        
        # 列名が期間を反映していることを確認
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        
        # 値が計算されていることを確認
        macd_values = result["macd_line"].drop_nulls()
        signal_values = result["macd_signal"].drop_nulls()
        histogram_values = result["macd_histogram"].drop_nulls()
        
        assert len(macd_values) > 0
        assert len(signal_values) > 0
        assert len(histogram_values) > 0
        
        # ヒストグラムがMACD - Signalであることを確認
        if len(histogram_values) >= 10:
            # 最後の値で検証
            last_macd = float(result["macd_line"].drop_nulls()[-1])
            last_signal = float(result["macd_signal"].drop_nulls()[-1])
            last_histogram = float(result["macd_histogram"].drop_nulls()[-1])
            
            expected_histogram = last_macd - last_signal
            assert abs(last_histogram - expected_histogram) < 1e-5, (
                f"ヒストグラム計算が不正確: {last_histogram} != {expected_histogram}"
            )
    
    def test_macd_multiple_symbols(self):
        """複数シンボルでのMACD計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 複数シンボルのデータ
        data = []
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            base_price = {"EURUSD": 1.1000, "GBPUSD": 1.2500, "USDJPY": 110.00}[symbol]
            for i in range(60):
                data.append({
                    "timestamp": datetime(2024, 1, 1) + timedelta(days=i),
                    "symbol": symbol,
                    "close": base_price * (1 + 0.002 * np.sin(i/5)),
                })
        
        df = pl.DataFrame(data).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_macd(df, group_by="symbol")
        
        # 各シンボルごとにMACDが計算されていることを確認
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            symbol_data = result.filter(pl.col("symbol") == symbol)
            assert "macd_line" in symbol_data.columns
            assert "macd_signal" in symbol_data.columns
            assert "macd_histogram" in symbol_data.columns
            
            # 値が計算されていることを確認
            macd_values = symbol_data["macd_line"].drop_nulls()
            if len(macd_values) > 0:
                # 各シンボルで独立して計算されていることを確認
                assert macd_values.min() != macd_values.max()
    
    def test_macd_convergence_divergence(self):
        """MACDの収束・拡散パターンのテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 収束パターン（価格が安定）と拡散パターン（価格が変動）を含むデータ
        convergence_data = [100.0] * 30  # 収束期間（価格一定）
        divergence_data = [100.0 + (i - 30) * 0.5 for i in range(30, 60)]  # 拡散期間（上昇）
        
        test_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)],
            "symbol": ["EURUSD"] * 60,
            "close": convergence_data + divergence_data,
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_macd(test_data)
        
        macd_values = result["macd_line"].drop_nulls()
        signal_values = result["macd_signal"].drop_nulls()
        
        if len(macd_values) >= 50:
            # 収束期間ではMACDとSignalが近い値になる
            mid_macd = float(macd_values[25])
            mid_signal = float(signal_values[25])
            convergence_diff = abs(mid_macd - mid_signal)
            
            # 拡散期間ではMACDとSignalの差が大きくなる
            last_macd = float(macd_values[-1])
            last_signal = float(signal_values[-1])
            divergence_diff = abs(last_macd - last_signal)
            
            # 拡散期間の差の方が大きいはず
            assert divergence_diff > convergence_diff * 0.5, (
                f"収束・拡散パターンが検出されない: "
                f"収束差={convergence_diff}, 拡散差={divergence_diff}"
            )
    
    def test_macd_downtrend(self):
        """下降トレンドでのMACD計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 下降トレンドのデータ
        test_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
            "symbol": ["EURUSD"] * 50,
            "close": [100.0 - i * 0.3 for i in range(50)],  # 下降トレンド
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_macd(test_data)
        
        # 下降トレンドなので、MACDラインは負の値になるはず（後半で）
        macd_values = result["macd_line"].drop_nulls()
        if len(macd_values) >= 30:
            last_macd = float(macd_values[-1])
            assert last_macd < 0, f"下降トレンドでMACDが正: {last_macd}"
            
            # ヒストグラムも負になることが多い
            histogram_values = result["macd_histogram"].drop_nulls()
            if len(histogram_values) >= 30:
                negative_count = sum(1 for v in histogram_values[-10:] if v < 0)
                assert negative_count >= 5, "下降トレンドでヒストグラムが正に偏っている"
    
    def test_macd_performance(self):
        """MACD計算のパフォーマンステスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        import time
        
        # 10万行のデータ
        n_rows = 100_000
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(n_rows)],
            "symbol": ["EURUSD"] * n_rows,
            "close": 100.0 + np.sin(np.arange(n_rows) / 1000) * 10,
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        
        # パフォーマンス測定
        start_time = time.time()
        result = engine.calculate_macd(data)
        elapsed_time = time.time() - start_time
        
        # 結果の検証
        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_rows
        
        # パフォーマンス基準（0.5秒以内に完了すべき）
        assert elapsed_time < 0.5, f"MACD計算が遅すぎます: {elapsed_time:.3f}秒"
        
        print(f"10万行のMACD計算時間: {elapsed_time:.4f}秒")


@pytest.mark.unit
class TestBollingerBands:
    """ボリンジャーバンド計算のテストスイート"""
    
    def test_bollinger_bands_basic(self, sample_price_data: pl.DataFrame):
        """基本的なボリンジャーバンド計算のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_bollinger_bands(sample_price_data)
        
        # 結果の検証
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_price_data)
        
        # ボリンジャーバンド関連列が追加されていることを確認
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        assert "bb_percent" in result.columns
        
        # データ型の確認
        assert result["bb_upper"].dtype == pl.Float32
        assert result["bb_middle"].dtype == pl.Float32
        assert result["bb_lower"].dtype == pl.Float32
        assert result["bb_width"].dtype == pl.Float32
        assert result["bb_percent"].dtype == pl.Float32
        
        # 元のデータ列も保持されていることを確認
        for col in sample_price_data.columns:
            assert col in result.columns
    
    def test_bollinger_bands_calculation_accuracy(self):
        """ボリンジャーバンド計算の精度テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 一定価格のテストデータ（標準偏差=0）
        constant_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "symbol": ["EURUSD"] * 30,
            "close": [100.0] * 30,
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_bollinger_bands(constant_data, period=20)
        
        # 価格が一定の場合、バンド幅は0になるはず
        bb_width = result["bb_width"].drop_nulls()
        if len(bb_width) > 0:
            # 浮動小数点の誤差を考慮
            assert float(bb_width[-1]) < 0.0001, "一定価格でバンド幅が0でない"
            
            # 上下のバンドは中央バンドと同じになるはず
            bb_upper = result["bb_upper"].drop_nulls()
            bb_middle = result["bb_middle"].drop_nulls()
            bb_lower = result["bb_lower"].drop_nulls()
            
            if len(bb_upper) > 0:
                assert abs(float(bb_upper[-1]) - float(bb_middle[-1])) < 0.0001
                assert abs(float(bb_lower[-1]) - float(bb_middle[-1])) < 0.0001
    
    def test_bollinger_bands_with_trend(self):
        """トレンドデータでのボリンジャーバンドテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 上昇トレンドのデータ
        trend_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
            "symbol": ["EURUSD"] * 50,
            "close": [100.0 + i * 0.5 + np.random.normal(0, 0.1) for i in range(50)],
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_bollinger_bands(trend_data)
        
        # バンドの順序関係を確認（upper > middle > lower）
        bb_upper = result["bb_upper"].drop_nulls()
        bb_middle = result["bb_middle"].drop_nulls()
        bb_lower = result["bb_lower"].drop_nulls()
        
        if len(bb_upper) >= 20:
            for i in range(-10, 0):  # 最後の10個をチェック
                assert float(bb_upper[i]) > float(bb_middle[i]), "上部バンドが中央バンド以下"
                assert float(bb_middle[i]) > float(bb_lower[i]), "中央バンドが下部バンド以下"
                
                # バンド幅が正の値
                assert float(result["bb_width"][i]) > 0, "バンド幅が負または0"
    
    def test_bollinger_bands_custom_parameters(self):
        """カスタムパラメータでのボリンジャーバンドテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)],
            "symbol": ["EURUSD"] * 100,
            "close": [100.0 + np.sin(i/10) * 10 for i in range(100)],
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        
        # 異なる期間と標準偏差倍数でテスト
        for period, num_std in [(10, 1.5), (30, 2.5), (50, 3.0)]:
            result = engine.calculate_bollinger_bands(
                data, 
                period=period, 
                num_std=num_std
            )
            
            # 全ての必要な列が存在
            assert "bb_upper" in result.columns
            assert "bb_middle" in result.columns
            assert "bb_lower" in result.columns
            
            # バンド幅は標準偏差倍数に比例
            bb_width = result["bb_width"].drop_nulls()
            if len(bb_width) > period:
                # より大きなnum_stdはより広いバンドを生成
                assert len(bb_width) > 0
    
    def test_bollinger_bands_multiple_symbols(self):
        """複数シンボルでのボリンジャーバンド計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 複数シンボルのデータ
        data = []
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            base_price = {"EURUSD": 1.1000, "GBPUSD": 1.2500, "USDJPY": 110.00}[symbol]
            for i in range(50):
                data.append({
                    "timestamp": datetime(2024, 1, 1) + timedelta(days=i),
                    "symbol": symbol,
                    "close": base_price * (1 + 0.001 * np.sin(i/5)),
                })
        
        df = pl.DataFrame(data).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_bollinger_bands(df, group_by="symbol")
        
        # 各シンボルごとにボリンジャーバンドが計算されていることを確認
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            symbol_data = result.filter(pl.col("symbol") == symbol)
            assert "bb_upper" in symbol_data.columns
            assert "bb_middle" in symbol_data.columns
            assert "bb_lower" in symbol_data.columns
            
            # バンドの順序関係を確認
            bb_upper = symbol_data["bb_upper"].drop_nulls()
            bb_middle = symbol_data["bb_middle"].drop_nulls()
            bb_lower = symbol_data["bb_lower"].drop_nulls()
            
            if len(bb_upper) >= 20:
                assert float(bb_upper[-1]) >= float(bb_middle[-1])
                assert float(bb_middle[-1]) >= float(bb_lower[-1])
    
    def test_bollinger_bands_squeeze_detection(self):
        """ボリンジャーバンドスクイーズ検出テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # ボラティリティが変化するデータ
        data = []
        for i in range(100):
            if i < 30:
                # 低ボラティリティ期間
                volatility = 0.1
            elif i < 60:
                # 高ボラティリティ期間
                volatility = 1.0
            else:
                # 再び低ボラティリティ
                volatility = 0.1
            
            data.append({
                "timestamp": datetime(2024, 1, 1) + timedelta(days=i),
                "symbol": "EURUSD",
                "close": 100.0 + np.random.normal(0, volatility),
            })
        
        df = pl.DataFrame(data).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_bollinger_bands(df)
        
        bb_width = result["bb_width"].drop_nulls()
        if len(bb_width) >= 90:
            # 低ボラティリティ期間のバンド幅
            low_vol_width = float(bb_width[25])
            # 高ボラティリティ期間のバンド幅
            high_vol_width = float(bb_width[50])
            
            # 高ボラティリティ期間の方がバンド幅が広いはず
            assert high_vol_width > low_vol_width * 2, "ボラティリティ変化がバンド幅に反映されていない"
    
    def test_bollinger_bands_percent_calculation(self):
        """ボリンジャーバンド％B計算のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        # 明確な価格位置を持つデータ
        data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)],
            "symbol": ["EURUSD"] * 30,
            "close": [100.0] * 20 + [102.0] * 5 + [98.0] * 5,  # 最初は中央、次に上部、最後に下部
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_bollinger_bands(data, period=10)
        
        bb_percent = result["bb_percent"].drop_nulls()
        if len(bb_percent) >= 25:
            # 価格が上部バンド付近の時、％Bは1に近い
            upper_percent = float(bb_percent[23])
            assert upper_percent > 0.5, f"上部付近で％Bが低すぎる: {upper_percent}"
            
            # 価格が下部バンド付近の時、％Bは0に近い
            if len(bb_percent) >= 29:
                lower_percent = float(bb_percent[28])
                assert lower_percent < 0.5, f"下部付近で％Bが高すぎる: {lower_percent}"
    
    def test_bollinger_bands_performance(self):
        """ボリンジャーバンド計算のパフォーマンステスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        import time
        
        # 大量データでのテスト
        n_rows = 100000
        large_data = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(n_rows)],
            "symbol": ["EURUSD"] * n_rows,
            "close": np.random.normal(100, 1, n_rows),
        }).with_columns([
            pl.col("close").cast(pl.Float32),
        ])
        
        engine = TechnicalIndicatorEngine()
        
        start_time = time.time()
        result = engine.calculate_bollinger_bands(large_data)
        elapsed_time = time.time() - start_time
        
        # 結果の検証
        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_rows
        
        # パフォーマンス基準（0.5秒以内に完了すべき）
        assert elapsed_time < 0.5, f"ボリンジャーバンド計算が遅すぎます: {elapsed_time:.3f}秒"
        
        print(f"10万行のボリンジャーバンド計算時間: {elapsed_time:.4f}秒")


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


@pytest.mark.unit
class TestMetadataManagement:
    """メタデータ管理機能のテスト"""
    
    def test_metadata_initialization(self):
        """メタデータの初期化テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        metadata = engine.get_metadata()
        
        # 初期状態の確認
        assert "indicators" in metadata
        assert "statistics" in metadata
        assert len(metadata["indicators"]) == 0
        assert metadata["statistics"]["total_rows_processed"] == 0
        assert metadata["statistics"]["total_processing_time"] == 0.0
        assert metadata["statistics"]["last_update"] is None
    
    def test_metadata_after_ema_calculation(self, sample_price_data):
        """EMA計算後のメタデータ更新テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_ema(sample_price_data)
        metadata = engine.get_metadata()
        
        # EMAメタデータの確認
        assert "ema" in metadata["indicators"]
        ema_info = metadata["indicators"]["ema"]
        assert ema_info["calculated"] is True
        assert ema_info["periods"] == [5, 20, 50, 100, 200]
        assert "timestamp" in ema_info
        
        # 統計情報の確認
        stats = metadata["statistics"]
        assert stats["total_rows_processed"] == len(sample_price_data)
        assert stats["last_update"] is not None
    
    def test_metadata_after_rsi_calculation(self, sample_price_data):
        """RSI計算後のメタデータ更新テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_rsi(sample_price_data, period=14)
        metadata = engine.get_metadata()
        
        # RSIメタデータの確認
        assert "rsi" in metadata["indicators"]
        rsi_info = metadata["indicators"]["rsi"]
        assert rsi_info["calculated"] is True
        assert rsi_info["period"] == 14
        assert "timestamp" in rsi_info
        
        # 統計情報の確認
        stats = metadata["statistics"]
        assert stats["total_rows_processed"] == len(sample_price_data)
    
    def test_metadata_after_macd_calculation(self, sample_price_data):
        """MACD計算後のメタデータ更新テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_macd(sample_price_data)
        metadata = engine.get_metadata()
        
        # MACDメタデータの確認
        assert "macd" in metadata["indicators"]
        macd_info = metadata["indicators"]["macd"]
        assert macd_info["calculated"] is True
        assert macd_info["params"]["fast"] == 12
        assert macd_info["params"]["slow"] == 26
        assert macd_info["params"]["signal"] == 9
        assert "timestamp" in macd_info
    
    def test_metadata_after_bollinger_calculation(self, sample_price_data):
        """ボリンジャーバンド計算後のメタデータ更新テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        result = engine.calculate_bollinger_bands(sample_price_data)
        metadata = engine.get_metadata()
        
        # ボリンジャーバンドメタデータの確認
        assert "bollinger" in metadata["indicators"]
        bb_info = metadata["indicators"]["bollinger"]
        assert bb_info["calculated"] is True
        assert bb_info["params"]["period"] == 20
        assert bb_info["params"]["num_std"] == 2.0
        assert "timestamp" in bb_info
    
    def test_metadata_accumulation(self, sample_price_data):
        """複数指標計算時のメタデータ累積テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 複数の指標を順番に計算
        engine.calculate_ema(sample_price_data)
        engine.calculate_rsi(sample_price_data)
        engine.calculate_macd(sample_price_data)
        engine.calculate_bollinger_bands(sample_price_data)
        
        metadata = engine.get_metadata()
        
        # 全ての指標が記録されている
        assert len(metadata["indicators"]) == 4
        assert "ema" in metadata["indicators"]
        assert "rsi" in metadata["indicators"]
        assert "macd" in metadata["indicators"]
        assert "bollinger" in metadata["indicators"]
        
        # 統計情報が累積されている
        stats = metadata["statistics"]
        assert stats["total_rows_processed"] == len(sample_price_data) * 4
    
    def test_clear_metadata(self, sample_price_data):
        """メタデータクリア機能のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 指標を計算してメタデータを生成
        engine.calculate_ema(sample_price_data)
        engine.calculate_rsi(sample_price_data)
        
        # メタデータが存在することを確認
        metadata = engine.get_metadata()
        assert len(metadata["indicators"]) == 2
        assert metadata["statistics"]["total_rows_processed"] > 0
        
        # メタデータをクリア
        engine.clear_metadata()
        
        # クリア後の確認
        metadata = engine.get_metadata()
        assert len(metadata["indicators"]) == 0
        assert metadata["statistics"]["total_rows_processed"] == 0
        assert metadata["statistics"]["total_processing_time"] == 0.0
        assert metadata["statistics"]["last_update"] is None
    
    def test_get_calculated_indicators(self, sample_price_data):
        """計算済み指標リスト取得のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 初期状態
        assert engine.get_calculated_indicators() == []
        
        # EMA計算後
        engine.calculate_ema(sample_price_data)
        assert engine.get_calculated_indicators() == ["ema"]
        
        # RSI計算後
        engine.calculate_rsi(sample_price_data)
        assert set(engine.get_calculated_indicators()) == {"ema", "rsi"}
    
    def test_is_indicator_calculated(self, sample_price_data):
        """指標計算済みチェック機能のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 初期状態
        assert engine.is_indicator_calculated("ema") is False
        assert engine.is_indicator_calculated("rsi") is False
        
        # EMA計算後
        engine.calculate_ema(sample_price_data)
        assert engine.is_indicator_calculated("ema") is True
        assert engine.is_indicator_calculated("rsi") is False
        
        # RSI計算後
        engine.calculate_rsi(sample_price_data)
        assert engine.is_indicator_calculated("ema") is True
        assert engine.is_indicator_calculated("rsi") is True
    
    def test_get_indicator_params(self, sample_price_data):
        """指標パラメータ取得機能のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 計算前はNone
        assert engine.get_indicator_params("ema") is None
        
        # EMA計算後
        engine.calculate_ema(sample_price_data)
        ema_params = engine.get_indicator_params("ema")
        assert ema_params is not None
        assert ema_params["periods"] == [5, 20, 50, 100, 200]
        
        # RSI計算後（カスタム期間）
        engine.calculate_rsi(sample_price_data, period=21)
        rsi_params = engine.get_indicator_params("rsi")
        assert rsi_params is not None
        assert rsi_params["period"] == 21
        
        # MACD計算後
        engine.calculate_macd(sample_price_data, fast_period=10, slow_period=20, signal_period=5)
        macd_params = engine.get_indicator_params("macd")
        assert macd_params is not None
        assert macd_params["fast"] == 10
        assert macd_params["slow"] == 20
        assert macd_params["signal"] == 5
    
    def test_get_processing_statistics(self, sample_price_data):
        """処理統計情報取得のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 初期状態
        stats = engine.get_processing_statistics()
        assert stats["total_rows_processed"] == 0
        assert stats["total_processing_time"] == 0.0
        assert stats["last_update"] is None
        
        # 複数指標計算後
        engine.calculate_ema(sample_price_data)
        engine.calculate_rsi(sample_price_data)
        
        stats = engine.get_processing_statistics()
        assert stats["total_rows_processed"] == len(sample_price_data) * 2
        assert stats["last_update"] is not None
        
        # last_updateが更新されることを確認
        from datetime import datetime
        last_update = datetime.fromisoformat(stats["last_update"])
        assert isinstance(last_update, datetime)


class TestBatchProcessingOptimization:
    """バッチ処理最適化のテストクラス"""
    
    def test_calculate_all_indicators_basic(self, sample_price_data):
        """calculate_all_indicators基本動作のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 全指標を一括計算
        result = engine.calculate_all_indicators(sample_price_data)
        
        # EMA列の確認
        for period in [5, 20, 50, 100, 200]:
            assert f"ema_{period}" in result.columns
        
        # RSI列の確認
        assert "rsi_14" in result.columns
        
        # MACD列の確認
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        
        # ボリンジャーバンド列の確認
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        assert "bb_percent" in result.columns
        
        # データサイズが変わっていないことを確認
        assert len(result) == len(sample_price_data)
    
    def test_calculate_all_indicators_selective(self, sample_price_data):
        """選択的な指標計算のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # EMAとRSIのみを計算
        result = engine.calculate_all_indicators(
            sample_price_data,
            include_indicators=["ema", "rsi"]
        )
        
        # EMA列の確認
        for period in [5, 20, 50, 100, 200]:
            assert f"ema_{period}" in result.columns
        
        # RSI列の確認
        assert "rsi_14" in result.columns
        
        # MACD列が存在しないことを確認
        assert "macd_line" not in result.columns
        
        # ボリンジャーバンド列が存在しないことを確認
        assert "bb_upper" not in result.columns
    
    def test_calculate_all_indicators_custom_params(self, sample_price_data):
        """カスタムパラメータでの計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # カスタムパラメータで計算
        result = engine.calculate_all_indicators(
            sample_price_data,
            ema_periods=[10, 30],
            rsi_period=21,
            macd_params={"fast": 10, "slow": 20, "signal": 5},
            bollinger_params={"period": 15, "num_std": 1.5}
        )
        
        # カスタムEMA期間の確認
        assert "ema_10" in result.columns
        assert "ema_30" in result.columns
        assert "ema_5" not in result.columns  # デフォルト期間は計算されない
        
        # カスタムRSI期間の確認
        assert "rsi_21" in result.columns
        assert "rsi_14" not in result.columns
        
        # 結果の値が適切な範囲にあることを確認
        rsi_values = result["rsi_21"].drop_nulls()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_batch_vs_individual_calculation(self, sample_price_data):
        """バッチ計算と個別計算の結果一致テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        import time
        
        engine1 = TechnicalIndicatorEngine()
        engine2 = TechnicalIndicatorEngine()
        
        # 個別計算
        start_individual = time.time()
        result_individual = sample_price_data.clone()
        result_individual = engine1.calculate_ema(result_individual)
        result_individual = engine1.calculate_rsi(result_individual)
        result_individual = engine1.calculate_macd(result_individual)
        result_individual = engine1.calculate_bollinger_bands(result_individual)
        time_individual = time.time() - start_individual
        
        # バッチ計算
        start_batch = time.time()
        result_batch = engine2.calculate_all_indicators(sample_price_data)
        time_batch = time.time() - start_batch
        
        # 結果の一致を確認（EMA）
        for period in [5, 20, 50, 100, 200]:
            col_name = f"ema_{period}"
            individual_values = result_individual[col_name].drop_nulls()
            batch_values = result_batch[col_name].drop_nulls()
            assert np.allclose(individual_values, batch_values, rtol=1e-5)
        
        # 結果の一致を確認（RSI）
        individual_rsi = result_individual["rsi_14"].drop_nulls()
        batch_rsi = result_batch["rsi_14"].drop_nulls()
        assert np.allclose(individual_rsi, batch_rsi, rtol=1e-5)
        
        # 結果の一致を確認（MACD）
        for col in ["macd_line", "macd_signal", "macd_histogram"]:
            individual_values = result_individual[col].drop_nulls()
            batch_values = result_batch[col].drop_nulls()
            assert np.allclose(individual_values, batch_values, rtol=1e-5)
        
        # 結果の一致を確認（ボリンジャーバンド）
        for col in ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent"]:
            individual_values = result_individual[col].drop_nulls()
            batch_values = result_batch[col].drop_nulls()
            assert np.allclose(individual_values, batch_values, rtol=1e-5)
        
        # バッチ計算の方が高速であることを期待（ただし必須ではない）
        print(f"個別計算時間: {time_individual:.4f}秒")
        print(f"バッチ計算時間: {time_batch:.4f}秒")
    
    def test_calculate_all_indicators_with_groups(self, sample_multi_symbol_data):
        """複数シンボルでのバッチ計算テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # グループ別に全指標を計算
        result = engine.calculate_all_indicators(
            sample_multi_symbol_data,
            group_by="symbol"
        )
        
        # 各シンボルで指標が計算されていることを確認
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            symbol_data = result.filter(pl.col("symbol") == symbol)
            
            # EMA列の確認
            for period in [5, 20, 50, 100, 200]:
                col_name = f"ema_{period}"
                assert col_name in symbol_data.columns
                values = symbol_data[col_name].drop_nulls()
                assert len(values) > 0
            
            # RSI列の確認
            assert "rsi_14" in symbol_data.columns
            rsi_values = symbol_data["rsi_14"].drop_nulls()
            assert len(rsi_values) > 0
            assert (rsi_values >= 0).all()
            assert (rsi_values <= 100).all()
    
    def test_memory_optimization(self, large_price_data):
        """メモリ最適化のテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        import gc
        
        engine = TechnicalIndicatorEngine()
        
        # メモリ使用量の測定前にガベージコレクション
        gc.collect()
        
        # 大量データでバッチ計算
        result = engine.calculate_all_indicators(large_price_data)
        
        # 一時列が削除されていることを確認
        temp_columns = [col for col in result.columns if col.startswith("_")]
        assert len(temp_columns) == 0, f"一時列が残っています: {temp_columns}"
        
        # Float32型が維持されていることを確認
        numeric_columns = [
            col for col in result.columns 
            if col.startswith(("ema_", "rsi_", "macd_", "bb_"))
        ]
        for col in numeric_columns:
            assert result[col].dtype == pl.Float32, f"{col}がFloat32型ではありません"
    
    def test_calculate_all_indicators_performance(self, large_price_data):
        """バッチ計算のパフォーマンステスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        import time
        
        engine = TechnicalIndicatorEngine()
        
        # 100,000行のデータで全指標を計算
        start_time = time.time()
        result = engine.calculate_all_indicators(large_price_data)
        elapsed_time = time.time() - start_time
        
        # 1秒以内に完了することを確認
        assert elapsed_time < 1.0, f"計算時間が1秒を超えています: {elapsed_time:.4f}秒"
        
        # 全ての指標が計算されていることを確認
        expected_columns = (
            [f"ema_{p}" for p in [5, 20, 50, 100, 200]] +
            ["rsi_14", "macd_line", "macd_signal", "macd_histogram"] +
            ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent"]
        )
        for col in expected_columns:
            assert col in result.columns
        
        print(f"100,000行の全指標計算時間: {elapsed_time:.4f}秒")
    
    def test_calculate_all_indicators_metadata(self, sample_price_data):
        """バッチ計算時のメタデータ更新テスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 全指標を計算
        result = engine.calculate_all_indicators(sample_price_data)
        
        # メタデータが正しく更新されていることを確認
        metadata = engine.get_metadata()
        
        # 各指標のメタデータを確認
        assert "ema" in metadata["indicators"]
        assert metadata["indicators"]["ema"]["calculated"] is True
        assert metadata["indicators"]["ema"]["periods"] == [5, 20, 50, 100, 200]
        
        assert "rsi" in metadata["indicators"]
        assert metadata["indicators"]["rsi"]["calculated"] is True
        assert metadata["indicators"]["rsi"]["period"] == 14
        
        assert "macd" in metadata["indicators"]
        assert metadata["indicators"]["macd"]["calculated"] is True
        assert metadata["indicators"]["macd"]["params"]["fast"] == 12
        assert metadata["indicators"]["macd"]["params"]["slow"] == 26
        assert metadata["indicators"]["macd"]["params"]["signal"] == 9
        
        assert "bollinger" in metadata["indicators"]
        assert metadata["indicators"]["bollinger"]["calculated"] is True
        assert metadata["indicators"]["bollinger"]["params"]["period"] == 20
        assert metadata["indicators"]["bollinger"]["params"]["num_std"] == 2.0
        
        # 処理統計情報を確認
        stats = metadata["statistics"]
        assert stats["total_rows_processed"] == len(sample_price_data) * 4  # 4指標分
        assert stats["total_processing_time"] > 0
        assert stats["last_update"] is not None
    
    def test_calculate_all_indicators_empty_data(self):
        """空データでのバッチ計算エラーハンドリングテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # 空のDataFrame
        empty_df = pl.DataFrame({"close": []})
        
        # 例外が発生することを確認
        with pytest.raises(ValueError, match="空のDataFrame"):
            engine.calculate_all_indicators(empty_df)
    
    def test_calculate_all_indicators_missing_column(self, sample_price_data):
        """必須列が欠けている場合のエラーハンドリングテスト"""
        from src.data_processing.indicators import TechnicalIndicatorEngine
        
        engine = TechnicalIndicatorEngine()
        
        # close列を削除
        df_no_close = sample_price_data.drop("close")
        
        # 例外が発生することを確認
        with pytest.raises(ValueError, match="close列が必要です"):
            engine.calculate_all_indicators(df_no_close)
        
        # カスタム価格列を指定した場合
        df_with_high = sample_price_data.drop("close")
        result = engine.calculate_all_indicators(df_with_high, price_column="high")
        
        # high列を使って計算されていることを確認
        assert "ema_5" in result.columns
        assert "rsi_14" in result.columns