"""
pytest共通設定とフィクスチャ

テスト実行で使用する共通設定、フィクスチャ、ヘルパー関数を定義します。
"""

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import polars as pl
import numpy as np
from unittest.mock import MagicMock, AsyncMock


# =====================================
# pytest設定
# =====================================

def pytest_configure(config):
    """pytest実行時の初期設定"""
    # カスタムマーカーの登録
    config.addinivalue_line(
        "markers", "unit: ユニットテストのマーカー"
    )
    config.addinivalue_line(
        "markers", "integration: 統合テストのマーカー"
    )
    config.addinivalue_line(
        "markers", "e2e: エンドツーエンドテストのマーカー"
    )
    config.addinivalue_line(
        "markers", "slow: 実行時間の長いテスト"
    )
    config.addinivalue_line(
        "markers", "mt5: MT5接続が必要なテスト"
    )
    config.addinivalue_line(
        "markers", "influxdb: InfluxDB接続が必要なテスト"
    )
    config.addinivalue_line(
        "markers", "gpu: GPU環境が必要なテスト"
    )


def pytest_collection_modifyitems(config, items):
    """テスト実行前の設定変更"""
    # 環境変数に基づいたテストスキップ
    skip_mt5 = pytest.mark.skip(reason="MT5環境が利用できません")
    skip_influxdb = pytest.mark.skip(reason="InfluxDB環境が利用できません")
    skip_gpu = pytest.mark.skip(reason="GPU環境が利用できません")
    
    for item in items:
        # MT5マーカーのあるテストをスキップ
        if "mt5" in item.keywords and not os.getenv("MT5_AVAILABLE"):
            item.add_marker(skip_mt5)
        
        # InfluxDBマーカーのあるテストをスキップ  
        if "influxdb" in item.keywords and not os.getenv("INFLUXDB_AVAILABLE"):
            item.add_marker(skip_influxdb)
            
        # GPUマーカーのあるテストをスキップ
        if "gpu" in item.keywords and not os.getenv("GPU_AVAILABLE"):
            item.add_marker(skip_gpu)


# =====================================
# 非同期テスト用設定
# =====================================

@pytest.fixture(scope="session")
def event_loop():
    """非同期テスト用のイベントループ"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =====================================
# ファイルシステム関連フィクスチャ
# =====================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """テスト用一時ディレクトリ"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_data_dir() -> Path:
    """テストデータディレクトリ"""
    return Path(__file__).parent / "fixtures" / "data"


# =====================================
# サンプルデータフィクスチャ
# =====================================

@pytest.fixture
def sample_tick_data() -> pl.DataFrame:
    """サンプルティックデータ"""
    np.random.seed(42)
    n_ticks = 1000
    
    # 基準時刻からの連続したタイムスタンプ
    base_time = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        base_time.timestamp() + i * 0.1 for i in range(n_ticks)
    ]
    
    # 価格データ（ランダムウォーク）
    base_price = 1.1000
    price_changes = np.random.normal(0, 0.0001, n_ticks)
    prices = base_price + np.cumsum(price_changes)
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["EURUSD"] * n_ticks,
        "bid": prices - 0.0001,
        "ask": prices + 0.0001,
        "volume": np.random.randint(1, 100, n_ticks),
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("bid").cast(pl.Float32),
        pl.col("ask").cast(pl.Float32),
    ])


@pytest.fixture
def sample_ohlc_data() -> pl.DataFrame:
    """サンプルOHLCデータ"""
    np.random.seed(42)
    n_bars = 100
    
    base_time = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        base_time.timestamp() + i * 60 for i in range(n_bars)  # 1分足
    ]
    
    # OHLC価格データ
    base_price = 1.1000
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_price = base_price
    for i in range(n_bars):
        open_price = current_price
        close_price = open_price + np.random.normal(0, 0.001)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.0005))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.0005))
        volume = np.random.randint(50, 500)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        
        current_price = close_price
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["EURUSD"] * n_bars,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
    ])


@pytest.fixture
def sample_prediction_data() -> pl.DataFrame:
    """サンプル予測データ"""
    np.random.seed(42)
    n_predictions = 50
    
    base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        base_time.timestamp() + i * 3600 for i in range(n_predictions)  # 1時間先予測
    ]
    
    predictions = 1.1000 + np.random.normal(0, 0.01, n_predictions)
    confidence_95 = abs(np.random.normal(0.01, 0.002, n_predictions))
    confidence_80 = confidence_95 * 0.7
    confidence_50 = confidence_95 * 0.4
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["EURUSD"] * n_predictions,
        "prediction": predictions,
        "confidence_95": confidence_95,
        "confidence_80": confidence_80,
        "confidence_50": confidence_50,
    }).with_columns([
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("prediction").cast(pl.Float32),
        pl.col("confidence_95").cast(pl.Float32),
        pl.col("confidence_80").cast(pl.Float32),
        pl.col("confidence_50").cast(pl.Float32),
    ])


# =====================================
# モック関連フィクスチャ  
# =====================================

@pytest.fixture
def mock_mt5_client() -> MagicMock:
    """MT5クライアントのモック"""
    mock = MagicMock()
    mock.initialize.return_value = True
    mock.shutdown.return_value = None
    mock.last_error.return_value = (0, "Success")
    
    # サンプルティックデータを返すモック
    mock.copy_ticks_from.return_value = np.array([
        (1704096000.0, 1.1000, 1.1002, 0, 100, 0, 0),
        (1704096001.0, 1.1001, 1.1003, 0, 150, 0, 0),
    ], dtype=[
        ('time', 'f8'), ('bid', 'f8'), ('ask', 'f8'),
        ('last', 'f8'), ('volume', 'u8'), ('time_msc', 'u8'), ('flags', 'u4')
    ])
    
    return mock


@pytest.fixture  
def mock_influx_client() -> AsyncMock:
    """InfluxDBクライアントのモック"""
    mock = AsyncMock()
    mock.write_api.return_value = AsyncMock()
    mock.query_api.return_value = AsyncMock()
    mock.health.return_value = {"status": "pass"}
    return mock


@pytest.fixture
def mock_torch_model() -> MagicMock:
    """PyTorchモデルのモック"""
    mock = MagicMock()
    mock.eval.return_value = mock
    mock.forward.return_value = MagicMock()
    mock.parameters.return_value = iter([])
    mock.state_dict.return_value = {}
    return mock


# =====================================
# 設定関連フィクスチャ
# =====================================

@pytest.fixture
def test_config() -> dict:
    """テスト用設定"""
    return {
        "mt5": {
            "server": "test-server",
            "login": 12345,
            "password": "test-password",
            "timeout": 5000,
        },
        "influxdb": {
            "url": "http://localhost:8086",
            "token": "test-token", 
            "org": "test-org",
            "bucket": "test-bucket",
        },
        "model": {
            "patch_size": 5,
            "num_heads": 8,
            "num_layers": 3,
            "hidden_dim": 256,
            "prediction_horizons": [1, 3, 6, 12],
        },
        "data_processing": {
            "batch_size": 1000,
            "chunk_size": 10000,
            "ema_periods": [5, 20, 50, 100, 200],
            "rci_periods": [9, 13, 24, 33, 48, 66, 108],
        },
    }


# =====================================
# パフォーマンステスト用フィクスチャ
# =====================================

@pytest.fixture
def benchmark_timer():
    """パフォーマンス測定用タイマー"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                raise ValueError("タイマーが正しく実行されていません")
            return self.end_time - self.start_time
    
    return Timer


# =====================================
# ヘルパー関数
# =====================================

def assert_dataframe_equal(df1: pl.DataFrame, df2: pl.DataFrame, tolerance: float = 1e-6):
    """DataFrameの値が等しいかを確認（浮動小数点数の許容誤差付き）"""
    assert df1.shape == df2.shape, f"DataFrameの形状が異なります: {df1.shape} != {df2.shape}"
    assert df1.columns == df2.columns, f"列名が異なります: {df1.columns} != {df2.columns}"
    
    for col in df1.columns:
        if df1[col].dtype in [pl.Float32, pl.Float64]:
            # 浮動小数点数の場合は許容誤差付きで比較
            diff = (df1[col] - df2[col]).abs()
            assert diff.max() <= tolerance, f"列 '{col}' の値が許容誤差を超えています"
        else:
            # その他のデータ型は厳密に比較
            assert df1[col].equals(df2[col]), f"列 '{col}' の値が異なります"


def create_test_database_url(temp_dir: Path) -> str:
    """テスト用データベースURLを生成"""
    db_path = temp_dir / "test.db"
    return f"sqlite:///{db_path}"