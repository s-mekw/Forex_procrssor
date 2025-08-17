"""Unit tests for common interfaces.

このモジュールは、共通インターフェースの実装とプロトコルの
動作を検証するテストを提供します。
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import pytest

from src.common.interfaces import (
    DataFetcher,
    DataFetcherProtocol,
    DataProcessor,
    DataProcessorProtocol,
    Predictor,
    PredictorProtocol,
    StorageHandler,
    StorageHandlerProtocol,
)
from src.common.models import Alert, OHLC, Prediction, Tick, TimeFrame


# === Mock implementations for testing ===

class MockDataFetcher(DataFetcher):
    """DataFetcherの具体的な実装例（テスト用）"""
    
    def __init__(self):
        self.connected = True
        self.symbols = ["USDJPY", "EURUSD", "GBPUSD"]
    
    async def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """モックのTickデータを生成"""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not available")
        
        # ダミーデータを生成
        timestamps = [start_time + timedelta(seconds=i) for i in range(10)]
        data = {
            "timestamp": timestamps,
            "symbol": [symbol] * 10,
            "bid": [150.0 + i * 0.001 for i in range(10)],
            "ask": [150.002 + i * 0.001 for i in range(10)],
            "volume": [1000.0] * 10
        }
        
        df = pl.DataFrame(data)
        if limit:
            df = df.head(limit)
        return df
    
    async def fetch_ohlc_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """モックのOHLCデータを生成"""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not available")
        
        # ダミーデータを生成
        timestamps = [start_time + timedelta(hours=i) for i in range(5)]
        data = {
            "timestamp": timestamps,
            "symbol": [symbol] * 5,
            "timeframe": [timeframe.value] * 5,
            "open": [150.0 + i * 0.1 for i in range(5)],
            "high": [150.1 + i * 0.1 for i in range(5)],
            "low": [149.9 + i * 0.1 for i in range(5)],
            "close": [150.05 + i * 0.1 for i in range(5)],
            "volume": [10000.0] * 5
        }
        
        df = pl.DataFrame(data)
        if limit:
            df = df.head(limit)
        return df
    
    async def get_available_symbols(self) -> List[str]:
        """利用可能なシンボルを返す"""
        if not self.connected:
            raise ConnectionError("Not connected to data source")
        return self.symbols


class MockDataProcessor(DataProcessor):
    """DataProcessorの具体的な実装例（テスト用）"""
    
    def process_tick_to_ohlc(
        self,
        tick_data: pl.DataFrame,
        timeframe: TimeFrame,
        symbol: Optional[str] = None
    ) -> pl.DataFrame:
        """TickデータをOHLCに変換（簡易実装）"""
        if tick_data.is_empty():
            raise ValueError("Empty tick data")
        
        # 簡易的な変換ロジック
        result = tick_data.group_by("symbol").agg([
            pl.col("bid").first().alias("open"),
            pl.col("bid").max().alias("high"),
            pl.col("bid").min().alias("low"),
            pl.col("bid").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("timestamp").first().alias("timestamp")
        ])
        
        result = result.with_columns(
            pl.lit(timeframe.value).alias("timeframe")
        )
        
        return result
    
    def calculate_indicators(
        self,
        ohlc_data: pl.DataFrame,
        indicators: List[str],
        **params
    ) -> pl.DataFrame:
        """テクニカル指標を計算（簡易実装）"""
        result = ohlc_data.clone()
        
        for indicator in indicators:
            if indicator == "SMA":
                period = params.get("sma_period", 20)
                result = result.with_columns(
                    pl.col("close").rolling_mean(period).alias(f"SMA_{period}")
                )
            elif indicator == "RSI":
                # 簡易的なRSI計算
                result = result.with_columns(
                    pl.lit(50.0).alias("RSI_14")  # ダミー値
                )
            else:
                raise ValueError(f"Unknown indicator: {indicator}")
        
        return result
    
    def validate_data(
        self,
        data: pl.DataFrame,
        data_type: str = "tick"
    ) -> tuple[bool, List[str]]:
        """データ検証（簡易実装）"""
        errors = []
        
        if data.is_empty():
            errors.append("Data is empty")
        
        if data_type == "tick":
            required_cols = {"timestamp", "symbol", "bid", "ask", "volume"}
        elif data_type == "ohlc":
            required_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        else:
            return False, ["Invalid data type"]
        
        actual_cols = set(data.columns)
        missing_cols = required_cols - actual_cols
        
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        return len(errors) == 0, errors


class MockStorageHandler(StorageHandler):
    """StorageHandlerの具体的な実装例（テスト用）"""
    
    def __init__(self):
        self.storage: Dict[str, Any] = {}
    
    async def save_data(
        self,
        data: Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]],
        key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """メモリにデータを保存"""
        self.storage[key] = {
            "data": data,
            "metadata": metadata,
            "saved_at": datetime.now(timezone.utc)
        }
        return True
    
    async def load_data(
        self,
        key: str,
        data_type: Optional[str] = None
    ) -> Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]]:
        """メモリからデータを読み込み"""
        if key not in self.storage:
            raise FileNotFoundError(f"Key {key} not found")
        return self.storage[key]["data"]
    
    async def delete_data(self, key: str) -> bool:
        """メモリからデータを削除"""
        if key not in self.storage:
            raise FileNotFoundError(f"Key {key} not found")
        del self.storage[key]
        return True
    
    async def query_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        """データをクエリ（簡易実装）"""
        # パターンマッチングの簡易実装
        matching_keys = [k for k in self.storage.keys() if query in k]
        
        if not matching_keys:
            return pl.DataFrame()
        
        # 最初のマッチを返す
        data = self.storage[matching_keys[0]]["data"]
        if isinstance(data, pl.DataFrame):
            return data
        else:
            return pl.DataFrame()


class MockPredictor(Predictor):
    """Predictorの具体的な実装例（テスト用）"""
    
    def __init__(self):
        self.model_trained = False
        self.model_params: Dict[str, Any] = {}
    
    async def train(
        self,
        training_data: pl.DataFrame,
        validation_data: Optional[pl.DataFrame] = None,
        **hyperparameters
    ) -> Dict[str, Any]:
        """モデルを訓練（モック）"""
        if training_data.is_empty():
            raise ValueError("Training data is empty")
        
        self.model_trained = True
        self.model_params = hyperparameters
        
        # ダミーのメトリクスを返す
        return {
            "loss": 0.1,
            "accuracy": 0.9,
            "val_loss": 0.15 if validation_data is not None else None
        }
    
    async def predict(
        self,
        input_data: pl.DataFrame,
        **params
    ) -> List[Prediction]:
        """予測を実行（モック）"""
        if not self.model_trained:
            raise RuntimeError("Model not trained")
        
        predictions = []
        now = datetime.now(timezone.utc)
        
        for i in range(min(len(input_data), 5)):
            pred = Prediction(
                symbol="USDJPY",
                predicted_at=now,
                target_timestamp=now + timedelta(hours=1),
                prediction_type="PRICE",
                predicted_value=150.0 + i * 0.1,
                confidence_score=0.8
            )
            predictions.append(pred)
        
        return predictions
    
    async def evaluate(
        self,
        test_data: pl.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """モデルを評価（モック）"""
        if not self.model_trained:
            raise RuntimeError("Model not trained")
        
        default_metrics = {"mae": 0.05, "rmse": 0.07, "mape": 0.03}
        
        if metrics:
            return {m: default_metrics.get(m, 0.0) for m in metrics}
        return default_metrics
    
    async def save_model(self, path: Path) -> bool:
        """モデルを保存（モック）"""
        if not self.model_trained:
            raise RuntimeError("Model not trained")
        
        # 実際のファイル保存は行わない（テスト用）
        return True
    
    async def load_model(self, path: Path) -> bool:
        """モデルを読み込み（モック）"""
        # 実際のファイル読み込みは行わない（テスト用）
        self.model_trained = True
        return True


# === Protocol implementations for testing ===

class ProtocolDataFetcher:
    """Protocolを満たす実装（継承なし）"""
    
    async def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        return pl.DataFrame()
    
    async def fetch_ohlc_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        return pl.DataFrame()
    
    async def get_available_symbols(self) -> List[str]:
        return []


# === Test cases ===

@pytest.mark.asyncio
class TestDataFetcher:
    """DataFetcherインターフェースのテスト"""
    
    async def test_fetch_tick_data(self):
        """Tickデータ取得のテスト"""
        fetcher = MockDataFetcher()
        now = datetime.now(timezone.utc)
        
        df = await fetcher.fetch_tick_data(
            "USDJPY",
            now - timedelta(hours=1),
            now,
            limit=5
        )
        
        assert not df.is_empty()
        assert len(df) == 5
        assert "timestamp" in df.columns
        assert "bid" in df.columns
        assert "ask" in df.columns
    
    async def test_fetch_ohlc_data(self):
        """OHLCデータ取得のテスト"""
        fetcher = MockDataFetcher()
        now = datetime.now(timezone.utc)
        
        df = await fetcher.fetch_ohlc_data(
            "EURUSD",
            TimeFrame.H1,
            now - timedelta(days=1),
            now,
            limit=3
        )
        
        assert not df.is_empty()
        assert len(df) == 3
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
    
    async def test_get_available_symbols(self):
        """利用可能シンボル取得のテスト"""
        fetcher = MockDataFetcher()
        symbols = await fetcher.get_available_symbols()
        
        assert len(symbols) > 0
        assert "USDJPY" in symbols
        assert "EURUSD" in symbols
    
    async def test_is_connected(self):
        """接続状態確認のテスト"""
        fetcher = MockDataFetcher()
        
        assert await fetcher.is_connected() is True
        
        fetcher.connected = False
        assert await fetcher.is_connected() is False
    
    async def test_invalid_symbol(self):
        """無効なシンボルのエラーハンドリング"""
        fetcher = MockDataFetcher()
        now = datetime.now(timezone.utc)
        
        with pytest.raises(ValueError):
            await fetcher.fetch_tick_data(
                "INVALID",
                now - timedelta(hours=1),
                now
            )


class TestDataProcessor:
    """DataProcessorインターフェースのテスト"""
    
    def test_process_tick_to_ohlc(self):
        """Tick→OHLC変換のテスト"""
        processor = MockDataProcessor()
        
        # テスト用Tickデータ作成
        tick_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 5,
            "symbol": ["USDJPY"] * 5,
            "bid": [150.0, 150.1, 149.9, 150.2, 150.05],
            "ask": [150.01, 150.11, 149.91, 150.21, 150.06],
            "volume": [1000.0] * 5
        })
        
        ohlc = processor.process_tick_to_ohlc(tick_data, TimeFrame.M1)
        
        assert not ohlc.is_empty()
        assert "open" in ohlc.columns
        assert "high" in ohlc.columns
        assert "low" in ohlc.columns
        assert "close" in ohlc.columns
    
    def test_calculate_indicators(self):
        """テクニカル指標計算のテスト"""
        processor = MockDataProcessor()
        
        # テスト用OHLCデータ作成
        ohlc_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 30,
            "symbol": ["USDJPY"] * 30,
            "open": [150.0 + i * 0.01 for i in range(30)],
            "high": [150.1 + i * 0.01 for i in range(30)],
            "low": [149.9 + i * 0.01 for i in range(30)],
            "close": [150.05 + i * 0.01 for i in range(30)],
            "volume": [10000.0] * 30
        })
        
        result = processor.calculate_indicators(
            ohlc_data,
            ["SMA", "RSI"],
            sma_period=10
        )
        
        assert "SMA_10" in result.columns
        assert "RSI_14" in result.columns
    
    def test_validate_data(self):
        """データ検証のテスト"""
        processor = MockDataProcessor()
        
        # 正常なTickデータ
        tick_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["USDJPY"],
            "bid": [150.0],
            "ask": [150.01],
            "volume": [1000.0]
        })
        
        is_valid, errors = processor.validate_data(tick_data, "tick")
        assert is_valid
        assert len(errors) == 0
        
        # 不正なデータ（カラム不足）
        invalid_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["USDJPY"]
        })
        
        is_valid, errors = processor.validate_data(invalid_data, "tick")
        assert not is_valid
        assert len(errors) > 0
    
    def test_empty_data_error(self):
        """空データのエラーハンドリング"""
        processor = MockDataProcessor()
        empty_data = pl.DataFrame()
        
        with pytest.raises(ValueError):
            processor.process_tick_to_ohlc(empty_data, TimeFrame.M1)


@pytest.mark.asyncio
class TestStorageHandler:
    """StorageHandlerインターフェースのテスト"""
    
    async def test_save_and_load_dataframe(self):
        """DataFrameの保存と読み込みテスト"""
        storage = MockStorageHandler()
        
        # テストデータ作成
        df = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "value": [100.0]
        })
        
        # 保存
        success = await storage.save_data(df, "test/data", {"type": "test"})
        assert success
        
        # 読み込み
        loaded = await storage.load_data("test/data")
        assert isinstance(loaded, pl.DataFrame)
        assert loaded.equals(df)
    
    async def test_save_and_load_models(self):
        """モデルオブジェクトの保存と読み込みテスト"""
        storage = MockStorageHandler()
        
        # テストデータ作成
        tick = Tick(
            timestamp=datetime.now(timezone.utc),
            symbol="USDJPY",
            bid=150.0,
            ask=150.01,
            volume=1000.0
        )
        
        # 保存
        success = await storage.save_data([tick], "test/tick")
        assert success
        
        # 読み込み
        loaded = await storage.load_data("test/tick")
        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0] == tick
    
    async def test_delete_data(self):
        """データ削除のテスト"""
        storage = MockStorageHandler()
        
        # データを保存
        await storage.save_data(pl.DataFrame(), "test/delete")
        
        # 削除
        success = await storage.delete_data("test/delete")
        assert success
        
        # 削除後の読み込みはエラー
        with pytest.raises(FileNotFoundError):
            await storage.load_data("test/delete")
    
    async def test_query_data(self):
        """データクエリのテスト"""
        storage = MockStorageHandler()
        
        # テストデータを複数保存
        df1 = pl.DataFrame({"id": [1, 2, 3]})
        df2 = pl.DataFrame({"id": [4, 5, 6]})
        
        await storage.save_data(df1, "data/test1")
        await storage.save_data(df2, "data/test2")
        
        # クエリ実行
        result = await storage.query_data("data/test")
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()


@pytest.mark.asyncio
class TestPredictor:
    """Predictorインターフェースのテスト"""
    
    async def test_train_model(self):
        """モデル訓練のテスト"""
        predictor = MockPredictor()
        
        # 訓練データ作成
        train_data = pl.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [7.0, 8.0, 9.0]
        })
        
        metrics = await predictor.train(
            train_data,
            learning_rate=0.01,
            epochs=100
        )
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert predictor.model_trained
    
    async def test_predict(self):
        """予測実行のテスト"""
        predictor = MockPredictor()
        
        # まず訓練
        train_data = pl.DataFrame({"feature": [1.0]})
        await predictor.train(train_data)
        
        # 予測実行
        input_data = pl.DataFrame({
            "feature1": [1.0, 2.0],
            "feature2": [3.0, 4.0]
        })
        
        predictions = await predictor.predict(input_data)
        
        assert len(predictions) > 0
        assert all(isinstance(p, Prediction) for p in predictions)
    
    async def test_evaluate(self):
        """モデル評価のテスト"""
        predictor = MockPredictor()
        
        # 訓練
        train_data = pl.DataFrame({"feature": [1.0]})
        await predictor.train(train_data)
        
        # 評価
        test_data = pl.DataFrame({
            "feature": [1.0, 2.0],
            "target": [3.0, 4.0]
        })
        
        metrics = await predictor.evaluate(test_data, ["mae", "rmse"])
        
        assert "mae" in metrics
        assert "rmse" in metrics
    
    async def test_save_and_load_model(self):
        """モデルの保存と読み込みテスト"""
        predictor = MockPredictor()
        
        # 訓練
        train_data = pl.DataFrame({"feature": [1.0]})
        await predictor.train(train_data)
        
        # 保存
        path = Path("test_model.pkl")
        success = await predictor.save_model(path)
        assert success
        
        # 新しいインスタンスで読み込み
        new_predictor = MockPredictor()
        success = await new_predictor.load_model(path)
        assert success
        assert new_predictor.model_trained
    
    async def test_predict_without_training(self):
        """訓練前の予測はエラー"""
        predictor = MockPredictor()
        input_data = pl.DataFrame({"feature": [1.0]})
        
        with pytest.raises(RuntimeError):
            await predictor.predict(input_data)


class TestProtocolCompatibility:
    """Protocolの互換性テスト"""
    
    def test_protocol_implementation(self):
        """Protocolを満たす実装の確認"""
        # 継承なしでProtocolを満たす
        fetcher = ProtocolDataFetcher()
        
        # 型チェック（実行時は常にTrue、型チェッカーが検証）
        assert isinstance(fetcher, object)
        
        # メソッドの存在確認
        assert hasattr(fetcher, "fetch_tick_data")
        assert hasattr(fetcher, "fetch_ohlc_data")
        assert hasattr(fetcher, "get_available_symbols")
    
    def test_abc_implementation(self):
        """ABCの実装確認"""
        # ABC継承の実装
        fetcher = MockDataFetcher()
        
        # DataFetcherのインスタンスであることを確認
        assert isinstance(fetcher, DataFetcher)
        
        # メソッドの存在確認
        assert hasattr(fetcher, "fetch_tick_data")
        assert hasattr(fetcher, "is_connected")


@pytest.mark.asyncio
class TestInterfaceIntegration:
    """インターフェース間の統合テスト"""
    
    async def test_fetcher_processor_integration(self):
        """DataFetcherとDataProcessorの連携"""
        fetcher = MockDataFetcher()
        processor = MockDataProcessor()
        
        # データ取得
        now = datetime.now(timezone.utc)
        tick_data = await fetcher.fetch_tick_data(
            "USDJPY",
            now - timedelta(hours=1),
            now
        )
        
        # データ処理
        ohlc_data = processor.process_tick_to_ohlc(tick_data, TimeFrame.M5)
        
        # 指標計算
        with_indicators = processor.calculate_indicators(
            ohlc_data,
            ["SMA"],
            sma_period=5
        )
        
        assert not with_indicators.is_empty()
        assert "SMA_5" in with_indicators.columns
    
    async def test_processor_storage_integration(self):
        """DataProcessorとStorageHandlerの連携"""
        processor = MockDataProcessor()
        storage = MockStorageHandler()
        
        # データ作成と処理
        tick_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)] * 5,
            "symbol": ["EURUSD"] * 5,
            "bid": [1.1000 + i * 0.0001 for i in range(5)],
            "ask": [1.1001 + i * 0.0001 for i in range(5)],
            "volume": [1000.0] * 5
        })
        
        ohlc_data = processor.process_tick_to_ohlc(tick_data, TimeFrame.H1)
        
        # 保存
        success = await storage.save_data(ohlc_data, "ohlc/EURUSD/H1")
        assert success
        
        # 読み込み
        loaded = await storage.load_data("ohlc/EURUSD/H1")
        assert loaded.equals(ohlc_data)
    
    async def test_storage_predictor_integration(self):
        """StorageHandlerとPredictorの連携"""
        storage = MockStorageHandler()
        predictor = MockPredictor()
        
        # 訓練データを保存
        train_data = pl.DataFrame({
            "feature1": list(range(100)),
            "feature2": list(range(100, 200)),
            "target": list(range(200, 300))
        })
        
        await storage.save_data(train_data, "training/data")
        
        # 読み込んで訓練
        loaded_data = await storage.load_data("training/data")
        metrics = await predictor.train(loaded_data)
        
        assert metrics["loss"] < 1.0
        
        # 予測結果を保存
        predictions = await predictor.predict(train_data.head(5))
        await storage.save_data(predictions, "predictions/batch1")
        
        # 予測結果を読み込み
        loaded_predictions = await storage.load_data("predictions/batch1")
        assert len(loaded_predictions) == len(predictions)