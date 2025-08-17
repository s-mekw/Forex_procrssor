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


class IncompleteDataFetcher(DataFetcher):
    """不完全な実装（テスト用）"""
    # fetch_tick_dataのみ実装
    async def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        return pl.DataFrame()
    # 他のメソッドは未実装（インスタンス化時にエラーになるはず）


class MinimalDataProcessor(DataProcessor):
    """最小限の実装（DataProcessor継承テスト用）"""
    def process_tick_to_ohlc(
        self,
        tick_data: pl.DataFrame,
        timeframe: TimeFrame,
        symbol: Optional[str] = None
    ) -> pl.DataFrame:
        return pl.DataFrame()
    
    def calculate_indicators(
        self,
        ohlc_data: pl.DataFrame,
        indicators: List[str],
        **params
    ) -> pl.DataFrame:
        return ohlc_data
    
    def validate_data(
        self,
        data: pl.DataFrame,
        data_type: str = "tick"
    ) -> tuple[bool, List[str]]:
        return True, []


class MinimalStorageHandler(StorageHandler):
    """最小限の実装（StorageHandler継承テスト用）"""
    async def save_data(
        self,
        data: Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]],
        key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        return True
    
    async def load_data(
        self,
        key: str,
        data_type: Optional[str] = None
    ) -> Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]]:
        return pl.DataFrame()
    
    async def delete_data(self, key: str) -> bool:
        return True
    
    async def query_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        return pl.DataFrame()


class MinimalPredictor(Predictor):
    """最小限の実装（Predictor継承テスト用）"""
    async def train(
        self,
        training_data: pl.DataFrame,
        validation_data: Optional[pl.DataFrame] = None,
        **hyperparameters
    ) -> Dict[str, Any]:
        return {"loss": 0.0}
    
    async def predict(
        self,
        input_data: pl.DataFrame,
        **params
    ) -> List[Prediction]:
        return []
    
    async def evaluate(
        self,
        test_data: pl.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        return {"mae": 0.0}
    
    async def save_model(self, path: Path) -> bool:
        return True
    
    async def load_model(self, path: Path) -> bool:
        return True


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
    
    @pytest.mark.asyncio
    async def test_protocol_implementation(self):
        """Protocolを満たす実装の確認"""
        # 継承なしでProtocolを満たす
        fetcher = ProtocolDataFetcher()
        
        # 型チェック（実行時は常にTrue、型チェッカーが検証）
        assert isinstance(fetcher, object)
        
        # メソッドの存在確認
        assert hasattr(fetcher, "fetch_tick_data")
        assert hasattr(fetcher, "fetch_ohlc_data")
        assert hasattr(fetcher, "get_available_symbols")
        
        # 実際にメソッドを呼び出してみる
        now = datetime.now(timezone.utc)
        tick_result = await fetcher.fetch_tick_data("TEST", now, now)
        assert isinstance(tick_result, pl.DataFrame)
        
        ohlc_result = await fetcher.fetch_ohlc_data("TEST", TimeFrame.M1, now, now)
        assert isinstance(ohlc_result, pl.DataFrame)
        
        symbols = await fetcher.get_available_symbols()
        assert isinstance(symbols, list)
    
    def test_abc_implementation(self):
        """ABCの実装確認"""
        # ABC継承の実装
        fetcher = MockDataFetcher()
        
        # DataFetcherのインスタンスであることを確認
        assert isinstance(fetcher, DataFetcher)
        
        # メソッドの存在確認
        assert hasattr(fetcher, "fetch_tick_data")
        assert hasattr(fetcher, "is_connected")
    
    def test_all_protocol_classes(self):
        """すべてのProtocolクラスの実装確認"""
        # DataProcessorProtocol
        class TestProcessor:
            def process_tick_to_ohlc(self, tick_data, timeframe, symbol=None):
                return pl.DataFrame()
            def calculate_indicators(self, ohlc_data, indicators, **params):
                return pl.DataFrame()
            def validate_data(self, data, data_type="tick"):
                return True, []
        
        processor = TestProcessor()
        assert hasattr(processor, "process_tick_to_ohlc")
        assert hasattr(processor, "calculate_indicators")
        assert hasattr(processor, "validate_data")
        
        # StorageHandlerProtocol
        class TestStorage:
            async def save_data(self, data, key, metadata=None):
                return True
            async def load_data(self, key, data_type=None):
                return pl.DataFrame()
            async def delete_data(self, key):
                return True
            async def query_data(self, query, params=None):
                return pl.DataFrame()
        
        storage = TestStorage()
        assert hasattr(storage, "save_data")
        assert hasattr(storage, "load_data")
        assert hasattr(storage, "delete_data")
        assert hasattr(storage, "query_data")
        
        # PredictorProtocol
        class TestPredictor:
            async def train(self, training_data, validation_data=None, **hyperparameters):
                return {}
            async def predict(self, input_data, **params):
                return []
            async def evaluate(self, test_data, metrics=None):
                return {}
            async def save_model(self, path):
                return True
            async def load_model(self, path):
                return True
        
        predictor = TestPredictor()
        assert hasattr(predictor, "train")
        assert hasattr(predictor, "predict")
        assert hasattr(predictor, "evaluate")
        assert hasattr(predictor, "save_model")
        assert hasattr(predictor, "load_model")


class TestAbstractMethods:
    """抽象メソッドのテスト"""
    
    def test_cannot_instantiate_abstract_classes(self):
        """抽象クラスは直接インスタンス化できないことを確認"""
        # DataFetcherの直接インスタンス化はエラー
        with pytest.raises(TypeError) as exc_info:
            DataFetcher()
        assert "Can't instantiate abstract class" in str(exc_info.value)
        
        # DataProcessorの直接インスタンス化はエラー
        with pytest.raises(TypeError) as exc_info:
            DataProcessor()
        assert "Can't instantiate abstract class" in str(exc_info.value)
        
        # StorageHandlerの直接インスタンス化はエラー
        with pytest.raises(TypeError) as exc_info:
            StorageHandler()
        assert "Can't instantiate abstract class" in str(exc_info.value)
        
        # Predictorの直接インスタンス化はエラー
        with pytest.raises(TypeError) as exc_info:
            Predictor()
        assert "Can't instantiate abstract class" in str(exc_info.value)
    
    def test_incomplete_implementation_fails(self):
        """不完全な実装はインスタンス化できないことを確認"""
        # 必要なメソッドが未実装の場合はエラー
        with pytest.raises(TypeError) as exc_info:
            IncompleteDataFetcher()
        assert "Can't instantiate abstract class" in str(exc_info.value)
    
    def test_dataprocessor_resample_ohlc_not_implemented(self):
        """DataProcessorのresample_ohlcメソッドのデフォルト実装テスト"""
        processor = MockDataProcessor()
        
        # テスト用OHLCデータ作成
        ohlc_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["USDJPY"],
            "open": [150.0],
            "high": [150.1],
            "low": [149.9],
            "close": [150.05],
            "volume": [10000.0]
        })
        
        # resample_ohlcはデフォルトでNotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            processor.resample_ohlc(ohlc_data, TimeFrame.M1, TimeFrame.H1)
        assert "Resampling is not implemented" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_storagehandler_list_keys_not_implemented(self):
        """StorageHandlerのlist_keysメソッドのデフォルト実装テスト"""
        storage = MockStorageHandler()
        
        # list_keysはデフォルトでNotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            await storage.list_keys()
        assert "List keys is not implemented" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_predictor_get_feature_importance_default(self):
        """Predictorのget_feature_importanceメソッドのデフォルト実装テスト"""
        predictor = MockPredictor()
        
        # デフォルト実装はNoneを返す
        importance = await predictor.get_feature_importance()
        assert importance is None
    
    def test_minimal_dataprocessor_implementation(self):
        """最小限のDataProcessor実装のテスト"""
        processor = MinimalDataProcessor()
        
        # インスタンス化可能であることを確認
        assert isinstance(processor, DataProcessor)
        
        # テスト用データ作成
        tick_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["USDJPY"],
            "bid": [150.0],
            "ask": [150.01],
            "volume": [1000.0]
        })
        
        # メソッドが正しく動作することを確認
        result = processor.process_tick_to_ohlc(tick_data, TimeFrame.M1)
        assert isinstance(result, pl.DataFrame)
        
        # calculate_indicatorsのテスト
        indicators_result = processor.calculate_indicators(tick_data, ["SMA"])
        assert indicators_result.equals(tick_data)
        
        # validate_dataのテスト
        is_valid, errors = processor.validate_data(tick_data)
        assert is_valid is True
        assert errors == []
        
        # resample_ohlcはデフォルト実装を使用
        with pytest.raises(NotImplementedError):
            processor.resample_ohlc(tick_data, TimeFrame.M1, TimeFrame.H1)
    
    @pytest.mark.asyncio
    async def test_minimal_storagehandler_implementation(self):
        """最小限のStorageHandler実装のテスト"""
        storage = MinimalStorageHandler()
        
        # インスタンス化可能であることを確認
        assert isinstance(storage, StorageHandler)
        
        # 各メソッドが正しく動作することを確認
        assert await storage.save_data(pl.DataFrame(), "test") is True
        result = await storage.load_data("test")
        assert isinstance(result, pl.DataFrame)
        assert await storage.delete_data("test") is True
        query_result = await storage.query_data("test")
        assert isinstance(query_result, pl.DataFrame)
        
        # list_keysはデフォルト実装
        with pytest.raises(NotImplementedError):
            await storage.list_keys()
    
    @pytest.mark.asyncio
    async def test_minimal_predictor_implementation(self):
        """最小限のPredictor実装のテスト"""
        predictor = MinimalPredictor()
        
        # インスタンス化可能であることを確認
        assert isinstance(predictor, Predictor)
        
        # 各メソッドが正しく動作することを確認
        train_result = await predictor.train(pl.DataFrame())
        assert "loss" in train_result
        
        predictions = await predictor.predict(pl.DataFrame())
        assert isinstance(predictions, list)
        
        eval_result = await predictor.evaluate(pl.DataFrame())
        assert "mae" in eval_result
        
        assert await predictor.save_model(Path("test.pkl")) is True
        assert await predictor.load_model(Path("test.pkl")) is True
        
        # get_feature_importanceはデフォルト実装
        importance = await predictor.get_feature_importance()
        assert importance is None


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


class TestTypeAliases:
    """型エイリアスのテスト"""
    
    def test_type_aliases_exist(self):
        """型エイリアスが定義されていることを確認"""
        from src.common.interfaces import (
            DataFetcherType,
            DataProcessorType,
            StorageHandlerType,
            PredictorType
        )
        
        # 型エイリアスが存在することを確認
        assert DataFetcherType is not None
        assert DataProcessorType is not None
        assert StorageHandlerType is not None
        assert PredictorType is not None


class TestMethodSignatures:
    """メソッドシグネチャのテスト"""
    
    def test_datafetcher_method_signatures(self):
        """DataFetcherのメソッドシグネチャ確認"""
        import inspect
        
        # fetch_tick_dataのシグネチャ確認
        sig = inspect.signature(DataFetcher.fetch_tick_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "symbol" in params
        assert "start_time" in params
        assert "end_time" in params
        assert "limit" in params
        
        # fetch_ohlc_dataのシグネチャ確認
        sig = inspect.signature(DataFetcher.fetch_ohlc_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "symbol" in params
        assert "timeframe" in params
        assert "start_time" in params
        assert "end_time" in params
        assert "limit" in params
        
        # get_available_symbolsのシグネチャ確認
        sig = inspect.signature(DataFetcher.get_available_symbols)
        params = list(sig.parameters.keys())
        assert "self" in params
    
    def test_dataprocessor_method_signatures(self):
        """DataProcessorのメソッドシグネチャ確認"""
        import inspect
        
        # process_tick_to_ohlcのシグネチャ確認
        sig = inspect.signature(DataProcessor.process_tick_to_ohlc)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "tick_data" in params
        assert "timeframe" in params
        assert "symbol" in params
        
        # calculate_indicatorsのシグネチャ確認
        sig = inspect.signature(DataProcessor.calculate_indicators)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "ohlc_data" in params
        assert "indicators" in params
        assert "params" in params
        
        # validate_dataのシグネチャ確認
        sig = inspect.signature(DataProcessor.validate_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "data" in params
        assert "data_type" in params
    
    def test_storagehandler_method_signatures(self):
        """StorageHandlerのメソッドシグネチャ確認"""
        import inspect
        
        # save_dataのシグネチャ確認
        sig = inspect.signature(StorageHandler.save_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "data" in params
        assert "key" in params
        assert "metadata" in params
        
        # load_dataのシグネチャ確認
        sig = inspect.signature(StorageHandler.load_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "key" in params
        assert "data_type" in params
        
        # delete_dataのシグネチャ確認
        sig = inspect.signature(StorageHandler.delete_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "key" in params
        
        # query_dataのシグネチャ確認
        sig = inspect.signature(StorageHandler.query_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "query" in params
        assert "params" in params
    
    def test_predictor_method_signatures(self):
        """Predictorのメソッドシグネチャ確認"""
        import inspect
        
        # trainのシグネチャ確認
        sig = inspect.signature(Predictor.train)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "training_data" in params
        assert "validation_data" in params
        assert "hyperparameters" in params
        
        # predictのシグネチャ確認
        sig = inspect.signature(Predictor.predict)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "input_data" in params
        assert "params" in params
        
        # evaluateのシグネチャ確認
        sig = inspect.signature(Predictor.evaluate)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "test_data" in params
        assert "metrics" in params
        
        # save_modelのシグネチャ確認
        sig = inspect.signature(Predictor.save_model)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "path" in params
        
        # load_modelのシグネチャ確認
        sig = inspect.signature(Predictor.load_model)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "path" in params