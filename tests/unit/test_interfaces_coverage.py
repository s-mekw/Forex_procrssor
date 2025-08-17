"""Additional tests to improve interfaces.py coverage.

このモジュールは、interfaces.pyのカバレッジを向上させるための
追加テストを提供します。主に、抽象メソッドとProtocolクラスの
動作確認に焦点を当てています。
"""

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
    DataProcessorType,
    DataFetcherType,
    StorageHandlerType,
    PredictorType,
    Predictor,
    PredictorProtocol,
    StorageHandler,
    StorageHandlerProtocol,
)
from src.common.models import Alert, OHLC, Prediction, Tick, TimeFrame


class CompleteDataFetcher(DataFetcher):
    """完全な実装のDataFetcher"""
    
    async def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        # 抽象メソッドを実装（pass文をカバー）
        return pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": [symbol],
            "bid": [150.0],
            "ask": [150.01],
            "volume": [1000.0]
        })
    
    async def fetch_ohlc_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        # 抽象メソッドを実装（pass文をカバー）
        return pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": [symbol],
            "timeframe": [timeframe.value],
            "open": [150.0],
            "high": [150.1],
            "low": [149.9],
            "close": [150.05],
            "volume": [10000.0]
        })
    
    async def get_available_symbols(self) -> List[str]:
        # 抽象メソッドを実装（pass文をカバー）
        return ["USDJPY", "EURUSD"]


class CompleteDataProcessor(DataProcessor):
    """完全な実装のDataProcessor"""
    
    def process_tick_to_ohlc(
        self,
        tick_data: pl.DataFrame,
        timeframe: TimeFrame,
        symbol: Optional[str] = None
    ) -> pl.DataFrame:
        # 抽象メソッドを実装
        return pl.DataFrame()
    
    def calculate_indicators(
        self,
        ohlc_data: pl.DataFrame,
        indicators: List[str],
        **params
    ) -> pl.DataFrame:
        # 抽象メソッドを実装
        return ohlc_data
    
    def validate_data(
        self,
        data: pl.DataFrame,
        data_type: str = "tick"
    ) -> tuple[bool, List[str]]:
        # 抽象メソッドを実装
        return True, []


class CompleteStorageHandler(StorageHandler):
    """完全な実装のStorageHandler"""
    
    async def save_data(
        self,
        data: Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]],
        key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        # 抽象メソッドを実装
        return True
    
    async def load_data(
        self,
        key: str,
        data_type: Optional[str] = None
    ) -> Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]]:
        # 抽象メソッドを実装
        return pl.DataFrame()
    
    async def delete_data(self, key: str) -> bool:
        # 抽象メソッドを実装
        return True
    
    async def query_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        # 抽象メソッドを実装
        return pl.DataFrame()


class CompletePredictor(Predictor):
    """完全な実装のPredictor"""
    
    async def train(
        self,
        training_data: pl.DataFrame,
        validation_data: Optional[pl.DataFrame] = None,
        **hyperparameters
    ) -> Dict[str, Any]:
        # 抽象メソッドを実装
        return {"loss": 0.1}
    
    async def predict(
        self,
        input_data: pl.DataFrame,
        **params
    ) -> List[Prediction]:
        # 抽象メソッドを実装
        return []
    
    async def evaluate(
        self,
        test_data: pl.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        # 抽象メソッドを実装
        return {"mae": 0.05}
    
    async def save_model(self, path: Path) -> bool:
        # 抽象メソッドを実装
        return True
    
    async def load_model(self, path: Path) -> bool:
        # 抽象メソッドを実装
        return True


class TestInterfaceCoverage:
    """インターフェースカバレッジ向上のためのテスト"""
    
    @pytest.mark.asyncio
    async def test_complete_datafetcher(self):
        """完全なDataFetcher実装のテスト"""
        fetcher = CompleteDataFetcher()
        now = datetime.now(timezone.utc)
        
        # fetch_tick_dataのテスト
        tick_data = await fetcher.fetch_tick_data(
            "USDJPY", now, now + timedelta(hours=1)
        )
        assert isinstance(tick_data, pl.DataFrame)
        assert not tick_data.is_empty()
        
        # fetch_ohlc_dataのテスト
        ohlc_data = await fetcher.fetch_ohlc_data(
            "EURUSD", TimeFrame.H1, now, now + timedelta(days=1)
        )
        assert isinstance(ohlc_data, pl.DataFrame)
        assert not ohlc_data.is_empty()
        
        # get_available_symbolsのテスト
        symbols = await fetcher.get_available_symbols()
        assert len(symbols) > 0
        
        # is_connectedのテスト（デフォルト実装）
        assert await fetcher.is_connected() is True
    
    def test_complete_dataprocessor(self):
        """完全なDataProcessor実装のテスト"""
        processor = CompleteDataProcessor()
        
        # テストデータ
        test_data = pl.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "value": [100.0]
        })
        
        # process_tick_to_ohlcのテスト
        ohlc = processor.process_tick_to_ohlc(test_data, TimeFrame.M5)
        assert isinstance(ohlc, pl.DataFrame)
        
        # calculate_indicatorsのテスト
        with_indicators = processor.calculate_indicators(test_data, ["SMA"])
        assert isinstance(with_indicators, pl.DataFrame)
        
        # validate_dataのテスト
        is_valid, errors = processor.validate_data(test_data)
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # resample_ohlcのテスト（デフォルト実装）
        with pytest.raises(NotImplementedError):
            processor.resample_ohlc(test_data, TimeFrame.M1, TimeFrame.H1)
    
    @pytest.mark.asyncio
    async def test_complete_storagehandler(self):
        """完全なStorageHandler実装のテスト"""
        storage = CompleteStorageHandler()
        
        # save_dataのテスト
        success = await storage.save_data(pl.DataFrame(), "test")
        assert success is True
        
        # load_dataのテスト
        data = await storage.load_data("test")
        assert isinstance(data, pl.DataFrame)
        
        # delete_dataのテスト
        deleted = await storage.delete_data("test")
        assert deleted is True
        
        # query_dataのテスト
        result = await storage.query_data("SELECT * FROM test")
        assert isinstance(result, pl.DataFrame)
        
        # list_keysのテスト（デフォルト実装）
        with pytest.raises(NotImplementedError):
            await storage.list_keys()
    
    @pytest.mark.asyncio
    async def test_complete_predictor(self):
        """完全なPredictor実装のテスト"""
        predictor = CompletePredictor()
        
        # trainのテスト
        train_result = await predictor.train(pl.DataFrame())
        assert "loss" in train_result
        
        # predictのテスト
        predictions = await predictor.predict(pl.DataFrame())
        assert isinstance(predictions, list)
        
        # evaluateのテスト
        eval_result = await predictor.evaluate(pl.DataFrame())
        assert "mae" in eval_result
        
        # save_modelのテスト
        saved = await predictor.save_model(Path("model.pkl"))
        assert saved is True
        
        # load_modelのテスト
        loaded = await predictor.load_model(Path("model.pkl"))
        assert loaded is True
        
        # get_feature_importanceのテスト（デフォルト実装）
        importance = await predictor.get_feature_importance()
        assert importance is None
    
    def test_type_aliases_usage(self):
        """型エイリアスの使用テスト"""
        # 型エイリアスが正しく動作することを確認
        fetcher: DataFetcherType = CompleteDataFetcher()
        processor: DataProcessorType = CompleteDataProcessor()
        storage: StorageHandlerType = CompleteStorageHandler()
        predictor: PredictorType = CompletePredictor()
        
        # 各インスタンスが正しく作成されていることを確認
        assert fetcher is not None
        assert processor is not None
        assert storage is not None
        assert predictor is not None
    
    def test_protocol_classes_structure(self):
        """Protocolクラスの構造テスト"""
        # Protocolクラスが定義されていることを確認
        assert DataFetcherProtocol is not None
        assert DataProcessorProtocol is not None
        assert StorageHandlerProtocol is not None
        assert PredictorProtocol is not None
        
        # Protocolクラスのメソッドが定義されていることを確認
        assert hasattr(DataFetcherProtocol, "fetch_tick_data")
        assert hasattr(DataProcessorProtocol, "process_tick_to_ohlc")
        assert hasattr(StorageHandlerProtocol, "save_data")
        assert hasattr(PredictorProtocol, "train")


class ProtocolImplementation:
    """Protocolを満たす独立した実装（継承なし）"""
    
    async def fetch_tick_data(
        self, symbol: str, start_time: datetime, 
        end_time: datetime, limit: Optional[int] = None
    ) -> pl.DataFrame:
        return pl.DataFrame()
    
    async def fetch_ohlc_data(
        self, symbol: str, timeframe: TimeFrame,
        start_time: datetime, end_time: datetime, 
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        return pl.DataFrame()
    
    async def get_available_symbols(self) -> List[str]:
        return []


class TestProtocolImplementation:
    """Protocol実装のテスト"""
    
    @pytest.mark.asyncio
    async def test_protocol_implementation_works(self):
        """Protocolを満たす実装が動作することを確認"""
        impl = ProtocolImplementation()
        
        # DataFetcherProtocolを満たすメソッドが動作する
        now = datetime.now(timezone.utc)
        tick_data = await impl.fetch_tick_data("TEST", now, now)
        assert isinstance(tick_data, pl.DataFrame)
        
        ohlc_data = await impl.fetch_ohlc_data("TEST", TimeFrame.M1, now, now)
        assert isinstance(ohlc_data, pl.DataFrame)
        
        symbols = await impl.get_available_symbols()
        assert isinstance(symbols, list)