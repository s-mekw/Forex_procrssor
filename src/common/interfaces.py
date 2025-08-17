"""Common interfaces for Forex Processor components.

このモジュールは、Forex Processorの各コンポーネント間で使用される
基底インターフェースを定義します。Protocol と ABC を併用することで、
柔軟性と厳密性のバランスを実現します。

すべてのインターフェースはPolarsをデータフレームライブラリとして使用し、
非同期処理を基本とした設計になっています。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import polars as pl

from .models import Alert, OHLC, Prediction, Tick, TimeFrame


class DataFetcher(ABC):
    """データ取得の抽象基底クラス
    
    外部データソースからのデータ取得を抽象化します。
    すべてのデータフェッチャーはこのインターフェースを実装する必要があります。
    """
    
    @abstractmethod
    async def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """Tickデータを取得
        
        Args:
            symbol: 通貨ペアシンボル（例: USDJPY）
            start_time: 取得開始時刻（UTC）
            end_time: 取得終了時刻（UTC）
            limit: 取得件数の上限（省略可）
        
        Returns:
            Tickデータを含むPolars DataFrame
            カラム: timestamp, symbol, bid, ask, volume
        
        Raises:
            ConnectionError: データソースへの接続に失敗した場合
            ValueError: 無効なパラメータが指定された場合
        """
        pass
    
    @abstractmethod
    async def fetch_ohlc_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """OHLCデータを取得
        
        Args:
            symbol: 通貨ペアシンボル（例: USDJPY）
            timeframe: 時間足（M1, M5, H1等）
            start_time: 取得開始時刻（UTC）
            end_time: 取得終了時刻（UTC）
            limit: 取得件数の上限（省略可）
        
        Returns:
            OHLCデータを含むPolars DataFrame
            カラム: timestamp, symbol, timeframe, open, high, low, close, volume
        
        Raises:
            ConnectionError: データソースへの接続に失敗した場合
            ValueError: 無効なパラメータが指定された場合
        """
        pass
    
    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """利用可能な通貨ペアシンボルのリストを取得
        
        Returns:
            利用可能なシンボルのリスト（例: ["USDJPY", "EURUSD"]）
        
        Raises:
            ConnectionError: データソースへの接続に失敗した場合
        """
        pass
    
    async def is_connected(self) -> bool:
        """データソースへの接続状態を確認
        
        Returns:
            接続されている場合True
        """
        try:
            # 最小限のデータ取得を試みて接続を確認
            await self.get_available_symbols()
            return True
        except Exception:
            return False


class DataProcessor(ABC):
    """データ処理の抽象基底クラス
    
    データの変換、指標計算、検証などのデータ処理を抽象化します。
    すべてのデータプロセッサーはこのインターフェースを実装する必要があります。
    """
    
    @abstractmethod
    def process_tick_to_ohlc(
        self,
        tick_data: pl.DataFrame,
        timeframe: TimeFrame,
        symbol: Optional[str] = None
    ) -> pl.DataFrame:
        """TickデータからOHLCデータへ変換
        
        Args:
            tick_data: Tickデータを含むPolars DataFrame
            timeframe: 変換先の時間足
            symbol: 対象シンボル（省略時は全シンボル）
        
        Returns:
            OHLCデータを含むPolars DataFrame
        
        Raises:
            ValueError: 無効なデータや時間足が指定された場合
        """
        pass
    
    @abstractmethod
    def calculate_indicators(
        self,
        ohlc_data: pl.DataFrame,
        indicators: List[str],
        **params
    ) -> pl.DataFrame:
        """テクニカル指標を計算
        
        Args:
            ohlc_data: OHLCデータを含むPolars DataFrame
            indicators: 計算する指標名のリスト（例: ["SMA", "RSI", "MACD"]）
            **params: 指標計算用のパラメータ（例: sma_period=20）
        
        Returns:
            指標カラムが追加されたPolars DataFrame
        
        Raises:
            ValueError: 無効な指標名やパラメータが指定された場合
            RuntimeError: 指標計算に失敗した場合
        """
        pass
    
    @abstractmethod
    def validate_data(
        self,
        data: pl.DataFrame,
        data_type: str = "tick"
    ) -> tuple[bool, List[str]]:
        """データの妥当性を検証
        
        Args:
            data: 検証対象のPolars DataFrame
            data_type: データタイプ（"tick" または "ohlc"）
        
        Returns:
            (検証成功フラグ, エラーメッセージのリスト)
        
        Raises:
            ValueError: 無効なデータタイプが指定された場合
        """
        pass
    
    def resample_ohlc(
        self,
        ohlc_data: pl.DataFrame,
        source_timeframe: TimeFrame,
        target_timeframe: TimeFrame
    ) -> pl.DataFrame:
        """OHLCデータを異なる時間足にリサンプリング
        
        Args:
            ohlc_data: 元のOHLCデータ
            source_timeframe: 元の時間足
            target_timeframe: 変換先の時間足
        
        Returns:
            リサンプリングされたOHLCデータ
        
        Raises:
            ValueError: ダウンサンプリングしようとした場合
        """
        # デフォルト実装を提供
        raise NotImplementedError("Resampling is not implemented in base class")


class StorageHandler(ABC):
    """ストレージ操作の抽象基底クラス
    
    データの永続化、読み込み、クエリなどのストレージ操作を抽象化します。
    すべてのストレージハンドラーはこのインターフェースを実装する必要があります。
    """
    
    @abstractmethod
    async def save_data(
        self,
        data: Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]],
        key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """データを保存
        
        Args:
            data: 保存するデータ（DataFrameまたはモデルオブジェクトのリスト）
            key: データを識別するキー（例: "tick/USDJPY/2025-01-17"）
            metadata: 追加メタデータ（省略可）
        
        Returns:
            保存成功時True
        
        Raises:
            IOError: 保存に失敗した場合
            ValueError: 無効なデータが指定された場合
        """
        pass
    
    @abstractmethod
    async def load_data(
        self,
        key: str,
        data_type: Optional[str] = None
    ) -> Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]]:
        """データを読み込み
        
        Args:
            key: データを識別するキー
            data_type: 期待するデータタイプ（"tick", "ohlc", "prediction", "alert"）
        
        Returns:
            読み込んだデータ
        
        Raises:
            FileNotFoundError: データが存在しない場合
            IOError: 読み込みに失敗した場合
        """
        pass
    
    @abstractmethod
    async def delete_data(self, key: str) -> bool:
        """データを削除
        
        Args:
            key: 削除するデータのキー
        
        Returns:
            削除成功時True
        
        Raises:
            FileNotFoundError: データが存在しない場合
            IOError: 削除に失敗した場合
        """
        pass
    
    @abstractmethod
    async def query_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        """データをクエリ
        
        Args:
            query: クエリ文字列（実装依存）
            params: クエリパラメータ（省略可）
        
        Returns:
            クエリ結果のDataFrame
        
        Raises:
            ValueError: 無効なクエリが指定された場合
            IOError: クエリ実行に失敗した場合
        """
        pass
    
    async def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """保存されているデータキーのリストを取得
        
        Args:
            pattern: キーのパターン（例: "tick/USDJPY/*"）
        
        Returns:
            マッチするキーのリスト
        """
        # デフォルト実装を提供
        raise NotImplementedError("List keys is not implemented in base class")


class Predictor(ABC):
    """予測モデルの抽象基底クラス
    
    機械学習モデルの訓練、予測、評価などの操作を抽象化します。
    すべての予測モデルはこのインターフェースを実装する必要があります。
    """
    
    @abstractmethod
    async def train(
        self,
        training_data: pl.DataFrame,
        validation_data: Optional[pl.DataFrame] = None,
        **hyperparameters
    ) -> Dict[str, Any]:
        """モデルを訓練
        
        Args:
            training_data: 訓練データ
            validation_data: 検証データ（省略可）
            **hyperparameters: ハイパーパラメータ
        
        Returns:
            訓練結果のメトリクス（loss, accuracy等）
        
        Raises:
            ValueError: 無効なデータやパラメータが指定された場合
            RuntimeError: 訓練に失敗した場合
        """
        pass
    
    @abstractmethod
    async def predict(
        self,
        input_data: pl.DataFrame,
        **params
    ) -> List[Prediction]:
        """予測を実行
        
        Args:
            input_data: 入力データ
            **params: 予測パラメータ
        
        Returns:
            Predictionオブジェクトのリスト
        
        Raises:
            ValueError: 無効な入力データが指定された場合
            RuntimeError: 予測に失敗した場合
        """
        pass
    
    @abstractmethod
    async def evaluate(
        self,
        test_data: pl.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """モデルを評価
        
        Args:
            test_data: テストデータ
            metrics: 計算する評価指標（省略時はデフォルト指標）
        
        Returns:
            評価指標の辞書（例: {"mae": 0.1, "rmse": 0.15}）
        
        Raises:
            ValueError: 無効なデータや指標が指定された場合
            RuntimeError: 評価に失敗した場合
        """
        pass
    
    @abstractmethod
    async def save_model(self, path: Path) -> bool:
        """モデルを保存
        
        Args:
            path: 保存先パス
        
        Returns:
            保存成功時True
        
        Raises:
            IOError: 保存に失敗した場合
        """
        pass
    
    @abstractmethod
    async def load_model(self, path: Path) -> bool:
        """モデルを読み込み
        
        Args:
            path: 読み込み元パス
        
        Returns:
            読み込み成功時True
        
        Raises:
            FileNotFoundError: モデルファイルが存在しない場合
            IOError: 読み込みに失敗した場合
        """
        pass
    
    async def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """特徴量の重要度を取得
        
        Returns:
            特徴量名と重要度のマッピング（実装されていない場合None）
        """
        # デフォルト実装を提供
        return None


class DataFetcherProtocol(Protocol):
    """DataFetcherのプロトコル定義
    
    構造的部分型を利用した柔軟なインターフェース定義。
    明示的な継承なしでインターフェースを満たすことができます。
    """
    
    async def fetch_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame: ...
    
    async def fetch_ohlc_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame: ...
    
    async def get_available_symbols(self) -> List[str]: ...


class DataProcessorProtocol(Protocol):
    """DataProcessorのプロトコル定義
    
    構造的部分型を利用した柔軟なインターフェース定義。
    """
    
    def process_tick_to_ohlc(
        self,
        tick_data: pl.DataFrame,
        timeframe: TimeFrame,
        symbol: Optional[str] = None
    ) -> pl.DataFrame: ...
    
    def calculate_indicators(
        self,
        ohlc_data: pl.DataFrame,
        indicators: List[str],
        **params
    ) -> pl.DataFrame: ...
    
    def validate_data(
        self,
        data: pl.DataFrame,
        data_type: str = "tick"
    ) -> tuple[bool, List[str]]: ...


class StorageHandlerProtocol(Protocol):
    """StorageHandlerのプロトコル定義
    
    構造的部分型を利用した柔軟なインターフェース定義。
    """
    
    async def save_data(
        self,
        data: Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]],
        key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool: ...
    
    async def load_data(
        self,
        key: str,
        data_type: Optional[str] = None
    ) -> Union[pl.DataFrame, List[Union[Tick, OHLC, Prediction, Alert]]]: ...
    
    async def delete_data(self, key: str) -> bool: ...
    
    async def query_data(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame: ...


class PredictorProtocol(Protocol):
    """Predictorのプロトコル定義
    
    構造的部分型を利用した柔軟なインターフェース定義。
    """
    
    async def train(
        self,
        training_data: pl.DataFrame,
        validation_data: Optional[pl.DataFrame] = None,
        **hyperparameters
    ) -> Dict[str, Any]: ...
    
    async def predict(
        self,
        input_data: pl.DataFrame,
        **params
    ) -> List[Prediction]: ...
    
    async def evaluate(
        self,
        test_data: pl.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]: ...
    
    async def save_model(self, path: Path) -> bool: ...
    
    async def load_model(self, path: Path) -> bool: ...


# 型エイリアスを定義（利便性のため）
DataFetcherType = Union[DataFetcher, DataFetcherProtocol]
DataProcessorType = Union[DataProcessor, DataProcessorProtocol]
StorageHandlerType = Union[StorageHandler, StorageHandlerProtocol]
PredictorType = Union[Predictor, PredictorProtocol]