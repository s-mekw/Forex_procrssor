# interfaces.py

## 概要
Forex Processorシステムの中核となる抽象基底クラス（ABC）とProtocolベースのインターフェースを定義。データ取得、処理、保存、予測の各コンポーネントの契約を規定し、システム全体の一貫性と拡張性を保証します。

## 依存関係
- 外部ライブラリ: polars, numpy, typing, abc
- 内部モジュール: models.py (Tick, OHLC, Prediction, Alert, TimeFrame)

## 主要コンポーネント

### 抽象基底クラス（ABC）

#### DataFetcher
**目的**: データ取得の抽象基底クラス。外部データソースからのデータ取得を抽象化

**メソッド**:
- `fetch_tick_data(symbol, start_time, end_time, limit)`: Tickデータを取得
- `fetch_ohlc_data(symbol, timeframe, start_time, end_time, limit)`: OHLCデータを取得
- `get_available_symbols()`: 利用可能な通貨ペアシンボルのリストを取得
- `is_connected()`: データソースへの接続状態を確認

#### DataProcessor
**目的**: データ処理の抽象基底クラス。データの変換、指標計算、検証などを抽象化

**メソッド**:
- `process_tick_to_ohlc(tick_data, timeframe, symbol)`: TickデータからOHLCデータへ変換
- `calculate_indicators(ohlc_data, indicators, **params)`: テクニカル指標を計算
- `validate_data(data, data_type)`: データの妥当性を検証
- `resample_ohlc(ohlc_data, source_timeframe, target_timeframe)`: OHLCデータをリサンプリング

#### StorageHandler
**目的**: データ保存の抽象基底クラス。データの永続化処理を抽象化

**メソッド**:
- `save_tick_data(tick_data, storage_type)`: Tickデータを保存
- `save_ohlc_data(ohlc_data, storage_type)`: OHLCデータを保存
- `load_tick_data(symbol, start_time, end_time, storage_type)`: Tickデータを読み込み
- `load_ohlc_data(symbol, timeframe, start_time, end_time, storage_type)`: OHLCデータを読み込み
- `delete_data(symbol, start_time, end_time, data_type, storage_type)`: データを削除
- `get_storage_info()`: ストレージ情報を取得

#### Predictor
**目的**: 予測処理の抽象基底クラス。機械学習モデルによる予測を抽象化

**メソッド**:
- `train(training_data, **params)`: モデルを訓練
- `predict(input_data, **params)`: 予測を実行
- `evaluate(test_data, predictions)`: モデル性能を評価
- `save_model(path)`: モデルを保存
- `load_model(path)`: モデルを読み込み
- `get_model_info()`: モデル情報を取得

### Protocolベースインターフェース

#### DataFetcherProtocol
**目的**: DataFetcherのProtocol版。ダックタイピングによる柔軟な実装を可能に

**入力**:
- 各メソッドはDataFetcherと同じシグネチャ

**出力**:
- 各メソッドはDataFetcherと同じ戻り値

**処理フロー**:
1. runtime_checkableデコレータにより実行時型チェックが可能
2. 継承不要でインターフェースを満たす実装が可能

#### DataProcessorProtocol
**目的**: DataProcessorのProtocol版

**特徴**:
- DataProcessorと同じメソッドシグネチャ
- 柔軟な実装選択が可能

#### StorageHandlerProtocol
**目的**: StorageHandlerのProtocol版

**特徴**:
- ストレージ処理の柔軟な実装
- 異なるストレージバックエンドの統一インターフェース

#### PredictorProtocol
**目的**: PredictorのProtocol版

**特徴**:
- ML/DLモデルの統一インターフェース
- モデル実装の柔軟性

### 型エイリアス

#### DataFetcherType
```python
Union[DataFetcher, DataFetcherProtocol]
```
ABC実装とProtocol実装の両方を受け入れる型

#### DataProcessorType
```python
Union[DataProcessor, DataProcessorProtocol]
```

#### StorageHandlerType
```python
Union[StorageHandler, StorageHandlerProtocol]
```

#### PredictorType
```python
Union[Predictor, PredictorProtocol]
```

## 使用例
```python
from src.common.interfaces import DataFetcher, DataFetcherProtocol
import polars as pl
from datetime import datetime

# ABC継承による実装
class MT5DataFetcher(DataFetcher):
    async def fetch_tick_data(
        self, symbol: str, start_time: datetime, 
        end_time: datetime, limit: Optional[int] = None
    ) -> pl.DataFrame:
        # MT5からTickデータを取得
        pass
    
    async def fetch_ohlc_data(
        self, symbol: str, timeframe: TimeFrame,
        start_time: datetime, end_time: datetime,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        # MT5からOHLCデータを取得
        pass
    
    async def get_available_symbols(self) -> List[str]:
        # 利用可能なシンボルリストを返す
        return ["USDJPY", "EURUSD", "GBPUSD"]

# Protocolベースの実装（継承不要）
class APIDataFetcher:  # DataFetcherProtocolを継承しない
    async def fetch_tick_data(self, symbol, start_time, end_time, limit=None):
        # API経由でデータ取得
        pass
    
    # 必要なメソッドを実装すればProtocolを満たす
```

## 注意事項
- **ABC vs Protocol**: ABCは厳密な契約、Protocolは柔軟な実装を提供
- **非同期処理**: DataFetcherのメソッドは非同期（async）で実装
- **Polars必須**: すべてのDataFrameはPolars形式（Pandas禁止）
- **Float32制約**: 数値データはnp.float32として処理
- **エラーハンドリング**: 各メソッドで適切な例外を発生させる必要がある
- **型エイリアス**: ABCとProtocolの両方を受け入れる柔軟な型システム