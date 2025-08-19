# TickDataStreamer API リファレンス

## 概要

`TickDataStreamer`は、MetaTrader 5 (MT5) からリアルタイムティックデータを取得し、非同期ストリーミングで配信するクラスです。高性能、低レイテンシ、高信頼性を実現するための様々な機能を提供します。

## 主な機能

- **非同期ティックストリーミング**: 指定通貨ペアの最新Bid/Askをリアルタイム取得
- **リングバッファ管理**: 最大10,000件のティックデータを効率的に管理
- **スパイクフィルター**: 3σルールによる異常値の自動検出・除外
- **バックプレッシャー制御**: データ処理の遅延を防ぐ自動制御機能
- **自動再購読**: 接続障害時の自動復旧機能
- **オブジェクトプール**: メモリ効率を最適化するTickオブジェクト再利用

## クラス定義

```python
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
```

## StreamerConfig

設定パラメータを管理するデータクラスです。

### パラメータ

| パラメータ | 型 | デフォルト値 | 説明 |
|-----------|-----|-------------|------|
| `buffer_size` | int | 10000 | リングバッファのサイズ |
| `spike_threshold` | float | 3.0 | スパイク検出の閾値（σ単位） |
| `backpressure_threshold` | float | 0.8 | バックプレッシャー発動閾値（バッファ使用率） |
| `stats_window_size` | int | 1000 | 統計計算用のウィンドウサイズ |
| `max_retries` | int | 5 | 再接続の最大試行回数 |
| `retry_delay` | float | 1.0 | 再接続試行の初期待機時間（秒） |
| `warmup_ticks` | int | 100 | 統計計算開始前の必要ティック数 |

### 使用例

```python
config = StreamerConfig(
    buffer_size=5000,
    spike_threshold=2.5,
    backpressure_threshold=0.9
)
```

## TickDataStreamer

### コンストラクタ

```python
def __init__(
    self,
    symbol: str,
    connection_manager: MT5ConnectionManager,
    config: StreamerConfig | None = None,
    logger: Any = None
)
```

#### パラメータ

- `symbol` (str): 取得対象の通貨ペア（例: "EURUSD", "USDJPY"）
- `connection_manager` (MT5ConnectionManager): MT5接続管理オブジェクト
- `config` (StreamerConfig | None): 設定オブジェクト（省略時はデフォルト値使用）
- `logger` (Any): ロガーインスタンス（省略時は標準ロガー使用）

### 主要メソッド

#### subscribe_to_ticks()

MT5へのティック購読を開始します。

```python
async def subscribe_to_ticks(self) -> bool
```

**戻り値**: 
- `bool`: 購読成功時はTrue、失敗時はFalse

**例外**:
- `ConnectionError`: MT5との接続に失敗した場合
- `ValueError`: 無効なシンボルが指定された場合

**使用例**:
```python
streamer = TickDataStreamer("EURUSD", connection_manager)
success = await streamer.subscribe_to_ticks()
if success:
    print("購読開始成功")
```

#### stream_ticks()

ティックデータの非同期ストリーミングを開始します。

```python
async def stream_ticks(
    self,
    max_ticks: int | None = None,
    timeout: float | None = None
) -> AsyncGenerator[Tick, None]
```

**パラメータ**:
- `max_ticks` (int | None): 最大取得ティック数（省略時は無制限）
- `timeout` (float | None): タイムアウト時間（秒）

**戻り値**:
- `AsyncGenerator[Tick, None]`: Tickオブジェクトの非同期ジェネレータ

**使用例**:
```python
async for tick in streamer.stream_ticks(max_ticks=1000):
    print(f"Bid: {tick.bid}, Ask: {tick.ask}")
```

#### add_listener()

イベントリスナーを追加します。

```python
def add_listener(
    self,
    event_type: str,
    callback: Callable
) -> None
```

**パラメータ**:
- `event_type` (str): イベントタイプ（"tick", "error", "backpressure"）
- `callback` (Callable): コールバック関数

**イベントタイプ**:
- `"tick"`: 新しいティックを受信した時
- `"error"`: エラーが発生した時
- `"backpressure"`: バックプレッシャーが発生した時

**使用例**:
```python
def on_tick(tick):
    print(f"新しいティック: {tick}")

def on_error(error):
    print(f"エラー発生: {error}")

streamer.add_listener("tick", on_tick)
streamer.add_listener("error", on_error)
```

#### get_recent_ticks()

最近のティックデータを取得します。

```python
def get_recent_ticks(self, count: int = 100) -> list[Tick]
```

**パラメータ**:
- `count` (int): 取得するティック数（デフォルト: 100）

**戻り値**:
- `list[Tick]`: 最新のTickオブジェクトのリスト

#### stop_streaming()

ストリーミングを停止します。

```python
async def stop_streaming(self) -> None
```

**使用例**:
```python
await streamer.stop_streaming()
```

#### unsubscribe()

MT5の購読を解除します。

```python
def unsubscribe(self) -> bool
```

**戻り値**:
- `bool`: 解除成功時はTrue、失敗時はFalse

### プロパティ

#### buffer_usage

バッファ使用率を取得します。

```python
@property
def buffer_usage(self) -> float
```

**戻り値**: 0.0〜1.0の使用率

#### current_stats

現在の統計情報を取得します。

```python
@property
def current_stats(self) -> dict
```

**戻り値の例**:
```python
{
    "symbol": "EURUSD",
    "total_ticks": 5432,
    "spike_count": 12,
    "dropped_ticks": 0,
    "backpressure_count": 2,
    "buffer_usage": 0.54,
    "mean_bid": 1.0856,
    "std_bid": 0.0002,
    "mean_ask": 1.0858,
    "std_ask": 0.0002,
    "error_stats": {...},
    "performance": {
        "latency_avg_ms": 2.8,
        "latency_p95_ms": 4.2,
        "latency_max_ms": 8.5
    },
    "object_pool": {
        "created": 100,
        "reused": 5332,
        "active": 54,
        "efficiency": 0.98
    }
}
```

#### is_connected

接続状態を確認します。

```python
@property
def is_connected(self) -> bool
```

**戻り値**: 接続中の場合True

## パフォーマンス特性

### レイテンシ
- **平均**: 3ms
- **95パーセンタイル**: 5ms
- **最大**: 10ms以内

### スループット
- **最大処理能力**: 1,000 ticks/秒
- **推奨レート**: 100-500 ticks/秒

### メモリ使用量
- **基本使用量**: 約50MB
- **フルバッファ時**: 約100MB（10,000ティック格納時）

## エラーハンドリング

### サーキットブレーカー
- 5回連続でエラーが発生すると自動的に一時停止
- 30秒後に自動復旧を試行
- HALF_OPEN状態で段階的に復旧

### 自動再購読
- エクスポネンシャルバックオフ（1, 2, 4, 8, 16秒）
- 最大5回まで自動再試行
- 成功時は統計情報をリセット

## ベストプラクティス

### 推奨設定

```python
# 高頻度取引向け
config = StreamerConfig(
    buffer_size=20000,
    spike_threshold=2.0,
    backpressure_threshold=0.7,
    stats_window_size=500
)

# 安定性重視
config = StreamerConfig(
    buffer_size=5000,
    spike_threshold=3.5,
    backpressure_threshold=0.9,
    stats_window_size=2000
)
```

### エラー処理の実装例

```python
async def run_with_error_handling():
    streamer = TickDataStreamer("EURUSD", connection_manager)
    
    def on_error(error):
        logging.error(f"Stream error: {error}")
        # カスタムエラー処理
    
    streamer.add_listener("error", on_error)
    
    try:
        async for tick in streamer.stream_ticks():
            # ティック処理
            process_tick(tick)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await streamer.stop_streaming()
```

### 複数通貨ペアの同時処理

```python
async def stream_multiple_symbols():
    symbols = ["EURUSD", "USDJPY", "GBPUSD"]
    streamers = []
    
    for symbol in symbols:
        streamer = TickDataStreamer(symbol, connection_manager)
        await streamer.subscribe_to_ticks()
        streamers.append(streamer)
    
    # 並行ストリーミング
    tasks = [
        process_symbol_stream(streamer) 
        for streamer in streamers
    ]
    await asyncio.gather(*tasks)
```

## 関連クラス

- [Tick](../models/tick.md): ティックデータモデル
- [MT5ConnectionManager](../mt5/connection_manager.md): MT5接続管理
- [CircuitBreaker](../patterns/circuit_breaker.md): サーキットブレーカーパターン

## 更新履歴

- v1.0.0 (2025-01-19): 初回リリース
  - 非同期ストリーミング実装
  - スパイクフィルター追加
  - バックプレッシャー制御実装
  - オブジェクトプール最適化