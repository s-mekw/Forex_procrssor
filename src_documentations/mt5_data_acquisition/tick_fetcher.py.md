# tick_fetcher.py

## 概要
MetaTrader 5からリアルタイムおよび履歴のティックデータを取得するモジュール。DataFetcherインターフェースを実装し、効率的なティックデータの取得とストリーミング機能を提供します。

## 依存関係
- 外部ライブラリ: MetaTrader5, polars, numpy, asyncio, datetime
- 内部モジュール: common.interfaces (DataFetcher), common.models (Tick), mt5_client (MT5ConnectionManager)

## 主要コンポーネント

### クラス

#### MT5TickFetcher
**目的**: MT5からティックデータを取得するDataFetcherの実装

**属性**:
- `_connection_manager` (MT5ConnectionManager): MT5接続マネージャー
- `_symbol_info_cache` (Dict[str, Any]): シンボル情報のキャッシュ
- `_streaming_tasks` (Dict[str, asyncio.Task]): ストリーミングタスクの管理

**メソッド**:
- `fetch_tick_data(symbol, start_time, end_time, limit)`: 履歴ティックデータを取得
- `stream_tick_data(symbol, callback)`: リアルタイムティックストリーミング
- `get_available_symbols()`: 利用可能なシンボルリストを取得
- `get_last_tick(symbol)`: 最新のティックを取得
- `validate_symbol(symbol)`: シンボルの妥当性を検証

### 関数

#### fetch_tick_data
**目的**: 指定期間のティックデータを取得

**入力**:
- `symbol` (str): 通貨ペアシンボル（例: USDJPY）
- `start_time` (datetime): 取得開始時刻（UTC）
- `end_time` (datetime): 取得終了時刻（UTC）
- `limit` (Optional[int]): 取得件数の上限

**出力**:
- pl.DataFrame: ティックデータのDataFrame

**処理フロー**:
1. シンボルの妥当性を検証
2. MT5のcopy_ticks_range関数を呼び出し
3. 取得したデータをPolars DataFrameに変換
4. Float32制約を適用
5. タイムスタンプでソート

**例外**:
- `ConnectionError`: MT5接続エラー
- `ValueError`: 無効なシンボルまたは期間

#### stream_tick_data
**目的**: リアルタイムティックデータのストリーミング

**入力**:
- `symbol` (str): 通貨ペアシンボル
- `callback` (Callable[[Tick], None]): ティック受信時のコールバック関数

**出力**:
- asyncio.Task: ストリーミングタスク

**処理フロー**:
1. シンボルのサブスクリプションを開始
2. 非同期ループでティックを監視
3. 新しいティックをTickモデルに変換
4. コールバック関数を呼び出し
5. エラー時は自動再接続

#### get_last_tick
**目的**: 最新のティック情報を取得

**入力**:
- `symbol` (str): 通貨ペアシンボル

**出力**:
- Tick: 最新のティックデータ

**処理フロー**:
1. MT5のsymbol_info_tick関数を呼び出し
2. Tickモデルに変換
3. Float32制約を適用

## 使用例
```python
from src.mt5_data_acquisition.tick_fetcher import MT5TickFetcher
from datetime import datetime, timedelta

# ティックフェッチャーの初期化
fetcher = MT5TickFetcher()

# 履歴ティックデータの取得
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=1)

tick_data = await fetcher.fetch_tick_data(
    symbol="USDJPY",
    start_time=start_time,
    end_time=end_time,
    limit=10000
)

print(f"取得したティック数: {len(tick_data)}")

# 最新ティックの取得
last_tick = await fetcher.get_last_tick("USDJPY")
print(f"最新ティック - Bid: {last_tick.bid}, Ask: {last_tick.ask}")

# リアルタイムストリーミング
async def on_tick(tick: Tick):
    print(f"新しいティック - {tick.symbol}: {tick.bid}/{tick.ask}")

stream_task = await fetcher.stream_tick_data("USDJPY", on_tick)

# ストリーミングを停止
stream_task.cancel()

# 利用可能なシンボルの取得
symbols = await fetcher.get_available_symbols()
print(f"利用可能なシンボル: {symbols}")
```

## 注意事項
- **非同期処理**: すべてのメソッドは非同期で実行
- **Float32制約**: すべての価格データはnp.float32で処理
- **UTC時刻**: すべてのタイムスタンプはUTCで統一
- **キャッシュ**: シンボル情報はキャッシュして効率化
- **ストリーミング管理**: 不要なストリーミングは適切に停止
- **エラーハンドリング**: 接続断時は自動的に再接続を試行