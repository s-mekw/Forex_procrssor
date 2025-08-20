# TickDataStreamer トラブルシューティングガイド

## 目次
1. [よくある問題と解決方法](#よくある問題と解決方法)
2. [パフォーマンスチューニング](#パフォーマンスチューニング)
3. [デバッグ方法](#デバッグ方法)
4. [ログ解析](#ログ解析)
5. [エラーコード一覧](#エラーコード一覧)

## よくある問題と解決方法

### 1. 接続エラー

#### 症状
```
ERROR: Connection failed: [Errno 10061] No connection could be made
```

#### 原因
- MT5が起動していない
- MT5の自動取引が無効
- ファイアウォールがブロックしている
- 認証情報が正しくない

#### 解決方法
```python
# 1. MT5が起動していることを確認
import MetaTrader5 as mt5
if not mt5.initialize():
    print("MT5初期化失敗:", mt5.last_error())

# 2. 自動取引を有効化
# MT5 GUI: ツール → オプション → エキスパートアドバイザー → 「自動売買を許可する」にチェック

# 3. 接続テスト
connection_manager = MT5ConnectionManager()
if connection_manager.connect():
    print("接続成功")
else:
    print("接続失敗")

# 4. ファイアウォール設定確認
# Windows: コントロールパネル → Windows Defender ファイアウォール
# MT5 (terminal64.exe) を許可リストに追加
```

### 2. シンボルが見つからない

#### 症状
```
ValueError: Symbol EURUSD not found or not visible
```

#### 原因
- シンボル名が正しくない
- シンボルが無効化されている
- ブローカーがそのシンボルを提供していない

#### 解決方法
```python
# 利用可能なシンボルを確認
import MetaTrader5 as mt5

mt5.initialize()
symbols = mt5.symbols_get()
print("利用可能なシンボル:")
for symbol in symbols[:10]:  # 最初の10個を表示
    print(f"  - {symbol.name}")

# シンボルを有効化
symbol_name = "EURUSD"
if not mt5.symbol_select(symbol_name, True):
    print(f"{symbol_name} の有効化失敗")
```

### 3. メモリリーク

#### 症状
- メモリ使用量が継続的に増加
- 長時間実行後にクラッシュ
- パフォーマンスが徐々に低下

#### 原因
- リスナーの解放忘れ
- バッファのクリア忘れ
- オブジェクトプールの不適切な使用

#### 解決方法
```python
# 適切なクリーンアップ
async def safe_streaming():
    streamer = TickDataStreamer("EURUSD", connection_manager)
    
    try:
        await streamer.subscribe_to_ticks()
        # ストリーミング処理
        async for tick in streamer.stream_ticks(max_ticks=1000):
            process_tick(tick)
    finally:
        # 必ずクリーンアップを実行
        streamer.clear_buffer()
        streamer.unsubscribe()
        await streamer.stop_streaming()
        
        # リスナーをクリア
        streamer._listeners.clear()

# メモリ監視
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"メモリ使用量: {mem_info.rss / 1024 / 1024:.2f} MB")
```

### 4. スパイクフィルターが機能しない

#### 症状
- 明らかな異常値が除外されない
- spike_countが増加しない

#### 原因
- ウォームアップ期間中
- 閾値設定が高すぎる
- 統計計算が正しくない

#### 解決方法
```python
# 設定の調整
config = StreamerConfig(
    spike_threshold=2.0,  # より厳しい閾値（デフォルト3.0）
    warmup_ticks=50,      # ウォームアップ期間を短く
    stats_window_size=500  # 統計ウィンドウを小さく
)

# スパイク検出のデバッグ
def debug_spike_filter(streamer):
    stats = streamer.current_stats
    print(f"平均Bid: {stats.get('mean_bid', 'N/A')}")
    print(f"標準偏差Bid: {stats.get('std_bid', 'N/A')}")
    print(f"スパイク検出数: {stats.get('spike_count', 0)}")
    
    # 手動でZスコアを計算
    recent_ticks = streamer.get_recent_ticks(10)
    for tick in recent_ticks:
        if stats.get('std_bid') and stats['std_bid'] > 0:
            z_score = abs(tick.bid - stats['mean_bid']) / stats['std_bid']
            print(f"Tick Bid={tick.bid:.5f}, Z-score={z_score:.2f}")
```

### 5. バックプレッシャーの頻発

#### 症状
```
WARNING: Backpressure triggered at 80% buffer usage
```

#### 原因
- ティック処理が遅い
- バッファサイズが小さい
- 閾値が低すぎる

#### 解決方法
```python
# 1. バッファサイズを増やす
config = StreamerConfig(
    buffer_size=20000,  # デフォルトの2倍
    backpressure_threshold=0.9  # 閾値を上げる
)

# 2. 処理を最適化
async def optimized_processing():
    batch = []
    async for tick in streamer.stream_ticks():
        batch.append(tick)
        
        # バッチ処理で効率化
        if len(batch) >= 100:
            await process_batch(batch)
            batch.clear()

# 3. 非同期処理の改善
async def async_processing():
    async def process_tick_async(tick):
        # 重い処理を非同期で実行
        await asyncio.sleep(0)  # イベントループに制御を返す
        # 実際の処理
        
    tasks = []
    async for tick in streamer.stream_ticks():
        task = asyncio.create_task(process_tick_async(tick))
        tasks.append(task)
        
        # 定期的にタスクを待機
        if len(tasks) >= 10:
            await asyncio.gather(*tasks)
            tasks.clear()
```

## パフォーマンスチューニング

### レイテンシの最適化

```python
# 1. 適切な待機時間の設定
class OptimizedStreamer(TickDataStreamer):
    async def stream_ticks(self, max_ticks=None, timeout=None):
        # デフォルトの5msから2msに短縮
        await asyncio.sleep(0.002)
        # ...

# 2. 不要な処理を削除
# BAD
async for tick in streamer.stream_ticks():
    print(f"Debug: {tick}")  # 本番環境では削除
    process_tick(tick)

# GOOD
if DEBUG:
    async for tick in streamer.stream_ticks():
        print(f"Debug: {tick}")
        process_tick(tick)
else:
    async for tick in streamer.stream_ticks():
        process_tick(tick)
```

### メモリ使用量の最適化

```python
# 1. 適切なバッファサイズ
def calculate_optimal_buffer_size(ticks_per_second, retention_seconds):
    """最適なバッファサイズを計算"""
    return int(ticks_per_second * retention_seconds * 1.2)  # 20%の余裕

# 例: 100 ticks/秒で60秒保持
buffer_size = calculate_optimal_buffer_size(100, 60)  # 7200

# 2. オブジェクトプールのサイズ調整
pool_size = min(buffer_size // 10, 1000)  # バッファの10%か1000の小さい方
```

### CPU使用率の最適化

```python
# 1. 統計計算の間引き
class OptimizedStats:
    def __init__(self, update_interval=10):
        self.update_interval = update_interval
        self.tick_count = 0
    
    def should_update(self):
        self.tick_count += 1
        return self.tick_count % self.update_interval == 0

# 2. 条件分岐の最適化
# BAD
if condition1:
    if condition2:
        if condition3:
            process()

# GOOD
if condition1 and condition2 and condition3:
    process()
```

## デバッグ方法

### ログレベルの設定

```python
import logging
import structlog

# 標準ログ
logging.basicConfig(
    level=logging.DEBUG,  # DEBUGレベルで詳細ログ
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tick_streamer.log'),
        logging.StreamHandler()
    ]
)

# structlog設定
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### プロファイリング

```python
import cProfile
import pstats
from io import StringIO

def profile_streaming():
    """ストリーミングのプロファイリング"""
    profiler = cProfile.Profile()
    
    # プロファイリング開始
    profiler.enable()
    
    # ストリーミング実行
    asyncio.run(stream_test())
    
    # プロファイリング終了
    profiler.disable()
    
    # 結果出力
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # 上位20個の関数
    print(s.getvalue())

# 時間計測
import time

async def measure_latency():
    """レイテンシ測定"""
    latencies = []
    
    async for tick in streamer.stream_ticks(max_ticks=100):
        start = time.perf_counter()
        process_tick(tick)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
    
    print(f"平均レイテンシ: {sum(latencies)/len(latencies):.2f}ms")
    print(f"最大レイテンシ: {max(latencies):.2f}ms")
    print(f"最小レイテンシ: {min(latencies):.2f}ms")
```

### メモリダンプ

```python
import tracemalloc
import gc

# メモリトレース開始
tracemalloc.start()

# ストリーミング実行
# ...

# スナップショット取得
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 メモリ使用箇所 ]")
for stat in top_stats[:10]:
    print(stat)

# ガベージコレクション情報
print(f"GCオブジェクト数: {len(gc.get_objects())}")
print(f"GC統計: {gc.get_stats()}")
```

## ログ解析

### ログフォーマット

```
2025-01-19 10:15:32.123 - tick_fetcher - INFO - Subscribe successful: {'symbol': 'EURUSD', 'timestamp': '2025-01-19T10:15:32.123456'}
```

### 重要なログメッセージ

| ログレベル | メッセージパターン | 意味 | 対処法 |
|-----------|-------------------|------|--------|
| ERROR | `Connection lost` | MT5接続が切断 | 自動再接続を待つか手動再接続 |
| ERROR | `Circuit breaker OPEN` | サーキットブレーカー作動 | 30秒待って自動復旧 |
| WARNING | `Spike detected` | 異常値検出 | spike_thresholdの調整を検討 |
| WARNING | `Backpressure at X%` | バッファ使用率警告 | 処理速度改善かバッファ拡大 |
| INFO | `Subscribe successful` | 購読成功 | 正常動作 |
| DEBUG | `Tick processed` | ティック処理完了 | デバッグ用 |

### ログ分析スクリプト

```python
import re
from collections import Counter
from datetime import datetime

def analyze_logs(log_file):
    """ログファイルの分析"""
    
    errors = []
    warnings = []
    spike_count = 0
    backpressure_count = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'ERROR' in line:
                errors.append(line)
            elif 'WARNING' in line:
                warnings.append(line)
                if 'Spike detected' in line:
                    spike_count += 1
                elif 'Backpressure' in line:
                    backpressure_count += 1
    
    print(f"エラー数: {len(errors)}")
    print(f"警告数: {len(warnings)}")
    print(f"スパイク検出: {spike_count}")
    print(f"バックプレッシャー: {backpressure_count}")
    
    # エラータイプの集計
    error_types = Counter()
    for error in errors:
        match = re.search(r'error_type.*?([A-Z]\w+Error)', error)
        if match:
            error_types[match.group(1)] += 1
    
    print("\nエラータイプ別集計:")
    for error_type, count in error_types.most_common():
        print(f"  {error_type}: {count}")
```

## エラーコード一覧

| エラーコード | エラータイプ | 説明 | 対処法 |
|-------------|-------------|------|--------|
| E001 | ConnectionError | MT5接続失敗 | MT5の起動確認、認証情報確認 |
| E002 | SymbolError | シンボル無効 | シンボル名確認、有効化 |
| E003 | TimeoutError | タイムアウト | タイムアウト値を増やす |
| E004 | DataError | データ不正 | データ検証、MT5再起動 |
| E005 | BufferOverflow | バッファ溢れ | バッファサイズ拡大 |
| E006 | CircuitBreakerOpen | 保護機能作動 | 自動復旧を待つ |
| E007 | MemoryError | メモリ不足 | メモリ解放、プロセス再起動 |
| E008 | PermissionError | 権限不足 | MT5の自動取引許可確認 |

## 問題が解決しない場合

1. **ログファイルの収集**
   - tick_streamer.log
   - MT5のログ（Experts/Logsフォルダ）
   - システムイベントログ

2. **環境情報の確認**
   ```python
   import sys
   import MetaTrader5 as mt5
   
   print(f"Python: {sys.version}")
   print(f"MT5: {mt5.version()}")
   print(f"OS: {sys.platform}")
   ```

3. **最小再現コードの作成**
   - 問題を再現する最小限のコード
   - 設定パラメータ
   - エラーメッセージ全文

4. **サポートへの連絡**
   - 上記情報を含めて報告
   - 再現手順を明確に記載