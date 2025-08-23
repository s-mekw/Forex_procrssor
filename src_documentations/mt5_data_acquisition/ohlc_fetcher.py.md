# ohlc_fetcher.py

## 概要
MetaTrader 5からOHLC（四本値）データを取得するモジュール。各種時間足のローソク足データの取得と、リアルタイム更新機能を提供します。

## 依存関係
- 外部ライブラリ: MetaTrader5, polars, numpy, asyncio, datetime
- 内部モジュール: common.models (OHLC, TimeFrame), mt5_client (MT5ConnectionManager)

## 主要コンポーネント

### クラス

#### MT5OHLCFetcher
**目的**: MT5からOHLCデータを効率的に取得

**属性**:
- `_connection_manager` (MT5ConnectionManager): MT5接続マネージャー
- `_timeframe_mapping` (Dict[TimeFrame, int]): TimeFrameとMT5定数のマッピング
- `_cache` (Dict[str, pl.DataFrame]): OHLCデータのキャッシュ

**メソッド**:
- `fetch_ohlc_data(symbol, timeframe, start_time, end_time, limit)`: OHLCデータを取得
- `fetch_latest_bars(symbol, timeframe, count)`: 最新のバーを取得
- `stream_ohlc_updates(symbol, timeframe, callback)`: OHLCデータのリアルタイム更新
- `convert_timeframe(timeframe)`: TimeFrameをMT5形式に変換

### 関数

#### fetch_ohlc_data
**目的**: 指定期間のOHLCデータを取得

**入力**:
- `symbol` (str): 通貨ペアシンボル
- `timeframe` (TimeFrame): 時間足（M1, M5, H1など）
- `start_time` (datetime): 開始時刻（UTC）
- `end_time` (datetime): 終了時刻（UTC）
- `limit` (Optional[int]): 取得バー数の上限

**出力**:
- pl.DataFrame: OHLCデータ

**処理フロー**:
1. TimeFrameをMT5形式に変換
2. copy_rates_range関数でデータ取得
3. Polars DataFrameに変換
4. Float32制約を適用
5. OHLC検証（high >= low など）

**例外**:
- `ValueError`: 無効な時間足または期間
- `ConnectionError`: MT5接続エラー

## 使用例
```python
from src.mt5_data_acquisition.ohlc_fetcher import MT5OHLCFetcher
from src.common.models import TimeFrame
from datetime import datetime, timedelta

fetcher = MT5OHLCFetcher()

# 1時間足データを取得
ohlc_data = await fetcher.fetch_ohlc_data(
    symbol="USDJPY",
    timeframe=TimeFrame.H1,
    start_time=datetime.utcnow() - timedelta(days=7),
    end_time=datetime.utcnow()
)

# 最新の20本のバーを取得
latest_bars = await fetcher.fetch_latest_bars(
    symbol="EURUSD",
    timeframe=TimeFrame.M5,
    count=20
)
```

## 注意事項
- **時間足マッピング**: TimeFrameとMT5のTIMEFRAME定数を正確にマッピング
- **データ検証**: OHLC関係（high >= {open,close} >= low）を検証
- **キャッシュ管理**: 頻繁にアクセスするデータはキャッシュで効率化