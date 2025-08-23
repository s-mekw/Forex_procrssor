# tick_to_bar.py

## 概要
ティックデータからOHLC（バー/ローソク足）データへの変換を担当するモジュール。リアルタイムストリーミングデータの集約と、効率的なバー生成機能を提供します。

## 依存関係
- 外部ライブラリ: polars, numpy, asyncio, datetime
- 内部モジュール: common.models (Tick, OHLC, TimeFrame), data_processing.processor

## 主要コンポーネント

### クラス

#### TickToBarConverter
**目的**: ティックデータをOHLCバーに変換するコンバーター

**属性**:
- `_buffer` (Dict[str, List[Tick]]): シンボル別のティックバッファ
- `_current_bars` (Dict[str, OHLC]): 生成中の現在のバー
- `_timeframe` (TimeFrame): 変換先の時間足
- `_aggregation_method` (str): 集約方法（time_based, tick_count, volume）

**メソッド**:
- `add_tick(tick)`: ティックを追加してバー生成を試行
- `force_close_bar(symbol)`: 現在のバーを強制的に閉じる
- `get_current_bar(symbol)`: 生成中のバーを取得
- `convert_batch(ticks, timeframe)`: バッチ変換を実行

### 関数

#### add_tick
**目的**: 新しいティックを追加し、必要に応じてバーを生成

**入力**:
- `tick` (Tick): 追加するティックデータ

**出力**:
- Optional[OHLC]: 完成したバー（ある場合）

**処理フロー**:
1. ティックをバッファに追加
2. 時間枠の境界をチェック
3. 境界を超えた場合、現在のバーを完成
4. 新しいバーを開始
5. Float32制約を維持

#### convert_batch
**目的**: ティックデータのバッチをOHLCに一括変換

**入力**:
- `ticks` (List[Tick]): ティックデータのリスト
- `timeframe` (TimeFrame): 目標時間足

**出力**:
- pl.DataFrame: OHLCデータ

**処理フロー**:
1. ティックを時間でグループ化
2. 各グループでOHLC値を計算
3. ボリュームを集計
4. DataFrameとして返す

## 使用例
```python
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter
from src.common.models import Tick, TimeFrame

# コンバーターの初期化
converter = TickToBarConverter(timeframe=TimeFrame.M1)

# ティックを追加
tick = Tick(
    timestamp=datetime.utcnow(),
    symbol="USDJPY",
    bid=150.123,
    ask=150.125,
    volume=100
)

# バーが完成した場合、OHLCが返される
completed_bar = converter.add_tick(tick)
if completed_bar:
    print(f"バー完成: O={completed_bar.open} H={completed_bar.high}")

# バッチ変換
ticks = [...]  # ティックのリスト
ohlc_df = converter.convert_batch(ticks, TimeFrame.H1)
```

## 注意事項
- **時間境界**: 時間足の境界を正確に判定
- **バッファ管理**: メモリ効率のためバッファサイズを制限
- **Float32精度**: すべての価格計算でFloat32を維持