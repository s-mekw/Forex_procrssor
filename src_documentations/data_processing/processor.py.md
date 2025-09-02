# processor.py

## 概要
Forex取引データの処理と変換を担当するモジュール。Tickデータの集約、OHLC生成、テクニカル指標計算、データ検証などのデータ処理機能を提供します。

## 依存関係
- 外部ライブラリ: polars, numpy, typing
- 内部モジュール: common.interfaces (DataProcessor), common.models (TimeFrame, Tick, OHLC)

## 主要コンポーネント

### クラス

#### DataProcessorImpl
**目的**: DataProcessorインターフェースの具体的な実装

**メソッド**:
- `process_tick_to_ohlc(tick_data, timeframe, symbol)`: TickデータをOHLCデータに変換
- `calculate_indicators(ohlc_data, indicators, **params)`: テクニカル指標を計算
- `validate_data(data, data_type)`: データの妥当性を検証
- `resample_ohlc(ohlc_data, source_timeframe, target_timeframe)`: OHLCデータをリサンプリング

### 関数

#### process_tick_to_ohlc
**目的**: TickデータからOHLCデータへの変換処理

**入力**:
- `tick_data` (pl.DataFrame): Tickデータを含むPolars DataFrame
- `timeframe` (TimeFrame): 変換先の時間足（M1, M5, H1など）
- `symbol` (Optional[str]): 対象シンボル（省略時は全シンボル）

**出力**:
- pl.DataFrame: OHLC形式に変換されたデータ

**処理フロー**:
1. Tickデータをタイムスタンプでソート
2. 指定された時間足でグループ化
3. 各グループでOHLC値を計算（Open: 最初, High: 最大, Low: 最小, Close: 最後）
4. ボリュームを集計
5. Float32制約を適用

**例外**:
- `ValueError`: 無効なデータや時間足が指定された場合

#### calculate_indicators
**目的**: テクニカル指標の計算

**入力**:
- `ohlc_data` (pl.DataFrame): OHLCデータ
- `indicators` (List[str]): 計算する指標名のリスト（例: ["SMA", "RSI", "MACD"]）
- `**params`: 指標計算用のパラメータ（例: sma_period=20, rsi_period=14）

**出力**:
- pl.DataFrame: 指標カラムが追加されたDataFrame

**処理フロー**:
1. 指定された各指標に対してループ
2. 指標に応じた計算ロジックを適用
3. 結果を新しいカラムとして追加
4. Float32制約を維持

**例外**:
- `ValueError`: 無効な指標名やパラメータが指定された場合
- `RuntimeError`: 指標計算に失敗した場合

#### validate_data
**目的**: データの整合性と妥当性を検証

**入力**:
- `data` (pl.DataFrame): 検証対象のデータ
- `data_type` (str): データタイプ（"tick" または "ohlc"）

**出力**:
- tuple[bool, List[str]]: (検証成功フラグ, エラーメッセージのリスト)

**処理フロー**:
1. 必須カラムの存在チェック
2. データ型の検証
3. 値の範囲チェック（価格が正、スプレッドが妥当など）
4. タイムスタンプの順序性検証
5. Float32制約の確認

## 使用例
```python
from src.data_processing.processor import DataProcessorImpl
from src.common.models import TimeFrame
import polars as pl

# プロセッサーのインスタンス化
processor = DataProcessorImpl()

# TickデータをOHLCに変換
tick_df = pl.DataFrame({
    "timestamp": [...],
    "symbol": ["USDJPY"] * n,
    "bid": [...],
    "ask": [...],
    "volume": [...]
})

ohlc_df = processor.process_tick_to_ohlc(
    tick_data=tick_df,
    timeframe=TimeFrame.H1,
    symbol="USDJPY"
)

# テクニカル指標を計算
with_indicators = processor.calculate_indicators(
    ohlc_data=ohlc_df,
    indicators=["SMA", "RSI"],
    sma_period=20,
    rsi_period=14
)

# データの検証
is_valid, errors = processor.validate_data(ohlc_df, "ohlc")
if not is_valid:
    print(f"Validation errors: {errors}")
```

## 注意事項
- **Polars必須**: すべてのDataFrameはPolars形式（Pandas禁止）
- **Float32制約**: すべての数値計算はnp.float32で実行
- **タイムゾーン**: すべてのタイムスタンプはUTCで処理
- **メモリ効率**: 大量データ処理時はバッチ処理を推奨
- **エラーハンドリング**: 無効なデータは早期に検証して除外