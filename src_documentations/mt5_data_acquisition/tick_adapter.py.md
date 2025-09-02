# tick_adapter.py

## 概要
MT5のティックデータを内部のTickモデル形式に変換するアダプターモジュール。データ形式の変換、正規化、検証機能を提供し、MT5固有の形式を抽象化します。

## 依存関係
- 外部ライブラリ: MetaTrader5, numpy, datetime
- 内部モジュール: common.models (Tick)

## 主要コンポーネント

### クラス

#### MT5TickAdapter
**目的**: MT5のティックデータを標準化されたTickモデルに変換

**属性**:
- `_symbol_info` (Dict[str, Any]): シンボル情報のキャッシュ
- `_digit_precision` (Dict[str, int]): シンボル別の価格精度

**メソッド**:
- `adapt_tick(mt5_tick, symbol)`: MT5ティックをTickモデルに変換
- `adapt_tick_batch(mt5_ticks, symbol)`: バッチ変換
- `normalize_price(price, symbol)`: 価格を正規化
- `validate_tick(tick)`: ティックデータを検証

### 関数

#### adapt_tick
**目的**: 単一のMT5ティックをTickモデルに変換

**入力**:
- `mt5_tick` (Any): MT5のティックオブジェクト
- `symbol` (str): 通貨ペアシンボル

**出力**:
- Tick: 変換されたTickモデル

**処理フロー**:
1. MT5ティックから各フィールドを抽出
2. タイムスタンプをUTCに変換
3. 価格をFloat32に変換
4. スプレッドを検証（ask > bid）
5. Tickモデルを生成

**例外**:
- `ValueError`: 無効なティックデータ
- `TypeError`: 型変換エラー

#### adapt_tick_batch
**目的**: 複数のMT5ティックを一括変換

**入力**:
- `mt5_ticks` (List[Any]): MT5ティックのリスト
- `symbol` (str): 通貨ペアシンボル

**出力**:
- List[Tick]: Tickモデルのリスト

**処理フロー**:
1. 各ティックを並列処理で変換
2. 無効なティックをフィルタリング
3. タイムスタンプでソート
4. 重複を除去

#### normalize_price
**目的**: 価格を適切な精度に正規化

**入力**:
- `price` (float): 元の価格
- `symbol` (str): 通貨ペアシンボル

**出力**:
- float: 正規化された価格（Float32）

**処理フロー**:
1. シンボルの価格精度を取得
2. 適切な小数点以下桁数に丸める
3. Float32に変換

## 使用例
```python
from src.mt5_data_acquisition.tick_adapter import MT5TickAdapter
import MetaTrader5 as mt5

# アダプターの初期化
adapter = MT5TickAdapter()

# MT5からティックを取得
mt5_ticks = mt5.copy_ticks_from("USDJPY", datetime.utcnow(), 100, mt5.COPY_TICKS_ALL)

# 単一ティックの変換
tick = adapter.adapt_tick(mt5_ticks[0], "USDJPY")
print(f"変換されたティック: {tick.bid}/{tick.ask}")

# バッチ変換
ticks = adapter.adapt_tick_batch(mt5_ticks, "USDJPY")
print(f"変換されたティック数: {len(ticks)}")

# 価格の正規化
normalized_price = adapter.normalize_price(150.12345, "USDJPY")
print(f"正規化された価格: {normalized_price}")
```

## 注意事項
- **データ型変換**: MT5の数値型をFloat32に統一
- **タイムゾーン**: MT5のローカル時刻をUTCに変換
- **精度管理**: シンボルごとの価格精度を適切に処理
- **検証**: スプレッドの妥当性など基本的な検証を実施
- **エラー処理**: 無効なデータは除外して処理を継続