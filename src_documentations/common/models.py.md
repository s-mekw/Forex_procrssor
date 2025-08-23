# models.py

## 概要
Forex取引データの中核となるPydanticモデルを定義するモジュール。Tick、OHLC、Prediction、Alertなどのデータモデルを提供し、Float32制約による メモリ最適化とバリデーション機能を実装しています。

## 依存関係
- 外部ライブラリ: pydantic, numpy, enum
- 内部モジュール: なし（独立したモデル定義）

## 主要コンポーネント

### Enum定義

#### TimeFrame
**目的**: OHLC データの時間枠を定義する列挙型

**値**:
- `M1`: 1分足
- `M5`: 5分足
- `M15`: 15分足
- `M30`: 30分足
- `H1`: 1時間足
- `H4`: 4時間足
- `D1`: 日足
- `W1`: 週足
- `MN`: 月足

#### PredictionType
**目的**: 予測の種類を定義する列挙型

**値**:
- `PRICE`: 価格予測
- `TREND`: トレンド予測
- `VOLATILITY`: ボラティリティ予測

#### AlertType
**目的**: アラートの種類を定義する列挙型

**値**:
- `PRICE_CHANGE`: 価格変動
- `TREND_REVERSAL`: トレンド転換
- `VOLUME_SPIKE`: 出来高急増
- `PATTERN_DETECTED`: パターン検出

#### AlertSeverity
**目的**: アラートの重要度を定義する列挙型

**値**:
- `INFO`: 情報
- `WARNING`: 警告
- `CRITICAL`: 重要

### クラス

#### Tick
**目的**: 為替レートのティックデータを表現するモデル

**属性**:
- `timestamp` (datetime): ティックのタイムスタンプ（UTC）
- `symbol` (str): 通貨ペアシンボル（例: USDJPY, EURUSD）、6-10文字
- `bid` (float): Bid価格（売値）、正の値
- `ask` (float): Ask価格（買値）、正の値
- `volume` (float): ティックボリューム、デフォルト0.0

**メソッド**:
- `ensure_float32(v)`: Float32型への変換を保証
- `validate_spread(v, info)`: スプレッドの妥当性を検証（ask > bid）
- `validate_symbol(v)`: シンボルを大文字に正規化

**プロパティ**:
- `spread`: スプレッドを計算（ask - bid）
- `mid_price`: 仲値（中間価格）を計算（(bid + ask) / 2）
- `to_float32_dict()`: Float32型で数値フィールドを含む辞書を返す

#### OHLC
**目的**: 為替レートのOHLC（四本値）データを表現するモデル

**属性**:
- `timestamp` (datetime): ローソク足の開始時刻（UTC）
- `symbol` (str): 通貨ペアシンボル（例: USDJPY, EURUSD）、6-10文字
- `timeframe` (TimeFrame): 時間足
- `open` (float): 開値、正の値
- `high` (float): 高値、正の値
- `low` (float): 安値、正の値
- `close` (float): 終値、正の値
- `volume` (float): 取引量、デフォルト0.0

**メソッド**:
- `ensure_float32(v)`: Float32型への変換を保証
- `validate_high(v, info)`: 高値が他の価格以上であることを検証
- `validate_low(v, info)`: 安値が他の価格以下であることを検証
- `validate_close(v, info)`: 終値が高値以下、安値以上であることを検証
- `validate_symbol(v)`: シンボルを大文字に正規化

**プロパティ**:
- `range`: 価格レンジ（高値-安値）を計算
- `is_bullish`: 陽線（bullish）かどうかを判定
- `is_bearish`: 陰線（bearish）かどうかを判定
- `body_size`: 実体サイズ（|終値-開値|）を計算
- `upper_shadow`: 上髭の長さを計算
- `lower_shadow`: 下髭の長さを計算
- `to_float32_dict()`: Float32型で数値フィールドを含む辞書を返す

#### Prediction
**目的**: 機械学習モデルによる予測結果を表現するモデル

**属性**:
- `timestamp` (datetime): 予測生成時刻（UTC）
- `symbol` (str): 通貨ペアシンボル
- `prediction_type` (PredictionType): 予測の種類
- `value` (float): 予測値
- `confidence` (float): 信頼度（0.0-1.0）
- `lower_bound` (Optional[float]): 予測区間の下限
- `upper_bound` (Optional[float]): 予測区間の上限
- `metadata` (Optional[dict]): 追加のメタデータ

**メソッド**:
- `ensure_float32(v)`: Float32型への変換を保証
- `validate_confidence(v)`: 信頼度が0-1の範囲内であることを検証
- `validate_bounds(v, info)`: 予測区間の妥当性を検証
- `validate_symbol(v)`: シンボルを大文字に正規化

**プロパティ**:
- `confidence_interval`: 信頼区間の幅を計算
- `to_float32_dict()`: Float32型で数値フィールドを含む辞書を返す

#### Alert
**目的**: 取引アラートを表現するモデル

**属性**:
- `timestamp` (datetime): アラート生成時刻（UTC）
- `symbol` (str): 通貨ペアシンボル
- `alert_type` (AlertType): アラートの種類
- `severity` (AlertSeverity): アラートの重要度
- `message` (str): アラートメッセージ（最大500文字）
- `value` (Optional[float]): 関連する数値
- `metadata` (Optional[dict]): 追加のメタデータ

**メソッド**:
- `ensure_float32(v)`: Float32型への変換を保証（valueフィールド）
- `validate_symbol(v)`: シンボルを大文字に正規化

**プロパティ**:
- `is_critical`: 重要度がCRITICALかどうかを判定
- `to_dict()`: 辞書形式に変換（valueはFloat32）

## 使用例
```python
from datetime import datetime
from src.common.models import Tick, OHLC, TimeFrame

# Tickデータの作成
tick = Tick(
    timestamp=datetime.utcnow(),
    symbol="USDJPY",
    bid=150.123,
    ask=150.125,
    volume=1000.0
)

# OHLCデータの作成
ohlc = OHLC(
    timestamp=datetime.utcnow(),
    symbol="USDJPY",
    timeframe=TimeFrame.H1,
    open=150.100,
    high=150.500,
    low=149.900,
    close=150.300,
    volume=10000.0
)

# スプレッドと仲値の計算
spread = tick.spread
mid_price = tick.mid_price

# ローソク足の判定
is_bullish = ohlc.is_bullish
body_size = ohlc.body_size
```

## 注意事項
- **Float32制約**: すべての数値フィールドは自動的にnp.float32に変換され、メモリ効率を最適化
- **バリデーション**: Pydanticのバリデーション機能により、データの整合性が自動的に検証される
- **スプレッド検証**: Tickモデルではask > bidが保証される
- **OHLC価格検証**: OHLCモデルでは high >= {open, close} >= low の関係が保証される
- **シンボル正規化**: すべてのシンボルは大文字に自動変換される