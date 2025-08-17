## タスク2: 共通データモデルとインターフェース定義 設計・実装計画

### 目的
Forexデータ処理基盤における共通データモデルと抽象インターフェースを定義し、システム全体の型安全性・一貫性・再利用性を向上させる。特に数値はデフォルトで Float32 を採用し、メモリ効率と計算速度を両立する。

### スコープ
- Pydantic v2 ベースの共通モデル作成：`Tick`、`OHLC`、`Prediction`、`Alert`
- 基底インターフェース（抽象クラス）定義：`DataFetcher`、`DataProcessor`、`StorageHandler`、`Predictor`
- 設定管理クラスの実装：環境変数と TOML の両対応

### 参照（詳細設計・ガイド）
- 設計全体: `../.kiro/specs/Forex_procrssor/design.md`
- 要件: `../.kiro/specs/Forex_procrssor/requirements.md`
- 実装タスク: `../.kiro/specs/Forex_procrssor/tasks.md`
- スペック概要: `../.kiro/specs/Forex_procrssor/spec.json`
- 技術方針: `../.kiro/steering/tech.md`
- 構造/モジュール方針: `../.kiro/steering/structure.md`
- 開発ガイドライン(Python): `../.kiro/steering/Python_Development_Guidelines.md`
- プロダクト方針: `../.kiro/steering/product.md`

### 成果物（ファイルと役割）
- `src/common/models.py`：共通データモデル（Pydantic）
- `src/common/interfaces.py`：抽象基底インターフェース（abc）
- `src/common/config.py`：設定管理（`pydantic-settings`、TOML/ENV）

## データモデル設計（Pydantic v2）

### 共通規約
- 浮動小数点は原則 `Float32`（NumPy/Polarsと整合）。入力は `float` を受け、内部で Float32 に正規化
- タイムスタンプは UTC の `datetime`（tz aware）。外部I/Oは ISO 8601
- シンボルは大文字の通貨ペア（例: `EURUSD`）
- `volume` は非負、`high >= low`、OHLC は `open/close` が [low, high] 範囲内

### モデル定義（仕様）
```python
# models.py（仕様イメージ）
from datetime import datetime
from typing import Optional, Literal, Dict, Any

class Tick:
    symbol: str
    timestamp: datetime  # UTC, tz-aware
    bid: float  # Float32化
    ask: float  # Float32化
    last: Optional[float] = None  # Float32化
    volume: Optional[float] = 0.0  # 非負、Float32化

class OHLC:
    symbol: str
    timeframe: Literal["M1","M5","M15","M30","H1","H4","D1"]
    open: float  # Float32化
    high: float  # Float32化
    low: float   # Float32化
    close: float # Float32化
    volume: float  # 非負、Float32化
    start_time: datetime  # バー開始（UTC）
    end_time: datetime    # バー終了（UTC, start < end）

class Prediction:
    symbol: str
    generated_at: datetime  # 予測生成時刻（UTC）
    horizon_minutes: int  # 予測ホライズン（分）
    value: float  # 予測値（Float32）
    ci50_low: Optional[float] = None
    ci50_high: Optional[float] = None
    ci80_low: Optional[float] = None
    ci80_high: Optional[float] = None
    ci95_low: Optional[float] = None
    ci95_high: Optional[float] = None
    model_version: Optional[str] = None

class Alert:
    type: Literal["PRICE","TECHNICAL","PREDICTION","SYSTEM"]
    timestamp: datetime
    symbol: Optional[str] = None
    severity: Literal["INFO","WARN","ERROR","CRITICAL"]
    message: str
    metadata: Optional[Dict[str, Any]] = None
```

### バリデーションポリシー
- `volume >= 0`
- `low <= open <= high` かつ `low <= close <= high`
- `end_time > start_time`
- `symbol` は英大文字の英数（例: 正規表現 `^[A-Z0-9_]+$`）
- 入力の `float` は Float32 にダウンクォート（内部正規化）

## インターフェース設計（抽象基底）

### 目的
実装差し替え可能な疎結合構成を実現（MT5実装/モック、ローカル/クラウドストレージ、各種予測モデル）。

### シグネチャ（仕様）
```python
# interfaces.py（仕様イメージ）
from abc import ABC, abstractmethod
from collections.abc import Iterable, AsyncIterable
from typing import Sequence, Optional

class DataFetcher(ABC):
    @abstractmethod
    def fetch_ohlc(self, symbol: str, timeframe: str, start, end) -> Iterable["OHLC"]: ...

    @abstractmethod
    def stream_ticks(self, symbols: Sequence[str]) -> AsyncIterable["Tick"]: ...

class DataProcessor(ABC):
    @abstractmethod
    def to_ohlc(self, ticks: Iterable["Tick"], timeframe: str) -> Iterable["OHLC"]: ...

    @abstractmethod
    def enrich(self, ohlc: Iterable["OHLC"]) -> Iterable["OHLC"]: ...  # 例: 指標付与

class StorageHandler(ABC):
    @abstractmethod
    def write_ohlc(self, data: Iterable["OHLC"]) -> None: ...

    @abstractmethod
    def read_ohlc(self, symbol: str, timeframe: str, start, end) -> Iterable["OHLC"]: ...

    @abstractmethod
    def write_predictions(self, data: Iterable["Prediction"]) -> None: ...

class Predictor(ABC):
    @abstractmethod
    def predict(self, data: Iterable["OHLC"], horizon_minutes: int) -> Iterable["Prediction"]: ...

    @property
    @abstractmethod
    def model_version(self) -> str: ...
```

### 非機能要件（関係）
- スループットとメモリ使用のバランス（Float32統一）
- 同期/非同期 API の両立（取得は非同期ストリーム、加工/保存は同期イテレータを基本）
- 実装クラスは `../.kiro/specs/Forex_procrssor/design.md` の責務分割に準拠

## 設定管理（`config.py`）

### 方針
- `pydantic-settings` を使用し、ENV と TOML（例: `src/config/settings.toml`）をマージ
- 環境変数は `FP_` プレフィックス（例: `FP_ENV`, `FP_LOG_LEVEL`, `FP_TIMEZONE`）
- ローディング優先度: ENV > TOML > デフォルト

### 例（仕様）
```python
# config.py（仕様イメージ）
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FP_", env_file=".env", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"
    timezone: str = "UTC"
    float_dtype: str = "float32"

    # MT5/DB などの接続情報は今後拡張
```

### TOML 例
```toml
[app]
env = "dev"
log_level = "INFO"
timezone = "UTC"
```

## テスト計画（タスク2範囲）

### ユニットテスト
- `models.py`
  - バリデーション（範囲、型、UTC/tz-aware）
  - Float32 正規化（誤差許容内での比較）
- `interfaces.py`
  - 抽象メソッドの存在確認、未実装呼び出しで `TypeError/NotImplementedError`
- `config.py`
  - ENV/TOML/デフォルトの優先順序
  - 無効値の拒否とデフォルトへのフォールバック

### カバレッジ
- 80%以上（`pyproject.toml` の設定に準拠）

## 受け入れ基準（DoD）
- 上記3ファイルが作成され、`pytest` がグリーン
- モデルに対し代表的入力のシリアライズ/デシリアライズ（dict/JSON）が成功
- Float32 統一ポリシーがテストで担保
- 参照リンクが全て有効で、設計方針が `../.kiro/` の内容と整合

## 実装指針
- 例外はドメイン例外でラップし、上位へ伝播（ロギングは `src/common/logger.py` へ将来委譲）
- 型注釈は公開APIに明示、内部は過剰注釈を避ける
- 変換コスト最小化のため、Polars/Numpy の型に合わせて早期に Float32 化

## リスクと緩和
- Float32 による精度低下：閾値比較は許容誤差を導入、集計系は必要に応じ Float64 層で再計算
- タイムゾーン不整合：入出力を UTC に統一し、境界はテストで検証

## 今後の拡張（他タスク連携）
- テクニカル指標（`src/data_processing/indicators.py`）へ `OHLC` を入力とする拡張
- 予測パイプライン（`src/patchTST_model/`）で `Prediction` を標準入出力とする
- ストレージ層（`src/storage/`）で `StorageHandler` 実装を追加（Influx/Parquet）

## 参考リンク（この計画が依拠する設計）
- `../.kiro/specs/Forex_procrssor/design.md`
- `../.kiro/specs/Forex_procrssor/requirements.md`
- `../.kiro/specs/Forex_procrssor/tasks.md`
- `../.kiro/steering/structure.md`
- `../.kiro/steering/tech.md`
- `../.kiro/steering/Python_Development_Guidelines.md`
- `../.kiro/steering/product.md`


