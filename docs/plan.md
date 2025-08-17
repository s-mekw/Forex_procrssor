# タスク2: 共通データモデルとインターフェース定義 - 実装計画

## 📋 実装ステップ

### Step 1: Pydanticモデルの定義 - Tickモデル ✅
- ファイル: src/common/models.py
- 作業: Tickデータモデルの作成（timestamp、symbol、bid、ask、volume）
- Float32型の使用とバリデーション設定
- 完了: [x]
- コミット: f52cf6f

### Step 2: Pydanticモデルの定義 - OHLCモデル ✅
- ファイル: src/common/models.py
- 作業: OHLCデータモデルの作成（timestamp、symbol、open、high、low、close、volume）
- 時間足（timeframe）フィールドの追加（Enum: M1, M5, M15, H1, H4, D1等）
- Float32型制約の適用（Step 1と同様のパターン）
- OHLC価格の論理的整合性検証の実装
- プロパティメソッド: range（高値-安値）、is_bullish（陽線判定）、body_size（実体サイズ）
- 完了: [x]

### Step 3: Pydanticモデルの定義 - Prediction/Alertモデル
- ファイル: src/common/models.py
- 作業: Predictionモデル（予測値、信頼区間、タイムスタンプ）とAlertモデル（アラートタイプ、閾値、メッセージ）の作成
- 完了: [ ]

### Step 4: 基底インターフェースの定義
- ファイル: src/common/interfaces.py
- 作業: DataFetcher、DataProcessor、StorageHandler、Predictorの抽象基底クラス作成
- プロトコルクラスとABCの使用
- 完了: [ ]

### Step 5: 設定管理システムの実装
- ファイル: src/common/config.py
- 作業: ConfigManagerクラスの実装（環境変数とTOMLファイルの読み込み）
- pydantic-settingsを使用した設定バリデーション
- 完了: [ ]

### Step 6: ユニットテストの作成 - モデル
- ファイル: tests/unit/test_models.py
- 作業: 各Pydanticモデルのバリデーションテスト、エッジケースのテスト
- 完了: [ ]

### Step 7: ユニットテストの作成 - インターフェース
- ファイル: tests/unit/test_interfaces.py
- 作業: 基底クラスの実装テスト、継承確認テスト
- 完了: [ ]

### Step 8: ユニットテストの作成 - 設定管理
- ファイル: tests/unit/test_config.py
- 作業: 環境変数読み込みテスト、TOMLファイル読み込みテスト、設定の優先順位テスト
- 完了: [ ]

## 📊 進捗状況
- 完了ステップ: 2/8
- 進行中ステップ: Step 3（Prediction/Alertモデル）
- 進捗率: 25%
- Step 1成果: テストカバレッジ95.65%達成
- Step 2成果: テストカバレッジ82.14%達成

## 🔍 実装詳細

### モデル設計の詳細

#### Step 1: Tickモデル（実装済み✅）
```python
class Tick(BaseModel):
    timestamp: datetime
    symbol: str
    bid: float  # Float32に制約
    ask: float  # Float32に制約
    volume: float  # Float32に制約
    
    @field_validator('bid', 'ask', 'volume', mode='before')
    def ensure_float32(cls, v: float) -> float:
        """Float32型への変換を保証"""
        return float(np.float32(v))
```

#### Step 2: OHLCモデル（実装済み✅）
```python
from enum import Enum

class TimeFrame(str, Enum):
    """時間足の定義"""
    M1 = "M1"   # 1分足
    M5 = "M5"   # 5分足
    M15 = "M15" # 15分足
    H1 = "H1"   # 1時間足
    H4 = "H4"   # 4時間足
    D1 = "D1"   # 日足

class OHLC(BaseModel):
    timestamp: datetime
    symbol: str
    timeframe: TimeFrame
    open: float   # Float32制約
    high: float   # Float32制約  
    low: float    # Float32制約
    close: float  # Float32制約
    volume: float # Float32制約
    
    @field_validator('high')
    def validate_high(cls, v, info):
        """高値が開値・終値・安値以上であることを検証"""
        # high >= low, high >= open, high >= close
    
    @property
    def range(self) -> float:
        """価格レンジ（高値-安値）"""
        return float(np.float32(self.high - self.low))
```

### インターフェース設計の詳細
```python
from abc import ABC, abstractmethod
from typing import Protocol

class DataFetcher(ABC):
    @abstractmethod
    async def fetch_data(self, symbol: str) -> pl.DataFrame:
        pass
```

### 設定管理の詳細
```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    # 環境変数から読み込み
    mt5_login: int
    mt5_password: str
    
    # TOMLファイルから読み込み
    class Config:
        env_file = ".env"
        toml_file = "settings.toml"
```

## 🎯 次のアクション
1. src/common/ディレクトリの作成
2. 必要な依存パッケージのインストール（pydantic、pydantic-settings）
3. Step 1から順番に実装を開始