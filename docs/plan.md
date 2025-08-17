# タスク2: 共通データモデルとインターフェース定義 - 実装計画

## 📋 実装ステップ

### Step 1: Pydanticモデルの定義 - Tickモデル
- ファイル: src/common/models.py
- 作業: Tickデータモデルの作成（timestamp、symbol、bid、ask、volume）
- Float32型の使用とバリデーション設定
- 完了: [x]

### Step 2: Pydanticモデルの定義 - OHLCモデル
- ファイル: src/common/models.py
- 作業: OHLCデータモデルの作成（timestamp、symbol、open、high、low、close、volume）
- 時間足（timeframe）フィールドの追加
- 完了: [ ]

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
- 完了ステップ: 1/8
- 進捗率: 12.5%

## 🔍 実装詳細

### モデル設計の詳細
```python
# Float32使用例
class Tick(BaseModel):
    timestamp: datetime
    symbol: str
    bid: float  # Pydanticでfloat32に制約
    ask: float
    volume: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
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