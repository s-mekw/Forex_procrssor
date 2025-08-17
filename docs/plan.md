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

### Step 3: Pydanticモデルの定義 - Prediction/Alertモデル ✅
- ファイル: src/common/models.py
- 作業内容:
  1. PredictionType Enum（PRICE, DIRECTION, VOLATILITY）の定義
  2. Predictionモデルの実装:
     - symbol, predicted_at, target_timestamp（必須フィールド）
     - prediction_type, predicted_value（Float32制約）
     - confidence_score（0.0-1.0の範囲制約）
     - confidence_upper, confidence_lower（信頼区間、Float32制約）
     - 信頼区間の論理的整合性検証（upper >= lower）
     - プロパティ: confidence_range（信頼区間の幅）
  3. AlertType Enum（PRICE_THRESHOLD, PATTERN_DETECTED, RISK_WARNING）の定義
  4. AlertSeverity Enum（INFO, WARNING, CRITICAL）の定義
  5. Alertモデルの実装:
     - symbol, timestamp, alert_type, severity（必須フィールド）
     - threshold_value（Float32制約、optional）
     - current_value（Float32制約、optional）
     - message（アラート詳細メッセージ）
     - condition（トリガー条件の記録）
     - プロパティ: is_critical（重要度判定）, threshold_exceeded（閾値超過判定）
- テスト: tests/common/test_prediction_alert_models.py
- 目標カバレッジ: 80%以上
- 完了: [x]

### Step 4: 基底インターフェースの定義 ✅
- ファイル: src/common/interfaces.py
- 作業: DataFetcher、DataProcessor、StorageHandler、Predictorの抽象基底クラス作成
- プロトコルクラスとABCの使用
- 完了: [x]

### Step 5: 設定管理システムの実装
- ファイル: src/common/config.py
- 作業: 環境変数とTOMLファイルから設定を読み込む階層的設定管理システム
- 完了: [ ]

#### 詳細実装計画

##### 5.1 設定クラスの構造
```python
# 基本設定クラス（pydantic-settingsベース）
class BaseConfig(BaseSettings):
    """基本設定クラス - 環境変数とTOMLファイルから読み込み"""
    # アプリケーション設定
    app_name: str = "forex_processor"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # MT5接続設定
    mt5_login: Optional[int] = None
    mt5_password: Optional[SecretStr] = None
    mt5_server: str = "MetaQuotes-Demo"
    mt5_timeout: int = 60000
    
    # データベース設定
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "forex_db"
    db_user: Optional[str] = None
    db_password: Optional[SecretStr] = None
    
    # ストレージ設定
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    model_dir: Path = Path("models")
    
    # データ処理設定
    batch_size: int = 1000
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # モデル設定
    model_type: str = "lstm"
    prediction_horizon: int = 24  # 時間単位
    confidence_threshold: float = 0.7
```

##### 5.2 環境変数とTOMLの優先順位
1. 環境変数（最優先）
2. .env.local（ローカル開発用）
3. .env（デフォルト環境変数）
4. config.toml（プロジェクト設定）
5. デフォルト値（クラス定義）

##### 5.3 ConfigManagerクラスの実装
```python
class ConfigManager:
    """設定管理クラス - シングルトンパターン"""
    _instance: Optional['ConfigManager'] = None
    _config: Optional[BaseConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, 
                   env_file: Optional[Path] = None,
                   toml_file: Optional[Path] = None) -> BaseConfig:
        """設定を読み込み"""
        # TOMLファイルの読み込み
        # 環境変数の上書き
        # バリデーション実行
        # キャッシュに保存
    
    def get_config(self) -> BaseConfig:
        """現在の設定を取得"""
    
    def reload_config(self) -> BaseConfig:
        """設定を再読み込み"""
    
    def validate_mt5_connection(self) -> bool:
        """MT5接続設定の検証"""
    
    def validate_db_connection(self) -> bool:
        """DB接続設定の検証"""
```

##### 5.4 バリデーション実装
- 必須フィールドの存在確認
- 型変換と範囲チェック
- パスの存在確認と作成
- 接続テスト機能
- 設定の相互依存チェック

##### 5.5 テストケース設計
1. **基本機能テスト**
   - 環境変数からの読み込み
   - TOMLファイルからの読み込み
   - デフォルト値の適用
   - 優先順位の確認

2. **バリデーションテスト**
   - 必須フィールドの検証
   - 型変換のテスト
   - 範囲チェックのテスト
   - 無効な値のエラー処理

3. **統合テスト**
   - 複数ソースからの読み込み
   - 設定の再読み込み
   - シングルトンパターンの動作
   - 環境別設定の切り替え

4. **エッジケーステスト**
   - ファイルが存在しない場合
   - 権限エラー
   - 不正なフォーマット
   - 循環参照の検出

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
- 完了ステップ: 4/8
- 進行中ステップ: Step 5（設定管理システムの実装）
- 進捗率: 50%
- Step 1成果: テストカバレッジ95.65%達成
- Step 2成果: テストカバレッジ82.14%達成
- Step 3成果: 全体テストカバレッジ85.63%達成
- Step 4成果: 全体テストカバレッジ86.29%達成

### Step 4実装チェックリスト ✅
- [x] DataFetcherインターフェースの定義
- [x] DataProcessorインターフェースの定義
- [x] StorageHandlerインターフェースの定義
- [x] Predictorインターフェースの定義
- [x] プロトコルクラスの実装
- [x] ドキュメント文字列の追加
- [x] ユニットテストの作成（目標カバレッジ80%以上）

### Step 3実装チェックリスト ✅
- [x] PredictionType, AlertType, AlertSeverity Enumの定義
- [x] Predictionモデルの基本フィールド実装
- [x] PredictionモデルのFloat32制約とバリデーション
- [x] Predictionモデルのプロパティメソッド
- [x] Alertモデルの基本フィールド実装
- [x] AlertモデルのFloat32制約とバリデーション
- [x] Alertモデルのプロパティメソッド
- [x] ユニットテストの作成（目標カバレッジ80%以上）

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
### Step 3の実装手順
1. PredictionType, AlertType, AlertSeverity Enumの定義から開始
2. Predictionモデルの実装（Step 1, 2と同様のパターンで）
3. Alertモデルの実装（Float32制約を一貫して適用）
4. 各モデルのプロパティメソッドを追加
5. tests/common/test_prediction_alert_models.pyでテストを作成
6. カバレッジ80%以上を確認してからコミット

### 実装のポイント
- Step 1, 2で確立したFloat32制約パターンを踏襲
- 包括的なドキュメント文字列を各レベルで記載
- バリデーションロジックを充実させる
- プロパティメソッドで実用的な派生値を提供