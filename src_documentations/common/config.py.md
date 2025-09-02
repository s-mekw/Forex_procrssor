# config.py

## 概要
Forex Processorシステムの設定管理モジュール。環境変数とTOMLファイルから設定を読み込み、Pydanticベースの型安全な設定管理を提供します。シングルトンパターンにより、アプリケーション全体で一貫した設定を保証します。

## 依存関係
- 外部ライブラリ: pydantic, pydantic-settings, numpy, pathlib, toml
- 内部モジュール: なし（独立した設定管理）

## 主要コンポーネント

### クラス

#### BaseConfig
**目的**: 基本設定クラス - 環境変数とTOMLファイルから設定を読み込み

**属性**:
- **アプリケーション設定**:
  - `app_name` (str): アプリケーション名、デフォルト "forex_processor"
  - `app_version` (str): アプリケーションバージョン、デフォルト "0.1.0"
  - `debug` (bool): デバッグモード、デフォルト False
  - `log_level` (str): ログレベル、デフォルト "INFO"

- **MT5接続設定**:
  - `mt5_login` (Optional[int]): MT5ログインID
  - `mt5_password` (Optional[SecretStr]): MT5パスワード（秘匿）
  - `mt5_server` (str): MT5サーバー名、デフォルト "MetaQuotes-Demo"
  - `mt5_timeout` (int): MT5タイムアウト（ミリ秒）、デフォルト 60000
  - `mt5_path` (Optional[str]): MT5実行ファイルのパス

- **データベース設定**:
  - `db_host` (str): データベースホスト、デフォルト "localhost"
  - `db_port` (int): データベースポート、デフォルト 5432
  - `db_name` (str): データベース名、デフォルト "forex_db"
  - `db_user` (Optional[str]): データベースユーザー
  - `db_password` (Optional[SecretStr]): データベースパスワード（秘匿）

- **ストレージ設定**:
  - `data_dir` (Path): データディレクトリ、デフォルト "data"
  - `cache_dir` (Path): キャッシュディレクトリ、デフォルト ".cache"
  - `model_dir` (Path): モデルディレクトリ、デフォルト "models"

- **データ処理設定**:
  - `batch_size` (int): バッチサイズ、デフォルト 1000（1-100000）
  - `max_workers` (int): 最大ワーカー数、デフォルト 4（1-32）
  - `memory_limit_gb` (float): メモリ制限（GB）、デフォルト 8.0（0-1024）

- **モデル設定**:
  - `model_type` (str): モデルタイプ、デフォルト "lstm"
  - `prediction_horizon` (int): 予測ホライズン（時間単位）、デフォルト 24（1-720）
  - `confidence_threshold` (float): 信頼度閾値、デフォルト 0.7（0.0-1.0）

**メソッド**:
- `validate_log_level(v)`: ログレベルの検証（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- `validate_model_type(v)`: モデルタイプの検証（lstm, gru, transformer, cnn, ensemble）
- `ensure_float32(v)`: Float32精度での保存を保証
- `ensure_float32_threshold(v)`: Float32精度での閾値保存を保証
- `ensure_path(v)`: Pathオブジェクトへの変換を保証
- `get_db_url()`: データベース接続URLを生成
- `ensure_directories()`: 設定されたディレクトリを作成
- `validate_mt5_settings()`: MT5接続設定の妥当性を検証
- `validate_db_settings()`: データベース接続設定の妥当性を検証

#### ConfigManager
**目的**: 設定管理のシングルトンクラス

**属性**:
- `_instance` (Optional[ConfigManager]): シングルトンインスタンス
- `_config` (Optional[BaseConfig]): 設定インスタンス
- `_config_file` (Optional[Path]): 設定ファイルパス

**メソッド**:
- `__new__(cls)`: シングルトンインスタンスの生成
- `load_config(config_file, reload)`: 設定をロード
- `get_config()`: 現在の設定を取得
- `reload_config()`: 設定を再読み込み
- `save_config(config_file)`: 設定をファイルに保存
- `update_config(**kwargs)`: 設定を部分的に更新
- `get_all_configs()`: すべての設定を辞書形式で取得

### 関数

#### get_config
**目的**: 設定を取得するヘルパー関数

**入力**:
- なし

**出力**:
- BaseConfig: 現在の設定インスタンス

**処理フロー**:
1. ConfigManagerのシングルトンインスタンスを取得
2. 設定がロードされていない場合はロード
3. 設定インスタンスを返す

#### load_config
**目的**: 設定をロードするヘルパー関数

**入力**:
- `config_file` (Optional[Union[str, Path]]): 設定ファイルパス

**出力**:
- BaseConfig: ロードされた設定インスタンス

**処理フロー**:
1. ConfigManagerのシングルトンインスタンスを取得
2. 指定されたファイルから設定をロード
3. 設定インスタンスを返す

## 使用例
```python
from src.common.config import get_config, load_config

# デフォルト設定を取得
config = get_config()

# 特定の設定ファイルから読み込み
config = load_config("config/production.toml")

# 設定値へのアクセス
print(f"App Name: {config.app_name}")
print(f"Debug Mode: {config.debug}")
print(f"MT5 Server: {config.mt5_server}")

# データベースURLの生成
db_url = config.get_db_url()

# ディレクトリの作成
config.ensure_directories()

# 設定の検証
if config.validate_mt5_settings():
    print("MT5設定は有効です")

# 設定の更新（ConfigManager経由）
from src.common.config import config_manager
config_manager.update_config(debug=True, log_level="DEBUG")

# 設定の保存
config_manager.save_config("config/custom.toml")
```

## 注意事項
- **環境変数プレフィックス**: すべての環境変数は`FOREX_`プレフィックスを使用
- **設定ファイル**: デフォルトで`.env`ファイルとTOMLファイルの両方をサポート
- **秘匿情報**: パスワードはSecretStrとして管理され、自動的にマスクされる
- **Float32制約**: memory_limit_gbとconfidence_thresholdは自動的にFloat32に変換
- **シングルトン**: ConfigManagerにより、アプリケーション全体で一貫した設定を保証
- **バリデーション**: Pydanticによる型チェックと値の範囲検証が自動実行
- **ディレクトリ作成**: ensure_directories()で必要なディレクトリを自動作成