"""設定管理モジュール.

環境変数とTOMLファイルから階層的に設定を読み込み、
アプリケーション全体で使用する設定を一元管理する。

優先順位（高い順）:
1. 環境変数
2. .env.local
3. .env
4. config.toml
5. デフォルト値
"""

from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os
if sys.version_info >= (3, 11):
    import tomllib as tomli
else:
    import tomli
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import numpy as np


class BaseConfig(BaseSettings):
    """基本設定クラス - 環境変数とTOMLファイルから読み込み.
    
    Attributes:
        app_name: アプリケーション名
        app_version: アプリケーションバージョン
        debug: デバッグモード
        log_level: ログレベル
        
        mt5_login: MT5ログインID
        mt5_password: MT5パスワード（秘匿）
        mt5_server: MT5サーバー名
        mt5_timeout: MT5タイムアウト（ミリ秒）
        
        db_host: データベースホスト
        db_port: データベースポート
        db_name: データベース名
        db_user: データベースユーザー
        db_password: データベースパスワード（秘匿）
        
        data_dir: データディレクトリ
        cache_dir: キャッシュディレクトリ
        model_dir: モデルディレクトリ
        
        batch_size: バッチサイズ
        max_workers: 最大ワーカー数
        memory_limit_gb: メモリ制限（GB）
        
        model_type: モデルタイプ
        prediction_horizon: 予測ホライズン（時間単位）
        confidence_threshold: 信頼度閾値
    """
    
    # アプリケーション設定
    app_name: str = Field(default="forex_processor", description="アプリケーション名")
    app_version: str = Field(default="0.1.0", description="アプリケーションバージョン")
    debug: bool = Field(default=False, description="デバッグモード")
    log_level: str = Field(default="INFO", description="ログレベル")
    
    # MT5接続設定
    mt5_login: Optional[int] = Field(default=None, description="MT5ログインID")
    mt5_password: Optional[SecretStr] = Field(default=None, description="MT5パスワード")
    mt5_server: str = Field(default="MetaQuotes-Demo", description="MT5サーバー名")
    mt5_timeout: int = Field(default=60000, gt=0, le=300000, description="MT5タイムアウト（ミリ秒）")
    
    # データベース設定
    db_host: str = Field(default="localhost", description="データベースホスト")
    db_port: int = Field(default=5432, gt=0, le=65535, description="データベースポート")
    db_name: str = Field(default="forex_db", description="データベース名")
    db_user: Optional[str] = Field(default=None, description="データベースユーザー")
    db_password: Optional[SecretStr] = Field(default=None, description="データベースパスワード")
    
    # ストレージ設定
    data_dir: Path = Field(default=Path("data"), description="データディレクトリ")
    cache_dir: Path = Field(default=Path(".cache"), description="キャッシュディレクトリ")
    model_dir: Path = Field(default=Path("models"), description="モデルディレクトリ")
    
    # データ処理設定
    batch_size: int = Field(default=1000, gt=0, le=100000, description="バッチサイズ")
    max_workers: int = Field(default=4, gt=0, le=32, description="最大ワーカー数")
    memory_limit_gb: float = Field(default=8.0, gt=0.0, le=1024.0, description="メモリ制限（GB）")
    
    # モデル設定
    model_type: str = Field(default="lstm", description="モデルタイプ")
    prediction_horizon: int = Field(default=24, gt=0, le=720, description="予測ホライズン（時間単位）")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="信頼度閾値")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="FOREX_",  # 環境変数プレフィックス
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """ログレベルの検証."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """モデルタイプの検証."""
        valid_types = ["lstm", "gru", "transformer", "cnn", "ensemble"]
        v_lower = v.lower()
        if v_lower not in valid_types:
            raise ValueError(f"Invalid model type: {v}. Must be one of {valid_types}")
        return v_lower
    
    @field_validator('memory_limit_gb')
    @classmethod
    def ensure_float32(cls, v: float) -> float:
        """Float32精度での保存を保証."""
        return float(np.float32(v))
    
    @field_validator('confidence_threshold')
    @classmethod
    def ensure_float32_threshold(cls, v: float) -> float:
        """Float32精度での保存を保証."""
        return float(np.float32(v))
    
    @field_validator('data_dir', 'cache_dir', 'model_dir')
    @classmethod
    def ensure_path(cls, v: Path) -> Path:
        """Pathオブジェクトへの変換を保証."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    def get_db_url(self) -> str:
        """データベース接続URLを生成.
        
        Returns:
            PostgreSQL接続URL
        """
        if self.db_user:
            user_part = self.db_user
            if self.db_password:
                pass_part = f":{self.db_password.get_secret_value()}"
            else:
                pass_part = ""
            auth = f"{user_part}{pass_part}@"
        else:
            auth = ""
        return f"postgresql://{auth}{self.db_host}:{self.db_port}/{self.db_name}"
    
    def ensure_directories(self) -> None:
        """設定されたディレクトリを作成."""
        for dir_path in [self.data_dir, self.cache_dir, self.model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_mt5_settings(self) -> bool:
        """MT5接続設定の妥当性を検証.
        
        Returns:
            設定が有効な場合True
        """
        return all([
            self.mt5_login is not None,
            self.mt5_password is not None,
            self.mt5_server,
            self.mt5_timeout > 0
        ])
    
    def validate_db_settings(self) -> bool:
        """データベース接続設定の妥当性を検証.
        
        Returns:
            設定が有効な場合True
        """
        return all([
            self.db_host,
            self.db_port > 0,
            self.db_name,
        ])


class ConfigManager:
    """設定管理クラス - シングルトンパターン.
    
    アプリケーション全体で一つのインスタンスのみを保持し、
    設定の一元管理を実現する。
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[BaseConfig] = None
    _toml_data: Optional[Dict[str, Any]] = None
    
    def __new__(cls) -> 'ConfigManager':
        """シングルトンインスタンスの生成."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self,
                   env_file: Optional[Path] = None,
                   toml_file: Optional[Path] = None) -> BaseConfig:
        """設定を読み込み.
        
        TOMLファイルから設定を読み込み、環境変数で上書きする。
        
        Args:
            env_file: 環境変数ファイルのパス
            toml_file: TOMLファイルのパス
            
        Returns:
            読み込まれた設定
            
        Raises:
            FileNotFoundError: TOMLファイルが存在しない場合
            tomli.TOMLDecodeError: TOMLファイルの形式が不正な場合
        """
        # TOMLファイルの読み込み
        if toml_file and toml_file.exists():
            with open(toml_file, "rb") as f:
                self._toml_data = tomli.load(f)
        else:
            self._toml_data = {}
        
        # 環境変数ファイルの指定
        env_files = []
        if env_file and env_file.exists():
            env_files.append(env_file)
        
        # .env.localを優先的に読み込み
        local_env = Path(".env.local")
        if local_env.exists():
            env_files.append(local_env)
        
        # デフォルトの.envファイル
        default_env = Path(".env")
        if default_env.exists():
            env_files.append(default_env)
        
        # 環境変数ファイルを手動で読み込み（優先順位を制御するため）
        # python-dotenvを使用して環境変数を設定
        if env_file and env_file.exists():
            # 指定された環境変数ファイルを優先
            load_dotenv(env_file, override=True)
        elif local_env.exists():
            # .env.localファイル
            load_dotenv(local_env, override=True)
        elif default_env.exists():
            # デフォルトの.envファイル
            load_dotenv(default_env, override=True)
        
        # BaseConfigのインスタンス化
        # 環境変数を優先させるため、TOMLデータは環境変数にない項目のみ適用
        # pydantic-settingsは環境変数を自動的に読み込む
        self._config = BaseConfig()
        
        # TOMLデータから環境変数にない項目を設定
        # 環境変数で設定されていない項目のみTOMLから補完
        if self._toml_data:
            current_data = self._config.model_dump()
            for key, value in self._toml_data.items():
                # 環境変数が設定されていない項目のみTOMLから設定
                env_key = f"FOREX_{key.upper()}"
                if env_key not in os.environ and hasattr(self._config, key):
                    setattr(self._config, key, value)
        
        # ディレクトリの作成
        self._config.ensure_directories()
        
        return self._config
    
    def get_config(self) -> BaseConfig:
        """現在の設定を取得.
        
        Returns:
            現在の設定
            
        Raises:
            RuntimeError: 設定が未読み込みの場合
        """
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        return self._config
    
    def reload_config(self) -> BaseConfig:
        """設定を再読み込み.
        
        Returns:
            再読み込みされた設定
        """
        self._config = None
        return self.load_config()
    
    def update_config(self, **kwargs) -> BaseConfig:
        """設定を部分的に更新.
        
        Args:
            **kwargs: 更新する設定項目
            
        Returns:
            更新された設定
            
        Raises:
            RuntimeError: 設定が未読み込みの場合
        """
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        
        # 現在の設定をダンプ
        current_data = self._config.model_dump()
        
        # 更新内容をマージ
        current_data.update(kwargs)
        
        # 新しいインスタンスを作成
        self._config = BaseConfig(**current_data)
        
        return self._config
    
    def validate_connections(self) -> Dict[str, bool]:
        """全ての接続設定を検証.
        
        Returns:
            接続名と検証結果のマッピング
        """
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        
        return {
            "mt5": self._config.validate_mt5_settings(),
            "database": self._config.validate_db_settings(),
        }
    
    def export_config(self, path: Path, include_secrets: bool = False) -> None:
        """設定をファイルにエクスポート.
        
        Args:
            path: エクスポート先のファイルパス
            include_secrets: 秘密情報を含めるかどうか
        """
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        
        config_data = self._config.model_dump()
        
        # 秘密情報を除外
        if not include_secrets:
            for key in ["mt5_password", "db_password"]:
                if key in config_data and config_data[key]:
                    config_data[key] = "***HIDDEN***"
        
        # PathオブジェクトをStringに変換
        for key in ["data_dir", "cache_dir", "model_dir"]:
            if key in config_data and isinstance(config_data[key], Path):
                config_data[key] = str(config_data[key])
        
        # TOMLファイルとして保存
        if sys.version_info >= (3, 11):
            import tomllib as tomli
            with open(path, "w", encoding="utf-8") as f:
                import toml
                toml.dump(config_data, f)
        else:
            import tomli_w
            with open(path, "wb") as f:
                tomli_w.dump(config_data, f)
    
    @classmethod
    def reset_instance(cls) -> None:
        """シングルトンインスタンスをリセット（主にテスト用）."""
        cls._instance = None
        cls._config = None
        cls._toml_data = None


# グローバルインスタンス
config_manager = ConfigManager()


def get_config() -> BaseConfig:
    """設定を取得するヘルパー関数.
    
    Returns:
        現在の設定
        
    Raises:
        RuntimeError: 設定が未読み込みの場合
    """
    return config_manager.get_config()


def load_config(env_file: Optional[Path] = None,
                toml_file: Optional[Path] = None) -> BaseConfig:
    """設定を読み込むヘルパー関数.
    
    Args:
        env_file: 環境変数ファイルのパス
        toml_file: TOMLファイルのパス
        
    Returns:
        読み込まれた設定
    """
    return config_manager.load_config(env_file, toml_file)