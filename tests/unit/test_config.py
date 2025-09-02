"""設定管理モジュールのユニットテスト."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from pydantic import ValidationError, SecretStr
import numpy as np

if sys.version_info >= (3, 11):
    import tomllib as tomli
else:
    import tomli

from src.common.config import BaseConfig, ConfigManager, get_config, load_config


class TestBaseConfig:
    """BaseConfigクラスのテスト."""
    
    def test_default_values(self):
        """デフォルト値の確認."""
        config = BaseConfig()
        
        # アプリケーション設定
        assert config.app_name == "forex_processor"
        assert config.app_version == "0.1.0"
        assert config.debug is False
        assert config.log_level == "INFO"
        
        # MT5設定
        assert config.mt5_login is None
        assert config.mt5_password is None
        assert config.mt5_server == "MetaQuotes-Demo"
        assert config.mt5_timeout == 60000
        assert config.mt5_path is None
        
        # DB設定
        assert config.db_host == "localhost"
        assert config.db_port == 5432
        assert config.db_name == "forex_db"
        assert config.db_user is None
        assert config.db_password is None
        
        # ストレージ設定
        assert config.data_dir == Path("data")
        assert config.cache_dir == Path(".cache")
        assert config.model_dir == Path("models")
        
        # データ処理設定
        assert config.batch_size == 1000
        assert config.max_workers == 4
        assert config.memory_limit_gb == 8.0
        
        # モデル設定
        assert config.model_type == "lstm"
        assert config.prediction_horizon == 24
        assert abs(config.confidence_threshold - 0.7) < 0.0001  # Float32精度
    
    def test_env_override(self):
        """環境変数による設定の上書き."""
        with patch.dict(os.environ, {
            "FOREX_APP_NAME": "test_app",
            "FOREX_DEBUG": "true",
            "FOREX_MT5_LOGIN": "12345",
            "FOREX_MT5_PATH": "C:\\Test\\MT5\\terminal64.exe",
            "FOREX_BATCH_SIZE": "5000",
        }):
            config = BaseConfig()
            assert config.app_name == "test_app"
            assert config.debug is True
            assert config.mt5_login == 12345
            assert config.mt5_path == "C:\\Test\\MT5\\terminal64.exe"
            assert config.batch_size == 5000
    
    def test_log_level_validation(self):
        """ログレベルのバリデーション."""
        # 有効なログレベル
        config = BaseConfig(log_level="debug")
        assert config.log_level == "DEBUG"
        
        config = BaseConfig(log_level="WARNING")
        assert config.log_level == "WARNING"
        
        # 無効なログレベル
        with pytest.raises(ValidationError) as exc_info:
            BaseConfig(log_level="INVALID")
        assert "Invalid log level" in str(exc_info.value)
    
    def test_model_type_validation(self):
        """モデルタイプのバリデーション."""
        # 有効なモデルタイプ
        config = BaseConfig(model_type="LSTM")
        assert config.model_type == "lstm"
        
        config = BaseConfig(model_type="transformer")
        assert config.model_type == "transformer"
        
        # 無効なモデルタイプ
        with pytest.raises(ValidationError) as exc_info:
            BaseConfig(model_type="invalid_model")
        assert "Invalid model type" in str(exc_info.value)
    
    def test_numeric_range_validation(self):
        """数値範囲のバリデーション."""
        # 有効な範囲
        config = BaseConfig(
            mt5_timeout=1000,
            db_port=3306,
            batch_size=500,
            max_workers=8,
            memory_limit_gb=16.0,
            prediction_horizon=48,
            confidence_threshold=0.5
        )
        assert config.mt5_timeout == 1000
        assert config.db_port == 3306
        assert config.batch_size == 500
        
        # 無効な範囲
        with pytest.raises(ValidationError):
            BaseConfig(mt5_timeout=0)  # must be > 0
        
        with pytest.raises(ValidationError):
            BaseConfig(db_port=70000)  # must be <= 65535
        
        with pytest.raises(ValidationError):
            BaseConfig(confidence_threshold=1.5)  # must be <= 1.0
    
    def test_float32_conversion(self):
        """Float32への変換."""
        config = BaseConfig(
            memory_limit_gb=16.123456789,
            confidence_threshold=0.123456789
        )
        
        # Float32精度での保存を確認
        assert config.memory_limit_gb == float(np.float32(16.123456789))
        assert config.confidence_threshold == float(np.float32(0.123456789))
    
    def test_path_conversion(self):
        """パスへの変換."""
        config = BaseConfig(
            data_dir="custom/data",
            cache_dir=Path("custom/cache"),
            model_dir="custom/models"
        )
        
        assert isinstance(config.data_dir, Path)
        assert config.data_dir == Path("custom/data")
        assert isinstance(config.cache_dir, Path)
        assert config.cache_dir == Path("custom/cache")
        assert isinstance(config.model_dir, Path)
    
    def test_secret_str(self):
        """SecretStrの処理."""
        config = BaseConfig(
            mt5_password="secret123",
            db_password="dbpass456"
        )
        
        assert isinstance(config.mt5_password, SecretStr)
        assert config.mt5_password.get_secret_value() == "secret123"
        assert isinstance(config.db_password, SecretStr)
        assert config.db_password.get_secret_value() == "dbpass456"
        
        # シリアライズ時に秘匿される
        dumped = config.model_dump()
        # pydantic v2ではSecretStrは秘匿されたまま
        assert isinstance(dumped["mt5_password"], SecretStr)
        assert dumped["mt5_password"].get_secret_value() == "secret123"
    
    def test_get_db_url(self):
        """データベースURL生成."""
        # 認証なし
        config = BaseConfig()
        assert config.get_db_url() == "postgresql://localhost:5432/forex_db"
        
        # ユーザーのみ
        config = BaseConfig(db_user="user")
        assert config.get_db_url() == "postgresql://user@localhost:5432/forex_db"
        
        # フル認証
        config = BaseConfig(
            db_user="user",
            db_password="pass",
            db_host="dbserver",
            db_port=5433
        )
        assert config.get_db_url() == "postgresql://user:pass@dbserver:5433/forex_db"
    
    def test_ensure_directories(self, tmp_path):
        """ディレクトリ作成."""
        base_dir = tmp_path / "test_config"
        config = BaseConfig(
            data_dir=base_dir / "data",
            cache_dir=base_dir / "cache",
            model_dir=base_dir / "models"
        )
        
        # ディレクトリが存在しない
        assert not config.data_dir.exists()
        assert not config.cache_dir.exists()
        assert not config.model_dir.exists()
        
        # ディレクトリを作成
        config.ensure_directories()
        
        # ディレクトリが作成された
        assert config.data_dir.exists()
        assert config.cache_dir.exists()
        assert config.model_dir.exists()
    
    def test_validate_mt5_settings(self):
        """MT5設定の検証."""
        # 不完全な設定
        config = BaseConfig()
        assert config.validate_mt5_settings() is False
        
        # 完全な設定
        config = BaseConfig(
            mt5_login=12345,
            mt5_password="password"
        )
        assert config.validate_mt5_settings() is True
    
    def test_validate_db_settings(self):
        """DB設定の検証."""
        # デフォルト設定でも有効
        config = BaseConfig()
        assert config.validate_db_settings() is True
        
        # カスタム設定
        config = BaseConfig(
            db_host="dbserver",
            db_port=3306,
            db_name="custom_db"
        )
        assert config.validate_db_settings() is True


class TestConfigManager:
    """ConfigManagerクラスのテスト."""
    
    def setup_method(self):
        """テストメソッドの前処理."""
        ConfigManager.reset_instance()
        # 環境変数をクリア
        for key in list(os.environ.keys()):
            if key.startswith("FOREX_"):
                del os.environ[key]
    
    def teardown_method(self):
        """テストメソッドの後処理."""
        ConfigManager.reset_instance()
        # 環境変数をクリア
        for key in list(os.environ.keys()):
            if key.startswith("FOREX_"):
                del os.environ[key]
    
    def test_singleton_pattern(self):
        """シングルトンパターンの確認."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2
    
    def test_load_config_default(self):
        """デフォルト設定の読み込み."""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert isinstance(config, BaseConfig)
        assert config.app_name == "forex_processor"
    
    def test_load_config_with_toml(self, tmp_path):
        """TOMLファイルからの読み込み."""
        toml_file = tmp_path / "config.toml"
        toml_content = """
app_name = "test_app"
debug = true
batch_size = 2000
model_type = "gru"
"""
        toml_file.write_text(toml_content)
        
        manager = ConfigManager()
        config = manager.load_config(toml_file=toml_file)
        
        assert config.app_name == "test_app"
        assert config.debug is True
        assert config.batch_size == 2000
        assert config.model_type == "gru"
    
    def test_load_config_with_env_file(self, tmp_path):
        """環境変数ファイルからの読み込み."""
        env_file = tmp_path / ".env"
        env_content = """
FOREX_APP_NAME=env_app
FOREX_DEBUG=true
FOREX_MT5_LOGIN=99999
"""
        env_file.write_text(env_content)
        
        manager = ConfigManager()
        config = manager.load_config(env_file=env_file)
        
        assert config.app_name == "env_app"
        assert config.debug is True
        assert config.mt5_login == 99999
    
    def test_get_config(self):
        """設定の取得."""
        manager = ConfigManager()
        
        # 未読み込み時はエラー
        with pytest.raises(RuntimeError) as exc_info:
            manager.get_config()
        assert "Config not loaded" in str(exc_info.value)
        
        # 読み込み後は取得可能
        loaded_config = manager.load_config()
        retrieved_config = manager.get_config()
        assert retrieved_config is loaded_config
    
    def test_reload_config(self):
        """設定の再読み込み."""
        manager = ConfigManager()
        
        # 初回読み込み
        config1 = manager.load_config()
        config1_id = id(config1)
        
        # 再読み込み
        config2 = manager.reload_config()
        config2_id = id(config2)
        
        # 新しいインスタンスが作成される
        assert config1_id != config2_id
        assert isinstance(config2, BaseConfig)
    
    def test_update_config(self):
        """設定の部分更新."""
        manager = ConfigManager()
        
        # 未読み込み時はエラー
        with pytest.raises(RuntimeError):
            manager.update_config(debug=True)
        
        # 読み込み後は更新可能
        manager.load_config()
        updated_config = manager.update_config(
            debug=True,
            batch_size=5000,
            model_type="transformer"
        )
        
        assert updated_config.debug is True
        assert updated_config.batch_size == 5000
        assert updated_config.model_type == "transformer"
        # 他の設定は保持される
        assert updated_config.app_name == "forex_processor"
    
    def test_validate_connections(self):
        """接続設定の検証."""
        manager = ConfigManager()
        
        # 未読み込み時はエラー
        with pytest.raises(RuntimeError):
            manager.validate_connections()
        
        # デフォルト設定での検証
        manager.load_config()
        validations = manager.validate_connections()
        
        assert "mt5" in validations
        assert "database" in validations
        assert validations["mt5"] is False  # MT5ログイン情報がない
        assert validations["database"] is True  # DB設定は有効
        
        # MT5設定を追加
        manager.update_config(mt5_login=12345, mt5_password="pass")
        validations = manager.validate_connections()
        assert validations["mt5"] is True
    
    def test_export_config(self, tmp_path):
        """設定のエクスポート."""
        manager = ConfigManager()
        
        # 未読み込み時はエラー
        export_path = tmp_path / "export.toml"
        with pytest.raises(RuntimeError):
            manager.export_config(export_path)
        
        # 設定を読み込んでエクスポート
        manager.load_config()
        manager.update_config(
            mt5_password="secret",
            db_password="dbsecret"
        )
        
        # 秘密情報を含めない
        export_path_no_secrets = tmp_path / "export_no_secrets.toml"
        manager.export_config(export_path_no_secrets, include_secrets=False)
        
        # ファイルの内容を確認
        content = export_path_no_secrets.read_text()
        assert "***HIDDEN***" in content
        assert "secret" not in content
        
        # 秘密情報を含める
        export_path_with_secrets = tmp_path / "export_with_secrets.toml"
        manager.export_config(export_path_with_secrets, include_secrets=True)
        
        # 秘密情報が含まれている（注：実際の値は取得できない）
        if sys.version_info >= (3, 11):
            with open(export_path_with_secrets, "rb") as f:
                exported_data = tomli.load(f)
        else:
            with open(export_path_with_secrets, "rb") as f:
                exported_data = tomli.load(f)
        # SecretStrは文字列として保存される
        assert exported_data["mt5_password"] is not None
    
    def test_reset_instance(self):
        """インスタンスのリセット."""
        manager1 = ConfigManager()
        manager1.load_config()
        
        ConfigManager.reset_instance()
        
        manager2 = ConfigManager()
        assert manager1 is not manager2
        
        # 新しいインスタンスは未読み込み状態
        with pytest.raises(RuntimeError):
            manager2.get_config()


class TestHelperFunctions:
    """ヘルパー関数のテスト."""
    
    def setup_method(self):
        """テストメソッドの前処理."""
        ConfigManager.reset_instance()
        # 環境変数をクリア
        for key in list(os.environ.keys()):
            if key.startswith("FOREX_"):
                del os.environ[key]
    
    def teardown_method(self):
        """テストメソッドの後処理."""
        ConfigManager.reset_instance()
        # 環境変数をクリア
        for key in list(os.environ.keys()):
            if key.startswith("FOREX_"):
                del os.environ[key]
    
    def test_get_config_helper(self):
        """get_config()ヘルパー関数."""
        # 未読み込み時はエラー
        with pytest.raises(RuntimeError):
            get_config()
        
        # 読み込み後は取得可能
        load_config()
        config = get_config()
        assert isinstance(config, BaseConfig)
    
    def test_load_config_helper(self, tmp_path):
        """load_config()ヘルパー関数."""
        # デフォルト読み込み
        config1 = load_config()
        assert isinstance(config1, BaseConfig)
        
        # TOMLファイル指定
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("app_name = 'helper_test'")
        
        config2 = load_config(toml_file=toml_file)
        assert config2.app_name == "helper_test"


class TestIntegration:
    """統合テスト."""
    
    def setup_method(self):
        """テストメソッドの前処理."""
        ConfigManager.reset_instance()
        # 環境変数をクリア
        for key in list(os.environ.keys()):
            if key.startswith("FOREX_"):
                del os.environ[key]
    
    def teardown_method(self):
        """テストメソッドの後処理."""
        ConfigManager.reset_instance()
        # 環境変数をクリア
        for key in list(os.environ.keys()):
            if key.startswith("FOREX_"):
                del os.environ[key]
    
    def test_priority_order(self, tmp_path):
        """設定の優先順位テスト."""
        # TOMLファイル
        toml_file = tmp_path / "config.toml"
        toml_content = """
app_name = "toml_app"
debug = false
batch_size = 1000
"""
        toml_file.write_text(toml_content)
        
        # 環境変数ファイル
        env_file = tmp_path / ".env"
        env_content = """
FOREX_APP_NAME=env_app
FOREX_DEBUG=true
"""
        env_file.write_text(env_content)
        
        # TOMLファイルと環境変数の両方を読み込み
        # 注: pydantic-settingsでは_env_fileパラメータで環境変数ファイルを指定
        # ここでは、TOMLファイルのデータがデフォルト値として渡される
        manager = ConfigManager()
        config = manager.load_config(env_file=env_file, toml_file=toml_file)
        
        # 環境変数ファイルから読み込まれた値が優先される
        assert config.app_name == "env_app"  # 環境変数が優先
        assert config.debug is True  # 環境変数が優先
        assert config.batch_size == 1000  # TOMLのみ
    
    def test_full_workflow(self, tmp_path):
        """完全なワークフローテスト."""
        # 初期設定
        manager = ConfigManager()
        config = manager.load_config()
        
        # ディレクトリ作成
        config.data_dir = tmp_path / "workflow_data"
        config.cache_dir = tmp_path / "workflow_cache"
        config.model_dir = tmp_path / "workflow_models"
        config.ensure_directories()
        
        assert config.data_dir.exists()
        assert config.cache_dir.exists()
        assert config.model_dir.exists()
        
        # 設定更新
        updated = manager.update_config(
            mt5_login=12345,
            mt5_password="test_pass",
            db_user="test_user",
            db_password="db_pass"
        )
        
        # 検証
        validations = manager.validate_connections()
        assert validations["mt5"] is True
        assert validations["database"] is True
        
        # DB URL生成
        db_url = updated.get_db_url()
        assert "test_user:db_pass@localhost:5432/forex_db" in db_url
        
        # エクスポート
        export_path = tmp_path / "final_config.toml"
        manager.export_config(export_path, include_secrets=False)
        assert export_path.exists()
        
        # 再読み込み
        reloaded = manager.reload_config()
        assert isinstance(reloaded, BaseConfig)