"""
Task7 Test Sandboxデモ用設定管理モジュール

このモジュールはtest_sandbox/task7_test_SANDBOX内のデモスクリプト専用の
設定管理機能を提供します。本番環境の設定とは独立して動作します。
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
if sys.version_info >= (3, 11):
    import tomllib as tomli
else:
    import tomli
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class TickStreamerConfig(BaseModel):
    """TickDataStreamerの設定"""
    buffer_size: int = Field(default=1000, gt=0, le=10000)
    spike_threshold_percent: float = Field(default=0.5, ge=0.0, le=10.0)
    backpressure_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.0, le=60.0)


class PolarsEngineConfig(BaseModel):
    """Polars処理エンジンの設定"""
    chunk_size: int = Field(default=100, gt=0, le=10000)
    memory_optimization: bool = Field(default=True)
    enable_lazy_evaluation: bool = Field(default=True)
    max_memory_gb: float = Field(default=8.0, gt=0.0, le=128.0)
    adaptive_chunk_size: bool = Field(default=False)
    min_chunk_size: int = Field(default=50, gt=0)
    max_chunk_size: int = Field(default=500, gt=0)
    streaming_batch_size: int = Field(default=5000, gt=0)
    aggregation_window: int = Field(default=60, gt=0)


class DisplayConfig(BaseModel):
    """表示設定"""
    refresh_rate: float = Field(default=2.0, ge=0.1, le=60.0)
    progress_bar: bool = Field(default=True)
    show_memory_stats: bool = Field(default=True)
    show_performance_metrics: bool = Field(default=True)
    max_display_rows: int = Field(default=10, gt=0, le=100)


class PerformanceConfig(BaseModel):
    """パフォーマンス設定"""
    max_workers: int = Field(default=4, gt=0, le=32)
    queue_size: int = Field(default=1000, gt=0, le=10000)
    batch_timeout: float = Field(default=5.0, gt=0.0, le=60.0)
    memory_check_interval: float = Field(default=10.0, gt=0.0, le=300.0)


class DemoConfig:
    """Task7デモ設定管理クラス"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        設定を初期化
        
        Args:
            config_file: 設定ファイルのパス（指定しない場合はデフォルトパスを使用）
        """
        if config_file is None:
            # デフォルトパス: 同じディレクトリのdemo_config.toml
            config_file = Path(__file__).parent / "demo_config.toml"
        
        self.config_file = config_file
        self.config_data = {}
        self.load_config()
        
    def load_config(self) -> None:
        """設定ファイルを読み込む"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_file}")
        
        with open(self.config_file, "rb") as f:
            self.config_data = tomli.load(f)
    
    def get_symbol(self, demo_name: str) -> str:
        """
        指定されたデモ用の通貨ペアを取得
        
        Args:
            demo_name: デモ名 ('showcase', 'realtime', 'streaming', 'aggregation')
        
        Returns:
            通貨ペア文字列
        """
        symbols = self.config_data.get("symbols", {})
        symbol = symbols.get(demo_name)
        
        if symbol is None:
            # デフォルトシンボルを返す
            return self.config_data.get("demo", {}).get("default_symbol", "EURUSD")
        
        return symbol
    
    def get_symbols_list(self, demo_name: str) -> List[str]:
        """
        複数通貨ペアのリストを取得（streaming用）
        
        Args:
            demo_name: デモ名
        
        Returns:
            通貨ペアのリスト
        """
        symbols = self.config_data.get("symbols", {})
        symbol_data = symbols.get(demo_name, [])
        
        if isinstance(symbol_data, list):
            return symbol_data
        elif isinstance(symbol_data, str):
            return [symbol_data]
        else:
            return ["EURUSD"]  # デフォルト
    
    def get_tick_streamer_config(self, demo_name: Optional[str] = None) -> TickStreamerConfig:
        """
        TickStreamerの設定を取得
        
        Args:
            demo_name: デモ名（指定するとデモ固有の設定を適用）
        
        Returns:
            TickStreamerConfig インスタンス
        """
        # デフォルト設定を取得
        default_config = self.config_data.get("tick_streamer", {})
        
        # デモ固有の設定があれば上書き
        if demo_name:
            demo_specific = self.config_data.get("tick_streamer", {}).get(demo_name, {})
            config_dict = {**default_config, **demo_specific}
        else:
            config_dict = default_config
        
        # 不要なネストしたキーを削除
        config_dict = {k: v for k, v in config_dict.items() if not isinstance(v, dict)}
        
        return TickStreamerConfig(**config_dict)
    
    def get_polars_engine_config(self, demo_name: Optional[str] = None) -> PolarsEngineConfig:
        """
        Polars処理エンジンの設定を取得
        
        Args:
            demo_name: デモ名（指定するとデモ固有の設定を適用）
        
        Returns:
            PolarsEngineConfig インスタンス
        """
        # デフォルト設定を取得
        default_config = self.config_data.get("polars_engine", {})
        
        # デモ固有の設定があれば上書き
        if demo_name:
            demo_specific = self.config_data.get("polars_engine", {}).get(demo_name, {})
            config_dict = {**default_config, **demo_specific}
        else:
            config_dict = default_config
        
        # 不要なネストしたキーを削除
        config_dict = {k: v for k, v in config_dict.items() if not isinstance(v, dict)}
        
        return PolarsEngineConfig(**config_dict)
    
    def get_display_config(self) -> DisplayConfig:
        """表示設定を取得"""
        config_dict = self.config_data.get("display", {})
        return DisplayConfig(**config_dict)
    
    def get_performance_config(self) -> PerformanceConfig:
        """パフォーマンス設定を取得"""
        config_dict = self.config_data.get("performance", {})
        return PerformanceConfig(**config_dict)
    
    def get_demo_config(self) -> Dict[str, Any]:
        """デモ全般設定を取得"""
        return self.config_data.get("demo", {})
    
    def get_collection_duration(self) -> int:
        """データ収集時間を取得（秒）"""
        return self.config_data.get("demo", {}).get("data_collection_duration", 30)
    
    def reload(self) -> None:
        """設定ファイルを再読み込み"""
        self.load_config()


# シングルトンインスタンス
_demo_config_instance: Optional[DemoConfig] = None


def get_demo_config(config_file: Optional[Path] = None) -> DemoConfig:
    """
    デモ設定のシングルトンインスタンスを取得
    
    Args:
        config_file: 設定ファイルのパス（初回のみ有効）
    
    Returns:
        DemoConfig インスタンス
    """
    global _demo_config_instance
    
    if _demo_config_instance is None:
        _demo_config_instance = DemoConfig(config_file)
    
    return _demo_config_instance


def reset_demo_config() -> None:
    """設定インスタンスをリセット（主にテスト用）"""
    global _demo_config_instance
    _demo_config_instance = None