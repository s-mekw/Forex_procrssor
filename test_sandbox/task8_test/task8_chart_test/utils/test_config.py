"""
Task 8 Chart Test Configuration
BTCUSD real-time chart with EMA indicators
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
from .config_loader import load_config, Task8Config, validate_config as validate_toml_config

@dataclass
class ChartConfig:
    """チャート表示設定"""
    # 基本設定
    symbol: str = "BTCUSD#"  # BTCUSD#を使用
    timeframe: str = "M1"  # 1分足
    
    # EMA設定
    ema_periods: List[int] = None
    ema_colors: Dict[int, str] = None
    
    # チャート表示設定
    chart_height: int = 800
    chart_width: int = 1200
    show_volume: bool = True
    show_grid: bool = True
    
    # データ取得設定
    initial_bars: int = 500  # 初期表示するバー数
    update_interval: float = 1.0  # 更新間隔（秒）
    
    # リアルタイム設定
    realtime_enabled: bool = True
    max_tick_buffer: int = 1000
    
    def __post_init__(self):
        """デフォルト値の設定"""
        if self.ema_periods is None:
            self.ema_periods = [20, 50, 200]
        
        if self.ema_colors is None:
            self.ema_colors = {
                20: "blue",      # EMA20 - 青
                50: "green",     # EMA50 - 緑  
                200: "red"       # EMA200 - 赤
            }

@dataclass
class MT5Config:
    """MT5接続設定"""
    # 接続パラメータ
    login: int = 51892669  # デモ口座
    password: str = "test123"
    server: str = "Exness-MT5Trial7"
    
    # タイムアウト設定
    timeout: int = 60000
    
    # リトライ設定
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass 
class TestConfig:
    """テスト全体の設定"""
    # 各設定の統合
    chart: ChartConfig = None
    mt5: MT5Config = None
    
    # テストモード設定
    use_mock_data: bool = False  # モックデータを使用するか
    save_results: bool = True     # 結果を保存するか
    verbose: bool = True          # 詳細ログ出力
    
    # パフォーマンス設定
    use_float32: bool = True      # Float32を使用（メモリ最適化）
    
    def __post_init__(self):
        """デフォルト設定の初期化"""
        if self.chart is None:
            self.chart = ChartConfig()
        if self.mt5 is None:
            self.mt5 = MT5Config()

# デフォルト設定のインスタンス
DEFAULT_CONFIG = TestConfig()

# プリセット設定
PRESETS = {
    "quick_test": TestConfig(
        chart=ChartConfig(
            initial_bars=100,
            update_interval=2.0,
            realtime_enabled=False
        )
    ),
    "full_test": TestConfig(
        chart=ChartConfig(
            initial_bars=500,
            update_interval=1.0,
            realtime_enabled=True
        )
    ),
    "performance_test": TestConfig(
        chart=ChartConfig(
            initial_bars=1000,
            update_interval=0.5,
            ema_periods=[10, 20, 50, 100, 200]
        ),
        use_float32=True
    )
}

def get_config(preset: str = None, use_toml: bool = True) -> TestConfig:
    """
    設定を取得
    
    Args:
        preset: プリセット名（"quick_test", "full_test", "performance_test"）
        use_toml: TOMLファイルから設定を読み込むか
        
    Returns:
        TestConfig: テスト設定
    """
    if use_toml:
        # TOMLファイルから設定を読み込み
        try:
            toml_config = load_config(preset=preset)
            # Task8ConfigからTestConfigに変換
            return TestConfig(
                chart=ChartConfig(
                    symbol=toml_config.symbol,
                    timeframe=toml_config.timeframe,
                    ema_periods=toml_config.ema_periods,
                    ema_colors=toml_config.chart.ema_colors,
                    chart_height=toml_config.chart.chart_height,
                    chart_width=toml_config.chart.chart_width,
                    show_volume=toml_config.chart.show_volume,
                    show_grid=toml_config.chart.show_grid,
                    initial_bars=toml_config.chart.initial_bars,
                    update_interval=toml_config.chart.update_interval,
                    realtime_enabled=toml_config.chart.realtime_enabled,
                    max_tick_buffer=toml_config.chart.max_tick_buffer
                ),
                mt5=MT5Config(
                    login=toml_config.mt5.login,
                    password=toml_config.mt5.password,
                    server=toml_config.mt5.server,
                    timeout=toml_config.mt5.timeout,
                    max_retries=toml_config.mt5.max_retries,
                    retry_delay=toml_config.mt5.retry_delay
                ),
                use_mock_data=toml_config.use_mock_data,
                save_results=toml_config.save_results,
                verbose=toml_config.verbose,
                use_float32=toml_config.use_float32
            )
        except Exception as e:
            print(f"Warning: Could not load TOML config: {e}")
            print("Falling back to default config")
    
    # TOMLを使わない場合はプリセットから取得
    if preset and preset in PRESETS:
        return PRESETS[preset]
    return DEFAULT_CONFIG

def validate_config(config: TestConfig) -> bool:
    """
    設定の妥当性を検証
    
    Args:
        config: テスト設定
        
    Returns:
        bool: 検証成功の場合True
    """
    # EMA期間の検証
    for period in config.chart.ema_periods:
        if period <= 0:
            raise ValueError(f"Invalid EMA period: {period}")
    
    # チャートサイズの検証
    if config.chart.chart_height <= 0 or config.chart.chart_width <= 0:
        raise ValueError("Invalid chart dimensions")
    
    # データ取得設定の検証
    if config.chart.initial_bars <= 0:
        raise ValueError("Initial bars must be positive")
    
    if config.chart.update_interval <= 0:
        raise ValueError("Update interval must be positive")
    
    return True

# カラーテーマ設定
CHART_THEME = {
    "background": "#1e1e1e",
    "grid": "#333333",
    "text": "#ffffff",
    "candlestick": {
        "bullish": "#26a69a",  # 陽線（緑）
        "bearish": "#ef5350"   # 陰線（赤）
    },
    "volume": {
        "bullish": "#26a69a80",  # 陽線ボリューム（半透明緑）
        "bearish": "#ef535080"   # 陰線ボリューム（半透明赤）
    }
}