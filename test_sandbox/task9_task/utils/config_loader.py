"""
Task 9 Configuration Loader
TOMLファイルから設定を読み込み、環境変数でオーバーライド可能
RCIインジケーター設定に対応
"""

import os
import toml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv

@dataclass
class ChartConfigFromToml:
    """TOMLから読み込んだチャート設定"""
    symbol: str
    timeframe: str
    initial_bars: int
    update_interval: float
    realtime_enabled: bool
    max_tick_buffer: int
    chart_height: int
    chart_width: int
    show_volume: bool
    show_grid: bool
    ema_periods: List[int]
    ema_colors: Dict[int, str]

@dataclass
class RCIConfigFromToml:
    """TOMLから読み込んだRCI設定"""
    periods_short: List[int]
    periods_long: List[int]
    use_float32: bool
    add_reliability: bool
    colors: Dict[int, str]
    levels: Dict[str, float]

@dataclass
class MT5ConfigFromToml:
    """TOMLから読み込んだMT5設定"""
    login: int
    password: str
    server: str
    timeout: int
    max_retries: int
    retry_delay: float

@dataclass
class DashConfigFromToml:
    """TOMLから読み込んだDash設定"""
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    auto_refresh_interval: int = 1000
    ui: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task9Config:
    """Task 9テストの統合設定"""
    chart: ChartConfigFromToml
    rci: RCIConfigFromToml
    mt5: MT5ConfigFromToml
    test_mode: str
    use_mock_data: bool
    save_results: bool
    verbose: bool
    use_float32: bool
    theme: Dict[str, Any]
    symbols: Dict[str, List[str]]
    dash: Optional[DashConfigFromToml] = None
    
    @property
    def symbol(self) -> str:
        """現在のシンボルを取得（環境変数でオーバーライド可能）"""
        return os.environ.get('TASK9_SYMBOL', self.chart.symbol)
    
    @property
    def timeframe(self) -> str:
        """現在のタイムフレームを取得"""
        return os.environ.get('TASK9_TIMEFRAME', self.chart.timeframe)
    
    @property
    def ema_periods(self) -> List[int]:
        """EMA期間リストを取得"""
        env_periods = os.environ.get('TASK9_EMA_PERIODS')
        if env_periods:
            return [int(p.strip()) for p in env_periods.split(',')]
        return self.chart.ema_periods
    
    @property
    def all_rci_periods(self) -> List[int]:
        """すべてのRCI期間を統合したリストを取得"""
        return self.rci.periods_short + self.rci.periods_long

def load_config(
    config_file: Optional[str] = None,
    preset: Optional[str] = None
) -> Task9Config:
    """
    設定ファイルを読み込む
    
    Args:
        config_file: TOMLファイルのパス（省略時はデフォルト）
        preset: プリセット名（"quick", "full", "performance"）
        
    Returns:
        Task9Config: 読み込んだ設定
    """
    # .envファイルを読み込み（プロジェクトルート）
    env_path = Path(__file__).parents[4] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # デフォルトの設定ファイルパス
    if config_file is None:
        config_file = Path(__file__).parent.parent / "task9_config.toml"
    
    # TOMLファイルを読み込み
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = toml.load(f)
    
    # プリセットが指定されている場合は適用
    if preset and f"presets.{preset}" in config_data:
        preset_data = config_data[f"presets"][preset]
        # プリセット設定でベース設定を上書き
        if "initial_bars" in preset_data:
            config_data["chart"]["initial_bars"] = preset_data["initial_bars"]
        if "update_interval" in preset_data:
            config_data["chart"]["update_interval"] = preset_data["update_interval"]
        if "realtime_enabled" in preset_data:
            config_data["chart"]["realtime_enabled"] = preset_data["realtime_enabled"]
        if "ema_periods" in preset_data:
            config_data["ema"]["periods"] = preset_data["ema_periods"]
        if "rci_periods_short" in preset_data:
            config_data["rci"]["periods_short"] = preset_data["rci_periods_short"]
        if "rci_periods_long" in preset_data:
            config_data["rci"]["periods_long"] = preset_data["rci_periods_long"]
    
    # 環境変数でオーバーライド
    if os.environ.get('TASK9_SYMBOL'):
        config_data["chart"]["symbol"] = os.environ['TASK9_SYMBOL']
    if os.environ.get('TASK9_TIMEFRAME'):
        config_data["chart"]["timeframe"] = os.environ['TASK9_TIMEFRAME']
    if os.environ.get('TASK9_TEST_MODE'):
        config_data["general"]["test_mode"] = os.environ['TASK9_TEST_MODE']
    
    # ChartConfig作成
    chart_config = ChartConfigFromToml(
        symbol=config_data["chart"]["symbol"],
        timeframe=config_data["chart"]["timeframe"],
        initial_bars=config_data["chart"]["initial_bars"],
        update_interval=config_data["chart"]["update_interval"],
        realtime_enabled=config_data["chart"]["realtime_enabled"],
        max_tick_buffer=config_data["chart"]["max_tick_buffer"],
        chart_height=config_data["chart"]["chart_height"],
        chart_width=config_data["chart"]["chart_width"],
        show_volume=config_data["chart"]["show_volume"],
        show_grid=config_data["chart"]["show_grid"],
        ema_periods=config_data["ema"]["periods"],
        ema_colors={int(k): v for k, v in config_data["ema"]["colors"].items()}
    )
    
    # RCIConfig作成
    rci_config = RCIConfigFromToml(
        periods_short=config_data["rci"]["periods_short"],
        periods_long=config_data["rci"]["periods_long"],
        use_float32=config_data["rci"]["use_float32"],
        add_reliability=config_data["rci"]["add_reliability"],
        colors={int(k): v for k, v in config_data["rci"]["colors"].items()},
        levels={k: v for k, v in config_data["rci"]["levels"].items()}
    )
    
    # MT5Config作成（環境変数を優先）
    mt5_config = MT5ConfigFromToml(
        login=int(os.environ.get('FOREX_MT5_LOGIN', config_data["mt5"]["login"])),
        password=os.environ.get('FOREX_MT5_PASSWORD', config_data["mt5"]["password"]),
        server=os.environ.get('FOREX_MT5_SERVER', config_data["mt5"]["server"]),
        timeout=int(os.environ.get('FOREX_MT5_TIMEOUT', config_data["mt5"]["timeout"])),
        max_retries=config_data["mt5"]["max_retries"],
        retry_delay=config_data["mt5"]["retry_delay"]
    )
    
    # Dash設定作成（存在する場合）
    dash_config = None
    if "dash" in config_data:
        dash_config = DashConfigFromToml(
            host=config_data["dash"].get("host", "0.0.0.0"),
            port=config_data["dash"].get("port", 8050),
            debug=config_data["dash"].get("debug", False),
            auto_refresh_interval=config_data["dash"].get("auto_refresh_interval", 1000),
            ui=config_data["dash"].get("ui", {})
        )
    
    # 統合設定作成
    return Task9Config(
        chart=chart_config,
        rci=rci_config,
        mt5=mt5_config,
        test_mode=config_data["general"]["test_mode"],
        use_mock_data=config_data["general"]["use_mock_data"],
        save_results=config_data["general"]["save_results"],
        verbose=config_data["general"]["verbose"],
        use_float32=config_data["performance"]["use_float32"],
        theme=config_data["theme"],
        symbols=config_data["symbols"],
        dash=dash_config
    )

def get_available_symbols(config: Task9Config, category: str = "all") -> List[str]:
    """
    利用可能なシンボルリストを取得
    
    Args:
        config: Task9設定
        category: "crypto", "forex", または "all"
        
    Returns:
        シンボルリスト
    """
    if category == "crypto":
        return config.symbols.get("crypto", [])
    elif category == "forex":
        return config.symbols.get("forex", [])
    else:
        crypto = config.symbols.get("crypto", [])
        forex = config.symbols.get("forex", [])
        return crypto + forex

def validate_config(config: Task9Config) -> bool:
    """
    設定の妥当性を検証
    
    Args:
        config: 検証する設定
        
    Returns:
        bool: 有効な場合True
        
    Raises:
        ValueError: 無効な設定の場合
    """
    # シンボルの検証
    if not config.chart.symbol:
        raise ValueError("Symbol is required")
    
    # タイムフレームの検証
    valid_timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
    if config.chart.timeframe not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {config.chart.timeframe}")
    
    # EMA期間の検証
    for period in config.chart.ema_periods:
        if period <= 0:
            raise ValueError(f"Invalid EMA period: {period}")
    
    # RCI期間の検証
    all_rci_periods = config.rci.periods_short + config.rci.periods_long
    for period in all_rci_periods:
        if period <= 0 or period > 999:
            raise ValueError(f"Invalid RCI period: {period}")
    
    # チャートサイズの検証
    if config.chart.chart_height <= 0 or config.chart.chart_width <= 0:
        raise ValueError("Invalid chart dimensions")
    
    # データ取得設定の検証
    if config.chart.initial_bars <= 0:
        raise ValueError("Initial bars must be positive")
    
    if config.chart.update_interval <= 0:
        raise ValueError("Update interval must be positive")
    
    return True

def print_config(config: Task9Config, show_source: bool = False):
    """
    設定を見やすく表示
    
    Args:
        config: 表示する設定
        show_source: 設定ソースを表示するか
    """
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # 基本設定テーブル
    table = Table(title="Task 9 RCI Chart Configuration", show_header=True)
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Setting", style="yellow", width=25)
    table.add_column("Value", style="white", width=30)
    
    # General設定
    table.add_row("General", "Test Mode", config.test_mode)
    table.add_row("", "Use Mock Data", str(config.use_mock_data))
    table.add_row("", "Save Results", str(config.save_results))
    table.add_row("", "Verbose", str(config.verbose))
    
    # Chart設定
    table.add_row("Chart", "Symbol", config.symbol)
    table.add_row("", "Timeframe", config.timeframe)
    table.add_row("", "Initial Bars", str(config.chart.initial_bars))
    table.add_row("", "Update Interval", f"{config.chart.update_interval}s")
    table.add_row("", "Real-time Enabled", str(config.chart.realtime_enabled))
    table.add_row("", "Chart Size", f"{config.chart.chart_width}x{config.chart.chart_height}")
    
    # EMA設定
    table.add_row("EMA", "Periods", str(config.ema_periods))
    ema_colors = ", ".join([f"{p}:{c}" for p, c in config.chart.ema_colors.items()])
    table.add_row("", "Colors", ema_colors)
    
    # RCI設定
    table.add_row("RCI", "Short Periods", str(config.rci.periods_short))
    table.add_row("", "Long Periods", str(config.rci.periods_long))
    table.add_row("", "Use Float32", str(config.rci.use_float32))
    rci_colors = ", ".join([f"{p}:{c}" for p, c in config.rci.colors.items()])
    table.add_row("", "Colors", rci_colors)
    table.add_row("", "Overbought Level", str(config.rci.levels['overbought']))
    table.add_row("", "Oversold Level", str(config.rci.levels['oversold']))
    
    # MT5設定
    table.add_row("MT5", "Server", config.mt5.server)
    table.add_row("", "Login", str(config.mt5.login))
    table.add_row("", "Timeout", f"{config.mt5.timeout}ms")
    
    # 設定ソース情報を表示
    if show_source:
        mt5_source = "TOML"
        if os.environ.get('FOREX_MT5_LOGIN'):
            mt5_source = ".env"
        table.add_row("", "Config Source", mt5_source)
    
    console.print(table)

# コマンドライン引数を処理するヘルパー関数
def parse_args():
    """コマンドライン引数を解析"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task 9 RCI Chart Configuration')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--preset', type=str, choices=['quick', 'full', 'performance'],
                       help='Preset configuration')
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., EURJPY#)')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., M1, M5)')
    parser.add_argument('--ema-periods', type=str, help='EMA periods (comma-separated)')
    parser.add_argument('--rci-periods-short', type=str, help='Short RCI periods (comma-separated)')
    parser.add_argument('--rci-periods-long', type=str, help='Long RCI periods (comma-separated)')
    parser.add_argument('--test-mode', type=str, choices=['quick', 'full', 'performance'],
                       help='Test mode')
    
    return parser.parse_args()

def apply_args_to_env(args):
    """コマンドライン引数を環境変数に設定"""
    if args.symbol:
        os.environ['TASK9_SYMBOL'] = args.symbol
    if args.timeframe:
        os.environ['TASK9_TIMEFRAME'] = args.timeframe
    if args.ema_periods:
        os.environ['TASK9_EMA_PERIODS'] = args.ema_periods
    if args.rci_periods_short:
        os.environ['TASK9_RCI_PERIODS_SHORT'] = args.rci_periods_short
    if args.rci_periods_long:
        os.environ['TASK9_RCI_PERIODS_LONG'] = args.rci_periods_long
    if args.test_mode:
        os.environ['TASK9_TEST_MODE'] = args.test_mode

if __name__ == "__main__":
    # テスト用のメイン処理
    args = parse_args()
    apply_args_to_env(args)
    
    config = load_config(config_file=args.config, preset=args.preset)
    
    if validate_config(config):
        print_config(config, show_source=True)
        print(f"\n✅ Configuration is valid!")
        print(f"Symbol: {config.symbol}")
        print(f"EMA Periods: {config.ema_periods}")
        print(f"RCI Periods: {config.all_rci_periods}")
        
        # .env使用状況を表示
        if os.environ.get('FOREX_MT5_LOGIN'):
            print(f"\n🔒 Using MT5 credentials from .env file")
        else:
            print(f"\n⚠️  Using default MT5 credentials from TOML (consider setting up .env)")