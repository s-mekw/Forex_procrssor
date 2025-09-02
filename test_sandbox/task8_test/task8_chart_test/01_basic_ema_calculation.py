"""
Basic EMA Calculation Test
BTCUSDのヒストリカルデータを取得してEMAを計算
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))

import asyncio
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import MetaTrader5 as mt5

# プロジェクトのインポート
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.data_processing.indicators import TechnicalIndicatorEngine
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Bar
from utils.test_config import get_config, validate_config
from utils.config_loader import parse_args, apply_args_to_env, print_config
from utils.chart_helpers import (
    create_statistics_panel,
    print_chart_summary,
    validate_ohlc_data,
    format_price
)

console = Console()

class EMACalculationTest:
    """EMA計算テストクラス"""
    
    def __init__(self, config=None):
        """
        初期化
        
        Args:
            config: テスト設定
        """
        self.config = config or get_config()
        validate_config(self.config)
        
        self.mt5_manager = None
        self.indicator_engine = TechnicalIndicatorEngine(
            ema_periods=self.config.chart.ema_periods
        )
        self.ohlc_data = None
        self.ema_data = {}
        
    async def setup(self):
        """MT5接続のセットアップ"""
        console.print("[yellow]Setting up MT5 connection...[/yellow]")
        
        # MT5初期化
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        # シンボル確認
        symbol_info = mt5.symbol_info(self.config.chart.symbol)
        if symbol_info is None:
            available_symbols = mt5.symbols_get()
            crypto_symbols = [s.name for s in available_symbols if "BTC" in s.name or "USD" in s.name]
            console.print(f"[red]Symbol {self.config.chart.symbol} not found[/red]")
            console.print(f"Available crypto symbols: {crypto_symbols[:10]}")
            raise ValueError(f"Symbol {self.config.chart.symbol} not available")
        
        if not symbol_info.visible:
            mt5.symbol_select(self.config.chart.symbol, True)
        
        console.print(f"[green]✓ Connected to MT5 - Symbol: {self.config.chart.symbol}[/green]")
        
    def fetch_historical_data(self) -> pl.DataFrame:
        """
        ヒストリカルデータを取得
        
        Returns:
            pl.DataFrame: OHLCデータ
        """
        console.print(f"[yellow]Fetching {self.config.chart.initial_bars} bars of historical data...[/yellow]")
        
        # タイムフレームの変換
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_map.get(self.config.chart.timeframe, mt5.TIMEFRAME_M1)
        
        # データ取得
        rates = mt5.copy_rates_from_pos(
            self.config.chart.symbol,
            mt5_timeframe,
            0,
            self.config.chart.initial_bars
        )
        
        if rates is None or len(rates) == 0:
            raise ValueError("Failed to fetch historical data")
        
        # Polars DataFrameに変換
        df = pl.DataFrame({
            "time": [datetime.fromtimestamp(r['time']) for r in rates],
            "open": np.array([r['open'] for r in rates], dtype=np.float32),
            "high": np.array([r['high'] for r in rates], dtype=np.float32),
            "low": np.array([r['low'] for r in rates], dtype=np.float32),
            "close": np.array([r['close'] for r in rates], dtype=np.float32),
            "volume": np.array([r['tick_volume'] for r in rates], dtype=np.float32)
        })
        
        console.print(f"[green]✓ Fetched {len(df)} bars[/green]")
        return df
    
    def calculate_ema(self) -> pl.DataFrame:
        """
        EMAを計算
        
        Returns:
            pl.DataFrame: EMA列が追加されたデータ
        """
        console.print("[yellow]Calculating EMA indicators...[/yellow]")
        
        # EMA計算
        df_with_ema = self.indicator_engine.calculate_ema(
            self.ohlc_data,
            price_column="close"
        )
        
        # EMA値を辞書に格納
        for period in self.config.chart.ema_periods:
            col_name = f"ema_{period}"
            if col_name in df_with_ema.columns:
                self.ema_data[period] = df_with_ema[col_name].to_list()
        
        console.print(f"[green]✓ Calculated EMA for periods: {self.config.chart.ema_periods}[/green]")
        return df_with_ema
    
    def validate_ema_calculation(self, df_with_ema: pl.DataFrame):
        """
        EMA計算の妥当性を検証
        
        Args:
            df_with_ema: EMA列を含むデータフレーム
        """
        console.print("\n[cyan]Validating EMA calculations...[/cyan]")
        
        validation_results = []
        
        for period in self.config.chart.ema_periods:
            col_name = f"ema_{period}"
            
            if col_name not in df_with_ema.columns:
                validation_results.append((period, "❌", "Column not found"))
                continue
            
            ema_values = df_with_ema[col_name]
            
            # NaN値のチェック
            nan_count = ema_values.null_count()
            if nan_count > period:  # 最初のperiod個はNaNが許容される
                validation_results.append((period, "⚠️", f"{nan_count} NaN values"))
                continue
            
            # 値の範囲チェック
            close_min = df_with_ema["close"].min()
            close_max = df_with_ema["close"].max()
            ema_min = ema_values.min()
            ema_max = ema_values.max()
            
            if ema_min < close_min * 0.5 or ema_max > close_max * 1.5:
                validation_results.append((period, "⚠️", "Values out of expected range"))
                continue
            
            validation_results.append((period, "✅", "Valid"))
        
        # 結果テーブル表示
        table = Table(title="EMA Validation Results")
        table.add_column("Period", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Notes", style="yellow")
        
        for period, status, notes in validation_results:
            table.add_row(f"EMA {period}", status, notes)
        
        console.print(table)
    
    def display_results(self, df_with_ema: pl.DataFrame):
        """
        結果を表示
        
        Args:
            df_with_ema: EMA列を含むデータフレーム
        """
        # 最新のデータを表示
        console.print("\n[bold cyan]Latest Data (Last 5 bars):[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Time", style="white", width=20)
        table.add_column("Close", style="cyan", width=12)
        
        for period in self.config.chart.ema_periods:
            table.add_column(f"EMA {period}", style="yellow", width=12)
        
        # 最後の5行を表示
        for row in df_with_ema.tail(5).iter_rows(named=True):
            row_data = [
                str(row["time"]),
                format_price(row["close"])
            ]
            
            for period in self.config.chart.ema_periods:
                col_name = f"ema_{period}"
                value = row.get(col_name, 0)
                row_data.append(format_price(value) if value else "N/A")
            
            table.add_row(*row_data)
        
        console.print(table)
        
        # EMAのクロスオーバー検出
        self.detect_ema_crossovers(df_with_ema)
    
    def detect_ema_crossovers(self, df: pl.DataFrame):
        """
        EMAのクロスオーバーを検出
        
        Args:
            df: EMA列を含むデータフレーム
        """
        console.print("\n[bold cyan]EMA Crossover Detection:[/bold cyan]")
        
        if len(self.config.chart.ema_periods) < 2:
            console.print("[yellow]Need at least 2 EMA periods for crossover detection[/yellow]")
            return
        
        # EMA20とEMA50のクロスオーバーを検出
        if 20 in self.config.chart.ema_periods and 50 in self.config.chart.ema_periods:
            ema20 = df["ema_20"]
            ema50 = df["ema_50"]
            
            # クロスオーバーポイントを検出
            crossovers = []
            for i in range(1, len(df)):
                prev_diff = ema20[i-1] - ema50[i-1]
                curr_diff = ema20[i] - ema50[i]
                
                if prev_diff < 0 and curr_diff > 0:
                    crossovers.append((df["time"][i], "Golden Cross (Bullish)", df["close"][i]))
                elif prev_diff > 0 and curr_diff < 0:
                    crossovers.append((df["time"][i], "Death Cross (Bearish)", df["close"][i]))
            
            if crossovers:
                table = Table(title="Recent Crossovers")
                table.add_column("Time", style="white")
                table.add_column("Type", style="cyan")
                table.add_column("Price", style="yellow")
                
                for time, cross_type, price in crossovers[-5:]:  # 最新5件
                    table.add_row(str(time), cross_type, format_price(price))
                
                console.print(table)
            else:
                console.print("[yellow]No crossovers detected in the data[/yellow]")
    
    async def run(self):
        """テストを実行"""
        try:
            # セットアップ
            await self.setup()
            
            # ヒストリカルデータ取得
            self.ohlc_data = self.fetch_historical_data()
            
            # データ検証
            if not validate_ohlc_data(self.ohlc_data):
                raise ValueError("Invalid OHLC data")
            
            # EMA計算
            df_with_ema = self.calculate_ema()
            
            # 検証
            self.validate_ema_calculation(df_with_ema)
            
            # 結果表示
            self.display_results(df_with_ema)
            
            # サマリー表示
            print_chart_summary(self.ohlc_data, self.ema_data, self.config.chart.symbol)
            
            # メタデータ表示
            metadata = self.indicator_engine.get_metadata()
            if metadata:
                console.print("\n[bold cyan]Calculation Metadata:[/bold cyan]")
                console.print(f"Indicators calculated: {list(metadata.keys())}")
                if "ema" in metadata:
                    console.print(f"EMA details: {metadata['ema']}")
            
            console.print("\n[bold green]✅ EMA calculation test completed successfully![/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise
        finally:
            # クリーンアップ
            if mt5.initialize():
                mt5.shutdown()
                console.print("[dim]MT5 connection closed[/dim]")

def main():
    """メイン関数"""
    # コマンドライン引数を解析
    args = parse_args()
    apply_args_to_env(args)
    
    console.print(Panel.fit(
        "[bold cyan]EMA Calculation Test[/bold cyan]\n"
        "Testing EMA calculation with configurable periods",
        border_style="cyan"
    ))
    
    # 設定選択
    if args.preset:
        # コマンドラインから指定
        preset = args.preset
        console.print(f"\n[cyan]Using preset: {preset}[/cyan]")
    elif os.environ.get('CI') or not sys.stdin.isatty():
        # CI環境や非対話環境ではデフォルトを使用
        preset = "full"
        console.print("\n[dim]Using default configuration (Full test)[/dim]")
    else:
        console.print("\nSelect test configuration:")
        console.print("1. Quick test (100 bars)")
        console.print("2. Full test (500 bars)")
        console.print("3. Performance test (1000 bars)")
        
        choice = input("Enter choice (1-3, default=2): ").strip() or "2"
        preset_map = {
            "1": "quick",
            "2": "full",
            "3": "performance"
        }
        preset = preset_map.get(choice, "full")
    
    # TOMLから設定を読み込み
    config = get_config(preset=preset, use_toml=True)
    
    console.print(f"\n[cyan]Using configuration: {preset}[/cyan]")
    console.print(f"Symbol: {config.chart.symbol}")
    console.print(f"Timeframe: {config.chart.timeframe}")
    console.print(f"Initial bars: {config.chart.initial_bars}")
    console.print(f"EMA periods: {config.chart.ema_periods}")
    
    # テスト実行
    test = EMACalculationTest(config)
    asyncio.run(test.run())

if __name__ == "__main__":
    main()