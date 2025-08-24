"""
Realtime Chart Display with Plotly
BTCUSDのリアルタイムチャートとEMAを表示
"""

import sys
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
from typing import List, Optional
import threading
import queue

# プロジェクトのインポート
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.data_processing.indicators import TechnicalIndicatorEngine
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Bar
from utils.test_config import get_config, validate_config
from utils.chart_helpers import (
    create_candlestick_chart,
    create_realtime_chart_update,
    create_statistics_panel,
    create_progress_indicator,
    print_chart_summary,
    validate_ohlc_data
)

console = Console()

# タイムフレーム変換辞書
TIMEFRAME_TO_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400
}

class RealtimeChartDisplay:
    """リアルタイムチャート表示クラス"""
    
    def __init__(self, config=None):
        """
        初期化
        
        Args:
            config: テスト設定
        """
        self.config = config or get_config("full_test")
        validate_config(self.config)
        
        self.mt5_manager = None
        self.indicator_engine = TechnicalIndicatorEngine(
            ema_periods=self.config.chart.ema_periods
        )
        # TickToBarConverterにsymbol引数を追加
        timeframe_seconds = TIMEFRAME_TO_SECONDS.get(self.config.chart.timeframe, 60)
        self.converter = TickToBarConverter(
            symbol=self.config.chart.symbol,
            timeframe=timeframe_seconds
        )
        
        # データ管理
        self.ohlc_data = None
        self.ema_data = {}
        self.tick_queue = queue.Queue()
        self.current_bar = None
        
        # 統計情報
        self.stats = {
            "ticks_received": 0,
            "bars_completed": 0,
            "start_time": None,
            "last_update": None
        }
        
        # チャート
        self.fig = None
        self.chart_file = "realtime_chart.html"
        
    async def setup(self):
        """MT5接続のセットアップ"""
        console.print("[yellow]Setting up MT5 connection...[/yellow]")
        
        # MT5初期化
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        # シンボル確認と選択
        symbol_info = mt5.symbol_info(self.config.chart.symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {self.config.chart.symbol} not available")
        
        if not symbol_info.visible:
            mt5.symbol_select(self.config.chart.symbol, True)
        
        console.print(f"[green]✓ Connected to MT5 - Symbol: {self.config.chart.symbol}[/green]")
        
        # 初期データ取得
        self.ohlc_data = self.fetch_initial_data()
        
        # 初期EMA計算
        self.calculate_all_ema()
        
        self.stats["start_time"] = datetime.now()
        
    def fetch_initial_data(self) -> pl.DataFrame:
        """初期データを取得"""
        console.print(f"[yellow]Fetching initial {self.config.chart.initial_bars} bars...[/yellow]")
        
        rates = mt5.copy_rates_from_pos(
            self.config.chart.symbol,
            mt5.TIMEFRAME_M1,
            0,
            self.config.chart.initial_bars
        )
        
        if rates is None or len(rates) == 0:
            raise ValueError("Failed to fetch initial data")
        
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
    
    def calculate_all_ema(self):
        """全EMAを計算"""
        if self.ohlc_data is None or self.ohlc_data.is_empty():
            return
        
        df_with_ema = self.indicator_engine.calculate_ema(
            self.ohlc_data,
            price_column="close"
        )
        
        # EMA値を辞書に格納
        for period in self.config.chart.ema_periods:
            col_name = f"ema_{period}"
            if col_name in df_with_ema.columns:
                self.ema_data[period] = df_with_ema[col_name].to_list()
    
    def update_ema_incremental(self, new_price: float):
        """
        EMAを増分更新
        
        Args:
            new_price: 新しい価格
        """
        for period in self.config.chart.ema_periods:
            if period in self.ema_data and len(self.ema_data[period]) > 0:
                # EMAの増分計算
                alpha = 2.0 / (period + 1)
                prev_ema = self.ema_data[period][-1]
                new_ema = alpha * new_price + (1 - alpha) * prev_ema
                
                # 現在のバーが進行中の場合は最後の値を更新
                # （新しいバーは add_new_bar で処理されるため、ここでは更新のみ）
                self.ema_data[period][-1] = new_ema
    
    def create_initial_chart(self):
        """初期チャートを作成"""
        console.print("[yellow]Creating initial chart...[/yellow]")
        
        self.fig = create_candlestick_chart(
            self.ohlc_data,
            self.ema_data,
            self.config,
            title=f"{self.config.chart.symbol} Real-time Chart - {self.config.chart.timeframe}"
        )
        
        # HTMLファイルに保存
        self.fig.write_html(
            self.chart_file,
            auto_open=False,
            config={'displayModeBar': True, 'responsive': True}
        )
        
        console.print(f"[green]✓ Chart created: {self.chart_file}[/green]")
    
    async def tick_receiver(self):
        """ティックデータを受信"""
        console.print("[yellow]Starting tick receiver...[/yellow]")
        
        last_tick_time = datetime.now()
        
        while self.config.chart.realtime_enabled:
            try:
                # 最新ティック取得
                tick = mt5.symbol_info_tick(self.config.chart.symbol)
                
                if tick is None:
                    await asyncio.sleep(0.1)
                    continue
                
                tick_time = datetime.fromtimestamp(tick.time)
                
                # 新しいティックの場合のみ処理
                if tick_time > last_tick_time:
                    tick_data = {
                        "time": tick_time,
                        "bid": float(tick.bid),
                        "ask": float(tick.ask),
                        "volume": float(tick.volume) if hasattr(tick, 'volume') else 0
                    }
                    
                    self.tick_queue.put(tick_data)
                    self.stats["ticks_received"] += 1
                    last_tick_time = tick_time
                
                await asyncio.sleep(0.1)  # 100ms間隔でポーリング
                
            except Exception as e:
                console.print(f"[red]Tick receiver error: {e}[/red]")
                await asyncio.sleep(1)
    
    async def process_ticks(self):
        """ティックを処理してバーに変換"""
        console.print("[yellow]Starting tick processor...[/yellow]")
        
        while self.config.chart.realtime_enabled:
            try:
                # キューからティック取得（タイムアウト付き）
                try:
                    tick = self.tick_queue.get(timeout=1)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                # ティックデータをTickオブジェクトに変換
                from src.common.models import Tick as CommonTick
                
                if isinstance(tick, dict):
                    # 辞書形式の場合、CommonTickに変換
                    tick_obj = CommonTick(
                        symbol=self.config.chart.symbol,
                        timestamp=tick["time"],
                        bid=tick["bid"],
                        ask=tick["ask"],
                        volume=tick.get("volume", 1.0)
                    )
                else:
                    tick_obj = tick
                
                # バーコンバーターに追加
                bar = self.converter.add_tick(tick_obj)
                
                if bar:
                    # 新しいバーが完成
                    self.add_new_bar(bar)
                    self.stats["bars_completed"] += 1
                    console.print(f"[green]New bar completed: {bar.time}[/green]")
                
                # 現在のバーを更新
                self.current_bar = self.converter.get_current_bar()
                
                # EMAを増分更新
                if self.current_bar:
                    self.update_ema_incremental(float(self.current_bar.close))
                
                self.stats["last_update"] = datetime.now()
                
            except Exception as e:
                console.print(f"[red]Tick processor error: {e}[/red]")
                await asyncio.sleep(1)
    
    def add_new_bar(self, bar: Bar):
        """
        新しいバーを追加
        
        Args:
            bar: 新しいバーデータ (Barオブジェクト)
        """
        new_row = pl.DataFrame({
            "time": [bar.time],
            "open": [np.float32(bar.open)],
            "high": [np.float32(bar.high)],
            "low": [np.float32(bar.low)],
            "close": [np.float32(bar.close)],
            "volume": [np.float32(bar.volume)]
        })
        
        self.ohlc_data = pl.concat([self.ohlc_data, new_row])
        
        # 古いデータを削除（メモリ管理）
        max_bars = self.config.chart.initial_bars * 2
        if len(self.ohlc_data) > max_bars:
            self.ohlc_data = self.ohlc_data.tail(max_bars)
    
    async def update_chart(self):
        """チャートを定期的に更新"""
        console.print("[yellow]Starting chart updater...[/yellow]")
        
        while self.config.chart.realtime_enabled:
            try:
                await asyncio.sleep(self.config.chart.update_interval)
                
                if self.stats["last_update"] and \
                   (datetime.now() - self.stats["last_update"]).total_seconds() < 60:
                    
                    # チャートデータを更新
                    self.update_chart_data()
                    
                    # HTMLファイルを再生成
                    self.fig.write_html(
                        self.chart_file,
                        auto_open=False,
                        config={'displayModeBar': True, 'responsive': True}
                    )
                    
                    console.print(f"[dim]Chart updated at {datetime.now().strftime('%H:%M:%S')}[/dim]")
                
            except Exception as e:
                console.print(f"[red]Chart update error: {e}[/red]")
                await asyncio.sleep(5)
    
    def update_chart_data(self):
        """チャートデータを更新"""
        if self.fig is None or self.ohlc_data is None:
            return
        
        # 最新のOHLCデータでチャートを再作成
        self.fig = create_candlestick_chart(
            self.ohlc_data.tail(self.config.chart.initial_bars),
            {k: v[-self.config.chart.initial_bars:] for k, v in self.ema_data.items()},
            self.config,
            title=f"{self.config.chart.symbol} Real-time Chart - Updated: {datetime.now().strftime('%H:%M:%S')}"
        )
    
    async def display_statistics(self):
        """統計情報を表示"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats", size=10),
            Layout(name="footer", size=3)
        )
        
        with Live(layout, refresh_per_second=1, console=console) as live:
            while self.config.chart.realtime_enabled:
                # ヘッダー
                layout["header"].update(
                    Panel(
                        f"[bold cyan]{self.config.chart.symbol} Real-time Chart Monitor[/bold cyan]\n"
                        f"Symbol: {self.config.chart.symbol} | Timeframe: {self.config.chart.timeframe}",
                        border_style="cyan"
                    )
                )
                
                # 統計情報
                if self.stats["start_time"]:
                    stats_panel = create_progress_indicator(
                        self.stats["ticks_received"],
                        self.stats["bars_completed"],
                        self.stats["start_time"]
                    )
                    
                    # 最新価格情報
                    if self.current_bar:
                        price_info = f"Current: ${float(self.current_bar.close):,.2f}"
                        if self.ema_data.get(20):
                            price_info += f" | EMA20: ${self.ema_data[20][-1]:,.2f}"
                        if self.ema_data.get(50):
                            price_info += f" | EMA50: ${self.ema_data[50][-1]:,.2f}"
                        
                        layout["stats"].update(stats_panel)
                        layout["footer"].update(
                            Panel(price_info, border_style="green")
                        )
                
                await asyncio.sleep(1)
    
    async def run(self):
        """メイン実行"""
        try:
            # セットアップ
            await self.setup()
            
            # 初期チャート作成
            self.create_initial_chart()
            
            # ブラウザでチャートを開く
            console.print(f"[cyan]Opening chart in browser: {self.chart_file}[/cyan]")
            webbrowser.open(f"file://{Path(self.chart_file).absolute()}")
            
            # 非同期タスクを開始
            tasks = [
                asyncio.create_task(self.tick_receiver()),
                asyncio.create_task(self.process_ticks()),
                asyncio.create_task(self.update_chart()),
                asyncio.create_task(self.display_statistics())
            ]
            
            console.print("[bold green]Real-time chart started. Press Ctrl+C to stop.[/bold green]")
            
            # タスクを実行
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping real-time chart...[/yellow]")
            self.config.chart.realtime_enabled = False
            
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise
            
        finally:
            # クリーンアップ
            if mt5.initialize():
                mt5.shutdown()
                console.print("[dim]MT5 connection closed[/dim]")
            
            # 最終サマリー
            print_chart_summary(self.ohlc_data, self.ema_data)
            console.print(f"\n[cyan]Chart saved to: {self.chart_file}[/cyan]")

def main():
    """メイン関数"""
    console.print(Panel.fit(
        "[bold cyan]Real-time Chart Display[/bold cyan]\n"
        "Live chart with EMA indicators",
        border_style="cyan"
    ))
    
    # 設定
    config = get_config("full_test")
    
    console.print(f"\n[cyan]Configuration:[/cyan]")
    console.print(f"Symbol: {config.chart.symbol}")
    console.print(f"Timeframe: {config.chart.timeframe}")
    console.print(f"Initial bars: {config.chart.initial_bars}")
    console.print(f"Update interval: {config.chart.update_interval}s")
    console.print(f"EMA periods: {config.chart.ema_periods}")
    
    # 実行
    display = RealtimeChartDisplay(config)
    asyncio.run(display.run())

if __name__ == "__main__":
    main()