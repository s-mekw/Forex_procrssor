"""
複数通貨ペアの同時変換テスト - 複数シンボルのティック→バー変換を並行処理
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any
from collections import defaultdict
import psutil
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Tick, Bar
from src.common.config import BaseConfig
from utils.bar_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, format_price, format_volume, format_timestamp,
    create_conversion_summary
)
from utils.converter_visualizer import ConverterVisualizer

console = Console()

class MultiSymbolConverter:
    """複数通貨ペア同時変換"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.converters: Dict[str, TickToBarConverter] = {}
        self.visualizers: Dict[str, ConverterVisualizer] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}
        
        # 各シンボルに対してコンバーターを作成
        for symbol in symbols:
            self.converters[symbol] = TickToBarConverter(
                symbol=symbol,
                timeframe=60,
                gap_threshold=15,
                max_completed_bars=20
            )
            
            self.visualizers[symbol] = ConverterVisualizer(symbol)
            
            self.stats[symbol] = {
                "ticks": 0,
                "bars": 0,
                "gaps": 0,
                "errors": 0,
                "last_tick_time": None,
                "last_bar_time": None,
                "current_bar": None,
                "avg_spread": 0.0,
                "tick_rate": 0.0,
                "bar_rate": 0.0
            }
            
            # コールバック設定
            self.converters[symbol].on_bar_complete = self.create_bar_callback(symbol)
        
        # グローバル統計
        self.global_stats = {
            "total_ticks": 0,
            "total_bars": 0,
            "start_time": datetime.now(),
            "cpu_percent": 0.0,
            "memory_mb": 0.0
        }
        
        # パフォーマンス計測
        self.process = psutil.Process()
        
    def create_bar_callback(self, symbol: str):
        """シンボル別のバー完成コールバック作成"""
        def callback(bar: Bar):
            self.stats[symbol]["bars"] += 1
            self.stats[symbol]["last_bar_time"] = bar.end_time
            self.global_stats["total_bars"] += 1
            
            # 平均スプレッド更新
            if bar.avg_spread:
                prev_avg = self.stats[symbol]["avg_spread"]
                bar_count = self.stats[symbol]["bars"]
                self.stats[symbol]["avg_spread"] = (
                    (prev_avg * (bar_count - 1) + float(bar.avg_spread)) / bar_count
                )
        
        return callback
    
    def process_tick(self, symbol: str, tick_data: dict):
        """特定シンボルのティックを処理"""
        try:
            # ティック作成
            tick = Tick(
                symbol=tick_data["symbol"],
                time=tick_data["time"],
                bid=Decimal(str(tick_data["bid"])),
                ask=Decimal(str(tick_data["ask"])),
                volume=Decimal(str(tick_data.get("volume", 1.0)))
            )
            
            # ギャップ検出
            gap = self.converters[symbol].check_tick_gap(tick.time)
            if gap:
                self.stats[symbol]["gaps"] += 1
            
            # コンバーターに追加
            self.converters[symbol].add_tick(tick)
            
            # 統計更新
            self.stats[symbol]["ticks"] += 1
            self.stats[symbol]["last_tick_time"] = tick.time
            self.stats[symbol]["current_bar"] = self.converters[symbol].get_current_bar()
            self.global_stats["total_ticks"] += 1
            
        except Exception as e:
            self.stats[symbol]["errors"] += 1
            print_error(f"Error processing {symbol}: {e}")
    
    def update_performance_metrics(self):
        """パフォーマンスメトリクス更新"""
        try:
            self.global_stats["cpu_percent"] = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            self.global_stats["memory_mb"] = memory_info.rss / 1024 / 1024
            
            # レート計算
            elapsed = (datetime.now() - self.global_stats["start_time"]).total_seconds()
            if elapsed > 0:
                for symbol in self.symbols:
                    self.stats[symbol]["tick_rate"] = self.stats[symbol]["ticks"] / elapsed
                    self.stats[symbol]["bar_rate"] = self.stats[symbol]["bars"] / elapsed * 60
                    
        except Exception:
            pass
    
    def create_symbol_status_table(self) -> Table:
        """シンボル別ステータステーブル"""
        table = Table(title="Symbol Status", box=box.ROUNDED)
        
        table.add_column("Symbol", style="cyan")
        table.add_column("Ticks", justify="right")
        table.add_column("Bars", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Gaps", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Tick Rate", justify="right")
        table.add_column("Bar Rate", justify="right")
        table.add_column("Avg Spread", justify="right")
        table.add_column("Status", justify="center")
        
        for symbol in self.symbols:
            stat = self.stats[symbol]
            current_bar = stat["current_bar"]
            
            # 現在価格
            current_price = "N/A"
            if current_bar:
                current_price = f"{float(current_bar.close):.5f}"
            
            # ステータス
            if stat["last_tick_time"]:
                age = (datetime.now() - stat["last_tick_time"]).total_seconds()
                if age < 2:
                    status = "[green]Active[/green]"
                elif age < 10:
                    status = "[yellow]Slow[/yellow]"
                else:
                    status = "[red]Stale[/red]"
            else:
                status = "[dim]Waiting[/dim]"
            
            # エラー/ギャップの色付け
            gap_str = str(stat["gaps"])
            if stat["gaps"] > 0:
                gap_str = f"[yellow]{stat['gaps']}[/yellow]"
            
            error_str = str(stat["errors"])
            if stat["errors"] > 0:
                error_str = f"[red]{stat['errors']}[/red]"
            
            table.add_row(
                symbol,
                str(stat["ticks"]),
                str(stat["bars"]),
                current_price,
                gap_str,
                error_str,
                f"{stat['tick_rate']:.1f}/s",
                f"{stat['bar_rate']:.1f}/m",
                f"{stat['avg_spread']:.5f}" if stat['avg_spread'] > 0 else "N/A",
                status
            )
        
        return table
    
    def create_bar_comparison_panel(self) -> Panel:
        """バー比較パネル"""
        lines = []
        lines.append("[bold]Current Bar Comparison:[/bold]")
        lines.append("")
        
        for symbol in self.symbols:
            current_bar = self.stats[symbol]["current_bar"]
            if current_bar:
                progress = current_bar.tick_count
                bar_visual = "█" * min(progress // 2, 30)
                
                lines.append(f"{symbol:8} [{bar_visual:<30}] {progress} ticks")
                lines.append(f"         OHLC: {float(current_bar.open):.5f} / "
                           f"{float(current_bar.high):.5f} / "
                           f"{float(current_bar.low):.5f} / "
                           f"{float(current_bar.close):.5f}")
                lines.append("")
            else:
                lines.append(f"{symbol:8} [{'':30}] No active bar")
                lines.append("")
        
        return Panel("\n".join(lines), title="Bar Formation", border_style="yellow")
    
    def create_performance_panel(self) -> Panel:
        """パフォーマンスパネル"""
        elapsed = (datetime.now() - self.global_stats["start_time"]).total_seconds()
        
        lines = []
        lines.append(f"[bold]System Performance:[/bold]")
        lines.append(f"  CPU Usage: {self.global_stats['cpu_percent']:.1f}%")
        lines.append(f"  Memory:    {self.global_stats['memory_mb']:.1f} MB")
        lines.append("")
        
        lines.append(f"[bold]Global Statistics:[/bold]")
        lines.append(f"  Total Ticks: {self.global_stats['total_ticks']}")
        lines.append(f"  Total Bars:  {self.global_stats['total_bars']}")
        
        if self.global_stats['total_ticks'] > 0:
            tps = self.global_stats['total_ticks'] / elapsed
            lines.append(f"  Throughput:  {tps:.1f} ticks/sec")
        
        lines.append("")
        lines.append(f"  Uptime: {int(elapsed//60)}m {int(elapsed%60)}s")
        
        # 各シンボルのメモリ使用量
        lines.append("")
        lines.append(f"[bold]Memory per Symbol:[/bold]")
        for symbol in self.symbols:
            bars_in_memory = len(self.converters[symbol].completed_bars)
            lines.append(f"  {symbol}: {bars_in_memory} bars")
        
        return Panel("\n".join(lines), title="Performance", border_style="blue")
    
    def create_display(self) -> Layout:
        """表示レイアウト作成"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=12)
        )
        
        # ヘッダー
        header_text = f"[bold cyan]Multi-Symbol Converter - {', '.join(self.symbols)}[/bold cyan]"
        layout["header"].update(Panel(header_text, style="cyan"))
        
        # メインを左右に分割
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # 左側 - バー比較
        layout["left"].update(self.create_bar_comparison_panel())
        
        # 右側 - パフォーマンス
        layout["right"].update(self.create_performance_panel())
        
        # フッター - ステータステーブル
        layout["footer"].update(self.create_symbol_status_table())
        
        return layout

async def main():
    """メイン関数"""
    print_section("Multi-Symbol Converter Test")
    
    # テストシンボル
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    print_info(f"Testing symbols: {', '.join(symbols)}")
    print_info("Processing multiple symbols simultaneously")
    print_info("Press Ctrl+C to stop")
    
    # コンバーター作成
    converter = MultiSymbolConverter(symbols)
    
    try:
        # MT5設定
        config = BaseConfig()
        mt5_config = {
            "login": config.mt5.login,
            "password": config.mt5.password,
            "server": config.mt5.server,
            "timeout": 60000,
            "retry_count": 3,
            "retry_delay": 1
        }
        
        # 接続
        connection_manager = MT5ConnectionManager(mt5_config)
        
        print_info("Connecting to MT5...")
        if not await connection_manager.connect():
            print_error("Failed to connect to MT5")
            return
        
        print_success("Connected to MT5")
        
        # ストリーマー設定
        streamer_config = StreamerConfig(
            symbols=symbols,
            buffer_size=2000,
            max_tick_age=5.0,
            enable_throttling=False
        )
        
        # ストリーマー作成・開始
        streamer = TickDataStreamer(connection_manager, streamer_config)
        await streamer.start_streaming()
        print_success("Streaming started for all symbols")
        
        # ライブ表示
        with Live(converter.create_display(), refresh_per_second=2, console=console) as live:
            while True:
                # 各シンボルのティックを処理
                for symbol in symbols:
                    ticks = await streamer.get_buffered_ticks(symbol)
                    
                    for tick_data in ticks:
                        converter.process_tick(symbol, tick_data)
                
                # パフォーマンスメトリクス更新
                converter.update_performance_metrics()
                
                # 表示更新
                live.update(converter.create_display())
                
                await asyncio.sleep(0.1)
                
    except KeyboardInterrupt:
        print_warning("\nStopping...")
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # クリーンアップ
        if 'streamer' in locals():
            await streamer.stop_streaming()
            
        if 'connection_manager' in locals():
            connection_manager.disconnect()
        
        # 最終統計
        print_section("Final Statistics")
        
        # グローバル統計
        print_info(f"Total ticks processed: {converter.global_stats['total_ticks']}")
        print_info(f"Total bars generated: {converter.global_stats['total_bars']}")
        
        # シンボル別統計
        print_info("\nPer-Symbol Statistics:")
        for symbol in symbols:
            stat = converter.stats[symbol]
            print_info(f"\n{symbol}:")
            print_info(f"  Ticks: {stat['ticks']}")
            print_info(f"  Bars: {stat['bars']}")
            print_info(f"  Gaps: {stat['gaps']}")
            print_info(f"  Errors: {stat['errors']}")
            
            if stat['bars'] > 0:
                print_info(f"  Avg ticks/bar: {stat['ticks']/stat['bars']:.1f}")
                print_info(f"  Avg spread: {stat['avg_spread']:.5f}")

if __name__ == "__main__":
    asyncio.run(main())