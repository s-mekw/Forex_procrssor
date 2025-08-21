"""
リアルタイムバー生成の可視化 - バーの形成過程をアニメーション表示
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List
from collections import deque
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
    create_bar_progress, create_alert_panel
)
from utils.converter_visualizer import (
    ConverterVisualizer, AnimatedFlowDiagram, ProgressTracker,
    create_gap_detection_visual
)

console = Console()

class RealtimeBarBuilder:
    """リアルタイムバービルダー"""
    
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.converter = TickToBarConverter(
            symbol=symbol,
            timeframe=60,  # 1分バー
            gap_threshold=10,  # 10秒でギャップ検出
            max_completed_bars=50
        )
        
        # ビジュアライザー
        self.visualizer = ConverterVisualizer(symbol)
        self.flow_diagram = AnimatedFlowDiagram()
        self.progress_tracker = ProgressTracker()
        
        # 統計
        self.tick_count = 0
        self.bar_count = 0
        self.gap_count = 0
        self.last_gap_seconds = 0.0
        self.max_gap_seconds = 0.0
        
        # アラート
        self.alerts: deque = deque(maxlen=10)
        
        # バー完成コールバック
        self.converter.on_bar_complete = self.on_bar_complete
        
        # ティック履歴（表示用）
        self.recent_ticks: deque = deque(maxlen=20)
        
    def on_bar_complete(self, bar: Bar):
        """バー完成時のコールバック"""
        self.bar_count += 1
        self.progress_tracker.add_bar()
        
        # アラート追加
        self.alerts.append({
            'timestamp': format_timestamp(datetime.now()),
            'level': 'INFO',
            'message': f'Bar completed: {format_timestamp(bar.time)} - {format_timestamp(bar.end_time)}'
        })
        
    def process_tick(self, tick_data: dict):
        """ティックを処理"""
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
            gap = self.converter.check_tick_gap(tick.time)
            if gap:
                self.gap_count += 1
                self.last_gap_seconds = gap
                self.max_gap_seconds = max(self.max_gap_seconds, gap)
                
                self.alerts.append({
                    'timestamp': format_timestamp(datetime.now()),
                    'level': 'WARNING',
                    'message': f'Gap detected: {gap:.1f} seconds'
                })
            
            # コンバーターに追加
            self.converter.add_tick(tick)
            
            # 統計更新
            self.tick_count += 1
            self.progress_tracker.add_tick()
            
            # 表示用に保存
            self.recent_ticks.append({
                'time': format_timestamp(tick.time),
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'spread': float(tick.ask - tick.bid),
                'volume': float(tick.volume)
            })
            
        except Exception as e:
            self.alerts.append({
                'timestamp': format_timestamp(datetime.now()),
                'level': 'ERROR',
                'message': f'Error: {str(e)}'
            })
    
    def create_bar_building_panel(self) -> Panel:
        """バー形成パネルを作成"""
        current_bar = self.converter.get_current_bar()
        
        if not current_bar:
            return Panel("[dim]No active bar[/dim]", title="Bar Building", border_style="yellow")
        
        lines = []
        
        # タイムウィンドウ
        lines.append(f"[cyan]Time Window:[/cyan]")
        lines.append(f"  Start: {format_timestamp(current_bar.time)}")
        lines.append(f"  End:   {format_timestamp(current_bar.end_time)}")
        
        # カウントダウン
        now = datetime.now()
        if now < current_bar.end_time:
            remaining = (current_bar.end_time - now).total_seconds()
            lines.append(f"  [yellow]Remaining: {int(remaining)}s[/yellow]")
        else:
            lines.append(f"  [green]Ready to complete![/green]")
        
        lines.append("")
        
        # OHLC値の変化をアニメーション
        lines.append("[bold]Current Values:[/bold]")
        lines.append(f"  Open:  {float(current_bar.open):.5f}")
        
        # High/Lowの更新を強調
        if self.recent_ticks:
            last_bid = self.recent_ticks[-1]['bid']
            
            high_marker = " ⬆" if last_bid >= float(current_bar.high) else ""
            low_marker = " ⬇" if last_bid <= float(current_bar.low) else ""
            
            lines.append(f"  High:  [green]{float(current_bar.high):.5f}{high_marker}[/green]")
            lines.append(f"  Low:   [red]{float(current_bar.low):.5f}{low_marker}[/red]")
            lines.append(f"  Close: {float(current_bar.close):.5f} → {last_bid:.5f}")
        else:
            lines.append(f"  High:  {float(current_bar.high):.5f}")
            lines.append(f"  Low:   {float(current_bar.low):.5f}")
            lines.append(f"  Close: {float(current_bar.close):.5f}")
        
        lines.append("")
        
        # 進捗バー
        lines.append(create_bar_progress(current_bar.tick_count))
        
        lines.append("")
        lines.append(f"Volume: [yellow]{format_volume(current_bar.volume)}[/yellow]")
        
        if current_bar.avg_spread:
            lines.append(f"Avg Spread: [blue]{float(current_bar.avg_spread):.5f}[/blue]")
        
        return Panel("\n".join(lines), title="Bar Building Process", border_style="yellow")
    
    def create_tick_stream_table(self) -> Table:
        """ティックストリームテーブル"""
        table = Table(title="Recent Ticks", box=box.SIMPLE)
        
        table.add_column("Time", style="cyan")
        table.add_column("Bid", justify="right")
        table.add_column("Ask", justify="right")
        table.add_column("Spread", justify="right", style="blue")
        table.add_column("Volume", justify="right", style="yellow")
        
        for tick in list(self.recent_ticks)[-10:]:
            table.add_row(
                tick['time'],
                f"{tick['bid']:.5f}",
                f"{tick['ask']:.5f}",
                f"{tick['spread']:.5f}",
                f"{tick['volume']:.2f}"
            )
        
        return table
    
    def create_performance_panel(self) -> Panel:
        """パフォーマンスパネル"""
        rates = self.progress_tracker.get_rates()
        
        lines = []
        lines.append(f"[bold]Processing Rates:[/bold]")
        lines.append(f"  Ticks: {rates['tick_rate']:.1f}/sec")
        lines.append(f"  Bars:  {rates['bar_rate']:.2f}/min")
        lines.append("")
        
        lines.append(f"[bold]Counts:[/bold]")
        lines.append(f"  Total Ticks: {self.tick_count}")
        lines.append(f"  Total Bars:  {self.bar_count}")
        
        if self.bar_count > 0:
            lines.append(f"  Avg Ticks/Bar: {self.tick_count/self.bar_count:.1f}")
        
        lines.append("")
        lines.append(f"[bold]Gap Detection:[/bold]")
        lines.append(f"  Gaps Found: {self.gap_count}")
        
        if self.gap_count > 0:
            lines.append(f"  Last Gap: {self.last_gap_seconds:.1f}s")
            lines.append(f"  Max Gap:  {self.max_gap_seconds:.1f}s")
        
        lines.append("")
        lines.append(f"Uptime: {int(rates['uptime'])}s")
        
        return Panel("\n".join(lines), title="Performance", border_style="blue")
    
    def create_display(self) -> Layout:
        """表示レイアウトを作成"""
        layout = Layout()
        
        # メインレイアウト
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=10)
        )
        
        # ヘッダー
        header_text = f"[bold cyan]Realtime Bar Builder - {self.symbol}[/bold cyan]"
        layout["header"].update(Panel(header_text, style="cyan"))
        
        # メインを3列に分割
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="center"),
            Layout(name="right")
        )
        
        # 左側 - フロー図とパイプライン
        layout["left"].split_column(
            Layout(name="flow"),
            Layout(name="pipeline")
        )
        
        flow_diagram = self.flow_diagram.create_flow_diagram(self.tick_count > 0)
        layout["left"]["flow"].update(Panel(flow_diagram, border_style="cyan"))
        
        current_bar = self.converter.get_current_bar()
        current_bar_ticks = current_bar.tick_count if current_bar else 0
        pipeline_panel = self.visualizer.create_pipeline_view(
            self.tick_count, current_bar_ticks, self.bar_count, current_bar is not None
        )
        layout["left"]["pipeline"].update(pipeline_panel)
        
        # 中央 - バー形成プロセス
        layout["center"].split_column(
            Layout(name="bar_building"),
            Layout(name="gap_detection")
        )
        
        layout["center"]["bar_building"].update(self.create_bar_building_panel())
        
        gap_visual = create_gap_detection_visual(
            self.last_gap_seconds,
            self.converter.gap_threshold
        )
        layout["center"]["gap_detection"].update(gap_visual)
        
        # 右側 - パフォーマンスとアラート
        layout["right"].split_column(
            Layout(name="performance"),
            Layout(name="alerts")
        )
        
        layout["right"]["performance"].update(self.create_performance_panel())
        
        alert_data = [dict(a) for a in self.alerts]
        layout["right"]["alerts"].update(create_alert_panel(alert_data))
        
        # フッター - ティックテーブル
        layout["footer"].update(self.create_tick_stream_table())
        
        return layout

async def main():
    """メイン関数"""
    print_section("Realtime Bar Builder Visualization")
    
    # 設定
    symbol = "EURUSD"
    print_info(f"Symbol: {symbol}")
    print_info("Visualizing bar formation process in real-time")
    print_info("Press Ctrl+C to stop")
    
    # ビルダー作成
    builder = RealtimeBarBuilder(symbol)
    
    try:
        # MT5設定
        config = BaseConfig()
        mt5_config = {
            "login": config.mt5_login,
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout,
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
            symbols=[symbol],
            buffer_size=1000,
            max_tick_age=5.0,
            enable_throttling=False
        )
        
        # ストリーマー作成・開始
        streamer = TickDataStreamer(connection_manager, streamer_config)
        await streamer.start_streaming()
        print_success("Streaming started")
        
        # ライブ表示
        with Live(builder.create_display(), refresh_per_second=4, console=console) as live:
            while True:
                # ティック取得・処理
                ticks = await streamer.get_buffered_ticks(symbol)
                
                for tick_data in ticks:
                    builder.process_tick(tick_data)
                
                # 表示更新
                live.update(builder.create_display())
                
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
        print_info(f"Total ticks: {builder.tick_count}")
        print_info(f"Total bars: {builder.bar_count}")
        print_info(f"Gaps detected: {builder.gap_count}")
        
        if builder.gap_count > 0:
            print_warning(f"Maximum gap: {builder.max_gap_seconds:.1f} seconds")

if __name__ == "__main__":
    asyncio.run(main())