"""
ティック欠損検知テスト - ギャップ検出機能の動作確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
import random
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter
from src.common.models import Tick
from src.common.config import BaseConfig
from utils.bar_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, format_timestamp
)
from utils.converter_visualizer import create_gap_detection_visual

console = Console()

class GapDetectionTest:
    """ギャップ検出テスト"""
    
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.converter = TickToBarConverter(
            symbol=symbol,
            timeframe=60,
            gap_threshold=5,  # 5秒でギャップとして検出
            max_completed_bars=50
        )
        
        # ギャップ統計
        self.gap_events: List[Dict[str, Any]] = []
        self.total_gaps = 0
        self.max_gap = 0.0
        self.total_gap_time = 0.0
        
        # シミュレーション設定
        self.simulate_gaps = True
        self.gap_probability = 0.1  # 10%の確率でギャップを発生
        self.min_gap_seconds = 6
        self.max_gap_seconds = 30
        
        # ティック統計
        self.total_ticks = 0
        self.last_tick_time: Optional[datetime] = None
        self.start_time = datetime.now()
        
        # 警告ログ
        self.warning_logs: List[str] = []
        
    def simulate_gap(self) -> float:
        """ギャップをシミュレート"""
        if self.simulate_gaps and random.random() < self.gap_probability:
            gap_seconds = random.uniform(self.min_gap_seconds, self.max_gap_seconds)
            return gap_seconds
        return 0
    
    def process_tick(self, tick: Tick, force_gap: bool = False):
        """ティックを処理（ギャップシミュレーション付き）"""
        try:
            # ギャップシミュレーション
            if force_gap or self.simulate_gap() > 0:
                gap_seconds = random.uniform(self.min_gap_seconds, self.max_gap_seconds)
                simulated_time = tick.timestamp + timedelta(seconds=gap_seconds)
                
                # ギャップイベントを記録
                self.gap_events.append({
                    "timestamp": datetime.now(),
                    "gap_seconds": gap_seconds,
                    "before_time": tick.timestamp,
                    "after_time": simulated_time,
                    "type": "simulated"
                })
                
                # 新しいTickオブジェクトを作成（タイムスタンプを変更）
                tick = Tick(
                    symbol=tick.symbol,
                    timestamp=simulated_time,
                    bid=tick.bid,
                    ask=tick.ask,
                    volume=tick.volume
                )
            
            # ギャップ検出
            gap = self.converter.check_tick_gap(tick.timestamp)
            if gap:
                self.total_gaps += 1
                self.max_gap = max(self.max_gap, gap)
                self.total_gap_time += gap
                
                # 実際に検出されたギャップを記録
                if not (force_gap or self.gap_events and 
                       self.gap_events[-1]["timestamp"] == datetime.now()):
                    self.gap_events.append({
                        "timestamp": datetime.now(),
                        "gap_seconds": gap,
                        "before_time": self.last_tick_time,
                        "after_time": tick.timestamp,
                        "type": "detected"
                    })
                
                # 警告ログに追加
                warning_msg = f"Gap detected: {gap:.1f}s at {format_timestamp(tick.timestamp)}"
                self.warning_logs.append(warning_msg)
                
                if len(self.warning_logs) > 20:
                    self.warning_logs.pop(0)
            
            # コンバーターに追加
            self.converter.add_tick(tick)
            
            # 統計更新
            self.total_ticks += 1
            self.last_tick_time = tick.timestamp
            
        except Exception as e:
            print_error(f"Error processing tick: {e}")
    
    def create_gap_timeline(self) -> Panel:
        """ギャップタイムライン表示"""
        lines = []
        lines.append("[bold]Gap Event Timeline:[/bold]")
        lines.append("")
        
        # 最新10件のギャップイベント
        recent_gaps = self.gap_events[-10:]
        
        if not recent_gaps:
            lines.append("[dim]No gaps detected yet[/dim]")
        else:
            for gap_event in recent_gaps:
                timestamp = gap_event["timestamp"].strftime("%H:%M:%S")
                gap_seconds = gap_event["gap_seconds"]
                gap_type = gap_event["type"]
                
                # ギャップの大きさによって色分け
                if gap_seconds < 10:
                    color = "yellow"
                elif gap_seconds < 20:
                    color = "orange1"
                else:
                    color = "red"
                
                # タイプによってマーク
                if gap_type == "simulated":
                    mark = "📍"
                else:
                    mark = "⚠️"
                
                bar_length = min(int(gap_seconds), 30)
                bar = "█" * bar_length
                
                lines.append(f"{mark} {timestamp} [{color}]{bar}[/{color}] {gap_seconds:.1f}s")
        
        return Panel("\n".join(lines), title="Gap Timeline", border_style="yellow")
    
    def create_statistics_panel(self) -> Panel:
        """統計パネル"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        lines = []
        lines.append(f"[bold]Gap Detection Statistics:[/bold]")
        lines.append(f"  Total Gaps: {self.total_gaps}")
        lines.append(f"  Max Gap: {self.max_gap:.1f}s")
        
        if self.total_gaps > 0:
            avg_gap = self.total_gap_time / self.total_gaps
            lines.append(f"  Avg Gap: {avg_gap:.1f}s")
            lines.append(f"  Total Gap Time: {self.total_gap_time:.1f}s")
            
            gap_ratio = (self.total_gap_time / elapsed) * 100 if elapsed > 0 else 0
            lines.append(f"  Gap Ratio: {gap_ratio:.1f}%")
        
        lines.append("")
        lines.append(f"[bold]Processing Statistics:[/bold]")
        lines.append(f"  Total Ticks: {self.total_ticks}")
        lines.append(f"  Bars Created: {len(self.converter.completed_bars)}")
        
        if elapsed > 0:
            tick_rate = self.total_ticks / elapsed
            lines.append(f"  Tick Rate: {tick_rate:.1f}/s")
        
        lines.append("")
        lines.append(f"[bold]Simulation Settings:[/bold]")
        lines.append(f"  Gap Threshold: {self.converter.gap_threshold}s")
        lines.append(f"  Gap Probability: {self.gap_probability*100:.0f}%")
        lines.append(f"  Gap Range: {self.min_gap_seconds}-{self.max_gap_seconds}s")
        
        return Panel("\n".join(lines), title="Statistics", border_style="blue")
    
    def create_warning_log_panel(self) -> Panel:
        """警告ログパネル"""
        if not self.warning_logs:
            content = "[dim]No warnings yet[/dim]"
        else:
            content = "\n".join(self.warning_logs[-10:])
        
        return Panel(content, title="Warning Log", border_style="orange1")
    
    def create_gap_analysis_table(self) -> Table:
        """ギャップ分析テーブル"""
        table = Table(title="Gap Analysis", box=box.ROUNDED)
        
        table.add_column("Time", style="cyan")
        table.add_column("Gap (s)", justify="right")
        table.add_column("Type", justify="center")
        table.add_column("Before", style="dim")
        table.add_column("After", style="dim")
        table.add_column("Impact", justify="center")
        
        for gap_event in self.gap_events[-15:]:
            gap_seconds = gap_event["gap_seconds"]
            
            # インパクト評価
            if gap_seconds < 10:
                impact = "[yellow]Low[/yellow]"
            elif gap_seconds < 20:
                impact = "[orange1]Medium[/orange1]"
            else:
                impact = "[red]High[/red]"
            
            # タイプ表示
            if gap_event["type"] == "simulated":
                type_str = "[blue]Simulated[/blue]"
            else:
                type_str = "[green]Detected[/green]"
            
            table.add_row(
                gap_event["timestamp"].strftime("%H:%M:%S"),
                f"{gap_seconds:.1f}",
                type_str,
                format_timestamp(gap_event["before_time"]) if gap_event["before_time"] else "N/A",
                format_timestamp(gap_event["after_time"]),
                impact
            )
        
        return table
    
    def create_display(self) -> Layout:
        """表示レイアウト作成"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=12)
        )
        
        # ヘッダー
        header_text = f"[bold cyan]Gap Detection Test - {self.symbol}[/bold cyan]"
        layout["header"].update(Panel(header_text, style="cyan"))
        
        # メインを3列に分割
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="center"),
            Layout(name="right")
        )
        
        # 左側 - ギャップタイムライン
        layout["left"].update(self.create_gap_timeline())
        
        # 中央 - 現在のギャップ状態
        current_gap = self.gap_events[-1]["gap_seconds"] if self.gap_events else 0
        gap_visual = create_gap_detection_visual(current_gap, self.converter.gap_threshold)
        layout["center"].split_column(
            Layout(gap_visual),
            Layout(self.create_warning_log_panel())
        )
        
        # 右側 - 統計
        layout["right"].update(self.create_statistics_panel())
        
        # フッター - ギャップ分析テーブル
        layout["footer"].update(self.create_gap_analysis_table())
        
        return layout

async def main():
    """メイン関数"""
    print_section("Gap Detection Test")
    
    symbol = "EURUSD"
    print_info(f"Testing gap detection for {symbol}")
    print_info("Simulating random gaps to test detection")
    print_warning("Gap threshold: 5 seconds")
    print_info("Press Ctrl+C to stop")
    
    # テスト作成
    test = GapDetectionTest(symbol)
    
    try:
        # MT5設定
        config = BaseConfig()
        mt5_config = {
            "account": config.mt5_login,  # "login"から"account"に修正
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout,
            "retry_count": 3,
            "retry_delay": 1
        }
        
        # 接続
        connection_manager = MT5ConnectionManager()
        
        print_info("Connecting to MT5...")
        if not connection_manager.connect(mt5_config):
            print_error("Failed to connect to MT5")
            return
        
        print_success("Connected to MT5")
        
        # ストリーマー作成・開始
        streamer = TickDataStreamer(
            symbol=symbol,
            buffer_size=1000,
            spike_threshold_percent=0.1,
            backpressure_threshold=0.8,
            mt5_client=connection_manager
        )
        await streamer.start_streaming()
        print_success("Streaming started")
        
        # 初期ギャップを強制的に発生させる
        print_warning("Forcing initial gap for demonstration...")
        
        # ライブ表示
        with Live(test.create_display(), refresh_per_second=2, console=console) as live:
            force_gap_counter = 0
            
            while True:
                # ティック取得・処理
                ticks = await streamer.get_new_ticks()
                
                for i, tick in enumerate(ticks):
                    # 定期的に強制ギャップを発生
                    force_gap = (force_gap_counter % 50 == 0 and force_gap_counter > 0)
                    test.process_tick(tick, force_gap=force_gap)
                    force_gap_counter += 1
                
                # 表示更新
                live.update(test.create_display())
                
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
        print_section("Final Gap Detection Results")
        print_info(f"Total gaps detected: {test.total_gaps}")
        
        if test.total_gaps > 0:
            print_info(f"Maximum gap: {test.max_gap:.1f} seconds")
            print_info(f"Average gap: {test.total_gap_time/test.total_gaps:.1f} seconds")
            print_info(f"Total gap time: {test.total_gap_time:.1f} seconds")
        
        print_info(f"Total ticks processed: {test.total_ticks}")
        
        # ギャップイベントサマリー
        if test.gap_events:
            print_info("\nGap Event Summary:")
            simulated = sum(1 for g in test.gap_events if g["type"] == "simulated")
            detected = sum(1 for g in test.gap_events if g["type"] == "detected")
            print_info(f"  Simulated gaps: {simulated}")
            print_info(f"  Detected gaps: {detected}")

if __name__ == "__main__":
    asyncio.run(main())