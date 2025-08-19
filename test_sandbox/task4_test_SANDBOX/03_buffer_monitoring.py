"""
バッファモニタリング - バッファ使用状況とパフォーマンスの監視
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
import psutil
import time
from datetime import datetime
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from src.common.models import Tick
from utils.test_config import get_mt5_config, TEST_SYMBOLS, create_tick_streamer, STREAMING_CONFIG
from utils.display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, format_percentage, format_price
)

console = Console()

class BufferMonitor:
    """バッファ監視クラス"""
    
    def __init__(self, streamer: TickDataStreamer):
        self.streamer = streamer
        self.start_time = time.time()
        self.metrics = {
            "ticks_received": 0,
            "ticks_dropped": 0,
            "backpressure_events": 0,
            "max_buffer_usage": 0.0,
            "avg_processing_time": 0.0,
            "memory_usage_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "network_latency_ms": 0.0,
        }
        self.processing_times = []
        self.last_tick_time = None
        self.process = psutil.Process()
        
    def on_tick(self, tick: Tick):
        """ティック受信時の処理"""
        current_time = time.time()
        if self.last_tick_time:
            processing_time = (current_time - self.last_tick_time) * 1000  # ms
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
        self.last_tick_time = current_time
        self.metrics["ticks_received"] += 1
        
    def on_backpressure(self, usage: float):
        """バックプレッシャー発生時の処理"""
        self.metrics["backpressure_events"] += 1
        self.metrics["max_buffer_usage"] = max(self.metrics["max_buffer_usage"], usage)
        
    def update_system_metrics(self):
        """システムメトリクスを更新"""
        # メモリ使用量
        memory_info = self.process.memory_info()
        self.metrics["memory_usage_mb"] = memory_info.rss / 1024 / 1024
        
        # CPU使用率
        self.metrics["cpu_usage_percent"] = self.process.cpu_percent()
        
        # 平均処理時間
        if self.processing_times:
            self.metrics["avg_processing_time"] = sum(self.processing_times) / len(self.processing_times)
        
        # ドロップされたティック数
        stats = self.streamer.current_stats
        self.metrics["ticks_dropped"] = stats.get("dropped_ticks", 0)
        
    def get_runtime(self) -> float:
        """実行時間を取得（秒）"""
        return time.time() - self.start_time
    
    def get_throughput(self) -> float:
        """スループットを取得（ティック/秒）"""
        runtime = self.get_runtime()
        if runtime > 0:
            return self.metrics["ticks_received"] / runtime
        return 0.0

def create_buffer_display(monitor: BufferMonitor, streamer: TickDataStreamer) -> Layout:
    """バッファ監視画面を作成"""
    layout = Layout()
    
    # レイアウトを分割
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="buffer", size=8),
        Layout(name="metrics", size=12),
        Layout(name="performance", size=10)
    )
    
    # ヘッダー
    runtime = monitor.get_runtime()
    header_text = Text(
        f"Buffer Monitor - Runtime: {runtime:.1f}s - Symbol: {streamer.symbol}",
        style="bold cyan"
    )
    layout["header"].update(Panel(header_text, border_style="cyan"))
    
    # バッファ状態
    buffer_usage = streamer.buffer_usage
    buffer_size = streamer.buffer_size
    used = int(buffer_usage * buffer_size)
    
    # プログレスバーを作成
    bar_width = 50
    filled = int(bar_width * buffer_usage)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    # 色を決定
    if buffer_usage < 0.5:
        color = "green"
    elif buffer_usage < 0.8:
        color = "yellow"
    else:
        color = "red"
    
    buffer_content = Text.from_markup(
        f"[{color}]{bar}[/{color}]\n"
        f"Usage: {buffer_usage * 100:.1f}% ({used}/{buffer_size})\n"
        f"Backpressure Threshold: {streamer.backpressure_threshold * 100:.0f}%\n"
        f"Backpressure Events: {monitor.metrics['backpressure_events']}\n"
        f"Max Usage: {monitor.metrics['max_buffer_usage'] * 100:.1f}%"
    )
    layout["buffer"].update(Panel(buffer_content, title="Buffer Status", border_style=color))
    
    # メトリクステーブル
    metrics_table = Table(title="Stream Metrics", show_header=True, header_style="bold cyan")
    metrics_table.add_column("Metric", style="magenta")
    metrics_table.add_column("Value", justify="right")
    
    stats = streamer.current_stats
    metrics_table.add_row("Total Ticks", str(monitor.metrics["ticks_received"]))
    metrics_table.add_row("Dropped Ticks", str(monitor.metrics["ticks_dropped"]))
    metrics_table.add_row("Throughput", f"{monitor.get_throughput():.2f} ticks/s")
    metrics_table.add_row("Spikes Detected", str(stats.get("spikes_detected", 0)))
    metrics_table.add_row("Mean Bid", format_price(stats.get("mean_bid", 0)))
    metrics_table.add_row("Mean Ask", format_price(stats.get("mean_ask", 0)))
    metrics_table.add_row("Std Dev Bid", format_price(stats.get("std_bid", 0)))
    metrics_table.add_row("Std Dev Ask", format_price(stats.get("std_ask", 0)))
    
    layout["metrics"].update(metrics_table)
    
    # パフォーマンステーブル
    perf_table = Table(title="System Performance", show_header=True, header_style="bold cyan")
    perf_table.add_column("Metric", style="magenta")
    perf_table.add_column("Value", justify="right")
    
    perf_table.add_row("Memory Usage", f"{monitor.metrics['memory_usage_mb']:.2f} MB")
    perf_table.add_row("CPU Usage", f"{monitor.metrics['cpu_usage_percent']:.1f}%")
    perf_table.add_row("Avg Processing Time", f"{monitor.metrics['avg_processing_time']:.2f} ms")
    perf_table.add_row("Circuit Breaker", "Open" if streamer.circuit_breaker_open else "Closed")
    
    # エラー統計
    error_stats = streamer.error_stats
    for error_type, count in error_stats.items():
        if count is not None and count > 0:
            perf_table.add_row(f"Errors ({error_type})", str(count))
    
    layout["performance"].update(perf_table)
    
    return layout

async def monitor_buffer(symbol: str, duration: int = 60):
    """バッファを監視"""
    print_section(f"Buffer Monitoring - {symbol}")
    
    # 接続
    config = get_mt5_config()
    connection_manager = MT5ConnectionManager(config=config)
    
    print_info("Connecting to MT5...")
    if not connection_manager.connect(config):
        print_error("Failed to connect to MT5")
        return
    
    print_success("Connected to MT5")
    
    # ストリーマーを設定（小さめのバッファでテスト）
    streamer = create_tick_streamer(symbol, connection_manager)
    
    # モニターを初期化
    monitor = BufferMonitor(streamer)
    
    # リスナーを設定
    streamer.add_tick_listener(monitor.on_tick)
    streamer.add_backpressure_listener(monitor.on_backpressure)
    
    # 購読開始
    print_info(f"Subscribing to {symbol}...")
    if not await streamer.subscribe_to_ticks():
        print_error(f"Failed to subscribe to {symbol}")
        connection_manager.disconnect()
        return
    
    print_success(f"Subscribed to {symbol}")
    print_info(f"Monitoring for {duration} seconds... Press Ctrl+C to stop")
    
    # ストリーミング開始
    await streamer.start_streaming()
    streaming_task = streamer._current_task
    
    # 監視を開始
    try:
        with Live(
            create_buffer_display(monitor, streamer),
            refresh_per_second=2,
            screen=True
        ) as live:
            end_time = asyncio.get_event_loop().time() + duration
            
            while asyncio.get_event_loop().time() < end_time:
                await asyncio.sleep(0.5)
                
                # システムメトリクスを更新
                monitor.update_system_metrics()
                
                # 表示を更新
                live.update(create_buffer_display(monitor, streamer))
                
                # ストリーミングタスクの状態をチェック
                if streaming_task.done():
                    print_warning("Streaming task ended unexpectedly")
                    break
                    
    except KeyboardInterrupt:
        print_warning("\nMonitoring interrupted by user")
    finally:
        # クリーンアップ
        await streamer.stop_streaming()
        await streamer.unsubscribe()
        connection_manager.disconnect()
        
        # 最終レポート
        print_section("Final Report")
        console.print(f"Total Runtime: {monitor.get_runtime():.1f} seconds")
        console.print(f"Total Ticks Received: {monitor.metrics['ticks_received']}")
        console.print(f"Total Ticks Dropped: {monitor.metrics['ticks_dropped']}")
        console.print(f"Backpressure Events: {monitor.metrics['backpressure_events']}")
        console.print(f"Max Buffer Usage: {monitor.metrics['max_buffer_usage'] * 100:.1f}%")
        console.print(f"Average Throughput: {monitor.get_throughput():.2f} ticks/s")
        console.print(f"Average Processing Time: {monitor.metrics['avg_processing_time']:.2f} ms")
        console.print(f"Peak Memory Usage: {monitor.metrics['memory_usage_mb']:.2f} MB")

async def main():
    """メイン関数"""
    symbol = TEST_SYMBOLS["default"]
    duration = 30  # 秒
    
    try:
        await monitor_buffer(symbol, duration)
        print_success("Test completed successfully")
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        console.print(traceback.format_exc(), style="red")

if __name__ == "__main__":
    asyncio.run(main())