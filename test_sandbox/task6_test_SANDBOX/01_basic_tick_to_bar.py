"""
基本的なティック→バー変換テスト - MT5からリアルタイムティックを取得してバーに変換
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from asyncio import Queue

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Tick, Bar
from src.common.config import BaseConfig
from utils.bar_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, create_bar_table, create_current_bar_panel,
    create_statistics_panel, create_mini_chart, format_timestamp
)
from utils.converter_visualizer import create_visual_tick_to_bar

console = Console()

class BasicTickToBarTest:
    """基本的なティック→バー変換テスト"""
    
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.converter = TickToBarConverter(
            symbol=symbol,
            timeframe=60,  # 1分バー
            gap_threshold=30,
            max_completed_bars=100
        )
        self.stats = {
            "total_ticks": 0,
            "total_bars": 0,
            "conversion_rate": 0.0,
            "gaps_detected": 0,
            "errors": 0,
            "warnings": 0,
            "memory_bars": 0,
            "start_time": datetime.now()
        }
        self.last_tick: Optional[Tick] = None
        self.completed_bars: List[Bar] = []
        
        # バー完成時のコールバックを設定
        self.converter.on_bar_complete = self.on_bar_complete
        
    def on_bar_complete(self, bar: Bar):
        """バー完成時のコールバック"""
        self.completed_bars.append(bar)
        self.stats["total_bars"] += 1
        
        # ライブ表示中はコンソール出力を抑制（パフォーマンス最適化）
        # 統計情報パネルで確認可能
        # print_success(f"Bar completed at {format_timestamp(bar.end_time)}")
        # print_info(f"  OHLC: {float(bar.open):.5f} / {float(bar.high):.5f} / {float(bar.low):.5f} / {float(bar.close):.5f}")
        # print_info(f"  Volume: {float(bar.volume):.2f}, Ticks: {bar.tick_count}")
        
    def process_tick(self, tick_data) -> Optional[Bar]:
        """ティックを処理"""
        try:
            # common.models.Tickの場合、tick_to_bar.Tickに変換
            if hasattr(tick_data, 'timestamp'):
                # common.models.Tickからtick_to_bar.Tickへ変換
                tick = Tick(
                    symbol=tick_data.symbol,
                    time=tick_data.timestamp,  # timestampをtimeにマップ
                    bid=Decimal(str(tick_data.bid)),
                    ask=Decimal(str(tick_data.ask)),
                    volume=Decimal(str(tick_data.volume)) if tick_data.volume else Decimal("1.0")
                )
            elif isinstance(tick_data, Tick):
                # すでにtick_to_bar.Tickの場合
                tick = tick_data
            else:
                # 辞書の場合は変換
                tick = Tick(
                    symbol=tick_data["symbol"],
                    time=tick_data["time"],
                    bid=Decimal(str(tick_data["bid"])),
                    ask=Decimal(str(tick_data["ask"])),
                    volume=Decimal(str(tick_data.get("volume", 1.0)))
                )
            
            self.last_tick = tick
            self.stats["total_ticks"] += 1
            
            # コンバーターに追加
            completed_bar = self.converter.add_tick(tick)
            
            # 統計更新
            if self.stats["total_bars"] > 0:
                self.stats["conversion_rate"] = self.stats["total_ticks"] / self.stats["total_bars"]
            
            self.stats["memory_bars"] = len(self.converter.completed_bars)
            
            return completed_bar
            
        except Exception as e:
            self.stats["errors"] += 1
            print_error(f"Error processing tick: {e}")
            return None
    
    def create_display(self) -> Layout:
        """表示レイアウトを作成"""
        layout = Layout()
        
        # メインレイアウトを3つに分割
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )
        
        # ヘッダー
        header_text = f"[bold cyan]Tick → Bar Converter Test - {self.symbol}[/bold cyan]"
        layout["header"].update(Panel(header_text, style="cyan"))
        
        # メインを左右に分割
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # 左側を上下に分割
        layout["left"].split_column(
            Layout(name="current_bar"),
            Layout(name="last_tick")
        )
        
        # 現在のバー情報
        current_bar = self.converter.get_current_bar()
        layout["left"]["current_bar"].update(
            create_current_bar_panel(current_bar, self.symbol)
        )
        
        # 最後のティック情報
        if self.last_tick:
            tick_visual = create_visual_tick_to_bar(self.last_tick, current_bar)
            layout["left"]["last_tick"].update(
                Panel(tick_visual, title="Last Tick → Bar", border_style="blue")
            )
        else:
            layout["left"]["last_tick"].update(
                Panel("[dim]Waiting for ticks...[/dim]", title="Last Tick", border_style="blue")
            )
        
        # 右側を上下に分割
        layout["right"].split_column(
            Layout(name="stats"),
            Layout(name="chart")
        )
        
        # 統計情報
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        self.stats["uptime"] = f"{int(uptime//60)}m {int(uptime%60)}s"
        layout["right"]["stats"].update(create_statistics_panel(self.stats))
        
        # ミニチャート
        if self.completed_bars:
            chart = create_mini_chart(self.completed_bars, width=40, height=8)
            layout["right"]["chart"].update(
                Panel(chart, title="Price Chart", border_style="green")
            )
        else:
            layout["right"]["chart"].update(
                Panel("[dim]No completed bars yet[/dim]", title="Price Chart", border_style="green")
            )
        
        # フッター - バーテーブル
        if self.completed_bars:
            table = create_bar_table(self.completed_bars, title="Completed Bars")
            layout["footer"].update(table)
        else:
            layout["footer"].update(
                Panel("[dim]No completed bars yet[/dim]", title="Completed Bars")
            )
        
        return layout

async def main():
    """メイン関数"""
    print_section("Basic Tick to Bar Conversion Test")
    
    # 設定
    symbol = "EURUSD"
    print_info(f"Testing with symbol: {symbol}")
    print_info("Timeframe: 1 minute bars")
    print_info("Press Ctrl+C to stop")
    
    # テストインスタンスを作成
    test = BasicTickToBarTest(symbol)
    
    try:
        # MT5設定
        config = BaseConfig()
        mt5_config = {
            "account": config.mt5_login,  # accountキーを使用
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout,
            "retry_count": 3,
            "retry_delay": 1
        }
        
        # MT5接続マネージャー
        connection_manager = MT5ConnectionManager()
        
        # 接続（同期メソッドなのでawaitを削除）
        print_info("Connecting to MT5...")
        if not connection_manager.connect(mt5_config):
            print_error("Failed to connect to MT5")
            return
        
        print_success("Connected to MT5")
        
        # ティックストリーマー作成（バッファサイズとbackpressure閾値を最適化）
        streamer = TickDataStreamer(
            symbol=symbol,
            buffer_size=5000,  # 1000 -> 5000に増加
            spike_threshold_percent=0.1,
            backpressure_threshold=0.9,  # 0.8 -> 0.9に調整（90%まで許容）
            mt5_client=connection_manager
        )
        
        # ストリーミング開始
        await streamer.start_streaming()
        print_success("Tick streaming started")
        
        # ティック処理とUI更新を分離（非同期タスク）
        async def tick_processor():
            """ティック処理タスク（バックグラウンド）"""
            while True:
                try:
                    # 新しいティックのみを取得（重複を避ける）
                    ticks = await streamer.get_new_ticks()
                    
                    if ticks:
                        # バッチ処理：すべてのティックを処理
                        for tick_data in ticks:
                            # ティックを処理
                            completed_bar = test.process_tick(tick_data)
                            
                            if completed_bar:
                                # バーが完成した場合の追加処理
                                pass
                    
                    # 短い待機（CPU使用率を抑える）
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    print_error(f"Tick processor error: {e}")
                    await asyncio.sleep(0.1)
        
        # ティック処理タスクを開始
        processor_task = asyncio.create_task(tick_processor())
        
        # ライブ表示（更新頻度を1回/秒に調整）
        with Live(test.create_display(), refresh_per_second=1, console=console) as live:
            while True:
                # 表示更新（ティック処理とは独立）
                live.update(test.create_display())
                
                # UI更新の待機
                await asyncio.sleep(0.5)
                
    except KeyboardInterrupt:
        print_warning("\nStopping test...")
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # クリーンアップ
        if 'processor_task' in locals():
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
        
        if 'streamer' in locals():
            await streamer.stop_streaming()
            print_info("Streaming stopped")
            
        if 'connection_manager' in locals():
            connection_manager.disconnect()
            print_info("Disconnected from MT5")
        
        # 最終統計を表示
        print_section("Final Statistics")
        print_info(f"Total ticks processed: {test.stats['total_ticks']}")
        print_info(f"Total bars completed: {test.stats['total_bars']}")
        
        if test.stats['total_bars'] > 0:
            print_info(f"Average ticks per bar: {test.stats['total_ticks'] / test.stats['total_bars']:.1f}")
        
        if test.completed_bars:
            print_info(f"First bar: {format_timestamp(test.completed_bars[0].time)}")
            print_info(f"Last bar: {format_timestamp(test.completed_bars[-1].time)}")

if __name__ == "__main__":
    asyncio.run(main())