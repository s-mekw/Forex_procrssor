"""
基本的なティック→バー変換テスト（デモモード版）
MT5接続なしでシミュレートされたティックデータを使用
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List
import random
import math
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Tick, Bar
from utils.bar_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, create_bar_table, create_current_bar_panel,
    create_statistics_panel, create_mini_chart, format_timestamp
)
from utils.converter_visualizer import create_visual_tick_to_bar

console = Console()

class TickSimulator:
    """ティックデータシミュレーター"""
    
    def __init__(self, symbol: str, base_price: float = 1.10000):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        self.tick_count = 0
        self.time = datetime.now()
        
    def generate_tick(self) -> dict:
        """リアルなティックデータをシミュレート"""
        # ランダムウォーク + トレンド
        trend = math.sin(self.tick_count / 100) * 0.0001
        random_walk = random.gauss(0, 0.00005)
        self.current_price += trend + random_walk
        
        # Bid/Askスプレッド（0.1〜0.3 pips）
        spread = random.uniform(0.00001, 0.00003)
        
        # ボリューム（対数正規分布）
        volume = max(0.01, random.lognormvariate(0, 0.5))
        
        # 時間進行（0.1〜2秒のランダム間隔）
        self.time += timedelta(seconds=random.uniform(0.1, 2.0))
        
        self.tick_count += 1
        
        return {
            "symbol": self.symbol,
            "time": self.time,
            "bid": self.current_price,
            "ask": self.current_price + spread,
            "volume": volume
        }

class BasicTickToBarDemo:
    """基本的なティック→バー変換デモ"""
    
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
        
        # ティックシミュレーター
        self.simulator = TickSimulator(symbol)
        
    def on_bar_complete(self, bar: Bar):
        """バー完成時のコールバック"""
        self.completed_bars.append(bar)
        self.stats["total_bars"] += 1
        
        # コンソールに通知
        print_success(f"Bar completed at {format_timestamp(bar.end_time)}")
        print_info(f"  OHLC: {float(bar.open):.5f} / {float(bar.high):.5f} / {float(bar.low):.5f} / {float(bar.close):.5f}")
        print_info(f"  Volume: {float(bar.volume):.2f}, Ticks: {bar.tick_count}")
        
    def process_tick(self, tick_data: dict) -> Optional[Bar]:
        """ティックを処理"""
        try:
            # ティックデータを変換
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
        header_text = f"[bold cyan]Tick → Bar Converter Demo - {self.symbol}[/bold cyan]"
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
    print_section("Basic Tick to Bar Conversion Demo")
    
    # 設定
    symbol = "EURUSD"
    print_info(f"Demo mode with simulated data for: {symbol}")
    print_info("Timeframe: 1 minute bars")
    print_warning("This is a demonstration using simulated tick data")
    print_info("Press Ctrl+C to stop")
    
    # テストインスタンスを作成
    demo = BasicTickToBarDemo(symbol)
    
    try:
        print_success("Starting tick simulation...")
        
        # ライブ表示
        with Live(demo.create_display(), refresh_per_second=2, console=console) as live:
            # 300ティック分のシミュレーション
            for i in range(300):
                # シミュレートされたティックを生成
                tick_data = demo.simulator.generate_tick()
                
                # ティックを処理
                completed_bar = demo.process_tick(tick_data)
                
                if completed_bar:
                    # バーが完成した場合の追加処理
                    pass
                
                # 表示更新
                live.update(demo.create_display())
                
                # リアルタイム感を出すための待機
                await asyncio.sleep(0.1)
                
    except KeyboardInterrupt:
        print_warning("\nStopping demo...")
        
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 最終統計を表示
        print_section("Final Statistics")
        print_info(f"Total ticks processed: {demo.stats['total_ticks']}")
        print_info(f"Total bars completed: {demo.stats['total_bars']}")
        
        if demo.stats['total_bars'] > 0:
            print_info(f"Average ticks per bar: {demo.stats['total_ticks'] / demo.stats['total_bars']:.1f}")
        
        if demo.completed_bars:
            print_info(f"First bar: {format_timestamp(demo.completed_bars[0].time)}")
            print_info(f"Last bar: {format_timestamp(demo.completed_bars[-1].time)}")
            
            # 価格範囲
            all_highs = [float(bar.high) for bar in demo.completed_bars]
            all_lows = [float(bar.low) for bar in demo.completed_bars]
            print_info(f"Price range: {min(all_lows):.5f} - {max(all_highs):.5f}")
            
            # 総ボリューム
            total_volume = sum(float(bar.volume) for bar in demo.completed_bars)
            print_info(f"Total volume: {total_volume:.2f}")

if __name__ == "__main__":
    asyncio.run(main())