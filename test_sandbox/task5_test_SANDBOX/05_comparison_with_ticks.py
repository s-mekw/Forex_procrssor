"""
ティックデータとOHLCデータの比較テスト - リアルタイムティックからOHLC生成と検証
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta
from dotenv import load_dotenv

# .envファイルを明示的に読み込む
load_dotenv(Path(__file__).parent.parent.parent / ".env")
import polars as pl
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import time
from typing import List, Dict

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.ohlc_fetcher import HistoricalDataFetcher
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.common.config import BaseConfig
from src.common.models import Tick
from utils.ohlc_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, format_price, format_timestamp
)

console = Console()

class TickToOHLCConverter:
    """ティックデータをOHLCに変換するクラス"""
    
    def __init__(self, timeframe_seconds: int = 60):
        self.timeframe_seconds = timeframe_seconds
        self.ticks = []
        self.ohlc_bars = []
        self.current_bar = None
        self.bar_start_time = None
        
    def add_tick(self, tick: Tick):
        """ティックを追加してOHLCバーを更新"""
        self.ticks.append({
            "time": tick.timestamp,
            "bid": tick.bid,
            "ask": tick.ask,
            "volume": tick.volume
        })
        
        # 現在のバーの時間枠を計算
        bar_time = self._get_bar_time(tick.timestamp)
        
        # 新しいバーを開始するか判定
        if self.bar_start_time is None or bar_time != self.bar_start_time:
            # 現在のバーを完了
            if self.current_bar is not None:
                self.ohlc_bars.append(self.current_bar)
            
            # 新しいバーを開始
            self.bar_start_time = bar_time
            self.current_bar = {
                "time": bar_time,
                "open": tick.bid,
                "high": tick.bid,
                "low": tick.bid,
                "close": tick.bid,
                "tick_volume": 1,
                "spread": int((tick.ask - tick.bid) * 100000)  # pips * 10
            }
        else:
            # 現在のバーを更新
            self.current_bar["high"] = max(self.current_bar["high"], tick.bid)
            self.current_bar["low"] = min(self.current_bar["low"], tick.bid)
            self.current_bar["close"] = tick.bid
            self.current_bar["tick_volume"] += 1
            self.current_bar["spread"] = int((tick.ask - tick.bid) * 100000)
    
    def _get_bar_time(self, tick_time: datetime) -> datetime:
        """ティック時間から対応するバーの開始時間を計算"""
        timestamp = tick_time.timestamp()
        bar_timestamp = (timestamp // self.timeframe_seconds) * self.timeframe_seconds
        return datetime.fromtimestamp(bar_timestamp)
    
    def get_current_bar(self) -> dict:
        """現在のバーを取得"""
        return self.current_bar
    
    def get_completed_bars(self) -> List[dict]:
        """完了したバーを取得"""
        return self.ohlc_bars

class ComparisonAnalyzer:
    """ティックとOHLCデータを比較分析するクラス"""
    
    def __init__(self):
        self.tick_generated_ohlc = []
        self.mt5_ohlc = None
        self.comparison_results = {}
        
    def compare_bars(self, tick_bar: dict, mt5_bar: pl.DataFrame) -> dict:
        """個別のバーを比較"""
        if mt5_bar.is_empty():
            return {"match": False, "reason": "No MT5 bar found"}
        
        mt5_row = mt5_bar.row(0, named=True)
        
        # 価格の差を計算
        open_diff = abs(tick_bar["open"] - mt5_row["open"])
        high_diff = abs(tick_bar["high"] - mt5_row["high"])
        low_diff = abs(tick_bar["low"] - mt5_row["low"])
        close_diff = abs(tick_bar["close"] - mt5_row["close"])
        
        # 許容誤差（0.1 pips = 0.00001）
        tolerance = 0.00001
        
        comparison = {
            "time": tick_bar["time"],
            "match": all([
                open_diff <= tolerance,
                high_diff <= tolerance,
                low_diff <= tolerance,
                close_diff <= tolerance
            ]),
            "open_diff": open_diff,
            "high_diff": high_diff,
            "low_diff": low_diff,
            "close_diff": close_diff,
            "tick_volume_tick": tick_bar["tick_volume"],
            "tick_volume_mt5": mt5_row.get("tick_volume", 0),
            "accuracy": 100 - (open_diff + high_diff + low_diff + close_diff) * 10000  # pips to percentage
        }
        
        return comparison
    
    def create_comparison_table(self) -> Table:
        """比較結果テーブルを作成"""
        table = Table(title="Tick vs MT5 OHLC Comparison", show_header=True, header_style="bold magenta")
        
        table.add_column("Time", style="cyan")
        table.add_column("Source", style="yellow")
        table.add_column("Open", justify="right")
        table.add_column("High", justify="right")
        table.add_column("Low", justify="right")
        table.add_column("Close", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Match", justify="center")
        
        # 最新の5バーを比較
        for i in range(min(5, len(self.tick_generated_ohlc))):
            tick_bar = self.tick_generated_ohlc[-(i+1)]
            
            # ティックから生成したバー
            table.add_row(
                format_timestamp(tick_bar["time"]),
                "Tick",
                format_price(tick_bar["open"]),
                format_price(tick_bar["high"]),
                format_price(tick_bar["low"]),
                format_price(tick_bar["close"]),
                str(tick_bar["tick_volume"]),
                ""
            )
            
            # 対応するMT5バーを探す
            if self.mt5_ohlc is not None and not self.mt5_ohlc.is_empty():
                mt5_bar = self.mt5_ohlc.filter(pl.col("time") == tick_bar["time"])
                if not mt5_bar.is_empty():
                    mt5_row = mt5_bar.row(0, named=True)
                    comparison = self.compare_bars(tick_bar, mt5_bar)
                    
                    match_icon = "✓" if comparison["match"] else "✗"
                    match_color = "green" if comparison["match"] else "red"
                    
                    table.add_row(
                        format_timestamp(mt5_row["time"]),
                        "MT5",
                        format_price(mt5_row["open"]),
                        format_price(mt5_row["high"]),
                        format_price(mt5_row["low"]),
                        format_price(mt5_row["close"]),
                        str(mt5_row.get("tick_volume", 0)),
                        f"[{match_color}]{match_icon}[/{match_color}]"
                    )
                    
                    # 差分を表示
                    if not comparison["match"]:
                        table.add_row(
                            "",
                            "Diff",
                            f"[dim]{comparison['open_diff']:.5f}[/dim]",
                            f"[dim]{comparison['high_diff']:.5f}[/dim]",
                            f"[dim]{comparison['low_diff']:.5f}[/dim]",
                            f"[dim]{comparison['close_diff']:.5f}[/dim]",
                            "",
                            f"[yellow]{comparison['accuracy']:.1f}%[/yellow]"
                        )
            
            # 区切り線
            if i < min(4, len(self.tick_generated_ohlc) - 1):
                table.add_row("─" * 19, "─" * 6, "─" * 7, "─" * 7, "─" * 7, "─" * 7, "─" * 6, "─" * 5)
        
        return table

async def collect_ticks_and_generate_ohlc(symbol: str, duration_seconds: int = 60):
    """ティックを収集してOHLCを生成"""
    converter = TickToOHLCConverter(timeframe_seconds=60)  # 1分足
    collected_ticks = []
    
    try:
        # MT5設定
        config = BaseConfig()
        mt5_config = {
            "account": config.mt5_login,
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout
        }
        mt5_client = MT5ConnectionManager(mt5_config)
        
        # MT5に接続
        if not mt5_client.connect(mt5_config):
            print_error("Failed to connect to MT5")
            return [], []
        
        # TickDataStreamerを作成
        streamer = TickDataStreamer(
            symbol=symbol,
            mt5_client=mt5_client,
            buffer_size=10000  # バッファサイズを増やして、オーバーフローを防ぐ
            # spike_threshold=3.0 (デフォルト値を使用)
        )
        
        print_info(f"Collecting ticks for {symbol} for {duration_seconds} seconds...")
        
        # ストリーミング開始（非同期コンテキストマネージャーを使わない）
        await streamer.start_streaming()
        try:
            start_time = time.time()
            tick_count = 0
            
            # プログレス表示用のレイアウト
            layout = Layout()
            layout.split_column(
                Layout(name="status", size=5),
                Layout(name="current", size=8),
                Layout(name="bars", size=15)
            )
            
            with Live(layout, refresh_per_second=2, console=console) as live:
                last_tick_count = 0
                while time.time() - start_time < duration_seconds:
                    # 最新のティックを取得（より頻繁に取得してバッファを消費）
                    recent_ticks = await streamer.get_recent_ticks()
                    
                    # 新しいティックのみ処理
                    if recent_ticks and len(recent_ticks) > last_tick_count:
                        new_ticks = recent_ticks[last_tick_count:]
                        for tick in new_ticks:
                            tick_count += 1
                            collected_ticks.append(tick)
                            converter.add_tick(tick)
                        last_tick_count = len(recent_ticks)
                    
                    # バッファを定期的に消費するため、短い待機時間を設定
                    await asyncio.sleep(0.01)  # 10msごとにチェック
                    
                    # ステータス更新
                    elapsed = time.time() - start_time
                    remaining = duration_seconds - elapsed
                    
                    status_text = f"""[bold cyan]Tick Collection Status[/bold cyan]
                    
Elapsed: {elapsed:.1f}s / {duration_seconds}s
Remaining: {remaining:.1f}s
Ticks Collected: {tick_count}
Bars Generated: {len(converter.ohlc_bars)}"""
                    
                    layout["status"].update(Panel(status_text, border_style="cyan"))
                    
                    # 現在のバー
                    current_bar = converter.get_current_bar()
                    if current_bar:
                        current_text = f"""[bold]Current Bar (Incomplete)[/bold]

Time: {format_timestamp(current_bar['time'])}
Open: {format_price(current_bar['open'])}
High: {format_price(current_bar['high'])}
Low: {format_price(current_bar['low'])}
Close: {format_price(current_bar['close'])}
Ticks: {current_bar['tick_volume']}"""
                        layout["current"].update(Panel(current_text, border_style="yellow"))
                    
                    # 完成したバー
                    completed_bars = converter.get_completed_bars()
                    if completed_bars:
                        bars_table = Table(show_header=True, header_style="bold magenta")
                        bars_table.add_column("Time")
                        bars_table.add_column("Open", justify="right")
                        bars_table.add_column("High", justify="right")
                        bars_table.add_column("Low", justify="right")
                        bars_table.add_column("Close", justify="right")
                        bars_table.add_column("Ticks", justify="right")
                        
                        for bar in completed_bars[-5:]:  # 最新5本
                            bars_table.add_row(
                                format_timestamp(bar["time"]),
                                format_price(bar["open"]),
                                format_price(bar["high"]),
                                format_price(bar["low"]),
                                format_price(bar["close"]),
                                str(bar["tick_volume"])
                            )
                        
                        layout["bars"].update(Panel(bars_table, title="Generated OHLC Bars", border_style="green"))
            
            # 最後のバーも追加
            if converter.current_bar:
                converter.ohlc_bars.append(converter.current_bar)
            
            print_success(f"Collected {tick_count} ticks and generated {len(converter.ohlc_bars)} OHLC bars")
            
            return converter.ohlc_bars, collected_ticks
        finally:
            await streamer.stop_streaming()
            mt5_client.disconnect()
        
    except Exception as e:
        print_error(f"Error collecting ticks: {str(e)}")
        return [], []

def main():
    """メインテスト関数"""
    print_section("Tick vs OHLC Comparison Test")
    
    # 設定
    symbol = "EURJPY"
    collection_duration = 30  # 30秒間（テスト用に短縮）ティックを収集
    
    print_info(f"Symbol: {symbol}")
    print_info(f"Collection Duration: {collection_duration} seconds")
    print_info("This test will:")
    print_info("  1. Collect live ticks and generate OHLC bars")
    print_info("  2. Fetch corresponding MT5 OHLC data")
    print_info("  3. Compare the generated bars with MT5 data")
    
    try:
        # ティック収集とOHLC生成
        print_section("Phase 1: Tick Collection")
        
        tick_ohlc_bars, collected_ticks = asyncio.run(
            collect_ticks_and_generate_ohlc(symbol, collection_duration)
        )
        
        if not tick_ohlc_bars:
            print_error("No OHLC bars were generated from ticks")
            return
        
        # MT5からOHLCデータを取得
        print_section("Phase 2: Fetching MT5 OHLC Data")
        
        config = BaseConfig()
        mt5_config = {
            "account": config.mt5_login,
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout
        }
        mt5_client = MT5ConnectionManager(mt5_config)
        fetcher = HistoricalDataFetcher(mt5_client=mt5_client, config={})
        
        if not fetcher.connect():
            print_error("Failed to connect to MT5")
            return
        
        # ティック収集期間と同じ時間範囲でMT5データを取得
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=10)  # 余裕を持って10分前から
        
        result = fetcher.fetch_ohlc_data(
            symbol=symbol,
            timeframe="M1",  # 1分足
            start_date=start_date,
            end_date=end_date
        )
        
        mt5_ohlc = None
        if result is not None:  # LazyFrameの存在確認のみ
            mt5_ohlc = result.collect()
            print_success(f"Fetched {len(mt5_ohlc)} MT5 OHLC bars")
        else:
            print_error("Failed to fetch MT5 OHLC data")
        
        # 比較分析
        print_section("Phase 3: Comparison Analysis")
        
        analyzer = ComparisonAnalyzer()
        analyzer.tick_generated_ohlc = tick_ohlc_bars
        analyzer.mt5_ohlc = mt5_ohlc
        
        # 比較テーブルを表示
        comparison_table = analyzer.create_comparison_table()
        console.print(comparison_table)
        
        # 統計サマリー
        print_section("Summary Statistics")
        
        summary = Panel(f"""
[bold cyan]Tick Collection Summary[/bold cyan]

Total Ticks Collected: {len(collected_ticks)}
OHLC Bars Generated: {len(tick_ohlc_bars)}
MT5 Bars Retrieved: {len(mt5_ohlc) if mt5_ohlc else 0}

[bold yellow]Analysis Notes:[/bold yellow]
• Tick-generated OHLC may differ slightly from MT5 due to timing
• MT5 bars are based on server-side aggregation
• Small differences (< 0.1 pips) are normal
• Volume counts may vary based on tick filtering

[bold green]Use Cases:[/bold green]
• Validate data quality
• Test trading strategies
• Understand market microstructure
• Debug data inconsistencies
        """, border_style="blue")
        
        console.print(summary)
        
        # 詳細な比較統計
        if mt5_ohlc and len(tick_ohlc_bars) > 0:
            matches = 0
            total_compared = 0
            accuracy_sum = 0
            
            for tick_bar in tick_ohlc_bars:
                mt5_bar = mt5_ohlc.filter(pl.col("time") == tick_bar["time"])
                if not mt5_bar.is_empty():
                    comparison = analyzer.compare_bars(tick_bar, mt5_bar)
                    total_compared += 1
                    if comparison["match"]:
                        matches += 1
                    accuracy_sum += comparison["accuracy"]
            
            if total_compared > 0:
                match_rate = (matches / total_compared) * 100
                avg_accuracy = accuracy_sum / total_compared
                
                print_info(f"Bars Compared: {total_compared}")
                print_info(f"Perfect Matches: {matches} ({match_rate:.1f}%)")
                print_info(f"Average Accuracy: {avg_accuracy:.2f}%")
        
    except Exception as e:
        print_error(f"Error in comparison test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'fetcher' in locals():
            fetcher.disconnect()
            print_info("Disconnected from MT5")

if __name__ == "__main__":
    main()