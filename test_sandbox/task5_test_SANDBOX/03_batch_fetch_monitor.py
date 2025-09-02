"""
バッチ処理監視テスト - 大量データの並列取得とパフォーマンス監視
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta
import polars as pl
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
import time
import psutil
import threading
from typing import Dict, Any

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.ohlc_fetcher import HistoricalDataFetcher
from src.common.config import BaseConfig
from utils.ohlc_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, format_price, format_volume
)

console = Console()

class BatchProcessMonitor:
    """バッチ処理モニタークラス"""
    
    def __init__(self):
        self.stats = {
            "total_records": 0,
            "total_batches": 0,
            "completed_batches": 0,
            "failed_batches": 0,
            "fetch_speed": 0.0,  # records/sec
            "memory_usage": 0.0,  # MB
            "cpu_usage": 0.0,  # %
            "start_time": None,
            "end_time": None,
            "errors": []
        }
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """システムモニタリングを開始"""
        self.monitoring = True
        self.stats["start_time"] = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """システムモニタリングを停止"""
        self.monitoring = False
        self.stats["end_time"] = datetime.now()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_system(self):
        """システムリソースを監視"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU使用率
                self.stats["cpu_usage"] = process.cpu_percent(interval=0.1)
                
                # メモリ使用量
                memory_info = process.memory_info()
                self.stats["memory_usage"] = memory_info.rss / 1024 / 1024  # MB
                
                # 取得速度を計算
                if self.stats["start_time"] and self.stats["total_records"] > 0:
                    elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
                    if elapsed > 0:
                        self.stats["fetch_speed"] = self.stats["total_records"] / elapsed
                
            except Exception as e:
                pass  # モニタリングエラーは無視
            
            time.sleep(0.5)
    
    def create_monitoring_layout(self) -> Layout:
        """モニタリング用レイアウトを作成"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=7),
            Layout(name="stats", size=12),
            Layout(name="performance", size=10)
        )
        
        # ヘッダー
        header_text = "[bold cyan]Batch Processing Monitor[/bold cyan]"
        layout["header"].update(Panel(header_text, border_style="cyan"))
        
        # 進捗情報
        progress_text = f"Total Batches: {self.stats['total_batches']}\n"
        progress_text += f"Completed: [green]{self.stats['completed_batches']}[/green]\n"
        progress_text += f"Failed: [red]{self.stats['failed_batches']}[/red]\n"
        progress_text += f"Progress: {self.stats['completed_batches']}/{self.stats['total_batches']} "
        
        if self.stats['total_batches'] > 0:
            pct = (self.stats['completed_batches'] / self.stats['total_batches']) * 100
            progress_text += f"([cyan]{pct:.1f}%[/cyan])"
        
        layout["progress"].update(Panel(progress_text, title="Progress", border_style="green"))
        
        # 統計情報
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="yellow")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row("Total Records", f"{self.stats['total_records']:,}")
        stats_table.add_row("Fetch Speed", f"{self.stats['fetch_speed']:.1f} rec/s")
        
        if self.stats['start_time']:
            elapsed = datetime.now() - self.stats['start_time']
            stats_table.add_row("Elapsed Time", str(elapsed).split('.')[0])
        
        if self.stats['total_records'] > 0 and self.stats['completed_batches'] > 0:
            avg_batch_size = self.stats['total_records'] / self.stats['completed_batches']
            stats_table.add_row("Avg Batch Size", f"{avg_batch_size:.0f}")
        
        layout["stats"].update(Panel(stats_table, title="Statistics", border_style="blue"))
        
        # パフォーマンス情報
        perf_table = Table(show_header=False, box=None)
        perf_table.add_column("Resource", style="cyan")
        perf_table.add_column("Usage", justify="right")
        
        # CPU使用率のバー
        cpu_bar = self._create_usage_bar(self.stats['cpu_usage'], 100)
        perf_table.add_row("CPU Usage", f"{cpu_bar} {self.stats['cpu_usage']:.1f}%")
        
        # メモリ使用量
        perf_table.add_row("Memory", f"{self.stats['memory_usage']:.1f} MB")
        
        # エラー情報
        if self.stats['errors']:
            perf_table.add_row("Last Error", f"[red]{self.stats['errors'][-1][:50]}...[/red]")
        
        layout["performance"].update(Panel(perf_table, title="Performance", border_style="yellow"))
        
        return layout
    
    def _create_usage_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """使用率バーを作成"""
        if max_value == 0:
            return "─" * width
        
        ratio = min(value / max_value, 1.0)
        filled = int(ratio * width)
        
        if ratio < 0.5:
            color = "green"
        elif ratio < 0.8:
            color = "yellow"
        else:
            color = "red"
        
        bar = f"[{color}]{'█' * filled}{'─' * (width - filled)}[/{color}]"
        return bar

def fetch_large_dataset(monitor: BatchProcessMonitor):
    """大規模データセットを取得"""
    symbols = ["EURJPY", "GBPJPY", "USDJPY"]
    timeframe = "M1"  # 1分足で大量データ
    days_back = 30  # 30日分
    
    print_info(f"Symbols: {', '.join(symbols)}")
    print_info(f"Timeframe: 1 Minute")
    print_info(f"Period: Last {days_back} days")
    
    all_data = {}
    
    try:
        # MT5設定を作成
        config = BaseConfig()
        
        # MT5クライアント用の設定を辞書形式で作成
        mt5_config = {
            "account": config.mt5_login,
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout,
            "path": config.mt5_path  # MT5実行ファイルのパス
        }
        
        # MT5クライアントを作成
        print_info("Creating MT5 client...")
        mt5_client = MT5ConnectionManager(mt5_config)
        
        # HistoricalDataFetcherを作成
        print_info("Initializing Historical Data Fetcher with parallel processing...")
        fetcher_config = {
            "batch_size": 5000,  # 大きめのバッチサイズ
            "max_workers": 4  # 並列処理
        }
        fetcher = HistoricalDataFetcher(
            mt5_client=mt5_client,
            config=fetcher_config
        )
        
        # 接続
        print_info("Connecting to MT5...")
        if not fetcher.connect():
            print_error("Failed to connect to MT5")
            return None
        
        print_success("Connected to MT5 successfully!")
        
        # 日付範囲
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # バッチ数を推定（1日あたり約1440本 × 日数 × シンボル数）
        estimated_records = 1440 * days_back * len(symbols)
        estimated_batches = (estimated_records // 5000) + len(symbols)
        monitor.stats["total_batches"] = estimated_batches
        
        # 各シンボルのデータを取得
        for symbol in symbols:
            print(f"\n[cyan]Fetching {symbol}...[/cyan]")
            
            try:
                start_fetch = time.time()
                
                # データ取得（内部で自動的にバッチ処理される）
                result = fetcher.fetch_ohlc_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if result is not None:
                    df = result.collect()
                    all_data[symbol] = df
                    
                    fetch_time = time.time() - start_fetch
                    records = len(df)
                    
                    monitor.stats["total_records"] += records
                    monitor.stats["completed_batches"] += (records // 5000) + 1
                    
                    print_success(f"  {symbol}: {records:,} records in {fetch_time:.2f}s")
                    print_info(f"  Speed: {records/fetch_time:.1f} records/sec")
                    
                    # データ品質の簡単なチェック
                    if records > 0:
                        print_info(f"  Period: {df['time'].min()} to {df['time'].max()}")
                        print_info(f"  Latest: {format_price(df['close'][-1])}")
                else:
                    monitor.stats["failed_batches"] += 1
                    monitor.stats["errors"].append(f"Failed to fetch {symbol}")
                    print_error(f"  Failed to fetch data for {symbol}")
                    
            except Exception as e:
                monitor.stats["failed_batches"] += 1
                monitor.stats["errors"].append(str(e))
                print_error(f"  Error fetching {symbol}: {str(e)}")
        
        return all_data
        
    except Exception as e:
        print_error(f"Error in batch processing: {str(e)}")
        monitor.stats["errors"].append(str(e))
        return None
        
    finally:
        if 'fetcher' in locals():
            fetcher.disconnect()
            print_info("Disconnected from MT5")

def main():
    """メインテスト関数"""
    print_section("Batch Processing Monitor Test")
    
    monitor = BatchProcessMonitor()
    
    # モニタリング開始
    monitor.start_monitoring()
    
    # ライブ表示を開始
    with Live(monitor.create_monitoring_layout(), refresh_per_second=2, console=console) as live:
        # 別スレッドでレイアウト更新
        def update_display():
            while monitor.monitoring:
                live.update(monitor.create_monitoring_layout())
                time.sleep(0.5)
        
        update_thread = threading.Thread(target=update_display)
        update_thread.daemon = True
        update_thread.start()
        
        # データ取得を実行
        data = fetch_large_dataset(monitor)
        
        # モニタリング停止
        monitor.stop_monitoring()
        time.sleep(1)  # 最終更新を待つ
    
    # 最終結果を表示
    print_section("Final Results")
    
    if data:
        # サマリーテーブル
        summary_table = Table(title="Data Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Symbol", style="cyan")
        summary_table.add_column("Records", justify="right")
        summary_table.add_column("Memory (MB)", justify="right")
        summary_table.add_column("First Time", style="yellow")
        summary_table.add_column("Last Time", style="yellow")
        summary_table.add_column("Latest Close", justify="right")
        
        total_memory = 0
        for symbol, df in data.items():
            memory_mb = df.estimated_size("mb")
            total_memory += memory_mb
            
            # カラム名を確認
            time_col = "timestamp" if "timestamp" in df.columns else "time"
            last_close = df.row(-1, named=True)["close"]
            
            summary_table.add_row(
                symbol,
                f"{len(df):,}",
                f"{memory_mb:.2f}",
                str(df[time_col].min()),
                str(df[time_col].max()),
                format_price(last_close)
            )
        
        console.print(summary_table)
        
        # パフォーマンス統計
        print_section("Performance Statistics")
        
        if monitor.stats["start_time"] and monitor.stats["end_time"]:
            total_time = (monitor.stats["end_time"] - monitor.stats["start_time"]).total_seconds()
            
            print_info(f"Total Time: {total_time:.2f} seconds")
            print_info(f"Total Records: {monitor.stats['total_records']:,}")
            print_info(f"Average Speed: {monitor.stats['total_records']/total_time:.1f} records/sec")
            print_info(f"Total Memory Used: {total_memory:.2f} MB")
            print_info(f"Peak Memory: {monitor.stats['memory_usage']:.2f} MB")
            print_info(f"Average CPU Usage: {monitor.stats['cpu_usage']:.1f}%")
            
            if monitor.stats['failed_batches'] > 0:
                print_warning(f"Failed Batches: {monitor.stats['failed_batches']}")
                for error in monitor.stats['errors'][-3:]:  # 最後の3つのエラー
                    print_error(f"  - {error}")
            else:
                print_success("All batches completed successfully!")
    else:
        print_error("No data was fetched")

if __name__ == "__main__":
    main()