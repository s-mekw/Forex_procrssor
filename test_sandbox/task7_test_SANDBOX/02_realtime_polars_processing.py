"""
リアルタイムPolars処理デモンストレーション

このテストでは以下を視覚的に確認できます：
- ライブティックストリームからのリアルタイム処理
- Polarsのチャンク処理によるメモリ効率化
- バッチ処理 vs ストリーミング処理の性能比較
- メモリ圧迫時の動的調整機能
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Deque
from collections import deque
import polars as pl
import psutil
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich import box
import threading
from queue import Queue

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.data_processing.processor import PolarsProcessingEngine
from src.common.models import Tick
from src.common.config import get_config, ConfigManager

console = Console()

class RealtimeProcessingDemo:
    """リアルタイムPolars処理デモクラス"""
    
    def __init__(self):
        self.console = Console()
        self.connection_manager = None
        self.tick_streamer = None
        self.polars_engine = None
        
        # データバッファ
        self.tick_buffer: Queue = Queue(maxsize=1000)
        self.processed_batches: Deque = deque(maxlen=10)  # 最新10バッチを保持
        
        # 統計情報
        self.stats = {
            'total_ticks_received': 0,
            'total_batches_processed': 0,
            'current_batch_size': 0,
            'processing_times': deque(maxlen=20),  # 最新20回の処理時間
            'memory_usage_history': deque(maxlen=50),
            'chunk_sizes_used': deque(maxlen=10),
            'errors': []
        }
        
        # 制御フラグ
        self.is_running = False
        self.processing_thread = None
        
    def setup_connections(self) -> bool:
        """接続とエンジンを初期化"""
        try:
            # ConfigManagerから設定を取得
            config_manager = ConfigManager()
            config_manager.load_config()
            config = get_config()
            
            # MT5接続用の設定辞書を作成
            mt5_config = {
                "account": int(config.mt5_login),  # int型に変換
                "password": config.mt5_password.get_secret_value() if config.mt5_password else "",  # SecretStrから値を取得
                "server": str(config.mt5_server),
                "timeout": int(config.mt5_timeout),
                "path": r"C:\Program Files\Axiory MetaTrader 5\terminal64.exe"  # Axiory MT5のパスを追加
            }
            
            self.connection_manager = MT5ConnectionManager()
            if not self.connection_manager.connect(mt5_config):
                console.print("[red]❌ MT5への接続に失敗[/red]")
                return False
                
            # TickStreamer設定
            self.tick_streamer = TickDataStreamer(
                symbol="EURUSD",
                buffer_size=500,
                spike_threshold_percent=0.3,
                backpressure_threshold=0.75,
                mt5_client=self.connection_manager
            )
            
            # Polarsエンジン（動的チャンクサイズ調整用）
            self.polars_engine = PolarsProcessingEngine(
                chunk_size=25      # 小さな初期チャンクサイズ
            )
            
            console.print("[green]✅ リアルタイム処理システム初期化完了[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ 初期化エラー: {e}[/red]")
            return False
    
    def get_system_metrics(self) -> Dict[str, float]:
        """システムメトリクスを取得"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'memory_mb': memory_info.rss / (1024 * 1024),
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'tick_buffer_size': self.tick_buffer.qsize(),
            'buffer_utilization': (self.tick_buffer.qsize() / 1000) * 100
        }
    
    def create_dataframe_from_ticks(self, ticks: List[Tick]) -> pl.DataFrame:
        """ティックリストからDataFrameを作成"""
        if not ticks:
            return pl.DataFrame()
            
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'volume': tick.volume,
                'spread': float(tick.ask) - float(tick.bid),
                'mid_price': (float(tick.bid) + float(tick.ask)) / 2
            })
        
        return pl.DataFrame(data)
    
    async def tick_collection_worker(self):
        """ティック収集ワーカー（バックグラウンド）"""
        await self.tick_streamer.subscribe_to_ticks()
        
        while self.is_running:
            try:
                # 新しいティックを取得
                new_ticks = self.tick_streamer.get_new_ticks()
                
                for tick in new_ticks:
                    if not self.tick_buffer.full():
                        self.tick_buffer.put(tick)
                        self.stats['total_ticks_received'] += 1
                    else:
                        # バッファが満杯の場合は古いティックを破棄
                        try:
                            self.tick_buffer.get_nowait()  # 古いティックを削除
                            self.tick_buffer.put(tick)
                            self.stats['total_ticks_received'] += 1
                        except:
                            pass
                
                await asyncio.sleep(0.1)  # 100msごとにチェック
                
            except Exception as e:
                self.stats['errors'].append(f"ティック収集エラー: {str(e)[:100]}")
                await asyncio.sleep(1)
        
        await self.tick_streamer.unsubscribe()
    
    def processing_worker(self):
        """データ処理ワーカー（別スレッド）"""
        batch_size = 20  # 初期バッチサイズ
        
        while self.is_running:
            try:
                ticks_batch = []
                
                # バッチサイズ分のティックを収集
                while len(ticks_batch) < batch_size and self.tick_buffer.qsize() > 0:
                    try:
                        tick = self.tick_buffer.get_nowait()
                        ticks_batch.append(tick)
                    except:
                        break
                
                if ticks_batch:
                    start_time = time.time()
                    
                    # DataFrameを作成
                    df = self.create_dataframe_from_ticks(ticks_batch)
                    
                    if not df.is_empty():
                        # データ型最適化
                        optimized_df = self.polars_engine.optimize_dtypes(df)
                        
                        # 簡単な集計処理
                        aggregated = optimized_df.group_by("symbol").agg([
                            pl.col("bid").mean().alias("avg_bid"),
                            pl.col("ask").mean().alias("avg_ask"),
                            pl.col("spread").mean().alias("avg_spread"),
                            pl.col("volume").sum().alias("total_volume"),
                            pl.col("timestamp").count().alias("tick_count")
                        ])
                        
                        # 処理時間を記録
                        processing_time = time.time() - start_time
                        self.stats['processing_times'].append(processing_time)
                        
                        # 処理済みバッチを保存
                        batch_result = {
                            'timestamp': datetime.now(),
                            'original_size': len(ticks_batch),
                            'processing_time': processing_time,
                            'aggregated_data': aggregated,
                            'memory_used_kb': optimized_df.estimated_size() / 1024
                        }
                        self.processed_batches.append(batch_result)
                        self.stats['total_batches_processed'] += 1
                        self.stats['current_batch_size'] = len(ticks_batch)
                        
                        # 動的バッチサイズ調整
                        if processing_time > 0.5:  # 500ms以上かかる場合
                            batch_size = max(10, batch_size - 5)
                        elif processing_time < 0.1:  # 100ms未満の場合
                            batch_size = min(50, batch_size + 5)
                            
                        self.stats['chunk_sizes_used'].append(batch_size)
                
                # メモリ使用量を記録
                metrics = self.get_system_metrics()
                self.stats['memory_usage_history'].append(metrics['memory_mb'])
                
                time.sleep(0.2)  # 200msごとに処理
                
            except Exception as e:
                self.stats['errors'].append(f"処理エラー: {str(e)[:100]}")
                time.sleep(1)
    
    def create_realtime_dashboard(self) -> Layout:
        """リアルタイムダッシュボードレイアウト"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="stats", size=8)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # ヘッダー
        header_text = Text("⚡ リアルタイムPolars処理デモンストレーション", style="bold cyan")
        layout["header"].update(Panel(header_text, box=box.ROUNDED))
        
        # 左側：リアルタイム統計
        realtime_table = Table(title="リアルタイム統計", box=box.SIMPLE)
        realtime_table.add_column("メトリクス", style="cyan")
        realtime_table.add_column("値", style="green")
        
        realtime_table.add_row("受信ティック総数", str(self.stats['total_ticks_received']))
        realtime_table.add_row("処理済みバッチ数", str(self.stats['total_batches_processed']))
        realtime_table.add_row("現在のバッチサイズ", str(self.stats['current_batch_size']))
        realtime_table.add_row("バッファ使用率", f"{self.get_system_metrics()['buffer_utilization']:.1f}%")
        
        # 平均処理時間
        if self.stats['processing_times']:
            avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            realtime_table.add_row("平均処理時間", f"{avg_processing_time:.3f}s")
        
        layout["left"].update(Panel(realtime_table, title="処理統計", box=box.ROUNDED))
        
        # 右側：最新バッチ情報
        if self.processed_batches:
            latest_batch = self.processed_batches[-1]
            
            batch_table = Table(title="最新バッチ処理結果", box=box.SIMPLE)
            batch_table.add_column("項目", style="cyan")
            batch_table.add_column("値", style="yellow")
            
            batch_table.add_row("処理時刻", latest_batch['timestamp'].strftime("%H:%M:%S"))
            batch_table.add_row("ティック数", str(latest_batch['original_size']))
            batch_table.add_row("処理時間", f"{latest_batch['processing_time']:.3f}s")
            batch_table.add_row("使用メモリ", f"{latest_batch['memory_used_kb']:.2f}KB")
            
            # 集計データ
            if not latest_batch['aggregated_data'].is_empty():
                agg_data = latest_batch['aggregated_data'].row(0)
                batch_table.add_row("平均Bid", f"{agg_data[1]:.5f}")
                batch_table.add_row("平均Ask", f"{agg_data[2]:.5f}")
                batch_table.add_row("平均スプレッド", f"{agg_data[3]:.5f}")
            
            layout["right"].update(Panel(batch_table, title="最新処理結果", box=box.ROUNDED))
        else:
            layout["right"].update(Panel("処理待機中...", title="最新処理結果", box=box.ROUNDED))
        
        # 下部：システム統計
        system_metrics = self.get_system_metrics()
        
        system_table = Table(title="システムリソース", box=box.SIMPLE)
        system_table.add_column("項目", style="cyan")
        system_table.add_column("現在値", style="green")
        system_table.add_column("傾向", style="yellow")
        
        system_table.add_row("メモリ使用量", f"{system_metrics['memory_mb']:.2f}MB", 
                           self._get_trend_indicator(self.stats['memory_usage_history']))
        system_table.add_row("CPU使用率", f"{system_metrics['cpu_percent']:.1f}%", "")
        system_table.add_row("バッファサイズ", str(system_metrics['tick_buffer_size']), "")
        
        # 最近のチャンクサイズ傾向
        if self.stats['chunk_sizes_used']:
            recent_chunk = self.stats['chunk_sizes_used'][-1]
            chunk_trend = self._get_trend_indicator(list(self.stats['chunk_sizes_used']))
            system_table.add_row("動的チャンクサイズ", str(recent_chunk), chunk_trend)
        
        # エラー情報
        error_count = len(self.stats['errors'])
        system_table.add_row("エラー数", str(error_count), 
                           "⚠️" if error_count > 0 else "✅")
        
        layout["stats"].update(Panel(system_table, title="システム監視", box=box.ROUNDED))
        
        return layout
    
    def _get_trend_indicator(self, values: list) -> str:
        """値のリストから傾向インジケーターを取得"""
        if len(values) < 2:
            return "➡️"
        
        recent_avg = sum(values[-3:]) / len(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = sum(values[:3]) / len(values[:3]) if len(values) >= 6 else values[0]
        
        if recent_avg > older_avg * 1.1:
            return "📈"
        elif recent_avg < older_avg * 0.9:
            return "📉"
        else:
            return "➡️"
    
    async def run_realtime_demo(self, duration_seconds: int = 60):
        """リアルタイム処理デモの実行"""
        console.print("[bold blue]⚡ リアルタイム処理デモを開始します[/bold blue]\n")
        
        if not self.setup_connections():
            return
        
        try:
            self.is_running = True
            
            # 処理スレッドを開始
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.start()
            
            # ライブダッシュボード表示
            with Live(self.create_realtime_dashboard(), console=console, refresh_per_second=4) as live:
                # ティック収集を開始（非同期）
                tick_task = asyncio.create_task(self.tick_collection_worker())
                
                console.print(f"[yellow]📊 {duration_seconds}秒間リアルタイム処理を実行します...[/yellow]\n")
                
                # メインループ
                for i in range(duration_seconds):
                    await asyncio.sleep(1)
                    live.update(self.create_realtime_dashboard())
                    
                    # 30秒ごとに中間レポート
                    if i % 30 == 29 and i < duration_seconds - 1:
                        console.print(f"\n[cyan]📝 中間レポート ({i+1}秒経過)[/cyan]")
                        self._print_performance_summary()
                        console.print()
                
                # 処理停止
                self.is_running = False
                tick_task.cancel()
                
                # 最終ダッシュボードを5秒間表示
                live.update(self.create_realtime_dashboard())
                await asyncio.sleep(5)
            
            # 最終レポート
            console.print("\n[bold green]🎉 リアルタイム処理完了[/bold green]")
            self._print_performance_summary()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ ユーザーによる中断[/yellow]")
            self.is_running = False
            
        except Exception as e:
            console.print(f"[red]❌ 実行エラー: {e}[/red]")
            self.is_running = False
            
        finally:
            # クリーンアップ
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            if self.connection_manager and self.connection_manager.is_connected():
                self.connection_manager.disconnect()
    
    def _print_performance_summary(self):
        """パフォーマンス要約を表示"""
        if not self.stats['processing_times']:
            console.print("[yellow]まだ処理データがありません[/yellow]")
            return
        
        avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        max_processing_time = max(self.stats['processing_times'])
        min_processing_time = min(self.stats['processing_times'])
        
        ticks_per_second = self.stats['total_ticks_received'] / max(1, len(self.stats['processing_times']))
        
        summary_table = Table(title="パフォーマンス要約", box=box.ROUNDED)
        summary_table.add_column("項目", style="cyan")
        summary_table.add_column("値", style="green")
        
        summary_table.add_row("総受信ティック数", str(self.stats['total_ticks_received']))
        summary_table.add_row("総処理バッチ数", str(self.stats['total_batches_processed']))
        summary_table.add_row("平均処理時間", f"{avg_processing_time:.3f}s")
        summary_table.add_row("最大処理時間", f"{max_processing_time:.3f}s")
        summary_table.add_row("最小処理時間", f"{min_processing_time:.3f}s")
        summary_table.add_row("ティック/秒", f"{ticks_per_second:.2f}")
        summary_table.add_row("エラー発生数", str(len(self.stats['errors'])))
        
        console.print(summary_table)
        
        # エラーがある場合は表示
        if self.stats['errors']:
            console.print("\n[red]発生したエラー:[/red]")
            for i, error in enumerate(self.stats['errors'][-5:], 1):  # 最新5件のみ
                console.print(f"  {i}. {error}")

async def main():
    """メイン実行関数"""
    demo = RealtimeProcessingDemo()
    await demo.run_realtime_demo(duration_seconds=90)  # 90秒間実行

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]プログラムが中断されました[/yellow]")
    except Exception as e:
        console.print(f"[red]実行エラー: {e}[/red]")