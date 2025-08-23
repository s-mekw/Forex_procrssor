"""
大容量データストリーミング処理デモンストレーション

このテストでは以下を視覚的に確認できます：
- 大量の履歴データのストリーミング処理
- Polarsのlazyフレーム機能による遅延評価
- メモリ効率的なチャンク処理
- 動的メモリ最適化とガベージコレクション
- 処理進行状況とパフォーマンス統計の可視化
"""

import sys
import asyncio
import time
import gc
from pathlib import Path
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional, Iterator
import polars as pl
import psutil
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.text import Text
from rich import box
from dataclasses import dataclass

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.ohlc_fetcher import OHLCFetcher
from src.data_processing.processor import PolarsProcessingEngine, MemoryLimitError
from src.common.models import OHLC
from src.common.config import get_config, ConfigManager
from demo_config import get_demo_config

console = Console()

@dataclass
class StreamingStats:
    """ストリーミング処理統計"""
    total_records_processed: int = 0
    total_chunks_processed: int = 0
    current_chunk_size: int = 0
    memory_usage_mb: float = 0.0
    processing_rate_per_sec: float = 0.0
    average_chunk_time: float = 0.0
    memory_optimization_savings: float = 0.0
    gc_collections: int = 0
    lazy_optimizations_applied: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class LargeDatasetStreaming:
    """大容量データセットストリーミングデモクラス"""
    
    def __init__(self):
        self.console = Console()
        self.connection_manager = None
        self.ohlc_fetcher = None
        self.polars_engine = None
        self.stats = StreamingStats()
        # デモ設定を読み込み
        self.demo_config = get_demo_config()
        
        # 処理設定（設定ファイルから取得）
        self.symbols = self.demo_config.get_symbols_list('streaming')
        self.timeframe = "H1"  # 1時間足
        self.days_back = 30    # 30日分のデータ
        
        # パフォーマンス追跡
        self.chunk_times = []
        self.memory_snapshots = []
        self.processing_start_time = None
        
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
                "path": config.mt5_path  # MT5実行ファイルのパス（環境変数から読み込み）
            }
            
            self.connection_manager = MT5ConnectionManager()
            if not self.connection_manager.connect(mt5_config):
                console.print("[red]❌ MT5への接続に失敗[/red]")
                return False
                
            self.ohlc_fetcher = OHLCFetcher(self.connection_manager)
            
            # ストリーミング用Polarsエンジン（設定から取得）
            polars_config = self.demo_config.get_polars_engine_config('streaming')
            self.polars_engine = PolarsProcessingEngine(
                chunk_size=polars_config.chunk_size
            )
            
            # ストリーミングバッチサイズを保存
            self.streaming_batch_size = polars_config.streaming_batch_size
            
            console.print("[green]✅ 大容量データ処理システム初期化完了[/green]")
            console.print(f"[cyan]📊 通貨ペア: {', '.join(self.symbols)}[/cyan]")
            console.print(f"[cyan]⚙️ ストリーミングバッチサイズ: {self.streaming_batch_size}[/cyan]")
            return True
            
        except Exception as e:
            self.stats.errors.append(f"初期化エラー: {str(e)}")
            console.print(f"[red]❌ 初期化エラー: {e}[/red]")
            return False
    
    def get_memory_metrics(self) -> Dict[str, float]:
        """メモリメトリクスを取得"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'percent': process.memory_percent()
        }
    
    async def fetch_historical_data_stream(self) -> Iterator[List[OHLC]]:
        """履歴データをストリーミング形式で取得"""
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=self.days_back)
        
        console.print(f"[yellow]📈 {len(self.symbols)}銘柄の{self.days_back}日分のデータを取得開始[/yellow]")
        
        for symbol in self.symbols:
            try:
                # シンボルごとにデータを取得
                console.print(f"[cyan]取得中: {symbol} ({self.timeframe})[/cyan]")
                
                ohlc_data = await asyncio.to_thread(
                    self.ohlc_fetcher.fetch_ohlc_range,
                    symbol=symbol,
                    timeframe=self.timeframe,
                    start_time=start_date,
                    end_time=end_date
                )
                
                if ohlc_data:
                    # データを小さなチャンクに分割してストリーミング
                    chunk_size = 50  # チャンクサイズ
                    for i in range(0, len(ohlc_data), chunk_size):
                        chunk = ohlc_data[i:i + chunk_size]
                        yield chunk
                        await asyncio.sleep(0.01)  # 少し待機してCPU負荷を軽減
                        
            except Exception as e:
                error_msg = f"{symbol}のデータ取得エラー: {str(e)[:100]}"
                self.stats.errors.append(error_msg)
                console.print(f"[red]⚠️ {error_msg}[/red]")
    
    def create_dataframe_from_ohlc(self, ohlc_list: List[OHLC]) -> pl.DataFrame:
        """OHLCリストからPolarsデータフレームを作成"""
        if not ohlc_list:
            return pl.DataFrame()
            
        data = []
        for ohlc in ohlc_list:
            data.append({
                'timestamp': ohlc.timestamp,
                'symbol': ohlc.symbol,
                'open': float(ohlc.open),
                'high': float(ohlc.high),
                'low': float(ohlc.low),
                'close': float(ohlc.close),
                'volume': ohlc.volume,
                'spread': float(ohlc.high) - float(ohlc.low),
                'body': abs(float(ohlc.close) - float(ohlc.open)),
                'direction': 1 if float(ohlc.close) > float(ohlc.open) else -1
            })
        
        return pl.DataFrame(data)
    
    def create_lazy_processing_pipeline(self, df: pl.DataFrame) -> pl.LazyFrame:
        """遅延評価パイプラインを作成"""
        self.stats.lazy_optimizations_applied += 1
        
        return (
            df.lazy()
            .with_columns([
                # 移動平均計算
                pl.col("close").rolling_mean(window_size=5).alias("sma_5"),
                pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
                # ボラティリティ指標
                pl.col("spread").rolling_std(window_size=10).alias("spread_volatility"),
                # 価格変化率
                pl.col("close").pct_change().alias("price_change_pct")
            ])
            .with_columns([
                # シグナル生成
                (pl.col("sma_5") > pl.col("sma_20")).alias("bullish_signal"),
                # 異常値検出
                (pl.col("spread") > pl.col("spread").quantile(0.95)).alias("high_spread")
            ])
            .filter(
                pl.col("volume") > 0  # ボリューム0のデータを除外
            )
        )
    
    async def process_streaming_chunks(self):
        """ストリーミングチャンク処理のメインループ"""
        self.processing_start_time = time.time()
        
        # 進行状況追跡用
        total_expected_chunks = len(self.symbols) * (self.days_back * 24 // 50)  # 概算
        processed_chunks = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("大容量データ処理中...", total=total_expected_chunks)
            
            # ダッシュボード用のライブ更新
            with Live(console=console, refresh_per_second=3) as live:
                
                async for data_chunk in self.fetch_historical_data_stream():
                    chunk_start_time = time.time()
                    
                    try:
                        # DataFrameを作成
                        df = self.create_dataframe_from_ohlc(data_chunk)
                        
                        if not df.is_empty():
                            # メモリ使用量チェック
                            memory_before = self.get_memory_metrics()['rss_mb']
                            
                            # データ型最適化
                            optimized_df = self.polars_engine.optimize_dtypes(df)
                            
                            # 遅延評価パイプライン作成
                            lazy_pipeline = self.create_lazy_processing_pipeline(optimized_df)
                            
                            # メモリ制限チェックと調整
                            try:
                                # パイプラインを実行（遅延評価を実際に評価）
                                result_df = lazy_pipeline.collect()
                                
                                # 集約処理
                                summary = result_df.group_by("symbol").agg([
                                    pl.col("close").mean().alias("avg_close"),
                                    pl.col("volume").sum().alias("total_volume"),
                                    pl.col("bullish_signal").sum().alias("bullish_count"),
                                    pl.col("high_spread").sum().alias("high_spread_count"),
                                    pl.col("price_change_pct").std().alias("volatility")
                                ])
                                
                                # メモリ最適化効果を計算
                                memory_after = self.get_memory_metrics()['rss_mb']
                                memory_saved = memory_before - memory_after
                                if memory_saved > 0:
                                    self.stats.memory_optimization_savings += memory_saved
                                
                            except MemoryLimitError:
                                # メモリ制限に引っかかった場合の対処
                                console.print("[yellow]⚠️ メモリ制限に達しました。チャンクサイズを調整中...[/yellow]")
                                
                                # チャンクサイズを動的に調整
                                self.polars_engine.adjust_chunk_size(-20)  # 20減らす
                                
                                # ガベージコレクション実行
                                gc.collect()
                                self.stats.gc_collections += 1
                                
                                # 再度処理を試行（簡易版）
                                simple_summary = optimized_df.group_by("symbol").agg([
                                    pl.col("close").mean().alias("avg_close"),
                                    pl.col("volume").sum().alias("total_volume")
                                ])
                            
                            # 統計情報更新
                            chunk_time = time.time() - chunk_start_time
                            self.chunk_times.append(chunk_time)
                            self.stats.total_records_processed += len(data_chunk)
                            self.stats.total_chunks_processed += 1
                            self.stats.current_chunk_size = len(data_chunk)
                            self.stats.memory_usage_mb = self.get_memory_metrics()['rss_mb']
                            
                            # 処理速度計算
                            elapsed_time = time.time() - self.processing_start_time
                            self.stats.processing_rate_per_sec = self.stats.total_records_processed / max(elapsed_time, 1)
                            self.stats.average_chunk_time = sum(self.chunk_times) / len(self.chunk_times)
                            
                            # メモリスナップショット
                            self.memory_snapshots.append(self.stats.memory_usage_mb)
                        
                    except Exception as e:
                        error_msg = f"チャンク処理エラー: {str(e)[:100]}"
                        self.stats.errors.append(error_msg)
                        console.print(f"[red]❌ {error_msg}[/red]")
                    
                    # プログレス更新
                    processed_chunks += 1
                    progress.update(task, advance=1, 
                                  description=f"処理済み: {self.stats.total_records_processed:,}レコード")
                    
                    # ライブダッシュボード更新
                    live.update(self.create_streaming_dashboard())
                    
                    # メモリ圧迫時の調整
                    if self.stats.memory_usage_mb > 200:  # 200MB以上の場合
                        gc.collect()
                        self.stats.gc_collections += 1
                        await asyncio.sleep(0.1)  # 少し待機
    
    def create_streaming_dashboard(self) -> Layout:
        """ストリーミング処理ダッシュボード"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="performance", size=10)
        )
        
        layout["main"].split_row(
            Layout(name="stats"),
            Layout(name="memory")
        )
        
        # ヘッダー
        header_text = Text("📊 大容量データストリーミング処理", style="bold green")
        layout["header"].update(Panel(header_text, box=box.ROUNDED))
        
        # 左側：処理統計
        stats_table = Table(title="処理統計", box=box.SIMPLE)
        stats_table.add_column("メトリクス", style="cyan")
        stats_table.add_column("値", style="green")
        
        stats_table.add_row("処理済みレコード", f"{self.stats.total_records_processed:,}")
        stats_table.add_row("処理済みチャンク", f"{self.stats.total_chunks_processed:,}")
        stats_table.add_row("現在のチャンクサイズ", str(self.stats.current_chunk_size))
        stats_table.add_row("処理速度", f"{self.stats.processing_rate_per_sec:.1f} レコード/秒")
        stats_table.add_row("平均チャンク処理時間", f"{self.stats.average_chunk_time:.3f}秒")
        stats_table.add_row("遅延評価最適化", str(self.stats.lazy_optimizations_applied))
        
        layout["stats"].update(Panel(stats_table, title="ストリーミング統計", box=box.ROUNDED))
        
        # 右側：メモリ管理
        memory_table = Table(title="メモリ管理", box=box.SIMPLE)
        memory_table.add_column("項目", style="cyan")
        memory_table.add_column("値", style="yellow")
        
        memory_table.add_row("現在のメモリ使用量", f"{self.stats.memory_usage_mb:.2f}MB")
        memory_table.add_row("メモリ最適化削減量", f"{self.stats.memory_optimization_savings:.2f}MB")
        memory_table.add_row("GC実行回数", str(self.stats.gc_collections))
        
        # メモリ傾向
        if len(self.memory_snapshots) >= 5:
            recent_trend = sum(self.memory_snapshots[-5:]) / 5
            older_trend = sum(self.memory_snapshots[:5]) / min(5, len(self.memory_snapshots))
            trend = "📈" if recent_trend > older_trend else "📉" if recent_trend < older_trend else "➡️"
            memory_table.add_row("メモリ使用傾向", trend)
        
        # 利用可能メモリ
        available_memory = self.get_memory_metrics()['available_mb']
        memory_table.add_row("利用可能メモリ", f"{available_memory:.2f}MB")
        
        layout["memory"].update(Panel(memory_table, title="メモリ監視", box=box.ROUNDED))
        
        # 下部：パフォーマンス詳細
        perf_table = Table(title="パフォーマンス詳細", box=box.SIMPLE)
        perf_table.add_column("項目", style="cyan")
        perf_table.add_column("値", style="green")
        perf_table.add_column("状態", style="yellow")
        
        # チャンクサイズ効率性
        current_chunk_size = self.polars_engine.chunk_size
        perf_table.add_row("動的チャンクサイズ", str(current_chunk_size), 
                          "⚡" if current_chunk_size > 50 else "🐌")
        
        # 処理効率
        if self.chunk_times:
            recent_avg = sum(self.chunk_times[-5:]) / min(5, len(self.chunk_times))
            efficiency = "🚀" if recent_avg < 0.1 else "⚡" if recent_avg < 0.5 else "🐌"
            perf_table.add_row("処理効率", f"{recent_avg:.3f}s/chunk", efficiency)
        
        # エラー状況
        error_count = len(self.stats.errors)
        error_status = "✅" if error_count == 0 else "⚠️" if error_count < 5 else "❌"
        perf_table.add_row("エラー数", str(error_count), error_status)
        
        # システム負荷
        cpu_percent = psutil.cpu_percent()
        cpu_status = "🔥" if cpu_percent > 80 else "⚡" if cpu_percent > 50 else "😴"
        perf_table.add_row("CPU使用率", f"{cpu_percent:.1f}%", cpu_status)
        
        layout["performance"].update(Panel(perf_table, title="システム状態", box=box.ROUNDED))
        
        return layout
    
    def print_final_report(self):
        """最終レポートを出力"""
        console.print("\n" + "="*60)
        console.print("[bold green]🎉 大容量データストリーミング処理完了レポート[/bold green]")
        console.print("="*60)
        
        # 処理サマリー
        total_time = time.time() - self.processing_start_time if self.processing_start_time else 0
        
        summary_table = Table(title="処理サマリー", box=box.ROUNDED)
        summary_table.add_column("項目", style="cyan")
        summary_table.add_column("値", style="green")
        
        summary_table.add_row("総処理時間", f"{total_time:.2f}秒")
        summary_table.add_row("処理レコード数", f"{self.stats.total_records_processed:,}")
        summary_table.add_row("処理チャンク数", f"{self.stats.total_chunks_processed:,}")
        summary_table.add_row("平均処理速度", f"{self.stats.processing_rate_per_sec:.1f} レコード/秒")
        summary_table.add_row("メモリ最適化削減量", f"{self.stats.memory_optimization_savings:.2f}MB")
        summary_table.add_row("遅延評価最適化実行回数", str(self.stats.lazy_optimizations_applied))
        summary_table.add_row("ガベージコレクション実行回数", str(self.stats.gc_collections))
        
        console.print(summary_table)
        
        # パフォーマンス統計
        if self.chunk_times:
            console.print("\n[bold]📈 処理時間統計:[/bold]")
            avg_time = sum(self.chunk_times) / len(self.chunk_times)
            min_time = min(self.chunk_times)
            max_time = max(self.chunk_times)
            
            perf_stats_table = Table(box=box.SIMPLE)
            perf_stats_table.add_column("統計", style="cyan")
            perf_stats_table.add_column("時間", style="green")
            
            perf_stats_table.add_row("平均処理時間", f"{avg_time:.4f}秒")
            perf_stats_table.add_row("最短処理時間", f"{min_time:.4f}秒")
            perf_stats_table.add_row("最長処理時間", f"{max_time:.4f}秒")
            
            console.print(perf_stats_table)
        
        # エラーレポート
        if self.stats.errors:
            console.print(f"\n[red]⚠️ 発生したエラー ({len(self.stats.errors)}件):[/red]")
            for i, error in enumerate(self.stats.errors[-10:], 1):  # 最新10件のみ
                console.print(f"  {i}. {error}")
        else:
            console.print("\n[green]✅ エラーなしで完了しました[/green]")
    
    async def run_large_dataset_demo(self):
        """大容量データセットデモの実行"""
        console.print("[bold blue]📊 大容量データストリーミング処理デモを開始します[/bold blue]\n")
        
        if not self.setup_connections():
            return
        
        try:
            await self.process_streaming_chunks()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ ユーザーによる中断[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ 実行エラー: {e}[/red]")
            self.stats.errors.append(f"実行エラー: {str(e)}")
            
        finally:
            # クリーンアップ
            if self.connection_manager and self.connection_manager.is_connected():
                self.connection_manager.disconnect()
            
            # 最終レポート
            self.print_final_report()

async def main():
    """メイン実行関数"""
    demo = LargeDatasetStreaming()
    await demo.run_large_dataset_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]プログラムが中断されました[/yellow]")
    except Exception as e:
        console.print(f"[red]実行エラー: {e}[/red]")