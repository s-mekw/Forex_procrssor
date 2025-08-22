"""
MT5データ + Polars処理基盤のデモンストレーション

このテストでは以下を視覚的に確認できます：
- MT5からのリアルタイムデータ取得
- Polarsによる高速データ処理
- メモリ最適化の効果
- データ型最適化前後の比較
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import polars as pl
import psutil
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich import box

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.data_processing.processor import PolarsProcessingEngine
from src.common.models import Tick
from src.common.config import get_config, ConfigManager

console = Console()

class PolarsDataShowcase:
    """MT5データ + Polarsデータ処理のデモクラス"""
    
    def __init__(self):
        self.console = Console()
        self.connection_manager = None
        self.tick_streamer = None
        self.polars_engine = None
        self.collected_ticks: List[Tick] = []
        
    def setup_connections(self) -> bool:
        """MT5接続とPolarsエンジンを初期化"""
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
                console.print("[red]❌ MT5への接続に失敗しました[/red]")
                return False
                
            # TickStreamerの設定
            self.tick_streamer = TickDataStreamer(
                symbol="EURUSD",
                buffer_size=1000,
                spike_threshold_percent=0.5,
                backpressure_threshold=0.8,
                mt5_client=self.connection_manager
            )
            
            # Polarsエンジンの初期化
            self.polars_engine = PolarsProcessingEngine(
                chunk_size=50  # テスト用に小さく設定
            )
            
            console.print("[green]✅ 接続とエンジン初期化完了[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ 初期化エラー: {e}[/red]")
            return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """現在のメモリ使用量を取得"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    
    def create_sample_dataframe(self, ticks: List[Tick]) -> pl.DataFrame:
        """ティックデータからPolarsデータフレームを作成"""
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
                'spread': float(tick.ask) - float(tick.bid)
            })
        
        return pl.DataFrame(data)
    
    def demonstrate_data_optimization(self, df: pl.DataFrame) -> Dict[str, Any]:
        """データ型最適化のデモンストレーション"""
        if df.is_empty():
            return {'error': 'データフレームが空です'}
            
        # 最適化前のメモリ使用量
        original_memory = df.estimated_size()
        
        # データ型最適化を実行
        optimized_df = self.polars_engine.optimize_dtypes(df)
        optimized_memory = optimized_df.estimated_size()
        
        # メモリレポート取得
        memory_report = self.polars_engine.get_memory_report(optimized_df)
        
        return {
            'original_memory_kb': original_memory / 1024,
            'optimized_memory_kb': optimized_memory / 1024,
            'reduction_percent': ((original_memory - optimized_memory) / original_memory) * 100 if original_memory > 0 else 0,
            'memory_report': memory_report,
            'optimized_df': optimized_df
        }
    
    def create_dashboard_layout(self, stats: Dict[str, Any]) -> Layout:
        """ダッシュボードレイアウトを作成"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )
        
        # メイン領域を分割
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # ヘッダー
        header_text = Text("🚀 Polars + MT5 データ処理デモンストレーション", style="bold magenta")
        layout["header"].update(Panel(header_text, box=box.ROUNDED))
        
        # 左側：接続状態とデータ収集
        connection_table = Table(title="接続状態", box=box.SIMPLE)
        connection_table.add_column("項目", style="cyan")
        connection_table.add_column("状態", style="green")
        
        mt5_status = "✅ 接続中" if self.connection_manager and self.connection_manager.is_connected() else "❌ 未接続"
        connection_table.add_row("MT5接続", mt5_status)
        connection_table.add_row("収集ティック数", str(len(self.collected_ticks)))
        connection_table.add_row("シンボル", "EURUSD")
        
        layout["left"].update(Panel(connection_table, title="システム状態", box=box.ROUNDED))
        
        # 右側：処理統計
        if 'optimization_result' in stats and not stats['optimization_result'].get('error'):
            opt = stats['optimization_result']
            
            optimization_table = Table(title="Polars最適化結果", box=box.SIMPLE)
            optimization_table.add_column("メトリクス", style="cyan")
            optimization_table.add_column("値", style="yellow")
            
            optimization_table.add_row("最適化前メモリ", f"{opt['original_memory_kb']:.2f} KB")
            optimization_table.add_row("最適化後メモリ", f"{opt['optimized_memory_kb']:.2f} KB")
            optimization_table.add_row("メモリ削減率", f"{opt['reduction_percent']:.2f}%")
            optimization_table.add_row("データ行数", str(opt['optimized_df'].height))
            optimization_table.add_row("カラム数", str(opt['optimized_df'].width))
            
            layout["right"].update(Panel(optimization_table, title="処理結果", box=box.ROUNDED))
        else:
            layout["right"].update(Panel("データ処理待機中...", title="処理結果", box=box.ROUNDED))
        
        # フッター：システムメモリ情報
        if 'memory_usage' in stats:
            mem = stats['memory_usage']
            memory_table = Table(title="システムメモリ使用量", box=box.SIMPLE)
            memory_table.add_column("項目", style="cyan")
            memory_table.add_column("値", style="green")
            
            memory_table.add_row("RSS Memory", f"{mem['rss_mb']:.2f} MB")
            memory_table.add_row("Virtual Memory", f"{mem['vms_mb']:.2f} MB")
            memory_table.add_row("Memory Percent", f"{mem['percent']:.2f}%")
            
            layout["footer"].update(Panel(memory_table, title="システム統計", box=box.ROUNDED))
        
        return layout
    
    async def collect_ticks_with_progress(self, duration_seconds: int = 30) -> None:
        """プログレスバー付きでティックデータを収集"""
        console.print(f"[yellow]📊 {duration_seconds}秒間ティックデータを収集します...[/yellow]")
        
        # ティックストリーマーを開始
        await self.tick_streamer.subscribe_to_ticks()
        
        # ストリーミングタスクを開始（重要：これがないとティックを受信しない）
        await self.tick_streamer.start_streaming()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("ティックデータ収集中...", total=duration_seconds)
            
            for i in range(duration_seconds):
                await asyncio.sleep(1)
                
                # 新しいティックを取得
                new_ticks = await self.tick_streamer.get_new_ticks()
                self.collected_ticks.extend(new_ticks)
                
                # プログレスバーを更新
                progress.update(task, advance=1, 
                              description=f"収集済み: {len(self.collected_ticks)}ティック")
        
        # ストリーミングを停止
        await self.tick_streamer.stop_streaming()
        await self.tick_streamer.unsubscribe()
        console.print(f"[green]✅ 収集完了: {len(self.collected_ticks)}ティック[/green]")
    
    async def run_demonstration(self) -> None:
        """デモンストレーションのメイン実行"""
        console.print("[bold blue]🎯 Polars + MT5 データ処理デモを開始します[/bold blue]\n")
        
        # 初期化
        if not self.setup_connections():
            return
            
        stats = {}
        
        try:
            # ライブダッシュボードでデータ収集
            with Live(console=console, refresh_per_second=2) as live:
                # 初期状態を表示
                stats['memory_usage'] = self.get_memory_usage()
                live.update(self.create_dashboard_layout(stats))
                
                await asyncio.sleep(2)  # 初期表示の時間
                
                # ティックデータ収集（バックグラウンド）
                await self.collect_ticks_with_progress(30)
                
                # データフレーム作成と最適化
                console.print("[yellow]🔄 Polarsデータフレーム作成中...[/yellow]")
                df = self.create_sample_dataframe(self.collected_ticks)
                
                if not df.is_empty():
                    # データ最適化のデモンストレーション
                    console.print("[yellow]⚡ データ型最適化実行中...[/yellow]")
                    optimization_result = self.demonstrate_data_optimization(df)
                    stats['optimization_result'] = optimization_result
                    
                    # 更新されたメモリ使用量
                    stats['memory_usage'] = self.get_memory_usage()
                    
                    # 最終ダッシュボード更新
                    live.update(self.create_dashboard_layout(stats))
                    
                    # 結果表示
                    if not optimization_result.get('error'):
                        console.print("\n[bold green]🎉 データ処理結果サマリー[/bold green]")
                        console.print(f"• 処理されたティック数: [cyan]{len(self.collected_ticks)}[/cyan]")
                        console.print(f"• メモリ削減率: [green]{optimization_result['reduction_percent']:.2f}%[/green]")
                        console.print(f"• データフレームサイズ: [yellow]{df.height} x {df.width}[/yellow]")
                        
                        # データサンプル表示
                        console.print("\n[bold]📋 データサンプル（最初の5行）:[/bold]")
                        sample_df = optimization_result['optimized_df'].head(5)
                        console.print(sample_df)
                        
                        # 統計情報表示
                        console.print("\n[bold]📊 基本統計情報:[/bold]")
                        console.print(optimization_result['optimized_df'].describe())
                    
                else:
                    console.print("[red]⚠️ 収集されたデータが不十分です[/red]")
                
                # ダッシュボードを10秒間表示
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ ユーザーによる中断[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ エラーが発生しました: {e}[/red]")
            
        finally:
            # クリーンアップ
            if self.tick_streamer:
                # ストリーミングが動作中の場合は停止
                if self.tick_streamer.is_streaming:
                    await self.tick_streamer.stop_streaming()
                # 購読中の場合は購読解除
                if self.tick_streamer.is_subscribed:
                    await self.tick_streamer.unsubscribe()
            if self.connection_manager and self.connection_manager.is_connected():
                self.connection_manager.disconnect()
                
            console.print("\n[green]✅ デモンストレーション完了[/green]")

async def main():
    """メイン実行関数"""
    showcase = PolarsDataShowcase()
    await showcase.run_demonstration()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]プログラムが中断されました[/yellow]")
    except Exception as e:
        console.print(f"[red]実行エラー: {e}[/red]")