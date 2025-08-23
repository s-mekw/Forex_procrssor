"""
データ集約処理パイプラインデモンストレーション

このテストでは以下を視覚的に確認できます：
- 複数の時間軸でのデータ集約処理
- 複雑なフィルタリングとグループ化処理
- カスタム集約関数の適用
- 処理パフォーマンスの可視化
- メモリ効率的なパイプライン処理
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import polars as pl
import psutil
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
from dataclasses import dataclass

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.data_processing.processor import PolarsProcessingEngine
from src.common.models import Tick
from src.common.config import get_config, ConfigManager
from demo_config import get_demo_config

console = Console()

class TimeFrame(Enum):
    """時間軸定義"""
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"

@dataclass
class PipelineStage:
    """パイプライン段階の定義"""
    name: str
    description: str
    execution_time: float = 0.0
    input_rows: int = 0
    output_rows: int = 0
    memory_usage_mb: float = 0.0
    completed: bool = False

class DataAggregationPipeline:
    """データ集約処理パイプラインデモクラス"""
    
    def __init__(self):
        self.console = Console()
        self.connection_manager = None
        self.tick_streamer = None
        self.polars_engine = None
        # デモ設定を読み込み
        self.demo_config = get_demo_config()
        
        # パイプライン段階
        self.pipeline_stages = [
            PipelineStage("data_collection", "📊 リアルタイムデータ収集"),
            PipelineStage("data_cleaning", "🧹 データクリーニング・検証"),
            PipelineStage("multi_timeframe_agg", "⏰ 複数時間軸集約"),
            PipelineStage("technical_indicators", "📈 テクニカル指標計算"),
            PipelineStage("statistical_analysis", "📊 統計分析"),
            PipelineStage("anomaly_detection", "🔍 異常値検出"),
            PipelineStage("final_aggregation", "🎯 最終集約・出力")
        ]
        
        # 収集データ
        self.raw_ticks: List[Tick] = []
        self.processed_data: Dict[str, pl.DataFrame] = {}
        
        # パフォーマンス追跡
        self.start_time = None
        self.stage_times = {}
        
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
                
            # デモ設定から通貨ペアとパラメータを取得
            symbol = self.demo_config.get_symbol('aggregation')
            tick_config = self.demo_config.get_tick_streamer_config('aggregation')
            
            # ティックストリーマー設定
            self.tick_streamer = TickDataStreamer(
                symbol=symbol,
                buffer_size=tick_config.buffer_size,
                spike_threshold_percent=tick_config.spike_threshold_percent,
                backpressure_threshold=tick_config.backpressure_threshold,
                mt5_client=self.connection_manager
            )
            
            # Polarsエンジン（設定から取得）
            polars_config = self.demo_config.get_polars_engine_config('aggregation')
            self.polars_engine = PolarsProcessingEngine(
                chunk_size=polars_config.chunk_size
            )
            
            # 集約ウィンドウを保存
            self.aggregation_window = polars_config.aggregation_window
            
            console.print("[green]✅ データ集約パイプライン初期化完了[/green]")
            console.print(f"[cyan]📊 通貨ペア: {symbol}[/cyan]")
            console.print(f"[cyan]⚙️ 集約ウィンドウ: {self.aggregation_window}秒[/cyan]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ 初期化エラー: {e}[/red]")
            return False
    
    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    async def stage_1_data_collection(self) -> None:
        """段階1: リアルタイムデータ収集"""
        stage = self.pipeline_stages[0]
        stage_start = time.time()
        
        console.print("[yellow]📊 段階1: データ収集を開始します...[/yellow]")
        
        await self.tick_streamer.subscribe_to_ticks()
        
        # ストリーミングタスクを開始（重要：これがないとティックを受信しない）
        await self.tick_streamer.start_streaming()
        
        # 60秒間データを収集
        collection_duration = 60
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("ティックデータ収集中...", total=collection_duration)
            
            for i in range(collection_duration):
                await asyncio.sleep(1)
                
                # 新しいティックを取得
                new_ticks = await self.tick_streamer.get_new_ticks()
                self.raw_ticks.extend(new_ticks)
                
                progress.update(task, advance=1, 
                              description=f"収集済み: {len(self.raw_ticks)}ティック")
        
        # ストリーミングを停止
        await self.tick_streamer.stop_streaming()
        await self.tick_streamer.unsubscribe()
        
        # 段階完了
        stage.execution_time = time.time() - stage_start
        stage.input_rows = 0
        stage.output_rows = len(self.raw_ticks)
        stage.memory_usage_mb = self.get_memory_usage()
        stage.completed = True
        
        console.print(f"[green]✅ データ収集完了: {len(self.raw_ticks)}ティック[/green]")
    
    def stage_2_data_cleaning(self) -> None:
        """段階2: データクリーニング・検証"""
        stage = self.pipeline_stages[1]
        stage_start = time.time()
        stage.input_rows = len(self.raw_ticks)
        
        console.print("[yellow]🧹 段階2: データクリーニングを実行中...[/yellow]")
        
        if not self.raw_ticks:
            console.print("[red]⚠️ クリーニング対象のデータがありません[/red]")
            return
        
        # DataFrameを作成
        data = []
        for tick in self.raw_ticks:
            data.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'bid': float(tick.bid),
                'ask': float(tick.ask),
                'volume': tick.volume,
                'spread': float(tick.ask) - float(tick.bid)
            })
        
        df = pl.DataFrame(data)
        
        # データクリーニングパイプライン
        cleaned_df = (
            df
            .filter(pl.col("bid") > 0)  # 無効なbid価格を除外
            .filter(pl.col("ask") > 0)  # 無効なask価格を除外
            .filter(pl.col("spread") >= 0)  # 負のスプレッドを除外
            .filter(pl.col("spread") < pl.col("bid") * 0.01)  # 異常に大きなスプレッドを除外
            .filter(pl.col("volume") >= 0)  # 負のボリュームを除外
            .with_columns([
                # 中値計算
                ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_price"),
                # 価格変化率
                pl.col("bid").pct_change().alias("bid_change_pct"),
                pl.col("ask").pct_change().alias("ask_change_pct")
            ])
            .drop_nulls()  # null値を除外
            .sort("timestamp")  # タイムスタンプでソート
        )
        
        # データ型最適化
        cleaned_df = self.polars_engine.optimize_dtypes(cleaned_df)
        self.processed_data["cleaned"] = cleaned_df
        
        stage.execution_time = time.time() - stage_start
        stage.output_rows = cleaned_df.height
        stage.memory_usage_mb = self.get_memory_usage()
        stage.completed = True
        
        console.print(f"[green]✅ クリーニング完了: {stage.input_rows} → {stage.output_rows}行[/green]")
    
    def stage_3_multi_timeframe_aggregation(self) -> None:
        """段階3: 複数時間軸集約"""
        stage = self.pipeline_stages[2]
        stage_start = time.time()
        
        if "cleaned" not in self.processed_data:
            console.print("[red]⚠️ クリーニング済みデータがありません[/red]")
            return
        
        df = self.processed_data["cleaned"]
        stage.input_rows = df.height
        
        console.print("[yellow]⏰ 段階3: 複数時間軸集約を実行中...[/yellow]")
        
        # 1秒間隔の集約
        df_1s = (
            df
            .with_columns(
                pl.col("timestamp").dt.truncate("1s").alias("time_bucket")
            )
            .group_by("time_bucket")
            .agg([
                pl.col("bid").first().alias("open_bid"),
                pl.col("bid").max().alias("high_bid"),
                pl.col("bid").min().alias("low_bid"),
                pl.col("bid").last().alias("close_bid"),
                pl.col("ask").first().alias("open_ask"),
                pl.col("ask").max().alias("high_ask"),
                pl.col("ask").min().alias("low_ask"),
                pl.col("ask").last().alias("close_ask"),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("spread").mean().alias("avg_spread"),
                pl.len().alias("tick_count")
            ])
            .sort("time_bucket")
        )
        
        # 5秒間隔の集約
        df_5s = (
            df
            .with_columns(
                pl.col("timestamp").dt.truncate("5s").alias("time_bucket")
            )
            .group_by("time_bucket")
            .agg([
                pl.col("mid_price").first().alias("open"),
                pl.col("mid_price").max().alias("high"),
                pl.col("mid_price").min().alias("low"),
                pl.col("mid_price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("spread").mean().alias("avg_spread"),
                pl.col("spread").std().alias("spread_volatility"),
                pl.len().alias("tick_count")
            ])
            .sort("time_bucket")
        )
        
        # 30秒間隔の集約
        df_30s = (
            df
            .with_columns(
                pl.col("timestamp").dt.truncate("30s").alias("time_bucket")
            )
            .group_by("time_bucket")
            .agg([
                pl.col("mid_price").first().alias("open"),
                pl.col("mid_price").max().alias("high"),
                pl.col("mid_price").min().alias("low"),
                pl.col("mid_price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("spread").mean().alias("avg_spread"),
                pl.col("bid_change_pct").std().alias("price_volatility"),
                pl.len().alias("tick_count")
            ])
            .sort("time_bucket")
        )
        
        self.processed_data["agg_1s"] = df_1s
        self.processed_data["agg_5s"] = df_5s
        self.processed_data["agg_30s"] = df_30s
        
        total_output_rows = df_1s.height + df_5s.height + df_30s.height
        
        stage.execution_time = time.time() - stage_start
        stage.output_rows = total_output_rows
        stage.memory_usage_mb = self.get_memory_usage()
        stage.completed = True
        
        console.print(f"[green]✅ 時間軸集約完了: 1s({df_1s.height}), 5s({df_5s.height}), 30s({df_30s.height})[/green]")
    
    def stage_4_technical_indicators(self) -> None:
        """段階4: テクニカル指標計算"""
        stage = self.pipeline_stages[3]
        stage_start = time.time()
        
        if "agg_5s" not in self.processed_data:
            console.print("[red]⚠️ 集約データがありません[/red]")
            return
        
        df = self.processed_data["agg_5s"]
        stage.input_rows = df.height
        
        console.print("[yellow]📈 段階4: テクニカル指標計算を実行中...[/yellow]")
        
        # テクニカル指標を追加
        df_with_indicators = df.with_columns([
            # 移動平均
            pl.col("close").rolling_mean(window_size=5).alias("sma_5"),
            pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
            pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
            
            # 指数移動平均（近似）
            pl.col("close").ewm_mean(half_life=5).alias("ema_5"),
            pl.col("close").ewm_mean(half_life=10).alias("ema_10"),
            
            # ボラティリティ指標
            pl.col("close").rolling_std(window_size=10).alias("volatility_10"),
            pl.col("high").rolling_max(window_size=10).alias("highest_10"),
            pl.col("low").rolling_min(window_size=10).alias("lowest_10"),
            
            # RSI近似（価格変化ベース）
            pl.col("close").pct_change().alias("price_change"),
        ]).with_columns([
            # ボリンジャーバンド近似
            (pl.col("sma_20") + 2 * pl.col("volatility_10")).alias("bb_upper"),
            (pl.col("sma_20") - 2 * pl.col("volatility_10")).alias("bb_lower"),
            
            # %K（ストキャスティクス近似）
            ((pl.col("close") - pl.col("lowest_10")) / 
             (pl.col("highest_10") - pl.col("lowest_10")) * 100).alias("stoch_k"),
        ]).with_columns([
            # %D（%Kの移動平均）
            pl.col("stoch_k").rolling_mean(window_size=3).alias("stoch_d"),
            
            # シグナル生成
            (pl.col("close") > pl.col("sma_20")).alias("bullish_ma"),
            (pl.col("ema_5") > pl.col("ema_10")).alias("bullish_ema"),
            (pl.col("close") > pl.col("bb_upper")).alias("overbought"),
            (pl.col("close") < pl.col("bb_lower")).alias("oversold"),
        ])
        
        self.processed_data["technical"] = df_with_indicators
        
        stage.execution_time = time.time() - stage_start
        stage.output_rows = df_with_indicators.height
        stage.memory_usage_mb = self.get_memory_usage()
        stage.completed = True
        
        console.print(f"[green]✅ テクニカル指標計算完了: {stage.output_rows}行[/green]")
    
    def stage_5_statistical_analysis(self) -> None:
        """段階5: 統計分析"""
        stage = self.pipeline_stages[4]
        stage_start = time.time()
        
        if "technical" not in self.processed_data:
            console.print("[red]⚠️ テクニカル指標データがありません[/red]")
            return
        
        df = self.processed_data["technical"]
        stage.input_rows = df.height
        
        console.print("[yellow]📊 段階5: 統計分析を実行中...[/yellow]")
        
        # 統計サマリー作成
        stats_summary = df.select([
            pl.col("close").mean().alias("avg_close"),
            pl.col("close").std().alias("std_close"),
            pl.col("close").min().alias("min_close"),
            pl.col("close").max().alias("max_close"),
            pl.col("close").quantile(0.25).alias("q25_close"),
            pl.col("close").quantile(0.75).alias("q75_close"),
            pl.col("volume").sum().alias("total_volume"),
            pl.col("volume").mean().alias("avg_volume"),
            pl.col("avg_spread").mean().alias("overall_avg_spread"),
            pl.col("volatility_10").mean().alias("avg_volatility"),
            pl.col("bullish_ma").sum().alias("bullish_ma_count"),
            pl.col("bullish_ema").sum().alias("bullish_ema_count"),
            pl.col("overbought").sum().alias("overbought_count"),
            pl.col("oversold").sum().alias("oversold_count"),
            pl.len().alias("total_periods")
        ])
        
        # 相関分析
        correlation_data = df.select([
            "close", "volume", "avg_spread", "volatility_10", "stoch_k"
        ]).drop_nulls()
        
        if correlation_data.height > 1:
            # Polarsで相関を計算（簡易版）
            corr_analysis = correlation_data.with_columns([
                pl.corr("close", "volume").alias("close_volume_corr"),
                pl.corr("close", "avg_spread").alias("close_spread_corr"),
                pl.corr("volume", "volatility_10").alias("volume_volatility_corr")
            ]).select([
                "close_volume_corr", "close_spread_corr", "volume_volatility_corr"
            ]).head(1)
        
        # 時系列分析
        trend_analysis = df.with_columns([
            pl.col("close").pct_change().alias("returns"),
        ]).filter(pl.col("returns").is_not_null()).with_columns([
            (pl.col("returns") > 0).alias("positive_return"),
        ]).select([
            pl.col("returns").mean().alias("avg_return"),
            pl.col("returns").std().alias("return_volatility"),
            pl.col("positive_return").sum().alias("positive_periods"),
            pl.len().alias("total_return_periods")
        ])
        
        self.processed_data["stats_summary"] = stats_summary
        if 'corr_analysis' in locals():
            self.processed_data["correlation"] = corr_analysis
        self.processed_data["trend_analysis"] = trend_analysis
        
        stage.execution_time = time.time() - stage_start
        stage.output_rows = stats_summary.height + trend_analysis.height
        stage.memory_usage_mb = self.get_memory_usage()
        stage.completed = True
        
        console.print(f"[green]✅ 統計分析完了[/green]")
    
    def stage_6_anomaly_detection(self) -> None:
        """段階6: 異常値検出（変化率ベース）"""
        stage = self.pipeline_stages[5]
        stage_start = time.time()
        
        if "technical" not in self.processed_data:
            console.print("[red]⚠️ テクニカル指標データがありません[/red]")
            return
        
        df = self.processed_data["technical"]
        stage.input_rows = df.height
        
        console.print("[yellow]🔍 段階6: 異常値検出を実行中...[/yellow]")
        
        # 設定から閾値を取得（aggregation用の設定があればそれを使用）
        if "anomaly_detection" in self.demo_config and "aggregation" in self.demo_config["anomaly_detection"]:
            thresholds = self.demo_config["anomaly_detection"]["aggregation"]
        elif "anomaly_detection" in self.demo_config:
            thresholds = self.demo_config["anomaly_detection"]
        else:
            # デフォルト値
            thresholds = {
                "price_change_threshold": 2.0,
                "spread_ratio_threshold": 0.5,
                "volume_multiplier": 5.0,
                "volatility_threshold": 3.0,
                "volume_ma_window": 10
            }
        
        # 閾値を取得
        price_threshold = thresholds.get("price_change_threshold", 2.0)
        spread_threshold = thresholds.get("spread_ratio_threshold", 0.5)
        volume_mult = thresholds.get("volume_multiplier", 5.0)
        volatility_thresh = thresholds.get("volatility_threshold", 3.0)
        ma_window = thresholds.get("volume_ma_window", 10)
        
        console.print(f"[dim]異常検出閾値: 価格変化 {price_threshold}%, スプレッド {spread_threshold}%, ボリューム x{volume_mult}, ボラティリティ {volatility_thresh}%[/dim]")
        
        # 変化率ベースの異常値検出
        df_with_anomalies = df.with_columns([
            # 価格変化率（前期比）
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1) * 100)
            .abs()
            .fill_null(0)
            .alias("price_change_pct"),
            
            # スプレッドの価格に対する比率
            (pl.col("avg_spread") / pl.col("close") * 100)
            .alias("spread_ratio_pct"),
            
            # ボリュームの移動平均
            pl.col("volume").rolling_mean(window_size=ma_window).alias("volume_ma"),
        ]).with_columns([
            # 価格異常: 設定された閾値以上の急変動
            (pl.col("price_change_pct") > price_threshold).alias("price_anomaly"),
            
            # スプレッド異常: 価格の設定された割合以上
            (pl.col("spread_ratio_pct") > spread_threshold).alias("spread_anomaly"),
            
            # ボリューム異常: 移動平均の設定倍数以上（移動平均が0より大きい場合のみ）
            pl.when(pl.col("volume_ma") > 0)
            .then(pl.col("volume") > pl.col("volume_ma") * volume_mult)
            .otherwise(False)
            .alias("volume_anomaly"),
            
            # ボラティリティ異常: 設定された閾値以上
            (pl.col("volatility_10") > volatility_thresh).alias("volatility_anomaly"),
        ]).with_columns([
            # 複合異常値（いずれかの異常が発生）
            (pl.col("price_anomaly") | pl.col("spread_anomaly") | 
             pl.col("volume_anomaly") | pl.col("volatility_anomaly")).alias("any_anomaly")
        ])
        
        # 異常値サマリー
        anomaly_summary = df_with_anomalies.select([
            pl.col("price_anomaly").sum().alias("price_anomalies"),
            pl.col("volume_anomaly").sum().alias("volume_anomalies"),
            pl.col("spread_anomaly").sum().alias("spread_anomalies"),
            pl.col("volatility_anomaly").sum().alias("volatility_anomalies"),
            pl.col("any_anomaly").sum().alias("total_anomalies"),
            pl.len().alias("total_records"),
            (pl.col("any_anomaly").sum() / pl.len() * 100).alias("anomaly_rate_percent")
        ])
        
        # デバッグ情報を出力（変化率ベースの統計）
        if df_with_anomalies.height > 0:
            avg_price_change = df_with_anomalies.select(pl.col("price_change_pct").mean()).item()
            avg_spread_ratio = df_with_anomalies.select(pl.col("spread_ratio_pct").mean()).item()
            max_price_change = df_with_anomalies.select(pl.col("price_change_pct").max()).item()
            console.print(f"[dim]価格変化: 平均 {avg_price_change:.2f}%, 最大 {max_price_change:.2f}%, スプレッド比率: 平均 {avg_spread_ratio:.3f}%[/dim]")
        
        self.processed_data["anomalies"] = df_with_anomalies
        self.processed_data["anomaly_summary"] = anomaly_summary
        
        stage.execution_time = time.time() - stage_start
        stage.output_rows = df_with_anomalies.height
        stage.memory_usage_mb = self.get_memory_usage()
        stage.completed = True
        
        total_anomalies = anomaly_summary.select("total_anomalies").item()
        anomaly_rate = anomaly_summary.select("anomaly_rate_percent").item()
        console.print(f"[green]✅ 異常値検出完了: {total_anomalies}件の異常値を検出 (異常率: {anomaly_rate:.1f}%)[/green]")
    
    def stage_7_final_aggregation(self) -> None:
        """段階7: 最終集約・出力"""
        stage = self.pipeline_stages[6]
        stage_start = time.time()
        
        console.print("[yellow]🎯 段階7: 最終集約・出力を実行中...[/yellow]")
        
        # 最終レポートデータを集約
        final_report = {}
        
        if "stats_summary" in self.processed_data:
            stats = self.processed_data["stats_summary"].row(0)
            final_report["statistics"] = {
                "avg_price": float(stats[0]),
                "price_volatility": float(stats[1]),
                "price_range": float(stats[3]) - float(stats[2]),
                "total_volume": float(stats[6]),
                "avg_spread": float(stats[8])
            }
        
        if "anomaly_summary" in self.processed_data:
            anomalies = self.processed_data["anomaly_summary"].row(0)
            final_report["anomalies"] = {
                "total_anomalies": int(anomalies[4]),
                "anomaly_rate": float(anomalies[6]),
                "price_anomalies": int(anomalies[0]),
                "volume_anomalies": int(anomalies[1])
            }
        
        if "trend_analysis" in self.processed_data:
            trend = self.processed_data["trend_analysis"].row(0)
            final_report["trends"] = {
                "avg_return": float(trend[0]) if trend[0] is not None else 0.0,
                "return_volatility": float(trend[1]) if trend[1] is not None else 0.0,
                "positive_rate": float(trend[2]) / float(trend[3]) * 100 if trend[3] > 0 else 0.0
            }
        
        # パフォーマンス統計
        total_processing_time = sum(s.execution_time for s in self.pipeline_stages if s.completed)
        total_input_rows = self.pipeline_stages[0].output_rows
        
        final_report["performance"] = {
            "total_processing_time": total_processing_time,
            "total_input_rows": total_input_rows,
            "processing_rate": total_input_rows / total_processing_time if total_processing_time > 0 else 0,
            "peak_memory_mb": max(s.memory_usage_mb for s in self.pipeline_stages if s.completed),
            "stages_completed": sum(1 for s in self.pipeline_stages if s.completed)
        }
        
        self.processed_data["final_report"] = final_report
        
        stage.execution_time = time.time() - stage_start
        stage.input_rows = sum(s.output_rows for s in self.pipeline_stages[:-1] if s.completed)
        stage.output_rows = 1  # 最終レポート
        stage.memory_usage_mb = self.get_memory_usage()
        stage.completed = True
        
        console.print("[green]✅ 最終集約完了[/green]")
    
    def create_pipeline_dashboard(self) -> Layout:
        """パイプライン処理ダッシュボード"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="details", size=12)
        )
        
        layout["main"].split_row(
            Layout(name="pipeline"),
            Layout(name="metrics")
        )
        
        # ヘッダー
        header_text = Text("🔄 データ集約処理パイプライン", style="bold magenta")
        layout["header"].update(Panel(header_text, box=box.ROUNDED))
        
        # パイプライン進行状況
        pipeline_table = Table(title="パイプライン段階", box=box.SIMPLE)
        pipeline_table.add_column("段階", style="cyan")
        pipeline_table.add_column("状態", style="green")
        pipeline_table.add_column("実行時間", style="yellow")
        pipeline_table.add_column("処理行数", style="blue")
        
        for stage in self.pipeline_stages:
            status = "✅" if stage.completed else "⏳" if stage.execution_time > 0 else "⏸️"
            execution_time = f"{stage.execution_time:.3f}s" if stage.execution_time > 0 else "-"
            row_info = f"{stage.input_rows} → {stage.output_rows}" if stage.completed else "-"
            
            pipeline_table.add_row(stage.description, status, execution_time, row_info)
        
        layout["pipeline"].update(Panel(pipeline_table, title="処理進行状況", box=box.ROUNDED))
        
        # メトリクス
        metrics_table = Table(title="パフォーマンスメトリクス", box=box.SIMPLE)
        metrics_table.add_column("項目", style="cyan")
        metrics_table.add_column("値", style="green")
        
        completed_stages = [s for s in self.pipeline_stages if s.completed]
        if completed_stages:
            total_time = sum(s.execution_time for s in completed_stages)
            total_input = self.pipeline_stages[0].output_rows if self.pipeline_stages[0].completed else 0
            current_memory = self.get_memory_usage()
            
            metrics_table.add_row("完了段階", f"{len(completed_stages)}/{len(self.pipeline_stages)}")
            metrics_table.add_row("総処理時間", f"{total_time:.3f}s")
            metrics_table.add_row("処理速度", f"{total_input/total_time:.1f} 行/秒" if total_time > 0 else "-")
            metrics_table.add_row("現在メモリ", f"{current_memory:.2f}MB")
            metrics_table.add_row("ピークメモリ", f"{max(s.memory_usage_mb for s in completed_stages):.2f}MB" if completed_stages else "-")
        else:
            metrics_table.add_row("処理状態", "待機中")
        
        layout["metrics"].update(Panel(metrics_table, title="システムメトリクス", box=box.ROUNDED))
        
        # 詳細結果
        if "final_report" in self.processed_data:
            report = self.processed_data["final_report"]
            
            results_table = Table(title="処理結果サマリー", box=box.SIMPLE)
            results_table.add_column("カテゴリ", style="cyan")
            results_table.add_column("項目", style="yellow")
            results_table.add_column("値", style="green")
            
            if "statistics" in report:
                stats = report["statistics"]
                results_table.add_row("統計", "平均価格", f"{stats['avg_price']:.5f}")
                results_table.add_row("", "価格変動幅", f"{stats['price_range']:.5f}")
                results_table.add_row("", "総ボリューム", f"{stats['total_volume']:,.0f}")
                results_table.add_row("", "平均スプレッド", f"{stats['avg_spread']:.5f}")
            
            if "anomalies" in report:
                anomalies = report["anomalies"]
                results_table.add_row("異常値", "検出数", str(anomalies["total_anomalies"]))
                results_table.add_row("", "異常値率", f"{anomalies['anomaly_rate']:.2f}%")
            
            if "trends" in report:
                trends = report["trends"]
                results_table.add_row("トレンド", "平均リターン", f"{trends['avg_return']:.6f}")
                results_table.add_row("", "上昇期間率", f"{trends['positive_rate']:.1f}%")
            
            layout["details"].update(Panel(results_table, title="最終結果", box=box.ROUNDED))
        else:
            layout["details"].update(Panel("処理結果待機中...", title="最終結果", box=box.ROUNDED))
        
        return layout
    
    def print_final_summary(self):
        """最終サマリーを出力"""
        console.print("\n" + "="*70)
        console.print("[bold green]🎉 データ集約処理パイプライン完了レポート[/bold green]")
        console.print("="*70)
        
        if "final_report" not in self.processed_data:
            console.print("[red]最終レポートが生成されていません[/red]")
            return
        
        report = self.processed_data["final_report"]
        
        # パフォーマンス情報
        if "performance" in report:
            perf = report["performance"]
            console.print(f"\n[bold cyan]📊 処理パフォーマンス:[/bold cyan]")
            console.print(f"• 総処理時間: [yellow]{perf['total_processing_time']:.3f}秒[/yellow]")
            console.print(f"• 処理行数: [yellow]{perf['total_input_rows']:,}行[/yellow]")
            console.print(f"• 処理速度: [green]{perf['processing_rate']:.1f}行/秒[/green]")
            console.print(f"• ピークメモリ: [yellow]{perf['peak_memory_mb']:.2f}MB[/yellow]")
            console.print(f"• 完了段階: [green]{perf['stages_completed']}/{len(self.pipeline_stages)}[/green]")
        
        # 統計情報
        if "statistics" in report:
            stats = report["statistics"]
            console.print(f"\n[bold cyan]📈 データ統計:[/bold cyan]")
            console.print(f"• 平均価格: [yellow]{stats['avg_price']:.5f}[/yellow]")
            console.print(f"• 価格変動幅: [yellow]{stats['price_range']:.5f}[/yellow]")
            console.print(f"• 総ボリューム: [yellow]{stats['total_volume']:,.0f}[/yellow]")
            console.print(f"• 平均スプレッド: [yellow]{stats['avg_spread']:.5f}[/yellow]")
        
        # 異常値情報
        if "anomalies" in report:
            anomalies = report["anomalies"]
            console.print(f"\n[bold cyan]🔍 異常値検出:[/bold cyan]")
            console.print(f"• 総異常値数: [red]{anomalies['total_anomalies']}[/red]")
            console.print(f"• 異常値率: [red]{anomalies['anomaly_rate']:.2f}%[/red]")
            console.print(f"• 価格異常: [yellow]{anomalies['price_anomalies']}[/yellow]")
            console.print(f"• ボリューム異常: [yellow]{anomalies['volume_anomalies']}[/yellow]")
        
        # トレンド情報
        if "trends" in report:
            trends = report["trends"]
            console.print(f"\n[bold cyan]📊 トレンド分析:[/bold cyan]")
            console.print(f"• 平均リターン: [green]{trends['avg_return']:.6f}[/green]")
            console.print(f"• リターン変動: [yellow]{trends['return_volatility']:.6f}[/yellow]")
            console.print(f"• 上昇期間率: [green]{trends['positive_rate']:.1f}%[/green]")
        
        console.print("\n[green]✅ 全パイプライン段階が正常に完了しました[/green]")
    
    async def run_aggregation_pipeline(self):
        """集約パイプラインの実行"""
        console.print("[bold blue]🔄 データ集約処理パイプラインを開始します[/bold blue]\n")
        
        if not self.setup_connections():
            return
        
        self.start_time = time.time()
        
        try:
            with Live(self.create_pipeline_dashboard(), console=console, refresh_per_second=2) as live:
                # 各段階を順次実行
                await self.stage_1_data_collection()
                live.update(self.create_pipeline_dashboard())
                
                self.stage_2_data_cleaning()
                live.update(self.create_pipeline_dashboard())
                
                self.stage_3_multi_timeframe_aggregation()
                live.update(self.create_pipeline_dashboard())
                
                self.stage_4_technical_indicators()
                live.update(self.create_pipeline_dashboard())
                
                self.stage_5_statistical_analysis()
                live.update(self.create_pipeline_dashboard())
                
                self.stage_6_anomaly_detection()
                live.update(self.create_pipeline_dashboard())
                
                self.stage_7_final_aggregation()
                live.update(self.create_pipeline_dashboard())
                
                # 最終ダッシュボードを10秒間表示
                await asyncio.sleep(10)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ ユーザーによる中断[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ パイプライン実行エラー: {e}[/red]")
            
        finally:
            # クリーンアップ
            # tick_streamerのクリーンアップ
            if self.tick_streamer:
                if self.tick_streamer.is_streaming:
                    await self.tick_streamer.stop_streaming()
                if self.tick_streamer.is_subscribed:
                    await self.tick_streamer.unsubscribe()
            if self.connection_manager and self.connection_manager.is_connected():
                self.connection_manager.disconnect()
            
            # 最終サマリー
            self.print_final_summary()

async def main():
    """メイン実行関数"""
    pipeline = DataAggregationPipeline()
    await pipeline.run_aggregation_pipeline()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]プログラムが中断されました[/yellow]")
    except Exception as e:
        console.print(f"[red]実行エラー: {e}[/red]")